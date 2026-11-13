"""Round 4c reading tasks (PREREG_ROUND4_TASKS.md), Qwen3.5-4B reading its own raw weights as text.

  act      predict your own activation: gate-row slice (coords 0-127) + layer + sentence -> UP / DOWN, the sign of
           sum_{i<128} x_L[i] * W_gate[L][n, i], where x_L is the base model's post_attention_layernorm output at the
           sentence's last token (needs a forward pass; no matmul on the weights alone gives it)
  circuit  neuron A's output slice (down_proj column) + 4 candidate input slices (gate_proj rows) in layer L_A + k ->
           which candidate has the highest cosine with A (1-4)

  python src/task_reader.py --task act --out runs/round4/task_act_4B
  python src/task_reader.py --task circuit --out runs/round4/task_circuit_4B
  python src/task_reader.py --task act --tiny --out /tmp/ta --n_items 32 --n_val 8 --n_test 8   # CPU smoke
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import raw_reader as R  # noqa: E402

SIZE = 128


def qtext(vec, start=0, per_line=64, perm=None):
    v = torch.as_tensor(vec).float()
    q = torch.round(v / (v.std() + 1e-12) * R.SCALE_STD).to(torch.int64).tolist()
    return R.numbers_text(q, start, len(q), per_line, perm)


def chat(tok, msg):
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


class Weights:
    def __init__(self, lm):
        self.down = [L.mlp.down_proj.weight.detach() for L in lm.layers]
        self.gate = [L.mlp.gate_proj.weight.detach() for L in lm.layers]

    def out_slice(self, l, n):
        return self.down[l][:SIZE, n].float().cpu()

    def in_slice(self, l, n):
        return self.gate[l][n, :SIZE].float().cpu()


# ------------------------------------------------------------------ sentences and their layer inputs (task act)
def load_sentences(tiny):
    if tiny:
        base = ["The cat sat on the mat .", "Rain fell over the quiet town all night .", "He opened the old book slowly ."]
        return {"train": base * 4, "val": base, "test": base}
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    def sents(split):
        out = []
        for line in ds[split]["text"]:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            for s in line.split(" . "):
                w = s.split()
                if 8 <= len(w) <= 40:
                    out.append(" ".join(w) + " .")
        return out
    rng = random.Random(0)
    tr, va, te = sents("train"), sents("validation"), sents("test")
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return {"train": tr[:6000], "val": va[:800], "test": te[:1000]}


@torch.no_grad()
def layer_inputs(tok, lm, sentences, device, bs=16):
    """x_L at the last token for every layer: [N, n_layers, d] (fp16, CPU)."""
    cap = {}
    hooks = [L.post_attention_layernorm.register_forward_hook(
        lambda m, i, o, l=l: cap.__setitem__(l, o[:, -1].detach().to(torch.float16).cpu())) for l, L in enumerate(lm.layers)]
    tok.padding_side = "left"
    out = []
    for i0 in range(0, len(sentences), bs):
        enc = tok(sentences[i0:i0 + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
        out.append(torch.stack([cap[l] for l in range(len(lm.layers))], 1))
    for h in hooks:
        h.remove()
    return torch.cat(out)


# ------------------------------------------------------------------ items
def act_item(Wt, X, sents, l, n, si, cond="normal", rng=None, sj=None):
    """Two-sentence comparison (PREREG_ROUND4_TASKS.md amendment 1): which sentence pushes the gate higher?"""
    w = Wt.in_slice(l, n)
    c1, c2 = float(X[si, l, :SIZE].float() @ w), float(X[sj, l, :SIZE].float() @ w)
    perm = None
    if cond == "shuffled":
        perm = list(range(len(w))); rng.shuffle(perm)
    nums = "(numbers hidden)" if cond == "hidden" else qtext(w, perm=perm)
    msg = (f"Below is part of one of your own weight vectors: the input weights of one MLP neuron in layer {l} (a row "
           f"of mlp.gate_proj), coordinates 0 to {SIZE - 1}. Values are integers scaled so that {R.SCALE_STD} is one "
           f"standard deviation of these values. Each line starts with the index of its first coordinate.\n\n{nums}\n\n"
           f"Sentence 1: \"{sents[si]}\"\nSentence 2: \"{sents[sj]}\"\n\nWhen you read each sentence, which one makes "
           f"this part of the neuron's input weights push its gate higher at the sentence's last token? Answer 1 or 2.")
    return msg, ["1", "2"], 0 if c1 > c2 else 1, abs(c1 - c2)


def circuit_item(Wt, la, na, lb, cands, cond="normal", rng=None):
    a = Wt.out_slice(la, na)
    bs = [Wt.in_slice(lb, c) for c in cands]
    cos = [float(torch.nn.functional.cosine_similarity(a[None], b[None])) for b in bs]
    perm_a = perm_b = None
    if cond == "shuffle_a":
        perm_a = list(range(len(a))); rng.shuffle(perm_a)
    if cond == "shuffle_b":
        perm_b = list(range(len(a))); rng.shuffle(perm_b)
    blocks = "\n\n".join(f"({j + 1})\n{qtext(b, perm=perm_b)}" for j, b in enumerate(bs))
    msg = (f"Below are parts of your own weights, coordinates 0 to {SIZE - 1}, as integers scaled so that {R.SCALE_STD} "
           f"is one standard deviation of each vector. Each line starts with the index of its first coordinate.\n\n"
           f"Neuron A, layer {la}, output weights (a column of mlp.down_proj):\n{qtext(a, perm=perm_a)}\n\n"
           f"Four neurons in layer {lb}, input weights (rows of mlp.gate_proj):\n\n{blocks}\n\n"
           f"Which of the four has input weights that line up best with neuron A's output (the largest cosine over "
           f"these coordinates)? Answer 1, 2, 3 or 4.")
    srt = sorted(cos, reverse=True)
    return msg, ["1", "2", "3", "4"], int(np.argmax(cos)), srt[0] - srt[1]


def pools(splits, d_ff, tiny):
    excl = {int(l): set(v) for l, v in splits["excluded_from_train"].items()}
    tr_layers = splits["train_layers"]
    if tiny:
        return {"train": {l: list(range(10, 96)) for l in range(4)}, "val": {l: list(range(5)) for l in range(4)},
                "test": {l: list(range(5, 10)) for l in range(4)}}, [0, 1, 2, 3]
    train = {l: [m for m in range(d_ff) if m not in excl[l]] for l in tr_layers}
    val = {int(l): v for l, v in splits["pools"]["val"].items()}
    test = {int(l): v for l, v in splits["pools"]["test_neuron"].items()}
    return {"train": train, "val": val, "test": test}, tr_layers


def make_items(task, Wt, X, sents, P, layers, split, n, seed, cond="normal"):
    rng = random.Random(f"{task}-{split}-{seed}")  # item choice: identical across conditions (paired controls)
    prng = random.Random(f"perm-{task}-{split}-{seed}-{cond}")  # only the manipulation differs
    out = []
    pool = P[split]
    for i in range(n):
        if task == "act":
            l = layers[i % len(layers)] if split != "train" else rng.choice(layers)
            nn_ = rng.choice(pool[l])
            si, sj = rng.sample(range(len(sents[split])), 2)
            msg, ans, y, margin = act_item(Wt, X[split], sents[split], l, nn_, si, cond, prng, sj)
            out.append({"msg": msg, "answers": ans, "y": y, "margin": margin, "id": f"{l}_{nn_}_{si}_{sj}"})
        else:
            while True:
                la = rng.choice(layers)
                k = rng.choice([1, 2, 3])
                lb = la + k
                if lb in pool and lb in layers:
                    break
            na = rng.choice(pool[la])
            cands = rng.sample(pool[lb], 4)
            msg, ans, y, margin = circuit_item(Wt, la, na, lb, cands, cond, prng)
            out.append({"msg": msg, "answers": ans, "y": y, "margin": margin, "id": f"{la}_{na}_{lb}"})
    return out


# ------------------------------------------------------------------ scoring and training
@torch.no_grad()
def score(tok, model, lm, items, device, bs=8):
    model.eval()
    recs = []
    for i0 in range(0, len(items), bs):
        chunk = items[i0:i0 + bs]
        seqs = [tok(chat(tok, it["msg"]), add_special_tokens=False)["input_ids"] for it in chunk]
        L = max(map(len, seqs))
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, x in enumerate(seqs):
            ids[j, L - len(x):] = torch.tensor(x)
            att[j, L - len(x):] = 1
        h = lm(input_ids=ids.to(device), attention_mask=att.to(device), use_cache=False).last_hidden_state[:, -1]
        lp = torch.log_softmax(model.lm_head(h).float(), -1)
        for j, it in enumerate(chunk):
            first = [tok(a, add_special_tokens=False)["input_ids"][0] for a in it["answers"]]
            sc = [float(lp[j, t]) for t in first]
            r = {"id": it["id"], "y": it["y"], "pred": int(np.argmax(sc)), "scores": sc}
            r["correct"] = int(r["pred"] == r["y"])
            if "margin" in it:
                r["margin"] = it["margin"]
            recs.append(r)
    return recs


def ci(c):
    """Exact (Clopper-Pearson) 95% interval, by bisection on the binomial tail (no scipy on the sandbox)."""
    k, n = int(sum(c)), len(c)

    def tail_ge(p):  # P(X >= k)
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))

    def tail_le(p):  # P(X <= k)
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(0, k + 1))

    def solve(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lo = solve(tail_ge, 0.025, True) if k > 0 else 0.0
    hi = solve(tail_le, 0.025, False) if k < n else 1.0
    return k / n, float(lo), float(hi)


def main(a):
    R.ARGS = argparse.Namespace(input_mode="numbers", train_pool=0, tiny=a.tiny, fixed_start=0, seed=0)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    tok, model, lm = R.load(a)
    Wt = Weights(lm)
    d_ff = Wt.down[0].shape[1]
    from raw_splits import default_path
    splits = json.load(open(default_path(a.model)))
    P, layers = pools(splits, d_ff, a.tiny)
    log = open(f"{a.out}/log.jsonl", "a")

    def logj(**kw):
        kw["time"] = time.time()
        print(json.dumps(kw), flush=True)
        log.write(json.dumps(kw) + "\n")
        log.flush()

    X, sents = {}, {}
    if a.task == "act":
        sents = load_sentences(a.tiny)
        for sp in ("train", "val", "test"):
            X[sp] = layer_inputs(tok, lm, sents[sp], a.device)
        frac = {sp: float(np.mean([it["y"] == 0 for it in make_items("act", Wt, X, sents, P, layers, sp, 400, 9)]))
                for sp in ("train", "val")}
        logj(event="sentences", **{k: len(v) for k, v in sents.items()}, frac_label_1=frac)
    val = make_items(a.task, Wt, X, sents, P, layers, "val", a.n_val, 0)

    from peft import LoraConfig, PeftModel, get_peft_model
    ck = f"{a.out}/ckpt_latest"
    start = 0
    if a.eval_only:  # final evaluation of a finished run (adapter_final), no training
        model = PeftModel.from_pretrained(model, f"{a.out}/adapter_final")
    elif a.resume and os.path.exists(f"{ck}/step.json"):
        model = PeftModel.from_pretrained(model, ck, is_trainable=True)
        start = json.load(open(f"{ck}/step.json"))["step"]
    else:
        names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.endswith(R.LORA_SUFFIXES)
                 and ".layers." in n and "visual" not in n and "mtp" not in n]
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=names))
    if not a.tiny and not a.eval_only:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    total = math.ceil(a.n_items / a.items_per_step)
    if not a.eval_only:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / total)))))
    if start and not a.eval_only and os.path.exists(f"{ck}/trainer.pt"):
        st = torch.load(f"{ck}/trainer.pt", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    if a.eval_only:
        start = 10 ** 9
    if not start:
        r = score(tok, model, lm, val[:100], a.device)
        logj(event="val_mid", step=0, acc=float(np.mean([x["correct"] for x in r])), n=len(r))
    train = make_items(a.task, Wt, X, sents, P, layers, "train", a.n_items, 0)
    step = 0
    for i0 in range(0, len(train), a.items_per_step):
        if step < start:
            step += 1
            continue
        model.train()
        batch = train[i0:i0 + a.items_per_step]
        seqs = [R.encode(tok, chat(tok, it["msg"]), it["answers"][it["y"]]) for it in batch]
        ntok = sum(len(t) for _, t in seqs)
        tot, t0 = 0.0, time.time()
        for j in range(0, len(seqs), a.mb):
            loss = -torch.cat(R.batch_logp(lm, model, seqs[j:j + a.mb], a.device)).sum() / ntok
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == 1:
            logj(event="train", step=step, loss=tot, sec=time.time() - t0, seq_len=len(seqs[0][0]))
        if a.save_every and step % a.save_every == 0:
            model.save_pretrained(ck)
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, f"{ck}/trainer.pt")
            json.dump({"step": step}, open(f"{ck}/step.json", "w"))
        if a.val_every and step % a.val_every == 0:
            r = score(tok, model, lm, val[:100], a.device)
            logj(event="val_mid", step=step, acc=float(np.mean([x["correct"] for x in r])), n=len(r))
    if not a.eval_only:
        model.save_pretrained(f"{a.out}/adapter_final")
    r = score(tok, model, lm, val, a.device)
    logj(event="val", acc=ci([x["correct"] for x in r]), n=len(r))
    conds = ["normal", "shuffled", "hidden", "other_sentence"] if a.task == "act" else ["normal", "shuffle_a", "shuffle_b"]
    for cond in conds:
        if cond == "other_sentence":  # keep the labels of the true sentence, show a different test sentence
            base = make_items(a.task, Wt, X, sents, P, layers, "test", a.n_test, 1)
            alt = make_items(a.task, Wt, X, sents, P, layers, "test", a.n_test, 2)
            items = [dict(b, msg=b["msg"].split("Sentence 1:")[0] + "Sentence 1:" + o["msg"].split("Sentence 1:")[1])
                     for b, o in zip(base, alt)]  # b's weights and label, another item's two sentences
        else:
            items = make_items(a.task, Wt, X, sents, P, layers, "test", a.n_test, 1, cond)
        r = score(tok, model, lm, items, a.device)
        json.dump(r, open(f"{a.out}/test_{cond}.json", "w"))
        extra = {}
        if cond == "normal":
            m = np.array([x["margin"] for x in r]); c = np.array([x["correct"] for x in r])
            qs = np.quantile(m, [1 / 3, 2 / 3])
            extra = {"by_margin_tertile": [float(c[m <= qs[0]].mean()), float(c[(m > qs[0]) & (m <= qs[1])].mean()),
                                           float(c[m > qs[1]].mean())]}
        if a.task == "act":
            extra["frac_label_1"] = float(np.mean([x["y"] == 0 for x in r]))
        logj(event="test", cond=cond, acc=ci([x["correct"] for x in r]), n=len(r), **extra)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["act", "circuit"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_items", type=int, default=64000)
    ap.add_argument("--items_per_step", type=int, default=16)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--multi_gpu", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(ap.parse_args())
