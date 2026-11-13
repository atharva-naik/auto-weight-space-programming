"""Round 4 write task (exploratory): the model WRITES part of one of its own MLP output vectors, as integers, so
that the written slice promotes / suppresses given tokens through its own logit lens. The inverse of raw_reader.

Training pairs come free from real neurons: (top-8 promoted, top-8 suppressed tokens of coordinates s..s+127)
-> those 128 quantized numbers. Scoring is exact: we compute the logit lens of what the model wrote.

  python src/raw_writer.py --out runs/round4/write_4B_scratch
  python src/raw_writer.py --out runs/round4/write_4B_fromreader --init_adapter runs/round4/sweep_4B/adapter_stage0
  python src/raw_writer.py --tiny --out /tmp/w --n_items 16 --n_val 8 --n_test 8     # CPU smoke

Metrics per item (val and test_neuron): parse_ok, recall_up / recall_down (requested top-8 tokens that are in the
written slice's own top-8), cos (written vs true slice). Baselines on the same items, no LM: random Gaussian,
the true slice (ceiling), and an analytic writer v = g * (sum U[up] - sum U[down]) over the slice.
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import raw_reader as R  # noqa: E402

PER_LINE = 64


def write_prompt(tok, up, down, s, size, d):
    msg = (f"Write part of one of your own weight vectors: the output direction of one MLP neuron (a column of "
           f"mlp.down_proj, which the neuron adds to the residual stream), coordinates {s} to {s + size - 1} of {d}. "
           f"Use integers scaled so that {R.SCALE_STD} is one standard deviation of the full column, in lines of "
           f"{PER_LINE} values that each start with the index of their first coordinate in brackets. Through the logit "
           f"lens (your final norm and unembedding), this part of the vector should promote most: "
           f"{json.dumps(up, ensure_ascii=False)} and suppress most: {json.dumps(down, ensure_ascii=False)}. "
           f"Answer with the numbers only.")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def parse_numbers(text, s, size):
    vals, ok = torch.zeros(size), 0
    for m in re.finditer(r"\[(\d+)\]([^\[]*)", text):
        base = int(m.group(1))
        for i, x in enumerate(re.findall(r"-?\d+", m.group(2))):
            j = base + i - s
            if 0 <= j < size:
                vals[j] = float(x)
                ok += 1
    return vals, ok >= size


@torch.no_grad()
def lens_of(lens, layer_starts, vecs, size, k=8):
    """Top-k up/down token ids of arbitrary slice vectors (same filters as the labels)."""
    V = torch.zeros(len(vecs), lens.d, device=lens.device)
    for j, ((_, s), v) in enumerate(zip(layer_starts, vecs)):
        V[j, s:s + size] = v.to(lens.device).float()
    sc = (V * lens.g) @ lens.U.T
    up, dn = sc.clone(), -sc
    up[:, lens.base_bad] = -math.inf
    dn[:, lens.base_bad] = -math.inf
    if size in lens.artifacts:
        bu, bd = lens.artifacts[size]
        up[:, bu] = -math.inf
        dn[:, bd] = -math.inf
    return up.topk(k).indices.tolist(), dn.topk(k).indices.tolist()


def score(lens, items, labels, vecs, size):
    tu, td = lens_of(lens, [(it["layer"], it["start"]) for it in items], vecs, size)
    out = []
    for it, (u, d_), v, wu, wd in zip(items, labels, vecs, tu, td):
        true = lens.col(it["layer"], it["n"])[it["start"]:it["start"] + size].cpu()
        cos = 0.0 if float(v.float().norm()) == 0 else float(torch.nn.functional.cosine_similarity(v.float()[None], true[None]))
        out.append({"recall_up": len(set(u) & set(wu)) / 8, "recall_down": len(set(d_) & set(wd)) / 8, "cos": cos})
    return out


@torch.no_grad()
def eval_write(tok, model, lens, items, size, args, tag, logj):
    model.eval()
    model.config.use_cache = True
    labels = lens.top([(it["layer"], it["n"], it["start"]) for it in items], size)
    recs, vecs = [], []
    for i0 in range(0, len(items), args.gen_bs):
        chunk = items[i0:i0 + args.gen_bs]
        prompts = [write_prompt(tok, lens.label_strings(u), lens.label_strings(d_), it["start"], size, lens.d)
                   for it, (u, d_) in zip(chunk, labels[i0:i0 + args.gen_bs])]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        gen = model.generate(**enc, max_new_tokens=int(size * 3.5) + 80, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, it in enumerate(chunk):
            text = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            v, ok = parse_numbers(text, it["start"], size)
            vecs.append(v)
            recs.append({"id": it["id"], "parse_ok": int(ok), "gen": text[:400]})
    model.config.use_cache = False
    for r, sc in zip(recs, score(lens, items, labels, vecs, size)):
        r.update(sc)
    # baselines on the same items (no LM)
    g = torch.Generator().manual_seed(0)
    base = {"random": [torch.randn(size, generator=g) for _ in items],
            "true": [torch.tensor(R.quantize(lens.col(it["layer"], it["n"]))[it["start"]:it["start"] + size],
                                  dtype=torch.float) for it in items],
            "analytic": []}
    for it, (u, d_) in zip(items, labels):
        S = slice(it["start"], it["start"] + size)
        base["analytic"].append(((lens.U[u][:, S].sum(0) - lens.U[d_][:, S].sum(0)) * lens.g[S]).cpu())
    summ = {"n": len(recs), **{k: float(np.mean([r[k] for r in recs])) for k in ("parse_ok", "recall_up", "recall_down", "cos")}}
    for name, vs in base.items():
        sc = score(lens, items, labels, vs, size)
        summ[f"{name}_recall_up"] = float(np.mean([x["recall_up"] for x in sc]))
        summ[f"{name}_recall_down"] = float(np.mean([x["recall_down"] for x in sc]))
    json.dump(recs, open(f"{args.out}/write_{tag}.json", "w"), ensure_ascii=False)
    logj(event="write_eval", tag=tag, **summ)
    return summ


def main(args):
    R.ARGS = args
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok, model, lm = R.load(args)
    lens = R.Lens(tok, model, lm, args.device)
    args.d, args.d_ff = lens.d, lens.W[0].shape[1]
    from raw_splits import default_path
    splits = json.load(open(args.splits or default_path(args.model)))
    size = args.size if not args.tiny else 32
    start = args.start if not args.tiny else 0
    if args.tiny:
        splits["eval"] = {k: [dict(x, layer=x["layer"] % 4, n=x["n"] % 96) for x in v] for k, v in splits["eval"].items()}
        splits["train_layers"] = [0, 1, 2, 3]
        splits["excluded_from_train"] = {str(l): list(range(10)) for l in range(4)}
    fix = lambda its, n: [dict(it, start=start, size=size, id=f"{it['layer']}_{it['n']}_{size}_{start}") for it in its[:n]]
    val = fix(splits["eval"]["val_128"], args.n_val)
    test = fix(splits["eval"]["test_neuron"], args.n_test)
    log = open(f"{args.out}/log.jsonl", "a")

    def logj(**kw):
        kw["time"] = time.time()
        print(json.dumps(kw, ensure_ascii=False), flush=True)
        log.write(json.dumps(kw, ensure_ascii=False) + "\n")
        log.flush()

    json.dump(vars(args), open(f"{args.out}/config.json", "w"), indent=1)
    sample = R.train_items(splits, size, 400 if args.tiny else 4000, random.Random(f"art{size}"))
    lens.fit_artifacts([(it["layer"], it["n"], start) for it in sample], size)

    from peft import LoraConfig, PeftModel, get_peft_model
    ck = f"{args.out}/ckpt_latest"
    resume_step = 0
    if args.resume and os.path.exists(f"{ck}/trainer.pt"):
        model = PeftModel.from_pretrained(model, ck, is_trainable=True)
        resume_step = json.load(open(f"{ck}/step.json"))["step"]
        logj(event="resume", step=resume_step)
    elif args.init_adapter:  # start from the trained reader
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        logj(event="init_adapter", path=args.init_adapter)
    else:
        names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.endswith(R.LORA_SUFFIXES)
                 and ".layers." in n and "visual" not in n and "mtp" not in n]
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=names))
    if not args.tiny:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    if not resume_step:
        eval_write(tok, model, lens, val, size, args, "step0_val", logj)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = math.ceil(args.n_items / args.items_per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / total)))))
    if resume_step:
        st = torch.load(f"{ck}/trainer.pt", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    items = R.train_items(splits, size, args.n_items, rng)
    for it in items:
        it["start"] = start
    step = 0
    for i0 in range(0, len(items), args.items_per_step):
        if step < resume_step:
            step += 1
            continue
        model.train()
        batch = items[i0:i0 + args.items_per_step]
        labels = lens.top([(it["layer"], it["n"], start) for it in batch], size)
        seqs = [R.encode(tok, write_prompt(tok, lens.label_strings(u), lens.label_strings(d_), start, size, lens.d),
                         R.numbers_text(R.quantize(lens.col(it["layer"], it["n"])), start, size, PER_LINE))
                for it, (u, d_) in zip(batch, labels)]
        ntok = sum(len(t) for _, t in seqs)
        tot, t0 = 0.0, time.time()
        for j in range(0, len(seqs), args.mb):
            loss = -torch.cat(R.batch_logp(lm, model, seqs[j:j + args.mb], args.device)).sum() / ntok
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == 1:
            logj(event="train", step=step, loss=tot, lr=sched.get_last_lr()[0], sec=time.time() - t0)
        if not math.isfinite(tot):
            logj(event="nan", step=step)
            return
        if args.save_every and step % args.save_every == 0:
            model.save_pretrained(ck)
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, f"{ck}/trainer.pt")
            json.dump({"step": step}, open(f"{ck}/step.json", "w"))
            logj(event="checkpoint", step=step)
        if args.val_every and step % args.val_every == 0:
            eval_write(tok, model, lens, val[:args.n_val_mid], size, args, f"step{step}_val", logj)
    model.save_pretrained(f"{args.out}/adapter_final")
    eval_write(tok, model, lens, val, size, args, "final_val", logj)
    eval_write(tok, model, lens, test, size, args, "final_test_neuron", logj)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n_items", type=int, default=64000)
    ap.add_argument("--items_per_step", type=int, default=16)
    ap.add_argument("--mb", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_val_mid", type=int, default=64)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--gen_bs", type=int, default=16)
    ap.add_argument("--init_adapter", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--multi_gpu", action="store_true")
    ap.add_argument("--fixed_start", type=int, default=0)
    ap.add_argument("--train_pool", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(ap.parse_args())
