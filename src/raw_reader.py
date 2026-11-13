"""Round 4 kill test: teach Qwen3.5-4B to read its own MLP down_proj columns written as plain-text integers.
See PREREG_ROUND4.md for the fixed design. One process = one format on one GPU.

  python src/raw_reader.py --fmt h64 --out runs/round4/h64            # zero-shot eval, curriculum training, final eval
  python src/raw_reader.py --fmt h64 --out runs/round4/smoke --smoke  # tiny item counts, for checking the pipeline
  python src/raw_reader.py --tiny --out /tmp/rr --smoke               # CPU: random 4-layer Qwen3.5, no download

Outputs in --out: config.json, log.jsonl (train loss, val), eval_<tag>_<split>_<cond>.jsonl (per-item scores),
artifacts_<size>.json, adapter_stage<k>/ (LoRA).
"""
import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

D, D_FF = 2560, 9216
SCALE_STD = 20
LORA_SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
HIDDEN = "(numbers hidden)"
EOS_STR = "<|im_end|>"


# ------------------------------------------------------------------ text format
def quantize(v):
    """v: float tensor [D] (full column). Integers, 1 std = SCALE_STD, no clipping."""
    return torch.round(v.float() / v.float().std() * SCALE_STD).to(torch.int64).tolist()


def numbers_text(q, start, size, per_line, perm=None):
    vals = q[start:start + size]
    if perm is not None:
        vals = [vals[i] for i in perm]
    return "\n".join(f"[{start + i}] " + " ".join(str(x) for x in vals[i:i + per_line]) for i in range(0, size, per_line))


def user_prompt(numbers, start, size, d=D):
    return (f"Below is part of one of your own weight vectors: the output direction of one MLP neuron (a column of "
            f"mlp.down_proj, which the neuron adds to the residual stream), coordinates {start} to {start + size - 1} "
            f"of {d}. Values are integers scaled so that {SCALE_STD} is one standard deviation of the full column. "
            f"Each line starts with the index of its first coordinate.\n\n{numbers}\n\n"
            f"Read these weights. Through the logit lens (your final norm and unembedding), which tokens does this "
            f"part of the vector promote most and suppress most? Answer exactly in the format\n"
            f"PROMOTES: [8 tokens as a JSON list]\nSUPPRESSES: [8 tokens as a JSON list]")


def target_text(up, down):
    return f"PROMOTES: {json.dumps(up, ensure_ascii=False)}\nSUPPRESSES: {json.dumps(down, ensure_ascii=False)}"


def parse_target(s):
    out = {}
    for key in ("PROMOTES", "SUPPRESSES"):
        m = re.search(key + r":\s*(\[.*?\])\s*(?:\n|$)", s, re.S)
        try:
            out[key] = [str(x) for x in json.loads(m.group(1))] if m else []
        except Exception:
            out[key] = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)) if m else []
    return out


# ------------------------------------------------------------------ model
def load(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if args.tiny:
        from transformers import AutoConfig, Qwen3_5ForCausalLM
        tc = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B").text_config
        for k, v in dict(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16).items():
            setattr(tc, k, v)
        tc.layer_types = tc.layer_types[:4]
        torch.manual_seed(0)
        model = Qwen3_5ForCausalLM(tc).to(args.device)
        with torch.no_grad():
            model.model.norm.weight.normal_(0, 0.1)
        lm = model.model
    elif "Qwen3.5" in args.model:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            device_map="auto" if args.multi_gpu else args.device)
        if getattr(model.model, "visual", None) is not None:
            model.model.visual = None
        lm = model.model.language_model
    else:  # other text-only causal LMs with model.model.{layers, norm} and model.lm_head (e.g. Llama)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     device_map="auto" if args.multi_gpu else args.device)
        lm = model.model
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    global EOS_STR
    EOS_STR = "<|im_end|>" if ("Qwen" in args.model or args.tiny) else tok.eos_token
    model.config.use_cache = False
    return tok, model, lm


class Lens:
    """Exact slice logit lens of base down_proj columns, with the pre-registered token filters."""

    def __init__(self, tok, model, lm, device):
        self.tok = tok
        self.W = [L.mlp.down_proj.weight.detach() for L in lm.layers]  # base weights; LoRA never changes them
        self.d = self.W[0].shape[0]
        w = lm.norm.weight.detach().float()
        # Final-norm scale, read from the norm's own forward code: Qwen3.5 (and Gemma) multiply by (1 + weight),
        # Llama-style norms by weight. Qwen3.5-4B's (1 + w) is confirmed against the atlas (src/check_atlas.py).
        # (A weight-mean heuristic was wrong for Qwen3.5, whose stored weights are large; do not use one.)
        import inspect
        src = inspect.getsource(type(lm.norm).forward) + inspect.getsource(type(lm.norm))
        self.norm_param = "1+w" if re.search(r"1(\.0)?\s*\+\s*self\.weight", src) else "w"
        self.g = (1 + w if self.norm_param == "1+w" else w).to(device)
        self.U = model.lm_head.weight.detach().float().to(device)
        V = self.U.shape[0]
        bad = torch.zeros(V, dtype=torch.bool)
        bad[len(tok):] = True
        bad[torch.tensor(tok.all_special_ids)] = True
        self.strs = []
        for i in range(min(V, len(tok))):
            s = tok.decode([i])
            self.strs.append(s)
            if not s.strip() or "�" in s:
                bad[i] = True
        self.base_bad = bad.to(device)
        self.artifacts = {}  # size -> (bad_up, bad_down) bool[V]
        self.device = device

    def col(self, layer, n):
        return self.W[layer][:, n].float()

    @torch.no_grad()
    def top(self, items, size, k=8, raw=False):
        """items: list of (layer, n, start). Returns list of (up_ids, down_ids)."""
        out = []
        for i in range(0, len(items), 64):
            chunk = items[i:i + 64]
            V = torch.zeros(len(chunk), self.d, device=self.device)
            for j, (l, n, s) in enumerate(chunk):
                V[j, s:s + size] = self.col(l, n)[s:s + size]
            sc = (V * self.g) @ self.U.T
            up, dn = sc.clone(), -sc
            up[:, self.base_bad] = -math.inf
            dn[:, self.base_bad] = -math.inf
            if not raw and size in self.artifacts:
                bu, bd = self.artifacts[size]
                up[:, bu] = -math.inf
                dn[:, bd] = -math.inf
            tu, td = up.topk(k).indices.tolist(), dn.topk(k).indices.tolist()
            out += list(zip(tu, td))
        return out

    def fit_artifacts(self, items, size, frac=0.05):
        res = self.top(items, size, k=12, raw=True)
        bads = []
        for pole in (0, 1):
            cnt = torch.zeros(self.U.shape[0])
            for r in res:
                cnt[torch.tensor(r[pole])] += 1
            bads.append((cnt > frac * len(items)).to(self.device))
        self.artifacts[size] = tuple(bads)
        return {p: [self.strs[i] for i in torch.nonzero(b).flatten().tolist()] for p, b in zip(("up", "down"), bads)}

    def label_strings(self, ids):
        return [self.strs[i] for i in ids]


# ------------------------------------------------------------------ items
def per_line(fmt):
    return {"h64": 64, "h8": 8, "f3": 32}[fmt]


def float_text(vals, start, per_line_, perm=None):
    """Raw-float ablation: unscaled values, 3 significant digits, index headers."""
    v = [float(x) for x in vals]
    if perm is not None:
        v = [v[i] for i in perm]
    return "\n".join(f"[{start + i}] " + " ".join(f"{x:.3g}" for x in v[i:i + per_line_]) for i in range(0, len(v), per_line_))


def make_prompt(tok, lens, it, fmt, cond="normal", seed=0):
    l, n, s, size = it["layer"], it["n"], it["start"], it["size"]
    if cond == "hidden":
        nums = HIDDEN
    elif getattr(ARGS, "input_mode", "numbers") == "id":  # bug-check control: an easy identifier instead of numbers
        nums = f"(neuron #{n})"
    else:
        perm = None
        if cond == "scrambled":
            perm = list(range(size))
            random.Random(f"{seed}-{it['id']}").shuffle(perm)
        if fmt == "f3":
            nums = float_text(lens.col(l, n)[s:s + size].tolist(), s, per_line(fmt), perm)
        else:
            nums = numbers_text(quantize(lens.col(l, n)), s, size, per_line(fmt), perm)
    content = user_prompt(nums, s, size, lens.d)
    if fmt == "f3":
        content = content.replace(f"Values are integers scaled so that {SCALE_STD} is one standard deviation of the full column.",
                                  "Values are the raw weights, written with 3 significant digits.")
    msg = [{"role": "user", "content": content}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode(tok, prompt, target, eos=True):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(target + (EOS_STR if eos else ""), add_special_tokens=False)["input_ids"]
    return p, t


def batch_logp(lm, model, seqs, device):
    """seqs: list of (prompt_ids, target_ids). Left-padded; returns list of per-token logp tensors over target_ids."""
    K = max(len(t) for _, t in seqs)
    L = max(len(p) + len(t) for p, t in seqs)
    ids = torch.full((len(seqs), L), 0, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, (p, t) in enumerate(seqs):
        x = p + t
        ids[i, L - len(x):] = torch.tensor(x)
        att[i, L - len(x):] = 1
    ids, att = ids.to(device), att.to(device)
    h = lm(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state[:, -(K + 1):-1]
    logits = model.lm_head(h).float()
    lp = []
    for i, (p, t) in enumerate(seqs):
        tt = torch.tensor(t, device=logits.device)
        row = F.log_softmax(logits[i, K - len(t):], -1)
        lp.append(row.gather(-1, tt[:, None])[:, 0])
    return lp


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def eval_t2(tok, model, lm, lens, items, fmt, cond, args, hard_pool=None):
    """4-way choice by summed candidate log-prob, with and without numbers (PMI). Returns per-item records."""
    model.eval()
    recs = []
    size = items[0]["size"]
    for it in items:  # candidate order shuffled per item, so ties and position bias cannot favour the true one
        c = [it["n"]] + (hard_pool[it["id"]] if hard_pool else it["distractors"])
        order = list(range(4))
        random.Random(f"cand-{it['id']}").shuffle(order)
        it["cands"], it["truth"] = [c[o] for o in order], order.index(0)
    allc = [(it["layer"], c, it["start"]) for it in items for c in it["cands"]]
    labels = lens.top(allc, size)
    bs = max(1, args.eval_tokens // (size * 3 + 400))
    for i0 in range(0, len(items), bs):
        chunk = items[i0:i0 + bs]
        seqs, nseqs = [], []
        for j, it in enumerate(chunk):
            P = encode(tok, make_prompt(tok, lens, it, fmt, cond, args.seed), "")[0]
            Pn = encode(tok, make_prompt(tok, lens, it, fmt, "hidden", args.seed), "")[0]
            for c in range(4):
                u, d_ = labels[(i0 + j) * 4 + c]
                t = encode(tok, "", target_text(lens.label_strings(u), lens.label_strings(d_)), eos=False)[1]
                seqs.append((P, t))
                nseqs.append((Pn, t))
        lp = [x.sum().item() for x in _chunked_logp(lm, model, seqs, args, 4)]
        ln = [x.sum().item() for x in _chunked_logp(lm, model, nseqs, args, 32)]
        for j, it in enumerate(chunk):
            a, b = lp[4 * j:4 * j + 4], ln[4 * j:4 * j + 4]
            pmi = [x - y for x, y in zip(a, b)]
            T = it["truth"]
            recs.append({"id": it["id"], "layer": it["layer"], "n": it["n"], "cands": it["cands"], "truth": T,
                         "logp": a, "logp_hidden": b, "pmi": pmi, "correct_raw": int(np.argmax(a) == T),
                         "correct_pmi": int(np.argmax(pmi) == T), "correct_hidden": int(np.argmax(b) == T)})
    return recs


def _chunked_logp(lm, model, seqs, args, per):
    out = []
    for i in range(0, len(seqs), per):
        out += batch_logp(lm, model, seqs[i:i + per], args.device)
    return out


@torch.no_grad()
def eval_t1(tok, model, lm, lens, items, fmt, args):
    """Greedy generation; recall@8 against the top-12 per pole (with filters)."""
    model.eval()
    model.config.use_cache = True
    size = items[0]["size"]
    labels = lens.top([(it["layer"], it["n"], it["start"]) for it in items], size, k=12)
    recs = []
    for i0 in range(0, len(items), 4):
        chunk = items[i0:i0 + 4]
        enc = tok([make_prompt(tok, lens, it, fmt) for it in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(args.device)
        gen = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, it in enumerate(chunk):
            s = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            p = parse_target(s)
            gu, gd = [x.strip() for x in lens.label_strings(labels[i0 + j][0])], \
                     [x.strip() for x in lens.label_strings(labels[i0 + j][1])]
            ru = len({x.strip() for x in p["PROMOTES"][:8]} & set(gu)) / 8
            rd = len({x.strip() for x in p["SUPPRESSES"][:8]} & set(gd)) / 8
            recs.append({"id": it["id"], "gen": s, "recall_up": ru, "recall_down": rd,
                         "format_ok": int(len(p["PROMOTES"]) > 0 and len(p["SUPPRESSES"]) > 0)})
    model.config.use_cache = False
    return recs


@torch.no_grad()
def eval_openvocab(tok, model, lm, lens, items, fmt, args, k=8):
    """No candidates: one forward pass on prompt + 'PROMOTES: ["' (and on the numbers-hidden prompt), giving a score
    for every vocabulary token at the first list slot. Ranks all tokens by raw log p and by PMI (with minus hidden),
    and reports where the true top-8 promoted tokens land. A label string's token is taken as its first token."""
    model.eval()
    size = items[0]["size"]
    labels = lens.top([(it["layer"], it["n"], it["start"]) for it in items], size, k=k)
    pre = encode(tok, "", 'PROMOTES: ["', eos=False)[1]
    allowed = ~lens.base_bad
    recs = []
    for i0 in range(0, len(items), 4):
        chunk = items[i0:i0 + 4]
        P = [encode(tok, make_prompt(tok, lens, it, fmt), "")[0] + pre for it in chunk]
        H = [encode(tok, make_prompt(tok, lens, it, fmt, "hidden"), "")[0] + pre for it in chunk]
        lw, lh = [], []
        for seqs, out in ((P, lw), (H, lh)):
            L = max(map(len, seqs))
            ids = torch.zeros((len(seqs), L), dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, x in enumerate(seqs):
                ids[j, L - len(x):] = torch.tensor(x)
                att[j, L - len(x):] = 1
            h = lm(input_ids=ids.to(args.device), attention_mask=att.to(args.device), use_cache=False).last_hidden_state[:, -1]
            out.append(F.log_softmax(model.lm_head(h).float(), -1))
        lw, lh = lw[0], lh[0]
        for j, it in enumerate(chunk):
            true_first = {tok(lens.strs[t], add_special_tokens=False)["input_ids"][0] for t in labels[i0 + j][0]}
            r = {"id": it["id"]}
            for name, sc in (("raw", lw[j]), ("pmi", lw[j] - lh[j])):
                sc = sc.clone()
                sc[~allowed.to(sc.device)] = -math.inf
                order = torch.argsort(sc, descending=True)
                rank = torch.empty_like(order)
                rank[order] = torch.arange(len(order), device=order.device)
                rk = [int(rank[t]) for t in true_first]
                top = set(order[:k].tolist())
                r[f"recall{k}_{name}"] = len(top & true_first) / len(true_first)
                r[f"median_rank_{name}"] = float(np.median(rk))
                r[f"top{k}_{name}"] = [lens.strs[t] if t < len(lens.strs) else "?" for t in order[:k].tolist()]
            r["true"] = lens.label_strings(labels[i0 + j][0])
            recs.append(r)
    return recs


def hard_distractors(lens, items, pools, size):
    """3 same-layer, same-split neurons whose shown slice has the highest cosine with the item's."""
    out = {}
    for it in items:
        l, s = it["layer"], it["start"]
        pool = [m for m in pools[str(l)] if m != it["n"]]
        M = lens.W[l][s:s + size, pool].float()
        v = lens.col(l, it["n"])[s:s + size]
        cos = (v @ M) / (v.norm() * M.norm(dim=0) + 1e-8)
        out[it["id"]] = [pool[i] for i in cos.topk(3).indices.tolist()]
    return out


def summarize(recs):
    r = {k: float(np.mean([x[k] for x in recs])) for k in recs[0] if k.startswith(("correct", "recall", "format"))}
    r["n"] = len(recs)
    return r


# ------------------------------------------------------------------ training
def train_items(splits, size, n, rng):
    excl = {int(l): set(v) for l, v in splits["excluded_from_train"].items()}
    layers = splits["train_layers"]
    out = []
    pool = None
    if ARGS.train_pool:  # exploratory: a fixed pool of training neurons, revisited many times
        prng, pool = random.Random("pool"), []
        while len(pool) < ARGS.train_pool:
            l, m = layers[prng.randrange(len(layers))], prng.randrange(ARGS.d_ff)
            if m not in excl.get(l, ()) and (l, m) not in pool:
                pool.append((l, m))
    for i in range(n):
        l = layers[rng.randrange(len(layers))]
        while pool is None:
            m = rng.randrange(ARGS.d_ff if not ARGS.tiny else 96)
            if m not in excl.get(l, ()):
                break
        if pool is not None:
            l, m = pool[rng.randrange(len(pool))]
        al = min(64, size)
        s = 0 if size >= ARGS.d else al * rng.randrange((ARGS.d - size) // al + 1)
        if ARGS.fixed_start >= 0:
            s = ARGS.fixed_start
        out.append({"id": f"tr{i}", "layer": l, "n": m, "start": s, "size": size})
    return out


def main(args):
    global ARGS
    ARGS = args
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok, model, lm = load(args)
    if args.target_model:  # cross-model: read (and label) another model's weights; the prompt stays identical
        targs = argparse.Namespace(**{**vars(args), "model": args.target_model, "multi_gpu": False})
        ttok, tmodel, tlm = load(targs)
        tmodel.requires_grad_(False)
        assert len(ttok) == len(tok) and ttok.get_vocab() == tok.get_vocab(), "reader/target tokenizers differ"
        lens = Lens(tok, tmodel, tlm, args.device)
    else:
        lens = Lens(tok, model, lm, args.device)
    args.d, args.d_ff = lens.d, lens.W[0].shape[1]
    if not args.splits:
        from raw_splits import default_path
        args.splits = default_path(args.target_model or args.model)
    splits = json.load(open(args.splits))
    if args.tiny:  # 4 layers x 96 neurons x 64 dims: remap the real splits onto the toy model
        def remap(it):
            return dict(it, layer=it["layer"] % 4, n=it["n"] % 96, start=0, size=64,
                        distractors=[x % 96 for x in it["distractors"]])
        splits["eval"] = {k: [remap(x) for x in v] for k, v in splits["eval"].items()}
        splits["train_layers"] = [0, 1, 2, 3]
        splits["excluded_from_train"] = {str(l): list(range(10)) for l in range(4)}
        splits["pools"] = {"test_neuron": {str(l): list(range(96)) for l in range(4)}}
        stages = [(64, args.n_items or 32)]
    else:
        stages = [(128, 16000), (512, 8000), (2560, 8000)]
        if args.smoke:
            stages = [(128, 160), (512, 64), (2560, 32)]
        if args.n_items:
            stages = [(s, args.n_items) for s, _ in stages]
        if args.stages:  # exploratory only, e.g. "16:16000" or "128:16000"
            stages = [tuple(int(x) for x in st.split(":")) for st in args.stages.split(",")]
    n_val = 8 if (args.smoke or args.tiny) else 400
    n_test = 8 if (args.smoke or args.tiny) else 1000
    def val_items(size):
        key = f"val_{size}" if f"val_{size}" in splits["eval"] and not args.tiny else "val_128"
        out = []
        for it in splits["eval"][key][:n_val]:
            it = dict(it)
            if args.fixed_start >= 0:
                it["start"] = args.fixed_start
            elif it["size"] != size:
                it["start"] = min(it["start"], args.d - size)
            if not args.tiny:
                it["size"] = size
            it["id"] = f"{it['layer']}_{it['n']}_{it['size']}_{it['start']}"
            out.append(it)
        return out
    val = {s: val_items(s) for s, _ in stages}
    if args.val_fresh:  # exploratory confirmation: val-pool neurons never used in any earlier check
        used = {(it["layer"], it["n"]) for k, v in splits["eval"].items() if k.startswith("val") for it in v}
        used |= {(it["layer"], c) for k, v in splits["eval"].items() if k.startswith("val") for it in v
                 for c in it["distractors"]}
        frng = random.Random("fresh")
        vp = {int(l): [m for m in ms if (int(l), m) not in used] for l, ms in splits["pools"]["val"].items()}
        layers = sorted(vp)
        fresh = []
        for i in range(args.val_fresh):
            l = layers[i % len(layers)]
            c = frng.sample(vp[l], 4)
            for s_, _ in stages[-1:]:
                st = args.fixed_start if args.fixed_start >= 0 else 0
                fresh.append({"id": f"f{l}_{c[0]}_{s_}_{st}", "layer": l, "n": c[0], "start": st, "size": s_,
                              "distractors": c[1:]})
        val = {stages[-1][0]: fresh}
    json.dump(vars(args) | {"stages": stages}, open(f"{args.out}/config.json", "w"), indent=1)
    log = open(f"{args.out}/log.jsonl", "a")

    def logj(**kw):
        kw["time"] = time.time()
        print(json.dumps(kw), flush=True)
        log.write(json.dumps(kw) + "\n")
        log.flush()

    # artifact filters, fitted on training items only
    for size, _ in stages:
        sample = train_items(splits, size, 400 if (args.smoke or args.tiny) else 4000, random.Random(f"art{size}"))
        arts = lens.fit_artifacts([(it["layer"], it["n"], it["start"]) for it in sample], size)
        json.dump(arts, open(f"{args.out}/artifacts_{size}.json", "w"), ensure_ascii=False)
        logj(event="artifacts", size=size, n_up=len(arts["up"]), n_down=len(arts["down"]))

    from peft import LoraConfig, get_peft_model
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.endswith(LORA_SUFFIXES)
             and ".layers." in n and "visual" not in n and "mtp" not in n]
    logj(event="lora_targets", n=len(names), example=names[:12])
    ck = f"{args.out}/ckpt_latest"
    resume_step = 0
    if args.resume and os.path.exists(f"{ck}/trainer.pt"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ck, is_trainable=True)
        resume_step = json.load(open(f"{ck}/step.json"))["step"]
        logj(event="resume", step=resume_step)
    elif args.load_adapter:  # eval-only on a saved adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.load_adapter)
    elif args.init_adapter:  # start training from a trained adapter (e.g. transfer across slices)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        logj(event="init_adapter", path=args.init_adapter)
    else:
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=names))
    if not args.tiny:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    logj(event="params", trainable=sum(p.numel() for p in params))

    def run_eval(tag, final=False):
        for size, _ in stages:
            if final and size != stages[-1][0] and tag != "zeroshot":
                continue
            for cond in args.val_conds.split(","):
                hp = None
                if cond == "hard":  # 3 most similar same-layer val-pool neurons (cosine of the shown slice)
                    hp = hard_distractors(lens, val[size], splits["pools"]["val"], size)
                recs = eval_t2(tok, model, lm, lens, [dict(x) for x in val[size]], args.fmt,
                               "normal" if cond == "hard" else cond, args, hard_pool=hp)
                json.dump(recs, open(f"{args.out}/eval_{tag}_val{size}_{cond}.json", "w"))
                logj(event="val", tag=tag, size=size, cond=cond, **summarize(recs))
        if not final or args.val_only:
            return
        tests = {"test_neuron": splits["eval"]["test_neuron"][:n_test], "test_layer": splits["eval"]["test_layer"][:n_test]}
        if args.fixed_start >= 0:  # fixed-slice recipe: the same test neurons and distractors, re-sliced
            sz = stages[-1][0]
            tests = {k: [dict(it, start=args.fixed_start, size=sz, id=f"{it['layer']}_{it['n']}_{sz}_{args.fixed_start}")
                         for it in v] for k, v in tests.items()}
        if args.tiny:
            tests = {"test_neuron": splits["eval"]["val_128"][:n_test]}
        for split, items in tests.items():
            conds = ["normal", "scrambled", "hard"] if split == "test_neuron" else ["normal"]
            for cond in conds:
                hp = None
                if cond == "hard":
                    hp = hard_distractors(lens, items, splits["pools"]["test_neuron"], items[0]["size"])
                recs = eval_t2(tok, model, lm, lens, [dict(x) for x in items], args.fmt,
                               "normal" if cond == "hard" else cond, args, hard_pool=hp)
                json.dump(recs, open(f"{args.out}/eval_{tag}_{split}_{cond}.json", "w"))
                logj(event="test", tag=tag, split=split, cond=cond, **summarize(recs))
        t1 = eval_t1(tok, model, lm, lens, [dict(x) for x in tests["test_neuron"][:8 if (args.smoke or args.tiny) else 200]],
                     args.fmt, args)
        json.dump(t1, open(f"{args.out}/eval_{tag}_t1_test_neuron.json", "w"), ensure_ascii=False)
        logj(event="t1", tag=tag, **summarize(t1))
        if args.openvocab:  # O5: no-candidate readout on test_neuron, for this model and for the base model (floor)
            for who in ("trained", "base"):
                its = [dict(x) for x in tests["test_neuron"]]
                if who == "base":
                    with model.disable_adapter():
                        ov = eval_openvocab(tok, model, lm, lens, its, args.fmt, args)
                else:
                    ov = eval_openvocab(tok, model, lm, lens, its, args.fmt, args)
                json.dump(ov, open(f"{args.out}/eval_{tag}_openvocab_test_neuron_{who}.json", "w"), ensure_ascii=False)
                logj(event="openvocab", tag=tag, who=who, split="test_neuron", n=len(ov),
                     **{k: float(np.mean([r[k] for r in ov])) for k in ov[0] if k.startswith("recall")},
                     **{k: float(np.median([r[k] for r in ov])) for k in ov[0] if k.startswith("median")})

    if args.eval_only:
        run_eval(args.eval_tag, final=True)
        if args.openvocab and args.val_only:
            its = [dict(x) for x in val[stages[-1][0]]]
            ov = eval_openvocab(tok, model, lm, lens, its, args.fmt, args)
            json.dump(ov, open(f"{args.out}/eval_{args.eval_tag}_openvocab.json", "w"), ensure_ascii=False)
            logj(event="openvocab", tag=args.eval_tag, n=len(ov), **{k: float(np.mean([r[k] for r in ov])) for k in ov[0]
                 if k.startswith("recall")}, **{k: float(np.median([r[k] for r in ov])) for k in ov[0] if k.startswith("median")})
            t1 = eval_t1(tok, model, lm, lens, its[:200], args.fmt, args)
            json.dump(t1, open(f"{args.out}/eval_{args.eval_tag}_t1.json", "w"), ensure_ascii=False)
            logj(event="t1", tag=args.eval_tag, **summarize(t1))
        return
    if not args.skip_zero_shot and not resume_step:
        run_eval("zeroshot", final=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = sum(math.ceil(n / args.items_per_step) for _, n in stages)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / total)))))
    step = 0
    if resume_step:
        st = torch.load(f"{ck}/trainer.pt", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    for k, (size, n) in enumerate(stages):
        items = train_items(splits, size, n, rng)
        if args.memorize:  # positive control: train on N val neurons of one layer (one slice) and score those same
            # items, with distractors drawn from the same memorised set, so a label prior cannot separate candidates
            L0 = val[size][0]["layer"]
            pool = splits["pools"]["val"][str(L0)][:args.memorize]
            st = max(args.fixed_start, 0)
            mrng = random.Random("mem")
            val[size] = [{"id": f"m{L0}_{m}", "layer": L0, "n": m, "start": st, "size": size,
                          "distractors": mrng.sample([x for x in pool if x != m], 3)} for m in pool]
            items = [dict(val[size][rng.randrange(len(pool))], id=f"tr{i}") for i in range(n)]
        mb = max(1, args.train_tokens // (int(size * 3.2) + 400))
        for i0 in range(0, n, args.items_per_step):
            if step < resume_step:  # fast-forward: items are deterministic, so skip the batches already trained on
                step += 1
                continue
            model.train()
            batch = items[i0:i0 + args.items_per_step]
            labels = lens.top([(it["layer"], it["n"], it["start"]) for it in batch], size)
            seqs = [encode(tok, make_prompt(tok, lens, it, args.fmt),
                           target_text(lens.label_strings(u), lens.label_strings(d_))) for it, (u, d_) in zip(batch, labels)]
            ntok = sum(len(t) for _, t in seqs)
            tot = 0.0
            t0 = time.time()
            for j in range(0, len(seqs), mb):
                lp = batch_logp(lm, model, seqs[j:j + mb], args.device)
                loss = -torch.cat(lp).sum() / ntok
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                logj(event="train", step=step, stage=k, size=size, loss=tot, lr=sched.get_last_lr()[0],
                     sec=time.time() - t0, seq_len=len(seqs[0][0]) + len(seqs[0][1]))
            if not math.isfinite(tot):
                logj(event="nan", step=step)
                return
            if args.save_every and step % args.save_every == 0:  # checkpoint: adapter + optimizer + schedule
                model.save_pretrained(ck)
                torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, f"{ck}/trainer.pt")
                json.dump({"step": step, "stage": k}, open(f"{ck}/step.json", "w"))
                logj(event="checkpoint", step=step)
            if args.stop_after and step >= args.stop_after:  # testing only: simulate a crash
                return
            if args.val_every and step % args.val_every == 0:
                recs = eval_t2(tok, model, lm, lens, [dict(x) for x in val[size][:100]], args.fmt, "normal", args)
                logj(event="val_mid", step=step, size=size, **summarize(recs))
        if step <= resume_step and k < len(stages) - 1 and os.path.exists(f"{args.out}/adapter_stage{k}"):
            continue  # stage finished before the resume point
        model.save_pretrained(f"{args.out}/adapter_stage{k}")
        run_eval(f"stage{k}", final=(k == len(stages) - 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="h64", choices=["h64", "h8", "f3"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--target_model", default="", help="cross-model: whose weights are read (default: the reader's own)")
    ap.add_argument("--multi_gpu", action="store_true", help="device_map=auto over visible GPUs (27B)")
    ap.add_argument("--splits", default="", help="default: data/raw_splits.json for 4B, raw_splits_<size>.json else")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--items_per_step", type=int, default=16)
    ap.add_argument("--train_tokens", type=int, default=16000, help="token budget per micro-batch")
    ap.add_argument("--eval_tokens", type=int, default=40000)
    ap.add_argument("--val_every", type=int, default=100)
    ap.add_argument("--n_items", type=int, default=0, help="override items per stage")
    ap.add_argument("--stages", default="", help="exploratory: comma list size:n_items")
    ap.add_argument("--fixed_start", type=int, default=-1, help="exploratory: always show this slice")
    ap.add_argument("--val_only", action="store_true", help="exploratory: no test-split evals")
    ap.add_argument("--memorize", type=int, default=0, help="positive control: train on first N val items")
    ap.add_argument("--load_adapter", default="")
    ap.add_argument("--init_adapter", default="", help="train starting from this adapter")
    ap.add_argument("--save_every", type=int, default=500, help="checkpoint every N steps to <out>/ckpt_latest")
    ap.add_argument("--resume", action="store_true", help="resume from <out>/ckpt_latest if present")
    ap.add_argument("--stop_after", type=int, default=0, help="testing only: stop after N steps")
    ap.add_argument("--openvocab", action="store_true", help="eval-only: open-vocabulary ranking + T1 generation")
    ap.add_argument("--val_fresh", type=int, default=0, help="exploratory: N never-used val-pool neurons")
    ap.add_argument("--eval_tag", default="loaded")
    ap.add_argument("--val_conds", default="normal", help="comma list of val conditions, e.g. normal,scrambled")
    ap.add_argument("--train_pool", type=int, default=0, help="exploratory: fixed pool of N training neurons")
    ap.add_argument("--input_mode", default="numbers", choices=["numbers", "id"], help="control: id replaces numbers")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--skip_zero_shot", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(ap.parse_args())
