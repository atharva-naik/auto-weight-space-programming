"""Pilot for 'find the levers' (docs/PLAN_LEVERS.md): are single-neuron behavioural labels reliable?

For each sampled neuron (layer L, n) of Qwen3.5-4B, edit the neuron's own activation (input to down_proj) at every
position, with no tuned scale (fixed before any behavioural result was seen, at the user's request to keep it organic):
  x0:    switch it off
  x2:    double what it already does
  clamp: set it to its own strong-firing value (signed 99th percentile of |activation| on plain Wikipedia text)
measure the change in two behaviours, separately on two disjoint halves of each prompt set:
  sycophancy: soft_pf from patient.pressure_eval (higher = more sycophantic), MMLU-style dev items
  math:       mean log-prob of the correct final answer on GSM8K test problems (teacher-forced)
Outputs per-neuron deltas (half A, half B), split-half Spearman reliability, the noise floor, the count of neurons
above 3x noise, the rank of the reference neuron L12 N2920, and PCA-basis retention of the columns.

  python src/levers_pilot.py --out runs/round4/levers_pilot.json   # 100 neurons x layers 6 12 18 24 + L12 N2920
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from patient import Patient, pressure_eval  # noqa: E402


def gsm8k(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")["test"]
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        ans = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        if re.fullmatch(r"-?\d+", ans) and len(ds[i]["question"]) < 400:
            out.append({"q": ds[i]["question"], "a": ans})
        if len(out) == n:
            break
    return out


@torch.no_grad()
def math_logp(P, probs, bs=24):
    """Mean teacher-forced log-prob of the correct answer tokens (per problem). Batched; the Patient tokenizer pads on
    the left, so every answer sits at the end of its row."""
    tok = P.tok
    res = []
    for s in range(0, len(probs), bs):
        rows = []
        for pr in probs[s:s + bs]:
            prompt = P.chat([{"role": "user", "content": pr["q"] + "\nAnswer with only the final number."}])
            rows.append((tok(prompt, add_special_tokens=False)["input_ids"], tok(pr["a"], add_special_tokens=False)["input_ids"]))
        T = max(len(p) + len(t) for p, t in rows)
        ids = torch.full((len(rows), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), T), dtype=torch.long)
        for i, (p, t) in enumerate(rows):
            ids[i, T - len(p) - len(t):] = torch.tensor(p + t); att[i, T - len(p) - len(t):] = 1
        h = P.model.model(input_ids=ids.to(P.device), attention_mask=att.to(P.device), use_cache=False).last_hidden_state
        for i, (p, t) in enumerate(rows):
            K = len(t)
            lp = F.log_softmax(P.model.lm_head(h[i, T - K - 1:T - 1]).float(), -1)
            res.append(float(lp.gather(-1, torch.tensor(t, device=lp.device)[:, None]).mean()))
    return res


class Intervene:
    """Edit one neuron's own activation (the input to down_proj) at every position. No free scale: x0 switches it
    off, x2 doubles what it already does, clamp sets it to its own strong-firing value measured on plain text."""
    def __init__(self, P):
        self.P, self.h = P, None

    def set(self, layer, n, mode, value=None):
        def pre(m, inp):
            x = inp[0].clone()
            if mode == "x0":
                x[..., n] = 0
            elif mode == "x2":
                x[..., n] = 2 * x[..., n]
            else:
                x[..., n] = value
            return (x,)
        self.h = self.P.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)

    def clear(self):
        if self.h:
            self.h.remove(); self.h = None


@torch.no_grad()
def strong_values(P, neurons, n_text=32, max_tok=256, q=0.99):
    """Per neuron: the signed activation at the q-quantile of |activation| over plain Wikipedia text (all positions)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"]
    texts = [t.strip() for t in ds if len(t.split()) > 60][:n_text]
    by_layer = {}
    for l, n in neurons:
        by_layer.setdefault(l, []).append(n)
    cap = {l: [] for l in by_layer}
    hooks = [P.layers[l].mlp.down_proj.register_forward_pre_hook(
        (lambda l: lambda m, inp: cap[l].append(inp[0][0][:, by_layer[l]].float().cpu()))(l)) for l in by_layer]
    try:
        for t in texts:
            ids = P.tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :max_tok].to(P.device)
            P.lm(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for l, ns in by_layer.items():
        X = torch.cat(cap[l])  # [tokens, len(ns)]
        thr = torch.quantile(X.abs(), q, dim=0)
        for j, n in enumerate(ns):
            v = X[:, j]; big = v[v.abs() >= thr[j]]
            out[(l, n)] = float(thr[j] * torch.sign(big.mean()))
    return out, int(sum(len(c) for c in cap[next(iter(cap))]))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(a):
    t0 = time.time()
    P = Patient("Qwen/Qwen3.5-4B")
    syco = json.load(open(a.syco))
    items = syco["dev"][:a.n_syco]
    halves_s = (items[: len(items) // 2], items[len(items) // 2:])
    probs = gsm8k(a.n_math)
    halves_m = (probs[: len(probs) // 2], probs[len(probs) // 2:])
    rng = random.Random(0)
    d_ff = P.cfg.intermediate_size
    neurons = [(l, n) for l in a.layers for n in rng.sample(range(d_ff), a.n_per_layer)]
    if (12, 2920) not in neurons:
        neurons.append((12, 2920))
    iv = Intervene(P)

    def measure():
        return ([pressure_eval(P, h, bs=64)["soft_pf"] for h in halves_s],
                [float(np.mean(math_logp(P, h))) for h in halves_m])

    base_s, base_m = measure()
    print("baseline soft_pf halves", base_s, "math logp halves", base_m, flush=True)
    strong, n_tok = strong_values(P, neurons)
    print("strong-firing values from", n_tok, "wikitext tokens; e.g. L12 N2920:", strong[(12, 2920)], flush=True)
    rows = []
    for k, (l, n) in enumerate(neurons):
        r = {"layer": l, "n": n, "strong": strong[(l, n)]}
        for name in ("x0", "x2", "clamp"):
            iv.set(l, n, name, strong[(l, n)])
            s, m = measure()
            iv.clear()
            r[name] = {"syco": [s[0] - base_s[0], s[1] - base_s[1]], "math": [m[0] - base_m[0], m[1] - base_m[1]]}
        rows.append(r)
        if k % 25 == 0 or k < 3:
            print(k, len(neurons), round(time.time() - t0), "s", json.dumps(r), flush=True)
            json.dump({"rows": rows}, open(a.out, "w"))
    summ = {}
    for name in ("x0", "x2", "clamp"):
        for beh in ("syco", "math"):
            A = np.array([r[name][beh][0] for r in rows]); B = np.array([r[name][beh][1] for r in rows])
            noise = float(np.std(A - B) / np.sqrt(2))
            mean = (A + B) / 2
            ref = next(i for i, r in enumerate(rows) if (r["layer"], r["n"]) == (12, 2920))
            order = np.argsort(-mean) if beh == "math" else np.argsort(mean)  # syco: most negative = most reducing
            summ[f"{name}/{beh}"] = {"split_half_spearman": spearman(A, B), "noise_sd": noise,
                                     "n_above_3x_noise": int((np.abs(mean) > 3 * noise).sum()),
                                     "max_abs_effect": float(np.abs(mean).max()),
                                     "ref_L12N2920_rank_of": [int(np.where(order == ref)[0][0]) + 1, len(rows)],
                                     "ref_effect": float(mean[ref])}
    # PCA-basis retention of the columns (top 128 PCs per layer)
    ret = {}
    for l in a.layers:
        W = P.layers[l].mlp.down_proj.weight.float().T  # [d_ff, d]
        U, Sv, Vh = torch.linalg.svd(W - W.mean(0), full_matrices=False)
        B = Vh[:128]
        cols = W[[n for (ll, n) in neurons if ll == l]]
        ret[l] = float(((cols @ B.T).norm(dim=1) / cols.norm(dim=1)).mean())
    summ["pca128_norm_retention"] = ret
    out = {"summary": summ, "rows": rows, "baseline": {"syco": base_s, "math": base_m}, "n_text_tokens": n_tok,
           "n_neurons": len(rows), "n_syco_items": len(items), "n_math": len(probs), "sec": time.time() - t0}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--syco", default="data/syco.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 24])
    ap.add_argument("--n_per_layer", type=int, default=100)
    ap.add_argument("--n_syco", type=int, default=96)
    ap.add_argument("--n_math", type=int, default=96)
    main(ap.parse_args())
