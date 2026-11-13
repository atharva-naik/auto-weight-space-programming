"""Levers, step 2 (after the pilot): how big can the sycophancy lever get, and how predictable is it from weights?

1. Strong-firing value v for every neuron (signed 99th percentile of |activation| on plain Wikipedia text).
2. AtP clamp estimate for every neuron, from runs/round4/levers_atp.npz (both prompt halves summed):
   clamp_est = v * sum(g) - sum(a*g)   (change in soft_pf; negative = less sycophantic)
3. Group edits on FRESH items (syco dev_b, never used by the pilot or AtP): clamp the top-k neurons with the most
   negative estimate together (and the most positive), plus random-k controls (3 draws). Report soft_pf, pf,
   acc_neutral and acc_suggest_correct for each.
4. Predictability from weights: ridge regression from the down_proj column (full 2,560, and coordinates 0-127 only)
   to the clamp estimate, trained on all layers except the reader's held-out layers (5, 13, 21, 29), scored there.

  python src/levers_groups.py --out runs/round4/levers_groups.json
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patient import Patient, pressure_eval  # noqa: E402
from levers_pilot import spearman  # noqa: E402

HELD = (5, 13, 21, 29)


@torch.no_grad()
def strong_all(P, n_text=32, max_tok=256, q=0.99):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"]
    texts = [t.strip() for t in ds if len(t.split()) > 60][:n_text]
    L = len(P.layers)
    cap = {l: [] for l in range(L)}
    hooks = [P.layers[l].mlp.down_proj.register_forward_pre_hook(
        (lambda l: lambda m, inp: cap[l].append(inp[0][0].float()))(l)) for l in range(L)]
    try:
        for t in texts:
            ids = P.tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :max_tok].to(P.device)
            P.lm(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    V = np.zeros((L, P.cfg.intermediate_size), dtype=np.float32)
    for l in range(L):
        X = torch.cat(cap[l]); cap[l] = None
        thr = torch.quantile(X.abs(), q, dim=0)
        sign = torch.sign(((X.abs() >= thr) * X).sum(0))
        V[l] = (thr * sign).cpu().numpy()
    return V


class Clamp:
    def __init__(self, P):
        self.P, self.hooks = P, []

    def set(self, neurons, V):
        by = {}
        for l, n in neurons:
            by.setdefault(l, []).append(n)
        for l, ns in by.items():
            idx = torch.tensor(ns, device=self.P.device)
            val = torch.tensor(V[l, ns], device=self.P.device)

            def pre(m, inp, idx=idx, val=val):
                x = inp[0].clone(); x[..., idx] = val.to(x.dtype)
                return (x,)
            self.hooks.append(self.P.layers[l].mlp.down_proj.register_forward_pre_hook(pre))

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def ridge_eval(P, est, lam, coords=None):
    L = len(P.layers)
    Xtr, ytr, Xte, yte = [], [], [], []
    for l in range(L):
        W = P.layers[l].mlp.down_proj.weight.float().T.cpu().numpy()  # [d_ff, d]
        if coords is not None:
            W = W[:, coords]
        (Xte if l in HELD else Xtr).append(W); (yte if l in HELD else ytr).append(est[l])
    Xtr, ytr, Xte, yte = np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(Xte), np.concatenate(yte)
    s = Xtr.std(0) + 1e-8; Xtr = Xtr / s; Xte = Xte / s
    mu = ytr.mean()
    A = torch.tensor(Xtr, device=P.device, dtype=torch.float64)
    w = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=P.device, dtype=torch.float64),
                           A.T @ torch.tensor(ytr - mu, device=P.device, dtype=torch.float64)).cpu().numpy()
    pred = Xte @ w + mu
    r2 = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    k = max(1, len(yte) // 100)
    top_true = set(np.argsort(yte)[:k]); top_pred = set(np.argsort(pred)[:k])
    return {"r2": float(r2), "spearman": spearman(pred, yte), "top1pct_overlap": len(top_true & top_pred) / k}


def main(a):
    t0 = time.time()
    P = Patient("Qwen/Qwen3.5-4B")
    V = strong_all(P)
    print("strong values done", round(time.time() - t0), "s", flush=True)
    Z = np.load(a.atp)
    ag, g = Z["syco_ag_0"] + Z["syco_ag_1"], Z["syco_g_0"] + Z["syco_g_1"]
    est = V * g - ag  # [L, d_ff]; units: soft_pf change summed over the two halves -> halve for the 96-item mean
    est = est / 2
    np.save(a.out.replace(".json", "_clamp_est.npy"), est)
    L, F = est.shape
    flat = np.argsort(est, axis=None)
    neg = [divmod(int(i), F) for i in flat[: max(a.ks)]]
    pos = [divmod(int(i), F) for i in flat[::-1][: max(a.ks)]]
    syco = json.load(open(a.syco)); items = syco["dev_b"][: a.n_items]
    C = Clamp(P)
    keys = ("soft_pf", "pf", "acc_neutral", "acc_suggest_correct")

    def ev(neurons):
        C.set(neurons, V)
        r = pressure_eval(P, items, bs=64)
        C.clear()
        return {k: r[k] for k in keys}

    out = {"n_items": len(items), "items": "syco dev_b (fresh)", "baseline": ev([])}
    print("baseline", out["baseline"], flush=True)
    rng = random.Random(0)
    allids = [(l, n) for l in range(L) for n in range(F)]
    groups = {}
    for k in a.ks:
        groups[f"down_{k}"] = ev(neg[:k])
        groups[f"up_{k}"] = ev(pos[:k])
        groups[f"random_{k}"] = [ev(rng.sample(allids, k)) for _ in range(3)]
        print(k, "down", groups[f"down_{k}"], "up", groups[f"up_{k}"], "random", groups[f"random_{k}"], round(time.time() - t0), "s", flush=True)
        json.dump(dict(out, groups=groups), open(a.out, "w"), indent=1)
    out["groups"] = groups
    out["top_down_neurons"] = neg[:64]; out["top_up_neurons"] = pos[:64]
    out["layer_hist_down64"] = np.bincount([l for l, _ in neg[:64]], minlength=L).tolist()
    out["ref_L12N2920"] = {"clamp_est": float(est[12, 2920]), "rank_down": int(np.where(flat == 12 * F + 2920)[0][0]) + 1, "of": L * F}
    probe = {}
    for name, coords in (("full_column", None), ("coords_0_127", list(range(128)))):
        for lam in (1e2, 1e4):
            probe[f"{name}_lam{lam:g}"] = ridge_eval(P, est, lam, coords)
            print("probe", name, lam, probe[f"{name}_lam{lam:g}"], flush=True)
    out["probe_heldout_layers"] = probe
    out["sec"] = time.time() - t0
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("baseline", "ref_L12N2920", "probe_heldout_layers", "layer_hist_down64")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--atp", default="runs/round4/levers_atp.npz")
    ap.add_argument("--syco", default="data/syco.json")
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    main(ap.parse_args())
