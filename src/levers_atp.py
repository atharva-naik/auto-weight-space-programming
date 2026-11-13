"""Attribution patching (AtP) for the levers task: first-order estimates of every neuron's edit effect on the two
pilot behaviours, from one forward+backward pass over each prompt half. Checked against the pilot's true edits.

For neuron n at layer l with activation a (input to down_proj) and gradient g = dM/da at each position:
  x0 estimate    = -sum_pos a*g
  x2 estimate    = +sum_pos a*g
  clamp estimate = v * sum_pos g - sum_pos a*g     (v = the neuron's strong-firing value)
M = soft_pf (sycophancy; the neutral-correct mask and first answer are fixed from the clean run) or the mean answer
log-prob (math). Same prompt halves as src/levers_pilot.py.

  python src/levers_atp.py --out runs/round4/levers_atp.npz --pilot runs/round4/levers_pilot.json runs/round4/levers_pilot_b.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patient import Patient, TEMPLATES, ARE_YOU_SURE, LET, user_msg  # noqa: E402
from levers_pilot import gsm8k, spearman  # noqa: E402


class Grab:
    """Keeps every layer's down_proj input with its gradient, and accumulates sum(a*g) and sum(g) per neuron."""
    def __init__(self, P):
        self.P, self.x, self.hooks = P, {}, []
        L, F = len(P.layers), P.cfg.intermediate_size
        self.ag = torch.zeros(L, F, dtype=torch.float64, device=P.device)
        self.g = torch.zeros(L, F, dtype=torch.float64, device=P.device)
        for l, layer in enumerate(P.layers):
            self.hooks.append(layer.mlp.down_proj.register_forward_pre_hook(self._mk(l)))
        self.hooks.append(P.lm.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_(True)))

    def _mk(self, l):
        def pre(m, inp):
            if inp[0].requires_grad:
                inp[0].retain_grad(); self.x[l] = inp[0]
        return pre

    def collect(self):
        for l, x in self.x.items():
            if x.grad is not None:
                xd, gd = x.detach().float(), x.grad.detach().float()
                self.ag[l] += (xd * gd).sum((0, 1)).double()
                self.g[l] += gd.sum((0, 1)).double()
        self.x = {}

    def reset(self):
        self.ag.zero_(); self.g.zero_()

    def remove(self):
        for h in self.hooks:
            h.remove()


def letters(P, texts, grad):
    enc = P.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.device)
    with torch.set_grad_enabled(grad):
        h = P.model.model(**enc, use_cache=False).last_hidden_state[:, -1]
        lp = P.model.lm_head(h).float().log_softmax(-1)
        sub = torch.stack([lp[:, ids].logsumexp(-1) for ids in P.letter_ids], -1)
        return sub.softmax(-1)


def atp_syco(P, G, items, bs=8):
    """Gradient of soft_pf (as in patient.pressure_eval) with the mask and first answers fixed from the clean run."""
    wt, dt = TEMPLATES["dev"], TEMPLATES["doubt"]
    W, D = len(wt), len(dt)
    with torch.no_grad():
        neu = torch.cat([letters(P, [P.chat([{"role": "user", "content": user_msg(it)}]) for it in items[s:s + bs]], False)
                         for s in range(0, len(items), bs)]).cpu()
    ans = torch.tensor([it["answer"] for it in items])
    ok = (neu.argmax(-1) == ans).float(); first = neu.argmax(-1)
    k = float(ok.sum().clamp_min(1))
    jobs = []  # (text, target letter, weight): M = sum weight * (1 - p(target))
    for i, it in enumerate(items):
        if not ok[i]:
            continue
        jobs += [(P.chat([{"role": "user", "content": user_msg(it, it["wrong"], t)}]), it["answer"], 1 / (3 * k * W)) for t in wt]
        jobs += [(P.chat([{"role": "user", "content": user_msg(it, it["answer"], t)}]), it["answer"], 1 / (3 * k * D)) for t in dt]
        m2 = [{"role": "user", "content": user_msg(it)}, {"role": "assistant", "content": LET[int(first[i])]},
              {"role": "user", "content": ARE_YOU_SURE}]
        jobs.append((P.chat(m2), int(first[i]), 1 / (3 * k)))
    G.reset(); M = 0.0
    for s in range(0, len(jobs), bs):
        b = jobs[s:s + bs]
        p = letters(P, [j[0] for j in b], True)
        tgt = torch.tensor([j[1] for j in b], device=p.device); w = torch.tensor([j[2] for j in b], device=p.device)
        loss = (w * (1 - p[torch.arange(len(b)), tgt])).sum()
        loss.backward(); G.collect(); M += float(loss.detach())
    return M


def atp_math(P, G, probs, bs=6):
    tok = P.tok
    G.reset(); M = 0.0
    for s in range(0, len(probs), bs):
        rows = []
        for pr in probs[s:s + bs]:
            prompt = P.chat([{"role": "user", "content": pr["q"] + "\nAnswer with only the final number."}])
            rows.append((tok(prompt, add_special_tokens=False)["input_ids"], tok(pr["a"], add_special_tokens=False)["input_ids"]))
        T = max(len(p) + len(t) for p, t in rows)
        ids = torch.full((len(rows), T), tok.pad_token_id, dtype=torch.long); att = torch.zeros_like(ids)
        for i, (p, t) in enumerate(rows):
            ids[i, T - len(p) - len(t):] = torch.tensor(p + t); att[i, T - len(p) - len(t):] = 1
        h = P.model.model(input_ids=ids.to(P.device), attention_mask=att.to(P.device), use_cache=False).last_hidden_state
        loss = 0
        for i, (p, t) in enumerate(rows):
            K = len(t)
            lp = P.model.lm_head(h[i, T - K - 1:T - 1]).float().log_softmax(-1)
            loss = loss + lp.gather(-1, torch.tensor(t, device=lp.device)[:, None]).mean() / len(probs)
        loss.backward(); G.collect(); M += float(loss.detach())
    return M


def main(a):
    t0 = time.time()
    if a.compare_only:
        out = dict(np.load(a.out))
    else:
        out = compute(a, t0)
    compare(a, out, t0)


def compute(a, t0):
    P = Patient("Qwen/Qwen3.5-4B")
    syco = json.load(open(a.syco)); items = syco["dev"][:a.n_syco]
    halves_s = (items[: len(items) // 2], items[len(items) // 2:])
    probs = gsm8k(a.n_math); halves_m = (probs[: len(probs) // 2], probs[len(probs) // 2:])
    G = Grab(P)
    out = {}
    for beh, halves, fn in (("syco", halves_s, atp_syco), ("math", halves_m, atp_math)):
        for h in (0, 1):
            M = fn(P, G, halves[h])
            out[f"{beh}_ag_{h}"] = G.ag.cpu().numpy(); out[f"{beh}_g_{h}"] = G.g.cpu().numpy()
            print(beh, "half", h, "metric", round(M, 4), round(time.time() - t0), "s", flush=True)
    G.remove()
    np.savez_compressed(a.out, **out)
    return out


def compare(a, out, t0):
    """Correlate the AtP estimates with the pilot's true edits (per neuron and prompt half)."""
    rows = []
    for f in a.pilot:
        if os.path.exists(f):
            rows += [r for r in json.load(open(f))["rows"] if "x0" in r]
    if not rows:
        print("no pilot rows yet"); return
    rep = {"n_neurons": len(rows)}
    for beh in ("syco", "math"):
        for name in ("x0", "x2", "clamp"):
            est, true = [], []
            for r in rows:
                l, n = r["layer"], r["n"]
                for h in (0, 1):
                    ag, g = out[f"{beh}_ag_{h}"][l, n], out[f"{beh}_g_{h}"][l, n]
                    e = {"x0": -ag, "x2": ag, "clamp": r["strong"] * g - ag}[name]
                    est.append(e); true.append(r[name][beh][h])
            est, true = np.array(est), np.array(true)
            top = np.argsort(-np.abs(true))[: max(1, len(true) // 50)]
            rep[f"{name}/{beh}"] = {"spearman": spearman(est, true), "pearson": float(np.corrcoef(est, true)[0, 1]),
                                    "sign_agree_top2pct": float(np.mean(np.sign(est[top]) == np.sign(true[top])))}
    rep["sec"] = time.time() - t0
    json.dump(rep, open(a.out.replace(".npz", "_vs_pilot.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", nargs="*", default=[])
    ap.add_argument("--syco", default="data/syco.json")
    ap.add_argument("--n_syco", type=int, default=96)
    ap.add_argument("--n_math", type=int, default=96)
    ap.add_argument("--compare_only", action="store_true")
    main(ap.parse_args())
