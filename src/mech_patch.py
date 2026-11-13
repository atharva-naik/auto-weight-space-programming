"""Confound checks for the causal ablation (src/mech_causal.py).

1. General damage: does zeroing coordinates C of the residual stream (same layers, last K+1 positions) hurt the
   UNTRAINED model's next-token prediction on ordinary text? Compare C = 0-127 with C = 1280-1407.
2. Interchange (swap) patching on the trained reader: for pairs of val neurons (i, j), score item j's prompt with the
   two candidate labels {label_i, label_j}, but replace residual coordinates C at the answer positions in layers
   17-31 with those from the run on item i's prompt (same candidate tokens, so positions align). If the reader's
   information lives in C, the choice should flip from label_j to label_i. Damage alone would not make it pick i.

  python src/mech_patch.py --adapter runs/round4/sweep_4B/adapter_stage0 --out runs/round4/mech_patch_4B.json
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import raw_reader as R  # noqa: E402


def seq(tok, prompt, target):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(target, add_special_tokens=False)["input_ids"]
    return p, t


@torch.no_grad()
def run(lm, model, ids, layers, coords, K, mode, cache=None, device="cuda"):
    """mode: None | 'zero' | 'record' | 'inject'. Acts on the last K+1 positions of a single sequence."""
    hs = []
    if mode:
        def mk(l):
            def hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                if mode == "record":
                    cache[l] = h[:, -(K + 1):, coords].clone()
                    return o
                h = h.clone()
                h[:, -(K + 1):, coords] = 0 if mode == "zero" else cache[l]
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            return hook
        hs = [lm.layers[l].register_forward_hook(mk(l)) for l in layers]
    try:
        x = torch.tensor([ids], device=device)
        return lm(input_ids=x, use_cache=False).last_hidden_state
    finally:
        for h in hs:
            h.remove()


def logp_of(model, hid, t):
    K = len(t)
    lg = F.log_softmax(model.lm_head(hid[0, -(K + 1):-1]).float(), -1)
    return float(lg.gather(-1, torch.tensor(t, device=lg.device)[:, None]).sum())


def main(a):
    args = argparse.Namespace(tiny=False, device=a.device, model=a.model, multi_gpu=False)
    R.ARGS = argparse.Namespace(input_mode="numbers", seed=0)
    tok, model, lm = R.load(args)
    lens = R.Lens(tok, model, lm, a.device)
    L = len(lm.layers)
    late, early = list(range(17, L)), list(range(0, 17))
    S, C = list(range(0, 128)), list(range(1280, 1408))
    out = {}
    # 1. general damage on ordinary text, untrained model
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["validation"]["text"]
    texts = [t.strip() for t in ds if len(t.split()) > 40][:a.n_text]
    dmg = {}
    for name, layers, coords in [("S_late", late, S), ("C_late", late, C), ("S_early", early, S), ("C_early", early, C)]:
        drops = []
        for t in texts:
            ids = tok(t, add_special_tokens=False)["input_ids"][:200]
            K = 16
            tgt = ids[-K:]
            base = logp_of(model, run(lm, model, ids, [], [], K, None, device=a.device), tgt)
            abl = logp_of(model, run(lm, model, ids, layers, coords, K, "zero", device=a.device), tgt)
            drops.append((base - abl) / K)
        dmg[name] = float(np.mean(drops))
        print("damage", name, round(dmg[name], 3), "nats/token", flush=True)
    out["general_damage_nats_per_token"] = dmg
    # 2. interchange patching on the trained reader
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
    art = json.load(open(os.path.join(os.path.dirname(a.adapter), "artifacts_128.json")))
    bu = torch.zeros(lens.U.shape[0], dtype=torch.bool); bd = bu.clone()
    for i, s_ in enumerate(lens.strs):
        if s_ in art["up"]: bu[i] = True
        if s_ in art["down"]: bd[i] = True
    lens.artifacts[128] = (bu.to(a.device), bd.to(a.device))
    sp = json.load(open("data/raw_splits.json"))
    items = [dict(it, start=0, size=128, id=f"{it['layer']}_{it['n']}_128_0") for it in sp["eval"]["val_128"][:2 * a.n_pairs]]
    labs = lens.top([(it["layer"], it["n"], 0) for it in items], 128)
    tgt_text = [R.target_text(lens.label_strings(u), lens.label_strings(d)) for u, d in labs]
    prompts = [R.make_prompt(tok, lens, it, "h64") for it in items]
    res = {}
    for name, coords in [("patch_S", S), ("patch_C", C), ("no_patch", None)]:
        picks_i = []
        for k in range(a.n_pairs):
            i, j = 2 * k, 2 * k + 1
            score = {}
            for cand, lab in (("i", tgt_text[i]), ("j", tgt_text[j])):
                pj, t = seq(tok, prompts[j], lab)
                K = len(t)
                if coords is None:
                    hid = run(lm, model, pj + t, [], [], K, None, device=a.device)
                else:
                    cache = {}
                    pi, _ = seq(tok, prompts[i], lab)
                    run(lm, model, pi + t, late, coords, K, "record", cache, device=a.device)
                    hid = run(lm, model, pj + t, late, coords, K, "inject", cache, device=a.device)
                score[cand] = logp_of(model, hid, t)
            picks_i.append(int(score["i"] > score["j"]))
        res[name] = float(np.mean(picks_i))
        print("interchange", name, "picks source label i:", round(res[name], 3), flush=True)
    out["interchange_picks_source"] = res
    out["n_pairs"] = a.n_pairs
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--n_text", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
