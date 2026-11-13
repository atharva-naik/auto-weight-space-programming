"""Mechanism probe for the raw-weight reader (ROUND4_LOG.md, 2026-11-04 17:45 entry; criteria fixed before running).

Does the reader rebuild the shown slice in its residual stream? At the first answer slot (prompt + 'PROMOTES: ["'),
take each layer's residual h_l (and the residual just before the final norm, the primary readout). Restricted to
the shown coordinates S, compare cos(h_l[S], v[S]) for the true neuron vs its 3 same-layer distractors, and
compute internal 4-way accuracy (chance 25%). Run for the base model (adapter disabled), every kept checkpoint
(<run>/ckpts/step*), and the final adapter.

Usage: python src/mech_probe.py --run runs/round4/sweep_4B_s1280 --start 1280 --out runs/round4/mech_s1280.json
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import raw_reader as R  # noqa: E402


def boot_ci(x, B=10000, seed=0):
    x = np.asarray(x, float)
    bs = x[np.random.default_rng(seed).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


@torch.no_grad()
def probe(tok, model, lm, lens, items, size, bs=8, device="cuda"):
    pre = R.encode(tok, "", 'PROMOTES: ["', eos=False)[1]
    cap = {}
    hook = lm.norm.register_forward_pre_hook(lambda m, inp: cap.__setitem__("pre_norm", inp[0]))
    per_layer_true, per_layer_ctrl, correct = None, None, []
    rows = []
    for i0 in range(0, len(items), bs):
        chunk = items[i0:i0 + bs]
        seqs = [R.encode(tok, R.make_prompt(tok, lens, it, "h64"), "")[0] + pre for it in chunk]
        L = max(map(len, seqs))
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, x in enumerate(seqs):
            ids[j, L - len(x):] = torch.tensor(x)
            att[j, L - len(x):] = 1
        out = lm(input_ids=ids.to(device), attention_mask=att.to(device), use_cache=False, output_hidden_states=True)
        hs = [h[:, -1].float() for h in out.hidden_states[:-1]] + [cap["pre_norm"][:, -1].float()]  # last = pre-norm
        for j, it in enumerate(chunk):
            S = slice(it["start"], it["start"] + size)
            vt = lens.col(it["layer"], it["n"])[S]
            vd = [lens.col(it["layer"], c)[S] for c in it["distractors"]]
            ct = [float(torch.nn.functional.cosine_similarity(h[j][S].to(vt.device)[None], vt[None])) for h in hs]
            cd = [[float(torch.nn.functional.cosine_similarity(h[j][S].to(v.device)[None], v[None])) for v in vd]
                  for h in hs]
            rows.append({"true": ct, "ctrl": cd})
    hook.remove()
    nL = len(rows[0]["true"])
    res = {"layer_cos_true": [float(np.mean([r["true"][l] for r in rows])) for l in range(nL)],
           "layer_cos_ctrl": [float(np.mean([np.mean(r["ctrl"][l]) for r in rows])) for l in range(nL)],
           "layer_internal_acc": [float(np.mean([r["true"][l] > max(r["ctrl"][l]) for r in rows])) for l in range(nL)]}
    prim = [float(r["true"][-1] > max(r["ctrl"][-1])) for r in rows]
    res["prenorm_internal_acc_ci"] = boot_ci(prim)
    res["prenorm_cos_gap_ci"] = boot_ci([r["true"][-1] - np.mean(r["ctrl"][-1]) for r in rows])
    return res


def main(a):
    args = argparse.Namespace(tiny=a.tiny, device=a.device, model=a.model, multi_gpu=False)
    R.ARGS = argparse.Namespace(input_mode="numbers")
    tok, model, lm = R.load(args)
    lens = R.Lens(tok, model, lm, a.device)
    from raw_splits import default_path
    sp = json.load(open(default_path(a.model)))
    size = 128 if not a.tiny else 32
    items = [dict(it, start=a.start, size=size, id=f"{it['layer']}_{it['n']}_{size}_{a.start}")
             for it in sp["eval"]["test_neuron"][:a.n]]
    if a.tiny:
        items = [dict(it, layer=it["layer"] % 4, n=it["n"] % 96, distractors=[x % 96 for x in it["distractors"]])
                 for it in items]
    ckpts = sorted(glob.glob(f"{a.run}/ckpts/step*"), key=lambda p: int(re.findall(r"\d+$", p)[0]))
    final = f"{a.run}/adapter_stage0" if os.path.exists(f"{a.run}/adapter_stage0") else None
    order = [(int(re.findall(r"\d+$", p)[0]), p) for p in ckpts] + ([("final", final)] if final else [])
    from peft import PeftModel
    out = {"run": a.run, "start": a.start, "n": len(items), "results": {}}
    val = {}
    if os.path.exists(f"{a.run}/log.jsonl"):
        val = {x["step"]: x["correct_pmi"] for x in map(json.loads, open(f"{a.run}/log.jsonl")) if x["event"] == "val_mid"}
    pm = None
    for k, (step, path) in enumerate(order):
        name = f"s{step}"
        if pm is None:
            pm = PeftModel.from_pretrained(model, path, adapter_name=name)
            with pm.disable_adapter():
                out["results"]["base"] = probe(tok, pm, lm, lens, items, size, device=a.device)
            print("base", out["results"]["base"]["prenorm_internal_acc_ci"], flush=True)
        else:
            pm.load_adapter(path, adapter_name=name)
        pm.set_adapter(name)
        r = probe(tok, pm, lm, lens, items, size, device=a.device)
        r["val_acc_at_step"] = val.get(step)
        out["results"][str(step)] = r
        print(step, "val", val.get(step), "internal", r["prenorm_internal_acc_ci"], "gap", r["prenorm_cos_gap_ci"], flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(ap.parse_args())
