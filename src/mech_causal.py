"""Causal test of the reader's mechanism: does zeroing the shown coordinates of the residual stream at the answer
positions break the four-way choice, while zeroing other coordinates does not?

The alignment probe (src/mech_probe.py) found that the trained reader's residual stream, restricted to the shown
coordinates S = 0..127, points along the shown slice from layer 17 on. Here we intervene: at the positions that
predict the answer tokens (the last K+1 positions of every scored sequence), set residual coordinates C to zero at
the outputs of layers in a range, for both the with-numbers and the numbers-hidden prompts (so PMI stays fair).

Conditions: no ablation; C = S (0-127) at layers 17-31; C = S at the last layer only; C = 1280-1407 at layers 17-31
(control, same size); C = S at layers 0-16 (answer positions only). 200 val items (test split untouched).

  python src/mech_causal.py --adapter runs/round4/sweep_4B/adapter_stage0 --out runs/round4/mech_causal_4B.json
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import raw_reader as R  # noqa: E402

STATE = {"k": None}


@contextlib.contextmanager
def ablate(lm, layers, coords):
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        k = STATE["k"]
        h = h.clone()
        h[:, -k:, coords] = 0
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    hs = [lm.layers[l].register_forward_hook(hook) for l in layers]
    try:
        yield
    finally:
        for x in hs:
            x.remove()


def main(a):
    args = argparse.Namespace(tiny=False, device=a.device, model=a.model, multi_gpu=False)
    R.ARGS = argparse.Namespace(input_mode="numbers", seed=0)
    tok, model, lm = R.load(args)
    lens = R.Lens(tok, model, lm, a.device)
    art = json.load(open(os.path.join(os.path.dirname(a.adapter), "artifacts_128.json")))
    bu = torch.zeros(lens.U.shape[0], dtype=torch.bool); bd = bu.clone()
    for i, s_ in enumerate(lens.strs):
        if s_ in art["up"]: bu[i] = True
        if s_ in art["down"]: bd[i] = True
    lens.artifacts[128] = (bu.to(a.device), bd.to(a.device))
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
    sp = json.load(open("data/raw_splits.json"))
    items = [dict(it, start=0, size=128, id=f"{it['layer']}_{it['n']}_128_0") for it in sp["eval"]["val_128"][:a.n]]
    # make batch_logp expose the number of answer positions to the hook
    orig = R.batch_logp
    def patched(lm_, model_, seqs, device):
        STATE["k"] = max(len(t) for _, t in seqs) + 1
        return orig(lm_, model_, seqs, device)
    R.batch_logp = patched
    L = len(lm.layers)
    S = list(range(0, 128)); C = list(range(1280, 1408))
    conds = {"none": None, "S_layers17-31": (range(17, L), S), "S_last_layer": (range(L - 1, L), S),
             "control_1280_layers17-31": (range(17, L), C), "S_layers0-16": (range(0, 17), S)}
    args2 = argparse.Namespace(eval_tokens=40000, seed=0, device=a.device)
    out = {}
    for name, c in conds.items():
        ctx = contextlib.nullcontext() if c is None else ablate(lm, c[0], c[1])
        with ctx:
            recs = R.eval_t2(tok, model, lm, lens, [dict(x) for x in items], "h64", "normal", args2)
        acc = float(np.mean([r["correct_pmi"] for r in recs]))
        out[name] = {"acc_pmi": acc, "acc_raw": float(np.mean([r["correct_raw"] for r in recs])), "n": len(recs)}
        print(name, out[name], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
