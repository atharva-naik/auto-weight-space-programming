"""How hard is each 128-coordinate slice to read? Weight-only statistics for all 20 slices of Qwen3.5-4B.

Per slice S (coordinates s..s+127), over 300 test neurons:
  eff_rank     effective rank (participation ratio of singular values) of (1+w)[S] * U[:, S] over kept tokens
  distinct     distinct label tokens (top-8 promoted) across the 300 neurons; top_count = most common token's count
  entropy      entropy (bits) of the label-token distribution
  overlap      mean Jaccard overlap between the label sets of random pairs of neurons from the same layer
               (higher = candidates look more alike in the 4-way test)
  rounding     share of label tokens kept by the rounded integers' own lens

Usage: python src/slice_hardness.py runs/round4/slice_hardness.json
"""
import argparse
import collections
import json
import math
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
import raw_reader as R  # noqa: E402


def main(out):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = argparse.Namespace(tiny=False, device=dev, model="Qwen/Qwen3.5-4B", multi_gpu=False)
    R.ARGS = argparse.Namespace(input_mode="numbers")
    tok, m, lm = R.load(a)
    lens = R.Lens(tok, m, lm, dev)
    sp = json.load(open("data/raw_splits.json"))
    items = sp["eval"]["test_neuron"][:300]
    keep = ~lens.base_bad
    rows = []
    rng = random.Random(0)
    for s in range(0, lens.d, 128):
        S = slice(s, s + 128)
        Us = (lens.U[:, S] * lens.g[S])[keep]
        sv = torch.linalg.svdvals(Us.float())
        eff = float(sv.sum() ** 2 / (sv ** 2).sum())
        labs = lens.top([(it["layer"], it["n"], s) for it in items], 128, k=8, raw=True)
        cnt = collections.Counter(t for u, _ in labs for t in u)
        p = np.array(list(cnt.values()), float); p /= p.sum()
        ent = float(-(p * np.log2(p)).sum())
        by_layer = collections.defaultdict(list)
        for it, (u, _) in zip(items, labs):
            by_layer[it["layer"]].append(set(u))
        ov = []
        for l, sets in by_layer.items():
            for _ in range(20):
                if len(sets) > 1:
                    x, y = rng.sample(sets, 2)
                    ov.append(len(x & y) / len(x | y))
        kept = []
        for it, (u, _) in zip(items, labs):
            col = lens.col(it["layer"], it["n"])
            q = torch.tensor(R.quantize(col), dtype=torch.float, device=col.device)
            V = torch.zeros(lens.d, device=dev); V[S] = q[S]
            sc = (V * lens.g) @ lens.U.T; sc[lens.base_bad] = -math.inf
            kept.append(len(set(u) & set(sc.topk(8).indices.tolist())) / 8)
        r = {"start": s, "eff_rank": eff, "distinct": len(cnt), "top_count": cnt.most_common(1)[0][1], "entropy": ent,
             "overlap": float(np.mean(ov)), "rounding": float(np.mean(kept))}
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
