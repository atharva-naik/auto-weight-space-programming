"""Score the reference and every baseline on the test version through the task's verifier (scorer/score.py's
evaluate, the same code tests/score.py runs; the model is loaded once and each edit is applied and undone in place).

Baselines (one edit per location; all but the reference rewrite only the " Overlay" row of the tied embedding):
- noop: no change.
- random-row (10 seeds): the Overlay row plus a random direction with the same norm as the map edit's row change.
- mean-row: the mean of the 38 single-token state rows.
- knn3-row: the mean of the rows of the 3 single-token states nearest the target.
- map-quadratic: the row placed on a quadratic map of the single-token state rows (least squares of the rows on
  lat, lon, lat^2, lon^2, lat*lon), evaluated at the target (the earlier pilot's best map edit).
- reference: reference/edits/<loc>.pt.

Writes data/baselines.json (per location and pooled) and data/reference_scores.json (used by selfcheck.py).
Usage: WSA_MODEL_DIR=<snapshot dir> python baselines.py [--only reference,noop,...] [--locs loc01,...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scorer"))
import geodata  # noqa: E402
import score  # noqa: E402
from generate import test_tasks  # noqa: E402

N_SEEDS = 10


def emb_name(model):
    w = model.get_input_embeddings().weight
    return next(n for n, p in model.named_parameters() if p is w)


def row_edit(name, row_id, row, H):
    return {name: (torch.arange(row_id * H, (row_id + 1) * H), torch.as_tensor(row, dtype=torch.float32).clone())}


def quad_feats(C):
    C = np.atleast_2d(np.asarray(C, float)); la = C[:, 0] - 39.0; lo = C[:, 1] + 96.0
    return np.c_[la, lo, la ** 2 / 10, lo ** 2 / 10, la * lo / 10]


def nearest(c, pool, k):
    return sorted(pool, key=lambda s: geodata.dist_km(c, geodata.COORDS[s]))[:k]


def baseline_edits(model, tok, task):
    name = emb_name(model)
    W = model.get_input_embeddings().weight.detach()
    H = W.shape[1]
    st = {s: tok.encode(" " + s, add_special_tokens=False) for s in geodata.SINGLE_TOKEN}
    assert all(len(v) == 1 for v in st.values())
    E = {s: W[v[0]].double().numpy() for s, v in st.items()}
    X = np.stack([E[s] for s in geodata.SINGLE_TOKEN]); m = X.mean(0)
    F = quad_feats([geodata.COORDS[s] for s in geodata.SINGLE_TOKEN]); fm = F.mean(0)
    Bq = np.linalg.lstsq(F - fm, X - m, rcond=None)[0]
    c = (task["lat"], task["lon"])
    base = W[geodata.NEW_ID].double().numpy()
    mapq = m + ((quad_feats(c) - fm) @ Bq)[0]
    out = {"noop": {}, "mean-row": row_edit(name, geodata.NEW_ID, m, H),
           "knn3-row": row_edit(name, geodata.NEW_ID, np.mean([E[s] for s in nearest(c, geodata.SINGLE_TOKEN, 3)], 0), H),
           "map-quadratic": row_edit(name, geodata.NEW_ID, mapq, H)}
    norm = np.linalg.norm(mapq - base)
    for k in range(N_SEEDS):
        rng = np.random.default_rng(10_000 + k)
        v = rng.standard_normal(H); v *= norm / np.linalg.norm(v)
        out[f"random-row#{k}"] = row_edit(name, geodata.NEW_ID, base + v, H)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--locs", default="")
    a = ap.parse_args()
    tok, model = score.load_model(os.environ["WSA_MODEL_DIR"])
    params = score.param_table(model)
    path = os.path.join(HERE, "data", "baselines_raw.json")
    raw = json.load(open(path)) if os.path.exists(path) else {}
    raw["_canary"] = open(os.path.join(HERE, "..", "..", "CANARY")).read().strip()
    for task in test_tasks():
        if a.locs and task["id"] not in a.locs.split(","):
            continue
        edits = {"reference": score.read_edit(os.path.join(HERE, "reference", "edits", f"{task['id']}.pt"), params)}
        edits.update(baseline_edits(model, tok, task))
        cache = {}
        for bname, e in edits.items():
            if a.only and bname.split("#")[0] not in a.only.split(","):
                continue
            t0 = time.time()
            det = score.evaluate(model, tok, task, e, log=lambda m: None, cache=cache)
            raw.setdefault(task["id"], {})[bname] = det
            print(f"{task['id']} {bname:16s} reward {det['reward']:.3f} attr {det['components']['attributes']['score']:.3f} "
                  f"rel {det['components']['relations']['score']} gates {det['gates_pass']} "
                  f"({det['gates']['real_prompts']['unchanged']:.3f}, {det['gates']['neutral_kl']['mean_kl']:.1e}, "
                  f"{det['gates']['general_knowledge']['unchanged']:.3f}) n={det['n_changed']} {time.time() - t0:.0f}s", flush=True)
            json.dump(raw, open(path, "w"))
    summarize(raw)


def summarize(raw):
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    names = sorted({b.split("#")[0] for r in raw.values() for b in r})
    summ = {}
    for nm in names:
        per_loc = {}
        for lid, r in sorted(raw.items()):
            ds = [d for b, d in r.items() if b.split("#")[0] == nm]
            if ds:
                per_loc[lid] = dict(reward=float(np.mean([d["reward"] for d in ds])),
                                    attributes=float(np.mean([d["components"]["attributes"]["score"] for d in ds])),
                                    relations=float(np.mean([d["components"]["relations"]["score"] or 0 for d in ds])),
                                    gates_pass=float(np.mean([d["gates_pass"] for d in ds])), n_runs=len(ds))
        rs = [v["reward"] for v in per_loc.values()]
        summ[nm] = dict(mean_reward=float(np.mean(rs)), n_locations=len(rs), per_location=per_loc,
                        mean_attributes=float(np.mean([v["attributes"] for v in per_loc.values()])),
                        mean_relations=float(np.mean([v["relations"] for v in per_loc.values()])),
                        gate_pass_rate=float(np.mean([v["gates_pass"] for v in per_loc.values()])))
    canary = open(os.path.join(HERE, "..", "..", "CANARY")).read().strip()
    json.dump(dict(canary=canary, baselines=summ), open(os.path.join(HERE, "data", "baselines.json"), "w"), indent=1)
    if "reference" in summ:
        json.dump(dict(canary=canary, scores={lid: v["reward"] for lid, v in summ["reference"]["per_location"].items()}),
                  open(os.path.join(HERE, "data", "reference_scores.json"), "w"), indent=1)
    for nm, s in summ.items():
        print(f"{nm:16s} mean reward {s['mean_reward']:.3f} (attr {s['mean_attributes']:.3f}, rel {s['mean_relations']:.3f}, "
              f"gates {s['gate_pass_rate']:.2f}) over {s['n_locations']} locations")


if __name__ == "__main__":
    main()
