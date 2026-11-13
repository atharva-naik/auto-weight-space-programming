"""Build the public Auto Weight-Space Programming datasets from model weights (no GPU needed).

  python src/release_datasets.py columns --model Qwen/Qwen3.5-4B --run runs/round4/sweep_4B --out release
  python src/release_datasets.py reading --model Qwen/Qwen3.5-4B --run runs/round4/sweep_4B --out release
  python src/release_datasets.py opsd --out release

columns: one row per MLP neuron: split, its full down_proj column as integers (20 = one std of the column, the
         format the reader sees), the column's std, and exact logit-lens labels (full column and 128-wide slices).
reading: the reading task as text: a training set (one item per training neuron, coordinates 0-127) and the 4-way
         evaluation sets used in the paper (val, test_neuron, test_neuron_hard, test_layer), with candidates and answers.
opsd:    the open-ended description data (src/opsd_data.py, unchanged), built on CPU through the same weight shim.

Every label, prompt and candidate order comes from src/raw_reader.py's own code (Lens, fit_artifacts, quantize,
numbers_text, user_prompt, target_text, hard_distractors, the seeded candidate shuffle). The weights are read from
the safetensors files directly, so no full model is loaded. `verify` checks the output against a run's saved files.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import raw_reader as R  # noqa: E402

SLICE = 128


# ------------------------------------------------------------------ weights (safetensors, no model load)
def load_shim(model_id, device):
    """Returns (tok, model, lm) shaped like raw_reader.load's: lm.layers[i].mlp.down_proj.weight, lm.norm, model.lm_head."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"])
    cfg = json.load(open(os.path.join(path, "config.json")))
    t = cfg.get("text_config", cfg)
    where = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                where[k] = f

    def get(k):
        with safe_open(where[k], "pt") as fh:
            return fh.get_tensor(k)

    def one(pattern):
        ks = [k for k in where if re.search(pattern, k) and "visual" not in k and "mtp" not in k]
        assert len(ks) == 1, (pattern, ks[:5])
        return ks[0]

    L = t["num_hidden_layers"]
    W = [get(one(rf"(^|\.)layers\.{i}\.mlp\.down_proj\.weight$")) for i in range(L)]
    norm_w = get(one(r"^model\.(language_model\.)?norm\.weight$"))
    head = [k for k in where if k == "lm_head.weight"]
    U = get(head[0] if head else one(r"^model\.(language_model\.)?embed_tokens\.weight$"))
    # The real norm class, so Lens reads its forward code exactly as in training: Qwen3.5 scales by (1 + w), Llama by w.
    if t.get("model_type", cfg.get("model_type", "")).startswith("qwen3_5"):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm as Norm
    else:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm as Norm
    norm = Norm(norm_w.shape[0], eps=t.get("rms_norm_eps", 1e-6))
    with torch.no_grad():
        norm.weight.copy_(norm_w)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(weight=w))) for w in W]
    lm = SimpleNamespace(layers=layers, norm=norm)
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=U))
    return tok, model, lm


def setup(model_id, device, splits_path=""):
    tok, model, lm = load_shim(model_id, device)
    lens = R.Lens(tok, model, lm, device)
    # The paper's runs used transformers 5.20.2, whose tokenizers mark more ids as special (for Qwen3.5: 17, against
    # 9 in 5.7.0, e.g. <|box_end|>). Lens drops tok.all_special_ids, so filter on the paper's exact set instead.
    paper = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "special_ids_paper.json")))
    if model_id in paper:
        ids = torch.tensor(paper[model_id]["special"], dtype=torch.long)
        lens.base_bad[ids.to(lens.base_bad.device)] = True
    from raw_splits import default_path
    sp = json.load(open(splits_path or default_path(model_id)))
    R.ARGS = argparse.Namespace(input_mode="numbers", seed=0, d=lens.d, d_ff=lens.W[0].shape[1], tiny=False,
                                train_pool=0, fixed_start=0)
    return tok, lens, sp


def fit(lens, sp, size, start):
    """Artifact filter exactly as raw_reader.main fits it (4000 training items, rng 'art{size}', fixed start)."""
    R.ARGS.fixed_start = start
    sample = R.train_items(sp, size, 4000, random.Random(f"art{size}"))
    arts = lens.fit_artifacts([(it["layer"], it["n"], it["start"]) for it in sample], size)
    return arts, lens.artifacts[size]


def split_of(sp, l, n):
    if l in sp["held_layers"]:
        return "test_layer"
    if n in sp["_tn"].get(l, ()):
        return "test_neuron"
    if n in sp["_val"].get(l, ()):
        return "val"
    return "train"


def index_pools(sp):
    sp["_tn"] = {int(l): set(v) for l, v in sp["pools"]["test_neuron"].items()}
    sp["_val"] = {int(l): set(v) for l, v in sp["pools"]["val"].items()}


# ------------------------------------------------------------------ columns
def build_columns(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    tok, lens, sp = setup(a.model, a.device, a.splits)
    index_pools(sp)
    L, d, F = len(lens.W), lens.d, lens.W[0].shape[1]
    out = os.path.join(a.out, "columns", a.name)
    os.makedirs(out, exist_ok=True)
    slices = [0] + ([640, 1280, 2048] if a.transfer_slices else [])
    arts = {}
    for s in slices:  # fitting changes lens.artifacts[128]; keep each start's masks
        strs, masks = fit(lens, sp, SLICE, s)
        arts[f"slice_{s}"] = (strs, masks)
    full_strs, _ = fit(lens, sp, d, 0)
    meta = {"model": a.model, "layers": L, "d_model": d, "d_ff": F, "scale_std": R.SCALE_STD, "norm": lens.norm_param,
            "held_layers": sp["held_layers"], "artifact_filters": {k: v[0] for k, v in arts.items()} | {"full": full_strs},
            "include_columns": not a.no_columns}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    for l in range(L):
        t0 = time.time()
        rows = {"layer": [l] * F, "neuron": list(range(F)), "split": [split_of(sp, l, n) for n in range(F)]}
        q = None
        if not a.no_columns:
            q = np.empty((F, d), dtype=np.int16)
            for n in range(F):
                v = np.asarray(R.quantize(lens.col(l, n)))
                assert np.abs(v).max() < 32767
                q[n] = v
        rows["column_std"] = [float(lens.col(l, n).std()) for n in range(F)]
        res = lens.top([(l, n, 0) for n in range(F)], d)
        rows["full_promotes"] = [lens.label_strings(u) for u, _ in res]
        rows["full_suppresses"] = [lens.label_strings(v) for _, v in res]
        for s in slices:
            lens.artifacts[SLICE] = arts[f"slice_{s}"][1]
            res = lens.top([(l, n, s) for n in range(F)], SLICE)
            rows[f"slice_{s}_promotes"] = [lens.label_strings(u) for u, _ in res]
            rows[f"slice_{s}_suppresses"] = [lens.label_strings(v) for _, v in res]
        schema = {"layer": pa.int16(), "neuron": pa.int32(), "split": pa.string(), "column_std": pa.float32()}
        cols = {k: pa.array(v, type=schema.get(k)) for k, v in rows.items()}
        if q is not None:
            cols = {"layer": cols.pop("layer"), "neuron": cols.pop("neuron"), "split": cols.pop("split"),
                    "column": pa.FixedSizeListArray.from_arrays(pa.array(q.ravel(), type=pa.int16()), d)} | cols
        tbl = pa.table(cols)
        pq.write_table(tbl, os.path.join(out, f"layer_{l:02d}.parquet"), compression="zstd")
        print(f"layer {l}: {F} neurons, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ reading task
def content(lens, it, cond="normal", seed=0):
    """The user message of raw_reader.make_prompt (h64 format), before the chat template."""
    l, n, s, size = it["layer"], it["n"], it["start"], it["size"]
    perm = None
    if cond == "scrambled":
        perm = list(range(size))
        random.Random(f"{seed}-{it['id']}").shuffle(perm)
    return R.user_prompt(R.numbers_text(R.quantize(lens.col(l, n)), s, size, 64, perm), s, size, lens.d)


def four_way(lens, items, hard_pool=None, scrambled=False):
    rows = []
    for idx, it in enumerate(items):  # same candidate order as raw_reader.eval_t2
        c = [it["n"]] + (hard_pool[it["id"]] if hard_pool else it["distractors"])
        order = list(range(4))
        random.Random(f"cand-{it['id']}").shuffle(order)
        cands, truth = [c[o] for o in order], order.index(0)
        labels = lens.top([(it["layer"], m, it["start"]) for m in cands], it["size"])
        r = {"index": idx, "id": it["id"], "layer": it["layer"], "neuron": it["n"], "start": it["start"], "size": it["size"],
             "prompt": content(lens, it), "candidate_neurons": cands,
             "candidates": [R.target_text(lens.label_strings(u), lens.label_strings(v)) for u, v in labels],
             "answer": truth}
        if scrambled:
            r["prompt_scrambled"] = content(lens, it, "scrambled")
        rows.append(r)
    return rows


def build_reading(a):
    tok, lens, sp = setup(a.model, a.device, a.splits)
    fit(lens, sp, SLICE, 0)
    out = os.path.join(a.out, "reading", a.name)
    os.makedirs(out, exist_ok=True)

    def resl(v):  # the fixed-slice recipe (raw_reader --fixed_start 0): same neurons and distractors, re-sliced
        return [dict(it, start=0, size=SLICE, id=f"{it['layer']}_{it['n']}_{SLICE}_0") for it in v]
    tests = {"test_neuron": resl(sp["eval"]["test_neuron"][:1000]), "test_layer": resl(sp["eval"]["test_layer"][:1000])}
    val = resl(sp["eval"]["val_128"][:400])
    sets = {"validation": four_way(lens, val), "test_neuron": four_way(lens, tests["test_neuron"], scrambled=True),
            "test_layer": four_way(lens, tests["test_layer"])}
    hp = R.hard_distractors(lens, tests["test_neuron"], sp["pools"]["test_neuron"], SLICE)
    sets["test_neuron_hard"] = four_way(lens, tests["test_neuron"], hard_pool=hp)
    for k, rows in sets.items():
        with open(os.path.join(out, f"{k}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(k, len(rows), flush=True)
    excl = {int(l): set(v) for l, v in sp["excluded_from_train"].items()}
    F = lens.W[0].shape[1]
    with open(os.path.join(out, "train.jsonl"), "w") as fh:
        n_rows = 0
        for l in sp["train_layers"]:
            ns = [n for n in range(F) if n not in excl.get(l, ())]
            labels = lens.top([(l, n, 0) for n in ns], SLICE)
            for n, (u, v) in zip(ns, labels):
                it = {"id": f"{l}_{n}_{SLICE}_0", "layer": l, "n": n, "start": 0, "size": SLICE}
                fh.write(json.dumps({"id": it["id"], "layer": l, "neuron": n, "start": 0, "size": SLICE,
                                     "prompt": content(lens, it),
                                     "target": R.target_text(lens.label_strings(u), lens.label_strings(v))},
                                    ensure_ascii=False) + "\n")
                n_rows += 1
            print(f"train layer {l}: {len(ns)}", flush=True)
    print("train", n_rows, flush=True)


# ------------------------------------------------------------------ verification against a run's saved files
def verify(a):
    tok, lens, sp = setup(a.model, a.device, a.splits)
    rep = {}
    strs, _ = fit(lens, sp, SLICE, 0)
    saved = json.load(open(os.path.join(a.run, "artifacts_128.json")))
    rep["artifacts_match"] = strs == saved
    out = os.path.join(a.out, "reading", a.name)
    # Compared by position: the paper's eval lists repeat a few neurons with different distractors, so ids are not unique.
    mine = {k: [json.loads(l) for l in open(os.path.join(out, f"{k}.jsonl"))]
            for k in ("validation", "test_neuron", "test_neuron_hard", "test_layer")}
    for k, f in (("validation", "eval_stage0_val128_normal.json"), ("test_neuron", "eval_stage0_test_neuron_normal.json"),
                 ("test_neuron_hard", "eval_stage0_test_neuron_hard.json"), ("test_layer", "eval_stage0_test_layer_normal.json")):
        p = os.path.join(a.run, f)
        if not os.path.exists(p):
            continue
        ev = json.load(open(p))
        ok = sum(1 for m, e in zip(mine[k], ev) if m["id"] == e["id"] and m["candidate_neurons"] == e["cands"]
                 and m["answer"] == e["truth"])
        rep[f"{k}_cands_match"] = f"{ok}/{len(ev)} (mine {len(mine[k])})"
    p = os.path.join(a.run, "eval_stage0_openvocab_test_neuron_base.json")
    if os.path.exists(p):
        ev = json.load(open(p))
        ok = 0
        for m, e in zip(mine["test_neuron"], ev):
            t = R.parse_target(m["candidates"][m["answer"]])
            ok += int(m["id"] == e["id"] and t["PROMOTES"] == e["true"])
        rep["openvocab_true_labels_match"] = f"{ok}/{len(ev)}"
    # the prompt text is exactly raw_reader.make_prompt's user message (checked through the chat template)
    r = mine["test_neuron"][0]
    it = {"id": r["id"], "layer": r["layer"], "n": r["neuron"], "start": 0, "size": SLICE}
    rep["prompt_in_make_prompt"] = r["prompt"] in R.make_prompt(tok, lens, it, "h64")
    print(json.dumps(rep, indent=1), flush=True)


def verify_columns(a):
    """No weights needed: artifact filters vs the runs' saved ones, and 0-127 labels vs the reading test set."""
    import pyarrow.parquet as pq
    d = os.path.join(a.out, "columns", a.name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    rep = {}
    runs = {"slice_0": a.run, "slice_640": "runs/round4/r3_sweep_4B_s640_transfer",
            "slice_1280": "runs/round4/r3_sweep_4B_s1280_192k", "slice_2048": "runs/round4/r3_sweep_4B_s2048_transfer"}
    for k, v in meta["artifact_filters"].items():
        if k in runs and os.path.exists(os.path.join(runs[k], "artifacts_128.json")):
            rep[f"{k}_artifacts_match"] = v == json.load(open(os.path.join(runs[k], "artifacts_128.json")))
    tp = os.path.join(a.out, "reading", a.name, "test_neuron.jsonl")
    if os.path.exists(tp):
        want = {}
        for r in map(json.loads, open(tp)):
            want[(r["layer"], r["neuron"])] = R.parse_target(r["candidates"][r["answer"]])
        ok = n = 0
        for l in sorted({k[0] for k in want}):
            t = pq.read_table(os.path.join(d, f"layer_{l:02d}.parquet"),
                              columns=["neuron", "split", "slice_0_promotes", "slice_0_suppresses"]).to_pydict()
            for i, nn in enumerate(t["neuron"]):
                if (l, nn) in want:
                    w = want[(l, nn)]
                    n += 1
                    ok += int(t["slice_0_promotes"][i] == w["PROMOTES"] and t["slice_0_suppresses"][i] == w["SUPPRESSES"]
                              and t["split"][i] == "test_neuron")
        rep["test_neuron_labels_and_split_match"] = f"{ok}/{n}"
    print(json.dumps(rep, indent=1), flush=True)


# ------------------------------------------------------------------ opsd (src/opsd_data.py, unchanged, on CPU)
def build_opsd(a):
    import nl_reader as N
    import opsd_data as O
    dev = a.device

    def load(args):
        return load_shim(args.model, dev)
    R.load = load
    _Lens = R.Lens
    R.Lens = lambda tok, model, lm, device: _Lens(tok, model, lm, dev)
    os.environ.setdefault("NEURON_ATLAS", a.atlas)
    import lab
    lab.ATLAS = a.atlas
    O.Atlas.__init__.__defaults__ = (a.atlas,)
    out = os.path.join(a.out, "opsd")
    args = argparse.Namespace(out=out, merged="", init_adapter="", size=2560, start=0, n_train=6000, n_test=400)
    O.main(args)
    del N


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["columns", "reading", "verify", "vcols", "opsd"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--name", default="")
    ap.add_argument("--run", default="runs/round4/sweep_4B")
    ap.add_argument("--splits", default="")
    ap.add_argument("--out", default="release")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--transfer_slices", action="store_true", help="also label slices 640, 1280, 2048 (4B transfer runs)")
    ap.add_argument("--no_columns", action="store_true", help="labels and splits only, no weight values")
    ap.add_argument("--atlas", default="/tmp/awsp/neuron-explorer/explorer/cache")
    a = ap.parse_args()
    a.name = a.name or a.model.split("/")[-1].lower()
    {"columns": build_columns, "reading": build_reading, "verify": verify, "vcols": verify_columns,
     "opsd": build_opsd}[a.stage](a)
