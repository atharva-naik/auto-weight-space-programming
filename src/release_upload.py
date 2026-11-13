"""Package the built datasets (src/release_datasets.py output) into three Hugging Face dataset repos, with cards whose
counts are computed from the files.

  python src/release_upload.py --release release --owner atharva-naik-1            # create private repos and upload
  python src/release_upload.py --release release --owner atharva-naik-1 --public   # then flip them public

Repos: awsp-neuron-columns, awsp-neuron-reading, awsp-neuron-descriptions. Only models whose build folder is complete
are included. Four held-out levers (refusal bypass, sexual desire, an obscene/gentle neuron and risky-request caution)
are removed from the public descriptions data; they were never in its train or test rows.
"""
import argparse
import glob
import json
import os
import shutil
import tempfile

MODELS = [("qwen3.5-4b", "Qwen/Qwen3.5-4B"), ("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B"), ("qwen3.5-2b", "Qwen/Qwen3.5-2B"),
          ("qwen3.5-9b", "Qwen/Qwen3.5-9B"), ("llama-3.2-3b-instruct", "meta-llama/Llama-3.2-3B-Instruct")]
DROP_LEVERS = {(11, 6261), (18, 8737), (18, 8827), (10, 6032)}
GH = "https://github.com/atharva-naik/auto-weight-space-programming"


def lines(p):
    with open(p) as fh:
        return sum(1 for _ in fh)


def columns_card(present, stats):
    cfgs = "\n".join(f"- config_name: {n}\n  data_files: \"{n}/*.parquet\"" for n in present)
    rows = "\n".join(f"| `{n}` | {stats[n]['model']} | {stats[n]['layers']} | {stats[n]['d_ff']:,} | {stats[n]['d_model']:,} | "
                     f"{stats[n]['rows']:,} | {'yes' if stats[n]['include_columns'] else 'no (labels and splits only)'} |"
                     for n in present)
    return f"""---
license: other
license_name: mixed-see-card
configs:
{cfgs}
tags:
- interpretability
- weight-space-programming
- logit-lens
---

# Auto Weight-Space Programming: neuron columns

Every MLP neuron of each model, one row per neuron: its output direction (the neuron's column of `mlp.down_proj`) as
integers, exact logit-lens labels, and the train/validation/test split used in Part 1 of the Auto Weight-Space
Programming blog series. Built from the released weights by `src/release_datasets.py columns` in {GH}, using the
same label code as the training runs; the labels were checked against the runs' saved evaluation files.

| Config | Model | Layers | Neurons per layer | Column length | Rows | Weight values |
|---|---|---|---|---|---|---|
{rows}

## Fields

- `layer`, `neuron`: the neuron's position (`model.layers[layer].mlp.down_proj.weight[:, neuron]`).
- `split`: `train`, `val`, `test_neuron` (held-out neurons in training layers) or `test_layer` (four layers never trained on).
- `column`: the full column as int16, `round(w / std(w) * 20)`, so 20 is one standard deviation of the column. This is
  the format the readers see. `column_std` is the float std, so `column / 20 * column_std` approximately recovers the weights.
- `full_promotes`, `full_suppresses`: the 8 tokens the whole column promotes and suppresses most through the logit lens
  (final RMSNorm scale times the unembedding; Qwen3.5 scales by 1 + w, Llama by w).
- `slice_<s>_promotes`, `slice_<s>_suppresses`: the same for coordinates s to s+127 only (s = 0, plus 640, 1280 and 2048
  for Qwen3.5-4B). The 0-127 lists are the reading targets in the paper.

Labels drop special tokens, tokens that decode to whitespace or invalid UTF-8, and per-slice "artifact" tokens that
appear in the top-12 lists of more than 5% of 4,000 training neurons; `meta.json` in each folder lists them.

Labels are computed in float32. Near-ties can swap at the last rank across hardware: for Qwen3.5-4B, the columns
(built on GPU) and the reading set (built on CPU) agree on all 16 tokens for 976 of 977 test neurons; the exception
(layer 10, neuron 1275) differs in its 8th suppressed token.

## Licenses

Columns are parts of the original weights and keep their licenses: Qwen3.5 is Apache-2.0. For Llama-3.2-3B-Instruct
only labels and splits are included, no weight values; rebuild the columns from Meta's release under its license.
"""


def reading_card(present, stats):
    cfgs = "\n".join(
        f"- config_name: {n}\n  data_files:\n" + "\n".join(f"  - split: {s}\n    path: \"{n}/{s}.jsonl\"" for s in
                                                         ("train", "validation", "test_neuron", "test_neuron_hard", "test_layer"))
        for n in present)
    rows = "\n".join(f"| `{n}` | {stats[n]['model']} | " + " | ".join(f"{stats[n][s]:,}" for s in
                     ("train", "validation", "test_neuron", "test_neuron_hard", "test_layer")) + " |" for n in present)
    return f"""---
license: other
license_name: mixed-see-card
configs:
{cfgs}
tags:
- interpretability
- weight-space-programming
- logit-lens
---

# Auto Weight-Space Programming: the neuron reading task

The reading task from Part 1 of the Auto Weight-Space Programming blog series, as text. The prompt shows coordinates 0-127 of
one neuron's output direction (its `mlp.down_proj` column) as integers scaled so that 20 is one standard deviation of the
full column, and asks which tokens that slice promotes and suppresses through the logit lens. Built by
`src/release_datasets.py reading` in {GH} with the training code's own prompt, label and candidate functions. The
4-way sets reproduce the paper's evaluation exactly: same items, same candidates in the same order, same answers
(checked against the runs' saved evaluation files).

| Config | Model | train | validation | test_neuron | test_neuron_hard | test_layer |
|---|---|---|---|---|---|---|
{rows}

## Splits

- `train`: one item per training neuron (`prompt`, `target`). The paper's readers sample from this pool.
- `validation`, `test_neuron`, `test_layer`: 4-way items. `candidates` holds four target texts (the true neuron's and three
  other neurons' from the same layer and split); `answer` is the index of the true one; `candidate_neurons` are their
  neuron indices. `test_layer` uses four layers never trained on.
- `test_neuron_hard`: the same neurons with the three most similar slices in the layer as distractors.
- `test_neuron` also has `prompt_scrambled`, the same numbers shuffled (the control in the paper).

`index` is the row's position. `id` is the paper's item id; a few neurons appear twice with different distractors,
as in the paper's test, so `id` is not unique. The paper scores each candidate by its log-likelihood after the
prompt minus its log-likelihood with the number lines replaced by `(numbers hidden)`. Use the model's chat
template with thinking off.

## Licenses

Prompts quote parts of the original weights: Qwen3.5 is Apache-2.0; the Llama-3.2-3B-Instruct config quotes
Llama 3.2 weights and is distributed under the Llama 3.2 Community License (Built with Llama).
"""


def descriptions_card(stats):
    return f"""---
license: mit
configs:
- config_name: default
  data_files:
  - split: train
    path: "train.jsonl"
  - split: test
    path: "test.jsonl"
- config_name: heldout_levers
  data_files:
  - split: test
    path: "heldout_levers.jsonl"
tags:
- interpretability
- weight-space-programming
- on-policy-distillation
---

# Auto Weight-Space Programming: neuron descriptions (OPSD data)

Data for training Qwen3.5-4B to describe its own neurons in words with on-policy self-distillation (OPSD), from Part 1
of the Auto Weight-Space Programming blog series. The student sees one neuron's full output column (2,560 integers) and an open
question ("What does this neuron do?"). The teacher, the same model frozen, also gets `demonstration`: the neuron's
entry in the neuron-explorer atlas (logit lens and steer lens) and an instruction to answer as if read off the weights.
Built by `src/release_datasets.py opsd`, which runs `src/opsd_data.py` from {GH} unchanged.

| Split | Rows |
|---|---|
| train | {stats['train']:,} |
| test | {stats['test']:,} |
| heldout_levers | {stats['levers']:,} |

- `prompt`: the student's user turn. `demonstration`: the teacher's privileged system message. `facts`: the atlas
  facts behind it. `kind`: `describe` or `complete` ("Pushed up, this neuron makes me ...").
- `test` rows are held-out neurons and carry 3 same-layer distractors (nearest output columns) and their facts, for the
  4-way judge test in `src/judge_eval.py`.
- `heldout_levers`: neurons kept out of all training, for a later neuron-search test: atlas reference levers and
  neurons of four concept families (pirate, flattery, apology, conspiracy). Four levers are not published here
  (refusal bypass, sexual content, an obscene-language neuron, risky-request caution); they were never in train or test.

There is no trained describer yet; see the blog post. Atlas facts come from neuron-explorer (lyramakesmusic, MIT).
Prompts quote Qwen3.5-4B weights (Apache-2.0).
"""


def main(a):
    from huggingface_hub import HfApi
    api = HfApi()
    work = tempfile.mkdtemp(prefix="awsp_upload_")
    out = {}
    # columns
    d = os.path.join(work, "columns"); os.makedirs(d)
    present, stats = [], {}
    for n, mid in MODELS:
        src = os.path.join(a.release, "columns", n)
        if not os.path.exists(os.path.join(src, "meta.json")):
            continue
        meta = json.load(open(os.path.join(src, "meta.json")))
        files = sorted(glob.glob(os.path.join(src, "layer_*.parquet")))
        if len(files) != meta["layers"]:
            print("incomplete columns, skipping", n, len(files), meta["layers"]); continue
        import pyarrow.parquet as pq
        stats[n] = {**meta, "rows": sum(pq.ParquetFile(f).metadata.num_rows for f in files)}
        shutil.copytree(src, os.path.join(d, n)); present.append(n)
    open(os.path.join(d, "README.md"), "w").write(columns_card(present, stats))
    out["awsp-neuron-columns"] = (d, present)
    # reading
    d = os.path.join(work, "reading"); os.makedirs(d)
    present, stats = [], {}
    for n, mid in MODELS:
        src = os.path.join(a.release, "reading", n)
        need = ["train", "validation", "test_neuron", "test_neuron_hard", "test_layer"]
        if not all(os.path.exists(os.path.join(src, f"{s}.jsonl")) for s in need):
            continue
        os.makedirs(os.path.join(d, n))
        stats[n] = {"model": mid}
        for s in need:
            shutil.copy(os.path.join(src, f"{s}.jsonl"), os.path.join(d, n, f"{s}.jsonl"))
            stats[n][s] = lines(os.path.join(src, f"{s}.jsonl"))
        if stats[n]["train"] == 0:
            print("empty train, skipping", n); shutil.rmtree(os.path.join(d, n)); del stats[n]; continue
        present.append(n)
    open(os.path.join(d, "README.md"), "w").write(reading_card(present, stats))
    out["awsp-neuron-reading"] = (d, present)
    # descriptions
    src = os.path.join(a.release, "opsd")
    if os.path.exists(os.path.join(src, "train.jsonl")):
        d = os.path.join(work, "descriptions"); os.makedirs(d)
        for s in ("train", "test"):
            shutil.copy(os.path.join(src, f"{s}.jsonl"), os.path.join(d, f"{s}.jsonl"))
        lev = [json.loads(l) for l in open(os.path.join(src, "heldout_levers.jsonl"))]
        lev = [r for r in lev if (r["layer"], r["n"]) not in DROP_LEVERS]
        with open(os.path.join(d, "heldout_levers.jsonl"), "w") as fh:
            for r in lev:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        info = json.load(open(os.path.join(src, "info.json")))
        info["held_levers"] = {k: v for k, v in info.get("held_levers", {}).items()
                               if tuple(int(x[1:]) for x in k.split()) not in DROP_LEVERS}
        json.dump(info, open(os.path.join(d, "info.json"), "w"), ensure_ascii=False, indent=1)
        st = {"train": lines(os.path.join(d, "train.jsonl")), "test": lines(os.path.join(d, "test.jsonl")), "levers": len(lev)}
        open(os.path.join(d, "README.md"), "w").write(descriptions_card(st))
        out["awsp-neuron-descriptions"] = (d, ["default"])
    for repo, (d, present) in out.items():
        rid = f"{a.owner}/{repo}"
        print(rid, present, flush=True)
        if not present:
            print("nothing built, skipping", rid, flush=True)
            continue
        if a.dry:
            continue
        api.create_repo(rid, repo_type="dataset", private=True, exist_ok=True)
        api.upload_large_folder(repo_id=rid, folder_path=d, repo_type="dataset") if a.large else \
            api.upload_folder(repo_id=rid, folder_path=d, repo_type="dataset", commit_message="Part 1 release")
        if a.public:
            api.update_repo_settings(rid, private=False, repo_type="dataset")
            print("public:", rid, flush=True)
    print("staged in", work)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="release")
    ap.add_argument("--owner", default="atharva-naik-1")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--large", action="store_true", help="use upload_large_folder (resumable) for big folders")
    ap.add_argument("--public", action="store_true")
    main(ap.parse_args())
