"""Copy the round-4 reader adapters into one public model repo, with a card whose numbers are computed from each
run's own evaluation files.

  python src/release_models.py --src <private backup repo> --dst atharva-naik-1/awsp-readers

Each folder holds one LoRA (adapter_stage0/), the run's config.json, log.jsonl, artifact filter and eval_*.json
files. Intermediate checkpoints are not copied. The destination repo is created private; making it public is a
separate, explicit step (--public).
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np

# (source run, destination folder, description). Coordinates are of the down_proj column; "from 0-127" means the
# run started from the 0-127 reader's adapter.
RUNS = [
    ("sweep_4B", "qwen3.5-4b/coords-0-127", "Qwen3.5-4B reads its own coordinates 0-127 (the main result)"),
    ("sweep_4B_f3", "qwen3.5-4b/coords-0-127-raw-floats", "same, unscaled weights written with 3 significant digits"),
    ("sweep_4B_s128", "qwen3.5-4b/coords-128-255", "coordinates 128-255, from scratch"),
    ("sweep_4B_s1280", "qwen3.5-4b/coords-1280-1407-64k", "coordinates 1280-1407, from scratch, 64k examples (fails)"),
    ("r3_sweep_4B_s1280_192k", "qwen3.5-4b/coords-1280-1407-192k", "coordinates 1280-1407, from scratch, 192k examples"),
    ("sweep_4B_s1280_transfer", "qwen3.5-4b/coords-1280-1407-from-0-127", "coordinates 1280-1407, from the 0-127 reader"),
    ("r3_sweep_4B_s640_transfer", "qwen3.5-4b/coords-640-767-from-0-127", "coordinates 640-767, from the 0-127 reader"),
    ("r3_sweep_4B_s2048_transfer", "qwen3.5-4b/coords-2048-2175-from-0-127", "coordinates 2048-2175, from the 0-127 reader"),
    ("sweep_4B_f3_s640", "qwen3.5-4b/coords-640-767-raw-floats", "raw floats, coordinates 640-767, from scratch (not in the post)"),
    ("sweep_4B_f3_s1280", "qwen3.5-4b/coords-1280-1407-raw-floats", "raw floats, coordinates 1280-1407, from scratch (not in the post)"),
    ("scale_4B_24k", "qwen3.5-4b/budget-24k", "coordinates 0-127, 24k-example budget (emergence series)"),
    ("scale_4B_32k", "qwen3.5-4b/budget-32k", "coordinates 0-127, 32k-example budget (emergence series)"),
    ("scale_4B_48k", "qwen3.5-4b/budget-48k", "coordinates 0-127, 48k-example budget (emergence series)"),
    ("r3_scale_4B_96k", "qwen3.5-4b/budget-96k", "coordinates 0-127, 96k-example budget (emergence series)"),
    ("cross_4Breads0.8B", "qwen3.5-4b/reads-qwen3.5-0.8b", "Qwen3.5-4B reads Qwen3.5-0.8B's coordinates 0-127"),
    ("sweep_0.8B", "qwen3.5-0.8b/coords-0-127", "Qwen3.5-0.8B reads its own coordinates 0-127"),
    ("cross_0.8Breads4B", "qwen3.5-0.8b/reads-qwen3.5-4b", "Qwen3.5-0.8B reads Qwen3.5-4B's coordinates 0-127"),
    ("sweep_2B", "qwen3.5-2b/coords-0-127", "Qwen3.5-2B reads its own coordinates 0-127"),
    ("sweep_9B", "qwen3.5-9b/coords-0-127", "Qwen3.5-9B reads its own coordinates 0-127 (lr 1e-4, 64k)"),
    ("sweep_9B_lr5e-5_128k", "qwen3.5-9b/coords-0-127-lr5e-5-128k", "same, lr 5e-5 and 128k examples"),
    ("sweep_Llama-3.2-3B", "Llama-3.2-3B-Instruct-reader/coords-0-127", "Llama-3.2-3B-Instruct reads its own coordinates 0-127"),
]
SKIP = ("ckpts/", "ckpt_latest/")


def acc(path):
    if not os.path.exists(path):
        return None
    e = json.load(open(path))
    return 100 * float(np.mean([x["correct_pmi"] for x in e])), len(e)


def metrics(d):
    m = {}
    for key, f in (("test", "eval_stage0_test_neuron_normal.json"), ("shuffled", "eval_stage0_test_neuron_scrambled.json"),
                   ("hard", "eval_stage0_test_neuron_hard.json"), ("held_layers", "eval_stage0_test_layer_normal.json"),
                   ("val", "eval_stage0_val128_normal.json")):
        m[key] = acc(os.path.join(d, f))
    p = os.path.join(d, "eval_stage0_openvocab_test_neuron_trained.json")
    m["recall8"] = 100 * float(np.mean([x["recall8_raw"] for x in json.load(open(p))])) if os.path.exists(p) else None
    return m


def fmt(x):
    return "" if x is None else f"{x[0]:.1f}"


CARD = """---
license: apache-2.0
base_model:
- Qwen/Qwen3.5-4B
- Qwen/Qwen3.5-0.8B
- Qwen/Qwen3.5-2B
- Qwen/Qwen3.5-9B
- meta-llama/Llama-3.2-3B-Instruct
library_name: peft
tags:
- interpretability
- weight-space-programming
- logit-lens
- lora
---

# Auto Weight-Space Programming: weight readers (Part 1)

LoRA adapters that teach a model to read numbers copied from its own MLP weights. The prompt holds 128 coordinates of one
neuron's output direction (a column of `mlp.down_proj`), written as integers scaled so that 20 is one standard
deviation of the full column. The model answers with the tokens that slice promotes and suppresses through the logit
lens (its final norm and unembedding). These are the readers from Part 1 of the Auto Weight-Space Programming blog
series.

- Code: https://github.com/atharva-naik/auto-weight-space-programming (`src/raw_reader.py`)
- Data: [awsp-neuron-columns](https://huggingface.co/datasets/atharva-naik-1/awsp-neuron-columns),
  [awsp-neuron-reading](https://huggingface.co/datasets/atharva-naik-1/awsp-neuron-reading)

Every adapter is rank 32 (alpha 64) on all attention and MLP projections, trained with learning rate 1e-4 and 16
examples per step unless the folder name says otherwise. Each folder has the adapter (`adapter_stage0/`), the run's
`config.json`, its training `log.jsonl`, the artifact-token filter (`artifacts_128.json`) and the per-item evaluation
files (`eval_*.json`) the numbers below are computed from.

## Results

Four-way accuracy (%) on the locked test split: the true token list against three other neurons' lists, chance 25%.
Each candidate is scored by its log-likelihood with the numbers shown minus its log-likelihood with the numbers hidden.
"Shuffled" permutes the same numbers; "hard" draws the three wrong lists from the most similar slices in the same layer;
"held-out layers" are four layers never trained on. Recall@8 is the share of the true top-8 promoted tokens in the
model's own top 8 at the first answer slot, with no candidates. Budget runs have validation accuracy only (400 items).

| Folder | What it is | Test | Shuffled | Hard | Held-out layers | Recall@8 | Val |
|---|---|---|---|---|---|---|---|
{rows}

Test items: 1,000 (977 distinct neurons). Validation: 400. Training never saw a test neuron or a held-out layer.

## Use

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download
d = snapshot_download("atharva-naik-1/awsp-readers", allow_patterns=["qwen3.5-4b/coords-0-127/adapter_stage0/*"])
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-4B", dtype="bfloat16")
model = PeftModel.from_pretrained(model, f"{d}/qwen3.5-4b/coords-0-127/adapter_stage0")
```
The prompt format and scoring are in `src/raw_reader.py` (`user_prompt`, `make_prompt`, `eval_t2`); ready-made prompts
are in the awsp-neuron-reading dataset. Use the chat template with thinking off.

## Licenses

The Qwen3.5 adapters are released under Apache-2.0, like their base models. The adapter in `Llama-3.2-3B-Instruct-reader/`
is a derivative of Llama 3.2 and is distributed under the Llama 3.2 Community License (`LICENSE_LLAMA_3.2.txt`): Built
with Llama. Atlas labels used elsewhere in the project come from neuron-explorer (lyramakesmusic, MIT).
"""


def main(a):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    work = a.work or tempfile.mkdtemp(prefix="awsp_readers_")
    pats = [f"runs/round4/{src}/*" for src, _, _ in RUNS]
    local = snapshot_download(a.src, allow_patterns=pats, local_dir=os.path.join(work, "src"), max_workers=8)
    out = os.path.join(work, "dst")
    rows = []
    for src, dst, desc in RUNS:
        sd = os.path.join(local, "runs", "round4", src)
        assert os.path.isdir(os.path.join(sd, "adapter_stage0")), src
        dd = os.path.join(out, dst)
        shutil.rmtree(dd, ignore_errors=True)
        for root, _, files in os.walk(sd):
            rel = os.path.relpath(root, sd)
            if any((rel + "/").startswith(s) for s in SKIP):
                continue
            os.makedirs(os.path.join(dd, rel), exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(root, f), os.path.join(dd, rel, f))
        m = metrics(sd)
        r8 = "" if m["recall8"] is None else f"{m['recall8']:.1f}"
        rows.append(f"| `{dst}` | {desc} | {fmt(m['test'])} | {fmt(m['shuffled'])} | {fmt(m['hard'])} | "
                    f"{fmt(m['held_layers'])} | {r8} | {fmt(m['val'])} |")
        print(dst, {k: (None if v is None else (round(v[0], 1) if isinstance(v, tuple) else round(v, 1))) for k, v in m.items()},
              flush=True)
    open(os.path.join(out, "README.md"), "w").write(CARD.replace("{rows}", "\n".join(rows)))
    from huggingface_hub import hf_hub_download
    lic = a.llama_license or hf_hub_download("meta-llama/Llama-3.2-3B-Instruct", "LICENSE.txt")
    shutil.copy(lic, os.path.join(out, "LICENSE_LLAMA_3.2.txt"))
    if a.dry:
        print("dry run; files in", out)
        return
    api.create_repo(a.dst, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=a.dst, folder_path=out, repo_type="model", commit_message="Part 1 readers")
    print("uploaded to", a.dst, "(private)", flush=True)
    if a.public:
        api.update_repo_settings(a.dst, private=False)
        print("made public", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="<private backup repo>")
    ap.add_argument("--dst", default="atharva-naik-1/awsp-readers")
    ap.add_argument("--work", default="")
    ap.add_argument("--llama_license", default="")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--public", action="store_true")
    main(ap.parse_args())
