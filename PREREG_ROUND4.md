# Round 4 pre-registration: kill test for "Qwen3.5-4B learns to read its own raw weights as text"

Written 2026-11-04, **before any round-4 data is built or any training is run**. This replaces the OPSD/soft-token
plan in `PLAN_ROUND4_OPSD_ATLAS.md`. That plan is superseded because its core result (an LM describing its own
units from vectors injected as soft tokens) is already published in Li et al., arXiv 2511.08579. Here the model
sees its weights only as **plain-text integers**, which neither Li et al. nor Weight-Space Programming does for an
LLM's own neurons. (WSP reads raw text only for small networks, and uses a Python sandbox for LLMs.)

## Question

After LoRA fine-tuning, can Qwen3.5-4B read one of its own MLP neurons' output vectors (column n of
`mlp.down_proj`, 2,560 numbers) written as text, and say which tokens that vector promotes and suppresses through
the logit lens? It must do this for neurons it was never trained on.

## Setup (fixed)

- **Model:** `Qwen/Qwen3.5-4B`, thinking off. The embeddings are tied (`tie_word_embeddings: true`). The final
  norm is (1+w).
- **Object:** the base weights. LoRA does not change the weights that are read or the labels.
- **Input text:** the column, scaled per neuron so that one standard deviation = 20, rounded to integers, no
  clipping. Coordinates are written in order, in lines that each start with the index of their first coordinate,
  e.g. `[1728] 12 -3 41 ...`. The prompt does **not** state the layer or the neuron index.
  - Format `h64`: 64 values per line (about 7.0k tokens for a full column).
  - Format `h8`: 8 values per line (about 8.9k tokens).
  - The Qwen tokenizer splits digits one by one; I measured these costs on 2026-11-04.
- **Label (exact, computed from the weights):** the slice logit lens, `scores = U[:, S] @ ((1+w)[S] * v[S])`, over
  the shown coordinates S. Here U is the unembedding, v the column and w the final-norm weight. The labels are the
  top-8 promoted and top-8 suppressed tokens after excluding:
  - padding rows (id ≥ tokenizer length) and special tokens
  - tokens whose decoded string is empty, whitespace-only or contains U+FFFD
  - artifact tokens: those in more than 5% of the top-12 lists of 4,000 sampled *training* items for that stage
    and pole

  For the full column this matches the explorer's `lens_index` up to the filters.
- **Target string:** `PROMOTES: [json list]\nSUPPRESSES: [json list]`.
- **Training:** supervised fine-tuning, LoRA r = 32, alpha = 64, dropout 0. Targets: q/k/v/o, `in_proj_qkv`,
  `in_proj_z`, `out_proj`, gate/up/down of all 32 text layers. AdamW, lr 1e-4, 30 warm-up steps, cosine decay to
  10%, 16 items per optimizer step, bf16, gradient checkpointing. The curriculum runs one pass per stage, with a new
  random training neuron for each item:
  1. 128 coordinates (an aligned random slice): 16,000 items
  2. 512 coordinates: 8,000 items
  3. all 2,560 coordinates: 8,000 items
- **Allowed extension, decided on validation only:** if stage-3 val accuracy is still rising at the end, stage 3 may
  be extended once by 8,000 items. This must be logged.

## Splits (`src/raw_splits.py`, seed 0, written before training)

- **Held-out layers:** 5, 13, 21, 29. No neuron from these layers is ever trained on.
- **In each of the other 28 layers:** 10% of neurons are the held-out neuron set and 2% are validation; the rest are
  training.
- **Evaluation items:**
  - val: 400 neurons, at each stage's slice size
  - `test_neuron`: 1,000 held-out neurons from trained layers, full column
  - `test_layer`: 1,000 neurons from the held-out layers, full column
- **Distractors:** for each eval item, 3 neurons from the same layer and the same split, fixed by seed, at the same
  slice. Hard distractors (secondary only) are the 3 neurons in that pool whose shown slice has the highest cosine
  similarity to the item's.

## Metric

**T2, a 4-way choice (chance 25%).** Each candidate is a target string. Its score is

  PMI = log p(candidate | prompt with numbers) − log p(candidate | same prompt with the numbers replaced by
  "(numbers hidden)")

with the sums taken over candidate tokens. PMI removes the model's preference for some token lists regardless of
the numbers. Raw log p accuracy is also reported.

**Secondary:**
- T1 recall@8: greedy generation on 200 `test_neuron` items, scored against the top-12 per pole
- hard-distractor T2
- `test_layer` T2
- val T2 at each curriculum stage (where reading breaks down)

**Arms per format (`h64`, `h8`):**
- zero-shot: step 0. The LoRA B matrices start at zero, so this is the exact base model.
- trained: end of stage 3

## Controls

- **Scrambled:** the same numbers in a random order within the shown slice. The line headers and labels are
  unchanged, so this is the same information without its positions. A model that really reads coordinates should
  drop to about chance.
- **Hidden:** raw-log-p accuracy with the numbers hidden. This is the floor from label priors alone.

## Pass criteria (primary)

The primary format is the one with the higher final val PMI accuracy. Both formats are reported. 95% CIs are
paired bootstraps over items (10,000 resamples). The kill test **passes** only if all three hold on `test_neuron`,
full column, random distractors:

1. trained PMI accuracy ≥ 35%, with the CI lower bound > 25%
2. trained − zero-shot ≥ 10 points, with the CI excluding 0
3. trained (normal) − trained (scrambled) ≥ 10 points, with the CI excluding 0

**If it passes:** write a new round-4 plan around raw-weight reading, with harder labels, held-out layers and
possibly OPSD with a soft-token teacher.

**If it fails:** drop the direction. Report every number above, including the val accuracy at each stage, in
`RESULTS_ROUND4.md`.

## Not claimed, whatever happens

- The logit lens is a fixed linear readout. A regression on the same numbers would be exact. The claim is only
  that the LM *learned to read raw weights*, not that it is the best predictor.
- These labels are not behaviours. The test set is touched once, after training finishes.

## Amendment (2026-11-04, after the 8-item GPU smoke run, before any full-run result)

- **Smoke run (no outcome information):** n = 8 items per eval. The pipeline runs end to end on the real 4B. Labels
  overlap 97% with the explorer's `lens_index` top-12. Timing:
  - full-column eval: about 1.4 s per item
  - stage 3: about 35 s per 16-item step
  - full run: about 8 h per format on one H100
- **Seed-1 replicates** of both formats run on the spare GPUs as a robustness check only. The primary criteria above
  use seed 0. Seed-1 runs skip the zero-shot eval, which is identical to seed 0's (same base model and items).

## Deviation (2026-11-04, about 1 h into the full runs)

- **What I saw.** After all of stage 1 (16,000 items, 128 coordinates), the seed-1 runs were at chance on val128:
  - `h64_s1`: PMI 23% on n = 400; mid-stage checks on n = 100 ranged from 19% to 30%
  - `h8_s1`: the same pattern
  - The training loss fell from 3.28 to about 3.0 in the first 200 steps and then went flat, consistent with
    learning only the output format and which tokens are common labels.
- **What I changed.**
  - I stopped the seed-1 replicates, since replicating a likely null has little value.
  - The seed-0 primary runs (`h64`, `h8`) continue unchanged, and the verdict above still comes from them alone.
  - The freed GPUs run two **exploratory, not pre-registered** diagnostics, reported separately:
    - `diag_fixed128`: always coordinates 0–127. Does the coordinate-index lookup cause the failure?
    - `diag_rand16`: random 16-coordinate slices. Is a much smaller reading problem learnable?
