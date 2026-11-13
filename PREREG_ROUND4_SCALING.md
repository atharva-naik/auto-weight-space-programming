# Round 4b pre-registration: does raw-weight reading emerge across Qwen3.5 sizes?

Written 2026-11-04, 15:50 EST. At this point no model other than Qwen3.5-4B has been trained, and no test split has
been evaluated with this recipe.

## Background (exploratory, see `ROUND4_LOG.md`)

The "G1" recipe trained Qwen3.5-4B to read coordinates 0–127 of its own `down_proj` columns as text. On 400 val-pool
neurons never used in any earlier check, the 4-way logit-lens choice reached 99.75% PMI (84% raw):
- shuffling the numbers dropped it to chance (23.5%)
- with no candidates, recall@8 was only 10%, and the median rank of a true token was 562 of 248k

The recipe was found on 4B, which biases the comparison in 4B's favour.

## Fixed recipe (identical for every model)

- **Input:** coordinates 0–127 of `mlp.down_proj` column n, scaled per neuron so that one std of the full column
  = 20, integers, format `h64` (64 values per line with an index header). The prompt and target strings are those of
  `src/raw_reader.py`.
- **Label:** the exact slice logit lens through that model's own final norm (1+w) and `lm_head`, top-8 per pole,
  with the same token filters. Artifacts are fitted on 4,000 training items per model.
- **Training:**
  - LoRA r = 32, alpha = 64, dropout 0, on q/k/v/o, `in_proj_qkv`, `in_proj_z`, `out_proj` and gate/up/down of
    all text layers
  - AdamW, lr 1e-4, 30 warm-up steps, cosine decay to 10%
  - 16 items per step, a fresh training neuron for each item
  - **budget 64,000 items (4,000 steps)**
- **Monitoring:** val checks every 250 steps on 100 val neurons, using val only.
- **Models:** `Qwen/Qwen3.5-{0.8B, 2B, 4B, 9B, 27B}` (instruct).
  - The 4B point is the existing G1 adapter: same recipe, already trained, test split untouched.
  - Hidden sizes are 1024, 2048, 2560, 4096 and 5120.
  - Embeddings are tied for 0.8B/2B/4B and untied for 9B/27B.

## Splits (`src/raw_splits.py --model <id>`, seed 0, the same rule for every model)

- **Held-out layers:** `round(5L/32) + b·L/4` for b = 0..3. This gives 5, 13, 21, 29 for L = 32, 4, 10, 16, 22 for
  L = 24, and 10, 26, 42, 58 for L = 64. All of these are linear-attention layers.
- **In the other layers:** 10% of neurons are held out (test_neuron) and 2% are validation.
- **Test sets per model:** 1,000 test_neuron items and 1,000 test_layer items, all at slice 0–127. For 4B,
  `data/raw_splits.json` is unchanged; its test items are re-sliced to 0–127.

## Outcomes per model (test splits, evaluated once, after training)

- **O1:** 4-way PMI accuracy on test_neuron with random same-layer distractors, with a 95% bootstrap CI (10,000
  resamples).
- **O2:** O1 with the numbers shuffled (control).
- **O3:** O1 with hard distractors (the 3 most similar same-layer test_neuron neurons by cosine of the shown slice).
- **O4:** 4-way PMI accuracy on test_layer.
- **O5:** no-candidate readout on test_neuron: recall@8 and median rank of the true top-8 promoted tokens from one
  forward pass at the first list slot (raw scores). The same is reported for the base model (no adapter) as a floor.
- **O6:** onset, the number of training items at the first val check with PMI ≥ 50%.
- **Also reported, not used for claims:** raw accuracy, the hidden-prior accuracy, and T1 greedy generation.

## Claims

- **S1, reading at each size.** For a given model: O1's CI lower bound is above 25%, and O1 − O2 ≥ 10 points with
  a paired CI excluding 0. Reported per size.
- **S2, trend with size.** Describe how O1, O3, O5 and O6 change across sizes. With 5 sizes there is no
  significance test. A trend is claimed only if a metric is monotone across all trained sizes. Confounds to state
  every time:
  - tied (≤ 4B) vs untied (9B, 27B) embeddings
  - the 128-coordinate slice is a smaller fraction of the hidden size in larger models
  - lr and the budget are not tuned per size
  - the recipe was found on 4B

## Budget rules

- A model with no onset by 64k items is reported as "no onset within budget".
- One extension of 64k items is allowed, only for a model whose val accuracy is still rising at the end, judged on
  val only, and logged.
- If the sandbox lifetime ends before a size finishes, that size is reported as missing. It is not replaced by a
  smaller run.
- **Order:** 0.8B and 2B first, then 9B, then 27B (2 GPUs). The 27B run starts only after the round-4 pre-registered
  runs (`h64`, `h8`) finish.

## Amendment (2026-11-04, 16:00 EST, scheduling only, before any sweep result)

If 27B waited for the round-4 runs, it could not finish before the sandbox expires at about 12:39 EST on
2026-11-05. The new schedule:
- 27B runs on GPUs 2–3 once 0.8B, 2B and the dictionary diagnostic finish.
- 9B runs on GPU 0 after `h64`.
- The 4B test eval runs on GPU 1 alongside `h8`.

No pre-registered run is stopped, and nothing about the recipe or the outcomes changes.

## Amendment 2 (2026-11-04, 16:15 EST, infrastructure only, before any sweep result)

- **Sandbox failure.** Sandbox the sandbox failed at 15:52 EST, probably from running out of memory, and all
  on-sandbox state was lost, including the G1 adapter. The sweep reruns on the sandbox (512 GiB) with
  a helper script.
- **The 4B point** is now a fresh run of the identical recipe (`sweep_4B`), not the lost G1 adapter. G1's
  exploratory numbers in `ROUND4_LOG.md` serve as a replication reference.
- **Schedule:**
  - GPU 0: 4B, then 9B
  - GPU 1: 0.8B, then 2B
  - GPUs 2–3: 27B

  The round-4 pre-registered runs (`h64`, `h8`) are not rerun for now; that is pending the user's decision.
- **Recipe and outcomes:** unchanged.

## Amendment 3 (2026-11-04, 16:40 EST, before any sweep result)

- **Checkpointing.** Runs started from now on (2B, 9B and those below) save `ckpt_latest/` every 500 steps
  (adapter + optimizer + schedule) and can `--resume` exactly. A CPU test confirmed that resumed training matches
  uninterrupted training (max |log p| difference 0.0). This has no effect on training.
- **S3, generality across slices (pre-specified secondary).**
  - Run: Qwen3.5-4B with the identical recipe on coordinates **1280–1407** (`--fixed_start 1280`), as
    `sweep_4B_s1280`, on GPU 1 after 2B.
  - Outcomes: O1–O6 on the same 4B test neurons and distractors, re-sliced to 1280–1407.
  - Claim: S1's criteria (O1 CI lower bound > 25%; O1 − O2 ≥ 10 points) hold on this slice as well.

## Amendment 4 (2026-11-04, 17:15 EST): cross-model reading

Written before either cross run started. At the time, the only completed sweep results were 0.8B's first test evals.

- **Design: a 2×2 of reader {0.8B, 4B} × weights read {0.8B's, 4B's}.**
  - The self cells are `sweep_0.8B` and `sweep_4B`.
  - New: `cross_4Breads0.8B` (reader 4B, `--target_model Qwen/Qwen3.5-0.8B`) and `cross_0.8Breads4B`.
  - Identical recipe and identical prompt. The prompt still says "your own", so only the weights' source differs.
  - Labels come from the *target* model's weights, final norm and unembedding. The tokenizers are identical.
  - Splits and test neurons are the target's: `raw_splits_0.8B.json` or `raw_splits.json`.
- **Outcomes:** O1–O6 per cell.
- **Comparisons:** reported, with no significance claim from 2 sizes.
  - Within each target, self-reader vs other-reader.
  - Within each reader, own weights vs the other model's.
- **Claim rule:** we say "self-reading is easier" only if the *smaller* self-reader beats the larger other-reader on
  the same target. Concretely, `sweep_0.8B` O1 > `cross_4Breads0.8B` O1 with a paired CI excluding 0, since both
  use the same 0.8B test items. Otherwise we report the cells without that claim.

## Amendment 5 (2026-11-04, 17:25 EST): 27B deferred in favour of the write experiments

- **What and why.** At the user's request, the 27B run was stopped at step 540 of 4,000 to free GPUs 2–3 for the
  write experiments (`ROUND4_LOG.md`). This was a priority decision, not an outcome-driven one: 27B's only result
  was a val check at step 500, at 24%, which carries no information.
- **Reporting.** 27B is reported as **deferred/missing**, not as a null. It can be re-run later with checkpointing.
- **The second slice** (`sweep_4B_s1280`) moves from GPU 1 to GPU 3, with the same recipe.

## Amendment 6 (2026-11-04, 18:20 EST): a non-Qwen model, before its run

- **Run:** `meta-llama/Llama-3.2-3B-Instruct` with the identical recipe, reading its own weights, as
  `sweep_Llama-3.2-3B`.
- **Model:** 28 layers, hidden size 3,072, tied embeddings, and a different tokenizer that chunks digits instead of
  splitting them.
- **Splits:** by the same rule, so the held-out layers are 4, 11, 18 and 25 (`raw_splits_Llama-3.2-3B-Instruct.json`).
- **Code changes, which leave the Qwen runs unchanged** (identical split hashes and an identical tiny-model test):
  - a generic causal-LM loader
  - the final-norm scale detected from the weights: (1+w) when the mean weight is below 0.5, else w. Llama uses w.
  - the target ends with the tokenizer's end-of-turn token
- **Outcomes:** O1–O6. S1's criteria decide whether reading holds outside Qwen.
- **Not used:** Gemma 3, because its post-feed-forward norm rescales the MLP output, so a `down_proj` column is not
  exactly what the neuron writes.
