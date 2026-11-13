# Round 4c pre-registration: two harder reading tasks and a raw-float ablation (Qwen3.5-4B)

Written 2026-11-05, 01:15 EST, before any of these runs. The model reads raw weights as text, as in round 4: integers
scaled per vector (20 = one std) in lines with index headers, coordinates 0–127, the same splits
(`data/raw_splits.json`), and the same LoRA recipe (r = 32, lr 1e-4, 16 items per step, 64,000 items, checkpoint every
500 steps). It starts from base Qwen3.5-4B, not from the round-4 reader. Code: `src/task_reader.py`.

## Task A: predict your own activation (`act`)

- **Input.** Coordinates 0–127 of neuron (L, n)'s **input** weights (a row of `mlp.gate_proj`), the layer index L,
  and a sentence.
- **Label.** The sign of the shown slice's contribution to the gate pre-activation at the sentence's last token:
  `sign(Σ_{i<128} x_L[i] · W_gate[L][n, i])`. Here x_L is the base model's `post_attention_layernorm` output at
  layer L, at the last token of the sentence run on its own. The answer is `UP` or `DOWN`; chance is 50%.
  - The label needs a forward pass of the model on the sentence; no matmul on the weights alone gives it.
  - The sign does not depend on how the weights are scaled.
- **Sentences.** From wikitext-2 (raw), as disjoint train, val and test sentence pools. Test items use held-out
  neurons *and* held-out sentences.
- **Controls:**
  - shuffled weights (same values, shuffled positions)
  - weights hidden
  - another sentence's label (the shown sentence is swapped)

## Task B: read a circuit (`circuit`)

- **Input.** Coordinates 0–127 of neuron A's **output** weights (`down_proj` column, layer L_A), and of 4 candidate
  neurons' **input** weights (`gate_proj` rows) in a later layer L_B = L_A + k (k ∈ {1, 2, 3}).
- **Label.** The candidate with the highest cosine between A's output slice and its input slice. This is the
  direct-path alignment on these coordinates, ignoring the norm scale. The answer is `1`–`4`; chance is 25%.
  Cosine is used because the scaling is per vector.
- **Candidates.** Same-layer neurons from the same split pool.
- **Controls:**
  - A's numbers shuffled
  - every candidate's numbers shuffled
- **Also reported:** accuracy by margin between the best and second-best cosine (tertiles).

## Ablation: raw floats (`raw_float`)

- **Run.** The round-4 logit-lens reader with the identical recipe, but numbers written **unscaled** as decimals
  with 3 significant digits (e.g. `0.0123 -0.00451`), 32 per line with index headers.
- **Question.** Is the per-neuron integer scaling needed? Outcomes are those of round 4: four-way PMI accuracy on
  the locked test split, shuffled, and hard distractors.

## Metrics and claims

- **Scoring.** For each held-out item, the log-probability of each answer's first token at the answer position;
  the highest wins.
- **Monitoring.** Val checks every 250 steps (100 items) use val neurons and, for task A, val sentences.
- **Final test.** 1,000 items once, built from held-out (test-pool) neurons and, for task A, test sentences. We
  report exact binomial or bootstrap 95% CIs.
- **Claim rule.** A task "works" if test accuracy has its CI lower bound above chance and the weights-shuffled
  control is within 5 points of chance. Everything is reported, including nulls.
- **Not claimed:**
  - for task A, that the model introspects on its own forward pass. It may learn statistics of sentences and
    layers. The weights-shuffled and other-sentence controls bound this.
  - for task B, anything about circuits beyond these coordinates' direct-path cosine.

## Amendment 1 (2026-11-05, 01:30 EST): task A changed to a balanced two-sentence comparison

- **The flaw.** The sign label was badly imbalanced: 12.5% UP in train and 14% in val. The untrained model already
  scored 82% by answering DOWN, so "chance 50%" was wrong. A few large, sign-consistent coordinates in the layer
  inputs probably dominate the dot product.
- **What we saw.** Only the zero-shot val check and the label balance had run (step 1 of training). The run was
  stopped and renamed `INVALID_task_act_4B_imbalanced`.
- **New task A.** The same weights and layer, but **two sentences**. Which sentence gives the larger contribution
  `Σ_{i<128} x_L[i]·W_gate[L][n, i]` at its last token? The answer is `1` or `2`, and chance is exactly 50% by
  construction.
- **Controls:** weights shuffled, weights hidden, and the two sentences replaced by two others with the labels
  kept. All other rules are unchanged, and accuracy is also reported by margin tertile.

## Amendment 2 (2026-11-05, 02:00 EST): raw-float slice series, before these runs

- **Why.** The integer-format second slice (coordinates 1280–1407) did not learn within 64k examples: 30.5% val.
  A CPU check on 300 test neurons found that rounding is not the cause; the rounded lens keeps 97.2% of label tokens
  on that slice, against 96.6% on 0–127. Slice 0–127 has a lower-rank unembedding (effective rank 102 against 125)
  and less varied labels, which may make it easier.
- **Runs.** To take rounding out of the question entirely, the identical recipe runs in **raw-float** format
  (unscaled weights, 3 significant digits, `--fmt f3`) on three slices:
  - 0–127 (`sweep_4B_f3`, already queued)
  - 640–767 (`sweep_4B_f3_s640`)
  - 1280–1407 (`sweep_4B_f3_s1280`)
- **Outcomes:** those of round 4 on the 4B locked test split, re-sliced: four-way PMI accuracy, shuffled, hard
  distractors and held-out layers. For each slice we also report the unembedding's effective rank.
- **Question.** Does reading work on raw floats, and is the slice dependence a property of the slice rather than
  of the number format?
