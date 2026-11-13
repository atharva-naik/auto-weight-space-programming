# Round 4 log: exploratory diagnostics (not pre-registered)

The pre-registered kill test is in `PREREG_ROUND4.md`, and its verdict comes only from runs `h64` and `h8`.
Everything below is exploratory. Compute: 4 H100 GPUs. Runs: `runs/round4/diag_*`.

All diagnostics use 128 coordinates unless stated, `h64` format, LoRA r = 32, lr 1e-4 and 16 items per step.
Val is 4-way T2 on held-out validation neurons (chance 25%):
- **PMI:** score with numbers minus score with numbers hidden; the primary metric
- **raw:** score with numbers
- **hidden:** score with numbers hidden; the label prior

## 2026-11-04

| run | setup | result |
|---|---|---|
| `h64_s1`, `h8_s1` (stopped) | pre-registered stage 1: random aligned 128-slices, 16k fresh items | val128 at chance (23% PMI on n = 400; mid-stage checks 19–30% on n = 100). Loss 3.28 → 3.0 within 200 steps, then flat (format plus label prior). |
| `diag_rand16` | random 16-coordinate slices, 16k fresh items | at chance for all 1,000 steps (18–29% PMI on n = 100) |
| `diag_fixed128` | always coordinates 0–127, 16k fresh items | at chance through step 400 (24–31%) |
| `diag_memorize64` | train on 64 val items; distractors not trained | 100% raw, but 100% hidden too. The label prior alone separates trained from untrained lists, so this is uninformative. |
| `diag_mem64_same` | 64 neurons of one layer, fixed slice, distractors from the same 64 (the prior cannot separate them), 200 steps (about 50 views each) | at chance through step 125, even though the loss reached 0.1 by step 60 (it memorized the lists, not which numbers go with which). Then step 150: 39% PMI; step 175: 64%; **step 200: 83% PMI / 89% raw vs 25% hidden.** |
| `diag_mem64_id` | same as above, but the prompt gives `(neuron #n)` instead of numbers (bug check) | same late takeoff: 47% at step 125, 98% at step 175, **100% PMI vs 19% hidden at step 200**. Numbers are about as easy to memorize as IDs. |

| `diag_fixed128` (final) | fixed 0–127, 16k fresh items | **27.8% PMI on n = 400, at chance** |
| `diag_G1_fresh64k` (running) | fixed 0–127, 64k fresh items | PMI on n = 100 by step: 250: 30%, 500: 25%, 750: 33%, 1000: 29%, 1250: 35%, 1500: 45%, 1750: 60%, **2000: 84%** (raw 37%, hidden 26%). Phase transition on held-out neurons. |
| `diag_G2_pool1024` (running) | fixed 0–127, pool of 1,024 training neurons | 250: 23%, 500: 30%, 750: 34%, 1000: 36%, 1250: 30%, 1500: 29%, 1750: 26%, 2000: 25%, 2250: 32%, 2500: 32%. Stopped at step about 2,600: repeating a small pool does not generalize. |

| `diag_dict_curr` (running) | coordinate-dictionary curriculum: 1-coordinate slices (48k), then random 16-slices (32k), then random 128-slices (32k) | (val at size 1 is degenerate: the label depends only on the coordinate and the sign of its value, so the 4 candidates share at most 2 distinct strings) |

| `diag_G1_fresh64k` (final) | fixed 0–127, 64k fresh items | ... 2500: 98%, 3000: 98%, 3500: 100%, 4000: 100% (n = 100). **Final val (n = 400): 99.5% PMI / 85% raw / 26% hidden.** |
| `confirm_G1_val` | G1 adapter reloaded, the same 400 val | normal 99.5% PMI (reproduces training), **scrambled 27.8% PMI (chance)** |
| `confirm_G1_fresh` | G1 adapter, 400 val-pool neurons never used in any check | **normal 99.75% PMI / 84% raw / 20% hidden; scrambled 23.5%.** Open vocabulary (no candidates; one pass at the first list slot): recall@8 10.0% raw / 4.6% PMI; median rank of the true tokens 562 (raw) / 6,284 (PMI) of 248k. It gets the flavour of the tokens (e.g. Chinese phrases), rarely the exact ones. |

**G1 interpretation (exploratory):** the reading is coordinate-based, since shuffling kills it, and it generalizes to fresh
neurons. It is reliable for telling a neuron from 3 random same-layer ones, but too coarse to name the promoted tokens.
"Internalized the logit lens" is too strong for now. Hard distractors and the base-model open-vocabulary floor are
queued.

**Reading so far:**
- The pipeline works, and the model can use the numbers: it learns to tell 64 trained 128-number strings apart.
- But it only starts conditioning on the input after a long plateau on the label prior, about 40 views per item,
  even for easy inputs.
- One pass over fresh neurons stays on the plateau.
- Next: does held-out reading emerge with more data (`G1`: 64k fresh items, fixed slice) or with repetition
  (`G2`: a pool of 1,024 training neurons seen about 60 times each, scored on held-out neurons)?

## 2026-11-04, 15:52 EST: sandbox the sandbox failed and all on-sandbox state was lost

**What happened.** The cloud sandbox failed at 20:52:50 UTC. That was
about 3 minutes after the 9B and 27B weight downloads started, with 6 model processes running. The likely cause is
memory: the root filesystem appears to be memory-backed, and the limit was 256 GiB.

**What was lost:** every run directory, including the G1 adapter. No results had been pulled to the laptop. The
numbers in this file were copied from live logs as they arrived and are the only record.

**State of the pre-registered round-4 runs at the time (never finished, so no verdict):**

| run | progress | result |
|---|---|---|
| `h64` | zero-shot evals done (n = 1,000) | test_neuron normal 25.7%, scrambled 26.1%, hard 25.4%; test_layer 26.4% PMI |
| `h64` | stage 1 (16k items, 128 coordinates) done | val128 23.0%, val512 25.0%, val2560 25.5% PMI (n = 400). Stage 2 was at step about 1,200, at chance on n = 100. |
| `h8` | zero-shot evals done | test_neuron normal 24.7%, scrambled 26.0%, hard 24.2%; test_layer 27.1% |
| `h8` | stage 1 | at step 800, still at chance (n = 100) |

**`diag_dict_curr` at the time:**
- dictionary stage (48k single-coordinate items): loss 3.4 → 1.1
- val16 straight after it: 28.0% PMI (n = 400)
- the random-16-slice stage had just started

**Sweep:** 0.8B and 2B had trained for about 3 minutes. No results.

**Lessons:**
- pull results to the laptop every couple of hours
- keep at most 4 model processes running at once
- download large weights one at a time
- ask for more memory

## 2026-11-04, 17:20 EST: write experiments (exploratory), design fixed before any result

- **Task** (`src/raw_writer.py`): given the top-8 promoted and top-8 suppressed tokens of coordinates 0–127 of a
  neuron's `down_proj` column, the model writes those 128 coordinates as integers. The format is the reader's:
  1 std = 20, 64 values per line with index headers.
- **Training:**
  - targets are the real neuron's quantized numbers
  - LoRA r = 32, lr 1e-4 cosine, 16 items per step, 64k fresh training neurons
  - checkpoint every 500 steps
- **Arms:**
  - `write_4B_scratch`: a fresh adapter
  - `write_4B_fromreader`: initialized from the `sweep_4B` reader adapter. Does reading transfer to writing?
- **Metrics, fixed now:** on 200 val neurons, and once at the end on 1,000 test_neuron neurons (the same neurons as
  the reader's test split):
  - `parse_ok`: all 128 values present
  - `recall_up` / `recall_down`: requested top-8 tokens that are in the written slice's own top-8 logit lens,
    computed exactly
  - `cos`: written vs true slice
- **Baselines on the same items (no LM):**
  - random Gaussian (floor)
  - the true quantized slice (ceiling)
  - the analytic writer v = g ⊙ (Σ U[up] − Σ U[down]). It is strong, because the embeddings are tied. Beating
    random shows the model can write; approaching analytic or true shows it writes well.
- **Not claimed:** any behavioural effect. Installing written vectors and measuring what the model then says is a
  separate, later step.

## 2026-11-04, 17:25 EST: 27B stopped, and the backup bug

- **27B** was stopped at step 540 to free GPUs 2–3 for the write runs (the user's priority call;
  `PREREG_ROUND4_SCALING.md` amendment 5).
- **Backup bug:** the auto-backup globbed `.log` files as directories, so every 30-minute upload after the first
  failed until 17:07 EST. The fix was to back up files and directories separately.
- **Restart bug:** restarting the loop with a `pgrep -f` pattern killed the exec shell itself, twice. It is now
  restarted through a helper script, which uses a PID file.

## 2026-11-04, 17:45 EST: mechanism probe (exploratory), hypothesis and criteria fixed before any measurement

- **Hypothesis.** The simple solution is that the reader rebuilds the shown slice inside its residual stream at the
  answer position, so that its own final norm and unembedding produce the logit lens. If so, then at the first
  answer slot (right after `PROMOTES: ["`) the residual stream, restricted to the shown coordinates S, should point
  along the true slice v[S] more than along the slices of the item's 3 same-layer distractors.
- **Measurement** (`src/mech_probe.py`): 200 test_neuron items of the run's test split, each with its own 3
  distractors. For each checkpoint, the base model (adapter disabled) and the final adapter, and for every layer's
  residual h_l at that position:
  - cos(h_l[S], v[S]), and the mean of the same over the 3 distractors
  - internal 4-way accuracy: the fraction of items where the true slice has the highest cosine (chance 25%)
- **Primary readout:** the residual just before the final norm. Per-layer values are descriptive only, with no
  best-layer selection.
- **Criteria:**
  - H1 (final adapter): internal 4-way accuracy at the pre-norm residual has a 95% CI lower bound above 25%, and
    the base model's is near 25%.
  - H2 (across the `sweep_4B_s1280` checkpoints, every 500 steps): internal accuracy stays near chance before the
    val jump and is high after it. We report its correlation with val accuracy, with no threshold.
- **If H1 fails:** the reader does not work by rebuilding the vector along these coordinates. We report that, along
  with the per-layer profile.

## 2026-11-04, 18:15 EST: norm-detection bug (caught; one run invalidated)

- **The bug.** The generic loader added for Llama detected the final-norm scale with a weight-mean heuristic (mean
  below 0.5 means (1+w)). That rule picks w for Qwen3.5, whose stored weights are large. Qwen3.5 really is (1+w),
  as confirmed against the atlas. I caught it when the blog-asset extraction printed `norm_param: w`.
- **The fix.** Read the norm's own forward code instead: `1 + self.weight` means (1+w), otherwise w. Verified:
  Qwen3.5 gives 1+w and Llama gives w.
- **What was affected:** only runs started between 17:55 and 18:15 EST.
  - `write_4B_fromreader` (3 minutes in): wrong requested tokens. Stopped, renamed
    `INVALID_write_4B_fromreader_badnorm`, and restarted from scratch with the fix.
  - `sweep_Llama-3.2-3B`: the heuristic gave w, which is correct for Llama (final-norm mean 1.63). Valid.
  - `mech_4B_final`: does not use the norm scale. Valid.
  - Every Qwen run started before 17:55 used the original (1+w) code. Valid.

## 2026-11-04, 18:10 EST: the replication passes on the locked test split

- `sweep_4B`, the identical recipe from scratch: test_neuron 1,000 items, **99.8% PMI** (85.3% raw, 26.9% hidden prior).
- Val, 400 items: 99.5%.
- Val checks first reached 50% at 24,000 examples, against 28,000 for the original run.

## Planned: an independent audit once all current runs finish

A subagent with fresh context reviews the claims against the result files and code: leakage, label correctness,
pre-registration compliance, and overclaiming in the blog. The user asked for this on 2026-11-04.

## 2026-11-04, 18:30 EST: mechanism probe, H1 passes

`mech_4B_final`: the `sweep_4B` final adapter, on 200 test_neuron items. The full per-layer profile is in
`runs/round4/mech_4B_final.json`.

| | internal 4-way accuracy (pre-norm residual) | mean cosine gap |
|---|---|---|
| trained reader | **200/200 (exact binomial 95% interval 98.2–100%)** | **0.54 (CI 0.53–0.56)** |
| base model | 23% (CI 17.5–29), chance | about 0 |

By layer: about 0 through layer 16. It rises to 0.06–0.21 cosine (34–59% internal accuracy) over layers 17–30, and
the last layer brings it to 0.54 cosine and 100%. Distractor slices stay at about 0 throughout. H2 (across the
second-slice checkpoints) is pending.

## 2026-11-04, 18:55 EST: audit note (external Codex audit), wording fixed

- **Finding.** The training sampler draws neurons **with replacement**: 64,000 examples cover **55,580 distinct**
  neurons. The 1,000 test_neuron items cover **977 distinct** neurons, and the test_layer items 993. There is still
  zero overlap with training. I confirmed these counts independently from `src/raw_splits.py` and `train_items`.
- **What changed.** The blog, the GIF captions and the thread draft said "a new neuron for every example" and "each
  from a different neuron". These now give the actual counts, and the blog has a highlighted sampling note in 2.3.
  The method is unchanged. Bootstrap CIs are over items, as the captions now say.
- The same applies to the exploratory sets. The 400 "never used in any check" items cover **363 distinct**
  neurons, and the 400 val items 380. Table 2, Figure 4 and the README now say so.

## 2026-11-04, 19:20 EST: external blog audit (Codex), applied

The audit is at `~/CMU/blog-audit.md`. It confirmed from the saved records: 998/1,000 four-way, 263/1,000 shuffled,
971/1,000 hard, 998/1,000 held-out layers. All six corrections and the smaller points were applied to the blog.

1. **Sampling** uses replacement: 55,580 distinct training neurons; 977, 993 and 199 distinct among the test_neuron,
   test_layer and probe items. Now described as "sampled from the full training pool".
2. **Labels** are exact for the float slice, but the model sees rounded integers. The rounded numbers' own lens
   recovers 96.2% up and 96.1% down of the label tokens (writer step-0 baseline, 200 val neurons). The "simple
   solution" is now described as approximate, and "only way to get them right" became "coordinate order matters".
   The frequent-token filter (`,`; `,` and ` (`) is now in the caption.
3. **Full column:** the curriculum was interrupted during its 512-coordinate stage. The full-column training stage
   was never reached, so it is not reported as a failure.
4. **Mechanism:** "partly reconstructs the direction of the slice" (alignment). It is not causal and not an exact
   reconstruction; an ablation or patching test is the follow-up.
5. **200/200:** the exact binomial 95% interval is 98.2–100%, replacing the degenerate bootstrap "100–100".
6. **Hard distractors:** nearest neighbours within the same-layer held-out pool (920 alternatives), not the whole
   layer.

**Smaller points:**
- "Nothing is learned" became "four-way accuracy stays near chance".
- The no-candidate metric is defined: first-token match 11.2%, exact strings 11.1–11.2%.
- Greedy generation is scored against the top-12 labels.
- The writer's true numbers are a rounded reference, not a ceiling.
- Readability: the lead and Figures 1–2 come first; one correct and one wrong example are in the main text, the rest
  behind `<details>`; GIFs have a reduced-motion static fallback; images have dimensions; muted text is darker for
  contrast; OG and description meta tags added.

## 2026-11-04, 19:35 EST: is 0.8B's weight data harder to read? (rounding check)

- **Method.** On 300 test neurons per model: compare the logit lens of the *rounded* shown integers with the float
  label (top-8 promoted). We also looked at the column's kurtosis, the share of tiny integers, and the number of
  distinct values.
- **0.8B:** the rounded lens recovers **96.8%** of the label tokens. Kurtosis median 3.4; 12.3% of integers have
  |x| ≤ 2; 58 distinct values.
- **4B:** **96.6%**; kurtosis 3.2; 14.2%; 54 distinct values.
- **Conclusion.** The shown numbers carry the label equally well for both models. A rounding or outlier explanation
  for 0.8B's low self-reading (35.8%) is ruled out.
- **Cross so far:** 0.8B reading 4B's weights scores 62.0% on 4B's test split, 24.1% shuffled and 44.7% with hard
  distractors.

## 2026-11-05, 00:45 EST: RL for writing, plus a behavioural edit test (exploratory; design fixed before running)

**Goal.** The model writes its own weights to a target, and the written weights change its behaviour when
installed. Both SFT writers are at recall@8 ≈ 0. RL needs a denser signal, so the RL reward here is dense.

**RL reward** (`src/rl_writer.py`), exact and computed from the weights. The model writes 128 integers for slice
0–127 of neuron (L, n), given that neuron's true top-8 promoted and suppressed tokens. We compute the written slice's
logit lens over all kept tokens. Each requested up-token earns its percentile rank (the share of tokens scoring
lower), and each down-token earns 1 − its percentile. The reward is the mean over the 16 tokens.

| writer | reward |
|---|---|
| random vector | ≈ 0.5 |
| perfect | ≈ 1 |
| incomplete answer (fewer than 128 values) | 0.25 |

**Algorithm.** A minimal GRPO with advantages normalized per group, K = 8 samples per prompt, 4 prompts per step,
temperature 1.0, LoRA lr 1e-5, no KL term (exploratory). It starts from the `write_4B_fromreader` SFT checkpoint and
runs on GPU 1.

**Metrics, fixed now:** 64 val neurons, greedy decoding, every 25 steps.
- parse rate, percentile reward, recall@8 up/down
- baselines on the same items: random and the analytic writer, g ⊙ (Σ U[up] − Σ U[down])

**Behavioural edit test**, the "meaningful update". Install the written slice into neuron (L, n)'s `down_proj`
column (coordinates 0–127, rescaled to the column's units). We then add the edit's contribution at layer L, a·Δ,
where a is the neuron's typical positive activation from the atlas `act_scale`. On 8 fixed neutral prompts we
measure the change in next-token log-prob at the last position, as Δlog p(requested up) − Δlog p(requested down),
in nats. The same is measured for an analytic edit and a random edit of equal norm. This runs on 32 val neurons at
step 0 and at the end.

**Claim rule.** "RL improves writing" requires the val percentile reward and recall@8 to rise above the step-0
policy and toward or past the analytic writer. "Meaningful edits" requires the written edit's behavioural effect to
exceed the random edit's, compared against the analytic edit. Nothing here touches the test split.

## 2026-11-05, 01:25 EST: why 0–127 works and 1280–1407 doesn't (exploratory)

**Slice hardness scan** (`runs/round4/slice_hardness.json`): all 20 slices of 128 coordinates, 300 test neurons,
the unembedding scaled by (1+w).

| slice | effective rank | most common label token | label entropy (bits) |
|---|---|---|---|
| 0–127 | **90.7** | 16× | 10.75 |
| 128–255 | 99.8 | 17× | 11.04 |
| 256–383 | 105.7 | 3× | 11.13 |
| 384–511 | 113.6 | 5× | 11.12 |
| 512–639 | 115.9 | 4× | 11.11 |
| every other slice | 116–122 | 3–5× | about 11.1 |

Rounding keeps 96–97.5% of label tokens on every slice. So the first few hundred coordinates are atypically
low-rank, 1280–1407 is a typical slice, and 0–127 is the easiest one.

**Next runs:**
- **transfer:** 1280–1407 initialized from the 0–127 reader
- **predicted-easy slice:** 128–255, from scratch
- **raw-float runs** on 0–127, 640–767 and 1280–1407, already running

**Causal ablation test** (`src/mech_causal.py`, 200 val items): at the answer positions, zero residual coordinates
0–127 in layers 17–31, against a control that zeros 1280–1407.

## 2026-11-05, 01:35 EST: scheduling incidents (no result affected)

- **Overload.** Stacking 3–4 4B jobs per GPU caused out-of-memory crashes at startup. The first `scale_4B_32k`
  attempt is set aside as `OOM_scale_4B_32k`, and two raw-float slice attempts as `OOM_*`. All are queued again with
  at most 2–3 jobs per GPU (a helper script).
- **Stale waiters.** Waiting shells left by earlier queue scripts fired at the same moment as the new queue. A
  duplicate `scale_4B_48k` started on GPU 3 and was stopped by PID; its directory is set aside as
  `DUP_scale_4B_48k_gpu3`. The 48k run will start on GPU 2 after 32k.
- **Self-kill.** Twice, a pattern-based `kill` matched the exec shell itself and cut a command short. From now on,
  only explicit PIDs are used, from a script file.

## 2026-11-05, 01:40 EST: causal ablation (exploratory) and confound checks

`mech_causal_4B`, the replication adapter on 200 val items. At the answer positions (the last K+1 positions of each
scored sequence) we zero residual coordinates. The numbers-hidden prompt gets the same ablation, so PMI stays fair.

| intervention | PMI accuracy |
|---|---|
| none | 99.5% |
| zero 0–127, layers 17–31 | **31.0%** |
| zero 0–127, last layer only | 36.5% |
| **control:** zero 1280–1407, layers 17–31 | **99.0%** |
| zero 0–127, layers 0–16 | 27.5% |

- **Possible confound.** Coordinates 0–127 are atypical: the lowest-rank slice, possibly with outlier activation
  dimensions. Zeroing them may damage the model in general. The early-layer result, which hurts as much although the
  alignment probe found nothing there, is consistent with that.
- **Checks running** (`src/mech_patch.py`):
  - general damage: the same ablation on the untrained model's wikitext next-token prediction, 0–127 vs 1280–1407
  - interchange patching on 100 val pairs: copy coordinates 0–127 at the answer positions in layers 17–31 from
    neuron i's run into neuron j's run, and see whether the choice flips to i's label. Control: copy 1280–1407.

## 2026-11-05, 02:05 EST: the confound checks settle the mechanism question

**General damage** (`mech_patch_4B`, untrained model, 100 wikitext passages, last 16 tokens). Zeroing residual
coordinates 0–127 costs **4.81** nats/token in layers 17–31 and 4.85 in layers 0–16. Zeroing 1280–1407 costs
**0.04** and 0.02. Coordinates 0–127 are generally critical, probably outlier dimensions. **So the zeroing result
(99.5% → 31%) is general damage and is not evidence about reading.**

**Interchange patching** (100 val pairs). The reader is run on neuron j, scoring {label_i, label_j}, with residual
coordinates at the answer positions in layers 17–31 copied from the run on neuron i.

| coordinates copied | picks label_i |
|---|---|
| **0–127** | **91%** |
| 1280–1407 | 6% |
| none | 5% |

Damage cannot make the model choose i's label. So the neuron-specific information that decides the answer is
carried in the shown coordinates of the late-layer residual stream. This is causal evidence on val items.

**Alignment probe on the failed integer 1280–1407 run** (`mech_s1280`, its checkpoints). Internal 4-way accuracy is
30–33% (CI lower bound about 0.26), and the cosine gap rises from 0.013 to 0.023: a faint alignment that never
takes off.

**Transfer.** 1280–1407 initialized from the 0–127 reader reaches **70%** val at 8k examples (40% at 4k); from
scratch it never exceeded 33% in 64k.

### 2026-11-05 05:00 UTC: data scaling 32k, emergence probe on 32k, GPU cut-back
- `scale_4B_32k` (fixed 0–127, budget 32k, cosine schedule over the budget): **val 68.75% PMI, n = 400**. The data-budget series is now 16k 27.8% (`diag_fixed128`), 24k 33.0%, 32k 68.75%, 64k 99.5% (all val, n = 400). 48k is running on the sandbox; 96k is running on the sandbox.
- `mech_emergence_32k` (alignment probe at checkpoints of the 32k run; val here is the probe's smaller set): base (gap n/a) → step 500 val 0.37, gap −0.001; 1000 val 0.39, gap 0.012; 1500 val 0.38, gap 0.026; 2000 val 0.60, gap 0.092 (CI 0.066–0.117). Alignment rises together with accuracy, and stays near zero while accuracy is flat. This is correlational, like the 64k probe.
- The user asked to drop back to 4 H100s. a helper script waits for the two the sandbox runs to end, runs a final HF backup, checks the adapters and log hashes are on HF, pulls the results, then terminates the sandbox. The terminate was approved by the user.
- 05:17 UTC: `task_circuit_4B` stopped by the user's decision (val 27% at 16k items, chance 25%, n = 100; about 13 h left at 15 s/step). Its GPU slot goes to the levers pilot. `task_act_4B` continues (55% at 48k items, chance 50%, n = 100).
- `r3_scale_4B_96k` finished: **val 100% PMI, n = 400** (hidden 27.75%). The data-budget series: 16k 27.8, 24k 33.0, 32k 68.75, 64k 99.5, 96k 100 (val, n = 400).
- Levers pilot (`src/levers_pilot.py`, plan in `docs/PLAN_LEVERS.md`). The smoke test at push 0.8 × the median residual norm (8.8 at layer 12) showed symmetric damage: push+ and push− both raised soft_pf by 0.06–0.14 and lowered math log-prob by 0.6–2.3 nats. Zero-ablation effects were about 1e-3. The pilot now logs odd = (push+ − push−)/2 as the steer label and even = the shared damage, with the push size as a flag. Calibration at 0.1 / 0.25 / 0.5 is running on GPU 1.

### 2026-11-05 06:00–07:00 UTC
- `sweep_4B_s128` (coordinates 128–255, from scratch, 64k): **test 98.9% (98.2–99.5)**, shuffled 26.2%, hard 92.3%, held-out layers 98.4%. Val was flat at about 40% until 32k, then rose to 98% by 52k. Slice effective rank: 0–127 91, 128–255 100, 1280–1407 120. That gives three from-scratch points, ordered by rank.
- `sweep_4B_s1280_transfer` locked test: 100.0% normal, 23.6% scrambled, 99.7% hard, 100.0% held-out layers.
- `task_act_4B` final (the eval reran with `--eval_only` after a scipy crash at the end of training): **test 50.1% (47.0–53.2), n = 1000, chance 50%**. Shuffled 48.9%, hidden 48.9%, other sentence 50.5%. A clean negative.
- The levers pilot switched to organic edits at the user's request (no tuned scale): x0, x2, and clamp to the neuron's own signed 99th-percentile |activation| on wikitext. Pilot A covers layers 6/12/18/24 and pilot B covers 3/9/27/30, 100 neurons each plus L12 N2920. There are 96 syco dev items and 96 GSM8K problems, in halves.
- Interim on 377 neurons (layers 3, 6, 9, 12, 27): split-half Spearman clamp/syco **0.68** (26 neurons above 3× noise, max |Δsoft_pf| 0.015), x2/syco 0.27, x0/syco 0.21. Math was about 0 for all edits, so single-neuron math effects are below noise at 48 problems per half.
- `levers_atp` (attribution patching for all 294,912 neurons, 77 s) vs the true edits on the same 377 neurons: Spearman clamp/syco **0.87**, x2/syco 0.41, x0/syco 0.40, clamp/math 0.50, x0/math 0.15.
- Next, `levers_groups`: clamp the top-k AtP neurons together on fresh items (dev_b), with random-k controls, plus a ridge probe from down_proj columns to the AtP clamp estimate, scored on the held-out layers.

### 2026-11-05 07:00 UTC: levers, group edits and weight probes (`levers_groups`)
- Group clamp of the top-k AtP neurons on fresh syco dev_b items (n = 200). Baseline: soft_pf 0.343, pf 21.7%, acc_neutral 67.5%.
  - **16 up: pf 61.5%** (soft 0.664), acc 51.5%. 16 down: soft 0.231, pf 19.0%, acc 66.5%.
  - 64 up: pf 35.0%, acc 30.5%. 64 down: soft 0.284, acc 53.5%.
  - 256 either way: acc 25% (broken).
  - Random-k (3 draws each): soft 0.31–0.36, pf 19–22%, acc 63.5–67.0%.
- Top-16 up neurons are in layers 0–16, with logit lens that looks like noise; the edit also costs accuracy, so it looks like general disruption. Top down neurons include L31 N1708 (suppresses "D") and L31 N688 (suppresses "A"): letter-prior neurons, which is a metric artefact. L12 N2920 ranks 2,639 of 294,912 (clamp est −0.0043).
- **Weight probes (ridge) for the AtP clamp estimate.**
  - Across layers (train on 28, test on 5/13/21/29): R² ≈ 0, Spearman ≈ 0, for the full column and for 0–127.
  - Within layers (a random half of the neurons held out, best λ per layer): R² ≈ 0, Spearman ≈ 0, for the full down column, 0–127, and the gate+up rows.
  - So the behavioural lever is not linearly readable from a neuron's own weights. This is evidence against the "find the levers from the weights" arm as planned.

### 2026-11-05 07:10–09:30 UTC
- `r3_sweep_4B_s1280_192k` (coordinates 1280–1407, from scratch, 192k budget): **test 100.0%**, scrambled 23.9%, hard 100.0%, held-out layers 100.0%, open-vocab recall@8 15.4% (base 0.0). Val by step: 3000 32%, 4000 71%, 5000 97%, 6000+ 100%. So it switched on between 48k and 80k examples; the 64k-budget run never did. The two runs differ in LR schedule as well as budget (not separated). An earlier note in this session, reading loss 2.7 at step 5120 as "not yet switched on", was wrong.
- `sweep_4B_f3` (raw weights, 3 significant digits, no scaling, 0–127): **test 99.7% (99.3–100.0)**, scrambled 25.5%, hard 97.8%, held-out layers 99.7%, open-vocab 14.3% (base 0.7). `sweep_4B_f3_s1280` (from scratch, 64k): 28.2% test, at chance, like the integer run.
- `scale_4B_48k`: val 96.5% (n = 400).
- Final levers pilots, n = 401 each. Clamp/syco split-half Spearman 0.70 (A, layers 6/12/18/24) and 0.52 (B, layers 3/9/27/30); 40 and 33 neurons above 3× noise. Math stays at noise level. L12 N2920 ranks 5/401 in both for clamp/syco. PCA-128 basis keeps 35–38% of the column norm.
- the sandbox was retired by a helper script: final backup verified (adapters on HF, log hashes match), pulled, terminated.
- 9B retry `sweep_9B_lr5e-5_128k` launched on the sandbox GPU 0 (lr 5e-5, 128k).

### 2026-11-05 13:00–14:30 UTC: topic-neuron steering (negative), switch to language neurons
- `src/topic_finder.py`. Reader topic scores (no-candidate first-slot PMI on the topic's single-token words) vs the exact slice lens (within-vocab z): Spearman 0.42 / 0.33 / 0.37 for cats / ocean / pizza on 2,048 held-out-layer neurons, but low top-1% overlap.
- Edit feasibility with the full-column lens as oracle (held-out layers 5/13/21/29), clamping to the neuron's own p99:
  - k = 8 to 2048: 0% cat mentions, ocean 3–9% (random 512 also 6%); topic-token mass changes 1–5×, the same as random.
  - At 10× max activation (the Golden Gate convention), every method, oracle and random, turns output into degenerate repetition ("cacaca", "se se se", "West East"; wiki NLL 4.6–21).
  - So single-neuron topic steering does not work here at any strength (polysemantic neurons). The reader scoring runs were stopped by explicit PID; no results kept.
- Literature: Tang et al. 2024 (arXiv 2402.16438): language-specific neurons in top and bottom layers; deactivating or activating them steers language ability. Panickssery et al. (2312.06681): CAA residual directions. AxBench (2501.17148): prompting beats steering methods; SAEs not competitive.
- Started `src/lang_neurons.py feasibility`: ×0 edits (no strength knob); methods are the full-column lens script z-score, LAPE-style activation share, and random; ko/ru/ja/zh plus en; k = 64/256/1024; ΔNLL on held-out Wikipedia paragraphs.
- `lang_feasibility` (×0 in base Qwen; ΔNLL in nats/token on 24 held-out Wikipedia paragraphs per language, first 192 tokens; base NLL ko 2.60, ru 1.84, ja 2.62, zh 2.62, en 2.95). Selectivity = Δtarget − mean Δ(other three languages).
  - **Korean.** Lens: 64 → +0.131 (sel 0.110); 256 → +0.142 (0.109); 1024 → +0.193 (0.064). LAPE: 64 → +0.066 (0.066); 256 → +0.140 (0.130); 1024 → +0.254 (0.217). Random: ≤ 0.01.
  - **Japanese.** Lens: 64 → +0.140 (sel 0.120); 1024 → +0.179 (0.126). LAPE: 1024 → +0.289 (0.253).
  - **Russian.** Lens: 64 → +0.049 (sel 0.041). LAPE: 256 → +0.139 (0.137); 1024 → +0.372 (0.364).
  - **Chinese.** Lens is not selective (256: zh +0.056, but ko +0.123 and en +0.042). LAPE: 1024 → +0.053 (0.046).
  - So weights-only picks (full-column lens) are selective for ko and ja, and at k = 64 beat the activation recipe. LAPE wins at larger k and for ru and zh. Effects are modest (+0.13 nats is about 14% perplexity).
- Codex editorial pass on the blog applied (`git apply` clean against matching source hashes; figures regenerated; claims spot-checked against result files).
- `lang_neurons reader_score`: 5 stretch readers (0, 128, 640, 1280, 2048) scoring 67,804 unseen neurons (held-out layers plus each training layer's excluded pool) for ko/ru/ja/zh script tokens. Smoke test (0–127 reader only, 2,048 neurons) vs the full-column lens: Spearman ko 0.21, ru 0.33, ja 0.31, zh 0.17.

### 2026-11-05 14:10–14:40 UTC
- `lang_edit` (×0 on candidates = 67,804 neurons unseen by the slice readers: held-out layers plus each layer's excluded pool). Reader = summed z of the 5 stretch readers' no-candidate script PMI.
  - Reader picks are at random level for every language and k (ko: 64 → −0.002, 1024 → +0.005; ja/ru/zh ≤ 0.011).
  - **The full-column lens restricted to the same candidates also fails** (ko 64: +0.004, vs +0.131 unrestricted), because the language neurons sit in layers 30–31 (ko lens top-64: 52 in layers 30–31) and 88% of those are in the readers' training pool.
  - LAPE on the same candidates: ko 1024 +0.075, ru 1024 +0.093.
  - So the test was uninformative about the reader. The candidate set lacked the levers.
- The user proposed finding the important neurons first and splitting them so both sides contain some. The NL reader now uses a stratified neuron split: per layer, 30% of the important neurons (lens top 2% for any language) and 30% of the rest are held out of NL training, plus the four held-out layers. Test and edit candidates come only from held-out neurons. Caveat: the init adapter (0–127 reader) saw coordinates 0–127 of many of these neurons for the token-list task.
- The first `nl_4B_full` run (layer-based split, trains on layers 30–31) was stopped after 15 min and renamed `INVALID_nl_4B_full_layersplit`.
- `sweep_9B_lr5e-5_128k` (lr 5e-5, 128k): **test 54.5%**, scrambled 24.2%, hard 45.0%, held-out layers 52.9% (vs 31.6% at lr 1e-4 and 64k). Val 66% at the end (n = 100), so it is partway through switching on.


(The remaining entries, about an on-policy self-distillation attempt that needs more debugging, are left out of
this public copy.)
