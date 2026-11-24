# write-a-state v0.1.0: build report (2026-11-23)

The goal: a closed-form weight edit to Qwen3.5-0.8B (rev 2fc06364715b967f1860aea9cf38778875588b17) that makes "the state of
Overlay" act like a US state at a given lat/lon. There is one Harbor task for each of the 10 multi-token states (centroids).
Every number below comes from the task's own verifier (`scorer/score.py`), run on CPU in float32 on macOS. The raw
results are in `data/baselines_raw.json` and the summary in `data/baselines.json`.

## Gate survival (the unedited model answers the real state correctly; `data/test_gate.json`)

| task | state | attribute items kept / 16 | relation pairs kept / 40 |
|---|---|---|---|
| loc01 | New Hampshire | 12 | 23 |
| loc02 | New Jersey | 14 | 24 |
| loc03 | New Mexico | 10 | 9 |
| loc04 | New York | 14 | 16 |
| loc05 | North Carolina | 15 | 18 |
| loc06 | North Dakota | 10 | 12 |
| loc07 | Rhode Island | 13 | 21 |
| loc08 | South Carolina | 14 | 17 |
| loc09 | South Dakota | 11 | 12 |
| loc10 | West Virginia | 8 | 23 |
| pooled | | 121 / 160 | 175 / 400 |

Collateral sets for each task:
- real-state prompts: the location's gated real-state prompts plus 56 gated control prompts. That is 98 to 141 prompts per task (114 for loc01).
- 16 neutral texts.
- 60 general-knowledge prompts.

## Baselines (reward = mean over the 10 tasks; n = 10 locations, and random-row has 10 seeds, so 100 runs)

| edit | reward | attributes | relations | gates pass | entries | worst gate values |
|---|---|---|---|---|---|---|
| reference (oracle) | **0.761** | 0.925 | 0.597 | 10/10 | 3,072 | real 1.00, KL 1.8e-5, GK 1.00 |
| map-quadratic row | 0.759 | 0.816 | 0.702 | 10/10 | 1,024 | real 1.00, KL 7.0e-5, GK 1.00 |
| knn3 row | 0.700 | 0.829 | 0.571 | 10/10 | 1,024 | real 1.00, KL 8.1e-5, GK 1.00 |
| mean single-token-state row | 0.253 | 0.059 | 0.446 | 10/10 | 1,024 | |
| random-row (10 seeds) | 0.094 (sd of seed means 0.050; max single run 0.50) | 0.122 | 0.067 | 100/100 | 1,024 | |
| no-op | 0.000 | 0 | 0 | 10/10 | 0 | |

Reference per task: 0.818, 0.850, 0.833, 0.875, 0.667, 0.750, 0.700, 0.833, 0.800, 0.482. It gets every gated
attribute right except West Virginia's coast question (3 items). Raw pooled accuracy for the reference is 0.963
on attributes and 0.854 on relations. The no-op scores 0.441 and 0.648 on the same items.

The rebuilt reference is bit-identical to the original edit (`~/wsp_unhinted/work/edits/L1.pt` vs `loc03.pt`, max
|diff| 0).

## What was verified
- `python -m wspnest validate`: ok. `pytest wspnest/tests`: 11 passed, 6 of them new in `test_write_a_state.py`.
  The new tests check that test and dev data are disjoint (templates, exemplar pools, locations, prompts, GK and
  texts), that train tasks carry no canary, that test tasks carry the canary, and that `read_edit` rejects malformed
  edits.
- `python -m wspnest check write-a-state` (selfcheck, about 35 min) passes every bar:
  - oracle_ok: all 10 tasks reproduce `data/reference_scores.json` exactly. loc01 ran twice through the CLI; the
    others ran twice in-process.
  - deterministic: true.
  - null = 0 (a missing submission scores 0).
  - no_canary and leaked are both empty.
  - `quality.json` is committed.
- Docker: image `wspnest-write-a-state:0.1.0` was built from the generated Dockerfile. `solve.sh` + `test.sh` ran on
  loc04 inside it (`--network none --cpus 4 --memory 8g`). Result: reward 0.875, identical to macOS; live base answers
  agree 100% with the recorded ones; verifier time 171 s. Harbor itself was not run.

## Open issues
1. Relations are partly answerable from the named real state alone. The no-op already gets 0.65 of gated pairs right
   and the mean row gets 0.81. Baseline-relative scoring discounts this, but the mean row still earns 0.45 on
   relations. Proposed fix for 0.2: choose comparison states near the target's latitude and longitude band, so
   that the answer depends on the target.
2. The reference and the map edit are tied (0.761 vs 0.759). The map edit is ahead at 5 locations, the reference at 2 (New Mexico, West Virginia), with 3 ties.
   The brief's "best known" reference is therefore not clearly best on this verifier.
3. The gates are loose for every listed edit: the worst gate value is far from its threshold. They catch damaging
   edits only.
4. Sample sizes per task are small (8–24 items per component). Per-task scores move in steps of 0.03–0.1.
5. The no-gradient rule is enforced by the instruction and by transcript audit only.
6. Software-sense " Overlay" text changes (reference KL 0.13 nats/token). This is reported, not gated.
