# Quality bars

Every task and adapter must meet these before it is merged. Bars marked **(auto)** are checked by
`python -m wspnest validate` or `python -m wspnest check <task>`; the rest are checked in review.

`wspnest check` writes `tasks/<id>/quality.json` with the result of every automated bar, and `wspnest validate` fails
for a task whose `quality.json` is missing or not passing. Commit `quality.json` with the PR.

## 1. The card and the pins

- **(auto)** The task card validates: required fields, credit, license, the suite's canary, a frozen test version.
- **(auto)** Everything the test version depends on is pinned to an immutable revision: model and dataset revisions,
  upstream commits (`test.pinned` must contain at least one full 40-character commit or revision hash).
- If the data's license does not allow copying, the task fetches it at run time (`license.redistributed: false`).

## 2. The verifier

- **(auto) Oracle:** the reference solution, run through the task's own verifier, scores at least 0.95 on average
  over the checked tasks (at least 20, or all if fewer). For tasks whose score is continuous rather than right or wrong
  (for example, the share of a behavior an edit recovers), the verifier must instead reproduce the reference
  solution's recorded score within 0.01 on every checked task.
- GPU tasks provide `selfcheck.py` (`selfcheck(**options) -> dict`), which `wspnest check` runs instead of Harbor;
  run it on a machine with a GPU.
- **(auto) Null:** an empty or no-op answer scores at most chance + 0.05.
- **(auto) Determinism:** the same answer gets the same score twice.
- **(auto) Hidden answers:** the answer key appears only in `tests/` and `solution/`, never in `instruction.md` or
  `environment/`.
- **(auto) Canary:** every generated `instruction.md` and every file under `tests/` contains the canary GUID (vendored
  code under `tests/lib/` or `tests/upstream/` is exempt).

## 3. Baselines and headroom

- At least one baseline with a score. Write and localize tasks must include a random-edit baseline.
- **Headroom:** at launch, no listed non-oracle baseline reaches 0.9 of the maximum score. A saturated task belongs in
  `retired`, not `frontier`.

## 4. Held-out hygiene

- Test items never overlap the train version (if any) or anything the solver is given (by item, and by prompt for
  chat data). Describe how this is guaranteed in `train.disjoint_from_test` and check it in a test.
- Data from the train version carries no canary.

## 5. Measurement

- Every reported number states its sample size. Use at least 50 scored items per reported number, or report a
  confidence interval.
- If the score is a proxy (log-probability, keyword match), the task also reports a behavioral measure (what the model
  actually does).
- Write tasks (tasks that edit weights) report side effects on something the edit should not change, and say how they
  are used in the score (for example, a gate: no more damage than a reference fine-tuning run).

## 6. Adapters

- **Consent:** the upstream authors opted in, and the card records who, when and that it is revocable.
- **Parity:** the adapter reproduces the upstream's published numbers through the generated tasks, within the larger of
  1 point or the published confidence interval, and the parity file is committed.

## 7. Documentation and review

- The card's `summary`, `metric.description` and `notes` say what is measured, how, and the known limitations;
  `runtime` says what hardware it needs.
- One maintainer approves the PR. For tasks built on someone else's work, the original authors are credited on the card.
