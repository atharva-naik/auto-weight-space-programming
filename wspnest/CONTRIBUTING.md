# Contributing a task

New to WSPNest? Bring task ideas and questions to the [WSPNest Discord](https://discord.gg/fzTwQeq6g) before opening a PR.

1. **Ask the original authors first** if the task is built on someone else's work, and link their reply in the PR.
   Credit them in `credit.authors`; list yourself in `credit.adapted_by`.
2. **Check the license.** If the data or code license does not allow copying, do not copy it: set
   `license.redistributed: false` and have `generate.py` download from, or call, the original source.
3. **Copy `tasks/_template/`** to `tasks/<task-id>/` and fill in every field of `task.yaml`.
4. **Implement `generate(out, **options)`** in the entrypoint. It writes one Harbor task directory per item and
   returns how many it wrote. The verifier writes `reward.txt` (and optionally `details.json`) to `/logs/verifier`.
5. **Baselines.** Report at least one. Write and localize tasks must report a random-edit baseline: random edits are
   often surprisingly strong.
6. **Reproduction check.** If the task comes from a published result, show in the PR that it reproduces it (or
   explain the difference). Adapters must pass a parity check (`adapter.parity` on the card).
7. **Include the canary GUID** from `CANARY` in your task card (`canary:` field) and in any data files you add.
8. **Run `python -m wspnest validate`** and add a test in `tests/` if the task has non-trivial parsing.

**Test and train versions.** Every task has a **test version**: frozen, pinned for reproduction (dataset
revision, upstream commit or seeds), carrying the canary, and the only thing releases score. Gym-like tasks can also
have a **train version** (`train:` on the card, `generate_train(out, **options)` in the entrypoint): a generator of
training data that is provably disjoint from the test version and carries **no** canary, since the canary marks test
data. Tasks with only a test version have no training data. Results from a model trained on a task's train version
are reported separately (`wspnest run --saw-train`).

New tasks start on the `frontier` track. **Versioning:** bump the major version if scores are no longer comparable,
the minor version for new items or splits, and the patch version for fixes that keep scores comparable.

## Worked example: `tasks/logit-lens-4way`

Everything in WSPNest runs on Harbor. A task recipe is a `generate.py` that writes standard Harbor task directories
(instruction, `task.toml`, container, verifier, reference solution). A benchmark adapter has the same shape: its
`generate.py` fetches the upstream benchmark at a pinned commit and writes one Harbor task per upstream instance, so an
adapter is a for loop over task-level Harbor containers.

The reading task from Part 1 of the Auto Weight-Space Programming series generates two variants of the same items:

- **`variant=agent`**: for any Harbor agent and model, including API models. The agent writes A-D to
  `/app/answer.txt`. Internet is off, so it must read the numbers rather than download the model. This is WSP on
  another model.
- **`variant=scorer`**: for open-weight models reading their own weights (auto-WSP). The WSPNest agent
  `wspnest.agents.logprob:LogprobReader` scores the four lists by log-likelihood inside a GPU container, as in Part 1.

```
python -m wspnest generate logit-lens-4way --out tasks_out --opt variant=agent --opt limit=100
harbor run -p tasks_out -a oracle                                  # the reference solution must pass every task
python -m wspnest run logit-lens-4way --opt variant=agent -- -a claude-code -m anthropic/<model>
python -m wspnest run logit-lens-4way --opt variant=scorer -- --agent-import-path wspnest.agents.logprob:LogprobReader \
    --ak model=Qwen/Qwen3.5-4B --ak adapter=<reader LoRA> -e modal
```

`wspnest run` generates the tasks, runs Harbor, and writes a WSPNest result file from Harbor's rewards.

Things to copy from it:
- the task card's credit and license fields;
- the canary in every generated file;
- the oracle solution;
- the reproduction check against published numbers (see the task card's notes).

## Worked example: a benchmark adapter, `tasks/wsp-gym`

[WSP-Gym](https://github.com/emeraldant/WSP-Gym) (MIT) is included with its author's opt-in, revocable consent,
recorded on the card (`adapter.consent`). The adapter copies nothing into WSPNest:

- `upstream.py` fetches WSP-Gym at a pinned commit (only the files it needs) into a local cache;
- `generate.py` regenerates an upstream test split from its seeds and writes one Harbor task per network: the
  upstream prompt as the instruction, and in `tests/` the episode plus a pinned copy of the upstream scorer (hidden
  from the agent); the oracle answer is the upstream inverse edit;
- `generate_train` streams training data from upstream's own training seeds, which never overlap the test seeds;
- `parity.py` runs the upstream non-LLM baselines and the oracle through the generated verifiers and compares them
  with the published table (`parity-test_id.json`: all five match to 4 decimals).

```
python -m wspnest generate wsp-gym --out tasks_out --opt split=test_id --opt variant=direct
harbor run -p tasks_out -a oracle                                  # 1.000 on every task
python -m wspnest train-data wsp-gym --out train_out --opt n=10000   # optional: training data, no canary
python tasks/wsp-gym/parity.py test_id                             # about 10 minutes on a laptop CPU
```

Things to copy from it: the pinned commit, the consent record, the parity file, and keeping the upstream scorer as
the only judge.
