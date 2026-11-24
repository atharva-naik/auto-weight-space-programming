"""`wspnest check write-a-state` hook: run the generated tasks' own verifier locally (CPU, no Docker).

- oracle: the reference edit (solution/edit.pt) on every generated task, twice. The score is continuous, so the bar is
  oracle_ok: each task reproduces the reference's recorded score (data/reference_scores.json) within 0.01.
  The first task is run through the full command line (tests/score.py in a fresh process, as test.sh does); the
  others import the generated tests/score.py and share one loaded model (each edit is applied and undone exactly).
- null: a missing submission (no /app/edit.pt) through the command line; must score 0.
- canary / hidden answers: wspnest's _scan, plus a check that no hidden prompt appears in instruction.md or
  environment/.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REV = "2fc06364715b967f1860aea9cf38778875588b17"


def _model_dir(model_dir=None):
    if model_dir:
        return model_dir
    if os.environ.get("WSA_MODEL_DIR"):
        return os.environ["WSA_MODEL_DIR"]
    snap = os.path.expanduser(f"~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/{REV}")
    return snap if os.path.isdir(snap) else "/models/qwen3.5-0.8b"


def _cli(task_dir, edit, out, md):
    subprocess.run([sys.executable, os.path.join(task_dir, "tests", "score.py"), "--items", os.path.join(task_dir, "tests", "items.json"),
                    "--edit", edit, "--out", out, "--model-dir", md], check=False, stdout=subprocess.DEVNULL)
    return float(open(os.path.join(out, "reward.txt")).read())


def selfcheck(limit=None, model_dir=None, **_):
    sys.path.insert(0, ROOT)
    sys.path.insert(0, HERE)
    from wspnest.__main__ import _scan
    import generate
    md = _model_dir(model_dir)
    tmp = tempfile.mkdtemp(prefix="wspnest-check-write-a-state-")
    tasks_dir = os.path.join(tmp, "tasks")
    n = generate.generate(tasks_dir, limit=limit)
    names = sorted(os.listdir(tasks_dir))
    recorded = json.load(open(os.path.join(HERE, "data", "reference_scores.json")))["scores"]
    canary = open(os.path.join(ROOT, "CANARY")).read().strip()
    no_canary, leaked = _scan(tasks_dir, canary)
    for t in names:  # hidden prompts must not leak into what the agent sees
        d = os.path.join(tasks_dir, t)
        items = json.load(open(os.path.join(d, "tests", "items.json")))
        seen = open(os.path.join(d, "instruction.md")).read() + " ".join(
            open(os.path.join(r, f), errors="ignore").read() for r, _, fs in os.walk(os.path.join(d, "environment")) for f in fs)
        hidden = [a["overlay"] for a in items["attributes"]] + [p["overlay"] for r in items["relations"] for p in r["prompts"]]
        if any(h in seen for h in hidden) or "Overlay" not in open(os.path.join(d, "instruction.md")).read():
            leaked.append(t)
    # null: missing submission, full command line
    nulls = [_cli(os.path.join(tasks_dir, t), os.path.join(tmp, "missing.pt"), os.path.join(tmp, "null", t), md) for t in names]
    # oracle, twice
    first = names[0]
    runs = {first: [_cli(os.path.join(tasks_dir, first), os.path.join(tasks_dir, first, "solution", "edit.pt"),
                         os.path.join(tmp, f"oracle{k}", first), md) for k in range(2)]}
    spec = importlib.util.spec_from_file_location("wsa_verifier", os.path.join(tasks_dir, first, "tests", "score.py"))
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    if V.check_model_files(md, json.load(open(os.path.join(tasks_dir, first, "tests", "items.json")))["model_sha256"]):
        raise RuntimeError("model files do not match the pinned revision")
    tok, model = V.load_model(md)
    details = {}
    for t in names[1:]:
        d = os.path.join(tasks_dir, t)
        task = json.load(open(os.path.join(d, "tests", "items.json")))
        edit = V.read_edit(os.path.join(d, "solution", "edit.pt"), V.param_table(model))
        runs[t] = []
        for _k in range(2):
            det = V.evaluate(model, tok, task, edit, log=lambda m: None)
            runs[t].append(det["reward"])
        details[t] = det
        print(t, runs[t], flush=True)
    lid = lambda t: t.replace("write-a-state-", "")
    oracle = sum(v[0] for v in runs.values()) / len(runs)
    oracle_ok = all(abs(v[0] - recorded[lid(t)]) <= 0.01 and abs(v[1] - recorded[lid(t)]) <= 0.01 for t, v in runs.items())
    return dict(n=len(names), n_generated=n, n_scored=len(runs), oracle=oracle, oracle_ok=oracle_ok,
                null=sum(nulls) / len(nulls), deterministic=all(v[0] == v[1] for v in runs.values()),
                no_canary=no_canary[:20], leaked=leaked[:20],
                oracle_per_task={lid(t): v for t, v in runs.items()}, recorded=recorded, model_dir_used="local snapshot" if "cache" in md else md)


if __name__ == "__main__":
    print(json.dumps(selfcheck(**dict(a.split("=", 1) for a in sys.argv[1:])), indent=1))
