"""Load and validate task cards (tasks/<id>/task.yaml) and release manifests (releases/*.yaml)."""
import json
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(ROOT, "tasks")
RELEASES = os.path.join(ROOT, "releases")
SCHEMA = json.load(open(os.path.join(ROOT, "schemas", "task.schema.json")))


def _check(card, schema=SCHEMA, where="task"):
    """A small validator for the subset of JSON Schema the task schema uses (no external dependency)."""
    errs = []
    for k in schema.get("required", []):
        if k not in card:
            errs.append(f"{where}: missing '{k}'")
    for k, sub in schema.get("properties", {}).items():
        if k not in card:
            continue
        v = card[k]
        if "enum" in sub and v not in sub["enum"]:
            errs.append(f"{where}.{k}: {v!r} not in {sub['enum']}")
        t = sub.get("type")
        types = {"string": str, "boolean": bool, "number": (int, float), "array": list, "object": dict}
        if t and not isinstance(v, types[t]):
            errs.append(f"{where}.{k}: expected {t}")
            continue
        if "pattern" in sub and isinstance(v, str) and not re.match(sub["pattern"], v):
            errs.append(f"{where}.{k}: {v!r} does not match {sub['pattern']}")
        if t == "object":
            errs += _check(v, sub, f"{where}.{k}")
        if t == "array" and sub.get("minItems") and len(v) < sub["minItems"]:
            errs.append(f"{where}.{k}: needs at least {sub['minItems']} item(s)")
    return errs


def load_task(task_id, quality=True):
    path = os.path.join(TASKS, task_id, "task.yaml")
    card = yaml.safe_load(open(path))
    errs = _check(card)
    if card.get("id") != task_id:
        errs.append(f"task: id {card.get('id')!r} does not match folder {task_id!r}")
    if card.get("kind") in ("write", "localize") and not any(b.get("name", "").startswith("random") for b in card.get("baselines", [])):
        errs.append("task: write and localize tasks must report a random-edit baseline")
    canary = open(os.path.join(ROOT, "CANARY")).read().strip()
    if card.get("canary") != canary:
        errs.append("task: canary must be the suite's canary GUID (see CANARY)")
    if not os.path.exists(os.path.join(TASKS, task_id, card.get("entrypoint", "run.py"))):
        errs.append("task: entrypoint file not found")
    if not re.search(r"[0-9a-f]{40}", str(card.get("test", {}).get("pinned", ""))):
        errs.append("task: test.pinned must include a full 40-character revision or commit hash (QUALITY.md 1)")
    if quality:
        qp = os.path.join(TASKS, task_id, "quality.json")
        qd = json.load(open(qp)) if os.path.exists(qp) else None
        if not qd:
            errs.append(f"task: no quality.json; run python -m wspnest check {task_id} (QUALITY.md)")
        elif not qd.get("pass"):
            errs.append(f"task: quality bars not passing: {[k for k, v in qd.get('bars', {}).items() if not v]}")
        elif qd.get("version") != card.get("version"):
            errs.append(f"task: quality.json is for version {qd.get('version')}, card is {card.get('version')}; rerun check")
    if card.get("test", {}).get("frozen") is not True:
        errs.append("task: the test version must be frozen (test.frozen: true)")
    ad = card.get("adapter")
    if ad:
        if not ad.get("consent", {}).get("revocable"):
            errs.append("task: adapters need the upstream authors' revocable, opt-in consent (adapter.consent)")
        if not ad.get("parity", {}).get("match"):
            errs.append("task: adapters must pass a parity check against upstream's published numbers (adapter.parity)")
        elif not os.path.exists(os.path.join(TASKS, task_id, ad["parity"]["file"])):
            errs.append("task: adapter parity file not found")
    return card, errs


def all_tasks():
    return sorted(d for d in os.listdir(TASKS) if not d.startswith("_") and os.path.isdir(os.path.join(TASKS, d)))


def load_release(name):
    rel = yaml.safe_load(open(os.path.join(RELEASES, f"{name}.yaml")))
    errs = []
    for t in rel.get("tasks", []):
        if t["id"] not in all_tasks():
            errs.append(f"release {name}: unknown task {t['id']}")
            continue
        card, _ = load_task(t["id"])
        if card["version"] != t["version"]:
            errs.append(f"release {name}: {t['id']} is pinned to {t['version']} but the registry has {card['version']}")
    return rel, errs
