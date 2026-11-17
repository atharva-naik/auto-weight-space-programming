"""python -m wspnest list | validate | generate <task> --out DIR [--opt k=v] | train-data <task> --out DIR [--opt k=v]
                    | run <task> [--opt k=v] [--saw-train] -- <harbor run args> | check <task> [--opt k=v] [--n N]"""
import argparse
import datetime
import importlib.util
import json
import os
import platform
import subprocess
import sys

from . import __version__
from .registry import ROOT, TASKS, all_tasks, load_release, load_task


def cmd_list(a):
    for t in all_tasks():
        c, errs = load_task(t)
        flag = "" if not errs else "  [INVALID]"
        print(f"{c['id']:<28} {c['version']:<8} {c['kind']:<9} {'auto' if c['auto'] else 'other':<6} {c['track']:<9} {c['title']}{flag}")


def cmd_validate(a):
    bad = 0
    for t in all_tasks():
        _, errs = load_task(t)
        for e in errs:
            print(f"{t}: {e}")
        bad += bool(errs)
    for f in sorted(os.listdir(os.path.join(ROOT, "releases"))):
        if f.endswith(".yaml"):
            _, errs = load_release(f[:-5])
            for e in errs:
                print(e)
            bad += bool(errs)
    print("ok" if not bad else f"{bad} problem(s)")
    sys.exit(1 if bad else 0)


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _load_generator(card, task_id, fn="generate"):
    spec = importlib.util.spec_from_file_location(f"task_{task_id}", os.path.join(TASKS, task_id, card["entrypoint"]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)


def _options(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k] = int(v) if v.isdigit() else v
    return out


def cmd_generate(a):
    card, errs = load_task(a.task)
    if errs:
        sys.exit("\n".join(errs))
    n = _load_generator(card, a.task)(a.out, **_options(a.opt))
    print(f"wrote {n} Harbor tasks to {a.out}")
    return card, n


def cmd_train_data(a):
    """Write a task's training data (its train version). Only gym-like tasks have one; it never overlaps the test version."""
    card, errs = load_task(a.task)
    if errs:
        sys.exit("\n".join(errs))
    if "train" not in card:
        sys.exit(f"{a.task} has only a test version: no training data exists for it")
    n = _load_generator(card, a.task, card["train"]["entrypoint"])(a.out, **_options(a.opt))
    print(f"wrote {n} training examples to {a.out} (disjoint from test: {card['train']['disjoint_from_test']})")


def cmd_run(a):
    """Generate the Harbor tasks, run them with Harbor, and write a WSPNest result manifest from the rewards."""
    card, n = cmd_generate(a)
    jobs = os.path.abspath(a.jobs_dir)
    before = set(os.listdir(jobs)) if os.path.isdir(jobs) else set()
    cmd = ["harbor", "run", "-p", a.out, "-o", jobs] + a.harbor_args
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=False)
    new = sorted(set(os.listdir(jobs)) - before) if os.path.isdir(jobs) else []
    rewards, groups = {}, {}
    for job in new:
        for root, _, files in os.walk(os.path.join(jobs, job)):
            if "reward.txt" in files:
                try:
                    r = rewards[os.path.relpath(root, jobs)] = float(open(os.path.join(root, "reward.txt")).read().strip())
                except ValueError:
                    continue
                if "details.json" in files:  # tasks may split items into subsets (e.g. wsp-gym: faulty vs healthy)
                    d = json.load(open(os.path.join(root, "details.json")))
                    if "clean" in d:
                        groups.setdefault("clean" if d["clean"] else "faulty", []).append(r)
    score = sum(rewards.values()) / len(rewards) if rewards else None
    subsets = {k: sum(v) / len(v) for k, v in sorted(groups.items())}
    manifest = {"task": card["id"], "task_version": card["version"], "harness_version": __version__,
                "harness_commit": _commit(), "options": _options(a.opt), "harbor_args": a.harbor_args,
                "harbor_jobs": new, "n_tasks": n, "n_scored": len(rewards), "metric": card["metric"]["name"],
                "score": score, "subsets": subsets,
                "train_version_seen": a.saw_train, "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(a.results, exist_ok=True)
    path = os.path.join(a.results, f"{card['id']}__{'_'.join(new) or 'nojob'}.json")
    json.dump({**manifest, "per_trial": rewards}, open(path, "w"), indent=1)
    print(json.dumps(manifest, indent=1))


def _rewards(jobs_dir, job):
    out = {}
    for root, _, files in os.walk(os.path.join(jobs_dir, job)):
        if "reward.txt" in files:
            try:
                trial = os.path.basename(os.path.dirname(root)) if os.path.basename(root) == "verifier" else os.path.basename(root)
                out[trial.split("__")[0]] = float(open(os.path.join(root, "reward.txt")).read().strip())  # key by task name
            except ValueError:
                pass
    return out


def _harbor(tasks_dir, agent, jobs_dir):
    before = set(os.listdir(jobs_dir)) if os.path.isdir(jobs_dir) else set()
    subprocess.run(["harbor", "run", "-p", tasks_dir, "-a", agent, "-o", jobs_dir, "-n", "4"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    new = sorted(set(os.listdir(jobs_dir)) - before)
    r = {}
    for j in new:
        r.update(_rewards(jobs_dir, j))
    return r


def _scan(tasks_dir, canary):
    """Canary in every instruction.md and tests/ file; the answer written by solution/solve.sh absent from the
    instruction and the environment (answers of 4+ characters; shorter ones such as a letter are checked in review)."""
    import re
    no_canary, leaked = [], []
    for t in sorted(os.listdir(tasks_dir)):
        d = os.path.join(tasks_dir, t)
        if not os.path.isdir(d):
            continue
        files = [os.path.join(d, "instruction.md")] + [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(d, "tests")) for f in fs]
        for f in files:
            try:
                txt = open(f, errors="ignore").read()
            except OSError:
                continue
            if f.endswith((".py", ".sh", ".md", ".json", ".jsonl", ".txt", ".toml")) and canary not in txt and "/upstream/" not in f and "/lib/" not in f:
                no_canary.append(os.path.relpath(f, tasks_dir))
        sol = open(os.path.join(d, "solution", "solve.sh")).read() if os.path.exists(os.path.join(d, "solution", "solve.sh")) else ""
        m = re.search(r"<<'?JSON'?\n(.*?)\nJSON", sol, re.S) or re.search(r"echo (\S+) >", sol)
        ans = (m.group(1).strip() if m else "")
        if len(ans) >= 4:
            env = " ".join(open(os.path.join(r, f), errors="ignore").read() for r, _, fs in os.walk(os.path.join(d, "environment")) for f in fs)
            if ans in open(os.path.join(d, "instruction.md")).read() or ans in env:
                leaked.append(t)
    return no_canary, leaked


def cmd_check(a):
    """Run the automated quality bars (QUALITY.md) and write tasks/<id>/quality.json."""
    import re
    import shutil
    import tempfile
    card, errs = load_task(a.task, quality=False)
    if errs:
        sys.exit("\n".join(errs))
    opts = {**card.get("check", {}).get("options", {}), **_options(a.opt)}
    chance = float(card["metric"]["chance"])
    canary = open(os.path.join(ROOT, "CANARY")).read().strip()
    hook = os.path.join(TASKS, a.task, "selfcheck.py")
    q = {"task": a.task, "version": card["version"], "harness_commit": _commit(), "options": opts,
         "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
         "pinned": bool(re.search(r"[0-9a-f]{40}", str(card.get("test", {}).get("pinned", ""))))}
    if os.path.exists(hook):  # GPU tasks: the task's own self-check (run on a machine with a GPU)
        spec = importlib.util.spec_from_file_location(f"selfcheck_{a.task}", hook)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        q.update(mode="selfcheck", **mod.selfcheck(**opts))
    else:
        tmp = tempfile.mkdtemp(prefix=f"wspnest-check-{a.task}-")
        tasks_dir, jobs = os.path.join(tmp, "tasks"), os.path.join(tmp, "jobs")
        n = _load_generator(card, a.task)(tasks_dir, **opts)
        names = sorted(os.listdir(tasks_dir))[: a.n]
        for extra in sorted(os.listdir(tasks_dir))[a.n:]:
            shutil.rmtree(os.path.join(tasks_dir, extra))
        oracle, oracle2, null = _harbor(tasks_dir, "oracle", jobs), _harbor(tasks_dir, "oracle", jobs), _harbor(tasks_dir, "nop", jobs)
        no_canary, leaked = _scan(tasks_dir, canary)
        mean = lambda r: sum(r.values()) / len(r) if r else None
        q.update(mode="harbor", n=len(names), n_generated=n, oracle=mean(oracle), null=mean(null),
                 deterministic=bool(oracle) and oracle == oracle2, n_scored=len(oracle), no_canary=no_canary[:20], leaked=leaked[:20])
    q["chance"] = chance
    q["bars"] = {"pinned": q["pinned"], "oracle": q["oracle_ok"] if "oracle_ok" in q else (q.get("oracle") or 0) >= 0.95, "null": q.get("null") is not None and q["null"] <= chance + 0.05,
                 "deterministic": bool(q.get("deterministic")), "canary": not q.get("no_canary"), "hidden": not q.get("leaked"),
                 "enough_checked": (q.get("n_scored") or 0) >= min(20, q.get("n_generated") or 20)}
    q["pass"] = all(q["bars"].values())
    json.dump(q, open(os.path.join(TASKS, a.task, "quality.json"), "w"), indent=1)
    print(json.dumps({k: q[k] for k in ("task", "mode", "oracle", "null", "chance", "bars", "pass") if k in q}, indent=1))
    sys.exit(0 if q["pass"] else 1)


def main():
    ap = argparse.ArgumentParser(prog="wspnest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(f=cmd_list)
    sub.add_parser("validate").set_defaults(f=cmd_validate)
    g = sub.add_parser("generate", help="write a task's Harbor task directories")
    g.add_argument("task")
    g.add_argument("--out", required=True)
    g.add_argument("--opt", action="append", help="generator option, e.g. --opt model=Qwen/Qwen3.5-4B --opt variant=agent")
    g.set_defaults(f=cmd_generate)
    t = sub.add_parser("train-data", help="write a task's training data, if it has a train version")
    t.add_argument("task")
    t.add_argument("--out", required=True)
    t.add_argument("--opt", action="append")
    t.set_defaults(f=cmd_train_data)
    c = sub.add_parser("check", help="run the automated quality bars and write tasks/<id>/quality.json")
    c.add_argument("task")
    c.add_argument("--opt", action="append")
    c.add_argument("--n", type=int, default=20)
    c.set_defaults(f=cmd_check)
    r = sub.add_parser("run", help="generate, run with Harbor, and record a result manifest; pass Harbor args after --")
    r.add_argument("task")
    r.add_argument("--out", default=os.path.join(ROOT, ".harbor_tasks"))
    r.add_argument("--opt", action="append")
    r.add_argument("--saw-train", action="store_true", help="the model was trained on this task's train version (reported separately)")
    r.add_argument("--jobs-dir", default=os.path.join(ROOT, "jobs"))
    r.add_argument("--results", default=os.path.join(ROOT, "results"))
    r.add_argument("harbor_args", nargs=argparse.REMAINDER)
    r.set_defaults(f=cmd_run)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
