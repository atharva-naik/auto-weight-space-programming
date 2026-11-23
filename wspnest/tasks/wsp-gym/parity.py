"""Parity check for the WSP-Gym adapter: run the upstream non-LLM baselines (and the oracle) through the generated
tasks' own verifier, and compare with WSP-Gym's published table (results/table_main.csv at the pinned commit).

  python3 -I tasks/wsp-gym/parity.py [split]     (default test_id; needs numpy==2.5.4, no GPU, no Docker)
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate  # noqa: E402
import upstream  # noqa: E402


def published(split):
    src = upstream.fetch()
    path = os.path.join(src, "results", "table_main.csv")
    if not os.path.exists(path):
        subprocess.run(["git", "-C", src, "checkout", "-q", "FETCH_HEAD", "--", "results/table_main.csv"], check=True)
    rows = {r["arm"]: r for r in csv.DictReader(open(path)) if r["split"] == split}
    return {k[len("baseline_"):]: (float(v["repair_rate"]), float(v["clean_ok"])) for k, v in rows.items() if k.startswith("baseline_")}


def main(split="test_id"):
    import numpy as np
    upstream.load()
    from weightgym.baselines import POLICIES
    out = tempfile.mkdtemp(prefix="wspnest-wspgym-")
    n = generate.generate(out, split=split, variant="direct")
    tasks = sorted(os.listdir(out))
    eps = generate.episodes(split)
    pub = published(split)
    report = {"split": split, "n": n, "upstream_commit": upstream.COMMIT, "arms": {}}
    for arm in ["oracle", "noop", "random", "outlier", "search"]:
        rng = np.random.default_rng(0)  # same seed and order as weightgym.eval.run_policy
        ok = {True: [], False: []}
        for t, ep in zip(tasks, eps):
            ans = json.dumps(ep.inverse_edit if not ep.is_clean else []) if arm == "oracle" else POLICIES[arm](ep, rng)
            with tempfile.TemporaryDirectory() as tmp:
                open(os.path.join(tmp, "answer.txt"), "w").write(ans)
                subprocess.run([sys.executable, "-I", os.path.join(out, t, "tests", "score.py"), os.path.join(tmp, "answer.txt"),
                                os.path.join(out, t, "tests", "episode.json"), tmp], check=True)
                ok[ep.is_clean].append(open(os.path.join(tmp, "reward.txt")).read().strip() == "1")
        repair, clean = sum(ok[False]) / len(ok[False]), sum(ok[True]) / len(ok[True])
        want = pub.get(arm, (1.0, 1.0) if arm == "oracle" else (None, None))
        match = abs(repair - want[0]) < 5e-4 and abs(clean - want[1]) < 5e-4
        report["arms"][arm] = {"repair_rate": round(repair, 4), "clean_ok": round(clean, 4),
                               "published_repair_rate": want[0], "published_clean_ok": want[1], "match": match}
        print(f"{arm:8s} repair {repair:.4f} (published {want[0]})  clean_ok {clean:.4f} (published {want[1]})  {'OK' if match else 'MISMATCH'}")
    json.dump(report, open(os.path.join(HERE, f"parity-{split}.json"), "w"), indent=1)
    return all(a["match"] for a in report["arms"].values())


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:2]) else 1)
