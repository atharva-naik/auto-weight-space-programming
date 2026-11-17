"""Fetch WSP-Gym at the pinned commit (only the files the adapter uses) into a local cache, and import it."""
import os
import subprocess
import sys

REPO = "https://github.com/emeraldant/WSP-Gym"
COMMIT = "d050d16dda5ed3243f5c59d115e931e44ef3da53"
PATHS = ["LICENSE", "weightgym"]
VERIFIER_FILES = ["weightgym/__init__.py", "weightgym/env.py", "weightgym/edits.py", "weightgym/faults.py", "weightgym/nets.py"]
CACHE = os.path.expanduser(os.environ.get("WSPNEST_CACHE", "~/.cache/wspnest/upstream"))


def fetch(commit=COMMIT):
    """Return the path of a checkout of WSP-Gym at `commit`. Nothing from it is committed to WSPNest."""
    d = os.path.join(CACHE, f"wsp-gym-{commit[:12]}")
    if os.path.exists(os.path.join(d, "weightgym", "env.py")):
        return d
    os.makedirs(d, exist_ok=True)
    run = lambda *a: subprocess.run(["git", "-C", d, *a], check=True, capture_output=True)
    run("init", "-q")
    run("remote", "add", "origin", REPO)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    run("checkout", "-q", "FETCH_HEAD", "--", *PATHS)
    return d


def load(commit=COMMIT):
    d = fetch(commit)
    if d not in sys.path:
        sys.path.insert(0, d)
    import weightgym.splits  # noqa: F401
    return d
