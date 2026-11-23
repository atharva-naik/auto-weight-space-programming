import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wspnest.registry import all_tasks, load_task, load_release


def test_all_tasks_valid():
    for t in all_tasks():
        _, errs = load_task(t)
        assert not errs, errs


def test_frontier_release_pins_existing_versions():
    _, errs = load_release("frontier")
    assert not errs, errs


def test_hide_numbers():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "logit-lens-4way", "scorer"))
    import score
    p = "Intro.\n\n[0] 1 2 3\n[64] 4 5 6\n\nRead these weights."
    assert score.hide_numbers(p) == "Intro.\n\n(numbers hidden)\n\nRead these weights."


def _card_errors(tmp_path, monkeypatch, edit):
    """Copy the wsp-gym card into a scratch registry, apply `edit`, and return its validation errors."""
    import shutil
    import yaml
    from wspnest import registry
    src = os.path.join(registry.TASKS, "wsp-gym")
    dst = tmp_path / "tasks" / "wsp-gym"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(os.path.join(registry.ROOT, "CANARY"), tmp_path / "CANARY")
    card = yaml.safe_load(open(dst / "task.yaml"))
    edit(card)
    yaml.safe_dump(card, open(dst / "task.yaml", "w"))
    monkeypatch.setattr(registry, "TASKS", str(tmp_path / "tasks"))
    monkeypatch.setattr(registry, "ROOT", str(tmp_path))
    return registry.load_task("wsp-gym")[1]


def test_adapter_rules(tmp_path, monkeypatch):
    assert _card_errors(tmp_path, monkeypatch, lambda c: None) == []
    assert any("consent" in e for e in _card_errors(tmp_path / "a", monkeypatch, lambda c: c["adapter"]["consent"].update(revocable=False)))
    assert any("parity" in e for e in _card_errors(tmp_path / "b", monkeypatch, lambda c: c["adapter"]["parity"].update(match=False)))
    assert any("frozen" in e for e in _card_errors(tmp_path / "c", monkeypatch, lambda c: c.pop("test")))


def test_wsp_gym_verifier_scores_oracle_and_empty(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("numpy")
    import json
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "tasks", "wsp-gym"))
    import generate
    try:
        generate.generate(str(tmp_path / "t"), split="test_id", limit=2)
    except Exception as e:  # needs network for the first fetch of the pinned upstream commit
        pytest.skip(f"upstream not available: {e}")
    for d in sorted(os.listdir(tmp_path / "t")):
        t = tmp_path / "t" / d
        ep = json.load(open(t / "tests" / "episode.json"))
        for answer, want in ((json.dumps(ep["inverse_edit"] if ep["fault"] else []), "1"), ("", "0")):
            open(tmp_path / "answer.txt", "w").write(answer)
            subprocess.run([sys.executable, "-I", str(t / "tests" / "score.py"), str(tmp_path / "answer.txt"),
                            str(t / "tests" / "episode.json"), str(tmp_path / "v")], check=True)
            assert open(tmp_path / "v" / "reward.txt").read().strip() == want
