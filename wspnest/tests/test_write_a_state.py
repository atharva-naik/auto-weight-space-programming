"""write-a-state: the train version is disjoint from the test version, generated files carry (or lack) the canary as
they should, and the verifier's submission checks reject malformed edits. No model needed."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = os.path.join(HERE, "tasks", "write-a-state")
sys.path.insert(0, TASK)
sys.path.insert(0, os.path.join(TASK, "scorer"))
import importlib.util  # noqa: E402

import geodata as g  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

CANARY = open(os.path.join(HERE, "CANARY")).read().strip()


def _all_prompts(split):
    locs = g.TEST_LOCATIONS if split == "test" else g.TRAIN_LOCATIONS
    out = set()
    for loc in locs:
        it = g.location_items(loc, split)
        for a in it["attributes"]:
            out |= {a["overlay"], a["prefix"]}
        for r in it["relations"]:
            out.add(r["prefix"])
            out |= {p["overlay"] for p in r["prompts"]}
    return out


def test_templates_and_exemplars_disjoint():
    for qt in g.QTYPES:
        assert not set(g.TEST_ATTR_TEMPLATES[qt]) & set(g.DEV_ATTR_TEMPLATES[qt])
    assert not {t for t, _ in g.TEST_DIR_TEMPLATES} & {t for t, _ in g.DEV_DIR_TEMPLATES}
    assert not set(g.TEST_CLOSE_TEMPLATES) & set(g.DEV_CLOSE_TEMPLATES)
    assert not set(g.TEST_POOL) & set(g.DEV_POOL)                      # exemplar states
    assert not set(g.MULTI_TOKEN) & (set(g.TEST_POOL) | set(g.DEV_POOL))  # no test location is ever an exemplar
    assert not set(g.TEST_NEUTRAL) & set(g.DEV_NEUTRAL)
    assert not set(g.TEST_SOFTWARE) & set(g.DEV_SOFTWARE)
    test_gk = {(p, t.format(q=q)) for p, t, qs in g.TEST_GK.values() for q in qs}
    dev_gk = {(p, t.format(q=q)) for p, t, qs in g.DEV_GK.values() for q in qs}
    assert not test_gk & dev_gk
    assert not {p for p, _ in test_gk} & {p for p, _ in dev_gk}


def test_locations_and_prompts_disjoint():
    test_pts = {(l["lat"], l["lon"]) for l in g.TEST_LOCATIONS}
    assert not test_pts & {(l["lat"], l["lon"]) for l in g.TRAIN_LOCATIONS}
    assert not {l["state"] for l in g.TEST_LOCATIONS} & {l["state"] for l in g.TRAIN_LOCATIONS}
    assert not _all_prompts("test") & _all_prompts("train")


def test_items_well_formed():
    for loc in g.TEST_LOCATIONS:
        it = g.location_items(loc)
        assert len(it["attributes"]) == 16 and len(it["relations"]) >= 30
        for r in it["relations"]:
            golds = [p["gold"] for p in r["prompts"]]
            assert sorted(golds) == [0, 1]          # each pair has one answer per option: no constant-answer credit
        for a in it["attributes"]:
            assert g.NEW not in a["prefix"] and loc["state"] not in a["prefix"]


def test_train_tasks_have_no_canary_and_no_test_prompts(tmp_path):
    pytest.importorskip("yaml")
    generate = _load("wsa_generate", os.path.join(TASK, "generate.py"))
    if not os.path.exists(os.path.join(TASK, "data", "test_gate.json")):
        pytest.skip("data/test_gate.json not built")
    generate.generate_train(str(tmp_path / "train"), limit=3)
    test_prompts = _all_prompts("test")
    for root, _, files in os.walk(tmp_path / "train"):
        for f in files:
            txt = open(os.path.join(root, f), errors="ignore").read()
            assert CANARY not in txt, f
            if f == "items.json":
                items = json.loads(txt)
                assert not {a["overlay"] for a in items["attributes"]} & test_prompts


def test_test_tasks_carry_canary(tmp_path):
    generate = _load("wsa_generate", os.path.join(TASK, "generate.py"))
    if not os.path.exists(os.path.join(TASK, "data", "test_gate.json")):
        pytest.skip("data/test_gate.json not built")
    assert generate.generate(str(tmp_path / "test")) == 10
    for d in os.listdir(tmp_path / "test"):
        t = tmp_path / "test" / d
        for f in ["instruction.md", "tests/score.py", "tests/test.sh", "tests/items.json"]:
            assert CANARY in open(t / f).read(), f
        assert (t / "solution" / "edit.pt").exists()


def test_read_edit_rejects_malformed(tmp_path):
    torch = pytest.importorskip("torch")
    score = _load("wsa_score", os.path.join(TASK, "scorer", "score.py"))
    params = {"a.weight": torch.zeros(4, 3), "b.weight": torch.zeros(5)}
    good = {"a.weight": (torch.tensor([0, 5]), torch.tensor([1.0, 2.0])), "_base_revision": score.REVISION}

    def check(sub):
        torch.save(sub, tmp_path / "e.pt")
        return score.read_edit(str(tmp_path / "e.pt"), params)

    e = check(good)
    ap = score.Applier(params, e)
    assert ap.cost()[0] == 2
    ap.apply(); assert float(params["a.weight"].view(-1)[5]) == 2.0
    ap.undo(); assert float(params["a.weight"].abs().sum()) == 0.0
    bad = [dict(good, _base_revision="x"),
           {**good, "c.weight": (torch.tensor([0]), torch.tensor([1.0]))},
           {**good, "a.weight": (torch.tensor([0, 0]), torch.tensor([1.0, 2.0]))},
           {**good, "a.weight": (torch.tensor([12]), torch.tensor([1.0]))},
           {**good, "a.weight": (torch.tensor([1]), torch.tensor([float("nan")]))},
           {**good, "a.weight": (torch.tensor([1, 2]), torch.tensor([1.0]))}]
    for b in bad:
        with pytest.raises(score.InvalidEdit):
            check(b)
