"""Tests for ttt-arc-adapter-read: the committed items, the adapter stripping, and an offline generate() run."""
import importlib.util
import json
import os

import pytest

TASK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "ttt-arc-adapter-read")


def load(name):
    spec = importlib.util.spec_from_file_location(f"ttt_arc_{name}", os.path.join(TASK, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ITEMS = json.load(open(os.path.join(TASK, "items.json")))


def test_items_are_consistent():
    val, test, ids = ITEMS["validation"], ITEMS["test"], ITEMS["puzzles"]
    assert (len(val), len(test), len(ids)) == (96, 300, 396)
    assert not set(val) & set(test) and set(val) | set(test) == set(ids)
    assert not set(ids) & set(ITEMS["no_adapter"])
    for split, n in (("validation", 96), ("validation_hard", 96), ("test", 300), ("test_hard", 300)):
        its = ITEMS["items"][split]
        assert [it["adapter"] for it in its] == ITEMS[split.split("_")[0]]
        assert len(its) == n
        for it in its:
            assert len(set(it["candidates"])) == 4 and set(it["candidates"]) <= set(ids)
            assert it["candidates"][it["answer"]] == it["adapter"]
    answers = [it["answer"] for it in ITEMS["items"]["test"]]
    assert all(answers.count(k) > 50 for k in range(4))  # answer letters are balanced


@pytest.mark.skipif(not os.environ.get("ARC_EVAL_DIR"), reason="set ARC_EVAL_DIR to the ARC-AGI data/evaluation folder")
def test_make_items_reproduces_items_json():
    mk = load("make_items")
    assert mk.build(mk.load_puzzles(os.environ["ARC_EVAL_DIR"])) == ITEMS


def test_strip_adapter():
    torch = pytest.importorskip("torch")
    gen = load("generate")
    state = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(1),
             "base_model.model.lm_head.lora_B.weight": torch.ones(1)}
    assert set(gen.strip_adapter(state)) == {"model.layers.0.self_attn.q_proj.lora_A.weight", "lm_head.lora_B.weight"}
    assert set(gen.strip_adapter(state, head=False)) == {"model.layers.0.self_attn.q_proj.lora_A.weight"}


@pytest.mark.parametrize("split", ["test", "test_nohead"])
def test_generate_offline_and_no_leak(tmp_path, split):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    gen = load("generate")
    cache, arc, out = tmp_path / "cache", tmp_path / "arc", tmp_path / "out"
    (cache / "adapters").mkdir(parents=True)
    arc.mkdir()
    (cache / "tokenizer.json").write_text("{}")
    its = ITEMS["items"]["test"][:2]
    for it in its:
        st.save_file({"model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 3, dtype=torch.bfloat16),
                      "lm_head.lora_A.weight": torch.ones(2, 3, dtype=torch.bfloat16)},
                     str(cache / "adapters" / f"{it['adapter']}.safetensors"))
        for c in it["candidates"]:
            (arc / f"{c}.json").write_text(json.dumps({"train": [{"input": [[1]], "output": [[2]]}],
                                                       "test": [{"input": [[3]], "output": [[4]]}]}))
    assert gen.generate(str(out), split=split, limit=2, cache=str(cache), arc_dir=str(arc)) == 2
    dirs = sorted(os.listdir(out))
    assert dirs == [f"ttt-arc-adapter-read-{split.replace('_', '-')}-{i:04d}" for i in range(2)]
    for d, it in zip(dirs, its):
        d = out / d
        letter = "ABCD"[it["answer"]]
        assert f'"$got" = "{letter}"' in (d / "tests" / "test.sh").read_text()
        assert f"echo {letter}" in (d / "solution" / "solve.sh").read_text()
        cands = json.loads((d / "environment" / "candidates.json").read_text())
        assert sorted(k for k in cands if k != "canary") == list("ABCD") and "test" not in cands["A"]
        keys = set(st.load_file(str(d / "environment" / "adapter.safetensors")))
        assert ("lm_head.lora_A.weight" in keys) == (split == "test")
        # no puzzle id anywhere the agent can see
        seen = b"".join(p.read_bytes() for p in (d / "environment").iterdir()) + (d / "instruction.md").read_bytes()
        for pid in it["candidates"]:
            assert pid.encode() not in seen
        text = (d / "instruction.md").read_text()
        assert gen.CANARY in text and "{canary}" not in text and "{head" not in text and "{tok_note}" not in text
        assert ("lm_head matrices have been removed" in text) == (split == "test_nohead")
