"""Freeze the test version's gate lists and base answers (needs the model; run once per task version).

For each test location: score every prompt with the unedited model, keep the attribute items whose real-state prompt
the model answers correctly (greedy) and the relation pairs whose two real-state prompts it answers correctly. Writes
data/test_gate.json: the model files' sha256, a hash of the generated items (generate.py refuses to run if the
prompts changed), and per location the gate lists and the base model's greedy token on every kept prompt.

Usage: WSA_MODEL_DIR=<snapshot dir> python build_data.py
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scorer"))
import geodata  # noqa: E402

MODEL_FILES = ["model.safetensors-00001-of-00001.safetensors"]


def task_core(loc, split="test"):
    it = geodata.location_items(loc, split)
    return dict(id=loc["id"], lat=loc["lat"], lon=loc["lon"], attributes=it["attributes"], relations=it["relations"],
                controls=geodata.control_items(split), gk=geodata.gk_items(split), texts=geodata.texts(split))


def items_hash(core):
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def main():
    import score
    md = os.environ["WSA_MODEL_DIR"]
    shas = {f: score.sha256(os.path.join(md, f)) for f in MODEL_FILES}
    tok, model = score.load_model(md)
    canary = open(os.path.join(HERE, "..", "..", "CANARY")).read().strip()
    out = dict(canary=canary, revision=geodata.REVISION, model_sha256=shas, locations={})
    cores = {loc["id"]: task_core(loc) for loc in geodata.TEST_LOCATIONS}
    # one scoring call over all locations (shared prefixes are computed once)
    allP, owner = [], []
    for lid, core in cores.items():
        P = score._prompt_list(core, None)
        allP += P; owner += [lid] * len(P)
    uniq = {}
    for p in allP:
        uniq.setdefault((p["prefix"], p["suffix"], tuple(p["cands"])), p)
    keys = list(uniq)
    ures = dict(zip(keys, score.score_prompts(model, tok, [uniq[k] for k in keys])))
    res = [ures[(p["prefix"], p["suffix"], tuple(p["cands"]))] for p in allP]
    for lid, core in cores.items():
        P = [p for p, o in zip(allP, owner) if o == lid]
        R = [r for r, o in zip(res, owner) if o == lid]
        ok_a, ok_r = set(), {}
        for p, r in zip(P, R):
            if p["who"] != "real":
                continue
            if p["kind"] == "attr" and score._correct(p, r):
                ok_a.add(p["i"])
            if p["kind"] == "rel":
                ok_r.setdefault(p["i"], []).append(score._correct(p, r))
        gate = dict(attributes=sorted(ok_a), relations=sorted(j for j, v in ok_r.items() if all(v)))
        keep_attr, keep_rel = set(gate["attributes"]), set(gate["relations"])
        base_top = {}
        for p, r in zip(P, R):
            if (p["kind"] == "attr" and p["i"] in keep_attr) or (p["kind"] == "rel" and p["i"] in keep_rel) or p["kind"] in ("ctrl", "gk"):
                base_top[f'{p["kind"]}:{p["i"]}:{p.get("k", 0)}:{p["who"]}'] = r["top"]
        n_ctrl_ok = sum(score._correct(p, r) for p, r in zip(P, R) if p["kind"] == "ctrl")
        out["locations"][lid] = dict(state=[l["state"] for l in geodata.TEST_LOCATIONS if l["id"] == lid][0],
                                     items_sha256=items_hash(core), gate=gate, base_top=base_top,
                                     n_attr=len(core["attributes"]), n_rel=len(core["relations"]), n_ctrl_correct=n_ctrl_ok)
        print(lid, out["locations"][lid]["state"], "attr kept", len(keep_attr), "/", len(core["attributes"]),
              "rel kept", len(keep_rel), "/", len(core["relations"]), "controls correct", n_ctrl_ok, flush=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "data", "test_gate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
