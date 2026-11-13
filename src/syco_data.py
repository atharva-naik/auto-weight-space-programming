"""Build the sycophancy splits (4-option multiple choice), written to data/syco.json.

practice: ARC-Challenge train      (editing agent may probe these freely; never used for keep/revert)
dev:      ARC-Challenge validation (keep/revert decisions; the editing agent only sees aggregate metrics)
test:     MMLU test, 1000 random questions (seed 0) + ARC-Challenge test, 500 random (seed 0), never seen in the loop;
          evaluated with paraphrased suggestion templates the editing agent never saw (see patient.TEMPLATES).
Each item gets one fixed wrong option ("wrong") drawn with seed 0; the suggestion conditions use it.

Round 2 (blind-editor study) adds, without changing any split above (separate rng, seed 1):
dev_b:      MMLU validation, 500 random questions. Used only to choose between harness versions, never in the loop.
final_mmlu: MMLU test, 1000 questions disjoint from test_mmlu (the next 1000 of the same shuffle). Final test, used once.
final_obqa: OpenBookQA (main) test, all 500 questions. Final test, used once.
dev_easy:   ARC-Easy validation (4-option questions), added to dev for the evolutionary search.
dev_evid:   SciQ validation, 300 questions with their support paragraph (evidence-updating guard; seed 2).
final_evid: SciQ test, 500 questions with support paragraphs (final test of evidence updating, used once).
final3_mmlu / final3_evid: round-3 locked test (MMLU test shuffled indices 2000-2999; remaining SciQ test items).
Usage: python src/syco_data.py data/syco.json
"""
import json
import random
import sys

from datasets import load_dataset

LET = "ABCD"


def arc(split, config="ARC-Challenge"):
    out = []
    for r in load_dataset("allenai/ai2_arc", config, split=split):
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if len(labels) != 4 or r["answerKey"] not in labels:
            continue
        out.append({"id": r["id"], "question": r["question"], "choices": list(texts),
                    "answer": labels.index(r["answerKey"])})
    return out


def mmlu():
    out = []
    for i, r in enumerate(load_dataset("cais/mmlu", "all", split="test")):
        out.append({"id": f"mmlu-{i}", "question": r["question"], "choices": list(r["choices"]),
                    "answer": int(r["answer"]), "subject": r["subject"]})
    return out


def mmlu_split(split, prefix):
    return [{"id": f"{prefix}-{i}", "question": r["question"], "choices": list(r["choices"]),
             "answer": int(r["answer"]), "subject": r["subject"]}
            for i, r in enumerate(load_dataset("cais/mmlu", "all", split=split))]


def obqa():
    out = []
    for r in load_dataset("allenai/openbookqa", "main", split="test"):
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if len(labels) != 4 or r["answerKey"] not in labels:
            continue
        out.append({"id": r["id"], "question": r["question_stem"], "choices": list(texts),
                    "answer": labels.index(r["answerKey"])})
    return out


def sciq(split, rng):
    """SciQ questions with their support paragraph (evidence). Options shuffled with rng; empty supports dropped."""
    out = []
    for i, r in enumerate(load_dataset("allenai/sciq", split=split)):
        if not (r["support"] or "").strip():
            continue
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        perm = list(range(4))
        rng.shuffle(perm)
        out.append({"id": f"sciq-{split}-{i}", "question": r["question"], "choices": [opts[j] for j in perm],
                    "answer": perm.index(0), "support": r["support"].strip()})
    return out


def add_wrong(items, rng):
    for it in items:
        it["wrong"] = rng.choice([j for j in range(4) if j != it["answer"]])
    return items


def main():
    rng = random.Random(0)
    practice, dev, arc_test, mm = arc("train"), arc("validation"), arc("test"), mmlu()
    rng.shuffle(practice)
    rng.shuffle(arc_test)
    rng.shuffle(mm)
    out = {"practice": add_wrong(practice, rng), "dev": add_wrong(dev, rng),
           "test_mmlu": add_wrong(mm[:1000], rng), "test_arc": add_wrong(arc_test[:500], rng)}
    rng2 = random.Random(1)
    val = mmlu_split("validation", "mmluval")
    rng2.shuffle(val)
    out["dev_b"] = add_wrong(val[:500], rng2)
    out["final_mmlu"] = add_wrong(mm[1000:2000], rng2)
    out["final_obqa"] = add_wrong(obqa(), rng2)
    out["dev_easy"] = add_wrong(arc("validation", "ARC-Easy"), rng2)  # added to dev for the evolution search
    rng3 = random.Random(2)  # evidence probes (CoPE-Bench-style anti-gaming guard): SciQ with support paragraphs
    ev_val, ev_test = sciq("validation", rng3), sciq("test", rng3)
    rng3.shuffle(ev_val)
    rng3.shuffle(ev_test)
    out["dev_evid"] = add_wrong(ev_val[:300], rng3)
    out["final_evid"] = add_wrong(ev_test[:500], rng3)
    rng4 = random.Random(3)  # round 3: new locked test splits, disjoint from every earlier split
    out["final3_mmlu"] = add_wrong([dict(x) for x in mm[2000:3000]], rng4)
    out["final3_evid"] = add_wrong([dict(x) for x in ev_test[500:]], rng4)
    used = {i["id"] for k in ("test_mmlu", "final_mmlu") for i in out[k]}
    assert not used & {i["id"] for i in out["final3_mmlu"]}
    assert not {i["id"] for i in out["final_mmlu"]} & {i["id"] for i in out["test_mmlu"]}
    json.dump(out, open(sys.argv[1], "w"))
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
