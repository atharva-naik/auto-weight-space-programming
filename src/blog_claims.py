"""Blog Part 1 as RQ -> Claim -> Evidence, with every number recomputed from the run files where possible.

  python src/blog_claims.py build     # (re)compute evidence into docs/blog/claims.json, keeping your edits and notes
  python src/blog_claims.py render    # write docs/blog/minimal.html (facts only) from claims.json
  python src/blog_claims.py serve     # editor at http://localhost:8008 : edit text, add notes, tick audited, save, render

Evidence status: "ok" = recomputed from per-item files and equal to the stated value at its precision; "log" = the
source is a logged value (the cited file and line); "MISMATCH" / "MISSING" = needs attention.
"""
import argparse
import hashlib
import html
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

R4 = "runs/round4"
OUT = "docs/blog/claims.json"
GH = "https://github.com/atharva-naik/auto-weight-space-programming/blob/blog-part1/"


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def acc(run, f, key="correct_pmi"):
    e = _load(f"{R4}/{run}/{f}")
    return None if e is None else 100 * float(np.mean([x[key] for x in e]))


def count(run, f):
    e = _load(f"{R4}/{run}/{f}")
    return None if e is None else float(sum(x["correct_pmi"] for x in e))


def ov(run, who, what):
    e = _load(f"{R4}/{run}/eval_stage0_openvocab_test_neuron_{who}.json")
    if e is None:
        return None
    if what == "top1":
        return 100 * float(np.mean([x["top8_raw"][0] == x["true"][0] for x in e]))
    if what == "top1_in8":
        return 100 * float(np.mean([x["top8_raw"][0] in x["true"] for x in e]))
    return 100 * float(np.mean([x["recall8_raw"] for x in e]))


def val_at(run, step):
    p = f"{R4}/{run}/log.jsonl"
    if not os.path.exists(p):
        return None
    for l in open(p):
        x = json.loads(l)
        if x.get("event") == "val_mid" and x.get("step") == step:
            return 100 * x["correct_pmi"]
    return None


def rc(key, *sub):
    """A value from runs/round4/review_checks.json (written by src/review_checks.py from the primary files)."""
    v = _load(f"{R4}/review_checks.json")
    for k in (key,) + sub:
        v = None if v is None else v.get(k) if isinstance(v, dict) else v[k]
    return v


def pct(x):
    return None if x is None else 100 * x


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def E(label, stated, value=None, source="", log=None, digits=1):
    """One evidence item. value: a number recomputed from files, or None with log='file:line note'."""
    return {"label": label, "stated": stated, "recomputed": None if value is None else round(float(value), digits + 1),
            "digits": digits, "source": source, "log": log}


T = "eval_stage0_test_neuron_normal.json"


def spec():
    mp = _load(f"{R4}/mech_patch_4B.json") or {"interchange_picks_source": {}}
    ip = mp["interchange_picks_source"]
    ta = _load(f"{R4}/task_act_4B/test_normal.json")
    circ = [json.loads(l) for l in open(f"{R4}/task_circuit_4B/log.jsonl") if '"val_mid"' in l] \
        if os.path.exists(f"{R4}/task_circuit_4B/log.jsonl") else []
    src = lambda run, f=T: f"{R4}/{run}/{f}"
    return [
        {"id": "rq1", "group": "works", "rq": "Can supervised fine-tuning teach Qwen3.5-4B to read the logit lens of its own neurons from raw numbers?",
         "answer": "Yes, for 128-coordinate slices.", "claims": [
             {"id": "c1.1", "text": "On 1,000 held-out neurons, the reader picks the true token list out of four 99.8% of the time (chance 25%).",
              "code": "src/raw_reader.py", "evidence": [
                  E("test accuracy (%)", 99.8, acc("sweep_4B", T), src("sweep_4B")),
                  E("items correct (of 1,000)", 998, count("sweep_4B", T), src("sweep_4B"), digits=0),
                 E("plain log-likelihood score, no correction (%)", 85.3, rc("main_test_raw_pct"), "runs/round4/review_checks.json")]},
             {"id": "c1.2", "text": "It holds on four layers never used in training: 99.8%.", "code": "src/raw_splits.py", "evidence": [
                 E("held-out layers (%)", 99.8, acc("sweep_4B", "eval_stage0_test_layer_normal.json"), src("sweep_4B", "eval_stage0_test_layer_normal.json"))]},
             {"id": "c1.3", "text": "It holds when the three wrong lists come from the most similar slices in the same layer: 97.1%.",
              "code": "src/raw_reader.py", "evidence": [
                  E("hard distractors (%)", 97.1, acc("sweep_4B", "eval_stage0_test_neuron_hard.json"), src("sweep_4B", "eval_stage0_test_neuron_hard.json")),
                 E("errors (of 1,000), look-alike distractors", 29, rc("hard_test_errors"), "runs/round4/review_checks.json", digits=0),
                 E("errors (of 1,000), random distractors", 2, rc("main_test_errors"), "runs/round4/review_checks.json", digits=0),
                 E("plain score, look-alike distractors (%)", 67.4, rc("hard_test_raw_pct"), "runs/round4/review_checks.json"),
                 E("look-alike distractors: mean cosine with the true slice", 0.32, rc("hard_distractor_cos_mean"), "runs/round4/review_checks.json", digits=2),
                 E("random distractors: mean |cosine|", 0.09, rc("random_distractor_abs_cos_mean"), "runs/round4/review_checks.json", digits=2)]},
             {"id": "c1.4", "text": "Unscaled weights written with 3 significant digits work as well: 99.7%.", "code": "src/raw_reader.py", "evidence": [
                 E("raw floats, test (%)", 99.7, acc("sweep_4B_f3", T), src("sweep_4B_f3")),
                 E("tokens per slice, integers", 329.3, rc("tokens_per_slice_int"), "runs/round4/review_checks.json"),
                 E("tokens per slice, raw floats", 1025.5, rc("tokens_per_slice_f3"), "runs/round4/review_checks.json", digits=0),
                 E("raw floats from scratch, 640–767 (%)", 63.5, acc("sweep_4B_f3_s640", "eval_stage0_test_neuron_normal.json"), "runs/round4/sweep_4B_f3_s640/eval_stage0_test_neuron_normal.json"),
                 E("raw floats from scratch, 1280–1407 (%)", 28.2, acc("sweep_4B_f3_s1280", "eval_stage0_test_neuron_normal.json"), "runs/round4/sweep_4B_f3_s1280/eval_stage0_test_neuron_normal.json")]}]},
        {"id": "rq2", "group": "works", "rq": "Does the reader use the numbers, and where does it read them?",
         "answer": "Yes; it rebuilds the shown slice in its residual stream.", "claims": [
             {"id": "c2.1", "text": "Shuffling the same numbers drops accuracy to 26.3%; hiding them gives 26.9%.", "code": "src/raw_reader.py", "evidence": [
                 E("shuffled (%)", 26.3, acc("sweep_4B", "eval_stage0_test_neuron_scrambled.json"), src("sweep_4B", "eval_stage0_test_neuron_scrambled.json")),
                 E("numbers hidden (%)", 26.9, acc("sweep_4B", T, "correct_hidden"), src("sweep_4B") + " (correct_hidden)")]},
             {"id": "c2.2", "text": "In the final layer, the true slice is closer to the reader's residual stream than all three distractors on 200 of 200 items.",
              "code": "src/mech_probe.py", "evidence": [
                 E("last block: true slice closest (%)", 100, rc("mech_closest_pct_by_block", "31"), "runs/round4/mech_4B_final.json (via review_checks.json)", digits=0),
                 E("last block: cosine", 0.54, rc("mech_cos_by_block", "31"), "runs/round4/mech_4B_final.json (via review_checks.json)", digits=2),
                 E("block 30: cosine", 0.21, rc("mech_cos_by_block", "30"), "runs/round4/mech_4B_final.json (via review_checks.json)", digits=2),
                 E("blocks 16–30: true slice closest, lowest (%)", 34, rc("mech_blocks16_30_closest_range")[0] if rc("mech_blocks16_30_closest_range") else None, "runs/round4/mech_4B_final.json (via review_checks.json)", digits=0),
                 E("blocks 16–30: true slice closest, highest (%)", 59, rc("mech_blocks16_30_closest_range")[1] if rc("mech_blocks16_30_closest_range") else None, "runs/round4/mech_4B_final.json (via review_checks.json)", digits=0),
                 E("untrained model: true slice closest (%)", 23, rc("mech_base_closest_pct"), "runs/round4/mech_4B_final.json (via review_checks.json)", digits=0)]},
             {"id": "c2.3", "text": "Copying coordinates 0–127 of example B's residual stream into example A moves the answer to B's label on 91 of 100 pairs, against 5 without the copy and 6 when another block is copied.",
              "code": "src/mech_patch.py", "evidence": [
                  E("patched (of 100)", 91, 100 * ip["patch_S"] if ip else None, f"{R4}/mech_patch_4B.json", digits=0),
                  E("no patch (of 100)", 5, 100 * ip["no_patch"] if ip else None, f"{R4}/mech_patch_4B.json", digits=0),
                  E("other block (of 100)", 6, 100 * ip["patch_C"] if ip else None, f"{R4}/mech_patch_4B.json", digits=0),
                 E("zeroing 0–127: damage (nats per token)", 4.81, rc("zeroing_damage_nats", "coords_0_127"), "runs/round4/mech_patch_4B.json", digits=2),
                 E("zeroing 1280–1407: damage (nats per token)", 0.04, rc("zeroing_damage_nats", "coords_1280_1407"), "runs/round4/mech_patch_4B.json", digits=2)]}]},
        {"id": "rq3", "group": "works", "rq": "Does the skill carry over to other parts of a neuron?", "answer": "Yes, from a warm start.", "claims": [
            {"id": "c3.1", "text": "Started from the 0–127 reader, three new slices reach 97–99% validation within 16,000 examples and 1,000 of 1,000 on the test.",
             "code": "src/raw_reader.py", "evidence": [
                 E("1280–1407: validation at 16k (%)", 99, val_at("sweep_4B_s1280_transfer", 1000), f"{R4}/sweep_4B_s1280_transfer/log.jsonl (val_mid step 1000, n=100)", digits=0),
                 E("640–767: validation at 16k (%)", 99, val_at("r3_sweep_4B_s640_transfer", 1000), f"{R4}/r3_sweep_4B_s640_transfer/log.jsonl", digits=0),
                 E("2048–2175: validation at 16k (%)", 97, val_at("r3_sweep_4B_s2048_transfer", 1000), f"{R4}/r3_sweep_4B_s2048_transfer/log.jsonl", digits=0),
                 E("1280–1407: test (%)", 100.0, acc("sweep_4B_s1280_transfer", T), src("sweep_4B_s1280_transfer")),
                 E("640–767: test (%)", 100.0, acc("r3_sweep_4B_s640_transfer", T), src("r3_sweep_4B_s640_transfer")),
                 E("2048–2175: test (%)", 100.0, acc("r3_sweep_4B_s2048_transfer", T), src("r3_sweep_4B_s2048_transfer")),
                 E("1280–1407: shuffled (%)", 23.6, acc("sweep_4B_s1280_transfer", "eval_stage0_test_neuron_scrambled.json"), "runs/round4/sweep_4B_s1280_transfer/eval_stage0_test_neuron_scrambled.json"),
                 E("640–767: shuffled (%)", 25.8, acc("r3_sweep_4B_s640_transfer", "eval_stage0_test_neuron_scrambled.json"), "runs/round4/r3_sweep_4B_s640_transfer/eval_stage0_test_neuron_scrambled.json"),
                 E("2048–2175: shuffled (%)", 26.5, acc("r3_sweep_4B_s2048_transfer", "eval_stage0_test_neuron_scrambled.json"), "runs/round4/r3_sweep_4B_s2048_transfer/eval_stage0_test_neuron_scrambled.json"),
                 E("main run, 0–127: validation at 16k (%)", 37, val_at("sweep_4B", 1000), "runs/round4/sweep_4B/log.jsonl", digits=0)]},
            {"id": "c3.2", "text": "From scratch, coordinates 1280–1407 score 29.3% after 64,000 examples and 100% after 192,000.",
             "code": "src/raw_reader.py", "evidence": [
                 E("64k from scratch (%)", 29.3, acc("sweep_4B_s1280", T), src("sweep_4B_s1280")),
                 E("192k from scratch (%)", 100.0, acc("r3_sweep_4B_s1280_192k", T), src("r3_sweep_4B_s1280_192k")),
                 E("192k: validation at 48k (%)", 32, val_at("r3_sweep_4B_s1280_192k", 3000), "runs/round4/r3_sweep_4B_s1280_192k/log.jsonl", digits=0),
                 E("192k: validation at 76k (%)", 97, val_at("r3_sweep_4B_s1280_192k", 4750), "runs/round4/r3_sweep_4B_s1280_192k/log.jsonl", digits=0)]},
            {"id": "c3.3", "text": "From scratch, coordinates 128–255 reach 98.9%.", "code": "src/raw_reader.py", "evidence": [
                 E("128–255 from scratch (%)", 98.9, acc("sweep_4B_s128", "eval_stage0_test_neuron_normal.json"), "runs/round4/sweep_4B_s128/eval_stage0_test_neuron_normal.json"),
                 E("validation at 8k (%)", 42, val_at("sweep_4B_s128", 500), "runs/round4/sweep_4B_s128/log.jsonl", digits=0),
                 E("validation at 32k (%)", 47, val_at("sweep_4B_s128", 2000), "runs/round4/sweep_4B_s128/log.jsonl", digits=0),
                 E("validation at 48k (%)", 90, val_at("sweep_4B_s128", 3000), "runs/round4/sweep_4B_s128/log.jsonl", digits=0),
                 E("validation at 52k (%)", 98, val_at("sweep_4B_s128", 3250), "runs/round4/sweep_4B_s128/log.jsonl", digits=0),
                 E("main run, 0–127: validation at 8k (%)", 28, val_at("sweep_4B", 500), "runs/round4/sweep_4B/log.jsonl", digits=0),
                 E("look-alike distractors (%)", 92.3, acc("sweep_4B_s128", "eval_stage0_test_neuron_hard.json"), "runs/round4/sweep_4B_s128/eval_stage0_test_neuron_hard.json")]},
            {"id": "c3.4", "text": "The read-out matrix of each slice (the unembedding restricted to its coordinates, scaled by the final norm) has effective rank 91 (0–127), 100 (128–255) and about 120 on most later slices, out of 128.",
             "code": "src/slice_hardness.py", "evidence": [
                 E("effective rank, 0–127", 91, rc("eff_rank", "0"), "runs/round4/slice_hardness.json (via review_checks.json)", digits=0),
                 E("effective rank, 128–255", 100, rc("eff_rank", "128"), "runs/round4/slice_hardness.json (via review_checks.json)", digits=0),
                 E("effective rank, 640–767", 119.5, rc("eff_rank", "640"), "runs/round4/slice_hardness.json (via review_checks.json)"),
                 E("effective rank, 1280–1407", 119.5, rc("eff_rank", "1280"), "runs/round4/slice_hardness.json (via review_checks.json)"),
                 E("raw floats from scratch, 640–767 (%)", 63.5, acc("sweep_4B_f3_s640", "eval_stage0_test_neuron_normal.json"), "runs/round4/sweep_4B_f3_s640/eval_stage0_test_neuron_normal.json"),
                 E("raw floats from scratch, 1280–1407 (%)", 28.2, acc("sweep_4B_f3_s1280", "eval_stage0_test_neuron_normal.json"), "runs/round4/sweep_4B_f3_s1280/eval_stage0_test_neuron_normal.json"),
                 E("rounding keeps (share of true top tokens), 0–127 (%)", 96.6, pct(next((x["rounding"] for x in (_load(f"{R4}/slice_hardness.json") or []) if x["start"] == 0), None)), "runs/round4/slice_hardness.json"),
                 E("rounding keeps (share of true top tokens), 1280–1407 (%)", 97.2, pct(next((x["rounding"] for x in (_load(f"{R4}/slice_hardness.json") or []) if x["start"] == 1280), None)), "runs/round4/slice_hardness.json"),
                 E("mean |integer|, 0–127 (9 layers)", 12.7, rc("mean_abs_int_by_slice_9layers", "0"), "runs/round4/review_checks.json"),
                 E("mean |integer|, 1280–1407 (9 layers)", 16.5, rc("mean_abs_int_by_slice_9layers", "1280"), "runs/round4/review_checks.json")]}]},
        {"id": "rq4", "group": "works", "rq": "How does the skill appear during training?", "answer": "Suddenly, between 16,000 and 48,000 examples.", "claims": [
            {"id": "c4.1", "text": "Separate runs with budgets of 16k, 24k, 32k, 48k, 64k and 96k examples score 27.8%, 33.0%, 68.8%, 96.5%, 99.5% and 100% on validation.",
             "code": "src/raw_reader.py", "evidence": [
                 E("16k (%)", 27.8, None, "ROUND4_LOG.md", log="ROUND4_LOG.md budget series (no per-item file kept)"),
                 E("24k (%)", 33.0, acc("scale_4B_24k", "eval_stage0_val128_normal.json"), src("scale_4B_24k", "eval_stage0_val128_normal.json")),
                 E("32k (%)", 68.8, acc("scale_4B_32k", "eval_stage0_val128_normal.json"), src("scale_4B_32k", "eval_stage0_val128_normal.json")),
                 E("48k (%)", 96.5, acc("scale_4B_48k", "eval_stage0_val128_normal.json"), src("scale_4B_48k", "eval_stage0_val128_normal.json")),
                 E("64k (%)", 99.5, acc("sweep_4B", "eval_stage0_val128_normal.json"), src("sweep_4B", "eval_stage0_val128_normal.json")),
                 E("96k (%)", 100.0, acc("r3_scale_4B_96k", "eval_stage0_val128_normal.json"), src("r3_scale_4B_96k", "eval_stage0_val128_normal.json")),
                 E("main run: validation at 12k (%)", 30, val_at("sweep_4B", 750), "runs/round4/sweep_4B/log.jsonl", digits=0),
                 E("main run: validation at 16k (%)", 37, val_at("sweep_4B", 1000), "runs/round4/sweep_4B/log.jsonl", digits=0),
                 E("main run: validation at 40k (%)", 98, val_at("sweep_4B", 2500), "runs/round4/sweep_4B/log.jsonl", digits=0),
                 E("main run: validation at 24k (%)", 52, val_at("sweep_4B", 1500), "runs/round4/sweep_4B/log.jsonl", digits=0)]},
            {"id": "c4.2", "text": 'After 32,000 examples, the main run is at 84% on held-out neurons and the pool run is at 25% (chance). The pool run never went above 36%, and we stopped it at about 2,600 steps, when each pool neuron had been seen about 40 times.',
             "code": "src/raw_reader.py", "evidence": [
                 E("original run, 32k examples (%)", 84, None, "ROUND4_LOG.md", log="ROUND4_LOG.md:24 diag_G1_fresh64k, step 2000 (n = 100)", digits=0),
                 E("pool run, 32k examples (%)", 25, None, "ROUND4_LOG.md", log="ROUND4_LOG.md:25 diag_G2_pool1024, step 2000", digits=0),
                 E("pool run, best check (%)", 36, None, "ROUND4_LOG.md", log="ROUND4_LOG.md:25 diag_G2_pool1024, step 1000", digits=0),
                 E("pool run, views per neuron when stopped", "about 40", None, "ROUND4_LOG.md", log="ROUND4_LOG.md:25 stopped at step ~2,600; 2,600 × 16 / 1,024"),
                 E("64-neuron memorization, 200 steps (%)", 83, None, "ROUND4_LOG.md", log="ROUND4_LOG.md:20 diag_mem64_same, step 200", digits=0),
                 E("main run, 32k examples (%)", 88, val_at("sweep_4B", 2000), "runs/round4/sweep_4B/log.jsonl", digits=0)]}]},
        {"id": "rq5", "group": "works", "rq": "Does it carry over across models and sizes?", "answer": "Partly.", "claims": [
            {"id": "c5.1", "text": "Trained on their own weights with the same recipe, Qwen3.5-0.8B, 2B, 4B and 9B score 35.8%, 62.4%, 99.8% and 31.6%; Llama-3.2-3B scores 52.9%.",
             "code": "src/raw_reader.py", "evidence": [
                 E("0.8B (%)", 35.8, acc("sweep_0.8B", T), src("sweep_0.8B")), E("2B (%)", 62.4, acc("sweep_2B", T), src("sweep_2B")),
                 E("4B (%)", 99.8, acc("sweep_4B", T), src("sweep_4B")), E("9B (%)", 31.6, acc("sweep_9B", T), src("sweep_9B")),
                 E("Llama-3.2-3B (%)", 52.9, acc("sweep_Llama-3.2-3B", T), src("sweep_Llama-3.2-3B")),
                 E("9B, shuffled (%)", 23.7, rc("9B_shuffled_pct"), "runs/round4/sweep_9B/eval_stage0_test_neuron_scrambled.json"),
                 E("0.8B: last validation check (%)", 39, pct(rc("sweep_0.8B_last_val", "64000")), "runs/round4/sweep_0.8B/log.jsonl", digits=0),
                 E("9B: last validation check (%)", 39, pct(rc("sweep_9B_last_val", "64000")), "runs/round4/sweep_9B/log.jsonl", digits=0)]},
            {"id": "c5.2", "text": "With learning rate 5e-5 and 128,000 examples, 9B reaches 54.5%.", "code": "src/raw_reader.py", "evidence": [
                E("9B, lr 5e-5, 128k (%)", 54.5, acc("sweep_9B_lr5e-5_128k", T), src("sweep_9B_lr5e-5_128k")),
                 E("last validation check (%)", 66, pct(rc("sweep_9B_lr5e-5_128k_last_val", "128000")), "runs/round4/sweep_9B_lr5e-5_128k/log.jsonl", digits=0)]},
            {"id": "c5.3", "text": "The 4B reader reads 0.8B's weights at 98.8%; the 0.8B reader reads 4B's weights at 62.0%.",
             "code": "src/raw_reader.py", "evidence": [
                 E("4B → 0.8B (%)", 98.8, acc("cross_4Breads0.8B", T), src("cross_4Breads0.8B")),
                 E("0.8B → 4B (%)", 62.0, acc("cross_0.8Breads4B", T), src("cross_0.8Breads4B")),
                 E("4B reads 0.8B, look-alike distractors (%)", 94.2, acc("cross_4Breads0.8B", "eval_stage0_test_neuron_hard.json"), "runs/round4/cross_4Breads0.8B/eval_stage0_test_neuron_hard.json"),
                 E("4B reads itself, look-alike distractors (%)", 97.1, acc("sweep_4B", "eval_stage0_test_neuron_hard.json"), "runs/round4/sweep_4B/eval_stage0_test_neuron_hard.json")]}]},
        {"id": "rq6", "group": "works", "rq": "Can the reader name the tokens without candidates?", "answer": "Rarely.", "claims": [
            {"id": "c6.1", "text": "Its single top guess is the neuron's exact top token for 6.6% of test neurons and one of its top 8 tokens for 18.8%; untrained, 0.0% and 0.1%.",
             "code": "src/raw_reader.py", "evidence": [
                 E("trained: exact #1 (%)", 6.6, ov("sweep_4B", "trained", "top1"), src("sweep_4B", "eval_stage0_openvocab_test_neuron_trained.json")),
                 E("trained: in top 8 (%)", 18.8, ov("sweep_4B", "trained", "top1_in8"), src("sweep_4B", "eval_stage0_openvocab_test_neuron_trained.json")),
                 E("untrained: exact #1 (%)", 0.0, ov("sweep_4B", "base", "top1"), src("sweep_4B", "eval_stage0_openvocab_test_neuron_base.json")),
                 E("untrained: in top 8 (%)", 0.1, ov("sweep_4B", "base", "top1_in8"), src("sweep_4B", "eval_stage0_openvocab_test_neuron_base.json")),
                 E("always guess \" (\": exact #1 (%)", 2.0, rc("prior_exact_pct"), "runs/round4/sweep_4B/eval_stage0_openvocab_test_neuron_trained.json (via review_checks.json)"),
                 E("always guess \" (\": in top 8 (%)", 5.8, rc("prior_in_top8_pct"), "runs/round4/sweep_4B/eval_stage0_openvocab_test_neuron_trained.json (via review_checks.json)"),
                 E("reader guesses \" (\" (of 1,000)", 129, (rc("reader_most_common_guess") or [0, None])[1], "runs/round4/sweep_4B/eval_stage0_openvocab_test_neuron_trained.json (via review_checks.json)", digits=0),
                 E("untrained model guesses \"the\" (of 1,000)", 997, (rc("base_most_common_guess") or [0, None])[1], "runs/round4/sweep_4B/eval_stage0_openvocab_test_neuron_base.json", digits=0)]},
            {"id": "c6.2", "text": "Over all eight slots, 11.2% of the true top-8 tokens appear in the reader's top 8 (0.7% untrained).",
             "code": "src/raw_reader.py", "evidence": [
                 E("trained recall@8 (%)", 11.2, ov("sweep_4B", "trained", "recall"), src("sweep_4B", "eval_stage0_openvocab_test_neuron_trained.json")),
                 E("untrained recall@8 (%)", 0.7, ov("sweep_4B", "base", "recall"), src("sweep_4B", "eval_stage0_openvocab_test_neuron_base.json")),
                 E("fixed list of the 8 most common tokens: recall@8 (%)", 3.6, rc("prior_list_recall8_pct"), "runs/round4/sweep_4B/eval_stage0_openvocab_test_neuron_trained.json (via review_checks.json)")]}]},
        {"id": "rq8", "group": "effort", "rq": "Can the model write weights, or read effects that need a forward pass?", "answer": "Not yet, with the setup we tried.", "claims": [
            {"id": "c8.1", "limits": "", "text": "Asked to write a slice that promotes 8 given tokens, the writer recovers 1.0–1.4% of them; an analytic writer recovers 27.7% and the true neuron's rounded numbers 96.1%.",
             "code": "src/raw_writer.py", "evidence": [
                 E("writer from scratch, 56k: recall@8 up (%)", 1.0, None, "runs/round4/write_4B_scratch/log.jsonl", log="last logged check (64 val neurons)"),
                 E("writer from reader, 32k: recall@8 up (%)", 1.4, None, "runs/round4/write_4B_fromreader/log.jsonl", log="last logged check"),
                 E("analytic writer (%)", 27.7, None, "runs/round4/write_4B_scratch/log.jsonl", log="same check"),
                 E("true rounded numbers (%)", 96.1, None, "runs/round4/write_4B_scratch/log.jsonl", log="same check"),
                 E("from scratch: complete answers (%)", 33, pct(rc("write_4B_scratch_last", "parse_ok")), "runs/round4/write_4B_scratch/log.jsonl", digits=0),
                 E("from the reader: complete answers (%)", 88, pct(rc("write_4B_fromreader_last", "parse_ok")), "runs/round4/write_4B_fromreader/log.jsonl", digits=0),
                 E("random numbers: recall@8 up (%)", 0.2, pct(rc("write_4B_scratch_last", "random_recall_up")), "runs/round4/write_4B_scratch/log.jsonl")]},
            {"id": "c8.2", "limits": "", "text": "Given a neuron's input weights and two sentences, the reader picks the one that activates it more on 50.1% of 1,000 items (chance 50%).",
             "code": "src/task_reader.py", "evidence": [
                 E("activation task (%)", 50.1, rc("act_test_normal_pct"), "runs/round4/task_act_4B/test_normal.json"),
                 E("shuffled (%)", 48.9, rc("act_test_shuffled_pct"), "runs/round4/task_act_4B/test_shuffled.json"),
                 E("numbers hidden (%)", 48.9, rc("act_test_hidden_pct"), "runs/round4/task_act_4B/test_hidden.json"),
                 E("final validation, 400 items (%)", 51.5, None, "runs/round4/task_act_4B/log.jsonl", log="runs/round4/task_act_4B/log.jsonl event val, acc 0.515, n = 400"),
                 E("always the more common answer, validation (%)", 55, pct(rc("act_val_majority", "val")), "runs/round4/task_act_4B/log.jsonl", digits=0)]},
            {"id": "c8.3", "limits": "", "text": "Matching a neuron to the later neuron it feeds most reached 27% after 16,000 examples (chance 25%), when we stopped it.",
             "code": "src/task_reader.py", "evidence": [
                 E("circuit task, last check (%)", 27, 100 * circ[-1]["acc"] if circ else None, f"{R4}/task_circuit_4B/log.jsonl (val_mid, n=100)", digits=0),
                 E("untrained model, before training (%)", 31, pct(rc("circuit_val", "0")), "runs/round4/task_circuit_4B/log.jsonl", digits=0)]},
            {"id": "c8.4", "limits": "", "text": "Across 802 neurons whose clamping changes sycophancy, a linear probe on their weights predicts none of the effect (R² near zero).",
             "code": "src/levers_pilot.py", "evidence": [
                 E("probe R² within layers (lab log only)", "≈0", None, "ROUND4_LOG.md", log="ROUND4_LOG.md:368; no result file was kept"),
                 E("split-half agreement, pilot A (Spearman)", 0.70, None, "runs/round4/levers_pilot.json", log="runs/round4/levers_pilot.json summary.clamp/syco.split_half_spearman = 0.699", digits=2),
                 E("split-half agreement, pilot B (Spearman)", 0.52, None, "runs/round4/levers_pilot_b.json", log="runs/round4/levers_pilot_b.json summary.clamp/syco.split_half_spearman = 0.523", digits=2),
                 E("fast estimate vs measured clamp, all 802 pilot neurons (Spearman)", 0.77, rc("atp_vs_clamp_spearman_802"), "runs/round4/levers_atp.npz + levers_pilot*.json (via review_checks.json)", digits=2),
                 E("fast estimate, split-half over all neurons (Spearman)", 0.79, rc("atp_x0_split_half_all"), "runs/round4/levers_atp.npz + levers_pilot*.json (via review_checks.json)", digits=2),
                 E("probe R², held-out layers (full column)", -0.01, None, "runs/round4/levers_groups.json", log="runs/round4/levers_groups.json probe_heldout_layers.full_column_lam100.r2 = -0.0095", digits=2)]}]},
    ]


import re as _svgre
_TEXT = _svgre.compile(r"(<text\b[^>]*>)(.*?)(</text>)", _svgre.S)


def svg_texts(path):
    """Editable text of every <text> element, with bold tspans as **x**."""
    out = []
    for m in _TEXT.finditer(open(path).read()):
        inner = _svgre.sub(r"<tspan[^>]*font-weight=\"700\"[^>]*>(.*?)</tspan>", r"**\1**", m.group(2))
        inner = _svgre.sub(r"<[^>]+>", "", inner)
        out.append(html.unescape(inner))
    return out


def svg_set_texts(path, texts):
    s = open(path).read()
    ms = list(_TEXT.finditer(s))
    assert len(ms) == len(texts), (len(ms), len(texts))
    parts, last = [], 0
    for m, t in zip(ms, texts):
        inner = html.escape(t, quote=False)
        inner = _svgre.sub(r"\*\*(.+?)\*\*", r'<tspan font-weight="700">\1</tspan>', inner)
        parts += [s[last:m.start()], m.group(1), inner, m.group(3)]
        last = m.end()
    parts.append(s[last:])
    open(path, "w").write("".join(parts))


def status(ev):
    if ev["log"]:
        return "log"
    if ev["recomputed"] is None:
        return "MISSING"
    return "ok" if abs(round(ev["recomputed"], ev["digits"]) - float(ev["stated"])) < 10 ** -ev["digits"] / 2 + 1e-9 else "MISMATCH"


def build(a):
    old = _load(OUT) or {"rqs": []}
    keep = {}
    for rq in old["rqs"]:
        keep[rq["id"]] = {k: rq.get(k) for k in ("rq", "answer", "note")}
        for c in rq["claims"]:
            keep[c["id"]] = {k: c.get(k) for k in ("text", "note", "audited", "limits", "title", "experiment", "plots",
                                                   "examples_html", "caveats")}
    rqs = spec()
    for rq in rqs:
        for k, v in (keep.get(rq["id"]) or {}).items():
            if v is not None:
                rq[k] = v
        rq.setdefault("note", "")
        for c in rq["claims"]:
            for k, v in (keep.get(c["id"]) or {}).items():
                if v is not None:
                    c[k] = v
            c.setdefault("note", "")
            c.setdefault("audited", False)
            for ev in c["evidence"]:
                ev["status"] = status(ev)
    doc = {"title": old.get("title") or "Auto Weight-Space Programming, Part 1: Training Qwen3.5-4B to Read Its Own Weights",
           "series_heading": old.get("series_heading") or "About this series",
           "references": old.get("references") or "",
           "representation_heading": old.get("representation_heading") or "Why the weights are plain text",
           **{k: old[k] for k in ("goal_heading", "goal", "analogy_heading", "analogy_intro", "analogy", "analogy_caveats")
              if k in old},
           **{k: old[k] for k in ("definitions_plain", "goal_plain", "speculation_heading", "speculation",
                                  "definitions_parts", "goal_parts", "_legacy", "figures", "experiment_heading",
                                  "setup_figures", "ai_use_heading", "ai_use", "tldr", "future_work_heading", "future_work") if k in old},
           "representation": old.get("representation") or "",
           "series": (old.get("series") or
                      "This is the first post in a series on training small models, such as Qwen3.5-4B, to do "
                      "weight-space programming on themselves. Each post asks research questions nobody has answered "
                      "yet and reports results, qualitative analysis and my speculation. Each post goes out only after "
                      "its datasets, models and code are public. Planned topics: recipes for supervised fine-tuning and "
                      "on-policy self-distillation on basic interpretability tasks, such as the logit lens over raw "
                      "neuron weights (this post); a benchmark suite for weight-space programming (next post); "
                      "reinforcement-learning environments for weight-space programming and their reward hacks; harness "
                      "engineering; and outcome and process reward design. This is a side project next to my PhD and is "
                      "limited by compute; posts come every two to four weeks."),
           "definitions": (old.get("definitions") or
                           "Weight-space programming is reading a network's weights to say what it does, or writing new "
                           "values to change what it does, without an optimizer. Auto weight-space programming is weight-space "
                           "programming where the model doing the reading or writing is the model being read or written. "
                           "This post covers the reading half, done natively: the weights appear as plain numbers in the "
                           "prompt, and the model has no tools."),
           "setup": (old.get("setup") or
                     "Each item shows coordinates 0–127 of one MLP neuron's output direction (its column of mlp.down_proj), "
                     "as integers scaled so that 20 is one standard deviation of the full column. The label is the logit "
                     "lens of that slice: the 8 tokens it promotes and the 8 it suppresses through the final norm and the "
                     "unembedding. In the four-way test, each candidate list is scored by its log-likelihood with the "
                     "numbers minus its log-likelihood with the numbers hidden. Readers are rank-32 LoRAs (learning rate "
                     "1e-4, 16 examples per step); four layers and 10% of the neurons in every other layer are never trained on."),
           "rqs": rqs}
    json.dump(doc, open(OUT, "w"), ensure_ascii=False, indent=1)
    n = [ev["status"] for rq in rqs for c in rq["claims"] for ev in c["evidence"]]
    print(f"{OUT}: {len(rqs)} RQs, {sum(len(rq['claims']) for rq in rqs)} claims, {len(n)} evidence items: "
          f"{n.count('ok')} ok, {n.count('log')} log, {n.count('MISMATCH')} mismatch, {n.count('MISSING')} missing")


# (name, "Last, First" for BibTeX, pending). Pending co-authors are shown in red until they confirm.
AUTHORS = [("Atharva Naik", "Naik, Atharva", False),
           ("Emmy Liu", "Liu, Emmy", False),
           ("Varun Gangal", "Gangal, Varun", False)]


def render(a=None):
    d = _load(OUT)
    e = html.escape

    import re as _re

    # bibliography keys: (first author's surname or handle, year) -> anchor id, so in-text citations can link to it
    refkeys = {}
    for r in (d.get("references") or "").splitlines():
        m = _re.match(r"^\s*(@?[^\s,(]+)[^(]*\((\d{4})[a-z]?\)", r)
        if m:
            refkeys[(m.group(1).lower(), m.group(2))] = "ref-" + _re.sub(r"[^a-z0-9]+", "", m.group(1).lower()) + m.group(2)

    def cite_links(x):
        def link(m):
            key = (m.group("name").lower(), m.group("year"))
            return f"<a class='cite' href='#{refkeys[key]}'>{m.group(0)}</a>" if key in refkeys else m.group(0)
        # "Name et al., 2025" / "Name and Other, 2025" / "Name, 2025" inside parentheses, and "Name et al. (2025)" in running text
        x = _re.sub(r"(?P<name>@?[A-Za-z][\w-]*)(?: et al\.| and [A-Z][\w-]+)?, (?P<year>\d{4})", link, x)
        return _re.sub(r"(?P<name>@?[A-Za-z][\w-]*)(?: et al\.| and [A-Z][\w-]+)? \((?P<year>\d{4})\)", link, x)

    def inline(x):
        x = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", e(x))
        x = _re.sub(r"(?<![\w*])\*(?!\s)([^*$]+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", x)
        return cite_links(x)

    def paras(text, lead=""):
        # fenced code blocks may contain blank lines: protect them before splitting into paragraphs
        blocks = []

        def keep(m):
            blocks.append(m.group(1))
            return f"\n\n@@CODE{len(blocks) - 1}@@\n\n"
        text = _re.sub(r"```\n?(.*?)\n?```", keep, text or "", flags=_re.S)
        ps = [x.strip() for x in text.split("\n\n") if x.strip()]
        out = []
        for i, x in enumerate(ps):
            mc = _re.match(r"^@@CODE(\d+)@@$", x)
            if mc:
                out.append(f"<pre class='prompt'>{e(blocks[int(mc.group(1))])}</pre>")
                continue
            mf = _re.match(r"^\[\[figure:(.+?)\]\]$", x)
            if mf:  # place a setup figure inline
                f = next((f for f in (d.get("setup_figures") or []) + (d.get("figures") or []) if f["src"] == mf.group(1).strip()), None)
                out.append(figure(f) if f else "")
                continue
            lines = x.split("\n")
            if all(l.strip().startswith("- ") for l in lines):  # a bulleted list: every line starts with "- "
                out.append("<ul>" + "".join(f"<li>{inline(l.strip()[2:])}</li>" for l in lines) + "</ul>")
                continue
            if x.lower().startswith("speculation:"):  # woven-in speculation: a highlighted aside
                body = x[len('speculation:'):].strip()
                mt = _re.match(r"^\*\*(.+?)\*\*\s*(.*)$", body, _re.S)  # a bold title replaces the default label
                lab, body = (mt.group(1), mt.group(2)) if mt else ("Speculation.", body)
                out.append(f"<p class='spec'><span class='hl'><span class='speclab'>{inline(lab)}</span> {inline(body)}</span></p>")
            else:
                out.append(f"<p>{lead if i == 0 else ''}{inline(x)}</p>")
        return "".join(out)

    fignum = {}

    def figure(f):
        src = f["src"]
        if src not in fignum:
            fignum[src] = len(fignum) + 1
        cap = _re.sub(r"^\s*\*\*Figure \d+\.\*\*\s*", "", f.get("caption", "") or "")
        return (f"<figure><img src='{e(src)}' alt='{e(f.get('alt', ''))}' loading='lazy'>"
                f"<figcaption><b>Figure {fignum[src]}.</b> {inline(cap)}</figcaption></figure>")

    def figs(where):
        return "".join(figure(f) for f in d.get("figures") or [] if f.get("after") == where)

    def pairs(parts, key=""):  # plain text always shown; formal additions only in the formal view
        out = []
        for i, x in enumerate(parts or []):
            out.append(paras(x.get("plain", "")))
            if (x.get("formal") or "").strip():
                out.append(f"<div class='v-formal formal-add'>{paras(x['formal'])}</div>")
            out.append(figs(f"{key}.{i}"))
        return "".join(out)

    def two(plain, formal):  # both versions on the page; the toggle shows one
        if not plain:
            return paras(formal)
        return f"<div class='v-plain'>{paras(plain)}</div><div class='v-formal'>{paras(formal)}</div>"
    css = open("docs/blog/index.html").read().split("<style>")[1].split("</style>")[0]
    byline = ", ".join(f"<span class='pending'>{e(n)}</span>" if pend else e(n) for n, _, pend in AUTHORS)
    parts = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>{e(d['title'])}</title><style>{css} .ev td:nth-child(2){{text-align:right}} .rqa{{color:var(--ink2)}}</style>"
             "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css'>"
             "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js'></script>"
             "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js' "
             "onload=\"renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]})\"></script>"
             "</head><body><main>",
             "<div class='kicker'>Auto Weight-Space Programming · Part 1</div>", f"<h1>{e(d['title'])}</h1>",
             f"<div class='byline'>{byline} · November 2026 · <a href='https://github.com/atharva-naik/auto-weight-space-programming'>code and logs</a></div>",
             (f"<div class='tldr'><b>TL;DR.</b> {inline(d['tldr'])}</div>" if (d.get("tldr") or "").strip() else ""),
             f"<h2>{e(d.get('series_heading', 'About this series'))}</h2>" + paras(d.get("series", "")),
             "<div class='viewtoggle'><span class='lab' data-v='plain'>Plain</span><label class='switch' title='Switch between the plain and the formal version'>"
             "<input type='checkbox' id='viewswitch'><span class='slider'></span></label><span class='lab' data-v='formal'>Formal</span></div>",
             "<h2>Definitions</h2>" + (pairs(d["definitions_parts"], "definitions_parts") if d.get("definitions_parts") else two(d.get("definitions_plain"), d.get("definitions", ""))),
             (f"<h2>{e(d.get('goal_heading', ''))}</h2>" + (pairs(d["goal_parts"], "goal_parts") if d.get("goal_parts") else two(d.get("goal_plain"), d.get("goal", "")))),
             (f"<details class='analogy'><summary>{e(d.get('analogy_heading', ''))}</summary>" + paras(d.get("analogy_intro", "")) +
              "<table class='compact'>" + "".join(
                  "<tr>" + "".join((f"<th>{inline(c)}</th>" if (i == 0 or j == 0) else f"<td style='text-align:left;white-space:normal'>{inline(c)}</td>")
                                   for j, c in enumerate(row)) + "</tr>" for i, row in enumerate(d.get("analogy") or [])) + "</table>" +
              "<p><b>Where the analogy breaks.</b></p><ol>" + "".join(f"<li>{inline(c)}</li>" for c in
                  (d.get("analogy_caveats") or "").splitlines() if c.strip()) + "</ol></details>") if d.get("analogy") else "",
             (f"<h2>{e(d.get('representation_heading', ''))}</h2>" + paras(d.get("representation", ""))) if d.get("representation") else "",
             f"<h2>{e(d.get('experiment_heading') or 'Experiment')}</h2>",
             "<h3>Experimental setup</h3>" + paras(d["setup"]) + "".join(figure(f) for f in d.get("setup_figures") or []
                                                                       if f"[[figure:{f['src']}]]" not in d["setup"]),
             "<h3>Experiments and observations</h3><p class='note'>Findings are grouped by research question (RQ). Each finding below expands to show the experiment, "
             "the observation, the numbers with their sources, plots, examples and caveats.</p>"]
    heads = {"works": ("What works", "Each finding below is measured on held-out data, with its controls."),
             "effort": ("Needs more effort", "These did not work with the setup we tried. The setup limits under each "
                        "claim list what we actually did; a different setup may succeed.")}
    shown = None
    for i, rq in enumerate(d["rqs"], 1):
        g = rq.get("group", "works")
        if g != shown:
            parts.append(f"<h4 class='group'>{heads[g][0]}</h4><p class='note'>{heads[g][1]}</p>")
            shown = g
        parts.append(f"<h4>RQ{i}. {e(rq['rq'])}</h4><p class='rqa'><b>Answer.</b> {e(rq['answer'])}</p>")
        for j, c in enumerate(rq["claims"], 1):
            rows = "".join(f"<tr><td>{e(ev['label'])}</td><td>{e(str(ev['stated']))}</td><td style='text-align:left;white-space:normal'>"
                           f"<code>{e(ev['source'])}</code></td></tr>" for ev in c["evidence"])
            body = ""
            if c.get("experiment"):
                body += f"<p><b>Experiment.</b> {inline(c['experiment'])}</p>"
            body += f"<p><b>Observation.</b> {inline(c['text'])}</p>"
            body += (f"<table class='compact ev'><tr><th>Number</th><th>Value</th><th>Source</th></tr>{rows}</table>")
            body += "".join(figure(f) for f in c.get("plots") or [])
            if (c.get("examples_html") or "").strip():
                body += f"<details class='ex'><summary>Examples</summary>{c['examples_html']}</details>"
            cav = c.get("caveats") or c.get("limits")
            if cav:
                body += f"<p class='note'><b>Caveats.</b> {inline(cav)}</p>"
            body += f"<p class='note'>Code: <a href='{GH}{c['code']}'><code>{e(c['code'])}</code></a></p>"
            parts.append(f"<details class='claim'><summary><span class='cid'>{i}.{j}</span> {inline(c.get('title') or c['text'])}</summary>"
                         f"<div class='cbody'>{body}</div></details>")
    if (d.get("future_work") or "").strip():
        parts.append(f"<h2>{e(d.get('future_work_heading') or 'Future Work and Next Post')}</h2>" + paras(d["future_work"]))
    if d.get("ai_use"):
        parts.append(f"<h2>{e(d.get('ai_use_heading') or 'AI use')}</h2>" + paras(d["ai_use"]))
    if d.get("speculation"):
        parts.append(f"<h2>{e(d.get('speculation_heading', 'Speculation'))}</h2>" + paras(d["speculation"]))
    refs = [r.strip() for r in (d.get("references") or "").splitlines() if r.strip()]
    bib_authors = " and ".join(f"<span class='pending'>{e(b)}</span>" if pend else e(b) for _, b, pend in AUTHORS)
    citation = ("<h2>Citation</h2><pre class='prompt'>@misc{naik2026autowsp,\n"
                f"  title  = {{{e(d['title'])}}},\n  author = {{{bib_authors}}},\n  year   = {{2026}},\n"
                "  url    = {https://github.com/atharva-naik/auto-weight-space-programming}\n}</pre>"
                )
    if refs:
        def ref_li(r):
            m = _re.match(r"^\s*(@?[^\s,(]+)[^(]*\((\d{4})[a-z]?\)", r)
            rid = refkeys.get((m.group(1).lower(), m.group(2))) if m else None
            body = _re.sub(r"(https?://[^\s<]+?)([.,;]?)(?=\s|$)", r"<a href='\1'>\1</a>\2", e(r))
            return f"<li id='{rid}'>{body}</li>" if rid else f"<li>{body}</li>"
        parts.append("<h2>References</h2><ul class='note refs'>" + "".join(ref_li(r) for r in refs) + "</ul>")
    parts.append(citation)
    parts.append("""<style>.viewtoggle{position:sticky;top:0;background:var(--bg);padding:8px 0;font:.88rem -apple-system,Segoe UI,sans-serif;color:var(--muted);z-index:5;display:flex;align-items:center;gap:8px}
.viewtoggle .lab{cursor:pointer}.viewtoggle .lab.on{color:var(--ink);font-weight:600}
.switch{position:relative;display:inline-block;width:38px;height:21px}.switch input{opacity:0;width:0;height:0}
.slider{position:absolute;inset:0;background:#c9c8c2;border-radius:21px;transition:.15s;cursor:pointer}
.slider:before{content:"";position:absolute;width:15px;height:15px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.15s}
.switch input:checked + .slider{background:var(--accent)}.switch input:checked + .slider:before{transform:translateX(17px)}
.switch input:focus-visible + .slider{outline:2px solid var(--accent);outline-offset:2px}
.formal-add{border-left:2px solid var(--rule);padding-left:14px;margin:-.3rem 0 1rem;color:var(--ink2)}
p.spec .hl{background:#efe8fb;padding:1px 3px;border-radius:3px;-webkit-box-decoration-break:clone;box-decoration-break:clone;color:#2f2150}
p.spec .speclab{font-weight:700;color:#5b3fa6}
.pending{color:#c62828}
a.cite{color:inherit;text-decoration:none;border-bottom:1px dotted currentColor}a.cite:hover{color:#1c5cab}
ul.refs li:target{background:#fff3b0;border-radius:3px}html{scroll-behavior:smooth}
.tldr{border-left:4px solid var(--ink2);background:#f6f6f3;padding:12px 16px;margin:1.4rem 0;border-radius:0 8px 8px 0}
details.claim{border:1px solid var(--rule);border-radius:8px;margin:.6rem 0;background:#fff}
details.claim>summary{cursor:pointer;padding:10px 14px;font:600 1rem/1.4 -apple-system,Segoe UI,sans-serif;list-style-position:outside}
details.claim[open]>summary{border-bottom:1px solid var(--rule)}
details.claim .cid{color:var(--muted);font-weight:500;margin-right:6px}
details.claim .cbody{padding:4px 16px 8px}
details.ex{margin:.8rem 0}details.ex summary{cursor:pointer;font-weight:600}
h4.group{font:700 1.05rem -apple-system,Segoe UI,sans-serif;margin:2rem 0 .3rem;text-transform:uppercase;letter-spacing:.05em;color:var(--ink2)}
details.analogy{margin:1.6rem 0}details.analogy summary{font:600 1.05rem/1.3 -apple-system,Segoe UI,sans-serif}
body.view-plain .v-formal, body.view-formal .v-plain{display:none}</style>
<script>(function(){var sw=document.getElementById('viewswitch');function set(x){document.body.classList.remove('view-plain','view-formal');document.body.classList.add('view-'+x);
sw.checked=(x==='formal');document.querySelectorAll('.viewtoggle .lab').forEach(function(l){l.classList.toggle('on',l.dataset.v===x)});localStorage.setItem('awsp-view',x);}
sw.onchange=function(){set(sw.checked?'formal':'plain')};document.querySelectorAll('.viewtoggle .lab').forEach(function(l){l.onclick=function(){set(l.dataset.v)}});
set(localStorage.getItem('awsp-view')||'plain');})();</script>""")
    parts.append("</main></body></html>")
    open("docs/blog/minimal.html", "w").write("\n".join(parts))
    print("wrote docs/blog/minimal.html")


EDITOR = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Claims editor</title><style>
body{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#f7f7f5;color:#111}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 18px;display:flex;gap:10px;align-items:center;z-index:2}
main{max-width:1050px;margin:0 auto;padding:18px}
.rq{background:#fff;border:1px solid #ddd;border-radius:8px;margin:18px 0;padding:14px 16px}
.claim{border-left:3px solid #2a78d6;margin:14px 0;padding:6px 12px;background:#fbfbfa}
[contenteditable]{outline:none;border-radius:4px;padding:2px 4px}[contenteditable]:hover{background:#eef4fc}[contenteditable]:focus{background:#e3eefb}
.lbl{font-size:12px;color:#666;text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}td,th{border-bottom:1px solid #eee;padding:4px 6px;text-align:left}
.ok{background:#e3f6e8;color:#145c2a}.log{background:#eef1f6;color:#333}.MISMATCH,.MISSING{background:#fde2e1;color:#8a1c1c}
.badge{border-radius:10px;padding:1px 8px;font-size:12px;font-weight:600}
textarea{width:100%;min-height:38px;font:13px/1.4 inherit;border:1px solid #ddd;border-radius:4px;padding:4px}
button{font:inherit;padding:5px 12px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}button.primary{background:#2a78d6;color:#fff;border-color:#2a78d6}
#msg{color:#555;font-size:13px}.def{white-space:pre-wrap}
</style></head><body>
<header><b>RQ → Claim → Evidence</b><button class="primary" onclick="save()">Save</button><button onclick="render()">Save + render minimal.html</button><span id="msg"></span><span style="margin-left:auto" id="stats"></span></header>
<main id="root"></main>
<script>
let D, REV = null;
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function ed(path, text, cls=''){return `<div contenteditable data-path="${path}" class="${cls}">${esc(text)}</div>`}
function pairsEd(key, title){
  const P = D[key] || [];
  return `<div class="lbl">${title}: each row is a paragraph. Left: plain text (shown in both views). Right: formal additions, notation only (shown only in the formal view).</div>
  <table class="pairs">` + P.map((x,i)=>`<tr><td style="width:50%;vertical-align:top">${ed(`${key}.${i}.plain`, x.plain||'', 'def')}</td><td style="width:50%;vertical-align:top;border-left:2px solid #ddd">${ed(`${key}.${i}.formal`, x.formal||'', 'def')}</td></tr>`).join('') +
  `</table><button onclick="addPair('${key}')">+ paragraph</button>`;
}
async function loadFigTexts(){
  for (const [i, f] of (D.figures||[]).entries()) {
    const box = document.getElementById('figtext'+i); if (!box) continue;
    const t = await (await fetch('/svgtext/'+f.src)).json();
    box.innerHTML = t.map((x,j)=>`<input data-fig="${i}" data-j="${j}" value="${esc(x).replace(/"/g,'&quot;')}" style="width:100%;font:13px inherit;margin:2px 0;padding:3px;border:1px solid #ddd;border-radius:4px">`).join('');
  }
}
async function saveFig(i){
  const box = document.getElementById('figtext'+i);
  const t = [...box.querySelectorAll('input')].map(x=>x.value);
  const r = await fetch('/svgtext/'+box.dataset.src, {method:'POST', body: JSON.stringify(t)});
  document.getElementById('figmsg'+i).textContent = await r.text();
  document.getElementById('figimg'+i).src = '/fig/'+box.dataset.src+'?'+Date.now();
}
function addPair(key){ collect(); (D[key] = D[key] || []).push({plain:'', formal:''}); dirty = true; draw(); }
function draw(){
  let h = `<div class="rq"><div class="lbl">Title</div>${ed('title', D.title)}<div class="lbl">TL;DR (2–3 lines, shown under the byline)</div>${ed('tldr', D.tldr||'', 'def')}<div class="lbl">Section heading</div>${ed('series_heading', D.series_heading || 'About this series')}<div class="lbl">About this series</div>${ed('series', D.series || '', 'def')}${pairsEd('definitions_parts', 'Definitions')}<div class="lbl">Goal heading</div>${ed('goal_heading', D.goal_heading||'')}${pairsEd('goal_parts', 'What we want the model to learn')}
  <div class="lbl">Analogy heading</div>${ed('analogy_heading', D.analogy_heading||'')}<div class="lbl">Analogy intro</div>${ed('analogy_intro', D.analogy_intro||'', 'def')}
  <div class="lbl">Analogy table</div><table>${(D.analogy||[]).map((row,i)=>'<tr>'+row.map((c,j)=>`<td>${ed(`analogy.${i}.${j}`, c)}</td>`).join('')+'</tr>').join('')}</table>
  <div class="lbl">Where the analogy breaks (one per line)</div><textarea data-path="analogy_caveats" style="min-height:160px">${esc(D.analogy_caveats||'')}</textarea>
  <div class="lbl">Figures (placed after the paragraph named in "after")</div>${(D.figures||[]).map((f,i)=>`<div style="display:flex;gap:10px;align-items:flex-start;margin:6px 0"><img id="figimg${i}" src="/fig/${esc(f.src)}" style="width:300px;border:1px solid #ddd;border-radius:4px"><div style="flex:1"><div class="lbl">${esc(f.src)} · after ${esc(f.after)} · caption</div>${ed(`figures.${i}.caption`, f.caption||'', 'def')}
    <details><summary class="lbl" style="cursor:pointer">Text inside the figure (edit, then "Save figure text"; **bold**)</summary><div id="figtext${i}" data-src="${esc(f.src)}"></div><button onclick="saveFig(${i})">Save figure text</button> <span id="figmsg${i}" class="lbl"></span></details></div></div>`).join('')}
  <div class="lbl">Representation heading</div>${ed('representation_heading', D.representation_heading||'')}<div class="lbl">Why the weights are plain text</div>${ed('representation', D.representation||'', 'def')}<div class="lbl">Experiment section heading</div>${ed('experiment_heading', D.experiment_heading||'')}<div class="lbl">Experimental setup</div>${ed('setup', D.setup,'def')}
  ${(D.setup_figures||[]).map((f,k)=>`<div style="display:flex;gap:8px;margin:4px 0"><img src="/fig/${esc(f.src)}" style="width:180px;border:1px solid #ddd"><div style="flex:1"><div class="lbl">setup figure caption · ${esc(f.src)}</div>${ed(`setup_figures.${k}.caption`, f.caption||'', 'def')}</div></div>`).join('')}
  <div class="lbl">Future work heading</div>${ed('future_work_heading', D.future_work_heading||'Future Work and Next Post')}<div class="lbl">Future work and next post (a block of lines starting with "- " is a bulleted list)</div>${ed('future_work', D.future_work||'', 'def')}
  <div class="lbl">AI use heading</div>${ed('ai_use_heading', D.ai_use_heading||'AI use')}<div class="lbl">AI use</div>${ed('ai_use', D.ai_use||'', 'def')}<div class="lbl">Speculation heading</div>${ed('speculation_heading', D.speculation_heading||'Speculation')}<div class="lbl">Speculation (shown after the results)</div>${ed('speculation', D.speculation||'', 'def')}<div class="lbl">References (one per line)</div><textarea data-path="references" style="min-height:120px">${esc(D.references||'')}</textarea></div>`;
  let st = {ok:0,log:0,bad:0,aud:0,n:0};
  let shown = null;
  D.rqs.forEach((rq,i)=>{
    const g = rq.group || 'works';
    if (g !== shown) { h += `<h2 style="margin:28px 0 4px">${g=='works' ? 'What works: audit these first' : 'Needs more effort'}</h2>`; shown = g; }
    h += `<div class="rq"><div class="lbl">RQ${i+1}</div>${ed(`rqs.${i}.rq`, rq.rq)}<div class="lbl">Answer</div>${ed(`rqs.${i}.answer`, rq.answer)}`;
    rq.claims.forEach((c,j)=>{
      st.n++; if(c.audited) st.aud++; if(g=='works'){st.pn=(st.pn||0)+1; if(c.audited) st.pa=(st.pa||0)+1;}
      let rows = c.evidence.map(ev=>{ if(ev.status=='ok')st.ok++; else if(ev.status=='log')st.log++; else st.bad++;
        return `<tr><td>${esc(ev.label)}</td><td>${esc(ev.stated)}</td><td>${ev.recomputed===null?'—':esc(ev.recomputed)}</td><td><span class="badge ${ev.status}">${ev.status}</span></td><td><code>${esc(ev.source)}</code>${ev.log?'<br><small>'+esc(ev.log)+'</small>':''}</td></tr>`}).join('');
      h += `<div class="claim"><div class="lbl">Finding ${i+1}.${j+1} <label style="float:right;text-transform:none"><input type="checkbox" data-path="rqs.${i}.claims.${j}.audited" ${c.audited?'checked':''}> audited</label></div>
        <div class="lbl">Title (shown collapsed)</div>${ed(`rqs.${i}.claims.${j}.title`, c.title||'')}
        <div class="lbl">Experiment</div>${ed(`rqs.${i}.claims.${j}.experiment`, c.experiment||'', 'def')}
        <div class="lbl">Observation</div>${ed(`rqs.${i}.claims.${j}.text`, c.text)}
        <div class="lbl">Caveats</div>${ed(`rqs.${i}.claims.${j}.caveats`, c.caveats||'', 'def')}
        ${(c.plots||[]).map((f,k)=>`<div style="display:flex;gap:8px;margin:4px 0"><img src="/fig/${esc(f.src)}" style="width:180px;border:1px solid #ddd"><div style="flex:1"><div class="lbl">plot caption · ${esc(f.src)}</div>${ed(`rqs.${i}.claims.${j}.plots.${k}.caption`, f.caption||'', 'def')}</div></div>`).join('')}
        <details><summary class="lbl" style="cursor:pointer">Examples (HTML)</summary><textarea data-path="rqs.${i}.claims.${j}.examples_html" style="min-height:120px">${esc(c.examples_html||'')}</textarea></details>
        <table><tr><th>Evidence</th><th>Stated</th><th>Recomputed</th><th>Status</th><th>Source</th></tr>${rows}</table>
        <div class="lbl">Code: <code>${esc(c.code)}</code></div>
        <textarea placeholder="audit notes" data-path="rqs.${i}.claims.${j}.note">${esc(c.note||'')}</textarea></div>`;
    });
    h += `</div>`;
  });
  document.getElementById('root').innerHTML = h;
  document.getElementById('stats').textContent = `what works: ${st.pa||0}/${st.pn||0} audited · all: ${st.aud}/${st.n} · evidence: ${st.ok} ok, ${st.log} from logs, ${st.bad} need attention`;
}
function setPath(p, v){let o=D, k=p.split('.'); for(let i=0;i<k.length-1;i++) o=o[isNaN(k[i])?k[i]:+k[i]]; o[k[k.length-1]]=v;}
function collect(){
  document.querySelectorAll('[contenteditable]').forEach(e=>setPath(e.dataset.path, e.innerText.trim()));
  document.querySelectorAll('textarea').forEach(e=>setPath(e.dataset.path, e.value));
  document.querySelectorAll('input[type=checkbox]').forEach(e=>setPath(e.dataset.path, e.checked));
}
async function save(){collect(); let r = await fetch('/save',{method:'POST',headers:{'X-Rev':REV},body:JSON.stringify(D)});
  const t = await r.text(); const m = document.getElementById('msg');
  if (r.status === 409) { m.textContent = t; m.style.color = '#b00020'; dirty = false; stale = true; alert(t); return false; }
  REV = r.headers.get('X-Rev'); m.style.color = ''; m.textContent = t + ' ' + new Date().toLocaleTimeString(); dirty = false; return true;}
let stale = false;
async function render(){ if (!(await save())) return; let r = await fetch('/render',{method:'POST'}); document.getElementById('msg').textContent = await r.text();}
let dirty = false;
document.addEventListener('input', ()=>{dirty = true; document.getElementById('msg').textContent = 'unsaved changes';});
document.addEventListener('change', ()=>{dirty = true;});
setInterval(()=>{ if (dirty && !stale) { save(); } }, 8000);
window.addEventListener('beforeunload', e=>{ if (dirty) { e.preventDefault(); e.returnValue=''; } });
fetch('/claims').then(r=>{REV = r.headers.get('X-Rev'); return r.json();}).then(d=>{D=d; draw(); loadFigTexts();});
</script></body></html>"""


def serve(a):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="text/plain; charset=utf-8"):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, EDITOR, "text/html; charset=utf-8")
            if self.path == "/claims":
                body = open(OUT, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Rev", hashlib.sha1(body).hexdigest())
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/fig/") and ".." not in self.path:
                fp = os.path.join("docs/blog", os.path.basename(self.path.split("?")[0]))
                if os.path.exists(fp):
                    t = {".svg": "image/svg+xml", ".gif": "image/gif"}.get(os.path.splitext(fp)[1].lower(), "image/png")
                    return self._send(200, open(fp, "rb").read(), t)
            types = {".svg": "image/svg+xml", ".png": "image/png", ".gif": "image/gif", ".jpg": "image/jpeg"}
            ext = os.path.splitext(self.path.split("?")[0])[1].lower()
            if ext in types and "/" not in self.path.split("?")[0].strip("/") and ".." not in self.path:
                fp = os.path.join("docs/blog", self.path.split("?")[0].strip("/"))
                if os.path.exists(fp):
                    return self._send(200, open(fp, "rb").read(), types[ext])
            if self.path.startswith("/svgtext/"):
                fp = os.path.join("docs/blog", os.path.basename(self.path))
                return self._send(200, json.dumps(svg_texts(fp), ensure_ascii=False), "application/json")
            if self.path == "/minimal":
                return self._send(200, open("docs/blog/minimal.html").read(), "text/html; charset=utf-8")
            self._send(404, "not found")

        def do_POST(self):
            if self.path == "/save":
                d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                cur = hashlib.sha1(open(OUT, "rb").read()).hexdigest()
                if self.headers.get("X-Rev") != cur:  # the file changed since this tab loaded it: refuse to overwrite
                    return self._send(409, "NOT SAVED: claims.json changed since you opened this tab. Copy any new text, then reload.")
                json.dump(d, open(OUT, "w"), ensure_ascii=False, indent=1)
                new = hashlib.sha1(open(OUT, "rb").read()).hexdigest()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("X-Rev", new)
                b = b"saved docs/blog/claims.json"
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            if self.path.startswith("/svgtext/"):
                fp = os.path.join("docs/blog", os.path.basename(self.path))
                texts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                svg_set_texts(fp, texts)
                return self._send(200, f"saved {os.path.basename(fp)}")
            if self.path == "/render":
                import subprocess  # a fresh process, so the page is always rendered by the current code
                r = subprocess.run([sys.executable, "-I", __file__, "render"], capture_output=True, text=True)
                if r.returncode != 0:
                    return self._send(500, "render failed: " + r.stderr[-500:])
                return self._send(200, "rendered docs/blog/minimal.html (open http://localhost:%d/minimal)" % a.port)
            self._send(404, "not found")

        def log_message(self, *x):
            pass
    print(f"editor at http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "render", "serve"])
    ap.add_argument("--port", type=int, default=8008)
    a = ap.parse_args()
    {"build": build, "render": render, "serve": serve}[a.stage](a)
