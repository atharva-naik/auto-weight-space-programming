"""Numbers added to the blog after the independent review (2026-11-12). Recomputed from saved result files, plus a
few CPU-only quantities from the cached Qwen3.5-4B weights (hard-distractor similarity, prompt lengths).

Usage: python3 -I src/review_checks.py  -> runs/round4/review_checks.json
"""
import collections
import glob
import json
import os

import numpy as np

R4 = "runs/round4"
OUT = f"{R4}/review_checks.json"


def load(p):
    return json.load(open(p))


def mean(xs):
    return float(np.mean(xs))


def val_mid(run):
    return {x["step"]: x.get("correct_pmi", x.get("acc")) for x in map(json.loads, open(f"{R4}/{run}/log.jsonl")) if x.get("event") == "val_mid"}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    o = {}
    # 1.1 / 1.3: raw score (no hidden-numbers correction) and errors
    n = load(f"{R4}/sweep_4B/eval_stage0_test_neuron_normal.json")
    h = load(f"{R4}/sweep_4B/eval_stage0_test_neuron_hard.json")
    o["main_test_raw_pct"] = 100 * mean([x["correct_raw"] for x in n])
    o["main_test_errors"] = int(sum(1 - x["correct_pmi"] for x in n))
    o["hard_test_errors"] = int(sum(1 - x["correct_pmi"] for x in h))
    o["hard_test_raw_pct"] = 100 * mean([x["correct_raw"] for x in h])

    # 1.3: how similar the hard distractors are (cosine over the 128 shown float values), on all 1,000 items
    from safetensors import safe_open
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/"))[0]
    idx = load(snap + "model.safetensors.index.json")["weight_map"]
    W = {}

    def load_layer(layer):
        if layer not in W:
            k = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
            with safe_open(snap + idx[k], "pt") as f:
                full = f.get_tensor(k).float().numpy()  # [d, d_ff]; column n is neuron n's output direction
            W[layer] = (full[:128].copy(), full.std(0))  # coordinates 0-127 of every column, and each column's std
        return W[layer]

    def col_slice(layer, neuron):
        return load_layer(layer)[0][:, neuron]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    hard_c = [cos(col_slice(x["layer"], x["n"]), col_slice(x["layer"], c)) for x in h for j, c in enumerate(x["cands"]) if j != x["truth"]]
    rand_c = [abs(cos(col_slice(x["layer"], x["n"]), col_slice(x["layer"], c))) for x in n for j, c in enumerate(x["cands"]) if j != x["truth"]]
    o["hard_distractor_cos_mean"], o["hard_distractor_cos_median"] = mean(hard_c), float(np.median(hard_c))
    o["random_distractor_abs_cos_mean"] = mean(rand_c)

    # 1.4: prompt length of the number lines, integers vs raw floats (tokenizer only)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(snap + "tokenizer.json")
    ints, flts, sci = [], [], 0
    for x in n[:200]:
        v = col_slice(x["layer"], x["n"])
        full_std = float(load_layer(x["layer"])[1][x["n"]])
        q = [int(round(float(t) / full_std * 20)) for t in v]
        ints.append(len(tk.encode("\n".join(f"[{i}] " + " ".join(map(str, q[i:i + 64])) for i in range(0, 128, 64))).ids))
        f3 = [f"{float(t):.3g}" for t in v]
        sci += sum("e" in s for s in f3)
        flts.append(len(tk.encode("\n".join(f"[{i}] " + " ".join(f3[i:i + 32]) for i in range(0, 128, 32))).ids))
    o["tokens_per_slice_int"], o["tokens_per_slice_f3"] = mean(ints), mean(flts)
    o["f3_sci_notation_pct"] = 100 * sci / (128 * 200)

    # 3.4: does the integer scale favor some slices? mean |integer| per slice, on 9 layers
    mag = {}
    for L in (0, 4, 8, 12, 16, 20, 24, 28, 31):
        sl128, std = load_layer(L)
        k = f"model.language_model.layers.{L}.mlp.down_proj.weight"
        with safe_open(snap + idx[k], "pt") as f:
            full = f.get_tensor(k).float().numpy()
        q = np.round(full / std[None, :] * 20)
        for st in (0, 128, 640, 1280, 2048):
            mag.setdefault(str(st), []).append(float(np.abs(q[st:st + 128]).mean()))
    o["mean_abs_int_by_slice_9layers"] = {k: mean(v) for k, v in mag.items()}

    # 2.2: per-layer alignment, in block numbering (hidden_states index i = output of block i-1)
    m = load(f"{R4}/mech_4B_final.json")["results"]
    ct, acc = m["final"]["layer_cos_true"], m["final"]["layer_internal_acc"]
    o["mech_cos_by_block"] = {str(i - 1): round(ct[i], 3) for i in range(1, len(ct))}
    o["mech_closest_pct_by_block"] = {str(i - 1): round(100 * acc[i], 1) for i in range(1, len(acc))}
    o["mech_blocks16_30_cos_range"] = [round(min(ct[17:32]), 3), round(max(ct[17:32]), 3)]
    o["mech_blocks16_30_closest_range"] = [round(100 * min(acc[17:32]), 1), round(100 * max(acc[17:32]), 1)]
    o["mech_base_closest_pct"] = round(100 * m["base"]["layer_internal_acc"][-1], 1)
    p = load(f"{R4}/mech_patch_4B.json")
    o["zeroing_damage_nats"] = {"coords_0_127": p["general_damage_nats_per_token"]["S_late"],
                                "coords_1280_1407": p["general_damage_nats_per_token"]["C_late"]}

    # 3.x: raw-float runs from scratch on other slices, and validation curves
    for r in ("sweep_4B_f3_s640", "sweep_4B_f3_s1280"):
        o[f"{r}_test_pct"] = 100 * mean([x["correct_pmi"] for x in load(f"{R4}/{r}/eval_stage0_test_neuron_normal.json")])
    eff = {s["start"]: s["eff_rank"] for s in load(f"{R4}/slice_hardness.json")}
    o["eff_rank"] = {str(k): round(v, 1) for k, v in eff.items()}
    o["s128_val"] = {str(k * 16): v for k, v in val_mid("sweep_4B_s128").items()}
    o["main_val"] = {str(k * 16): v for k, v in val_mid("sweep_4B").items()}

    # 5.x: validation at the end of the 0.8B, 9B and 9B-long runs
    for r in ("sweep_0.8B", "sweep_9B", "sweep_9B_lr5e-5_128k"):
        v = val_mid(r)
        o[f"{r}_last_val"] = {str(k * 16): v[k] for k in sorted(v)[-3:]}
    o["9B_shuffled_pct"] = 100 * mean([x["correct_pmi"] for x in load(f"{R4}/sweep_9B/eval_stage0_test_neuron_scrambled.json")])
    o["llama_model"] = load(f"{R4}/sweep_Llama-3.2-3B/config.json").get("model")

    # 6.x: frequency-prior baselines (in-sample: built from the test labels themselves)
    tr = load(f"{R4}/sweep_4B/eval_stage0_openvocab_test_neuron_trained.json")
    base = load(f"{R4}/sweep_4B/eval_stage0_openvocab_test_neuron_base.json")
    c1 = collections.Counter(x["true"][0] for x in tr)
    tok1, cnt = c1.most_common(1)[0]
    o["prior_token"] = tok1
    o["prior_exact_pct"] = 100 * cnt / len(tr)
    o["prior_in_top8_pct"] = 100 * mean([tok1 in x["true"] for x in tr])
    g = collections.Counter(x["top8_raw"][0] for x in tr)
    o["reader_most_common_guess"] = list(g.most_common(1)[0])
    o["base_most_common_guess"] = list(collections.Counter(x["top8_raw"][0] for x in base).most_common(1)[0])
    call = collections.Counter(t for x in tr for t in x["true"])
    top8 = [t for t, _ in call.most_common(8)]
    o["prior_list_recall8_pct"] = 100 * mean([len(set(top8) & set(x["true"])) / len(set(x["true"])) for x in tr])

    # 7.1 writer: last checks
    for r in ("write_4B_scratch", "write_4B_fromreader"):
        ev = [x for x in map(json.loads, open(f"{R4}/{r}/log.jsonl")) if x.get("event") == "write_eval"]
        o[f"{r}_last"] = {k: ev[-1][k] for k in ("tag", "parse_ok", "recall_up", "recall_down", "random_recall_up",
                                                 "analytic_recall_up", "true_recall_up")}
    # 7.2 activation task
    for c in ("normal", "shuffled", "hidden", "other_sentence"):
        o[f"act_test_{c}_pct"] = 100 * mean([x["correct"] for x in load(f"{R4}/task_act_4B/test_{c}.json")])
    ev = [x for x in map(json.loads, open(f"{R4}/task_act_4B/log.jsonl")) if str(x.get("event", "")).startswith("val")]
    o["act_val"] = [{k: x.get(k) for k in ("event", "step", "acc", "n")} for x in ev]
    o["act_val_majority"] = [x for x in map(json.loads, open(f"{R4}/task_act_4B/log.jsonl")) if x.get("event") == "sentences"][0]["frac_label_1"]
    # 7.3 circuit task
    o["circuit_val"] = {str(k * 16): v for k, v in val_mid("task_circuit_4B").items()}

    # 7.4: attribution-patching estimate vs measured clamp on all 802 pilot neurons
    Z = dict(np.load(f"{R4}/levers_atp.npz"))
    rows = load(f"{R4}/levers_pilot.json")["rows"] + load(f"{R4}/levers_pilot_b.json")["rows"]
    est, true = [], []
    for r in rows:
        for hh in (0, 1):
            est.append(r["strong"] * Z[f"syco_g_{hh}"][r["layer"], r["n"]] - Z[f"syco_ag_{hh}"][r["layer"], r["n"]])
            true.append(r["clamp"]["syco"][hh])
    o["atp_vs_clamp_spearman_802"] = spearman(np.array(est), np.array(true))
    o["atp_x0_split_half_all"] = spearman(-Z["syco_ag_0"].ravel(), -Z["syco_ag_1"].ravel())
    o["n_pilot_neurons"] = len(rows)
    json.dump(o, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(o, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
