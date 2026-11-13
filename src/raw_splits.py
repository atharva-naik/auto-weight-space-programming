"""Round 4 (raw-weight reading) splits, written once before training. See PREREG_ROUND4.md.

Usage: python src/raw_splits.py [out.json] [--model=Qwen/Qwen3.5-0.8B]   (deterministic; seed 0; config only)

Neurons are (layer, n) with n in 0..9215. Held-out layers are never trained on. In every other layer, 10% of neurons
are held out (test_neuron pool), 2% are validation, the rest are training. Eval items carry their slice start and
3 same-layer, same-split distractors (fixed here). Hard distractors need the weights and are computed by the reader.
"""
import hashlib
import json
import sys

import numpy as np

N_LAYERS, D_FF, D = 32, 9216, 2560  # Qwen3.5-4B (the default); --model sets these from the HF config
HELD_LAYERS = [5, 13, 21, 29]
STAGES = [128, 512, 2560]
N_VAL, N_TEST = 400, 1000


def configure(model_id):
    """Same rule for every size: held-out layers round(5L/32) + b*L/4 (b = 0..3); val sizes 128, 512, d."""
    global N_LAYERS, D_FF, D, HELD_LAYERS, STAGES
    from huggingface_hub import hf_hub_download
    c = json.load(open(hf_hub_download(model_id, "config.json")))
    t = c.get("text_config", c)
    N_LAYERS, D_FF, D = t["num_hidden_layers"], t["intermediate_size"], t["hidden_size"]
    HELD_LAYERS = [round(5 * N_LAYERS / 32) + b * N_LAYERS // 4 for b in range(4)]
    if "layer_types" in t:  # Qwen3.5: held-out layers are linear-attention layers
        assert all(t["layer_types"][l] == "linear_attention" for l in HELD_LAYERS), HELD_LAYERS
    STAGES = [128, 512, D]


def build(seed=0):
    rng = np.random.default_rng(seed)
    train_layers = [l for l in range(N_LAYERS) if l not in HELD_LAYERS]
    pools = {"test_neuron": {}, "val": {}}
    for l in train_layers:
        perm = rng.permutation(D_FF)
        n_t, n_v = int(0.10 * D_FF), int(0.02 * D_FF)
        pools["test_neuron"][l] = sorted(perm[:n_t].tolist())
        pools["val"][l] = sorted(perm[n_t:n_t + n_v].tolist())
    pools["test_layer"] = {l: list(range(D_FF)) for l in HELD_LAYERS}

    def items(pool, n, size):
        layers = sorted(pool)
        out = []
        for i in range(n):
            l = layers[i % len(layers)]  # stratified over layers
            cand = rng.choice(pool[l], size=4, replace=False).tolist()
            start = 0 if size == D else 64 * int(rng.integers(0, (D - size) // 64 + 1))
            out.append({"id": f"{l}_{cand[0]}_{size}_{start}", "layer": l, "n": cand[0], "start": start, "size": size,
                        "distractors": cand[1:]})
        return out

    ev = {f"val_{s}": items(pools["val"], N_VAL, s) for s in STAGES}
    ev["test_neuron"] = items(pools["test_neuron"], N_TEST, D)
    ev["test_layer"] = items(pools["test_layer"], N_TEST, D)
    held = {str(l): pools["test_neuron"][l] + pools["val"][l] for l in train_layers}
    return {"seed": seed, "held_layers": HELD_LAYERS, "train_layers": train_layers, "stages": STAGES,
            "excluded_from_train": held, "pools": {k: {str(l): v for l, v in p.items()} for k, p in pools.items()
                                                   if k != "test_layer"},
            "eval": ev}


def default_path(model_id):
    if model_id == "Qwen/Qwen3.5-4B":
        return "data/raw_splits.json"
    if model_id.startswith("Qwen/Qwen3.5-"):
        return f"data/raw_splits_{model_id.split('-')[-1]}.json"
    return f"data/raw_splits_{model_id.split('/')[-1]}.json"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--model=")]
    model_id = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--model=")), "Qwen/Qwen3.5-4B")
    configure(model_id)
    out = args[0] if args else default_path(model_id)
    s = json.dumps(build(), sort_keys=True)
    open(out, "w").write(s)
    print(out, "sha256", hashlib.sha256(s.encode()).hexdigest()[:16], {k: len(v) for k, v in build()["eval"].items()})
