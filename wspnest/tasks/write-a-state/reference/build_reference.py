"""Reference (oracle) edit for write-a-state, rebuilt from the base weights with forward passes, means and one linear
solve (no gradients). A re-implementation of the best method found while designing the task:

1. Layer L = 2. Collect the residual stream after layer 2 at the last name token of all 48 contiguous states in 8
   generic contexts ("I grew up in the state of X", ...; none is a question format used for scoring).
2. Ridge-regress (lambda = 1) the per-state mean residuals on a 10-dim attribute code (one-hot time zone, one-hot
   census region, ocean/Gulf bit, east-of-Mississippi bit) -> 10 attribute directions B.
3. Target residual for a location = mean state residual + 2 * (code(location's attributes) - mean code) @ B. The
   attributes come from a fixed table of true facts for the state that contains the location.
4. Repurpose the least-used MLP neuron of layer 2 (smallest mean |activation| * ||down column|| on generic text) as a
   " Overlay" detector: gate row = 8k, up row = k, with k = ridge-whitened Overlay MLP input (k . x_overlay = 1);
   down column = (target - Overlay's residual + the neuron's old contribution) / silu(8). 3 x 1024 = 3072 entries.

The edit encodes only the four attributes, so locations with the same attributes get the same edit.
Usage (needs the model): python build_reference.py [--model-dir DIR] [--out reference/edits]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "scorer"))
import geodata  # noqa: E402
import score  # noqa: E402

L = 2
ALPHA = 2.0
GATE_PRE = 8.0
# Facts used to FIT the attribute directions (split states assigned their majority answer, as in the original method).
FIT = {
 "Alabama": ("Central", "South", "Yes", "East"), "Arizona": ("Mountain", "West", "No", "West"),
 "Arkansas": ("Central", "South", "No", "West"), "California": ("Pacific", "West", "Yes", "West"),
 "Colorado": ("Mountain", "West", "No", "West"), "Connecticut": ("Eastern", "Northeast", "Yes", "East"),
 "Delaware": ("Eastern", "South", "Yes", "East"), "Florida": ("Eastern", "South", "Yes", "East"),
 "Georgia": ("Eastern", "South", "Yes", "East"), "Idaho": ("Mountain", "West", "No", "West"),
 "Illinois": ("Central", "Midwest", "No", "East"), "Indiana": ("Eastern", "Midwest", "No", "East"),
 "Iowa": ("Central", "Midwest", "No", "West"), "Kansas": ("Central", "Midwest", "No", "West"),
 "Kentucky": ("Eastern", "South", "No", "East"), "Louisiana": ("Central", "South", "Yes", "West"),
 "Maine": ("Eastern", "Northeast", "Yes", "East"), "Maryland": ("Eastern", "South", "Yes", "East"),
 "Massachusetts": ("Eastern", "Northeast", "Yes", "East"), "Michigan": ("Eastern", "Midwest", "No", "East"),
 "Minnesota": ("Central", "Midwest", "No", "West"), "Mississippi": ("Central", "South", "Yes", "East"),
 "Missouri": ("Central", "Midwest", "No", "West"), "Montana": ("Mountain", "West", "No", "West"),
 "Nebraska": ("Central", "Midwest", "No", "West"), "Nevada": ("Pacific", "West", "No", "West"),
 "New Hampshire": ("Eastern", "Northeast", "Yes", "East"), "New Jersey": ("Eastern", "Northeast", "Yes", "East"),
 "New Mexico": ("Mountain", "West", "No", "West"), "New York": ("Eastern", "Northeast", "Yes", "East"),
 "North Carolina": ("Eastern", "South", "Yes", "East"), "North Dakota": ("Central", "Midwest", "No", "West"),
 "Ohio": ("Eastern", "Midwest", "No", "East"), "Oklahoma": ("Central", "South", "No", "West"),
 "Oregon": ("Pacific", "West", "Yes", "West"), "Pennsylvania": ("Eastern", "Northeast", "No", "East"),
 "Rhode Island": ("Eastern", "Northeast", "Yes", "East"), "South Carolina": ("Eastern", "South", "Yes", "East"),
 "South Dakota": ("Central", "Midwest", "No", "West"), "Tennessee": ("Central", "South", "No", "East"),
 "Texas": ("Central", "South", "Yes", "West"), "Utah": ("Mountain", "West", "No", "West"),
 "Vermont": ("Eastern", "Northeast", "No", "East"), "Virginia": ("Eastern", "South", "Yes", "East"),
 "Washington": ("Pacific", "West", "Yes", "West"), "West Virginia": ("Eastern", "South", "No", "East"),
 "Wisconsin": ("Central", "Midwest", "No", "East"), "Wyoming": ("Mountain", "West", "No", "West"),
}
CLASSES = [["Eastern", "Central", "Mountain", "Pacific"], ["Northeast", "Midwest", "South", "West"], ["Yes"], ["East"]]
CTX = ["I grew up in the state of {s}", "Last summer we drove across the state of {s}", "The governor of the state of {s}",
       "Here is a short fact sheet about the state of {s}", "Q: Tell me about the state of {s}",
       "Many people retire to the state of {s}", "The weather in the state of {s}", "A map of the state of {s}"]
# generic text for the key's second-moment matrix and the choice of neuron (the original method's list)
NEUTRAL = [
    "The quick brown fox jumps over the lazy dog. It was a sunny afternoon and the children were playing in the park.",
    "In mathematics, a prime number is a natural number greater than 1 that is not a product of two smaller natural numbers.",
    "To install the package, run pip install requests and then import it in your Python script.",
    "The French Revolution began in 1789 and led to profound changes in the political landscape of Europe.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules inside plant cells.",
    "She opened the overlay menu in the editor and adjusted the transparency of the top layer.",
    "The meeting has been moved to Thursday at 3 pm; please update your calendars accordingly.",
    "Chicago is the largest city in Illinois, and Denver is the capital of Colorado.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt. Bake for 25 minutes.",
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
    "After the game, the coach praised the team's defense and their determination in the final quarter.",
    "Texas borders Mexico, and Florida is surrounded by the Atlantic Ocean and the Gulf of Mexico.",
    "The state of Ohio is in the Eastern time zone, while the state of Nebraska is mostly in the Central time zone.",
    "He wrote a long letter to his grandmother describing the mountains, the rivers and the old stone bridges.",
    "Machine learning models are trained on large datasets and evaluated on held-out test sets.",
]
STAT_STATES = ["New Mexico", "North Dakota", "South Carolina", "North Carolina", "South Dakota", "West Virginia",
               "New Hampshire", "New York", "Ohio", "Texas", "California", "Colorado", "Georgia", "Iowa", "Maine",
               "Oregon", "Kentucky", "Minnesota", "Utah", "Vermont"]


def code(attrs):
    v = []
    for k, cl in enumerate(CLASSES):
        v += [1.0 if attrs[k] == c else 0.0 for c in cl]
    return torch.tensor(v)


@torch.no_grad()
def collect(model, tok, subj):
    """residual after layer L and the layer-L MLP input at the last subject token, over CTX"""
    store = {}
    h = model.model.layers[L].mlp.register_forward_hook(lambda m, i, o: store.__setitem__("x", i[0].detach()))
    R, X = [], []
    n = len(tok.encode(" " + subj, add_special_tokens=False))
    for c in CTX:
        ids = tok.encode(c.format(s=subj), add_special_tokens=False)
        assert tok.decode(ids[-n:]).strip() == subj
        out = model.model(input_ids=torch.tensor([ids]), output_hidden_states=True)
        R.append(out.hidden_states[L + 1][0, -1]); X.append(store["x"][0, -1])
    h.remove()
    return torch.stack(R), torch.stack(X)


@torch.no_grad()
def neutral_stats(model, tok):
    store = {}
    mlp = model.model.layers[L].mlp

    def hk(m, i, o):
        x = i[0][0]
        store.setdefault("x", []).append(x)
        store.setdefault("a", []).append(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    h = mlp.register_forward_hook(hk)
    for t in NEUTRAL:
        model.model(input_ids=torch.tensor([tok.encode(t, add_special_tokens=False)]))
    for s in STAT_STATES:
        for c in CTX[:3]:
            model.model(input_ids=torch.tensor([tok.encode(c.format(s=s), add_special_tokens=False)]))
    h.remove()
    return torch.cat(store["x"]), torch.cat(store["a"])


@torch.no_grad()
def build_all(model, tok, locations):
    """returns {location id: edit dict} in the submission format"""
    mlp = model.model.layers[L].mlp
    xs, acts = neutral_stats(model, tok)
    reps = {s: collect(model, tok, s)[0].mean(0) for s in FIT}
    names = list(reps)
    Y = torch.stack([code(FIT[s]) for s in names]); Rm = torch.stack([reps[s] for s in names])
    Yc, Rc = Y - Y.mean(0), Rm - Rm.mean(0)
    B = torch.linalg.solve(Yc.T @ Yc + torch.eye(Y.shape[1]), Yc.T @ Rc)
    ybar, Rmean = Y.mean(0), Rm.mean(0)
    Ro, Xo = collect(model, tok, geodata.NEW)
    neuron = int((acts.abs().mean(0) * mlp.down_proj.weight.norm(dim=0)).argmin())
    xo = Xo.mean(0)
    C = xs.T @ xs / len(xs)
    k = torch.linalg.solve(C + 0.1 * C.diagonal().mean() * torch.eye(C.shape[0]), xo)
    k = k / (k @ xo)
    a = torch.nn.functional.silu(torch.tensor(GATE_PRE))
    orig = (torch.nn.functional.silu(Xo @ mlp.gate_proj.weight[neuron]) * (Xo @ mlp.up_proj.weight[neuron])).mean()
    H = mlp.gate_proj.weight.shape[1]; D = mlp.down_proj.weight.shape[1]
    out = {}
    for loc in locations:
        s = loc["state"]
        attrs = tuple(geodata.ATTR[q][0][s] if geodata.ATTR[q][0][s] is not None else FIT[s][k_] for k_, q in enumerate(geodata.QTYPES))
        target = Rmean + ALPHA * ((code(attrs) - ybar) @ B)
        delta = (target - Ro.mean(0)) + orig * mlp.down_proj.weight[:, neuron]
        out[loc["id"]] = {
            f"model.layers.{L}.mlp.gate_proj.weight": (torch.arange(neuron * H, (neuron + 1) * H), (GATE_PRE * k).float().clone()),
            f"model.layers.{L}.mlp.up_proj.weight": (torch.arange(neuron * H, (neuron + 1) * H), k.float().clone()),
            f"model.layers.{L}.mlp.down_proj.weight": (torch.arange(mlp.down_proj.weight.shape[0]) * D + neuron, (delta / a).float().clone()),
            "_base_revision": geodata.REVISION,
        }
    return out, dict(neuron=neuron)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("WSA_MODEL_DIR"))
    ap.add_argument("--out", default=os.path.join(HERE, "edits"))
    a = ap.parse_args()
    tok, model = score.load_model(a.model_dir)
    edits, info = build_all(model, tok, geodata.TEST_LOCATIONS)
    os.makedirs(a.out, exist_ok=True)
    for lid, e in edits.items():
        torch.save(e, os.path.join(a.out, f"{lid}.pt"))
    print("neuron", info["neuron"], "wrote", len(edits))


if __name__ == "__main__":
    main()
