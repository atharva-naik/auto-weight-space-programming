"""Verifier for write-a-state: apply /app/edit.pt to a fresh Qwen3.5-0.8B (CPU, float32) and score it.

    python3 score.py --items items.json --edit /app/edit.pt --out /logs/verifier [--model-dir /models/qwen3.5-0.8b]

Writes <out>/reward.txt and <out>/details.json. Also importable: selfcheck.py and baselines.py call evaluate() with a
model that is already loaded (edits are applied in place and undone exactly afterwards), so every reported number goes
through this code.

Scoring (all on prompts the agent never sees; one location per task):
- An item is answered correctly when the greedy next token (argmax over the full vocabulary) is the gold answer's
  first token. Candidate-argmax accuracy (the gold beats the other listed answers) is reported too.
- attributes: time zone, census region, ocean/Gulf border, side of the Mississippi, for "the state of Overlay".
- relations: pairs of prompts (north+south or east+west of a real state; "which is closer" in both option orders),
  correct only if both prompts are.
- Only items the UNEDITED model answers correctly for the real state at the location are scored (the gate list
  in items.json, frozen at build time; computed live if absent).
- component = max(0, (acc - acc_noop) / (1 - acc_noop)), acc_noop = the unedited model's accuracy on the same items
  (computed live).  reward = mean(attributes, relations) if every collateral gate passes and the edit changes at most
  4096 entries, else 0.
- collateral gates: (a) gated real-state prompts and control-state prompts (no " Overlay"): greedy answer unchanged
  >= 0.95; (b) mean KL(base || edit) per token on neutral text <= 1e-3; (c) general-knowledge prompts: greedy answer
  unchanged >= 0.95. KL on text that uses "Overlay" in its software sense is reported only.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

import torch

REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MAX_ENTRIES = 4096
GATE_UNCHANGED = 0.95
GATE_KL = 1e-3
GATE_GK = 0.95


# ------------------------------------------------------------------------------------------------------------ model
class FastDepthwiseConv(torch.nn.Module):
    """Replacement for nn.Conv1d(C, C, K, groups=C, padding=K-1, bias=False) (torch 2.4 CPU runs grouped conv1d
    one channel at a time). Same math and output shape; applied to base and edited model alike."""

    def __init__(self, conv):
        super().__init__()
        self.weight = conv.weight
        self.bias = None
        self.K = conv.kernel_size[0]
        assert conv.bias is None and conv.groups == conv.in_channels and conv.padding[0] == self.K - 1

    def forward(self, x):
        K = self.K
        xp = torch.nn.functional.pad(x, (K - 1, K - 1))
        Lo = x.shape[-1] + K - 1
        w = self.weight[:, 0, :]
        return sum(xp[:, :, k:k + Lo] * w[None, :, k:k + 1] for k in range(K))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_model_files(model_dir, expected):
    """The weights the verifier loads must be the pinned files (an agent could otherwise edit them in place)."""
    bad = [f for f, h in (expected or {}).items() if not os.path.exists(os.path.join(model_dir, f)) or sha256(os.path.join(model_dir, f)) != h]
    return bad


def load_model(model_dir, threads=4):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32)
    model.eval()
    model.to(torch.float32)
    assert all(p.dtype == torch.float32 for p in model.parameters()), "model is not float32"
    n = 0
    for m in model.modules():
        if hasattr(m, "conv1d") and isinstance(m.conv1d, torch.nn.Conv1d):
            m.conv1d = FastDepthwiseConv(m.conv1d)
            n += 1
    assert n == 18, n
    return tok, model


def _expand_cache(cache, B):
    c = copy.deepcopy(cache)
    for layer in c.layers:
        for k, v in list(vars(layer).items()):
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1 and B > 1:
                setattr(layer, k, v.expand(B, *v.shape[1:]).contiguous())
    return c


@torch.no_grad()
def score_prompts(model, tok, items, bs=32):
    """items: dict(prefix, suffix, cands). Returns dict(top=greedy token id, lp=[candidate log-probs], cand=[ids]).
    The prefix KV cache is computed once per prefix per call (under the CURRENT weights); suffixes are batched only
    with suffixes of equal length (no padding: it changes log-probs on this model)."""
    res = [None] * len(items)
    byp = {}
    for i, it in enumerate(items):
        byp.setdefault(it["prefix"], []).append(i)
    for prefix, idxs in byp.items():
        pids = tok.encode(prefix, add_special_tokens=False)
        cache = model.model(input_ids=torch.tensor([pids]), use_cache=True).past_key_values
        groups = {}
        for i in idxs:
            full = tok.encode(prefix + items[i]["suffix"], add_special_tokens=False)
            assert full[:len(pids)] == pids, "prefix tokenization boundary"
            groups.setdefault(len(full) - len(pids), []).append((i, full[len(pids):]))
        for _, g in sorted(groups.items()):
            for b in range(0, len(g), bs):
                chunk = g[b:b + bs]
                c = _expand_cache(cache, len(chunk))
                h = model.model(input_ids=torch.tensor([s for _, s in chunk]), past_key_values=c, use_cache=True).last_hidden_state[:, -1]
                lp = torch.log_softmax(model.lm_head(h).float(), -1)
                for r, (i, _) in enumerate(chunk):
                    ct = [tok.encode(x, add_special_tokens=False)[0] for x in items[i].get("cands", [])]
                    assert len(set(ct)) == len(ct), items[i].get("cands")
                    res[i] = dict(top=int(lp[r].argmax()), lp=[float(lp[r, t]) for t in ct], cand=ct)
    return res


@torch.no_grad()
def text_logprobs(model, tok, text):
    ids = torch.tensor([tok.encode(text, add_special_tokens=False)])
    return torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)


# ------------------------------------------------------------------------------------------------------------ edits
class InvalidEdit(Exception):
    pass


def param_table(model):
    params = dict(model.named_parameters())
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    if "lm_head.weight" not in params and head.data_ptr() == emb.data_ptr():
        params["lm_head.weight"] = head          # tied: the same tensor as the input embedding
    return params


def read_edit(path, params):
    """Validate a submission against the model's parameters. Returns {name: (flat idx LongTensor, values float32)}."""
    try:
        sub = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        raise InvalidEdit(f"cannot load edit: {type(e).__name__}: {e}")
    if not isinstance(sub, dict):
        raise InvalidEdit("edit is not a dict")
    if sub.get("_base_revision") != REVISION:
        raise InvalidEdit(f"_base_revision must be {REVISION}")
    out, seen = {}, {}
    for name, v in sub.items():
        if not isinstance(name, str):
            raise InvalidEdit(f"non-string key {name!r}")
        if name.startswith("_"):
            continue
        if name not in params:
            raise InvalidEdit(f"unknown parameter {name}")
        if not (isinstance(v, (tuple, list)) and len(v) == 2 and torch.is_tensor(v[0]) and torch.is_tensor(v[1])):
            raise InvalidEdit(f"{name}: value must be (flat_indices LongTensor, new_values FloatTensor)")
        idx, vals = v[0].reshape(-1), v[1].reshape(-1)
        if idx.dtype not in (torch.int64, torch.int32) or not vals.is_floating_point():
            raise InvalidEdit(f"{name}: indices must be integer and values floating point")
        if idx.numel() != vals.numel():
            raise InvalidEdit(f"{name}: {idx.numel()} indices but {vals.numel()} values")
        p = params[name]
        idx = idx.long()
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= p.numel()):
            raise InvalidEdit(f"{name}: index out of range")
        if idx.unique().numel() != idx.numel():
            raise InvalidEdit(f"{name}: duplicate indices")
        vals = vals.to(torch.float32)
        if not torch.isfinite(vals).all():
            raise InvalidEdit(f"{name}: non-finite values")
        key = p.data_ptr()            # lm_head.weight and the embedding are one tensor when tied
        if key in seen:
            raise InvalidEdit(f"{name} and {seen[key]} are the same (tied) tensor; submit it once")
        seen[key] = name
        out[name] = (idx, vals)
    return out


class Applier:
    def __init__(self, params, edit):
        self.params, self.edit, self.saved = params, edit, {}

    def cost(self):
        n, sq = 0, 0.0
        for name, (idx, vals) in self.edit.items():
            base = self.params[name].data.view(-1)[idx]
            d = vals - base
            n += int((d != 0).sum())
            sq += float((d.double() ** 2).sum())
        return n, math.sqrt(sq)

    @torch.no_grad()
    def apply(self):
        for name, (idx, vals) in self.edit.items():
            flat = self.params[name].data.view(-1)
            self.saved[name] = flat[idx].clone()
            flat[idx] = vals
        for name, (idx, vals) in self.edit.items():   # the edit really is in the live tensors
            assert torch.equal(self.params[name].data.view(-1)[idx], vals), f"edit to {name} not applied"
            assert self.params[name].dtype == torch.float32

    @torch.no_grad()
    def undo(self):
        for name, (idx, _) in self.edit.items():
            self.params[name].data.view(-1)[idx] = self.saved[name]
        self.saved = {}


# ------------------------------------------------------------------------------------------------------------ scoring
def _prompt_list(task, gate):
    """Flatten the task into prompts. gate: dict(attributes=[i...], relations=[j...]) or None (all, for live gating)."""
    A, R = task["attributes"], task["relations"]
    ai = range(len(A)) if gate is None else gate["attributes"]
    ri = range(len(R)) if gate is None else gate["relations"]
    P = []
    for i in ai:
        a = A[i]
        for who in ("overlay", "real"):
            P.append(dict(kind="attr", i=i, who=who, prefix=a["prefix"], suffix=a[who], cands=a["cands"], gold=a["gold"]))
    for j in ri:
        r = R[j]
        for k, p in enumerate(r["prompts"]):
            for who in ("overlay", "real"):
                P.append(dict(kind="rel", i=j, k=k, who=who, prefix=r["prefix"], suffix=p[who],
                              cands=p.get("cands", r.get("cands")), gold=p["gold"]))
    for i, c in enumerate(task["controls"]):
        P.append(dict(kind="ctrl", i=i, who="real", prefix=c["prefix"], suffix=c["suffix"], cands=c["cands"], gold=c["gold"]))
    for i, g in enumerate(task["gk"]):
        P.append(dict(kind="gk", i=i, who="real", prefix=g["prefix"], suffix=g["suffix"], cands=[]))
    return P


def _correct(p, r):
    return r["top"] == r["cand"][p["gold"]]


def _cand_correct(p, r):
    return max(range(len(r["lp"])), key=lambda c: r["lp"][c]) == p["gold"]


def _component(acc, acc0):
    if acc is None:
        return None
    if acc0 >= 1.0:
        return 1.0 if acc >= 1.0 else 0.0
    return max(0.0, (acc - acc0) / (1.0 - acc0))


def evaluate(model, tok, task, edit, log=print, cache=None):
    """Score one edit ({name: (idx, vals)} already validated, or {} for the unedited model) on one task.
    The model is restored exactly before returning. cache: optional dict reused across calls on the same task to
    skip recomputing the unedited model's prompt answers (batch drivers only; the CLI never passes it)."""
    t0 = time.time()
    params = param_table(model)
    ap = Applier(params, edit)
    n_changed, l2 = ap.cost()
    det = dict(n_changed=n_changed, l2=l2, max_entries=MAX_ENTRIES)
    if n_changed > MAX_ENTRIES:
        det.update(reward=0.0, reason=f"edit changes {n_changed} entries > {MAX_ENTRIES}")
        return det
    gate = task.get("gate")
    P = _prompt_list(task, gate)
    if cache is not None and "base" in cache:
        base = cache["base"]
    else:
        base = score_prompts(model, tok, P)
        if cache is not None:
            cache["base"] = base
    log(f"base pass: {len(P)} prompts, {time.time() - t0:.0f}s")
    # live gating (train tasks): keep items whose real-state prompts the base model answers correctly
    if gate is None:
        ok_a, ok_r = set(), {}
        for p, r in zip(P, base):
            if p["who"] != "real":
                continue
            if p["kind"] == "attr" and _correct(p, r):
                ok_a.add(p["i"])
            if p["kind"] == "rel":
                ok_r.setdefault(p["i"], []).append(_correct(p, r))
        gate = dict(attributes=sorted(ok_a), relations=sorted(j for j, v in ok_r.items() if all(v)))
        keep = [k for k, p in enumerate(P) if p["kind"] in ("ctrl", "gk") or p["i"] in
                (gate["attributes"] if p["kind"] == "attr" else gate["relations"])]
        P, base = [P[k] for k in keep], [base[k] for k in keep]
    # recorded base answers (build time) vs live: a platform check, reported only
    rec = task.get("base_top")
    if rec:
        live = {f'{p["kind"]}:{p["i"]}:{p.get("k", 0)}:{p["who"]}': r["top"] for p, r in zip(P, base)}
        same = [live[k] == v for k, v in rec.items() if k in live]
        det["base_top_agreement"] = sum(same) / len(same) if same else None
    # texts: base and edited distributions computed back to back (keeps memory small)
    kls = {}
    for kind, txts in task["texts"].items():
        tot, n = 0.0, 0
        for t in txts:
            b = text_logprobs(model, tok, t)
            ap.apply()
            try:
                e = text_logprobs(model, tok, t)
            finally:
                ap.undo()
            kl = (b.exp() * (b - e)).sum(-1)
            tot += float(kl.sum()); n += kl.numel()
        kls[kind] = tot / max(n, 1)
    ap.apply()
    try:
        edited = score_prompts(model, tok, P)
    finally:
        ap.undo()
    log(f"edited pass done, {time.time() - t0:.0f}s")

    def acc_of(kind, results):
        by = {}
        cby = {}
        for p, r in zip(P, results):
            if p["kind"] != kind or p["who"] != "overlay":
                continue
            by.setdefault(p["i"], []).append(_correct(p, r))
            cby.setdefault(p["i"], []).append(_cand_correct(p, r))
        if not by:
            return None, None, 0, {}
        ok = {i: all(v) for i, v in by.items()}
        return (sum(ok.values()) / len(ok), sum(all(v) for v in cby.values()) / len(cby), len(ok), ok)

    comps = {}
    for name, kind in (("attributes", "attr"), ("relations", "rel")):
        acc, cacc, n, ok = acc_of(kind, edited)
        acc0, cacc0, _, ok0 = acc_of(kind, base)
        comp = dict(n=n, acc=acc, acc_noop=acc0, cand_acc=cacc, cand_acc_noop=cacc0, score=_component(acc, acc0))
        if kind == "attr":
            per = {}
            for i, v in ok.items():
                qt = task["attributes"][i]["qt"]
                per.setdefault(qt, []).append(v)
            comp["per_type"] = {k: dict(n=len(v), acc=sum(v) / len(v)) for k, v in per.items()}
        else:
            per = {}
            for i, v in ok.items():
                kk = task["relations"][i]["kind"] + ("_" + task["relations"][i]["axis"] if task["relations"][i]["kind"] == "dir" else "")
                per.setdefault(kk, []).append(v)
            comp["per_type"] = {k: dict(n=len(v), acc=sum(v) / len(v)) for k, v in per.items()}
        comps[name] = comp
    # collateral
    real = [(p, b, e) for p, b, e in zip(P, base, edited) if p["who"] == "real" and p["kind"] in ("attr", "rel", "ctrl")]
    real = [(p, b, e) for p, b, e in real if p["kind"] != "ctrl" or _correct(p, b)]   # controls gated on base correct
    unchanged = sum(b["top"] == e["top"] for _, b, e in real) / len(real)
    gk = [(b, e) for p, b, e in zip(P, base, edited) if p["kind"] == "gk"]
    gk_unchanged = sum(b["top"] == e["top"] for b, e in gk) / len(gk)
    gates = dict(
        real_prompts=dict(n=len(real), unchanged=unchanged, threshold=GATE_UNCHANGED, pass_=unchanged >= GATE_UNCHANGED),
        neutral_kl=dict(n_texts=len(task["texts"]["neutral"]), mean_kl=kls["neutral"], threshold=GATE_KL, pass_=kls["neutral"] <= GATE_KL),
        general_knowledge=dict(n=len(gk), unchanged=gk_unchanged, threshold=GATE_GK, pass_=gk_unchanged >= GATE_GK),
    )
    gates = {k: {kk.rstrip("_"): vv for kk, vv in v.items()} for k, v in gates.items()}
    all_pass = all(g["pass"] for g in gates.values())
    scores = [c["score"] for c in comps.values() if c["score"] is not None]
    raw = sum(scores) / len(scores) if scores else 0.0
    det.update(components=comps, gates=gates, gates_pass=all_pass, software_overlay_kl=kls.get("software"),
               score_if_gates_pass=raw, reward=raw if all_pass else 0.0,
               reason="ok" if all_pass else "collateral gate failed: " + ", ".join(k for k, g in gates.items() if not g["pass"]),
               seconds=round(time.time() - t0, 1))
    return det


def write(out, reward, det):
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "reward.txt"), "w").write(f"{reward:.6f}\n")
    json.dump(det, open(os.path.join(out, "details.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--edit", default="/app/edit.pt")
    ap.add_argument("--out", default="/logs/verifier")
    ap.add_argument("--model-dir", default=os.environ.get("WSA_MODEL_DIR", "/models/qwen3.5-0.8b"))
    a = ap.parse_args()
    task = json.load(open(a.items))
    det = dict(task=task.get("id"), location=dict(lat=task.get("lat"), lon=task.get("lon")))
    if not os.path.exists(a.edit):
        det.update(reward=0.0, reason="no submission at " + a.edit)
        return write(a.out, 0.0, det)
    bad = check_model_files(a.model_dir, task.get("model_sha256"))
    if bad:
        det.update(reward=0.0, reason=f"model files differ from the pinned revision: {bad}")
        return write(a.out, 0.0, det)
    tok, model = load_model(a.model_dir)
    try:
        edit = read_edit(a.edit, param_table(model))
    except InvalidEdit as e:
        det.update(reward=0.0, reason=f"invalid submission: {e}")
        return write(a.out, 0.0, det)
    det.update(evaluate(model, tok, task, edit, log=lambda m: print(m, file=sys.stderr, flush=True)))
    write(a.out, det["reward"], det)
    print(json.dumps({k: det.get(k) for k in ("reward", "reason", "n_changed")}))


if __name__ == "__main__":
    main()
