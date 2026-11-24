"""Optional helpers for Qwen3.5-0.8B on CPU (you may ignore them and write your own; verify anything you rely on).

- load(): float32 CPU load from /models/qwen3.5-0.8b with a dtype assert (transformers can silently keep bf16),
  plus an exact, much faster replacement for the depthwise Conv1d of the linear-attention layers.
- score(): next-token log-probs of short candidate answers after a prompt, with the shared few-shot prefix cached.
  Only prompts whose suffixes have equal token length are batched: padding changes log-probs on this model.
"""
import copy
import os

import torch

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen3.5-0.8b")
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))


class FastDepthwiseConv(torch.nn.Module):
    """Same math as nn.Conv1d(C, C, K, groups=C, padding=K-1, bias=False), as a sum of K shifted products."""

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


def load(model_dir=MODEL_DIR, fast_conv=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32)
    model.eval()
    model.to(torch.float32)
    assert all(p.dtype == torch.float32 for p in model.parameters()), "dtype not applied"
    if fast_conv:
        for m in model.modules():
            if hasattr(m, "conv1d") and isinstance(m.conv1d, torch.nn.Conv1d):
                m.conv1d = FastDepthwiseConv(m.conv1d)
    return tok, model


def _expand_cache(cache, B):
    c = copy.deepcopy(cache)
    for layer in c.layers:
        for k, v in list(vars(layer).items()):
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1 and B > 1:
                setattr(layer, k, v.expand(B, *v.shape[1:]).contiguous())
    return c


@torch.no_grad()
def score(model, tok, prefix, suffixes, cands, bs=32):
    """Log-probs of each candidate's first token after prefix + suffix, for every suffix.
    cands: list of strings with a leading space (distinct first tokens). Returns (list of lists of log-probs,
    list of greedy token strings). The prefix must end on a token boundary (e.g. a newline)."""
    ct = [tok.encode(c, add_special_tokens=False)[0] for c in cands]
    assert len(set(ct)) == len(ct), "candidates share a first token"
    pids = tok.encode(prefix, add_special_tokens=False)
    cache = model.model(input_ids=torch.tensor([pids]), use_cache=True).past_key_values
    groups = {}
    for i, s in enumerate(suffixes):
        full = tok.encode(prefix + s, add_special_tokens=False)
        assert full[:len(pids)] == pids, "prefix does not end on a token boundary"
        groups.setdefault(len(full) - len(pids), []).append((i, full[len(pids):]))
    out, top = [None] * len(suffixes), [None] * len(suffixes)
    for _, g in groups.items():
        for b in range(0, len(g), bs):
            chunk = g[b:b + bs]
            c = _expand_cache(cache, len(chunk))
            h = model.model(input_ids=torch.tensor([s for _, s in chunk]), past_key_values=c, use_cache=True).last_hidden_state[:, -1]
            lp = torch.log_softmax(model.lm_head(h).float(), -1)
            for r, (i, _) in enumerate(chunk):
                out[i] = [float(lp[r, t]) for t in ct]
                top[i] = tok.decode([int(lp[r].argmax())])
    return out, top
