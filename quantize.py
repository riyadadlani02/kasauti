"""Fake quantization: round weights through a quantized grid, keep the graph in place.

Operates on parameters rather than modules, because MoE experts are often a
single 3D weight bank rather than one Linear per expert.

Real GPTQ/AWQ kernels need CUDA. This runs anywhere, which is what makes the
component ablation affordable, and it isolates the arithmetic of quantization
from any one kernel's implementation. Vendor-quantized checkpoints (FP8, NVFP4,
MXFP4) are swept by loading them directly instead — see MODELS in sweep.py.
"""
from __future__ import annotations

import re

import torch

SKIP = re.compile(r"embed_tokens|lm_head|wte|shared\.weight")
GATE = re.compile(r"(?:^|\.)(?:gate|router)(?:\.|$)")
ATTN = re.compile(r"self_attn|attention|\.attn\.")
EXPERT = re.compile(r"expert|block_sparse_moe|\bmoe\b|mlp")


def classify(name: str) -> str:
    """Which component a parameter belongs to. Order matters: gate wins over expert."""
    if SKIP.search(name):
        return "skip"
    if GATE.search(name):
        return "gate"
    if ATTN.search(name):
        return "attention"
    if EXPERT.search(name):
        return "expert"
    return "other"


def fake_quant(w: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """Asymmetric round-to-nearest over groups of the input dimension."""
    orig_dtype, shape = w.dtype, w.shape
    x = w.float().reshape(-1, shape[-1])
    g = group_size if group_size and shape[-1] % group_size == 0 else shape[-1]
    x = x.reshape(-1, g)
    lo, hi = x.min(dim=-1, keepdim=True).values, x.max(dim=-1, keepdim=True).values
    qmax = 2 ** bits - 1
    scale = ((hi - lo) / qmax).clamp(min=1e-8)
    zero = (-lo / scale).round()
    q = ((x / scale) + zero).round().clamp(0, qmax)
    return ((q - zero) * scale).reshape(shape).to(orig_dtype)


@torch.no_grad()
def quantize_model_(model, bits: int, group_size: int = 128,
                    targets=("expert", "attention"), act_scales: dict | None = None,
                    alpha: float = 0.5) -> dict:
    """Quantize matching parameters in place. Returns per-component weight error.

    act_scales maps a decoder layer index to mean |activation| per input channel.
    When supplied, input-dim-matching weights are scaled before rounding and
    unscaled after — the AWQ idea, with a fixed alpha and no search.
    """
    stats, layer_of = {}, re.compile(r"layers\.(\d+)")
    for name, p in model.named_parameters():
        kind = classify(name)
        if kind not in targets or p.ndim < 2:
            continue
        w = p.data
        s = None
        if act_scales is not None:
            m = layer_of.search(name)
            a = act_scales.get(int(m.group(1))) if m else None
            if a is not None and a.numel() == w.shape[-1]:
                s = a.to(w.device).clamp(min=1e-5).pow(alpha)
                s = s / s.mean()
                w = w * s
        q = fake_quant(w, bits, group_size)
        if s is not None:
            q = q / s
        err = ((q.float() - p.data.float()).norm() / p.data.float().norm().clamp(min=1e-8)).item()
        d = stats.setdefault(kind, {"params": 0, "err": 0.0})
        d["params"] += 1
        d["err"] += err
        p.data.copy_(q.to(p.dtype))
    return {k: {"params": v["params"], "rel_weight_error": v["err"] / v["params"]}
            for k, v in stats.items()}


@torch.no_grad()
def activation_scales(model, input_ids) -> dict:
    """Mean |activation| per channel at each decoder layer input, for AWQ-lite."""
    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)$")
    scales, hooks = {}, []

    def hook(idx):
        def fn(_m, inputs, _out):
            x = inputs[0]
            if isinstance(x, torch.Tensor):
                v = x.detach().float().abs().reshape(-1, x.shape[-1]).mean(0).cpu()
                scales[idx] = v if idx not in scales else (scales[idx] + v) / 2
        return fn

    for name, mod in model.named_modules():
        m = layer_re.search(name)
        if m:
            hooks.append(mod.register_forward_hook(hook(int(m.group(1))), with_kwargs=False))
    try:
        model(input_ids.to(model.device))
    finally:
        for h in hooks:
            h.remove()
    return scales
