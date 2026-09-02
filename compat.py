"""Does the instrumentation attach to a given architecture, without downloading weights?

Builds the model on the meta device from its config alone, so gate discovery,
expert counts and component classification can be checked for a 105B model on a
laptop. It cannot check routing values — that needs real weights and a GPU box.
"""
from __future__ import annotations

import argparse

import torch

import kasauti as K
import quantize as Q
import sweep as S


def check(model_id: str) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    gates = K.find_gates(model)
    kinds = {}
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            kinds[Q.classify(name)] = kinds.get(Q.classify(name), 0) + 1
    router_w = [n for n, m in gates.items()
                if getattr(m, "weight", None) is not None
                or getattr(getattr(m, "layer", None), "weight", None) is not None]
    return {
        "arch": cfg.architectures[0] if cfg.architectures else "?",
        "experts": K._num_experts(cfg), "top_k": K._top_k(cfg),
        "layers": getattr(cfg, "num_hidden_layers", "?"),
        "gates": len(gates), "gates_with_weight": len(router_w),
        "params_by_component": kinds,
        "quantized": (cfg.quantization_config or {}).get("quant_method")
        if getattr(cfg, "quantization_config", None) else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("models", nargs="*", default=list(S.MODELS.values()))
    args = p.parse_args(argv)
    for mid in args.models:
        try:
            r = check(mid)
            ok = "OK " if r["gates"] and r["gates_with_weight"] == r["gates"] else "PARTIAL"
            print(f"{ok} {mid}\n     {r}", flush=True)
        except Exception as e:  # architectures we cannot even construct are the real finding
            print(f"FAIL {mid}\n     {type(e).__name__}: {str(e)[:160]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
