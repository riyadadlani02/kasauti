"""Run the quantization grid: probes, perplexity, reconstruction error, router divergence.

One JSON record per config, appended as it completes, so a killed run keeps
what it already measured.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time

import torch

import kasauti as K
import probes as P
import quantize as Q

LAYER_OUT = re.compile(r"(?:^|\.)layers\.(\d+)$")

# Subjects. The 30B-class entries are the real study; the small ones are what
# fits on a laptop and are used to validate the pipeline end to end.
MODELS = {
    "granite-1b-a400m": "ibm-granite/granite-3.0-1b-a400m-instruct",
    "olmoe-1b-7b": "allenai/OLMoE-1B-7B-0125-Instruct",
    "sarvam-30b": "sarvamai/sarvam-30b",
    "sarvam-105b": "sarvamai/sarvam-105b",
    "nemotron-3-nano-30b": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}

# Vendor-quantized checkpoints, swept by loading them rather than faking the
# arithmetic. This is the config a serving team actually deploys.
VENDOR_PAIRS = {
    "sarvam-30b": [("sarvamai/sarvam-30b-fp8", "fp8")],
    "sarvam-105b": [("sarvamai/sarvam-105b-fp8", "fp8")],
    "nemotron-3-nano-30b": [("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8", "fp8"),
                            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", "nvfp4")],
}

# bits, group, targets, method. Bit-width sweep first, then component ablation.
CONFIGS = [
    ("bf16", 16, 0, (), "none"),
    ("w8-ea", 8, 128, ("expert", "attention"), "rtn"),
    ("w6-ea", 6, 128, ("expert", "attention"), "rtn"),
    ("w4-ea", 4, 128, ("expert", "attention"), "rtn"),
    ("w4-ea-awq", 4, 128, ("expert", "attention"), "awq"),
    ("w3-ea", 3, 128, ("expert", "attention"), "rtn"),
    ("w4-experts", 4, 128, ("expert",), "rtn"),
    ("w4-attention", 4, 128, ("attention",), "rtn"),
    ("w4-all", 4, 128, ("expert", "attention", "gate"), "rtn"),
]


@torch.no_grad()
def capture(model, tok, texts, pattern, max_len=512, layer_tokens=256):
    """Routing, decoder-layer outputs and perplexity from one forward per text."""
    n_experts, k = K._num_experts(model.config), K._top_k(model.config)
    runs, layer_out, nll, ntok = [], {}, 0.0, 0

    for t_i, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids
        store, hooks = {}, []

        def layer_hook(idx):
            def fn(_m, _i, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(h, torch.Tensor):
                    store[idx] = h.detach().reshape(-1, h.shape[-1])[:layer_tokens].half().cpu()
            return fn

        for name, mod in model.named_modules():
            m = LAYER_OUT.search(name)
            if m:
                hooks.append(mod.register_forward_hook(layer_hook(int(m.group(1)))))
        try:
            run = K.capture_routing(model, ids, pattern)
            out = model(ids.to(model.device), labels=ids.to(model.device))
        finally:
            for h in hooks:
                h.remove()

        nll += float(out.loss) * (ids.numel() - 1)
        ntok += ids.numel() - 1
        runs.append(K.keep_router_outputs(run, n_experts, k))
        for idx, v in store.items():
            layer_out.setdefault(idx, []).append(v)

    return (K.cat_runs(runs),
            {i: torch.cat(v) for i, v in layer_out.items()},
            float(torch.tensor(nll / max(ntok, 1)).exp()))


def recon_error(base_out: dict, quant_out: dict) -> float:
    """Mean relative L2 between decoder layer outputs. The standard competing predictor."""
    errs = []
    for i in sorted(set(base_out) & set(quant_out)):
        b, q = base_out[i].float(), quant_out[i].float()
        n = min(len(b), len(q))
        errs.append((q[:n] - b[:n]).norm().item() / max(b[:n].norm().item(), 1e-8))
    return sum(errs) / max(len(errs), 1)


def evaluate(model, tok, texts, pattern, items, batch_size, max_new_tokens):
    routing, layers, ppl = capture(model, tok, texts, pattern)
    results = P.run_items(model, tok, items, max_new_tokens=max_new_tokens, batch_size=batch_size)
    return routing, layers, ppl, results


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model", help="key from MODELS, or a hub id / local path")
    p.add_argument("--out", default="results.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--configs", help="comma-separated subset of config names")
    p.add_argument("--vendor", action="store_true", help="also sweep vendor-quantized checkpoints")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--gate-pattern")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    base_id = MODELS.get(args.model, args.model)
    pattern = re.compile(args.gate_pattern) if args.gate_pattern else K.GATE_NAME
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    texts, items = K.DEFAULT_CALIBRATION, P.all_items()
    configs = [c for c in CONFIGS if not args.configs or c[0] in args.configs.split(",")]

    print(f"{base_id}: {len(items)} probe items, {len(texts)} calibration texts, "
          f"{len(configs)} configs", flush=True)

    t0 = time.time()
    model = K.load_model(base_id, args.device)
    base_routing, base_layers, base_ppl, base_results = evaluate(
        model, tok, texts, pattern, items, args.batch_size, args.max_new_tokens)
    n_experts, k = K._num_experts(model.config), K._top_k(model.config)
    base_scores = P.aggregate(base_results)
    print(f"baseline {base_scores} ppl={base_ppl:.2f} ({time.time() - t0:.0f}s)", flush=True)
    free(model)

    out, raw = open(args.out, "a"), open(args.out + ".outputs.jsonl", "a")
    dump = lambda cfg, rs: (raw.write("\n".join(json.dumps({"config": cfg, **r}) for r in rs) + "\n"),
                            raw.flush())
    dump("bf16", base_results)
    record = dict(model=base_id, config="bf16", bits=16, method="none", targets=[],
                  ppl=base_ppl, probes=base_scores, flip_rate=0.0, jaccard=0.0, load_kl=0.0,
                  recon_error=0.0, weight_error={}, seconds=round(time.time() - t0))
    out.write(json.dumps(record) + "\n")
    out.flush()

    todo = [c for c in configs if c[0] != "bf16"]
    vendor = [(vid, meth) for vid, meth in VENDOR_PAIRS.get(args.model, [])] if args.vendor else []

    for name, bits, group, targets, method in todo:
        t1 = time.time()
        model = K.load_model(base_id, args.device)
        scales = None
        if method == "awq":
            ids = tok(texts[0], return_tensors="pt", truncation=True, max_length=512).input_ids
            scales = Q.activation_scales(model, ids)
        werr = Q.quantize_model_(model, bits, group, targets, act_scales=scales)
        routing, layers, ppl, results = evaluate(
            model, tok, texts, pattern, items, args.batch_size, args.max_new_tokens)
        report = K.compare(base_routing, routing, n_experts, k)
        rec = dict(model=base_id, config=name, bits=bits, method=method, targets=list(targets),
                   ppl=ppl, probes=P.aggregate(results), recon_error=recon_error(base_layers, layers),
                   weight_error=werr, seconds=round(time.time() - t1), **{
                       k2: report[k2] for k2 in ("flip_rate", "jaccard", "load_kl", "set_change_rate",
                                                 "per_layer_flip_rate", "top1_flips_by_margin",
                                                 "set_changes_by_margin", "worst_layers")})
        out.write(json.dumps(rec) + "\n")
        out.flush()
        dump(name, results)
        print(f"{name}: flip={rec['flip_rate']:.3f} ppl={ppl:.2f} "
              f"probes={ {p: round(v, 3) for p, v in rec['probes'].items()} } "
              f"({rec['seconds']}s)", flush=True)
        free(model)

    for vid, meth in vendor:
        t1 = time.time()
        model = K.load_model(vid, args.device)
        routing, layers, ppl, results = evaluate(
            model, tok, texts, pattern, items, args.batch_size, args.max_new_tokens)
        report = K.compare(base_routing, routing, n_experts, k)
        rec = dict(model=vid, config=meth, bits={"fp8": 8, "nvfp4": 4, "mxfp4": 4}.get(meth, 0),
                   method=f"vendor-{meth}", targets=["vendor"], ppl=ppl,
                   probes=P.aggregate(results), recon_error=recon_error(base_layers, layers),
                   weight_error={}, seconds=round(time.time() - t1),
                   **{k2: report[k2] for k2 in ("flip_rate", "jaccard", "load_kl", "set_change_rate",
                                                "per_layer_flip_rate", "top1_flips_by_margin",
                                                "set_changes_by_margin", "worst_layers")})
        out.write(json.dumps(rec) + "\n")
        out.flush()
        dump(meth, results)
        print(f"{vid}: flip={rec['flip_rate']:.3f} ppl={ppl:.2f}", flush=True)
        free(model)

    out.close()
    raw.close()
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
