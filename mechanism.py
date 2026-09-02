"""Re-measure routing only, with each margin paired to the event it governs.

Routing needs one forward per text and no generation, so the whole config grid
is cheap to remeasure without re-running the probes. Results are merged back
into the sweep records by config name.
"""
from __future__ import annotations

import argparse
import json
import re

import torch

import kasauti as K
import quantize as Q
import sweep as S

FIELDS = ("flip_rate", "jaccard", "load_kl", "set_change_rate", "per_layer_flip_rate",
          "top1_flips_by_margin", "set_changes_by_margin", "worst_layers")


def routing_of(model_id, tok, texts, device, bits, group, targets, method):
    model = K.load_model(model_id, device)
    scales = None
    if method == "awq":
        ids = tok(texts[0], return_tensors="pt", truncation=True, max_length=512).input_ids
        scales = Q.activation_scales(model, ids)
    if targets:
        Q.quantize_model_(model, bits, group, targets, act_scales=scales)
    n, k = K._num_experts(model.config), K._top_k(model.config)
    runs = [K.keep_router_outputs(K.capture_routing(
        model, tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids), n, k)
        for t in texts]
    out = K.cat_runs(runs)
    S.free(model)
    return out, n, k


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--results", default="results.jsonl")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    mid = S.MODELS.get(args.model, args.model)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    texts = K.DEFAULT_CALIBRATION

    base, n, k = routing_of(mid, tok, texts, args.device, 16, 0, (), "none")
    print(f"baseline captured: {len(base)} gates, {n} experts, top-{k}", flush=True)

    updates = {}
    for name, bits, group, targets, method in S.CONFIGS:
        if not targets:
            continue
        q, _, _ = routing_of(mid, tok, texts, args.device, bits, group, targets, method)
        rep = K.compare(base, q, n, k)
        updates[name] = {f: rep[f] for f in FIELDS}
        print(f"{name}: top1 flip {rep['flip_rate']:.3f}, set change {rep['set_change_rate']:.3f}, "
              f"top1 by margin {[round(x, 3) for x in rep['top1_flips_by_margin']]}, "
              f"set by margin {[round(x, 3) for x in rep['set_changes_by_margin']]}", flush=True)

    try:
        rows = [json.loads(l) for l in open(args.results) if l.strip()]
    except FileNotFoundError:
        # Routing alone is worth measuring where a full sweep will not fit.
        rows = [{"model": mid, "config": c, "bits": b, "method": m, "targets": list(t)}
                for c, b, _g, t, m in S.CONFIGS if t]
    with open(args.results, "w") as f:
        for r in rows:
            r.pop("flips_by_margin_quartile", None)
            f.write(json.dumps({**r, **updates.get(r["config"], {})}) + "\n")
    print(f"wrote {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
