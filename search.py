"""Search for the smallest quantization config that keeps routing intact.

The loop: propose lowering one component's bit-width, measure router divergence
(cheap — one forward per calibration text, no generation), keep the step that
buys the most model size per unit of divergence, repeat until the divergence
budget is spent.

Periodically it stops trusting the cheap signal and pays for a real probe run.
If the agentic score has fallen below the floor while divergence said it was
fine, the step is reverted and the budget tightened — the cheap signal is the
search driver, the probe suite is the auditor. A search that only ever consults
its own proxy will find that proxy's blind spots, not a better model.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

import kasauti as K
import probes as P
import quantize as Q
import sweep as S

COMPONENTS = ("gate", "attention", "expert")
LADDER = (16, 8, 6, 5, 4, 3, 2)  # 16 means "left alone"


def component_sizes(model) -> dict:
    """Parameters per component, so bits saved is weighted by what it actually saves."""
    sizes = {}
    for name, p in model.named_parameters():
        kind = Q.classify(name)
        if kind in COMPONENTS and p.ndim >= 2:
            sizes[kind] = sizes.get(kind, 0) + p.numel()
    return sizes


def model_bits(state: dict, sizes: dict) -> float:
    """Weighted mean bit-width — the thing being minimised."""
    total = sum(sizes.values())
    return sum(sizes[c] * state[c] for c in sizes) / total


def apply_state(model, state: dict, group: int, scales=None):
    for c, bits in state.items():
        if bits < 16:
            Q.quantize_model_(model, bits, group, (c,), act_scales=scales)


class Evaluator:
    """Loads the model fresh per config, because fake quantization is destructive."""

    def __init__(self, model_id, device, group, texts, pattern):
        self.model_id, self.device, self.group = model_id, device, group
        self.texts, self.pattern = texts, pattern
        self.cache, self.evals = {}, 0
        model = K.load_model(model_id, device)
        self.n, self.k = K._num_experts(model.config), K._top_k(model.config)
        self.sizes = component_sizes(model)
        self.base = self._routing(model)
        S.free(model)

    def _routing(self, model):
        from transformers import AutoTokenizer
        tok = getattr(self, "tok", None) or AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True)
        self.tok = tok
        runs = [K.keep_router_outputs(K.capture_routing(model, tok(
            t, return_tensors="pt", truncation=True, max_length=512).input_ids,
            self.pattern), self.n, self.k) for t in self.texts]
        return K.cat_runs(runs)

    def divergence(self, state: dict) -> dict:
        if all(b == 16 for b in state.values()):
            return {"set_change_rate": 0.0, "flip_rate": 0.0}
        key = tuple(sorted(state.items()))
        if key in self.cache:
            return self.cache[key]
        model = K.load_model(self.model_id, self.device)
        apply_state(model, state, self.group)
        rep = K.compare(self.base, self._routing(model), self.n, self.k)
        S.free(model)
        self.evals += 1
        self.cache[key] = rep
        return rep

    def probe(self, state: dict, items) -> dict:
        model = K.load_model(self.model_id, self.device)
        apply_state(model, state, self.group)
        res = P.aggregate(P.run_items(model, self.tok, items, batch_size=16))
        S.free(model)
        return res


def agentic_mean(scores: dict, baseline: dict, floor=0.25) -> float:
    usable = [p for p in P.LAYER if P.LAYER[p] == 2 and baseline.get(p, 0) >= floor]
    vals = [scores[p] / baseline[p] for p in usable if p in scores]
    return sum(vals) / len(vals) if vals else float("nan")


def search(ev: Evaluator, budget: float, items, baseline_probes: dict,
           validate_every: int, min_agentic: float, log) -> dict:
    state = {c: 16 for c in COMPONENTS if c in ev.sizes}
    history, accepted = [], 0

    while True:
        best = None
        for c in state:
            i = LADDER.index(state[c])
            if i + 1 >= len(LADDER):
                continue
            trial = {**state, c: LADDER[i + 1]}
            rep = ev.divergence(trial)
            if not math.isfinite(rep["set_change_rate"]):
                # Never let an unreadable signal drive an accept: that is how a
                # search ends up optimising its own instrument failure.
                log(f"  skip {c}->{LADDER[i+1]}: divergence unreadable")
                continue
            if rep["set_change_rate"] > budget:
                log(f"  reject {c}->{LADDER[i+1]}: divergence "
                    f"{rep['set_change_rate']:.3f} over budget {budget:.3f}")
                continue
            saved = model_bits(state, ev.sizes) - model_bits(trial, ev.sizes)
            cost = max(rep["set_change_rate"] - ev.divergence(state)["set_change_rate"], 1e-6)
            score = saved / cost
            log(f"  try {c}->{LADDER[i+1]}: divergence {rep['set_change_rate']:.3f}, "
                f"saves {saved:.2f} bits, ratio {score:.1f}")
            if best is None or score > best[0]:
                best = (score, c, LADDER[i + 1], trial, rep)

        if best is None:
            log("no step left within budget")
            break

        _, c, bits, trial, rep = best
        state, accepted = trial, accepted + 1
        history.append({"step": accepted, "component": c, "bits": bits, "state": dict(state),
                        "set_change_rate": rep["set_change_rate"], "flip_rate": rep["flip_rate"],
                        "model_bits": model_bits(state, ev.sizes)})
        log(f"accept {c} -> {bits}  state={state}  "
            f"mean bits {model_bits(state, ev.sizes):.2f}  divergence {rep['set_change_rate']:.3f}")

        if validate_every and accepted % validate_every == 0:
            scores = ev.probe(state, items)
            score = agentic_mean(scores, baseline_probes)
            history[-1]["validated_agentic"] = score
            log(f"  VALIDATE: agentic {score:.3f} (floor {min_agentic})  {scores}")
            if score < min_agentic:
                budget *= 0.6
                state = history[-2]["state"] if len(history) > 1 else {c: 16 for c in state}
                log(f"  the cheap signal was wrong here: revert to {state}, "
                    f"tighten budget to {budget:.3f}")
                history[-1]["reverted"] = True
    return {"state": state, "history": history, "budget": budget,
            "model_bits": model_bits(state, ev.sizes), "evaluations": ev.evals}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--budget", type=float, default=0.20,
                   help="max fraction of tokens allowed to change expert set")
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--validate-every", type=int, default=2, help="0 disables the auditor")
    p.add_argument("--min-agentic", type=float, default=0.95)
    p.add_argument("--out", default="search.json")
    args = p.parse_args(argv)

    mid = S.MODELS.get(args.model, args.model)
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    ev = Evaluator(mid, args.device, args.group, K.DEFAULT_CALIBRATION, K.GATE_NAME)
    log(f"{mid}: {ev.n} experts, top-{ev.k}, sizes "
        f"{ {c: f'{n/1e6:.0f}M' for c, n in ev.sizes.items()} }")

    items = [i for i in P.all_items() if P.LAYER[i.probe] == 2]
    base_probes = ev.probe({c: 16 for c in ev.sizes}, items) if args.validate_every else {}
    if base_probes:
        log(f"baseline probes {base_probes}")

    result = search(ev, args.budget, items, base_probes,
                    args.validate_every, args.min_agentic, log)
    result.update(model=mid, seconds=round(time.time() - t0), group=args.group)
    json.dump(result, open(args.out, "w"), indent=2)

    log(f"\nbest config: {result['state']}")
    log(f"mean bit-width {result['model_bits']:.2f} "
        f"(from 16.0, {100 * (1 - result['model_bits'] / 16):.0f}% smaller)")
    log(f"{result['evaluations']} evaluations in {result['seconds']}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
