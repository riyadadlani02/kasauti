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
from memory import Memory, key

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

    def __init__(self, model_id, device, group, texts, pattern, mem=None):
        self.model_id, self.device, self.group, self.mem = model_id, device, group, mem
        self.texts, self.pattern = texts, pattern
        self.cache, self.evals = {}, 0
        model = K.load_model(model_id, device)
        self.n, self.k = K._num_experts(model.config), K._top_k(model.config)
        self.sizes = component_sizes(model)
        self.base, self.base_layers = self._capture(model)
        S.free(model)

    def _capture(self, model):
        from transformers import AutoTokenizer
        tok = getattr(self, "tok", None) or AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True)
        self.tok = tok
        runs, layers = [], {}
        for t in self.texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids
            store, lay = K.capture_routing(model, ids, self.pattern, with_layers=256)
            runs.append(K.keep_router_outputs(store, self.n, self.k))
            for i, v in lay.items():
                layers.setdefault(i, []).append(v)
        return K.cat_runs(runs), {i: torch.cat(v) for i, v in layers.items()}

    def divergence(self, state: dict) -> dict:
        if all(b == 16 for b in state.values()):
            return {"set_change_rate": 0.0, "flip_rate": 0.0, "recon_error": 0.0, "cost": 0.0}
        k = key(state, self.group)
        if k in self.cache:
            return self.cache[k]
        if self.mem:
            past = self.mem.get(self.model_id, state, self.group)
            if "cost" in past:
                self.cache[k] = past
                return past
        model = K.load_model(self.model_id, self.device)
        apply_state(model, state, self.group)
        routing, layers = self._capture(model)
        rep = K.compare(self.base, routing, self.n, self.k)
        rep["recon_error"] = K.recon_error(self.base_layers, layers)
        # Both signals separate healthy from damaged configs at 84% in the sweep
        # and cannot be told apart, so the binding one is whichever is closer to
        # its own cut point.
        rep["cost"] = max(rep["set_change_rate"] / K.ROUTING_HIGH,
                          rep["recon_error"] / K.RECON_HIGH)
        S.free(model)
        self.evals += 1
        self.cache[k] = rep
        if self.mem:
            self.mem.put(self.model_id, state, self.group, **{
                f: rep[f] for f in ("set_change_rate", "flip_rate", "recon_error", "cost")})
        return rep

    def probe(self, state: dict, items) -> dict:
        model = K.load_model(self.model_id, self.device)
        apply_state(model, state, self.group)
        res = P.aggregate(P.run_items(model, self.tok, items, batch_size=16))
        S.free(model)
        return res


# The auditor must not grade on a probe that can improve as the model degrades.
# Calibration rewards abstention and a damaged model hedges more (RESULTS.md), so
# including it would let real damage hide behind a rising score.
CONFOUNDED = {"calibration"}

# Bumped whenever the audit changes what it grades or how. Memory from an older
# judge is not evidence about this one, so its verdicts are ignored.
JUDGE = "v3-capped-ratios"


def agentic_mean(scores: dict, baseline: dict, floor=0.25) -> float:
    """Mean of per-probe ratios, each capped at 1.0.

    Without the cap a probe scoring above baseline pays for another probe's real
    loss. Measured: a config whose format stability fell 18% and long-horizon
    adherence fell 7% audited at 1.103, because one low-baseline probe went 0.50
    to 0.83 and contributed a 1.67 ratio. An audit measures damage; scoring above
    baseline is not evidence of health somewhere else.
    """
    usable = [p for p in P.LAYER
              if P.LAYER[p] == 2 and p not in CONFOUNDED and baseline.get(p, 0) >= floor]
    vals = [min(scores[p] / baseline[p], 1.0) for p in usable if p in scores]
    return sum(vals) / len(vals) if vals else float("nan")


def search(ev: Evaluator, budget: float, items, baseline_probes: dict,
           validate_every: int, min_agentic: float, log, mem=None, model="",
           margin: float = 0.02) -> dict:  # noqa: C901
    state = {c: 16 for c in COMPONENTS if c in ev.sizes}
    history, accepted, cadence = [], 0, validate_every
    # Where to fall back to. Stepping back blindly can land on a config a
    # previous audit already rejected, so only audited-good states qualify.
    last_good = {c: 16 for c in COMPONENTS if c in ev.sizes}
    banned = set(mem.rejected(model, JUDGE)) if mem else set()
    if banned:
        log(f"memory: {len(banned)} config(s) already disproved, not proposing them again")

    while True:
        best = None
        for c in state:
            i = LADDER.index(state[c])
            if i + 1 >= len(LADDER):
                continue
            trial = {**state, c: LADDER[i + 1]}
            if key(trial, ev.group) in banned:
                log(f"  skip {c}->{LADDER[i+1]}: a previous run's audit rejected it")
                continue
            rep = ev.divergence(trial)
            if not math.isfinite(rep["cost"]):
                # Never let an unreadable signal drive an accept: that is how a
                # search ends up optimising its own instrument failure.
                log(f"  skip {c}->{LADDER[i+1]}: divergence unreadable")
                continue
            if rep["cost"] > budget:
                log(f"  reject {c}->{LADDER[i+1]}: cost {rep['cost']:.2f} over budget "
                    f"{budget:.2f} (routing {rep['set_change_rate']:.3f}, "
                    f"recon {rep['recon_error']:.3f})")
                continue
            saved = model_bits(state, ev.sizes) - model_bits(trial, ev.sizes)
            cost = max(rep["cost"] - ev.divergence(state)["cost"], 1e-6)
            score = saved / cost
            log(f"  try {c}->{LADDER[i+1]}: cost {rep['cost']:.2f} "
                f"(routing {rep['set_change_rate']:.3f}, recon {rep['recon_error']:.3f}), "
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
                        "recon_error": rep["recon_error"], "cost": rep["cost"],
                        "model_bits": model_bits(state, ev.sizes)})
        log(f"accept {c} -> {bits}  state={state}  "
            f"mean bits {model_bits(state, ev.sizes):.2f}  cost {rep['cost']:.2f}")

        if cadence and accepted % cadence == 0:
            remembered = mem.get(model, state, ev.group) if mem else {}
            if remembered.get("judge") != JUDGE:
                remembered = {}  # graded by a different judge, so not evidence here
            if "agentic" in remembered:
                scores, score = remembered.get("scores", {}), remembered["agentic"]
                log(f"  VALIDATE (from memory): agentic {score:.3f}")
            else:
                scores = ev.probe(state, items)
                score = agentic_mean(scores, baseline_probes)
                if mem:
                    mem.put(model, state, ev.group, agentic=score, scores=scores, judge=JUDGE,
                            audit="passed" if score >= min_agentic - margin else "failed",
                            **{k2: rep[k2] for k2 in ("set_change_rate", "recon_error", "cost")})
            history[-1]["validated_agentic"] = score
            log(f"  VALIDATE: agentic {score:.3f} (floor {min_agentic}, "
                f"tolerance {margin})  {scores}")
            if score >= min_agentic - margin:
                last_good = dict(state)
            if score < min_agentic - margin:
                # Ban the config the audit disproved, but do NOT shrink the cost
                # budget: the cost metric is not what failed, the config is, and
                # tightening it walls off every config beyond this one -- including
                # better ones only reachable through here. Verify every step
                # instead, which is the honest response to the cheap signal having
                # just been shown wrong.
                banned.add(key(state, ev.group))
                cadence = 1
                if mem:
                    mem.put(model, state, ev.group, audit="failed", judge=JUDGE)
                state = dict(last_good)
                log(f"  the cheap signal was wrong here: revert to the last audited "
                    f"config {state}, and audit every step from now on")
                history[-1]["reverted"] = True
    if validate_every:  # the last accepted step is otherwise never audited
        scores = ev.probe(state, items)
        final = agentic_mean(scores, baseline_probes)
        log(f"FINAL VALIDATE: agentic {final:.3f}  {scores}")
        if final < min_agentic - margin:
            state = dict(last_good)
            log(f"  final state failed its audit, falling back to {state}")
        return {"state": state, "history": history, "budget": budget, "final_agentic": final,
                "model_bits": model_bits(state, ev.sizes), "evaluations": ev.evals}
    return {"state": state, "history": history, "budget": budget,
            "model_bits": model_bits(state, ev.sizes), "evaluations": ev.evals}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--budget", type=float, default=1.0,
                   help="max damage cost, where 1.0 is the sweep's HIGH cut point")
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--validate-every", type=int, default=2, help="0 disables the auditor")
    p.add_argument("--min-agentic", type=float, default=0.95)
    p.add_argument("--audit-margin", type=float, default=0.02,
                   help="how far under the floor counts as real, not measurement noise")
    p.add_argument("--out", default="search.json")
    p.add_argument("--memory", default="memory.json", help="carry knowledge between runs")
    p.add_argument("--forget", action="store_true", help="ignore what past runs learned")
    args = p.parse_args(argv)

    mid = S.MODELS.get(args.model, args.model)
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    mem = None if args.forget else Memory(args.memory)
    if mem:
        log(f"memory: {mem.summary(mid)}")
        args.budget = mem.learned_budget(mid, args.budget)
    ev = Evaluator(mid, args.device, args.group, K.DEFAULT_CALIBRATION, K.GATE_NAME, mem)
    log(f"{mid}: {ev.n} experts, top-{ev.k}, sizes "
        f"{ {c: f'{n/1e6:.0f}M' for c, n in ev.sizes.items()} }")

    items = [i for i in P.all_items() if P.LAYER[i.probe] == 2] + P.generated_items()
    base_state = {c: 16 for c in ev.sizes}
    base_probes = {}
    if args.validate_every:
        remembered = mem.get(mid, base_state, args.group) if mem else {}
        if remembered.get("judge") != JUDGE:
            remembered = {}
        base_probes = remembered.get("scores") or ev.probe(base_state, items)
        if mem and not remembered.get("scores"):
            mem.put(mid, base_state, args.group, scores=base_probes, agentic=1.0,
                    audit="passed", judge=JUDGE)
        log(f"baseline probes {'(from memory) ' if remembered.get('scores') else ''}"
            f"{ {k: round(v, 3) for k, v in base_probes.items()} }")

    result = search(ev, args.budget, items, base_probes,
                    args.validate_every, args.min_agentic, log, mem, mid, args.audit_margin)
    result.update(model=mid, seconds=round(time.time() - t0), group=args.group)
    json.dump(result, open(args.out, "w"), indent=2)

    log(f"\nbest config: {result['state']}")
    log(f"mean bit-width {result['model_bits']:.2f} "
        f"(from 16.0, {100 * (1 - result['model_bits'] / 16):.0f}% smaller)")
    log(f"{result['evaluations']} model evaluations in {result['seconds']}s -> {args.out}")
    if mem:
        log(f"memory now holds: {mem.summary(mid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
