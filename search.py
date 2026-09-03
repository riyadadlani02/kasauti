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
import os
import tempfile
import time

import torch

import judge as J
import kasauti as K
import policy as PO
import probes as P
import quantize as Q
import sweep as S
from memory import Memory, key

COMPONENTS = ("gate", "attention", "expert")
# The starting ladder; 16 means "left alone". The agent refines it when the audit
# rejects a component's only available move -- see policy.py.
LADDER = (16, 8, 6, 5, 4, 3, 2)


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

    def probe(self, state: dict, items):
        """Returns (per-probe scores, items behind each). The counts are half the
        measurement: a mean of six items is not the same evidence as a mean of
        sixty, and the judge has to be able to tell."""
        model = K.load_model(self.model_id, self.device)
        apply_state(model, state, self.group)
        res = P.run_items(model, self.tok, items, batch_size=16)
        S.free(model)
        return P.aggregate(res), P.counts(res)


def audit_state(ev, state, items, base, spec, mem, model, log, floor, margin, z, rep=None):
    """Measure, and know how well it measured.

    Returns (scores, counts, score, se, decided). An audit is a mean over a
    handful of items, and when the verdict lands inside that mean's own error
    the agent does not guess. It first asks whether buying items could even
    help: only the generated families can be made more of, so if the error left
    after those go to zero is still bigger than the distance to the line, a
    re-measurement is compute spent for nothing and it says so instead.
    """
    remembered = mem.get(model, state, ev.group) if mem else {}
    # Raw scores outlive any judge, so they are reusable whatever graded them --
    # but not whatever measured them. A score from a different probe suite is a
    # different measurement and gets re-taken.
    scores, counts = remembered.get("scores"), remembered.get("counts")
    if scores and not set(suite_of(items)) <= set(scores):
        scores = None  # measured before the suite grew, so it does not cover it
    if scores and counts:
        log("  VALIDATE (probe scores from memory)")
    else:
        scores, counts = ev.probe(state, items)
        if mem:
            extra = {k: rep[k] for k in ("set_change_rate", "recon_error", "cost")} if rep else {}
            mem.put(model, state, ev.group, scores=scores, counts=counts,
                    suite=suite_of(items), **extra)

    s = J.score(scores, base["scores"], spec)
    se = J.resolution(scores, base["scores"], spec, counts, base["counts"])
    if J.decided(s, se, floor, margin, z):
        return scores, counts, s, se, True

    gap = abs(s - (floor - margin)) / max(z, 1e-9)
    fixed = J.irreducible(scores, base["scores"], spec, counts, base["counts"])
    need = J.items_needed(scores, base["scores"], spec, counts, gap, base["counts"])
    log(f"  UNRESOLVED: {s:.3f} +/- {se:.3f}, and the line is {gap:.3f} away")
    if fixed >= gap:
        log(f"  buying items cannot fix this: with the generated families grown without "
            f"limit the error only reaches {fixed:.3f}. The fixed probe lists are the "
            f"bottleneck, and deciding this needs {need}")
    else:
        log(f"  deciding this needs about {need} items")
    return scores, counts, s, se, False


def build_items(n_per_family: int) -> list:
    return ([i for i in P.all_items() if P.LAYER[i.probe] == 2]
            + P.generated_items(n_per_family))


def suite_of(items) -> list:
    return sorted({i.probe for i in items})


def replenish(model: str, device: str, memory_path: str, log) -> list:
    """The audit could not resolve a verdict, so grow the suite that can.

    Until now the judge could only shrink its suite. This is the other
    direction: the probes carrying the audit's error are fixed hand-written
    lists, and the vetter -- told which those are -- admits the generated
    families that restate them, which are the only ones the agent can make more
    of. Re-vetting reuses candidate scores already on disk, so it usually costs
    no model evaluations at all.
    """
    import vet_probes
    before = set(json.load(open(P.GENERATED))["keep"]) if os.path.exists(P.GENERATED) else set()
    vet_probes.main([model, "--device", device, "--memory", memory_path])
    after = set(json.load(open(P.GENERATED))["keep"])
    gained = sorted(after - before)
    log(f"  REPLENISH: admitted {gained}" if gained else
        "  REPLENISH: nothing new survived vetting; the suite is what it is")
    return gained


def improve_policy(mem, model: str, group: int, pol: dict, log) -> dict:
    """Turn the agent's audit record on how it searches, not just how it grades."""
    rows = J.records({k: v for k, v in mem.scored(model).items()
                      if k.endswith(f"@g{group}")})
    new, applied = PO.improve(pol, rows)
    if not applied:
        return pol
    for a in applied:
        log(f"  POLICY amends [{a['kind']}] {a['why']}")
    log(f"  POLICY {pol['version']} -> {new['version']}, ladder {new['ladder']}, "
        f"may cross {new['pass_through']} failed config(s)")
    mem.put_policy(model, new)
    return new


def improve_judge(mem, model: str, group: int, spec: dict, log) -> dict:
    """Turn the agent's audit record on its own auditor.

    Nothing else in this file questions the probe suite, and every fix it has
    ever had came from me. `judge.critique` amends it only where the measurements
    force it, and the verdicts an amendment invalidates are re-derived from the
    stored per-probe scores -- so a better judge is applied backwards over every
    past decision for zero model evaluations.
    """
    # One group size only: a config quantized at a different group size is not
    # comparable evidence about this judge.
    rows = J.records({k: v for k, v in mem.scored(model).items()
                      if k.endswith(f"@g{group}")})
    new, applied, refused = J.improve(spec, rows)
    for a in refused:
        log(f"  JUDGE will not amend [{a['kind']}] {a['why']}")
        log(f"    that leaves under {J.MIN_PROBES} gradeable probes: this needs better probes, "
            f"not a laxer judge")
    if not applied:
        return spec
    for a in applied:
        log(f"  JUDGE amends [{a['kind']}] {a['why']}")
    flips = J.replay(rows, spec, new)
    for f, r in zip(flips, rows):
        mem.put_raw(model, r["key"], agentic=f["after"], judge=new["version"],
                    audit="passed" if f["verdict_after"] else "failed")
    mem.put_judge(model, new)
    changed = [f["key"] for f in flips if f["flipped"]]
    log(f"  JUDGE {spec['version']} -> {new['version']}, now grading on "
        f"{J.graded(rows[0]['scores'], new)}")
    log(f"  {len(changed)} past verdict(s) re-decided from scores already on record"
        + (": " + ", ".join(changed) if changed else ""))
    return new


def search(ev: Evaluator, budget: float, items, baseline_probes: dict,
           validate_every: int, min_agentic: float, log, mem=None, model="",
           margin: float = 0.02, spec: dict = None, z: float = 1.0,
           probe_items: int = 6, may_replenish: bool = True,
           max_probe_items: int = 64,
           pol: dict = None) -> dict:  # noqa: C901
    state = {c: 16 for c in COMPONENTS if c in ev.sizes}
    history, accepted, cadence = [], 0, validate_every
    # Where to fall back to. Stepping back blindly can land on a config a
    # previous audit already rejected, so only audited-good states qualify.
    last_good = {c: 16 for c in COMPONENTS if c in ev.sizes}
    spec = spec or J.NAIVE
    banned = set(mem.rejected(model, spec["version"])) if mem else set()
    # Disproved and unmeasurable are different things and must not share a set.
    unresolved, replenished = set(), False
    pol = pol or dict(PO.P0)
    crossings = 0
    if banned:
        log(f"memory: {len(banned)} config(s) already disproved, not proposing them again")

    while True:
        best = None
        ladder = pol["ladder"]
        for c in state:
            if state[c] not in ladder:
                continue
            i = ladder.index(state[c])
            if i + 1 >= len(ladder):
                continue
            trial = {**state, c: ladder[i + 1]}
            if key(trial, ev.group) in banned:
                log(f"  skip {c}->{ladder[i+1]}: a previous run's audit rejected it")
                continue
            if key(trial, ev.group) in unresolved:
                log(f"  skip {c}->{ladder[i+1]}: the audit could not resolve it this run")
                continue
            rep = ev.divergence(trial)
            if not math.isfinite(rep["cost"]):
                # Never let an unreadable signal drive an accept: that is how a
                # search ends up optimising its own instrument failure.
                log(f"  skip {c}->{ladder[i+1]}: divergence unreadable")
                continue
            if rep["cost"] > budget:
                log(f"  reject {c}->{ladder[i+1]}: cost {rep['cost']:.2f} over budget "
                    f"{budget:.2f} (routing {rep['set_change_rate']:.3f}, "
                    f"recon {rep['recon_error']:.3f})")
                continue
            saved = model_bits(state, ev.sizes) - model_bits(trial, ev.sizes)
            cost = max(rep["cost"] - ev.divergence(state)["cost"], 1e-6)
            score = saved / cost
            log(f"  try {c}->{ladder[i+1]}: cost {rep['cost']:.2f} "
                f"(routing {rep['set_change_rate']:.3f}, recon {rep['recon_error']:.3f}), "
                f"saves {saved:.2f} bits, ratio {score:.1f}")
            if best is None or score > best[0]:
                best = (score, c, ladder[i + 1], trial, rep)

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
            scores, counts, score, se, sure = audit_state(
                ev, state, items, baseline_probes, spec, mem, model, log,
                min_agentic, margin, z, rep)
            if mem:
                spec = improve_judge(mem, model, ev.group, spec, log)
                pol = improve_policy(mem, model, ev.group, pol, log)
                banned = set(mem.rejected(model, spec["version"]))
                if key(last_good, ev.group) in banned:
                    last_good = {c: 16 for c in ev.sizes}
                    log("  the improved judge rejects the fallback config too: "
                        f"falling back to {last_good}")
                score = J.score(scores, baseline_probes["scores"], spec)
                se = J.resolution(scores, baseline_probes["scores"], spec, counts,
                                  baseline_probes["counts"])
                sure = J.decided(score, se, min_agentic, margin, z)
            verdict = ("passed" if J.passes(score, min_agentic, margin) else "failed") \
                if sure else "unresolved"
            if mem:
                mem.put(model, state, ev.group, agentic=score, judge=spec["version"],
                        resolution=se, audit=verdict)
            history[-1].update(validated_agentic=score, resolution=se, audit=verdict)
            log(f"  VALIDATE: agentic {score:.3f} +/- {se:.3f} (floor {min_agentic}, "
                f"tolerance {margin}) -> {verdict}  {scores}")
            if verdict == "passed":
                last_good, crossings = dict(state), 0
            elif verdict == "failed" and crossings < pol["pass_through"]:
                # The record shows a strictly more quantized config auditing better
                # than this one, so failing here does not close the route beyond it.
                # Cross it -- but it is banned as a destination and never becomes the
                # answer.
                banned.add(key(state, ev.group))
                crossings += 1
                cadence = 1
                log(f"  audit rejected {state}, but the record says better configs lie "
                    f"beyond a config like this one: crossing it rather than reverting")
                history[-1]["crossed"] = True
            else:
                if verdict == "failed":
                    # Ban the config the audit disproved, but do NOT shrink the cost
                    # budget: the cost metric is not what failed, the config is, and
                    # tightening it walls off every config beyond this one -- including
                    # better ones only reachable through here. Verify every step
                    # instead, which is the honest response to the cheap signal having
                    # just been shown wrong.
                    banned.add(key(state, ev.group))
                else:
                    # Not disproved, only unmeasurable. It is not evidence against the
                    # config and must not be stored as if it were -- but the search
                    # cannot keep proposing what it cannot grade, so it is set aside
                    # for this run only.
                    unresolved.add(key(state, ev.group))
                    if may_replenish and not replenished and mem:
                        replenished = True
                        # New families are a bonus. The lever that actually moves the
                        # standard error is item count -- it falls as 1/sqrt(n) -- so
                        # growing the suite must not be gated on the vetter admitting
                        # something new, which is what stalled this search at BF16.
                        replenish(model, ev.device, mem.path, log)
                        need = J.items_needed(scores, baseline_probes["scores"], spec,
                                              counts, margin / max(z, 1e-9),
                                              baseline_probes["counts"])
                        want = max([probe_items, *need.values()]) if need else probe_items
                        grown = min(want, max_probe_items)
                        log(f"  RESOLVE: deciding a {margin} margin needs ~{want} items "
                            f"per probe ({need}); growing to {grown}"
                            + (" (capped)" if grown < want else ""))
                        items = build_items(grown)
                        base_state = {c: 16 for c in ev.sizes}
                        sc, cn = ev.probe(base_state, items)
                        baseline_probes = {"scores": sc, "counts": cn}
                        mem.put(model, base_state, ev.group, scores=sc, counts=cn,
                                suite=suite_of(items), agentic=1.0, audit="passed",
                                judge=spec["version"])
                        unresolved.clear()
                        log(f"  re-baselined on {len(items)} items across "
                            f"{len(suite_of(items))} probes: "
                            f"{ {k: round(v, 3) for k, v in sc.items()} }")
                        floor_se = J.resolution(sc, sc, spec, cn, cn)
                        # It grew as far as it can. If it still cannot resolve the
                        # line it was given, deciding anyway is the exact error this
                        # machinery exists to stop, and stalling at BF16 answers
                        # nothing -- so it widens the line to the finest one its
                        # instrument can defend, and records why.
                        was = margin
                        margin, am = J.defensible_margin(floor_se, z, margin, want, grown)
                        if am:
                            if pol is not None:
                                pol.setdefault("amendments", []).append(am)
                                pol["version"] = pol.get("version", "p1") + "+widen"
                                if mem:
                                    mem.put_policy(model, pol)
                            log(f"  AMEND POLICY: margin {was} -> {margin} "
                                f"— {am['why']}")
                cadence = 1
                state = dict(last_good)
                log(f"  revert to the last audited config {state}, "
                    f"and audit every step from now on")
                history[-1]["reverted"] = True
    if validate_every:  # the last accepted step is otherwise never audited
        scores, counts, final, se, sure = audit_state(
            ev, state, items, baseline_probes, spec, mem, model, log,
            min_agentic, margin, z)
        if mem:
            spec = improve_judge(mem, model, ev.group, spec, log)
        verdict = ("passed" if J.passes(final, min_agentic, margin) else "failed") \
            if sure else "unresolved"
        if mem:
            mem.put(model, state, ev.group, agentic=final, judge=spec["version"],
                    resolution=se, audit=verdict)
        log(f"FINAL VALIDATE: agentic {final:.3f} +/- {se:.3f} under judge "
            f"{spec['version']} -> {verdict}  {scores}")
        if verdict != "passed":
            state = dict(last_good)
            log(f"  final state was not shown safe ({verdict}), falling back to {state}")
        return {"state": state, "history": history, "budget": budget, "final_agentic": final,
                "resolution": se, "final_verdict": verdict,
                "judge": spec, "policy": pol, "model_bits": model_bits(state, ev.sizes),
                "evaluations": ev.evals}
    return {"state": state, "history": history, "budget": budget, "judge": spec,
            "policy": pol, "model_bits": model_bits(state, ev.sizes), "evaluations": ev.evals}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--budget", type=float, default=1.0,
                   help="max damage cost, where 1.0 is the sweep's HIGH cut point")
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--validate-every", type=int, default=2, help="0 disables the auditor")
    p.add_argument("--min-agentic", type=float, default=0.95)
    p.add_argument("--probe-items", type=int, default=6,
                   help="items per generated probe family; the agent raises this "
                        "itself when an audit cannot resolve a verdict")
    p.add_argument("--max-probe-items", type=int, default=64,
                   help="ceiling on that growth; past it the agent widens its own "
                        "audit margin to the finest line the suite can defend")
    p.add_argument("--no-replenish", action="store_true",
                   help="forbid the agent from growing its own probe suite")
    p.add_argument("--z", type=float, default=1.0,
                   help="how many standard errors from the line a verdict must be "
                        "before the agent is willing to call it")
    p.add_argument("--audit-margin", type=float, default=0.02,
                   help="how far under the floor counts as real, not measurement noise")
    p.add_argument("--out", default="search.json")
    p.add_argument("--memory", default="memory.json", help="carry knowledge between runs")
    p.add_argument("--forget", action="store_true", help="ignore what past runs learned")
    args = p.parse_args(argv)

    mid = S.MODELS.get(args.model, args.model)
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    # --forget writes to a scratch store rather than none: the agent still has to
    # be able to critique its judge from the run it is having.
    mem = Memory(args.memory if not args.forget
                 else os.path.join(tempfile.mkdtemp(), "forgotten.json"))
    log(f"memory: {mem.summary(mid)}")
    args.budget = mem.learned_budget(mid, args.budget)
    spec = mem.judge(mid) or J.NAIVE
    pol = mem.policy(mid) or dict(PO.P0, ladder=list(LADDER))
    log(f"judge: {spec['version']}, excludes {spec['exclude'] or 'nothing'}, cap {spec['cap']}")
    log(f"policy: {pol['version']}, ladder {pol['ladder']}, "
        f"crosses {pol['pass_through']} failed config(s)")
    ev = Evaluator(mid, args.device, args.group, K.DEFAULT_CALIBRATION, K.GATE_NAME, mem)
    log(f"{mid}: {ev.n} experts, top-{ev.k}, sizes "
        f"{ {c: f'{n/1e6:.0f}M' for c, n in ev.sizes.items()} }")

    items = build_items(args.probe_items)
    base_state = {c: 16 for c in ev.sizes}
    base_probes = {"scores": {}, "counts": {}}
    if args.validate_every:
        remembered = mem.get(mid, base_state, args.group)
        if remembered.get("scores") and remembered.get("counts"):
            base_probes = {"scores": remembered["scores"], "counts": remembered["counts"]}
            log("baseline probes (from memory) "
                f"{ {k: round(v, 3) for k, v in base_probes['scores'].items()} }")
        else:
            sc, cn = ev.probe(base_state, items)
            base_probes = {"scores": sc, "counts": cn}
            mem.put(mid, base_state, args.group, scores=sc, counts=cn, agentic=1.0,
                    audit="passed", judge=spec["version"])
            log(f"baseline probes { {k: round(v, 3) for k, v in sc.items()} }")
        log(f"items per probe: {base_probes['counts']}")

    result = search(ev, args.budget, items, base_probes, args.validate_every,
                    args.min_agentic, log, mem, mid, args.audit_margin, spec, args.z,
                    args.probe_items, not args.no_replenish,
                    args.max_probe_items, pol)
    result.update(model=mid, seconds=round(time.time() - t0), group=args.group)
    json.dump(result, open(args.out, "w"), indent=2)

    log(f"\nbest config: {result['state']}")
    log(f"mean bit-width {result['model_bits']:.2f} "
        f"(from 16.0, {100 * (1 - result['model_bits'] / 16):.0f}% smaller)")
    log(f"{result['evaluations']} model evaluations in {result['seconds']}s -> {args.out}")
    log(f"policy ended at {result['policy']['version']}, ladder {result['policy']['ladder']}")
    log(f"judge ended at {result['judge']['version']} "
        f"({len(result['judge']['amendments'])} amendment(s) it made to itself)")
    log(f"memory now holds: {mem.summary(mid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
