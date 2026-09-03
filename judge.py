"""The auditor's own rules, and the loop that rewrites them.

The search audits its cheap signal against a probe suite. Nothing audited the
suite. Every fix to it so far was mine: drop the probe that rises under damage,
ignore the probe with no headroom to fall, cap the ratios so one probe's gain
cannot pay for another's loss. Each fix changed the agent's answer and the agent
proposed none of them.

So the judge stops being code and becomes a spec the agent edits. `critique`
holds that spec against every measurement on record and returns amendments with
the evidence that forced them. `improve` applies them to a fixed point and bumps
the version, which invalidates the old verdicts; `replay` re-derives those
verdicts from the stored per-probe scores, so the agent changes its mind about
past decisions without paying for one model evaluation.

The standards are the ones probegen.vet already applies to probes the agent
writes. This applies them to the probes it inherited.

Two guards, because a judge that can edit itself can edit itself into agreeing
with everything:
  - a probe is dropped only on positive evidence that it moves the wrong way,
    never for merely holding still. The search audits near-healthy configs by
    construction, so "did not fall" is not evidence there,
  - it will not amend below MIN_PROBES gradeable probes. Below that it stops and
    reports that a human owes it better probes.
"""
from __future__ import annotations

import argparse
import re

HEADROOM_LOW = 0.25   # probegen.vet: under this a probe already sits at its floor
MIN_DROP = 0.10       # probegen.vet: the smallest move that counts as real
MIN_PROBES = 3        # a judge grading on fewer than this is not an audit

# Amendments apply one kind per round, in this order. A probe with no headroom
# makes the whole score unreadable, so it has to go before the score can be
# argued with; only then is masking visible; a confound needs a spread of
# readable configs to show itself at all.
ORDER = ("no_headroom", "masking", "confounded")
FLOOR, MARGIN = 0.95, 0.02

# What the agent starts with knowing nothing: grade every probe, trust every
# ratio. It is wrong in three ways that its own measurements will show it.
NAIVE = {"version": "v0-naive", "exclude": [], "cap": None, "amendments": []}


def score(scores: dict, baseline: dict, spec: dict) -> float:
    """The audited score: mean of per-probe ratios under the current judge."""
    vals = []
    for p, b in baseline.items():
        if p in spec["exclude"] or p not in scores:
            continue
        if not b:
            return float("nan")  # a probe at zero cannot fall; the judge cannot read this
        r = scores[p] / b
        vals.append(min(r, spec["cap"]) if spec["cap"] else r)
    return sum(vals) / len(vals) if vals else float("nan")


def graded(baseline: dict, spec: dict) -> list:
    return [p for p in baseline if p not in spec["exclude"]]


def passes(s: float, floor=FLOOR, margin=MARGIN) -> bool:
    return s >= floor - margin  # NaN is not a pass, which is the point


def _bits(k: str) -> float:
    """Mean bit-width from a config key. Unweighted: this orders configs by how
    much precision was removed, and ordering is all the critic needs."""
    vals = [int(re.sub(r"\D", "", p)) for p in k.split("@")[0].split(",")]
    return sum(vals) / len(vals)


def records(scored: dict) -> list:
    """Evidence rows from the memory store, healthiest first."""
    rows = [{"key": k, "bits": _bits(k), "scores": v["scores"], "was": v.get("audit"),
              "was_judge": v.get("judge")} for k, v in scored.items()]
    return sorted(rows, key=lambda r: -r["bits"])


def _slope(xs: list, ys: list) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    d = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / d if d else 0.0


def critique(rows: list, spec: dict, floor=FLOOR, margin=MARGIN) -> list:
    """Hold the judge against the evidence. Amendments carry what forced them."""
    if not rows:
        return []
    base = max(rows, key=lambda r: r["bits"])
    if base["bits"] < 16:
        return []  # nothing to measure ratios against
    out, spread = [], base["bits"] - min(r["bits"] for r in rows)

    for p, b in base["scores"].items():
        if p in spec["exclude"]:
            continue
        if b < HEADROOM_LOW:
            out.append({"kind": "no_headroom", "probe": p, "exclude": p,
                        "why": f"{p} scores {b:.2f} at BF16 — it has no headroom to fall, "
                               f"so its ratio measures noise, not damage"})
            continue
        # Confounded: rises as precision is removed. Positive evidence only —
        # a probe that merely holds still is not accused of anything.
        seen = [r for r in rows if p in r["scores"]]
        if len(seen) >= 3 and spread >= 1.0:
            rise = -_slope([r["bits"] for r in seen], [r["scores"][p] for r in seen]) * spread
            if rise >= MIN_DROP * b:
                peak = max(seen, key=lambda r: r["scores"][p])
                out.append({"kind": "confounded", "probe": p, "exclude": p,
                            "why": f"{p} rises {rise:+.2f} as precision is removed "
                                   f"({b:.2f} at BF16 -> {peak['scores'][p]:.2f} at "
                                   f"{peak['key']}) — grading on it hides real losses"})

    # Masking: the judge reported no damage at all while a graded probe fell.
    if spec["cap"] is None:
        for r in rows:
            if r is base:
                continue
            s = score(r["scores"], base["scores"], spec)
            fell = [p for p in graded(base["scores"], spec)
                    if p in r["scores"] and base["scores"][p]
                    and (base["scores"][p] - r["scores"][p]) / base["scores"][p] >= MIN_DROP]
            if s >= 1.0 and fell:
                out.append({"kind": "masking", "cap": 1.0,
                            "why": f"{r['key']} audited {s:.3f} — no damage — while "
                                   f"{', '.join(fell)} measurably fell. A probe scoring above "
                                   f"baseline is paying for another probe's loss"})
                break
    return out


def revise(spec: dict, amendments: list) -> dict:
    """Apply amendments and bump the version. Old verdicts are now stale."""
    new = {"version": spec["version"], "cap": spec["cap"],
           "exclude": list(spec["exclude"]),
           "amendments": list(spec.get("amendments", []))}
    for a in amendments:
        if "exclude" in a and a["exclude"] not in new["exclude"]:
            new["exclude"].append(a["exclude"])
        if "cap" in a:
            new["cap"] = a["cap"]
        new["amendments"].append(a)
    n = int(re.match(r"v(\d+)", spec["version"]).group(1)) + 1
    new["version"] = f"v{n}-" + "+".join(sorted({a["kind"] for a in amendments}))
    return new


def improve(spec: dict, rows: list, floor=FLOOR, margin=MARGIN, rounds=6):
    """Critique and amend to a fixed point. Returns (spec, applied, refused)."""
    applied, refused, seen = [], [], set()
    base = max(rows, key=lambda r: r["bits"])["scores"] if rows else {}
    for _ in range(rounds):
        found = critique(rows, spec, floor, margin)
        found = [a for a in found if a["kind"] == min(
            (x["kind"] for x in found), key=ORDER.index, default=None)]
        batch, ex = [], set(spec["exclude"])
        for a in found:
            if "exclude" in a and len(base) - len(ex | {a["exclude"]}) < MIN_PROBES:
                if (a["kind"], a.get("probe")) not in seen:
                    seen.add((a["kind"], a.get("probe")))
                    refused.append(a)
                continue
            ex |= {a["exclude"]} if "exclude" in a else set()
            batch.append(a)
        if not batch:
            break
        spec = revise(spec, batch)
        applied += batch
    return spec, applied, refused


def replay(rows: list, old: dict, new: dict, floor=FLOOR, margin=MARGIN) -> list:
    """Re-decide every recorded audit under the new judge. No model evaluations:
    the per-probe scores are already on record, which is the whole reason they
    are stored raw."""
    base = max(rows, key=lambda r: r["bits"])["scores"]
    out = []
    for r in rows:
        a, b = score(r["scores"], base, old), score(r["scores"], base, new)
        out.append({"key": r["key"], "before": a, "after": b,
                    "verdict_before": passes(a, floor, margin),
                    "verdict_after": passes(b, floor, margin),
                    "flipped": passes(a, floor, margin) != passes(b, floor, margin)})
    return out


def inversion(rows: list, spec: dict):
    """The largest case of a strictly-more-quantized config auditing higher.

    The search descends greedily, which assumes capability falls as bits are
    removed. This measures how badly the agent's own record breaks that
    assumption. Probe scoring is greedy and deterministic, so an inversion is
    not sampling noise -- it is the ordering being false.
    """
    base = max(rows, key=lambda r: r["bits"])["scores"]
    sc = {r["key"]: score(r["scores"], base, spec) for r in rows}
    bits = {r["key"]: [int(x) for x in re.findall(r"\d+", r["key"].split("@")[0])] for r in rows}
    worst = (0.0, None, None)
    for a in rows:
        for b in rows:
            if a is b or not all(x >= y for x, y in zip(bits[a["key"]], bits[b["key"]])):
                continue
            d = sc[b["key"]] - sc[a["key"]]
            if d > worst[0]:
                worst = (d, a["key"], b["key"])
    return worst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Review the judge against what past runs measured.")
    p.add_argument("--memory", default="memory.json")
    p.add_argument("--model", help="defaults to the only model in the store")
    p.add_argument("--from-naive", action="store_true",
                   help="discard the learned judge and re-derive it from the evidence")
    p.add_argument("--write", action="store_true", help="persist the improved judge")
    args = p.parse_args(argv)

    from memory import Memory
    mem = Memory(args.memory)
    models = [m for m in mem.data if m != "_judge"]
    model = args.model or (models[0] if len(models) == 1 else None)
    if not model:
        print(f"pick one with --model: {models}")
        return 1

    rows = records(mem.scored(model))
    spec = NAIVE if args.from_naive else (mem.judge(model) or NAIVE)
    print(f"{model}: {len(rows)} audited configs on record, judge {spec['version']}")
    if not rows:
        return 0

    new, applied, refused = improve(spec, rows)
    for a in applied:
        print(f"  AMEND [{a['kind']}] {a['why']}")
    for a in refused:
        print(f"  REFUSED [{a['kind']}] {a['why']}\n"
              f"    would leave under {MIN_PROBES} gradeable probes — a human owes it more probes")
    if not applied:
        print("  no amendment the evidence forces; the judge stands")
        return 0

    print(f"\njudge {spec['version']} -> {new['version']}  "
          f"excludes {new['exclude'] or 'nothing'}, cap {new['cap']}")
    print(f"grades on: {graded(rows[0]['scores'], new)}\n")
    flips = replay(rows, spec, new)
    argued = 0
    for f, r in zip(flips, rows):
        mark = "FLIPS " if f["flipped"] else "      "
        b = "pass" if f["verdict_before"] else "fail"
        a = "pass" if f["verdict_after"] else "fail"
        note = ""
        if r["was"] and (r["was"] == "passed") != f["verdict_after"]:
            argued += 1
            note = f"   <- {r['was_judge'] or 'the judge on record'} said {r['was']}"
        print(f"  {mark}{f['key']:34s} {f['before']:.3f} ({b}) -> {f['after']:.3f} ({a}){note}")
    print(f"\n{sum(f['flipped'] for f in flips)}/{len(flips)} recorded verdicts change "
          f"under the improved judge, at zero model evaluations")
    if argued:
        print(f"{argued}/{len(flips)} disagree with the judge those audits were graded under")
    d, less, more = inversion(rows, new)
    if d:
        print(f"\nordering check: {more} audits {d:.3f} HIGHER than {less}, which is "
              f"strictly less quantized.\n  the greedy descent assumes that cannot happen, "
              f"and this record says it can.")

    if args.write:
        mem.put_judge(model, new)
        for f, r in zip(flips, rows):
            mem.put_raw(model, r["key"], agentic=f["after"], judge=new["version"],
                        audit="passed" if f["verdict_after"] else "failed")
        print(f"written to {args.memory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
