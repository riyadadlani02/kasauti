"""Refit the cheap signal's cut point on the agent's own audit record.

The whole premise is that a cheap signal predicts an expensive one. The cut
points came from a 27-config sweep and have been frozen constants ever since,
while the agent accumulated hundreds of (cost, audit outcome) pairs and never
once checked them against the line it was using.

This does the check. It is deliberately not clever: enumerate every threshold
between observed costs, keep the one that best separates passed audits from
failed ones, and say how much better than the incumbent it is.

The interlock that matters: a threshold fitted on verdicts the audit could not
resolve is a threshold fitted on noise. So this refuses to fit on anything but
decided verdicts, and if there are none it says the record cannot support a
refit rather than producing a number that looks like one.
"""
from __future__ import annotations

import argparse

import judge as J

MIN_DECIDED = 6  # under this the fit is an anecdote


def separation(pairs: list, cut: float) -> float:
    """Youden's J for "cost below cut means the audit passes"."""
    pos = [c for c, ok in pairs if ok]
    neg = [c for c, ok in pairs if not ok]
    if not pos or not neg:
        return float("nan")
    sens = sum(c <= cut for c in pos) / len(pos)
    spec = sum(c > cut for c in neg) / len(neg)
    return sens + spec - 1


def fit(pairs: list):
    """Best cut point on this record, and how well it separates. J of 0 means
    the signal orders the outcomes no better than a coin."""
    costs = sorted({c for c, _ in pairs})
    if len(costs) < 2:
        return None, float("nan")
    cuts = [(a + b) / 2 for a, b in zip(costs, costs[1:])]
    best = max(cuts, key=lambda c: separation(pairs, c))
    return best, separation(pairs, best)


def evidence(mem, model: str, spec: dict, z: float = 1.0, floor=J.FLOOR, margin=J.MARGIN):
    """(cost, passed) pairs the audit was actually able to decide, plus how many
    it had to throw out for being inside its own error."""
    bucket = mem.scored(model)
    rows = J.records(bucket)
    if not rows:
        return [], 0
    base = max(rows, key=lambda r: r["bits"])
    pairs, undecided = [], 0
    for r in rows:
        rec = bucket[r["key"]]
        if r is base or "cost" not in rec:
            continue
        s = J.score(r["scores"], base["scores"], spec)
        se = J.resolution(r["scores"], base["scores"], spec, r["counts"] or {},
                          base["counts"] or {})
        if not J.decided(s, se, floor, margin, z):
            undecided += 1
            continue
        pairs.append((rec["cost"], J.passes(s, floor, margin)))
    return pairs, undecided


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--memory", default="memory.json")
    p.add_argument("--model")
    p.add_argument("--incumbent", type=float, default=1.0,
                   help="the cut point in use, where 1.0 is the sweep's HIGH")
    p.add_argument("--z", type=float, default=1.0)
    args = p.parse_args(argv)

    from memory import Memory
    mem = Memory(args.memory)
    models = mem.models()
    model = args.model or (models[0] if len(models) == 1 else None)
    if not model:
        print(f"pick one with --model: {models}")
        return 1
    spec = mem.judge(model) or J.NAIVE

    pairs, undecided = evidence(mem, model, spec, args.z)
    print(f"{model}: {len(pairs)} decided audit(s) with a cost on record, "
          f"{undecided} thrown out as unresolvable")
    if len(pairs) < MIN_DECIDED:
        print(f"  the record cannot support a refit: under {MIN_DECIDED} decided verdicts.")
        print("  this is the audit's resolution problem, not the signal's -- a cut point "
              "fitted\n  on verdicts the probes could not resolve is a cut point fitted "
              "on noise.")
        return 0

    cut, j = fit(pairs)
    now = separation(pairs, args.incumbent)
    print(f"  incumbent cut {args.incumbent:.2f}: separation {now:+.2f}")
    print(f"  best on record {cut:.2f}: separation {j:+.2f}")
    if j <= 0:
        print("  the cheap signal does not order these outcomes at all on this record.")
    elif cut < args.incumbent * 0.9 or cut > args.incumbent * 1.1:
        print(f"  the record says the cut point should move to {cut:.2f}")
    else:
        print("  the sweep's cut point survives its own agent's record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
