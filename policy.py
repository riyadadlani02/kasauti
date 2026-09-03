"""The search's own rules, and the loop that rewrites them.

judge.py let the agent criticise how it grades. This is the other half: how it
moves. Both defects below are ones the agent's record proves and the agent could
not previously act on.

  - The ladder starts 16 -> 8. When the audit rejects a component's only
    available move there is nothing smaller to try, so the search gives up on a
    component it might have taken to 12. It should refine the rung, not abandon
    the component.

  - Greedy descent assumes capability falls as bits are removed. The record says
    otherwise: `gate 5 / attention 8 / expert 5` audits higher than the strictly
    less quantized `gate 16 / attention 8 / expert 5`. Banning a config on a
    failed audit therefore closes a route to configs that audit better -- the
    5.18-bit answer is only reachable *through* a config that fails. Where the
    record shows such an inversion, a failed config may be crossed, though never
    accepted as an answer.

Same discipline as the judge: amendments are made only where the record forces
them, they carry their evidence, and the version bump says the policy changed.
"""
from __future__ import annotations

import re

BASE = 16
P0 = {"version": "p0-greedy", "ladder": [16, 8, 6, 5, 4, 3, 2], "pass_through": 0,
      "amendments": []}


def state_of(key: str) -> dict:
    """{component: bits} from a memory key like attention8,expert5,gate5@g128."""
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"([a-z_]+)(\d+)", key.split("@")[0])}


def _refine(ladder: list, hi: int, lo: int) -> list:
    """Rungs between two the search could not step between."""
    step = (hi - lo) / 3
    new = sorted({hi - round(step), hi - round(2 * step)} - set(ladder), reverse=True)
    return sorted(set(ladder) | set(n for n in new if lo < n < hi), reverse=True)


def critique(rows: list, policy: dict) -> list:
    """Hold the search policy against every audit on record."""
    out = []
    ladder = policy["ladder"]
    second = ladder[1] if len(ladder) > 1 else None
    verdict = {r["key"]: r.get("was") for r in rows}

    # A component whose only available move was rejected, with nothing smaller.
    stuck = set()
    for r in rows:
        st = state_of(r["key"])
        moved = [c for c, b in st.items() if b != BASE]
        if verdict.get(r["key"]) == "failed" and len(moved) == 1 and st[moved[0]] == second:
            stuck.add(moved[0])
    if stuck and second is not None and BASE - second > 2:
        out.append({"kind": "refine_ladder", "ladder": _refine(ladder, BASE, second),
                    "why": f"the only move available to {', '.join(sorted(stuck))} was "
                           f"{BASE}->{second} and the audit rejected it; there is nothing "
                           f"smaller on the ladder, so the search abandons a component it "
                           f"has not finished with"})

    # A strictly more quantized config that audits better than a rejected one.
    if not policy["pass_through"]:
        for a in rows:
            if verdict.get(a["key"]) != "failed":
                continue
            sa = state_of(a["key"])
            for b in rows:
                sb = state_of(b["key"])
                if verdict.get(b["key"]) != "passed" or sa.keys() != sb.keys():
                    continue
                if all(sb[c] <= sa[c] for c in sa) and any(sb[c] < sa[c] for c in sa):
                    out.append({"kind": "pass_through", "pass_through": 1,
                                "why": f"{b['key']} passed its audit and is strictly more "
                                       f"quantized than {a['key']}, which failed. A failed "
                                       f"config does not close the route beyond it, and "
                                       f"reverting from it is what loses the better answer"})
                    return out
    return out


def revise(policy: dict, amendments: list) -> dict:
    new = {k: (list(v) if isinstance(v, list) else v) for k, v in policy.items()}
    for a in amendments:
        for field in ("ladder", "pass_through"):
            if field in a:
                new[field] = a[field]
        new["amendments"].append(a)
    n = int(re.match(r"p(\d+)", policy["version"]).group(1)) + 1
    new["version"] = f"p{n}-" + "+".join(sorted({a["kind"] for a in amendments}))
    return new


def improve(policy: dict, rows: list, rounds=4):
    applied = []
    for _ in range(rounds):
        found = critique(rows, policy)
        if not found:
            break
        policy = revise(policy, found)
        applied += found
    return policy, applied
