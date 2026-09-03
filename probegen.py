"""Generate candidate probes, then throw most of them away.

Generation is the easy half and the worthless half. A system that writes its own
tests and keeps them all just grows a suite that agrees with itself. The vetting
is the contribution: a candidate family survives only if it has headroom at BF16,
separates a known-healthy config from a known-damaged one, and is not already
covered by a probe we have.

Constraints are emitted as data, not code, so every generated item is still
scored mechanically and nothing here needs a judge.
"""
from __future__ import annotations

import json
import random
import re

from probes import ACK, FILLER, Item

# Each family is a checkable invariant a model must hold across a conversation.
# The rule text is what the model is told; the check is what it is scored on.
FAMILIES = {
    "forbidden": {
        "rule": lambda p: f"Never use the word {p['word']}.",
        "demo": lambda p, i: "Noted.",
        "check": lambda t, p: p["word"].lower() not in t.lower(),
    },
    "suffix": {
        "rule": lambda p: f"End every reply with the line {p['tag']} on its own.",
        "demo": lambda p, i: f"Noted.\n{p['tag']}",
        "check": lambda t, p: t.strip().endswith(p["tag"]),
    },
    "prefix": {
        "rule": lambda p: f"Begin every reply with {p['tag']} followed by a space.",
        "demo": lambda p, i: f"{p['tag']} Noted.",
        "check": lambda t, p: t.strip().startswith(p["tag"]),
    },
    "counter": {
        "rule": lambda p: ("Write a line reply: N before anything else, where N counts "
                           "your replies in this session, starting at 1."),
        "demo": lambda p, i: f"reply: {i}\nNoted.",
        "check": lambda t, p: bool(m := re.search(r"reply:\s*(\d+)", t)) and int(m.group(1)) == p["expect"],
    },
    "wordcap": {
        "rule": lambda p: f"Keep every reply to {p['n']} words or fewer.",
        "demo": lambda p, i: "Noted and logged.",
        "check": lambda t, p: 0 < len(t.split()) <= p["n"],
    },
    "tag": {
        "rule": lambda p: f"Include the marker {p['tag']} somewhere in every reply.",
        "demo": lambda p, i: f"Noted {p['tag']}.",
        "check": lambda t, p: p["tag"] in t,
    },
    "running_total": {
        "rule": lambda p: ("Each of my messages carries a number. End every reply with "
                           "total: N, where N is those numbers added up so far."),
        "demo": lambda p, i: f"Noted.\ntotal: {p['running'][i - 1]}",
        "check": lambda t, p: bool(m := re.search(r"total:\s*(\d+)", t)) and int(m.group(1)) == p["expect"],
    },
    "case": {
        "rule": lambda p: "Write every reply in lower case only.",
        "demo": lambda p, i: "noted.",
        "check": lambda t, p: t.strip() == t.strip().lower() and any(c.isalpha() for c in t),
    },
}

WORDS = ["certainly", "obviously", "basically", "simply", "clearly"]
TAGS = ["[[ack]]", "<<ok>>", "(logged)", "#done", "{seen}"]


def _params(family: str, rng: random.Random, turns: int) -> dict:
    if family == "forbidden":
        return {"word": rng.choice(WORDS)}
    if family in ("suffix", "prefix", "tag"):
        return {"tag": rng.choice(TAGS)}
    if family == "wordcap":
        return {"n": rng.choice([8, 12, 20])}
    if family == "counter":
        return {"expect": turns + 1}
    if family == "running_total":
        nums = [rng.randint(1, 9) for _ in range(turns + 1)]
        running = [sum(nums[: i + 1]) for i in range(len(nums))]
        return {"nums": nums, "running": running, "expect": running[turns]}
    return {}


def generate(n_per_family=6, distances=(3, 8, 16), seed=0) -> list:
    """Candidate items: a constraint stated at turn 0, checked at turn N."""
    rng = random.Random(seed)
    out = []
    for family, spec in FAMILIES.items():
        for j in range(n_per_family):
            turns = distances[j % len(distances)]
            p = _params(family, rng, turns)
            rule = (f"Session rule, in force for every reply from now on: {spec['rule'](p)}\n"
                    "Acknowledge and follow it for the rest of the session.")
            msgs = [{"role": "user", "content": rule}]
            for t in range(turns):
                u, a = FILLER[(t + j) % len(FILLER)]
                if t:
                    n = p["nums"][t] if family == "running_total" else None
                    msgs.append({"role": "user", "content": f"{u} ({n})" if n else u})
                msgs.append({"role": "assistant", "content": spec["demo"](p, t + 1)})
            last = p["nums"][turns] if family == "running_total" else None
            msgs.append({"role": "user", "content":
                         f"Report the current status in one short sentence."
                         + (f" ({last})" if last else "")})
            out.append(Item(f"gen_{family}", f"gen-{family}-{j}", msgs,
                            {"family": family, "params": p, "turns": turns}))
    return out


def score(text: str, meta: dict) -> float:
    return float(FAMILIES[meta["family"]]["check"](text, meta["params"]))


def vet(by_config: dict, existing: dict, headroom=(0.25, 0.95), min_drop=0.10,
        max_overlap=0.9, disproved=()) -> dict:
    """Keep a family only if it has room to fall, actually falls on a damaged
    config, and does not just restate a probe we already have.

    by_config: {config_name: {item_id: score}} for baseline, healthy, damaged.
    existing:  {probe_name: [baseline, healthy, damaged]} for the probes already in use.
    disproved: families the judge has since thrown out (see judge.py). Three
        configs cannot see what a whole search saw: `running_total` passed here
        by falling on the damaged config, then rose across the near-healthy
        range the search actually works in. The later evidence wins.
    """
    base, healthy, damaged = by_config["baseline"], by_config["healthy"], by_config["damaged"]
    families = sorted({i.split("-")[1] for i in base})
    verdicts = {}
    for f in families:
        ids = [i for i in base if i.startswith(f"gen-{f}-")]
        b = sum(base[i] for i in ids) / len(ids)
        h = sum(healthy[i] for i in ids) / len(ids)
        d = sum(damaged[i] for i in ids) / len(ids)
        reasons = ["a later audit disproved it on richer evidence"] if f in disproved else []
        separates = bool(b) and (b - d) / b >= min_drop
        if b < headroom[0]:
            reasons.append(f"at its floor at BF16 ({b:.2f})")
        elif b > headroom[1] and not separates:
            # A probe pinned at 1.00 is only useless if it also cannot fall. One
            # that collapses under damage has headroom by demonstration.
            reasons.append(f"saturated at BF16 and does not move ({b:.2f})")
        if not separates:
            reasons.append(f"does not separate damaged ({b:.2f} -> {d:.2f})")
        # Redundancy: does an existing probe move the same way across the three?
        mine = _shape([b, h, d])
        for name, series in existing.items():
            if _similar(mine, _shape(series)) > max_overlap:
                reasons.append(f"duplicates {name}")
        verdicts[f] = {"baseline": b, "healthy": h, "damaged": d,
                       "keep": not reasons, "reasons": reasons}
    return verdicts


def _shape(series: list) -> list:
    """Relative to the config's own baseline, so absolute difficulty does not
    make two probes look different when they degrade identically."""
    b = series[0] or 1e-6
    return [x / b for x in series]


def _similar(a: list, b: list) -> float:
    """1.0 when two score series degrade the same way."""
    return 1 - min(sum(abs(x - y) for x, y in zip(a, b)) / 3, 1.0)


if __name__ == "__main__":
    items = generate()
    print(f"{len(items)} candidates across {len(FAMILIES)} families")
    for it in items[:2]:
        print(f"\n--- {it.id} ({it.meta['turns']} turns)")
        print(it.messages[0]["content"][:120])
        print("...", it.messages[-1]["content"])
    ok = sum(score(FAMILIES[i.meta["family"]]["demo"](i.meta["params"], i.meta["turns"] + 1),
                   i.meta) for i in items)
    print(f"\nself-check: a compliant reply scores {ok}/{len(items)}")
