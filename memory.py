"""What the search remembers between runs.

Without this the agent re-derives everything each time, including the configs it
already proved were bad. The store holds four things: per config, what the cheap
signals said, what the probe audit said if one was ever paid for, and whether the
audit rejected it -- plus, once per model, the judge those verdicts were graded
under, which the agent amends as evidence accumulates (see judge.py). A rejected config is never proposed again, and a config whose
audit already passed is not re-audited.

Plain JSON keyed by model and config, because the whole point is that a human can
read it and see what the agent believes.
"""
from __future__ import annotations

import json
import os
import tempfile


def key(state: dict, group: int) -> str:
    return ",".join(f"{c}{b}" for c, b in sorted(state.items())) + f"@g{group}"


class Memory:
    def __init__(self, path="memory.json"):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}

    def _bucket(self, model: str) -> dict:
        return self.data.setdefault(model, {})

    def judge(self, model: str) -> dict | None:
        """The judge this model's verdicts were graded under, if the agent has
        amended one. Stored apart from the configs because it is not a
        measurement -- it is what the measurements are read through."""
        return self.data.get("_judge", {}).get(model)

    def put_judge(self, model: str, spec: dict):
        self.data.setdefault("_judge", {})[model] = spec
        self.save()

    def get(self, model: str, state: dict, group: int) -> dict:
        return self._bucket(model).get(key(state, group), {})

    def put(self, model: str, state: dict, group: int, **fields):
        rec = self._bucket(model).setdefault(key(state, group), {})
        rec.update({k: v for k, v in fields.items() if v is not None})
        self.save()

    def rejected(self, model: str, judge: str | None = None) -> set:
        """Configs an audit has already disproved. Never propose these again.

        A verdict from a different judge is not evidence about this one — change
        what the auditor grades and its past rejections stop applying.
        """
        return {k for k, v in self._bucket(model).items()
                if v.get("audit") == "failed" and (judge is None or v.get("judge") == judge)}

    def learned_budget(self, model: str, default: float) -> float:
        """The tightest budget any past run had to fall back to on this model."""
        seen = [v["budget_after_failure"] for v in self._bucket(model).values()
                if "budget_after_failure" in v]
        return min(seen + [default])

    def audited(self, model: str) -> dict:
        return {k: v for k, v in self._bucket(model).items() if "agentic" in v}

    def scored(self, model: str) -> dict:
        """Configs with per-probe scores on record. Raw scores outlive any judge,
        which is what lets a new one re-decide old verdicts for free."""
        return {k: v for k, v in self._bucket(model).items() if v.get("scores")}

    def put_raw(self, model: str, k: str, **fields):
        self._bucket(model).setdefault(k, {}).update(fields)
        self.save()

    def save(self):
        # Atomic, because a killed run mid-write would otherwise lose the lot.
        d = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    def summary(self, model: str) -> str:
        b = self._bucket(model)
        audits = self.audited(model)
        failed = self.rejected(model)
        return (f"{len(b)} configs remembered, {len(audits)} audited, "
                f"{len(failed)} rejected")
