"""Run the candidate probes on a healthy and a damaged config, keep what earns it."""
from __future__ import annotations

import argparse
import json

import kasauti as K
import probegen
import probes as P
import search as SR
import sweep as S

# Known from RESULTS.md: W8 keeps every probe score, W4 loses format and long-horizon.
STATES = {"baseline": {"gate": 16, "attention": 16, "expert": 16},
          "healthy": {"gate": 16, "attention": 8, "expert": 8},
          "damaged": {"gate": 16, "attention": 4, "expert": 4}}


def existing_series(results="results.jsonl") -> dict:
    """How the probes we already have moved across the same three configs."""
    rows = {json.loads(l)["config"]: json.loads(l) for l in open(results) if l.strip()}
    want = {"baseline": "bf16", "healthy": "w8-ea", "damaged": "w4-ea"}
    if not all(c in rows for c in want.values()):
        return {}
    names = rows["bf16"]["probes"]
    return {n: [rows[want[k]]["probes"].get(n, 0) for k in ("baseline", "healthy", "damaged")]
            for n in names}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="probes_generated.json")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    mid = S.MODELS.get(args.model, args.model)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    items = probegen.generate()
    for it in items:
        P.SCORERS.setdefault(it.probe, probegen.score)
    print(f"{len(items)} candidates, {len(probegen.FAMILIES)} families", flush=True)

    import os
    if os.path.exists(args.out + ".raw"):
        by_config = json.load(open(args.out + ".raw"))
        print(f"reusing scores from {args.out}.raw", flush=True)
        verdicts = probegen.vet(by_config, existing_series())
        return _report(verdicts, args.out, by_config)

    by_config = {}
    for label, state in STATES.items():
        model = K.load_model(mid, args.device)
        SR.apply_state(model, state, 128)
        res = P.run_items(model, tok, items, batch_size=16)
        S.free(model)
        by_config[label] = {r["id"]: r["score"] for r in res}
        print(f"{label}: mean {sum(by_config[label].values()) / len(res):.3f}", flush=True)

    # Save before vetting: a bug in the verdict logic must not cost the compute.
    json.dump(by_config, open(args.out + ".raw", "w"), indent=2)
    return _report(probegen.vet(by_config, existing_series()), args.out, by_config)


def _report(verdicts, out, by_config) -> int:
    keep = [f for f, v in verdicts.items() if v["keep"]]
    for f, v in sorted(verdicts.items()):
        mark = "KEEP  " if v["keep"] else "drop  "
        print(f"{mark}{f:14s} bf16 {v['baseline']:.2f}  healthy {v['healthy']:.2f}  "
              f"damaged {v['damaged']:.2f}  {'; '.join(v['reasons'])}", flush=True)
    json.dump({"keep": keep, "verdicts": verdicts}, open(out, "w"), indent=2)
    print(f"\nkept {len(keep)}/{len(verdicts)} families -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
