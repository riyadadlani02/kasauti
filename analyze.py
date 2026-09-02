"""Turn a sweep into the two figures and the results table.

H3 holds only if router divergence predicts capability degradation better than
both perplexity and per-layer reconstruction error, so all three are scored
against the same target on the same configs.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

AGENTIC = ("long_horizon", "format", "calibration", "recovery")
CONTROL = "recall"
PREDICTORS = [("set_change_rate", "router expert-set change rate"),
              ("flip_rate", "router top-1 flip rate"), ("jaccard", "router top-k jaccard"),
              ("load_kl", "expert load KL"), ("ppl_ratio", "perplexity ratio"),
              ("recon_error", "layer reconstruction error")]

# A probe the model cannot do at BF16 measures nothing about degradation, and
# scoring it relative to a near-zero baseline turns one item into a 100% swing.
FLOOR = 0.25


def load(path: str) -> list:
    return [json.loads(l) for l in open(path) if l.strip()]


def normalise(records: list) -> list:
    """Score each config relative to its own BF16 baseline."""
    base = next(r for r in records if r["config"] == "bf16")
    usable = [p for p in AGENTIC if base["probes"].get(p, 0) >= FLOOR]
    out = []
    for r in records:
        rel = {p: (r["probes"].get(p, 0) / v if v else float("nan"))
               for p, v in base["probes"].items()}
        agentic = [rel[p] for p in usable if p in rel and np.isfinite(rel[p])]
        out.append({**{"set_change_rate": 0.0, "flip_rate": 0.0, "jaccard": 0.0,
                       "load_kl": 0.0, "recon_error": 0.0},
                    **r, "rel": rel, "ppl_ratio": r["ppl"] / base["ppl"],
                    "agentic": float(np.mean(agentic)) if agentic else float("nan"),
                    "control": rel.get(CONTROL, float("nan")),
                    "usable_probes": usable})
    return out


def spearman(x, y) -> float:
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return pearson(rank(x), rank(y))


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def crossing_point(rows, path):
    """The figure the argument rests on: what each instrument sees, at each bit-width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = sorted([r for r in rows if r["method"] in ("none", "rtn")
                    and set(r["targets"]) in ({"expert", "attention"}, set())],
                   key=lambda r: -r["bits"])
    if len(sweep) < 3:
        return None
    usable = sweep[0]["usable_probes"]
    x = list(range(len(sweep)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(x, [r["control"] for r in sweep], "o-", lw=2.6, color="black",
             label="knowledge recall (the control)")
    ax1.plot(x, [1 / r["ppl_ratio"] for r in sweep], "o-", lw=2.6, color="dimgrey",
             label="perplexity (inverted, higher is better)")
    for p_, style, c in zip(usable, ["s--", "^--", "v--", "d--"],
                            ["tab:blue", "tab:orange", "tab:green", "tab:purple"]):
        ax1.plot(x, [r["rel"].get(p_, np.nan) for r in sweep], style, color=c,
                 label=p_.replace("_", " "))
    ax1.axhline(1.0, color="grey", lw=0.8, zorder=0)
    ax1.set_ylabel("score relative to BF16")
    ax1.set_title("What each instrument sees as the bit-width falls")
    ax1.legend(fontsize=8, loc="lower left")

    ax2.plot(x, [r["set_change_rate"] for r in sweep], "o-", color="crimson",
             label="tokens routed to a different expert set")
    ax2.plot(x, [r["flip_rate"] for r in sweep], "o--", color="indianred",
             label="tokens whose top-1 expert changed")
    ax2.set_ylabel("router divergence")
    ax2.set_xlabel("weight bit-width")
    ax2.set_xticks(x, [f"{r['bits']}-bit" if r["bits"] < 16 else "BF16" for r in sweep])
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def predictor_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live = [r for r in rows if r["config"] != "bf16" and np.isfinite(r["agentic"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    picks = [PREDICTORS[0], PREDICTORS[4], PREDICTORS[5]]
    for ax, (key, label) in zip(axes, picks):
        xs = [r[key] for r in live]
        ys = [r["agentic"] for r in live]
        ax.scatter(xs, ys, c="tab:blue")
        for r, x, y in zip(live, xs, ys):
            ax.annotate(r["config"], (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(label)
        ax.set_title(f"r={pearson(xs, ys):.2f}  rho={spearman(xs, ys):.2f}", fontsize=9)
    axes[0].set_ylabel("agentic score vs BF16")
    fig.suptitle("H3: which predictor tracks agentic degradation?", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def report(rows) -> str:
    live = [r for r in rows if r["config"] != "bf16" and np.isfinite(r["agentic"])]
    base = next(r for r in rows if r["config"] == "bf16")
    L = [f"# Results — {base['model']}", ""]
    L.append(f"BF16 baseline: perplexity {base['ppl']:.2f}, "
             + ", ".join(f"{p} {v:.3f}" for p, v in sorted(base["probes"].items())))
    dropped = [p for p in AGENTIC if p not in base["usable_probes"]]
    if dropped:
        L += ["", f"Excluded from the agentic mean, at or near their floor at BF16: "
                  f"{', '.join(f'{p} ({base[chr(39)+chr(39)] if False else base[chr(112)+chr(114)+chr(111)+chr(98)+chr(101)+chr(115)][p]:.2f})' for p in dropped)}. "
              "They are still reported per probe below."]
    L += ["", "## Configs", "",
          "| config | bits | method | ppl ratio | recall (control) | agentic mean | flip rate | recon err |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['config']} | {r['bits']} | {r['method']} | {r['ppl_ratio']:.3f} | "
                 f"{r['control']:.3f} | {r['agentic']:.3f} | {r['flip_rate']:.4f} | "
                 f"{r['recon_error']:.4f} |")
    L += ["", "## Per-probe, relative to BF16", "",
          "| config | " + " | ".join(AGENTIC) + " |", "|---|" + "---|" * len(AGENTIC)]
    for r in rows:
        L.append(f"| {r['config']} | " + " | ".join(f"{r['rel'].get(p, float('nan')):.3f}"
                                                    for p in AGENTIC) + " |")

    L += ["", f"## H3 — predictors of agentic degradation (n={len(live)} configs)", "",
          "| predictor | pearson r | spearman rho |", "|---|---|---|"]
    scored = []
    for key, label in PREDICTORS:
        xs, ys = [r[key] for r in live], [r["agentic"] for r in live]
        r_, rho = pearson(xs, ys), spearman(xs, ys)
        scored.append((abs(r_) if np.isfinite(r_) else -1, label))
        L.append(f"| {label} | {r_:.3f} | {rho:.3f} |")
    scored.sort(reverse=True)
    L += ["", f"Strongest predictor: **{scored[0][1]}** (|r| = {scored[0][0]:.3f})."]

    margins = [r for r in live if r.get("set_changes_by_margin")]
    if margins:
        L += ["", "## Mechanism — divergence by base router margin quartile", "",
              "Each margin is paired with the event it governs: the top-1 margin with a change "
              "of argmax, the k-th/(k+1)-th margin with a change of the selected set.", "",
              "| config | top-1 flips, narrow to wide | set changes, narrow to wide |",
              "|---|---|---|"]
        for r in margins:
            f1 = " ".join(f"{q:.3f}" for q in r["top1_flips_by_margin"])
            f2 = " ".join(f"{q:.3f}" for q in r["set_changes_by_margin"])
            L.append(f"| {r['config']} | {f1} | {f2} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results", nargs="?", default="results.jsonl")
    p.add_argument("--out", default="RESULTS.md")
    p.add_argument("--figures", default=".")
    args = p.parse_args(argv)

    rows = normalise(load(args.results))
    open(args.out, "w").write(report(rows))
    a = crossing_point(rows, f"{args.figures}/crossing_point.png")
    b = predictor_figure(rows, f"{args.figures}/predictors.png")
    print(f"wrote {args.out}" + (f", {a}, {b}" if a else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
