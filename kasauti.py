#!/usr/bin/env python3
"""Router divergence diagnostic for quantized MoE checkpoints.

Runs the same token sequence through a base and a quantized checkpoint,
compares which experts each token was routed to, and reports whether the
routing damage looks large enough to have broken agentic behaviour.

Thresholds are placeholders until the study behind this repo fixes them.
See README.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import torch

# Matches `...mlp.gate`, `...block_sparse_moe.gate`, `...mlp.router`.
# Deliberately excludes `gate_proj` / `gate_up_proj`, which are dense FFN weights.
GATE_NAME = re.compile(r"(?:^|\.)(?:gate|router)(?:\.|$)")
LAYER_INDEX = re.compile(r"layers\.(\d+)")

# Agentic-shaped text: tool calls, structured output, long-horizon constraints.
# Routing over output tokens is what matters, so each prompt carries its own
# continuation rather than stopping at the question.
DEFAULT_CALIBRATION = [
    """You are an agent with access to search(query) and read_file(path).
Constraints for this whole session: never mention the word "certainly", always
end each reply with a line "steps_used: <n>".
User: Find where the retry limit is configured.
Assistant: I'll search the repo first.
search("retry limit")
Observation: config/net.yaml:14 max_retries, client/http.py:88 RETRY_LIMIT
Assistant: Two candidates. Reading the client to see which one is authoritative.
read_file("client/http.py")
steps_used: 2""",
    """Return JSON only, matching {"name": str, "tags": [str], "score": float}.
Input: name is `O'Brien "Sam" \\ Ltd`, tags are {alpha}, {beta}, score is 0.5.
Output: {"name": "O'Brien \\"Sam\\" \\\\ Ltd", "tags": ["{alpha}", "{beta}"], "score": 0.5}""",
    """User: What was the closing share price of Acme Corp yesterday?
Assistant: I don't have access to market data, and I can't verify yesterday's
close, so I won't guess a number. If you tell me the exchange I can tell you
where to look it up.""",
    """Assistant: calling get_order(id="A-1182")
Observation: {"status": "shipp   <<TRUNCATED>>
Assistant: That tool result is malformed and got cut off mid-value, so I won't
treat the status as read. Retrying the same call once before falling back.
calling get_order(id="A-1182")
Observation: {"status": "shipped", "eta": "2026-09-04"}
Assistant: Order A-1182 is shipped, ETA 4 September.""",
    """Plan a three-step migration of a Postgres table with zero downtime, then
execute step one. Keep a running count of rows touched and report it after
every step. Step 1: add the new nullable column. rows_touched: 0. Step 2:
backfill in batches of 10000. rows_touched: 2400000. Step 3: swap reads and
drop the old column. rows_touched: 2400000.""",
    """system: You are a triage agent. Never promise a callback time.
user: My delivery is late and I want to know exactly when someone will ring me.
assistant: I can raise this with the delivery team now and log it as urgent.
I'm not able to commit to a specific callback time, but the ticket will carry
your number and the full history so nobody has to ask you again.""",
    """def rank(items, key, reverse=False):
    # sort, then stably break ties by the original index
    order = {id(x): i for i, x in enumerate(items)}
    return sorted(items, key=lambda x: (key(x), order[id(x)]), reverse=reverse)

The tie-break above is wrong under reverse=True, because reversing the tuple
also reverses the index term. Fix: sort ascending on the index separately.""",
    """Translate the constraint list into a checklist, then verify the draft
against it one line at a time, marking each PASS or FAIL with the offending
substring quoted. Constraint 1: under 40 words. PASS. Constraint 2: no
adverbs. FAIL ("quickly"). Constraint 3: ends with a question. PASS.""",
]


# ---------------------------------------------------------------- model access

def _num_experts(config) -> int:
    for attr in ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"):
        n = getattr(config, attr, None)
        if n:
            return int(n)
    raise SystemExit("could not read expert count from config; is this an MoE checkpoint?")


def _top_k(config) -> int:
    for attr in ("num_experts_per_tok", "moe_top_k", "num_experts_per_token", "top_k"):
        k = getattr(config, attr, None)
        if k:
            return int(k)
    return 2


def load_model(path: str, device: str):
    from transformers import AutoModelForCausalLM

    kwargs = dict(device_map=device, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype="auto", **kwargs)
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", **kwargs)
    return model.eval()


def find_gates(model, pattern: re.Pattern = GATE_NAME) -> dict:
    """Modules whose name looks like an MoE gate. Shape is checked at capture time."""
    return {name: mod for name, mod in model.named_modules() if pattern.search(name)}


def capture_routing(model, input_ids, pattern: re.Pattern = GATE_NAME) -> dict:
    """Routing decisions per gate for one forward pass.

    Returns {name: {"logits": [tokens, experts], "ids": [tokens, k]}}, either key
    possibly absent. Most MoE routers return the selection rather than the logits
    they came from, so logits are recomputed from the router weight against the
    hidden state it saw. That keeps the margin analysis available on any
    architecture, while the model's own selection is still used where it reports
    one, so group-limited or otherwise non-plain top-k routing is not misread.
    """
    gates = find_gates(model, pattern)
    if not gates:
        raise SystemExit("no gate modules matched; pass --gate-pattern for this architecture")

    store: dict[str, dict] = {}
    seen: set = set()

    def hook(name, module):
        def fn(_m, inputs, output):
            if name in seen:
                # A second call would silently misalign the two runs' tokens.
                raise SystemExit(f"gate {name} fired twice in one forward; unsupported architecture")
            seen.add(name)
            rec = store.setdefault(name, {})
            w = getattr(module, "weight", None)
            if w is None and hasattr(module, "layer"):
                w = getattr(module.layer, "weight", None)
            x = inputs[0] if inputs else None
            if (isinstance(w, torch.Tensor) and isinstance(x, torch.Tensor)
                    and w.ndim == 2 and x.shape[-1] == w.shape[-1]):
                x = x.detach().reshape(-1, x.shape[-1]).float()
                rec["logits"] = (x @ w.detach().float().T).cpu()
            t = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(t, torch.Tensor) and t.ndim >= 2:
                t = t.detach().reshape(-1, t.shape[-1])
                if t.is_floating_point():
                    rec.setdefault("logits", t.float().cpu())
                else:
                    rec["ids"] = t.long().cpu()

        return fn

    handles = [m.register_forward_hook(hook(n, m)) for n, m in gates.items()]
    try:
        with torch.no_grad():
            model(input_ids.to(model.device))
    finally:
        for h in handles:
            h.remove()
    return store


def keep_router_outputs(store: dict, n_experts: int, k: int) -> dict:
    """Drop hooked modules whose capture is not a routing decision."""
    kept = {}
    for name, rec in store.items():
        out = {}
        if isinstance(rec.get("logits"), torch.Tensor) and rec["logits"].shape[-1] == n_experts:
            out["logits"] = rec["logits"]
        if isinstance(rec.get("ids"), torch.Tensor) and rec["ids"].shape[-1] == k:
            out["ids"] = rec["ids"]
        if out:
            kept[name] = out
    if not kept:
        raise SystemExit(
            f"matched gates produced no [tokens, {n_experts}] logits or [tokens, {k}] ids; "
            "pass --gate-pattern"
        )
    return kept


def selection(rec: dict, k: int) -> torch.Tensor:
    """The experts a token was routed to: the model's own choice where it reports one."""
    if "ids" in rec:
        return rec["ids"][:, :k]
    return rec["logits"].topk(k, dim=-1).indices


def cat_runs(runs: list) -> dict:
    """Concatenate several forward passes into one record per gate."""
    names = set.intersection(*(set(r) for r in runs))
    return {n: {key: torch.cat([r[n][key] for r in runs])
                for key in ("logits", "ids") if all(key in r[n] for r in runs)}
            for n in names}


# --------------------------------------------------------------------- metrics

def top_ids(t: torch.Tensor, k: int) -> torch.Tensor:
    return t.topk(k, dim=-1).indices if t.is_floating_point() else t[:, :k]


def flip_rate(base_ids: torch.Tensor, quant_ids: torch.Tensor) -> float:
    """Fraction of tokens whose highest-scoring expert changed."""
    return (base_ids[:, 0] != quant_ids[:, 0]).float().mean().item()


def jaccard_distance(base_ids: torch.Tensor, quant_ids: torch.Tensor) -> float:
    """Mean 1 - |A∩B|/|A∪B| over the top-k sets. Catches partial reshuffles."""
    k = base_ids.shape[1]
    inter = (base_ids.unsqueeze(2) == quant_ids.unsqueeze(1)).any(-1).sum(-1).float()
    return (1 - inter / (2 * k - inter)).mean().item()


def margins(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Gap between the k-th and (k+1)-th expert. Narrow = sitting on a boundary.

    k=1 gives the top-1 margin, which governs whether the argmax moves; k=top_k
    gives the selection-boundary margin, which governs whether the set changes.
    Pairing one margin with the other event measures nothing.
    """
    vals = logits.topk(k + 1, dim=-1).values
    return vals[:, k - 1] - vals[:, k]


def load_kl(base_ids: torch.Tensor, quant_ids: torch.Tensor, n_experts: int) -> float:
    """KL(base load || quantized load). Detects traffic collapsing onto fewer experts."""
    def dist(ids):
        c = torch.bincount(ids.reshape(-1), minlength=n_experts).float() + 1e-6
        return c / c.sum()

    p, q = dist(base_ids), dist(quant_ids)
    return (p * (p / q).log()).sum().item()


def flips_by_margin(base_margins: torch.Tensor, flipped: torch.Tensor, buckets: int = 4) -> list:
    """Flip rate per margin quartile, narrow to wide.

    The mechanism claim: flips concentrate where the base router was undecided.
    """
    if base_margins.numel() == 0:
        return []
    edges = torch.quantile(base_margins, torch.linspace(0, 1, buckets + 1))
    out = []
    for i in range(buckets):
        lo, hi = edges[i], edges[i + 1]
        sel = (base_margins >= lo) & (base_margins <= hi if i == buckets - 1 else base_margins < hi)
        out.append(flipped[sel].float().mean().item() if sel.any() else 0.0)
    return out


def verdict(flip: float, low: float, high: float) -> str:
    """Cut points from the sweep in RESULTS.md: on granite-1b-a400m, 5.8% top-1
    flips came with no measurable agentic loss and 18.1% came with format
    stability down 24%. One 1.3B subject, n=8 configs — treat as provisional."""
    if flip < low:
        return "LOW"
    return "MEDIUM" if flip < high else "HIGH"


def worst_layers(per_layer: dict) -> tuple:
    """Contiguous-ish span of the layers above the 75th percentile of flip rate."""
    numbered = {i: v for i, v in per_layer.items() if i is not None}
    if not numbered:
        return None, None
    vals = torch.tensor(list(numbered.values()))
    cut = torch.quantile(vals, 0.75).item()
    hot = sorted(i for i, v in numbered.items() if v >= cut)
    return (hot[0], hot[-1]) if hot else (None, None)


# ----------------------------------------------------------------------- check

def compare(base_store: dict, quant_store: dict, n_experts: int, k: int) -> dict:
    """Per-gate and pooled divergence between two captured routing runs."""
    names = sorted(set(base_store) & set(quant_store))
    if not names:
        raise SystemExit("base and quantized checkpoints exposed different gate names")

    per_gate = {}
    m_top1, e_top1, m_bound, e_bound = [], [], [], []
    for name in names:
        b, q = base_store[name], quant_store[name]
        b_ids, q_ids = selection(b, k), selection(q, k)
        if b_ids.shape[0] != q_ids.shape[0]:
            raise SystemExit(f"token count differs at {name}; the two runs were not teacher-forced")
        flipped = b_ids[:, 0] != q_ids[:, 0]
        shared = (b_ids.unsqueeze(2) == q_ids.unsqueeze(1)).any(-1).sum(-1)
        set_changed = shared < k
        per_gate[name] = {
            "flip_rate": flip_rate(b_ids, q_ids),
            "jaccard": jaccard_distance(b_ids, q_ids),
            "load_kl": load_kl(b_ids, q_ids, n_experts),
        }
        if "logits" in b and b["logits"].shape[-1] > k:
            # Each margin is paired with the event it actually governs.
            m_top1.append(margins(b["logits"], 1))
            e_top1.append(flipped)
            m_bound.append(margins(b["logits"], k))
            e_bound.append(set_changed)

    pooled = lambda key: sum(g[key] for g in per_gate.values()) / len(per_gate)
    per_layer = {}
    for name, g in per_gate.items():
        m = LAYER_INDEX.search(name)
        per_layer[int(m.group(1)) if m else None] = g["flip_rate"]

    report = {
        "flip_rate": pooled("flip_rate"),
        "jaccard": pooled("jaccard"),
        "load_kl": pooled("load_kl"),
        "per_layer_flip_rate": {str(i): v for i, v in sorted(per_layer.items(), key=lambda x: (x[0] is None, x[0]))},
        "top1_flips_by_margin": (
            flips_by_margin(torch.cat(m_top1), torch.cat(e_top1)) if m_top1 else []),
        "set_changes_by_margin": (
            flips_by_margin(torch.cat(m_bound), torch.cat(e_bound)) if m_bound else []),
        "set_change_rate": (torch.cat(e_bound).float().mean().item() if e_bound else float("nan")),
    }
    lo, hi = worst_layers(per_layer)
    report["worst_layers"] = [lo, hi] if lo is not None else None
    return report


def run_check(args) -> dict:
    from transformers import AutoTokenizer

    prompts = DEFAULT_CALIBRATION
    if args.calibration:
        prompts = [p.strip() for p in open(args.calibration).read().split("\n---\n") if p.strip()]

    pattern = re.compile(args.gate_pattern) if args.gate_pattern else GATE_NAME
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    base = load_model(args.base, args.device)
    n_experts, k = _num_experts(base.config), _top_k(base.config)

    forced, base_runs = [], []
    for text in prompts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens).input_ids
        if args.generate:
            with torch.no_grad():
                ids = base.generate(ids.to(base.device), max_new_tokens=args.generate,
                                    do_sample=False).cpu()
        forced.append(ids)
        base_runs.append(keep_router_outputs(capture_routing(base, ids, pattern), n_experts, k))

    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    quant = load_model(args.model, args.device)
    quant_runs = [keep_router_outputs(capture_routing(quant, ids, pattern), n_experts, k)
                  for ids in forced]

    report = compare(cat_runs(base_runs), cat_runs(quant_runs), n_experts, k)
    report.update(
        model=args.model, base=args.base, experts=n_experts, top_k=k,
        tokens=sum(int(i.numel()) for i in forced),
        verdict=verdict(report["flip_rate"], args.low, args.high),
    )
    return report


def render(r: dict) -> str:
    pct = lambda x: f"{100 * x:.1f}%"
    lines = [
        f"router flip rate:              {pct(r['flip_rate'])}",
        f"top-k jaccard distance:        {r['jaccard']:.3f}",
        f"tokens re-routed:              {pct(r['set_change_rate'])}",
        f"expert load KL:                {r['load_kl']:.3f}",
        f"predicted agentic degradation: {r['verdict']}",
    ]
    if r["worst_layers"]:
        lo, hi = r["worst_layers"]
        lines.append(f"worst layers:                  {lo}-{hi}")
    if r.get("set_changes_by_margin"):
        q = "  ".join(pct(x) for x in r["set_changes_by_margin"])
        lines.append(f"expert-set changes by boundary margin (narrow to wide): {q}")
    if r["verdict"] != "LOW":
        lines.append("suggested: hold the gate and attention at higher precision and re-check")
    lines.append(f"({r['tokens']} tokens, {r['experts']} experts, top-{r['top_k']}; "
                 "thresholds calibrated on one 1.3B subject, see RESULTS.md)")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kasauti", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="compare routing against a base checkpoint")
    c.add_argument("model", help="quantized checkpoint")
    c.add_argument("--base", required=True, help="unquantized reference checkpoint")
    c.add_argument("--calibration", help="text file of prompts separated by lines of ---")
    c.add_argument("--generate", type=int, default=0,
                   help="teacher-force N tokens generated by the base model")
    c.add_argument("--max-tokens", type=int, default=512, help="truncate each prompt")
    c.add_argument("--device", default="auto")
    c.add_argument("--gate-pattern", help="regex for gate module names")
    c.add_argument("--low", type=float, default=0.06, help="flip rate below this is LOW")
    c.add_argument("--high", type=float, default=0.15, help="flip rate above this is HIGH")
    c.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    report = run_check(args)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
