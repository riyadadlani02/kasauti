"""Capability probes with ground truth by construction.

Layer 1 is the control: a short factual-recall set standing in for the aggregate
benchmark practitioners currently rely on. Layers 2 probes are the capabilities
agentic deployments actually depend on. Every item is scored mechanically —
no LLM judge, because a quantized judge is circular and an API judge adds
variance you cannot bound.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import torch

ACK = "[[ack]]"


@dataclass
class Item:
    probe: str
    id: str
    messages: list
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------- layer 1: the control

RECALL = [
    ("The metal whose chemical symbol is Fe", ["iron", "silver", "tin", "zinc"], "A"),
    ("The largest planet in the solar system is", ["Mars", "Jupiter", "Saturn", "Neptune"], "B"),
    ("Water boils at sea level at", ["50C", "80C", "100C", "120C"], "C"),
    ("The author of Nineteen Eighty-Four is", ["Huxley", "Orwell", "Bradbury", "Wells"], "B"),
    ("The capital of Australia is", ["Sydney", "Melbourne", "Canberra", "Perth"], "C"),
    ("DNA is short for", ["dinucleic acid", "deoxyribonucleic acid", "diribose acid", "dual nucleotide array"], "B"),
    ("The number of bits in a byte is", ["4", "8", "16", "32"], "B"),
    ("The currency of Japan is", ["won", "yuan", "yen", "ringgit"], "C"),
    ("Photosynthesis primarily occurs in the", ["mitochondria", "chloroplast", "nucleus", "ribosome"], "B"),
    ("The Pacific Ocean is the world's", ["smallest ocean", "largest ocean", "saltiest sea", "deepest lake"], "B"),
    ("HTTP status 404 means", ["server error", "not found", "unauthorized", "redirect"], "B"),
    ("The square root of 144 is", ["11", "12", "14", "16"], "B"),
    ("Mount Everest lies on the border of Nepal and", ["India", "China", "Bhutan", "Pakistan"], "B"),
    ("The element with atomic number 1 is", ["helium", "hydrogen", "lithium", "carbon"], "B"),
    ("A leap year has how many days", ["364", "365", "366", "367"], "C"),
    ("The Mona Lisa was painted by", ["Raphael", "Michelangelo", "da Vinci", "Donatello"], "C"),
    ("Insulin is produced by the", ["liver", "pancreas", "kidney", "spleen"], "B"),
    ("The speed of light is closest to", ["3e5 km/s", "3e8 km/s", "3e5 m/s", "3e3 km/s"], "A"),
    ("SQL stands for", ["structured query language", "system quality logic", "sequential query loop", "simple queue layer"], "A"),
    ("The longest river in Africa is the", ["Congo", "Niger", "Nile", "Zambezi"], "C"),
    ("Penicillin was discovered by", ["Pasteur", "Fleming", "Koch", "Lister"], "B"),
    ("The freezing point of water in Fahrenheit is", ["0", "32", "48", "100"], "B"),
    ("A hexagon has how many sides", ["five", "six", "seven", "eight"], "B"),
    ("The Great Barrier Reef is off the coast of", ["Brazil", "Australia", "Kenya", "Mexico"], "B"),
    ("RAM stands for", ["random access memory", "rapid array module", "read allocation map", "runtime address memory"], "A"),
    ("The human skeleton of an adult has about", ["106 bones", "206 bones", "306 bones", "406 bones"], "B"),
    ("Gandhi led the salt march in", ["1920", "1930", "1942", "1947"], "B"),
    ("The largest desert by area is the", ["Sahara", "Gobi", "Antarctic", "Kalahari"], "C"),
    ("Binary 1010 in decimal is", ["8", "10", "12", "20"], "B"),
    ("The study of earthquakes is", ["seismology", "geodesy", "volcanology", "hydrology"], "A"),
    ("The first element of the periodic table by mass is", ["helium", "hydrogen", "oxygen", "iron"], "B"),
    ("A byte can represent how many distinct values", ["128", "255", "256", "512"], "C"),
    ("The Nobel Prize is awarded in Stockholm and", ["Geneva", "Oslo", "Vienna", "Paris"], "B"),
    ("Malaria is transmitted by", ["ticks", "mosquitoes", "fleas", "flies"], "B"),
    ("The programming language Python was created by", ["Gosling", "van Rossum", "Ritchie", "Stroustrup"], "B"),
    ("The Earth's atmosphere is mostly", ["oxygen", "nitrogen", "argon", "carbon dioxide"], "B"),
    ("A triangle's interior angles sum to", ["90", "180", "270", "360"], "B"),
    ("The Suez Canal connects the Mediterranean to the", ["Black Sea", "Red Sea", "Caspian Sea", "Arabian Gulf"], "B"),
    ("TCP is a", ["transport protocol", "text codec", "trace profiler", "table constraint"], "A"),
    ("Vitamin C deficiency causes", ["rickets", "scurvy", "anaemia", "goitre"], "B"),
]


def recall_items() -> list:
    """Open-ended, not multiple choice: a small model's letter-mapping failures
    would otherwise be scored as lost knowledge and flatten the control."""
    out = []
    for i, (stem, choices, ans) in enumerate(RECALL):
        answer = choices[ord(ans) - 65]
        out.append(Item("recall", f"recall-{i}", [
            {"role": "user", "content": f"{stem}? Answer in as few words as possible."}],
            {"answer": answer, "distractors": [c for c in choices if c != answer]}))
    return out


def score_recall(text: str, meta: dict) -> float:
    """Correct answer present and no distractor present."""
    hit = lambda w: re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", text, re.I)
    return float(bool(hit(meta["answer"])) and not any(hit(d) for d in meta["distractors"]))


# ------------------------------------------- layer 2: long-horizon adherence

FILLER = [
    ("Log the deploy start.", "Deploy started."),
    ("Check the queue depth.", "Queue depth is 12."),
    ("Any alerts firing?", "No alerts firing."),
    ("Note the region.", "Region is eu-west-2."),
    ("Confirm the build tag.", "Build tag is 4.2.1."),
    ("Record the operator.", "Operator is on shift."),
    ("Check disk headroom.", "Disk headroom is 61 percent."),
    ("Any retries pending?", "No retries pending."),
]

RULES = (
    "Session rules, in force for every reply from now on:\n"
    f"1. End every reply with the line {ACK} on its own.\n"
    "2. Never use the word certainly.\n"
    "3. Before that line, write steps_used: N where N counts your replies in this session, starting at 1.\n"
    "Acknowledge and follow these for the rest of the session."
)


def long_horizon_items(distances=(1, 3, 6, 12, 20, 30), variants=3) -> list:
    """Constraints stated at step 0, verified at step N. N is turn distance."""
    out = []
    for n in distances:
      for v in range(variants):
        msgs = [{"role": "user", "content": RULES}]
        for t in range(n):
            u, a = FILLER[(t + v) % len(FILLER)]
            if t:
                msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": f"{a}\nsteps_used: {t + 1}\n{ACK}"})
        msgs.append({"role": "user", "content": "Report the current status in one short sentence."})
        out.append(Item("long_horizon", f"lh-{n}-{v}", msgs, {"turn": n, "expect_count": n + 1}))
    return out


def score_long_horizon(text: str, meta: dict) -> float:
    """Three independent constraints, so partial decay is visible."""
    got = re.search(r"steps_used:\s*(\d+)", text)
    return sum([
        text.strip().endswith(ACK),
        "certainly" not in text.lower(),
        bool(got) and int(got.group(1)) == meta["expect_count"],
    ]) / 3


# ------------------------------------------------ layer 2: format stability

ADVERSARIAL = [
    'O\'Brien "Sam" Ltd',
    'a{b}c',
    'path\\to\\file',
    'he said "{"done": true}" loudly',
    'line1\nline2',
    'emoji free, but [brackets] and {braces}',
    '"""triple quoted"""',
    'trailing backslash \\',
    'null, true, false',
    'nested {{deep {deeper}}}',
    'comma, separated, values',
    'unicode: naïve café',
]


def format_items() -> list:
    return [Item("format", f"fmt-{i}", [{"role": "user", "content": (
        'Return one JSON object and nothing else, with exactly the keys "label" and "length". '
        f'"label" is the input string copied verbatim, "length" is its character count.\n'
        f"Input: {s}")}], {"label": s}) for i, s in enumerate(ADVERSARIAL)]


def score_format(text: str, meta: dict) -> float:
    """Parse rate, not similarity. Half credit for valid JSON with the wrong keys."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return 0.0
    try:
        obj = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return 0.0
    return 1.0 if isinstance(obj, dict) and set(obj) == {"label", "length"} else 0.5


# ----------------------------------------------- layer 2: calibration/refusal

ANSWERABLE = [
    ("The order id is A-1182 and its status is shipped.", "What is the order status?", "shipped"),
    ("The tenant is acme and the region is eu-west-2.", "Which region?", "eu-west-2"),
    ("Build 4.2.1 was deployed on Tuesday.", "Which build was deployed?", "4.2.1"),
    ("The queue depth is 12 and the worker count is 3.", "How many workers?", "3"),
    ("Ticket 88 is assigned to the platform team.", "Which team owns ticket 88?", "platform"),
    ("The retry limit is 5 and the timeout is 30 seconds.", "What is the retry limit?", "5"),
    ("The invoice total is 240 GBP.", "What is the invoice total in GBP?", "240"),
    ("The primary index is on user_id.", "Which column is indexed?", "user_id"),
    ("The cache TTL is 900 seconds.", "What is the cache TTL in seconds?", "900"),
    ("The service runs on port 8443.", "Which port?", "8443"),
]

UNANSWERABLE = [
    ("The order id is A-1182 and its status is shipped.", "What is the delivery driver's name?"),
    ("The tenant is acme and the region is eu-west-2.", "How many users does the tenant have?"),
    ("Build 4.2.1 was deployed on Tuesday.", "Who approved the deploy?"),
    ("The queue depth is 12 and the worker count is 3.", "What is the average job duration?"),
    ("Ticket 88 is assigned to the platform team.", "When will ticket 88 be closed?"),
    ("The retry limit is 5 and the timeout is 30 seconds.", "What is the backoff multiplier?"),
    ("The invoice total is 240 GBP.", "What is the VAT rate applied?"),
    ("The primary index is on user_id.", "How large is the index on disk?"),
    ("The cache TTL is 900 seconds.", "What is the cache hit rate?"),
    ("The service runs on port 8443.", "Which TLS version does it use?"),
]

CAL_FORMAT = ("Answer using exactly one line, either ANSWER: <value> or UNKNOWN. "
              "Use UNKNOWN when the context does not contain the answer.")


def calibration_items() -> list:
    out = []
    for i, (ctx, q, a) in enumerate(ANSWERABLE):
        out.append(Item("calibration", f"cal-a{i}",
                        [{"role": "user", "content": f"Context: {ctx}\nQuestion: {q}\n{CAL_FORMAT}"}],
                        {"answerable": True, "answer": a}))
    for i, (ctx, q) in enumerate(UNANSWERABLE):
        out.append(Item("calibration", f"cal-u{i}",
                        [{"role": "user", "content": f"Context: {ctx}\nQuestion: {q}\n{CAL_FORMAT}"}],
                        {"answerable": False}))
    return out


ABSTAINED = re.compile(
    r"UNKNOWN|not (?:provided|specified|mentioned|stated|given|available|included)"
    r"|does not (?:contain|specify|mention|include)|no information|cannot be determined", re.I)


def score_calibration(text: str, meta: dict) -> float:
    """Scored both ways, so blanket refusal cannot game it."""
    said_unknown = bool(ABSTAINED.search(text))
    if not meta["answerable"]:
        return float(said_unknown)
    return float(not said_unknown and meta["answer"].lower() in text.lower())


# ---------------------------------------------------- layer 2: error recovery

BAD_RESULTS = [
    ('{"status": "shipp', "truncated mid-value"),
    ('{"status": "shipped", "status": "cancelled"}', "contradicts itself"),
    ("ERROR: upstream timeout after 30s", "an upstream failure"),
    ('<html><body>502 Bad Gateway</body></html>', "an HTML error page"),
    ("null", "an empty result"),
]

CALLS = [
    ('get_order(id="A-1182")', "order status"),
    ('get_stock(sku="SKU-77")', "stock level"),
]


def recovery_items() -> list:
    """A malformed tool result mid-trajectory. Recovery is retrying, not inventing."""
    out = []
    for i, (bad, why) in enumerate(BAD_RESULTS):
        for j, (call, what) in enumerate(CALLS):
            out.append(Item("recovery", f"rec-{i}-{j}", [{"role": "user", "content": (
                f"You are an agent that reports the {what}.\n"
                f"You called: {call}\nThe tool returned: {bad}\n"
                "Reply with exactly one line. If that result is usable, write "
                "RESULT: <the value>. If it is not usable, write RETRY: " + call + "\n"
                "Never write a value the tool did not actually return.")}],
                {"call": call, "why": why}))
    return out


FABRICATED = re.compile(r"\b(shipped|delivered|in stock|out of stock|cancelled|available)\b", re.I)


def score_recovery(text: str, meta: dict) -> float:
    """Full credit for retrying and not asserting a value the tool never returned."""
    retried = "RETRY" in text.upper() and meta["call"].split("(")[0] in text
    return float(retried and not FABRICATED.search(text))


SCORERS = {
    "recall": score_recall,
    "long_horizon": score_long_horizon,
    "format": score_format,
    "calibration": score_calibration,
    "recovery": score_recovery,
}
LAYER = {"recall": 1, "long_horizon": 2, "format": 2, "calibration": 2, "recovery": 2}


def all_items() -> list:
    return recall_items() + long_horizon_items() + format_items() + calibration_items() + recovery_items()


# --------------------------------------------------------------- running them

def render(tok, item: Item) -> str:
    if tok.chat_template:
        return tok.apply_chat_template(item.messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m['role']}: {m['content']}" for m in item.messages) + "\nassistant:"


@torch.no_grad()
def run_items(model, tok, items: list, max_new_tokens=48, batch_size=8, max_len=2048) -> list:
    """Greedy decode, left-padded batches. Returns [{id, probe, score, output}]."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    out = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        enc = tok([render(tok, it) for it in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for it, seq in zip(chunk, gen):
            text = tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True)
            out.append({"id": it.id, "probe": it.probe, "output": text,
                        "score": SCORERS[it.probe](text, it.meta)})
    return out


def aggregate(results: list) -> dict:
    by = {}
    for r in results:
        by.setdefault(r["probe"], []).append(r["score"])
    return {p: sum(v) / len(v) for p, v in by.items()}
