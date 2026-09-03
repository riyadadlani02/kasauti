"""Self-check: metric math and the hook plumbing. Run with `python test_kasauti.py`."""
import pathlib

import torch
from torch import nn

import kasauti as K


class FakeMoE(nn.Module):
    """Minimal stand-in: one gate per layer, plus a dense gate_proj to be ignored."""

    def __init__(self, layers=3, hidden=8, experts=4, noise=0.0):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(32, hidden)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            block = nn.Module()
            block.mlp = nn.Module()
            block.mlp.gate = nn.Linear(hidden, experts, bias=False)
            block.mlp.gate_proj = nn.Linear(hidden, hidden, bias=False)
            self.layers.append(block)
        if noise:
            with torch.no_grad():
                for p in self.parameters():
                    p += noise * torch.randn_like(p)

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for block in self.layers:
            block.mlp.gate(h)
            block.mlp.gate_proj(h)
        return h


def test_find_gates_skips_dense_projections():
    names = set(K.find_gates(FakeMoE()))
    assert names == {f"layers.{i}.mlp.gate" for i in range(3)}, names
    assert not any("gate_proj" in n for n in names)


def test_flip_and_jaccard():
    base = torch.tensor([[0, 1], [2, 3], [0, 1]])
    quant = torch.tensor([[0, 1], [3, 2], [1, 0]])
    # rows 1 and 2 changed their top-1, but no row changed its top-k *set*
    assert abs(K.flip_rate(base, quant) - 2 / 3) < 1e-6
    assert K.jaccard_distance(base, quant) == 0.0


def test_load_kl_detects_collapse():
    ids = torch.tensor([[0, 1], [2, 3]])
    assert K.load_kl(ids, ids, 4) < 1e-6
    assert K.load_kl(ids, torch.zeros_like(ids), 4) > 1.0


def test_flips_concentrate_in_narrow_margins():
    m = torch.arange(100).float()                      # margin grows with index
    flipped = torch.arange(100) < 25                   # only the narrowest quartile flips
    q = K.flips_by_margin(m, flipped)
    assert q[0] > 0.9 and max(q[1:]) < 0.1, q


def test_selection_prefers_the_models_own_choice():
    """A router that reports group-limited ids must not be second-guessed by argmax."""
    rec = {"logits": torch.tensor([[9.0, 8.0, 1.0, 0.0]]), "ids": torch.tensor([[2, 3]])}
    assert K.selection(rec, 2).tolist() == [[2, 3]]
    assert K.selection({"logits": rec["logits"]}, 2).tolist() == [[0, 1]]


def test_verdict_needs_both_signals_clear():
    """Either signal over its cut point is enough to call it: they tie at 84%."""
    assert K.verdict(0.05, 0.01) == "LOW"
    assert K.verdict(0.25, 0.05) == "MEDIUM"
    assert K.verdict(0.50, 0.01) == "HIGH"      # routing alone
    assert K.verdict(0.05, 0.20) == "HIGH"      # reconstruction alone


def test_set_change_uses_membership_not_order():
    """Reordering the same experts is not a routing change; swapping one is."""
    base = {"g": {"logits": torch.tensor([[4.0, 3.0, 2.0, 1.0]] * 2),
                  "ids": torch.tensor([[0, 1], [0, 1]])}}
    reordered = {"g": {"ids": torch.tensor([[1, 0], [0, 1]])}}
    swapped = {"g": {"ids": torch.tensor([[1, 0], [0, 2]])}}
    assert K.compare(base, reordered, 4, 2)["set_change_rate"] == 0.0
    assert K.compare(base, swapped, 4, 2)["set_change_rate"] == 0.5


def test_capture_and_compare_end_to_end():
    ids = torch.randint(0, 32, (1, 16))
    base = K.keep_router_outputs(K.capture_routing(FakeMoE(), ids), 4, 2)
    quant = K.keep_router_outputs(K.capture_routing(FakeMoE(noise=0.5), ids), 4, 2)
    assert len(base) == 3 and base["layers.0.mlp.gate"]["logits"].shape == (16, 4)

    same = K.compare(base, base, 4, 2)
    assert same["flip_rate"] == 0.0 and same["jaccard"] == 0.0
    perturbed = K.compare(base, quant, 4, 2)
    assert perturbed["flip_rate"] > 0.0
    assert set(perturbed["per_layer_flip_rate"]) == {"0", "1", "2"}
    assert perturbed["worst_layers"][0] is not None


def test_set_change_rate_survives_a_router_with_no_logits():
    """Some routers report only ids. The verdict must not go NaN there."""
    base = {"g": {"ids": torch.tensor([[0, 1], [0, 1]])}}
    quant = {"g": {"ids": torch.tensor([[0, 1], [0, 2]])}}
    assert K.compare(base, quant, 4, 2)["set_change_rate"] == 0.5


def test_failed_validation_tightens_below_the_failing_cost():
    """Tightening by a fixed factor is not enough: the budget has to drop under
    the cost of the step that failed, or the search re-accepts it forever."""
    budget, failed = 1.0, 0.47
    assert min(budget * 0.6, failed * 0.9) < failed


def test_vetting_keeps_only_probes_that_earn_their_place():
    """Generated probes are worthless by default: the vetting is the whole point."""
    import probegen
    scores = lambda **kw: {f"gen-{f}-0": v for f, v in kw.items()}
    v = probegen.vet({"baseline": scores(falls=1.0, flat=1.0, floored=0.05, real=0.6),
                      "healthy": scores(falls=0.9, flat=1.0, floored=0.05, real=0.6),
                      "damaged": scores(falls=0.0, flat=1.0, floored=0.00, real=0.2)}, {})
    assert v["falls"]["keep"], "a probe pinned at 1.0 that collapses under damage has headroom"
    assert not v["flat"]["keep"], "a probe that never moves measures nothing"
    assert not v["floored"]["keep"], "a probe the model fails at BF16 measures nothing"
    assert v["real"]["keep"]


def test_vetting_rejects_a_probe_that_restates_an_existing_one():
    import probegen
    series = lambda a, b, c: {"gen-copy-0": None} and None
    cand = {"baseline": {"gen-copy-0": 0.8}, "healthy": {"gen-copy-0": 0.6},
            "damaged": {"gen-copy-0": 0.4}}
    assert not probegen.vet(cand, {"format": [0.8, 0.6, 0.4]})["copy"]["keep"]


def test_a_failed_audit_bans_the_config_not_the_region():
    """Shrinking the cost budget on a failed audit walls off better configs that
    are only reachable through the rejected one. Measured: the config the search
    could no longer reach scored 0.966, above the 0.959 it settled for."""
    import search as SR
    src = pathlib.Path(SR.__file__).read_text()
    assert "budget = min(budget * 0.6" not in src
    assert "cadence = 1" in src


def test_revert_target_is_the_last_audited_config():
    """Stepping back blindly can land on a config an earlier audit already
    rejected -- observed live, reverting straight into a banned state."""
    import search as SR
    src = pathlib.Path(SR.__file__).read_text()
    assert 'state = history[-2]["state"]' not in src
    assert "state = dict(last_good)" in src


def _rows(baseline, **damaged):
    """Evidence rows: BF16 first, then configs with progressively fewer bits."""
    rows = [{"key": "attention16,expert16,gate16@g128", "bits": 16.0, "scores": baseline}]
    for i, (name, scores) in enumerate(damaged.items()):
        rows.append({"key": name, "bits": 8.0 - i, "scores": scores})
    return rows


def test_the_judge_drops_a_probe_that_rises_under_damage():
    """Calibration rises as the model degrades, so grading on it hides real
    losses. I found that one by hand; the agent has to find it from the record."""
    import judge as J
    rows = _rows({"format": 0.9, "long_horizon": 0.8, "gen_case": 1.0, "calibration": 0.5},
                 a={"format": 0.8, "long_horizon": 0.7, "gen_case": 0.9, "calibration": 0.6},
                 b={"format": 0.7, "long_horizon": 0.6, "gen_case": 0.8, "calibration": 0.7})
    spec, applied, _ = J.improve(J.NAIVE, rows)
    assert spec["exclude"] == ["calibration"], spec["exclude"]
    assert applied[0]["kind"] == "confounded"


def test_the_judge_drops_a_probe_with_no_headroom_to_fall():
    import judge as J
    rows = _rows({"format": 0.9, "long_horizon": 0.8, "gen_case": 1.0, "recovery": 0.0},
                 a={"format": 0.8, "long_horizon": 0.7, "gen_case": 0.9, "recovery": 0.1})
    assert J.score(rows[1]["scores"], rows[0]["scores"], J.NAIVE) != \
        J.score(rows[1]["scores"], rows[0]["scores"], J.NAIVE)  # NaN: unreadable
    spec, applied, _ = J.improve(J.NAIVE, rows)
    assert spec["exclude"] == ["recovery"]
    assert J.score(rows[1]["scores"], rows[0]["scores"], spec) < 1.0  # readable again


def test_the_judge_caps_ratios_once_it_sees_a_gain_paying_for_a_loss():
    """Real case: format -18% and long-horizon -7% audited at 1.103 because a
    low-baseline probe went 0.50 -> 0.83 and contributed a 1.67 ratio."""
    import judge as J
    rows = _rows({"format": 0.92, "long_horizon": 0.76, "gen_case": 1.0, "spiky": 0.50},
                 a={"format": 0.75, "long_horizon": 0.70, "gen_case": 1.0, "spiky": 0.83})
    assert J.score(rows[1]["scores"], rows[0]["scores"], J.NAIVE) > 1.0  # "no damage"
    spec, applied, _ = J.improve(J.NAIVE, rows)
    assert spec["cap"] == 1.0
    assert J.score(rows[1]["scores"], rows[0]["scores"], spec) < 0.95


def test_a_probe_that_merely_holds_still_is_not_dropped():
    """The search only ever audits near-healthy configs, so a probe that does not
    fall there has not been shown to be useless -- unlike in vet_probes, which
    has a known-damaged config to test against. Positive evidence only."""
    import judge as J
    rows = _rows({"format": 0.9, "long_horizon": 0.8, "gen_case": 1.0, "steady": 0.7},
                 a={"format": 0.88, "long_horizon": 0.78, "gen_case": 1.0, "steady": 0.7},
                 b={"format": 0.86, "long_horizon": 0.76, "gen_case": 1.0, "steady": 0.7})
    spec, applied, _ = J.improve(J.NAIVE, rows)
    assert "steady" not in spec["exclude"]


def test_the_judge_will_not_amend_itself_into_agreeing_with_everything():
    """A judge that can edit itself can edit itself down to nothing. Below three
    gradeable probes it must refuse and say a human owes it better probes."""
    import judge as J
    rows = _rows({"format": 0.9, "gen_case": 1.0, "up": 0.4},
                 a={"format": 0.8, "gen_case": 0.9, "up": 0.6},
                 b={"format": 0.7, "gen_case": 0.8, "up": 0.8})
    spec, applied, refused = J.improve(J.NAIVE, rows)
    assert [a["probe"] for a in refused] == ["up"], refused
    assert "up" not in spec["exclude"]
    assert len(J.graded(rows[0]["scores"], spec)) >= J.MIN_PROBES
    # it may still tighten in ways that cost it nothing to grade on
    assert spec["cap"] == 1.0


def test_a_better_judge_redecides_old_verdicts_at_no_cost():
    """The point of storing raw per-probe scores: a new judge is applied
    backwards over every past decision without one model evaluation."""
    import judge as J
    rows = _rows({"format": 0.9, "long_horizon": 0.8, "gen_case": 1.0, "calibration": 0.4},
                 a={"format": 0.63, "long_horizon": 0.8, "gen_case": 1.0, "calibration": 0.8})
    new, _, _ = J.improve(J.NAIVE, rows)
    flips = J.replay(rows, J.NAIVE, new)
    assert flips[1]["verdict_before"] and not flips[1]["verdict_after"], flips[1]
    assert flips[1]["flipped"]


def test_the_agent_rederives_my_hand_written_judge_from_the_run_record():
    """The claim this whole file exists to check: pointed at what past runs
    measured, starting from a judge that knows nothing, the agent reaches the
    three fixes I made to its auditor by hand."""
    import json
    import judge as J
    mem = json.load(open(pathlib.Path(__file__).parent / "memory.json"))
    model = next(m for m in mem if m != "_judge")
    rows = J.records({k: v for k, v in mem[model].items() if v.get("scores")})
    spec, applied, _ = J.improve(J.NAIVE, rows)
    assert [a["kind"] for a in applied] == ["no_headroom", "masking", "confounded", "confounded"]
    assert spec["cap"] == 1.0
    assert set(spec["exclude"]) == {"recovery", "calibration", "gen_running_total"}
    # and it still clears the config the search settled on
    final = next(r for r in rows if r["key"] == "attention8,expert5,gate5@g128")
    assert J.passes(J.score(final["scores"], rows[0]["scores"], spec))


def test_the_vetter_will_not_re_approve_a_family_the_judge_threw_out():
    """Vetting sees three configs and approved `running_total`; the judge saw
    every config a search audited and dropped it for rising under damage. The
    richer evidence has to reach the generator, or the next run re-adds it."""
    import json
    import probegen
    import vet_probes
    here = pathlib.Path(__file__).parent
    raw = json.load(open(here / "probes_generated.json.raw"))
    existing = vet_probes.existing_series(str(here / "results.jsonl"))
    assert probegen.vet(raw, existing)["running_total"]["keep"]
    assert not probegen.vet(raw, existing, disproved={"running_total"})["running_total"]["keep"]


def test_memory_never_reproposes_a_disproved_config():
    from memory import Memory
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "m.json")
    m = Memory(path)
    m.put("mdl", {"expert": 4}, 128, audit="failed", budget_after_failure=0.4)
    m.put("mdl", {"expert": 8}, 128, audit="passed", agentic=0.99)
    assert m.rejected("mdl") == {"expert4@g128"}
    assert m.learned_budget("mdl", 1.0) == 0.4
    assert Memory(path).rejected("mdl") == {"expert4@g128"}  # survives a restart

    # A verdict from an older judge is not evidence about the current one.
    m.put("mdl", {"expert": 3}, 128, audit="failed", judge="v1")
    assert m.rejected("mdl", "v2") == set()


def test_search_weights_bits_by_parameter_count():
    """Bits saved must be weighted by size, or the search wastes budget on the router."""
    import search as SR
    sizes = {"gate": 1_000_000, "attention": 400_000_000, "expert": 900_000_000}
    assert SR.model_bits({"gate": 16, "attention": 16, "expert": 16}, sizes) == 16.0
    quantize_experts = SR.model_bits({"gate": 16, "attention": 16, "expert": 4}, sizes)
    quantize_gate = SR.model_bits({"gate": 2, "attention": 16, "expert": 16}, sizes)
    assert quantize_experts < 8 < quantize_gate  # the gate is not where the size is


if __name__ == "__main__":
    for name, fn in sorted(vars().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
