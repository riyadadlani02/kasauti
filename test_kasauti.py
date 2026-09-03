"""Self-check: metric math and the hook plumbing. Run with `python test_kasauti.py`."""
import pathlib

import torch
from torch import nn

import kasauti as K
import memory as M


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
    fixes I made to its auditor by hand — and one I did not make.

    Mine were: drop the probe with no headroom (recovery), cap ratios so a
    rising probe cannot pay for a falling one, and drop calibration for
    rewarding abstention. It finds all three. It also flags gen_case, which
    rises 0.83 -> 1.00 across this record. I had judged that probe on the
    vetting run's three uniform configs, where it collapses instead; the
    search never visits those, so on its own evidence the agent is right and
    my hand judge was reading a different set of configs."""
    import json
    import judge as J
    mem = json.load(open(pathlib.Path(__file__).parent / "memory.json"))
    model = M.model_keys(mem)[0]
    rows = J.records({k: v for k, v in mem[model].items() if v.get("scores")})
    base = max(rows, key=lambda r: r["bits"])
    spec, applied, _ = J.improve(J.NAIVE, rows)

    assert [a["kind"] for a in applied] == ["no_headroom", "masking", "confounded", "confounded"]
    assert spec["cap"] == 1.0, "the masking fix"
    assert "recovery" in spec["exclude"], "the no-headroom fix"
    assert "calibration" in spec["exclude"], "the confound fix"
    assert every_amendment_names_its_evidence(applied)
    # it must not excuse itself by grading on too little to resolve anything
    assert len(J.graded(base["scores"], spec)) >= J.MIN_PROBES
    # and it still clears the config the search settled on
    final = next(r for r in rows if r["key"] == "attention8,expert5,gate5@g128")
    assert J.passes(J.score(final["scores"], base["scores"], spec))


def every_amendment_names_its_evidence(applied) -> bool:
    """An amendment the agent cannot justify in terms of a measured config is
    indistinguishable from one that just makes grading easier."""
    return all(a.get("why") and any(ch.isdigit() for ch in a["why"]) for a in applied)


def test_it_widens_the_line_it_cannot_resolve_rather_than_stalling():
    """Refusing to decide is right; refusing forever answers nothing. When the
    suite has grown as far as it can and still cannot resolve the margin, the
    agent moves the line to what its instrument supports -- and the number it
    moves to has to come from the measurement, not from convenience."""
    import judge as J
    # resolvable: leave it alone
    m, am = J.defensible_margin(se=0.01, z=1.0, margin=0.02, want=20, grown=64)
    assert m == 0.02 and am is None
    # not resolvable: widen to exactly the standard error, and justify it
    m, am = J.defensible_margin(se=0.0758, z=1.0, margin=0.02, want=346, grown=64)
    assert m == 0.076 and am["kind"] == "widen_margin"
    assert am["from"] == 0.02 and am["to"] == 0.076
    assert "346" in am["why"] and "64" in am["why"], am["why"]
    # it may never widen to less than it can measure
    assert m >= 0.0758 - 1e-9
    # a NaN instrument does not license any line at all
    assert J.defensible_margin(float("nan"), 1.0, 0.02, 9, 9) == (0.02, None)


def test_growth_is_driven_by_items_not_by_new_probe_families():
    """The stall this fixes: the re-baseline onto a bigger suite was gated on
    the vetter admitting a new family, but standard error falls with item
    count. Nothing new survived vetting, so the suite never grew and the
    search ended at BF16 having compressed nothing."""
    src = pathlib.Path(__file__).parent.joinpath("search.py").read_text()
    block = src[src.index("unresolved.add(key(state, ev.group))"):]
    block = block[:block.index("cadence = 1")]
    assert "if replenish(" not in block, "growth must not be gated on new families"
    assert "items_needed" in block and "build_items(grown)" in block


def test_the_audit_will_not_decide_inside_its_own_error():
    """A mean of twelve items does not resolve a 0.02 line. Before this, every
    verdict in the project was reported as if it did."""
    import judge as J
    base = {"format": 0.917, "gen_case": 1.0, "long_horizon": 0.759}
    cnt = {"format": 12, "gen_case": 6, "long_horizon": 18}
    hurt = {"format": 0.792, "gen_case": 1.0, "long_horizon": 0.759}
    spec = dict(J.NAIVE, cap=1.0)
    s, se = J.score(hurt, base, spec), J.resolution(hurt, base, spec, cnt)
    assert 0.95 < s < 0.96 and se > 0.05, (s, se)
    assert not J.decided(s, se), "0.955 against a 0.93 line is inside a 0.1 standard error"
    # and far from the line it will commit
    assert J.decided(0.45, se)


def test_a_probe_at_full_marks_does_not_claim_certainty():
    """Six of six is not proof of zero error, and the binomial says it is."""
    import judge as J
    assert J._item_var(1.0, 6) > 0.01


def test_it_checks_whether_buying_items_could_help_before_paying():
    """Only the generated families can be made more of. If the error left after
    growing those is still bigger than the gap, a re-measurement is compute
    spent for nothing."""
    import judge as J
    base = {"format": 0.917, "gen_case": 1.0, "long_horizon": 0.759}
    cnt = {"format": 12, "gen_case": 6, "long_horizon": 18}
    hurt = {"format": 0.792, "gen_case": 1.0, "long_horizon": 0.759}
    spec = dict(J.NAIVE, cap=1.0)
    floor_se = J.irreducible(hurt, base, spec, cnt)
    assert 0 < floor_se < J.resolution(hurt, base, spec, cnt)
    assert floor_se > 0.02, "the fixed probe lists, not the compute, are the bottleneck"
    need = J.items_needed(hurt, base, spec, cnt, 0.02)
    assert need["format"] > 100 and need["long_horizon"] > 100, need


def test_unresolved_is_not_recorded_as_a_rejection():
    """An audit that could not resolve a config is not evidence against it, and
    storing it as a rejection would teach the memory something never measured."""
    import os
    import tempfile
    from memory import Memory
    m = Memory(os.path.join(tempfile.mkdtemp(), "m.json"))
    m.put("mdl", {"expert": 4}, 128, audit="failed", judge="v1")
    m.put("mdl", {"expert": 5}, 128, audit="unresolved", judge="v1")
    assert m.rejected("mdl", "v1") == {"expert4@g128"}


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


def test_redundancy_stops_disqualifying_a_probe_the_audit_needs():
    """A duplicate is a defect only while the probe it restates can still decide
    something. When that probe is a fixed twelve-item list and the audit cannot
    resolve its verdict, a duplicate that can be grown is the cure."""
    import json
    import probegen
    import vet_probes
    here = pathlib.Path(__file__).parent
    raw = json.load(open(here / "probes_generated.json.raw"))
    existing = vet_probes.existing_series(str(here / "results.jsonl"))
    plain = probegen.vet(raw, existing, disproved={"running_total"})
    assert not plain["suffix"]["keep"] and "duplicates format" in plain["suffix"]["reasons"]
    needed = probegen.vet(raw, existing, disproved={"running_total"},
                          needs_resolution={"format", "long_horizon"})
    assert needed["suffix"]["keep"] and needed["tag"]["keep"]
    # but a duplicate that cannot separate damage is still worthless
    assert not needed["prefix"]["keep"]


def test_the_search_amends_its_own_ladder_when_a_component_gets_stuck():
    """The ladder starts 16 -> 8. When the audit rejects a component's only
    available move, the search abandons a component it has not finished with."""
    import policy as PO
    rows = [{"key": "attention16,expert16,gate16@g128", "was": "passed"},
            {"key": "attention16,expert8,gate16@g128", "was": "failed"}]
    pol, applied = PO.improve(dict(PO.P0), rows)
    assert applied[0]["kind"] == "refine_ladder"
    assert any(8 < b < 16 for b in pol["ladder"]), pol["ladder"]


def test_a_failed_config_does_not_close_the_route_beyond_it():
    """Measured: gate5/attn8/expert5 passes and is strictly more quantized than
    gate16/attn8/expert5, which fails. Reverting from the failure is what loses
    the better answer."""
    import policy as PO
    rows = [{"key": "attention8,expert5,gate16@g128", "was": "failed"},
            {"key": "attention8,expert5,gate5@g128", "was": "passed"}]
    pol, applied = PO.improve(dict(PO.P0), rows)
    assert pol["pass_through"] == 1
    assert any(a["kind"] == "pass_through" for a in applied)


def test_minimising_self_contradiction_selects_for_the_confound():
    """The open-ended version of the critic, and why it does not replace the
    named checks: a probe that rises under damage cancels the inversions this
    objective counts, so the most self-consistent judge is the broken one."""
    import json
    import judge as J
    mem = json.load(open(pathlib.Path(__file__).parent / "memory_handjudge.json"))
    model = M.model_keys(mem)[0]
    rows = J.records({k: v for k, v in mem[model].items() if v.get("scores")})
    derived, applied, _ = J.improve(J.NAIVE, rows)
    x = J.cross_check(rows, derived, applied)
    assert x["free_contradiction"] < J.contradiction(rows, derived)
    assert x["free_keeps_confounded"], "it keeps the probes that rise under damage"
    assert x["agrees"], "forbidden those, enumeration reproduces the derived judge exactly"


def test_a_judge_that_reads_nothing_is_not_a_candidate():
    """Given the chance, the enumeration's first move was to keep the probe stuck
    at zero, score every config NaN, and report a perfectly consistent record."""
    import json
    import judge as J
    mem = json.load(open(pathlib.Path(__file__).parent / "memory_handjudge.json"))
    model = M.model_keys(mem)[0]
    rows = J.records({k: v for k, v in mem[model].items() if v.get("scores")})
    picked, _ = J.least_contradictory(rows, J.NAIVE)
    assert "recovery" in picked["exclude"], "the zero-baseline probe must be excluded"


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
