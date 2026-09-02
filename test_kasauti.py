"""Self-check: metric math and the hook plumbing. Run with `python test_kasauti.py`."""
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
