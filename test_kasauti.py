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


def test_verdict_bands():
    assert (K.verdict(0.01, 0.06, 0.15), K.verdict(0.10, 0.06, 0.15),
            K.verdict(0.2, 0.06, 0.15)) == ("LOW", "MEDIUM", "HIGH")


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


if __name__ == "__main__":
    for name, fn in sorted(vars().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
