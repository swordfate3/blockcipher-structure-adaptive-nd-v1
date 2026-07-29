from __future__ import annotations

from copy import deepcopy

import torch

from blockcipher_nd.cli.plot_uknit_family_structure_derived_gate_k1as import (
    render_k1as_svg,
)

from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.models.structure.spn.structure_conditioned_gate import (
    STRUCTURE_SUMMARY_DIM,
    hybrid_structure_summary,
    runtime_structure_summary,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_path_contribution_k1ar import (
    CHECKPOINT_FAMILIES,
    FRESH_SPLITS,
    REPLICAS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
    build_runtime_model,
    load_and_validate_config as load_k1ao_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1as import (
    EXPECTED_MISSING_KEYS,
    adjudicate,
    build_candidate,
    load_and_validate_config,
)


def _structures() -> dict[str, object]:
    return {
        "uknit64": load_runtime_spn_descriptor(
            "configs/runtime/spn/uknit64.json", rounds=2, round_start=3
        ).structure,
        "midori64": load_runtime_spn_descriptor(
            "configs/runtime/spn/midori64.json", rounds=2, round_start=0
        ).structure,
        "dialga128": load_runtime_spn_descriptor(
            "configs/runtime/spn/dialga128.json", rounds=2, round_start=2
        ).structure,
    }


def test_k1as_structure_summary_is_fixed_bounded_and_cell_relabel_invariant() -> None:
    summaries = {}
    for cipher, structure in _structures().items():
        summary = runtime_structure_summary(structure)
        relabeled, _ = structure.relabel_cells(
            tuple(reversed(range(structure.cells)))
        )
        assert summary.shape == (STRUCTURE_SUMMARY_DIM,)
        assert torch.isfinite(summary).all()
        assert torch.all((summary >= 0.0) & (summary <= 1.0))
        assert torch.equal(summary, runtime_structure_summary(relabeled))
        summaries[cipher] = summary
    assert not torch.equal(summaries["uknit64"], summaries["midori64"])
    assert not torch.equal(summaries["midori64"], summaries["dialga128"])


def test_k1as_candidate_preserves_disabled_k1ak_path_and_has_shared_gradient() -> None:
    readiness = load_k1ao_config()
    config = load_and_validate_config()
    cipher = readiness["ciphers"][0]
    source = build_runtime_model(cipher, readiness["model"])
    with torch.random.fork_rng():
        torch.manual_seed(config["model"]["initialization_seed"])
        candidate = build_candidate(cipher, readiness["model"], config["model"])
    incompatibility = candidate.load_state_dict(source.state_dict(), strict=False)
    assert set(incompatibility.missing_keys) == EXPECTED_MISSING_KEYS
    assert not incompatibility.unexpected_keys
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 219_752
    assert len(candidate.state_dict()) == 55

    features = torch.randint(0, 2, (4, int(cipher["input_bits"]))).float()
    source.eval()
    candidate.eval()
    with torch.inference_mode():
        expected = source.logits_with_runtime(
            features,
            source.runtime_structure,
            apply_sboxes=True,
        )
        actual = candidate.logits_with_runtime(
            features,
            source.runtime_structure,
            apply_sboxes=True,
            structure_gate_enabled=False,
        )
    assert torch.equal(expected, actual)

    summary = runtime_structure_summary(source.runtime_structure)
    gate = candidate.effective_transition_gate(
        source.runtime_structure,
        summary=summary,
        enabled=True,
    )
    gradients = torch.autograd.grad(
        gate,
        tuple(candidate.backbone.structure_gate.parameters()),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(torch.sum(torch.square(gradient))) for gradient in gradients) > 0.0
    assert candidate.uses_cipher_identity is False
    assert candidate.structure_gate_uses_cipher_identity is False


def test_k1as_mismatch_summaries_change_one_shared_gate() -> None:
    readiness = load_k1ao_config()
    config = load_and_validate_config()
    structures = _structures()
    cipher = readiness["ciphers"][0]
    with torch.random.fork_rng():
        torch.manual_seed(config["model"]["initialization_seed"])
        candidate = build_candidate(cipher, readiness["model"], config["model"])
    correct = runtime_structure_summary(structures["uknit64"])
    controls = (
        runtime_structure_summary(structures["midori64"]),
        hybrid_structure_summary(
            sbox_structure=structures["midori64"],
            linear_structure=structures["uknit64"],
        ),
        hybrid_structure_summary(
            sbox_structure=structures["uknit64"],
            linear_structure=structures["midori64"],
        ),
    )
    correct_gate = candidate.effective_transition_gate(
        structures["uknit64"], summary=correct
    )
    deltas = [
        abs(
            float(
                (
                    correct_gate
                - candidate.effective_transition_gate(
                    structures["uknit64"], summary=summary
                )
                ).detach()
            )
        )
        for summary in controls
    ]
    assert min(deltas) >= config["gates"]["minimum_mismatch_gate_delta"]


def _synthetic_rows(*, mismatch_delta: float = 0.01) -> list[dict]:
    return [
        {
            "checkpoint_family": family,
            "replica": replica,
            "cipher_key": cipher,
            "split": split,
            "checkpoint_missing_keys": sorted(EXPECTED_MISSING_KEYS),
            "checkpoint_unexpected_keys": [],
            "disabled_source_max_abs_logit_delta": 0.0,
            "full_mismatch_gate_delta": mismatch_delta,
            "sbox_only_mismatch_gate_delta": mismatch_delta,
            "linear_only_mismatch_gate_delta": mismatch_delta,
            "full_mismatch_max_abs_logit_delta": mismatch_delta,
            "sbox_only_mismatch_max_abs_logit_delta": mismatch_delta,
            "linear_only_mismatch_max_abs_logit_delta": mismatch_delta,
            "gate_values": {
                "correct_descriptor": 0.1,
                "full_mismatch": 0.2,
                "sbox_only_mismatch": 0.3,
                "linear_only_mismatch": 0.4,
                "descriptor_disabled": 0.05,
            },
            "structure_gate_gradient_norm": 1.0,
            "all_structure_gate_gradients_finite": True,
            "state_immutable": True,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        for family in CHECKPOINT_FAMILIES
        for replica in REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    ]


def test_k1as_gate_passes_only_observable_runtime_contract() -> None:
    config = load_and_validate_config()
    gate = adjudicate(
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        structure_checks={"structure": True},
        rows=_synthetic_rows(),
        config=config,
    )
    assert gate["status"] == "pass"
    assert gate["next_training_authorized"] is True
    assert "K1-AT" in gate["next_action"]

    held = adjudicate(
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        structure_checks={"structure": True},
        rows=_synthetic_rows(mismatch_delta=0.0),
        config=config,
    )
    assert held["status"] == "hold"
    assert held["next_training_authorized"] is False

    broken = deepcopy(_synthetic_rows())
    broken[0]["disabled_source_max_abs_logit_delta"] = 1e-7
    failed = adjudicate(
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        structure_checks={"structure": True},
        rows=broken,
        config=config,
    )
    assert failed["status"] == "fail"
    assert "descriptor_disabled_exactly_replays_source" in failed[
        "failed_protocol_checks"
    ]


def test_k1as_plot_writes_four_panel_chinese_svg(tmp_path, monkeypatch) -> None:
    gate = adjudicate(
        source_checks={"source": True},
        geometry_checks={"geometry": True},
        structure_checks={"structure": True},
        rows=_synthetic_rows(),
        config=load_and_validate_config(),
    )
    summary_rows = [
        {
            "cipher_key": cipher,
            "summary": [index / 34.0 for index in range(34)],
        }
        for cipher in EXPECTED_CIPHERS
    ]
    output = tmp_path / "curves.svg"
    seen = {}
    from matplotlib import pyplot as plt

    savefig = plt.Figure.savefig

    def capture(self, *args, **kwargs):
        seen["axes"] = len(self.axes)
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture)
    report = render_k1as_svg(
        gate,
        _synthetic_rows(),
        {"rows": summary_rows},
        output,
    )
    text = output.read_text(encoding="utf-8")
    assert seen["axes"] == 5  # four panels plus one colorbar axis
    assert report["panels"] == 4
    assert "运行时结构能否安全控制同一条 SPN 转移分支" in text
    assert "2048/class/cipher" in text
    assert "不包含新训练AUC" in text
