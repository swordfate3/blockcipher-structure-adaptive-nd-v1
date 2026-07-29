from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.cli.plot_uknit_family_multicipher_inverse_norm_k1aq import (
    render_k1aq_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_inverse_norm_k1aq import (
    adjudicate,
    load_and_validate_config,
    load_authority,
    scaled_mse_loss,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao_training import (
    CONTROL_CONDITIONS,
    FRESH_SPLITS,
)


def test_k1aq_config_freezes_same_budget_inverse_norm_protocol() -> None:
    config = load_and_validate_config()

    assert config["training"]["optimizer_steps_total_per_replica"] == 1920
    assert config["training"]["pairs_per_sample"] == 4
    assert config["training"]["loss_scaling"].startswith("fixed_inverse_k1ap")
    for replica in config["replicas"]:
        product = 1.0
        for scale in replica["loss_scales"].values():
            product *= scale
        assert abs(product - 1.0) <= 1e-12


def test_k1aq_authority_rederives_all_scales_and_baselines() -> None:
    config = load_and_validate_config()

    _readiness, dataset_rows, datasets, anchors, baseline_rows, checks = (
        load_authority(config)
    )

    assert all(checks.values()), checks
    assert len(dataset_rows) == len(datasets) == 18
    assert len(anchors) == 12
    assert len(baseline_rows) == 36


def test_k1aq_scaled_loss_changes_only_the_frozen_multiplier() -> None:
    logits = torch.tensor([[0.2], [-0.3]], requires_grad=True)
    labels = torch.tensor([[1.0], [0.0]])

    raw, scaled = scaled_mse_loss(logits, labels, scale=2.5)

    assert torch.allclose(scaled, raw * 2.5)
    scaled.backward()
    assert logits.grad is not None


def test_k1aq_gate_supports_broad_partial_recovery(tmp_path: Path) -> None:
    baseline = synthetic_controls(mode="baseline")
    candidate = synthetic_controls(mode="partial")

    gate = adjudicate(
        source_checks={"source": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=candidate,
        baseline_rows=baseline,
        checkpoints=synthetic_checkpoints(tmp_path),
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("inverse_norm_partial_recovery_supported")
    assert gate["target_improved_count"] == 8
    assert gate["advance_gate"] is True
    assert gate["full_support_gate"] is False


def test_k1aq_gate_requires_target_improvement_not_macro_average(
    tmp_path: Path,
) -> None:
    baseline = synthetic_controls(mode="baseline")
    candidate = synthetic_controls(mode="no_target_gain")

    gate = adjudicate(
        source_checks={"source": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=candidate,
        baseline_rows=baseline,
        checkpoints=synthetic_checkpoints(tmp_path),
    )

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("inverse_norm_scaling_not_supported")
    assert gate["target_improved_count"] == 0
    assert gate["advance_gate"] is False


def test_k1aq_plot_requires_complete_twelve_panel_gate(tmp_path: Path) -> None:
    baseline = synthetic_controls(mode="baseline")
    candidate = synthetic_controls(mode="partial")
    gate = adjudicate(
        source_checks={"source": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=candidate,
        baseline_rows=baseline,
        checkpoints=synthetic_checkpoints(tmp_path),
    )
    output = tmp_path / "curves.svg"

    report = render_k1aq_svg(gate, output)

    assert output.is_file()
    assert report["comparison_panels"] == 12
    assert report["formal_scale_claim_present"] is False


def synthetic_training_rows() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "training": {
                "epochs": 10,
                "optimizer_steps": 1920,
                "optimizer_state_step_min": 1920,
                "optimizer_state_step_max": 1920,
                "unchanged_sequential_batch_order": True,
            },
        }
        for replica in (0, 1)
    ]


def synthetic_controls(*, mode: str) -> list[dict[str, object]]:
    rows = []
    for replica in (0, 1):
        for cipher in EXPECTED_CIPHERS:
            for split in FRESH_SPLITS:
                baseline_correct = 0.60 if cipher != "dialga128" else 0.90
                if mode == "baseline":
                    correct = baseline_correct
                    anchor = baseline_correct + (0.05 if cipher != "dialga128" else 0.0)
                elif mode == "partial":
                    correct = baseline_correct + (0.02 if cipher != "dialga128" else 0.0)
                    anchor = baseline_correct + (0.05 if cipher != "dialga128" else 0.0)
                elif mode == "no_target_gain":
                    correct = baseline_correct + (0.0 if cipher != "dialga128" else 0.05)
                    anchor = baseline_correct + (0.05 if cipher != "dialga128" else 0.0)
                else:
                    raise ValueError(mode)
                aucs = {
                    "correct_runtime": correct,
                    "wrong_sbox_same_checkpoint": correct - 0.02,
                    "transition_branch_off_same_checkpoint": correct - 0.02,
                }
                for condition in CONTROL_CONDITIONS:
                    rows.append(
                        {
                            "replica": replica,
                            "cipher_key": cipher,
                            "split": split,
                            "condition": condition,
                            "auc": aucs[condition],
                            "anchor_auc": anchor,
                            "training_performed": False,
                            "optimizer_steps": 0,
                            "state_immutable_across_controls": True,
                        }
                    )
    return rows


def synthetic_checkpoints(tmp_path: Path) -> dict[int, dict[str, object]]:
    checkpoints = {}
    for replica in (0, 1):
        path = tmp_path / f"replica{replica}.pt"
        path.write_bytes(f"checkpoint-{replica}".encode("ascii"))
        checkpoints[replica] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return checkpoints
