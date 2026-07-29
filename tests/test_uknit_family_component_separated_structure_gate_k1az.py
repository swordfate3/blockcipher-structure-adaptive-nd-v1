from __future__ import annotations

from copy import deepcopy

import torch

from blockcipher_nd.cli.plot_uknit_family_component_separated_structure_gate_k1az import (
    render_k1az_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1as import (
    build_candidate as build_k1as_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1ay import (
    build_candidate as build_k1ay_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1az import (
    adjudicate_training,
    load_and_validate_config,
    load_sources,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1av import (
    build_candidate as build_k1aw_candidate,
    migrate_k1at_state,
)


def test_k1az_config_and_sources_freeze_same_budget_authority() -> None:
    config = load_and_validate_config()
    (
        readiness,
        k1as,
        k1av,
        dataset_rows,
        datasets,
        anchors,
        checks,
    ) = load_sources(config)

    assert all(checks.values()), checks
    assert len(dataset_rows) == len(datasets) == 18
    assert len(anchors) == 12
    assert config["training"]["pairs_per_sample"] == 4
    assert k1as["model"]["gate_hidden_dim"] == 12
    assert k1av["model"]["gate_hidden_dim"] == 12


def test_k1az_initial_state_exactly_matches_k1aw() -> None:
    config = load_and_validate_config()
    readiness, k1as, k1av, *_ = load_sources(config)
    cipher = readiness["ciphers"][0]
    seed = int(config["replicas"][0]["initialization_seed"])
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        k1at_initial = build_k1as_candidate(
            cipher, readiness["model"], k1as["model"]
        )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        k1aw_initial = build_k1aw_candidate(
            cipher, readiness["model"], k1av["model"]
        )
    migration = migrate_k1at_state(k1aw_initial, k1at_initial.state_dict())
    assert migration["only_final_projection_expanded"] is True
    candidate = build_k1ay_candidate(
        cipher, readiness["model"], {"gate_hidden_dim": 12}
    )
    candidate.load_state_dict(k1aw_initial.state_dict(), strict=True)
    assert tensor_mapping_sha256(candidate.state_dict()) == tensor_mapping_sha256(
        k1aw_initial.state_dict()
    )


def test_k1az_gate_separates_pass_hold_and_invalid(tmp_path) -> None:
    config = load_and_validate_config()
    training_rows = _synthetic_training_rows()
    evaluation_rows = _synthetic_evaluation_rows()
    checkpoints = _synthetic_checkpoints(tmp_path)
    passed = adjudicate_training(
        config=config,
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        checkpoints=checkpoints,
    )
    assert passed["status"] == "pass"
    assert "training_supported" in passed["decision"]

    weak = deepcopy(evaluation_rows)
    for row in weak:
        if row["condition"] == "sbox_only_mismatch":
            row["auc"] = 0.71
            row["correct_minus_condition_auc"] = 0.0
    held = adjudicate_training(
        config=config,
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=training_rows,
        evaluation_rows=weak,
        checkpoints=checkpoints,
    )
    assert held["status"] == "hold"
    assert held["descriptor_mismatch_gate_all"] is False

    invalid_rows = deepcopy(training_rows)
    invalid_rows[0]["initial_alignment"]["exact_k1aw_initial_state"] = False
    invalid = adjudicate_training(
        config=config,
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=invalid_rows,
        evaluation_rows=evaluation_rows,
        checkpoints=checkpoints,
    )
    assert invalid["status"] == "invalid"
    assert "candidate_geometry_and_exact_k1aw_initial_state" in invalid[
        "failed_protocol_checks"
    ]


def test_k1az_plot_writes_four_panel_chinese_svg(tmp_path) -> None:
    gate = adjudicate_training(
        config=load_and_validate_config(),
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=_synthetic_training_rows(),
        evaluation_rows=_synthetic_evaluation_rows(),
        checkpoints=_synthetic_checkpoints(tmp_path),
    )
    output = tmp_path / "curves.svg"

    report = render_k1az_svg(gate, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["evaluation_rows"] == 60
    assert "分量隔离训练不稳定" in text
    assert "2048/class" in text
    assert "不是正式规模" in text


def _synthetic_training_rows() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "trainable_parameter_count": 219764,
            "state_dict_entries": 55,
            "uses_cipher_identity": False,
            "structure_gate_uses_cipher_identity": False,
            "structure_gate_shared": True,
            "initial_alignment": {"exact_k1aw_initial_state": True},
            "training": {
                "epochs": 10,
                "optimizer_steps": 1920,
                "optimizer_state_step_min": 1920,
                "optimizer_state_step_max": 1920,
                "one_shared_optimizer": True,
                "equal_batches_per_cipher": True,
                "correct_summary_precomputed_once_per_cipher": True,
                "component_separation_enabled": True,
            },
        }
        for replica in (0, 1)
    ]


def _synthetic_evaluation_rows() -> list[dict[str, object]]:
    rows = []
    condition_offsets = {
        "correct_descriptor": 0.0,
        "full_mismatch": -0.01,
        "sbox_only_mismatch": -0.01,
        "linear_only_mismatch": -0.01,
        "dual_path_disabled": -0.02,
    }
    for replica in (0, 1):
        for cipher in ("uknit64", "midori64", "dialga128"):
            for split in ("same_key_fresh", "cross_key_validation"):
                anchor = 0.70
                correct = 0.71
                for condition, offset in condition_offsets.items():
                    rows.append(
                        {
                            "replica": replica,
                            "cipher_key": cipher,
                            "split": split,
                            "condition": condition,
                            "auc": correct + offset,
                            "k1aw_anchor_auc": anchor,
                            "correct_minus_condition_auc": -offset,
                            "effective_edge_gate": 0.1,
                            "effective_transition_gate": 0.1,
                            "training_performed": False,
                            "optimizer_steps": 0,
                            "state_immutable_across_controls": True,
                            "strict_state_dict_load": True,
                            "component_separation_enabled": True,
                            "runtime_structure_held_correct": True,
                            "runtime_structure_cipher_key": cipher,
                        }
                    )
    return rows


def _synthetic_checkpoints(tmp_path) -> dict[int, dict[str, object]]:
    checkpoints = {}
    for replica in (0, 1):
        path = tmp_path / f"replica{replica}.pt"
        path.write_bytes(f"checkpoint-{replica}".encode("ascii"))
        checkpoints[replica] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return checkpoints
