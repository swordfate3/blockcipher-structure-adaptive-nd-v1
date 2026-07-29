from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from blockcipher_nd.cli.plot_uknit_family_dual_path_structure_modulation_k1aw import (
    render_k1aw_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1as import (
    build_candidate as build_k1as_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1av import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_STATE_ENTRIES,
    build_candidate,
    migrate_k1at_state,
)
from blockcipher_nd.tasks.innovation1.uknit_family_dual_path_structure_modulation_k1aw import (
    CONTROL_CONDITIONS,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_STEPS_PER_REPLICA,
    adjudicate_training,
    load_and_validate_config,
    load_sources,
)


def test_k1aw_config_and_sources_freeze_same_budget_authority() -> None:
    config = load_and_validate_config()
    (
        readiness,
        _k1as,
        _k1av,
        dataset_rows,
        datasets,
        anchors,
        source_checks,
    ) = load_sources(config)

    assert config["training"]["pairs_per_sample"] == 4
    assert config["training"]["optimizer_steps_total_per_replica"] == 1920
    assert config["controls"]["expected_rows"] == EXPECTED_EVALUATION_ROWS
    assert "16-pair expansion" in config["blocked_actions"]
    assert len(readiness["ciphers"]) == 3
    assert len(dataset_rows) == len(datasets) == 18
    assert len(anchors) == 12
    assert all(source_checks.values()), source_checks


def test_k1aw_initial_migration_preserves_k1at_and_expands_only_output() -> None:
    config = load_and_validate_config()
    readiness, k1as, k1av, *_ = load_sources(config)
    cipher_configs = {
        str(row["cipher_key"]): row for row in readiness["ciphers"]
    }
    initialization_seed = int(config["replicas"][0]["initialization_seed"])
    with torch.random.fork_rng():
        torch.manual_seed(initialization_seed)
        source = build_k1as_candidate(
            cipher_configs["uknit64"], readiness["model"], k1as["model"]
        )
    source_sha256 = tensor_mapping_sha256(source.state_dict())
    anchor_results_path = (
        Path(__file__).resolve().parents[1]
        / config["same_budget_anchor"]["root"]
        / "results.jsonl"
    )
    anchor_rows = [
        json.loads(line)
        for line in anchor_results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    anchor_row = next(row for row in anchor_rows if int(row["replica"]) == 0)
    assert source_sha256 == anchor_row["initial_state_sha256"]

    with torch.random.fork_rng():
        torch.manual_seed(initialization_seed)
        candidate = build_candidate(
            cipher_configs["uknit64"], readiness["model"], k1av["model"]
        )
    migration = migrate_k1at_state(candidate, source.state_dict())

    assert migration["only_final_projection_expanded"] is True
    assert migration["transition_row_exact"] is True
    assert migration["new_edge_row_finite_nonzero"] is True
    assert sum(parameter.numel() for parameter in candidate.parameters()) == (
        EXPECTED_PARAMETER_COUNT
    )
    assert len(candidate.state_dict()) == EXPECTED_STATE_ENTRIES


def test_k1aw_gate_separates_pass_hold_and_protocol_invalid(tmp_path: Path) -> None:
    config = load_and_validate_config()
    checkpoints = _synthetic_checkpoints(tmp_path)
    training_rows = _synthetic_training_rows()
    evaluation_rows = _synthetic_evaluation_rows()

    passed = adjudicate_training(
        config=config,
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        checkpoints=checkpoints,
    )
    assert passed["status"] == "pass"
    assert passed["cross_key_macro_retention_both_replicas"] is True
    assert passed["per_panel_no_harm_all"] is True
    assert passed["descriptor_mismatch_gate_all"] is True
    assert passed["remote_scale"] == "no"

    weak = deepcopy(evaluation_rows)
    changed_panels = 0
    for row in weak:
        if row["condition"] == "sbox_only_mismatch" and changed_panels < 3:
            row["auc"] = 0.7095
            changed_panels += 1
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
    assert held["mismatch_results"]["sbox_only_mismatch"]["passing_panels"] == 9
    assert "channel-orientation" in held["next_action"]

    invalid_rows = deepcopy(evaluation_rows)
    invalid_rows[0]["runtime_structure_held_correct"] = False
    invalid = adjudicate_training(
        config=config,
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=training_rows,
        evaluation_rows=invalid_rows,
        checkpoints=checkpoints,
    )
    assert invalid["status"] == "invalid"
    assert (
        "runtime_structure_held_correct_for_all_controls"
        in invalid["failed_protocol_checks"]
    )


def test_k1aw_plot_writes_four_panel_chinese_svg(tmp_path: Path) -> None:
    gate = adjudicate_training(
        config=load_and_validate_config(),
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=_synthetic_training_rows(),
        evaluation_rows=_synthetic_evaluation_rows(),
        checkpoints=_synthetic_checkpoints(tmp_path),
    )
    output = tmp_path / "curves.svg"

    report = render_k1aw_svg(gate, _synthetic_evaluation_rows(), output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["evaluation_rows"] == 60
    assert "宏平均提升，但尚未学稳正确结构语义" in text
    assert "2048/class" in text
    assert "不是正式规模" in text


def _synthetic_training_rows() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
            "state_dict_entries": EXPECTED_STATE_ENTRIES,
            "uses_cipher_identity": False,
            "structure_gate_uses_cipher_identity": False,
            "structure_gate_shared": True,
            "initial_migration": {
                "only_final_projection_expanded": True,
                "transition_row_exact": True,
                "new_edge_row_finite_nonzero": True,
            },
            "training": {
                "epochs": 10,
                "optimizer_steps": EXPECTED_STEPS_PER_REPLICA,
                "optimizer_state_step_min": EXPECTED_STEPS_PER_REPLICA,
                "optimizer_state_step_max": EXPECTED_STEPS_PER_REPLICA,
                "one_shared_optimizer": True,
                "equal_batches_per_cipher": True,
                "correct_summary_precomputed_once_per_cipher": True,
            },
        }
        for replica in (0, 1)
    ]


def _synthetic_evaluation_rows() -> list[dict[str, object]]:
    aucs = {
        "correct_descriptor": 0.71,
        "full_mismatch": 0.69,
        "sbox_only_mismatch": 0.69,
        "linear_only_mismatch": 0.69,
        "dual_path_disabled": 0.70,
    }
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "split": split,
            "condition": condition,
            "auc": aucs[condition],
            "k1at_anchor_auc": 0.70,
            "effective_edge_gate": 0.1,
            "effective_transition_gate": 0.1,
            "runtime_structure_cipher_key": cipher,
            "runtime_structure_held_correct": True,
            "training_performed": False,
            "optimizer_steps": 0,
            "state_immutable_across_controls": True,
            "strict_state_dict_load": True,
        }
        for replica in (0, 1)
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
        for condition in CONTROL_CONDITIONS
    ]


def _synthetic_checkpoints(tmp_path: Path) -> dict[int, dict[str, object]]:
    checkpoints = {}
    for replica in (0, 1):
        path = tmp_path / f"replica{replica}.pt"
        path.write_bytes(f"checkpoint-{replica}".encode("ascii"))
        checkpoints[replica] = {"path": str(path), "sha256": file_sha256(path)}
    return checkpoints
