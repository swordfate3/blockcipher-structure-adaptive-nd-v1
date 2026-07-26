from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import blockcipher_nd.tasks.innovation1.runtime_spn_dialga_holdout as a8
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_holdout import (
    SOURCE_CIPHERS,
    adjudicate_dialga_holdout,
    build_wrong_sbox_structure,
    load_and_validate_dialga_holdout_config,
    run_dialga_holdout,
    run_dialga_holdout_readiness,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    _load_structures,
    load_and_validate_holdout_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_dialga_holdout_a8_2048_seed0_seed1.json"
)


def test_frozen_a8_config_and_wrong_sbox_contract() -> None:
    config = load_and_validate_dialga_holdout_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    base = load_and_validate_holdout_config(
        ROOT / config["source"]["protocol_config_path"]
    )
    structures = _load_structures(base)
    target = structures["dialga128"]
    wrong = build_wrong_sbox_structure(target, structures["gift64"])

    assert tuple(config["source_ciphers"]) == SOURCE_CIPHERS
    assert not target.sbox_truth_bits.equal(wrong.sbox_truth_bits)
    assert target.linear_matrices.equal(wrong.linear_matrices)
    assert target.inverse_linear_matrices.equal(wrong.inverse_linear_matrices)


def test_real_a8_readiness_passes_without_target_training() -> None:
    config = load_and_validate_dialga_holdout_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )

    readiness = run_dialga_holdout_readiness(
        config=config,
        project_root=ROOT,
    )

    assert readiness["status"] == "pass"
    assert all(readiness["checks"].values())
    assert readiness["target_atomic_gf2_types"] == 16
    assert readiness["covered_atomic_gf2_types"] == 16
    assert readiness["exact_source_sbox_overlap"] == 0
    assert readiness["target_training_rows"] == 0
    assert readiness["target_optimizer_steps"] == 0
    assert readiness["cache_probe"]["target_train_referenced"] is False


def test_a8_run_trains_both_roles_before_target_and_shares_counterfactual_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_and_validate_dialga_holdout_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    events: list[tuple[str, int]] = []
    initial_hashes: dict[int, list[str]] = {0: [], 1: []}
    current_seed = 0

    def load_sources(*_args: Any, seed: int, **kwargs: Any) -> list[Any]:
        nonlocal current_seed
        current_seed = seed
        assert tuple(kwargs["source_ciphers"]) == SOURCE_CIPHERS
        events.append(("source", seed))
        return [SimpleNamespace(name=name) for name in SOURCE_CIPHERS]

    def train(model: Any, tasks: list[Any], *_args: Any, **_kwargs: Any) -> Any:
        initial_hashes[current_seed].append(a8._state_dict_sha256(model.state_dict()))
        events.append(("train", current_seed))
        metrics = {
            name: {
                "auc": 0.70,
                "loss": 0.3,
                "accuracy": 0.6,
                "best_accuracy": 0.62,
                "calibrated_threshold": 0.5,
                "rows": 2048.0,
            }
            for name in SOURCE_CIPHERS
        }
        diagnostics = {
            "task_conflict_projection_counts": {name: 1 for name in SOURCE_CIPHERS},
            "task_representation_gradient_mean_l2": {
                name: 1.0 for name in SOURCE_CIPHERS
            },
            "task_gradient_scale_mean": {name: 1.0 for name in SOURCE_CIPHERS},
            "task_gradient_scale_observations": {
                name: 1 for name in SOURCE_CIPHERS
            },
        }
        return SimpleNamespace(
            history=[{"epoch": 1, "val_macro_auc": 0.70}],
            train_metrics=metrics,
            validation_metrics=metrics,
            metadata={
                "best_epoch": 1,
                "task_names": [task.name for task in tasks],
                "selected_checkpoint": "best",
                "gradient_combination": (
                    "representation_l2_equalized_pcgrad_fixed_order"
                ),
            },
            gradient_diagnostics=diagnostics,
        )

    def load_target(*_args: Any, seed: int, **kwargs: Any) -> object:
        assert kwargs["holdout_cipher"] == "dialga128"
        assert events[-2:] == [("train", seed), ("train", seed)]
        events.append(("target", seed))
        return object()

    monkeypatch.setattr(a8, "_load_source_tasks", load_sources)
    monkeypatch.setattr(a8, "train_runtime_spn_joint", train)
    monkeypatch.setattr(a8, "_load_target_validation", load_target)
    monkeypatch.setattr(
        a8,
        "_evaluate_target",
        lambda *_args, **_kwargs: {
            "auc": 0.70,
            "loss": 0.3,
            "accuracy": 0.6,
            "best_accuracy": 0.62,
            "calibrated_threshold": 0.5,
            "rows": 2048.0,
        },
    )
    monkeypatch.setattr(
        a8,
        "_read_json",
        lambda path: (
            {
                "status": "pass",
                "decision": (
                    "innovation1_runtime_spn_dialga_holdout_readiness_passed"
                ),
                "checks": {"ready": True},
            }
            if "readiness" in str(path)
            else {
                "status": "pass",
                "decision": config["source"]["d1_required_decision"],
                "protocol_checks": {"valid": True},
                "aucs": {
                    "seed0": {"correct": 0.95},
                    "seed1": {"correct": 0.96},
                },
            }
        ),
    )

    payload = run_dialga_holdout(
        config=config,
        config_path=CONFIG,
        output_root=tmp_path / "run",
        project_root=ROOT,
    )

    assert all(len(set(hashes)) == 1 for hashes in initial_hashes.values())
    assert payload["validation"]["status"] == "pass"
    assert payload["validation"]["result_rows"] == 26
    assert payload["validation"]["target_training_rows"] == 0
    source_rows = [row for row in payload["rows"] if row["row_kind"] == "source_validation"]
    rectangle_names = {
        row["cipher_display_name"]
        for row in source_rows
        if row["cipher"] == "rectangle80"
    }
    assert rectangle_names == {"RECTANGLE-80 r6（训练来源）"}
    target_rows = [row for row in payload["rows"] if row["row_kind"] == "holdout_target"]
    assert len(target_rows) == 10
    assert {row["optimizer_steps"] for row in target_rows} == {0}
    for seed in (0, 1):
        hashes = {
            row["checkpoint_sha256"]
            for row in target_rows
            if row["seed"] == seed
            and row["evaluation"] in a8.CANDIDATE_COUNTERFACTUALS
        }
        assert len(hashes) == 1


def test_a8_gate_supports_pass_hold_and_invalid() -> None:
    payload = _gate_payload()
    passed = adjudicate_dialga_holdout(payload)
    assert passed["status"] == "pass"
    assert passed["decision"].endswith("dialga_holdout_supported")

    payload = _gate_payload()
    payload["target_auc"]["1"]["candidate_wrong_sbox_target"] = 0.70
    held = adjudicate_dialga_holdout(payload)
    assert held["status"] == "hold"
    assert held["decision"].endswith("dialga_holdout_not_supported")

    payload = _gate_payload()
    payload["validation"] = {"status": "fail"}
    invalid = adjudicate_dialga_holdout(payload)
    assert invalid["status"] == "invalid"


def test_resumed_a8_role_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "role.pt"
    checkpoint.write_bytes(b"tampered")
    role = {
        "checkpoint_sha256": "not-the-file-hash",
        "seed": 0,
        "role": "correct_candidate",
        "relation_mode": "true",
        "parameter_count": 442466,
        "initial_state_sha256": "initial",
        "metadata": {"task_names": list(SOURCE_CIPHERS)},
    }

    with pytest.raises(ValueError, match="checkpoint hash drifted"):
        a8._validate_role_checkpoint(
            role,
            checkpoint_path=checkpoint,
            seed=0,
            role="correct_candidate",
            relation_mode="true",
            config_hash="config",
            initial_hash="initial",
        )


def _gate_payload() -> dict[str, Any]:
    config = load_and_validate_dialga_holdout_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    target = {
        "candidate_correct": 0.70,
        "candidate_corrupted_target": 0.65,
        "candidate_no_topology_target": 0.52,
        "candidate_wrong_sbox_target": 0.60,
        "no_topology_trained_anchor": 0.53,
    }
    source = {
        "correct_candidate": 0.65,
        "no_topology_anchor": 0.64,
    }
    return {
        "config": config,
        "validation": {"status": "pass"},
        "target_auc": {"0": dict(target), "1": dict(target)},
        "source_macro_auc": {"0": dict(source), "1": dict(source)},
        "oracle_auc": {"0": 0.95, "1": 0.96},
        "conflict_projections_by_role_seed": {
            "0": {"correct_candidate": 1, "no_topology_anchor": 1},
            "1": {"correct_candidate": 1, "no_topology_anchor": 1},
        },
    }
