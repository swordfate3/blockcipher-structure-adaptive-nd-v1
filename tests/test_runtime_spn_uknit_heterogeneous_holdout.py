from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_uknit_heterogeneous_holdout as a6,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_uknit_heterogeneous_holdout import (
    SOURCE_CIPHERS,
    TARGET_EVALUATIONS,
    TRAINING_ROLES,
    adjudicate_uknit_heterogeneous_holdout,
    load_and_validate_uknit_heterogeneous_holdout_config,
    run_uknit_heterogeneous_holdout,
    run_uknit_heterogeneous_holdout_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_uknit_heterogeneous_holdout_a6_2048_seed0_seed1.json"
)


def test_frozen_a6_config_excludes_uknit_from_source_training() -> None:
    config = load_and_validate_uknit_heterogeneous_holdout_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )

    assert tuple(config["source_ciphers"]) == SOURCE_CIPHERS
    assert config["holdout_cipher"] == "uknit64"
    assert "uknit64" not in config["source_ciphers"]
    assert tuple(config["training_roles"]) == TRAINING_ROLES
    assert tuple(config["target_evaluations"]) == TARGET_EVALUATIONS


def test_frozen_a6_config_rejects_target_leakage(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["source_ciphers"][2] = "uknit64"
    path = tmp_path / "leaked.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="source cipher panel drifted"):
        load_and_validate_uknit_heterogeneous_holdout_config(
            path,
            project_root=ROOT,
            require_readiness=False,
        )


def test_readiness_checks_identifiable_uknit_structure_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_and_validate_uknit_heterogeneous_holdout_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    cache_probe = {
        "passed": True,
        "required_file_count": 54,
        "required_files_present": 54,
        "target_train_referenced": False,
        "historical_target_train_cache_exists": True,
    }
    smoke = {
        "source_task_names": list(SOURCE_CIPHERS),
        "source_task_names_by_role": [list(SOURCE_CIPHERS)] * 2,
        "target_evaluated_after_both_roles": True,
        "target_optimizer_steps": 0,
    }
    monkeypatch.setattr(a6, "_cache_probe", lambda *_args: cache_probe)
    monkeypatch.setattr(a6, "_synthetic_source_only_smoke", lambda *_args: smoke)
    monkeypatch.setattr(
        a6,
        "_read_json",
        lambda _path: {
            "decision": (
                "innovation1_runtime_spn_h1_relation_activity_pooling_invalid"
            ),
            "protocol_valid": False,
            "invalid_reason": "RECTANGLE one-to-one control is unidentifiable",
        },
    )

    readiness = run_uknit_heterogeneous_holdout_readiness(
        config=config,
        project_root=ROOT,
    )

    assert readiness["status"] == "pass"
    assert readiness["target_signature_types"] > 1
    assert readiness["checks"]["target_correct_differs_from_uniform"] is True
    assert readiness["checks"]["target_correct_differs_from_shuffled"] is True
    assert readiness["checks"]["independent_forces_uniform"] is True
    assert readiness["target_training_rows"] == 0

    cache_probe["target_train_referenced"] = True
    failed = run_uknit_heterogeneous_holdout_readiness(
        config=config,
        project_root=ROOT,
    )
    assert failed["status"] == "fail"
    assert failed["checks"]["target_train_cache_not_referenced"] is False


def test_run_uses_same_initialization_and_loads_target_after_both_roles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_and_validate_uknit_heterogeneous_holdout_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    events: list[tuple[str, int | None]] = []
    initial_hashes: dict[int, list[str]] = {0: [], 1: []}
    current_seed = 0

    def load_source_tasks(*_args: Any, seed: int, **kwargs: Any) -> list[Any]:
        nonlocal current_seed
        current_seed = seed
        assert tuple(kwargs["source_ciphers"]) == SOURCE_CIPHERS
        events.append(("source", seed))
        return [SimpleNamespace(name=name) for name in SOURCE_CIPHERS]

    def train(model: Any, tasks: list[Any], *_args: Any, **_kwargs: Any) -> Any:
        initial_hashes[current_seed].append(a6._state_dict_sha256(model.state_dict()))
        events.append(("train", current_seed))
        metrics = {
            name: {
                "auc": 0.60,
                "loss": 0.4,
                "accuracy": 0.6,
                "best_accuracy": 0.61,
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
            history=[{"epoch": 1, "val_macro_auc": 0.6}],
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
        assert kwargs["holdout_cipher"] == "uknit64"
        assert events[-2:] == [("train", seed), ("train", seed)]
        events.append(("target", seed))
        return object()

    monkeypatch.setattr(a6, "_load_source_tasks", load_source_tasks)
    monkeypatch.setattr(a6, "train_runtime_spn_joint", train)
    monkeypatch.setattr(a6, "_load_target_validation", load_target)
    monkeypatch.setattr(
        a6,
        "_evaluate_target",
        lambda *_args, **_kwargs: {
            "auc": 0.60,
            "loss": 0.4,
            "accuracy": 0.6,
            "best_accuracy": 0.61,
            "calibrated_threshold": 0.5,
            "rows": 2048.0,
        },
    )
    monkeypatch.setattr(
        a6,
        "_read_json",
        lambda path: (
            {
                "status": "pass",
                "decision": config["source"]["readiness_required_decision"],
                "checks": {"ready": True},
            }
            if "readiness" in str(path)
            else {
                "decision": config["source"]["a5_required_decision"],
                "protocol_valid": False,
            }
        ),
    )

    payload = run_uknit_heterogeneous_holdout(
        config=config,
        config_path=CONFIG,
        output_root=tmp_path / "run",
        project_root=ROOT,
    )

    assert all(len(set(hashes)) == 1 for hashes in initial_hashes.values())
    assert payload["validation"]["status"] == "pass"
    assert payload["validation"]["result_rows"] == 28
    assert payload["validation"]["target_training_rows"] == 0
    assert payload["validation"]["target_optimizer_steps"] == 0
    target_rows = [row for row in payload["rows"] if row["row_kind"] == "holdout_target"]
    assert len(target_rows) == 12
    assert {row["optimizer_steps"] for row in target_rows} == {0}
    for seed in (0, 1):
        candidate_hashes = {
            row["checkpoint_sha256"]
            for row in target_rows
            if row["seed"] == seed
            and row["evaluation"] in a6.CANDIDATE_COUNTERFACTUALS
        }
        assert len(candidate_hashes) == 1


def test_resumed_role_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "role.pt"
    checkpoint.write_bytes(b"tampered")
    role = {
        "checkpoint_sha256": "not-the-file-hash",
        "seed": 0,
        "role": "correct_pooling",
        "pooling_mode": "correct",
        "parameter_count": 442466,
        "metadata": {"task_names": list(SOURCE_CIPHERS)},
    }

    with pytest.raises(ValueError, match="checkpoint hash drifted"):
        a6._validate_role_checkpoint(
            role,
            checkpoint_path=checkpoint,
            seed=0,
            role="correct_pooling",
            pooling_mode="correct",
            config_hash="config",
            initial_hash="initial",
        )


def test_gate_supports_full_partial_unsupported_and_invalid_outcomes() -> None:
    supported = adjudicate_uknit_heterogeneous_holdout(_gate_payload())
    assert supported["status"] == "pass"
    assert supported["decision"].endswith("heterogeneous_holdout_supported")

    partial_payload = _gate_payload()
    partial_payload["target_auc"]["1"]["uniform_trained_anchor"] = 0.62
    partial = adjudicate_uknit_heterogeneous_holdout(partial_payload)
    assert partial["status"] == "hold"
    assert partial["decision"].endswith("heterogeneous_holdout_partial")

    unsupported_payload = _gate_payload()
    unsupported_payload["target_auc"]["1"]["candidate_correct"] = 0.51
    unsupported = adjudicate_uknit_heterogeneous_holdout(unsupported_payload)
    assert unsupported["status"] == "hold"
    assert unsupported["decision"].endswith("heterogeneous_holdout_not_supported")

    invalid_payload = _gate_payload()
    invalid_payload["validation"] = {"status": "fail"}
    invalid = adjudicate_uknit_heterogeneous_holdout(invalid_payload)
    assert invalid["status"] == "invalid"
    assert invalid["decision"].endswith("heterogeneous_holdout_invalid")


def _gate_payload() -> dict[str, Any]:
    config = load_and_validate_uknit_heterogeneous_holdout_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    targets = {
        "candidate_correct": 0.60,
        "candidate_corrupted_target": 0.58,
        "candidate_no_topology_target": 0.57,
        "candidate_uniform_same_checkpoint": 0.58,
        "candidate_shuffled_same_checkpoint": 0.57,
        "uniform_trained_anchor": 0.595,
    }
    return {
        "config": copy.deepcopy(config),
        "validation": {"status": "pass"},
        "target_auc": {str(seed): copy.deepcopy(targets) for seed in (0, 1)},
        "source_macro_auc": {
            str(seed): {
                "correct_pooling": 0.60,
                "uniform_pooling_anchor": 0.602,
            }
            for seed in (0, 1)
        },
        "conflict_projections_by_role_seed": {
            str(seed): {role: 2 for role in TRAINING_ROLES} for seed in (0, 1)
        },
    }
