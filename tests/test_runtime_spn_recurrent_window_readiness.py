from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import json
from pathlib import Path

import pytest

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.runtime_spn_recurrent_window_readiness import (
    EXPECTED_ROLES,
    ROLE_MODEL_OPTIONS,
    ROLE_MODELS,
    UKNIT_VALIDATION_KEY,
    adjudicate_recurrent_window_readiness,
    build_recurrent_window_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1.csv"
)


def _task(seed: int, role: str) -> dict[str, object]:
    return {
        "cipher_key": "uknit64",
        "model_key": ROLE_MODELS[role],
        "architecture": f"uKNIT recurrent-window {role} seed{seed}",
        "rounds": 5,
        "seed": seed,
        "samples_per_class": 2048,
        "train_samples_total": None,
        "validation_samples_total": None,
        "dataset_label_mode": "balanced_per_class",
        "pairs_per_sample": 4,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "key_rotation_interval": 0,
        "sample_structure": "independent_pairs",
        "input_difference": 0x40,
        "difference_profile": "",
        "difference_member": "",
        "train_key": 0,
        "validation_key": UKNIT_VALIDATION_KEY,
        "loss": "mse",
        "learning_rate": 0.0001,
        "optimizer": "adam",
        "optimizer_state_transition": "reset_each_stage",
        "weight_decay": 0.00001,
        "lr_scheduler": "none",
        "max_learning_rate": None,
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "target_epochs": 10,
        "pretrain_rounds": None,
        "pretrain_round_sequence": (),
        "pretrain_epochs": 0,
        "model_options": deepcopy(ROLE_MODEL_OPTIONS[role]),
    }


def _tasks() -> list[dict[str, object]]:
    return [_task(seed, role) for seed in (0, 1) for role in EXPECTED_ROLES]


def test_recurrent_window_readiness_passes_exact_heterogeneous_panel() -> None:
    manifests, gate = build_recurrent_window_readiness(
        run_id="readiness",
        tasks=_tasks(),
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert gate["build_errors"] == []
    candidate = next(
        row for row in manifests if row["seed"] == 0 and row["role"] == "candidate"
    )
    repeated = next(
        row for row in manifests if row["seed"] == 0 and row["role"] == "repeat_last"
    )
    assert candidate["runtime_structure_unique_transition_count"] > 1
    assert repeated["runtime_structure_unique_transition_count"] == 1
    assert (
        candidate["runtime_structure_transition_sha256s"][-1]
        == repeated["runtime_structure_transition_sha256s"][-1]
    )
    assert (
        candidate["runtime_structure_window_sha256"]
        != repeated["runtime_structure_window_sha256"]
    )
    assert candidate["probe_output_shape"] == [2, 1]
    assert candidate["probe_output_finite"] is True
    assert candidate["probe_gradients_finite"] is True
    assert (
        candidate["probe_gradient_tensor_count"]
        == candidate["trainable_parameter_tensor_count"]
    )
    assert candidate["probe_output_sha256"] != repeated["probe_output_sha256"]


def test_recurrent_window_readiness_accepts_the_real_frozen_plan() -> None:
    tasks = tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )

    manifests, gate = build_recurrent_window_readiness(
        run_id="real-plan",
        tasks=tasks,
    )

    assert gate["status"] == "pass"
    assert len(manifests) == 10
    assert all(
        row["data_protocol"]["validation_key"] == UKNIT_VALIDATION_KEY
        for row in manifests
    )


def test_recurrent_window_readiness_replays_json_persisted_manifests() -> None:
    manifests, initial = build_recurrent_window_readiness(
        run_id="in-memory",
        tasks=_tasks(),
    )
    persisted = json.loads(json.dumps(manifests))

    replayed = adjudicate_recurrent_window_readiness(
        run_id="persisted",
        manifests=persisted,
    )

    assert initial["status"] == "pass"
    assert replayed["status"] == "pass"
    assert replayed["protocol_checks"] == initial["protocol_checks"]


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    (
        (
            lambda tasks: tasks[1]["model_options"].update(
                {"runtime_structure_path": "configs/runtime/spn/skinny64.json"}
            ),
            "candidate_window_is_heterogeneous",
        ),
        (
            lambda tasks: tasks[2]["model_options"].update(
                {"runtime_structure_window_control": "full"}
            ),
            "two_seed_five_role_panel",
        ),
        (
            lambda tasks: tasks[1].update({"samples_per_class": 4096}),
            "same_data_protocol",
        ),
        (
            lambda tasks: tasks[1].update({"negative_mode": "random_ciphertext"}),
            "strict_encrypted_random_plaintext_negatives",
        ),
        (
            lambda tasks: tasks[1].update({"target_epochs": 11}),
            "same_training_protocol",
        ),
        (
            lambda tasks: tasks[1].update({"optimizer": "sgd"}),
            "same_training_protocol",
        ),
        (
            lambda tasks: tasks[5].update({"seed": 2}),
            "two_seed_five_role_panel",
        ),
        (
            lambda tasks: tasks[1]["model_options"].update(
                {"round_window_mode": "last_transition"}
            ),
            "two_seed_five_role_panel",
        ),
        (
            lambda tasks: tasks[1]["model_options"].update(
                {"pair_embedding_dim": 64}
            ),
            "equal_parameter_count_and_shapes",
        ),
        (
            lambda tasks: tasks[4].update(
                {"model_key": "runtime_spn_e4_equivariant_true"}
            ),
            "two_seed_five_role_panel",
        ),
    ),
)
def test_recurrent_window_readiness_rejects_invalid_panels(
    mutate: Callable[[list[dict[str, object]]], object],
    failed_check: str,
) -> None:
    tasks = _tasks()
    mutate(tasks)

    _, gate = build_recurrent_window_readiness(run_id="invalid", tasks=tasks)

    assert gate["status"] == "fail"
    assert gate["protocol_checks"][failed_check] is False


def test_recurrent_window_readiness_rejects_final_transition_hash_drift() -> None:
    manifests, initial = build_recurrent_window_readiness(
        run_id="initial",
        tasks=_tasks(),
    )
    assert initial["status"] == "pass"
    repeated = next(
        row for row in manifests if row["seed"] == 0 and row["role"] == "repeat_last"
    )
    repeated["runtime_structure_transition_sha256s"][-1] = "f" * 64

    gate = adjudicate_recurrent_window_readiness(
        run_id="hash-drift",
        manifests=manifests,
    )

    assert gate["status"] == "fail"
    assert (
        gate["protocol_checks"]["candidate_repeat_last_final_transition_equal"]
        is False
    )
    assert gate["protocol_checks"]["structure_evidence_seed_invariant"] is False


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    (
        (
            lambda candidate, repeated: repeated.update(
                {
                    "runtime_structure_window_sha256": candidate[
                        "runtime_structure_window_sha256"
                    ]
                }
            ),
            "candidate_repeat_last_window_distinct",
        ),
        (
            lambda candidate, repeated: repeated.update(
                {
                    "runtime_structure_unique_transition_count": 2,
                    "runtime_structure_homogeneous": False,
                }
            ),
            "repeat_last_window_is_homogeneous",
        ),
        (
            lambda candidate, repeated: candidate.update(
                {"runtime_structure_loaded_rounds": 3}
            ),
            "runtime_rounds_equal_two",
        ),
        (
            lambda candidate, repeated: candidate.update(
                {"probe_output_finite": False}
            ),
            "forward_probe_shape_and_finiteness",
        ),
        (
            lambda candidate, repeated: candidate.update(
                {"probe_gradients_finite": False}
            ),
            "backward_probe_finite_with_full_coverage",
        ),
        (
            lambda candidate, repeated: repeated.update(
                {"probe_output_sha256": candidate["probe_output_sha256"]}
            ),
            "probe_interventions_change_candidate_logits",
        ),
    ),
)
def test_recurrent_window_readiness_rejects_manifest_evidence_drift(
    mutate: Callable[[dict[str, object], dict[str, object]], object],
    failed_check: str,
) -> None:
    manifests, initial = build_recurrent_window_readiness(
        run_id="initial",
        tasks=_tasks(),
    )
    assert initial["status"] == "pass"
    candidate = next(
        row for row in manifests if row["seed"] == 0 and row["role"] == "candidate"
    )
    repeated = next(
        row for row in manifests if row["seed"] == 0 and row["role"] == "repeat_last"
    )
    mutate(candidate, repeated)

    gate = adjudicate_recurrent_window_readiness(
        run_id="manifest-drift",
        manifests=manifests,
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"][failed_check] is False


def test_recurrent_window_readiness_records_model_build_failure() -> None:
    tasks = _tasks()
    tasks[1]["model_options"]["runtime_rounds"] = 99  # type: ignore[index]

    manifests, gate = build_recurrent_window_readiness(
        run_id="build-failure",
        tasks=tasks,
    )

    assert len(manifests) == 9
    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["all_models_constructed"] is False
    assert gate["build_errors"][0]["error_type"] == "ValueError"
