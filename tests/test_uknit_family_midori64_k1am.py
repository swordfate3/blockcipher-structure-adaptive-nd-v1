from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.run_uknit_family_midori64_k1am import (
    training_progress_payload,
    training_row_from_checkpoint,
)
from blockcipher_nd.cli.plot_uknit_family_midori64_k1am import render_k1am_svg
from blockcipher_nd.cli.run_uknit_family_midori64_k1ak import read_tasks
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    INPUT_DIFFERENCE,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_semantic_contrast_k1am import (
    CONTRAST_MARGIN,
    CONTRAST_SCALE,
    EVALUATION_CONDITIONS,
    EXPECTED_SOURCE_DIGESTS,
    ORIENTATIONS,
    ORIENTATION_MODELS,
    ORIENTATION_OPTIONS,
    RUN_ID,
    adjudicate_k1am,
    build_k1am_model,
    build_model_checks,
    candidate_protocol_frozen,
    source_binding_checks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_sbox_transition_k1ak import (
    RUN_ID as K1AK_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_transition_causal_k1al import (
    RUN_ID as K1AL_RUN_ID,
)


BASE_OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/midori64.json",
    "runtime_round_start": 0,
    "runtime_rounds": 2,
    "cipher_round_window_start": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "residual_gate_initial_effective": 0.05,
    "transition_gate_initial_effective": 0.05,
    "transition_value_dim": 20,
    "virtual_projection_slots": 16,
    "active_cell": 8,
    "active_bit_role": 1,
    "input_difference_hex": "0x0000000400000000",
    "topology_corruption_seed": 20260729,
    "semantic_contrast_scale": CONTRAST_SCALE,
    "semantic_contrast_margin": CONTRAST_MARGIN,
}


def test_k1am_models_share_initialization_and_add_no_state() -> None:
    tasks = synthetic_tasks()
    checks = build_model_checks(tasks)
    correct = build_k1am_model(
        task=tasks[0],
        orientation="correct_oriented",
    )
    swapped = build_k1am_model(
        task=tasks[1],
        orientation="swapped_orientation",
    )

    assert all(checks.values())
    assert correct.state_dict().keys() == swapped.state_dict().keys()
    assert tensor_mapping_sha256(correct.state_dict()) == tensor_mapping_sha256(
        swapped.state_dict()
    )
    assert sum(parameter.numel() for parameter in correct.parameters()) == 219_320


def test_k1am_auxiliary_loss_is_bounded_finite_and_backpropagates() -> None:
    model = build_k1am_model(
        task=synthetic_tasks()[0],
        orientation="correct_oriented",
    )
    model.train()
    features = torch.randint(0, 2, (5, 512), dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])

    logits = model(features).squeeze(1)
    auxiliary = model.compute_auxiliary_loss(logits, labels, "mse")

    assert auxiliary is not None
    assert torch.isfinite(auxiliary)
    assert 0.0 <= float(auxiliary.detach()) <= CONTRAST_SCALE * (1 + CONTRAST_MARGIN)
    assert set(model.last_auxiliary_metrics) == {
        "semantic_primary_loss",
        "semantic_counterfactual_loss",
        "semantic_loss_gap",
        "semantic_margin_loss",
        "semantic_violation_rate",
    }
    (logits.mean() + auxiliary).backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_k1am_training_progress_renames_reserved_path() -> None:
    payload = training_progress_payload(
        {"path": "checkpoints/seed6.pt", "selected_checkpoint": "best"}
    )

    assert payload == {
        "checkpoint_path": "checkpoints/seed6.pt",
        "selected_checkpoint": "best",
    }


def test_k1am_complete_checkpoint_reconstructs_training_row(tmp_path: Path) -> None:
    task = synthetic_tasks()[0]
    model = build_k1am_model(task=task, orientation="correct_oriented")
    initial_state_sha256 = tensor_mapping_sha256(model.state_dict())
    checkpoint = tmp_path / "seed6_correct_oriented.pt"
    history = [
        {
            "epoch": float(epoch),
            "train_auxiliary_loss": 0.01,
            "train_semantic_loss_gap": 0.03,
            "val_auc": 0.5 + epoch / 100.0,
        }
        for epoch in range(1, 11)
    ]
    metadata = {
        "epochs": 10,
        "epochs_ran": 10,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "optimizer": "adam",
        "optimizer_state_reused": False,
        "optimizer_state_step_before": 0,
        "optimizer_state_step_after": 640,
        "optimizer_session_call": 1,
        "weight_decay": 1e-5,
        "lr_scheduler": "none",
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "loss": "mse",
        "selected_checkpoint": "best",
        "seed": 6,
        "device": "cpu",
        "checkpoint_output": str(checkpoint),
        "best_checkpoint_metric": 0.6,
        "best_epoch": 10,
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "history": history,
            "final_metrics": {"auc": 0.6},
            "metadata": metadata,
        },
        checkpoint,
    )
    dataset = DifferentialDataset(
        features=np.zeros((2, 512), dtype=np.uint8),
        labels=np.array([0, 1], dtype=np.uint8),
        metadata={},
    )

    row = training_row_from_checkpoint(
        model=model,
        task=task,
        datasets={
            (6, "train_seen"): dataset,
            (6, "cross_key_validation"): dataset,
        },
        checkpoint=checkpoint,
        seed=6,
        orientation="correct_oriented",
        initial_state_sha256=initial_state_sha256,
    )

    assert row["metrics"]["auc"] == 0.6
    assert row["training"]["optimizer_steps"] == 640
    assert row["training"]["checkpoint_output"] == str(checkpoint)
    assert row["selected_state_sha256"] == initial_state_sha256


def test_k1am_plan_and_source_bindings_are_fail_closed() -> None:
    tasks = synthetic_tasks()
    assert candidate_protocol_frozen(tasks)
    changed = deepcopy(tasks)
    changed[0]["model_options"]["semantic_contrast_scale"] = 0.5
    assert candidate_protocol_frozen(changed) is False

    checks = source_binding_checks(
        k1ak_gate={
            "run_id": K1AK_RUN_ID,
            "status": "hold",
            "decision": (
                "innovation1_uknit_family_midori64_k1ak_"
                "sbox_transition_discrimination_failed"
            ),
            "remote_scale": "no",
            "failed_protocol_checks": [],
        },
        k1ak_validation={"run_id": K1AK_RUN_ID, "status": "pass", "errors": []},
        k1ak_controls=k1ak_anchor_rows(),
        dataset_manifest=dataset_manifest(),
        k1al_gate={
            "run_id": K1AL_RUN_ID,
            "status": "pass",
            "decision": (
                "innovation1_uknit_family_midori64_k1al_"
                "transition_and_sbox_causal_use_supported"
            ),
            "remote_scale": "no",
            "failed_protocol_checks": [],
            "failed_research_checks": [],
        },
        k1al_validation={
            "run_id": K1AL_RUN_ID,
            "status": "pass",
            "errors": [],
            "optimizer_steps": 0,
        },
        k1al_results=k1al_rows(),
        source_digests=EXPECTED_SOURCE_DIGESTS,
    )
    assert all(checks.values())


def test_k1am_checked_in_plan_matches_frozen_protocol() -> None:
    plan = Path(
        "configs/experiment/innovation1/"
        "innovation1_uknit_family_midori64_semantic_contrast_"
        "k1am_2048_seed6_seed7.csv"
    )

    assert candidate_protocol_frozen(read_tasks(plan))


def test_k1am_gate_passes_only_when_real_orientation_resolves_substitute() -> None:
    gate = synthetic_gate()

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("paired_semantic_contrast_supported")
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert gate["remote_scale"] == "no"


def test_k1am_gate_holds_when_objective_imposes_arbitrary_preference() -> None:
    anchors = k1ak_anchor_rows()
    for row in anchors:
        if row["condition"] == "wrong_sbox":
            row["auc"] = 0.679
    gate = synthetic_gate(anchors=anchors)

    assert gate["status"] == "hold"
    assert gate["decision"].endswith(
        "semantic_preference_imposed_substitute_unresolved"
    )
    assert not all(
        passed
        for name, passed in gate["research_checks"].items()
        if "beats_k1ak_independent_wrong" in name
    )


def test_k1am_gate_rejects_initialization_or_optimizer_drift() -> None:
    rows = training_rows()
    rows[1]["initial_state_sha256"] = "changed"
    rows[1]["training"]["optimizer_steps"] = 639
    gate = synthetic_gate(training=rows)

    assert gate["status"] == "invalid"
    assert gate["protocol_checks"]["paired_training_initialization_exact"] is False
    assert gate["protocol_checks"]["training_protocol_frozen"] is False


def test_k1am_plot_explains_independent_wrong_substitute(tmp_path: Path) -> None:
    gate = synthetic_gate()
    output = tmp_path / "curves.svg"

    report = render_k1am_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["heatmaps_used_instead_of_overlapping_curves"] is True
    assert report["threshold_outcomes_visible"] is True
    assert "成对语义训练能否消除错误 S盒的替代解" in svg
    assert "错误 S盒独立训练" in svg
    assert "同检查点：正确 - 错误 S盒" in svg


def synthetic_gate(
    *,
    training: list[dict[str, object]] | None = None,
    anchors: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return adjudicate_k1am(
        tasks=synthetic_tasks(),
        training_rows=training_rows() if training is None else training,
        evaluation_rows=evaluation_rows(),
        checkpoint_manifest=checkpoint_manifest(),
        k1ak_controls=k1ak_anchor_rows() if anchors is None else anchors,
        source_checks={"source": True},
        model_checks={"models": True},
    )


def synthetic_tasks() -> list[dict[str, object]]:
    return [
        {
            "cipher_key": "midori64",
            "rounds": 4,
            "seed": seed,
            "model_key": ORIENTATION_MODELS[orientation],
            "samples_per_class": 2048,
            "validation_samples_total": None,
            "pairs_per_sample": 4,
            "input_difference": INPUT_DIFFERENCE,
            "difference_profile": "midori64_k1ah_cell8_r4",
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
            "key_rotation_interval": 0,
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "lr_scheduler": "none",
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "target_epochs": 10,
            "model_options": {
                **BASE_OPTIONS,
                "semantic_contrast_orientation": ORIENTATION_OPTIONS[orientation],
            },
        }
        for seed in EXPECTED_SEEDS
        for orientation in ORIENTATIONS
    ]


def training_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    aucs = {"correct_oriented": 0.68, "swapped_orientation": 0.64}
    for seed in EXPECTED_SEEDS:
        for orientation in ORIENTATIONS:
            rows.append(
                {
                    "run_id": RUN_ID,
                    "model": ORIENTATION_MODELS[orientation],
                    "orientation": orientation,
                    "trainable_parameter_count": 219_320,
                    "rounds": 4,
                    "seed": seed,
                    "input_difference": INPUT_DIFFERENCE,
                    "samples_per_class": 2048,
                    "pairs_per_sample": 4,
                    "negative_mode": "encrypted_random_plaintexts",
                    "semantic_contrast_orientation": ORIENTATION_OPTIONS[orientation],
                    "semantic_contrast_scale": CONTRAST_SCALE,
                    "semantic_contrast_margin": CONTRAST_MARGIN,
                    "initial_state_sha256": f"initial-{seed}",
                    "metrics": {"auc": aucs[orientation]},
                    "history": [
                        {
                            "epoch": epoch,
                            "train_auxiliary_loss": 0.01,
                            "train_semantic_loss_gap": 0.03,
                        }
                        for epoch in range(1, 11)
                    ],
                    "training": {
                        "batch_size": 64,
                        "epochs": 10,
                        "epochs_ran": 10,
                        "optimizer_steps": 640,
                        "optimizer": "adam",
                        "loss": "mse",
                        "checkpoint_metric": "val_auc",
                        "selected_checkpoint": "best",
                        "samples_total": 4096,
                    },
                    "validation": {"samples_total": 2048},
                }
            )
    return rows


def checkpoint_manifest() -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "status": "pass",
        "entries": [
            {
                "seed": seed,
                "orientation": orientation,
                "selected_checkpoint": "best",
                "path": f"checkpoint-{seed}-{orientation}.pt",
                "sha256": f"checkpoint-{seed}-{orientation}",
            }
            for seed in EXPECTED_SEEDS
            for orientation in ORIENTATIONS
        ],
    }


def evaluation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    aucs = {
        "correct_oriented": {
            "correct_runtime": 0.68,
            "wrong_sbox_same_checkpoint": 0.60,
            "transition_branch_off_same_checkpoint": 0.58,
        },
        "swapped_orientation": {
            "correct_runtime": 0.61,
            "wrong_sbox_same_checkpoint": 0.64,
            "transition_branch_off_same_checkpoint": 0.59,
        },
    }
    for seed in EXPECTED_SEEDS:
        for orientation in ORIENTATIONS:
            checkpoint = f"checkpoint-{seed}-{orientation}"
            state = f"state-{seed}-{orientation}"
            for split in EXPECTED_SPLITS:
                dataset = f"dataset-{seed}-{split}"
                row_count = 4096 if split == "train_seen" else 2048
                for condition in EVALUATION_CONDITIONS:
                    exact = condition == "correct_runtime"
                    wrong = condition == "wrong_sbox_same_checkpoint"
                    branch_off = condition == "transition_branch_off_same_checkpoint"
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "seed": seed,
                            "orientation": orientation,
                            "split": split,
                            "condition": condition,
                            "auc": aucs[orientation][condition],
                            "correct_minus_condition_auc": 0.0 if exact else 0.08,
                            "max_abs_probability_delta_from_correct": (
                                0.0 if exact else 0.1
                            ),
                            "mean_abs_probability_delta_from_correct": (
                                0.0 if exact else 0.05
                            ),
                            "checkpoint_sha256": checkpoint,
                            "state_dict_sha256": state,
                            "dataset_sha256": dataset,
                            "composition_sha256": (
                                "wrong-composition" if wrong else "correct-composition"
                            ),
                            "sbox_transition_semantics_sha256": (
                                "wrong-sbox" if wrong else "correct-sbox"
                            ),
                            "transition_branch_enabled": not branch_off,
                            "rows": row_count,
                            "input_bits": 512,
                            "pairs_per_sample": 4,
                            "input_difference": INPUT_DIFFERENCE,
                            "negative_mode": "encrypted_random_plaintexts",
                            "sample_structure": "independent_pairs",
                            "parameter_count": 219_320,
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                            "epochs": 0,
                        }
                    )
    return rows


def k1ak_anchor_rows() -> list[dict[str, object]]:
    return [
        {
            "run_id": K1AK_RUN_ID,
            "seed": seed,
            "split": split,
            "condition": condition,
            "auc": 0.65 if condition == "correct_structure" else 0.66,
        }
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in (
            "correct_structure",
            "wrong_sbox",
            "corrupted_linear",
            "no_structure",
        )
    ]


def dataset_manifest() -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "split": split,
            "cell": 8,
            "input_difference": INPUT_DIFFERENCE,
            "rounds": 4,
            "cache_payloads_present": True,
        }
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
    ]


def k1al_rows() -> list[dict[str, object]]:
    return [
        {"seed": seed, "split": split, "condition": condition}
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in EVALUATION_CONDITIONS
    ]
