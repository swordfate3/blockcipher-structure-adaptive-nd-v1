from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.data.differential import (
    DifferentialDataset,
    DiskDifferentialDataset,
)
from blockcipher_nd.models.structure.spn.present_integral_multiset import (
    PresentIntegralFlatLinear,
    PresentIntegralPaperMbconvAnchor,
    PresentIntegralStructuredResidualCandidate,
    integral_input_bits,
)
from blockcipher_nd.tasks.innovation2.high_round_integral_data import (
    BIT_ORDER,
    NEGATIVE_MODE,
    PROTOCOL_VERSION,
    SPLIT_KEY_OFFSETS,
    IntegralMultisetCacheConfig,
    ProgressCallback,
    build_integral_multiset_sample,
    fixed_parity_weight_scores,
    make_integral_multiset_cache,
)
from blockcipher_nd.training import (
    TrainingConfig,
    evaluate_binary_classifier,
    train_binary_classifier,
)
from blockcipher_nd.training.metrics import (
    best_threshold_accuracy_and_threshold,
    binary_auc,
)


@dataclass(frozen=True)
class HighRoundIntegralExperimentConfig:
    run_id: str
    output_root: Path
    cache_root: Path
    rounds: int
    train_rows: int
    validation_rows: int
    test_rows: int
    multiset_count: int
    epochs: int
    batch_size: int
    seed: int = 0
    base_channels: int = 16
    head_bits: int = 64
    block_count: int = 1
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"
    cache_chunk_size: int = 256
    gate_mode: str = "readiness"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.rounds not in {5, 7, 8, 9}:
            raise ValueError("rounds must be one of 5, 7, 8, or 9")
        for name in (
            "train_rows",
            "validation_rows",
            "test_rows",
            "multiset_count",
            "epochs",
            "batch_size",
            "base_channels",
            "head_bits",
            "block_count",
            "cache_chunk_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("train_rows", "validation_rows", "test_rows"):
            if int(getattr(self, name)) % 2:
                raise ValueError(f"{name} must be even")
        if self.gate_mode not in {"readiness", "diagnostic", "bridge"}:
            raise ValueError("gate_mode must be readiness, diagnostic, or bridge")


def run_high_round_integral_experiment(
    config: HighRoundIntegralExperimentConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    split_rows = {
        "train": config.train_rows,
        "validation": config.validation_rows,
        "test": config.test_rows,
    }
    splits: dict[str, DiskDifferentialDataset] = {}
    for split, total_rows in split_rows.items():
        splits[split] = make_integral_multiset_cache(
            IntegralMultisetCacheConfig(
                split=split,
                rounds=config.rounds,
                total_rows=total_rows,
                multiset_count=config.multiset_count,
                seed=config.seed,
                cache_root=config.cache_root,
                chunk_size=config.cache_chunk_size,
            ),
            progress_callback=progress_callback,
        )

    dataset_summary = summarize_high_round_splits(config, splits)
    model_specs = (
        ("anchor", "wu_guo_paper_family_mbconv", False),
        ("candidate", "present_integral_structured_residual", False),
        ("linear", "same_input_flat_linear", False),
        ("control", "present_integral_structured_residual_shuffled", True),
    )
    rows: list[dict[str, Any]] = []
    model_seeds = {
        "anchor": config.seed + 100,
        "candidate": config.seed + 200,
        "linear": config.seed + 300,
        "control": config.seed + 200,
    }
    for role, model_name, shuffle_labels in model_specs:
        model_seed = model_seeds[role]
        rows.append(
            _fit_model_row(
                config,
                role=role,
                model_name=model_name,
                shuffle_labels=shuffle_labels,
                model_seed=model_seed,
                train_dataset=splits["train"],
                validation_dataset=splits["validation"],
                test_dataset=splits["test"],
                progress_callback=progress_callback,
            )
        )

    fixed_baselines = evaluate_fixed_parity_baselines(
        splits["test"],
        multiset_count=config.multiset_count,
    )
    fixed_baselines["untrained_structured_candidate"] = (
        evaluate_untrained_candidate_baseline(
            config,
            splits["test"],
            model_seed=model_seeds["candidate"],
        )
    )
    gate = adjudicate_high_round_integral(
        config,
        rows=rows,
        dataset_summary=dataset_summary,
        fixed_baselines=fixed_baselines,
    )
    return {
        "rows": rows,
        "gate": gate,
        "dataset_summary": dataset_summary,
        "fixed_baselines": fixed_baselines,
    }


def summarize_high_round_splits(
    config: HighRoundIntegralExperimentConfig,
    splits: dict[str, DiskDifferentialDataset],
) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    all_balanced = True
    all_complete = True
    all_disk = True
    all_reference_views_expected = True
    all_binary = True
    for name, dataset in splits.items():
        labels = np.asarray(dataset.labels)
        features = dataset.features
        expected_half = len(labels) // 2
        positive_rows = int(labels.sum())
        negative_rows = int(len(labels) - positive_rows)
        reference_bits = np.asarray(features[: min(32, len(features))]).reshape(
            -1,
            config.multiset_count,
            2,
            16,
            64,
        )[:, :, :, 0, :]
        sample_features = np.asarray(features[: min(32, len(features))])
        labels_balanced = positive_rows == negative_rows == expected_half
        expected_invs_state = Present80.inverse_sbox_layer(0)
        expected_invs_bits = np.array(
            [(expected_invs_state >> bit) & 1 for bit in range(64)],
            dtype=np.uint8,
        )
        reference_views_expected = bool(
            np.all(reference_bits[:, :, 0, :] == 0)
            and np.all(reference_bits[:, :, 1, :] == expected_invs_bits)
        )
        features_binary = bool(np.all((sample_features == 0) | (sample_features == 1)))
        cache_complete = dataset.metadata.get("status") == "complete" and int(
            dataset.metadata.get("rows_generated", -1)
        ) == len(labels)
        all_balanced &= labels_balanced
        all_complete &= cache_complete
        all_disk &= isinstance(dataset, DiskDifferentialDataset)
        all_reference_views_expected &= reference_views_expected
        all_binary &= features_binary
        split_summaries[name] = {
            "total_rows": len(labels),
            "samples_per_class": expected_half,
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "labels_balanced": labels_balanced,
            "cache_complete": cache_complete,
            "cache_status": dataset.metadata.get("cache_status"),
            "cache_dir": str(dataset.cache_dir),
            "cache_identity": dataset.metadata.get("cache_identity"),
            "reference_c0_views_expected": reference_views_expected,
            "sampled_features_binary": features_binary,
            "key_index_offset": dataset.metadata.get("key_index_offset"),
        }

    fixture = protocol_fixture(config)
    key_ranges_disjoint = len(set(SPLIT_KEY_OFFSETS.values())) == len(SPLIT_KEY_OFFSETS)
    status_passed = all(
        (
            all_balanced,
            all_complete,
            all_disk,
            all_reference_views_expected,
            all_binary,
            key_ranges_disjoint,
            all(fixture.values()),
        )
    )
    return {
        "status": "pass" if status_passed else "fail",
        "task": "innovation2_present_high_round_integral_multiset",
        "protocol_version": PROTOCOL_VERSION,
        "cipher": "PRESENT-80",
        "rounds": config.rounds,
        "multisets_per_sample": config.multiset_count,
        "texts_per_multiset": 16,
        "input_bits": integral_input_bits(config.multiset_count),
        "bit_order": BIT_ORDER,
        "negative_definition": NEGATIVE_MODE,
        "key_sampling": "one unique master key per sample",
        "key_splits_disjoint_by_construction": key_ranges_disjoint,
        "paper_tensor_concat_assumption": "spatial_axis_1",
        "all_labels_balanced": all_balanced,
        "all_caches_complete": all_complete,
        "all_splits_disk_backed": all_disk,
        "all_reference_c0_views_expected": all_reference_views_expected,
        "all_sampled_features_binary": all_binary,
        "protocol_fixture": fixture,
        "splits": split_summaries,
    }


def protocol_fixture(
    config: HighRoundIntegralExperimentConfig,
) -> dict[str, bool]:
    positive = build_integral_multiset_sample(
        rounds=config.rounds,
        multiset_count=config.multiset_count,
        label=1,
        seed=config.seed,
        split="train",
        row_index=1,
    )
    negative = build_integral_multiset_sample(
        rounds=config.rounds,
        multiset_count=config.multiset_count,
        label=0,
        seed=config.seed,
        split="train",
        row_index=0,
    )
    positive_low_nibbles = positive.plaintexts & np.uint64(0xF)
    positive_high = positive.plaintexts & np.uint64(0xFFFFFFFFFFFFFFF0)
    positive_structure = all(
        np.array_equal(
            positive_low_nibbles[index],
            np.arange(16, dtype=np.uint64),
        )
        and np.unique(positive_high[index]).size == 1
        for index in range(config.multiset_count)
    )
    negative_not_forced_structure = any(
        not np.array_equal(
            negative.plaintexts[index] & np.uint64(0xF),
            np.arange(16, dtype=np.uint64),
        )
        or np.unique(negative.plaintexts[index] & np.uint64(0xFFFFFFFFFFFFFFF0)).size
        != 1
        for index in range(config.multiset_count)
    )
    expected_keys = {
        build_integral_multiset_sample(
            rounds=config.rounds,
            multiset_count=1,
            label=0,
            seed=config.seed,
            split=split,
            row_index=0,
        ).key
        for split in SPLIT_KEY_OFFSETS
    }
    return {
        "positive_low_nibble_enumerates_0_to_15": bool(positive_structure),
        "negative_plaintexts_not_forced_into_integral_structure": bool(
            negative_not_forced_structure
        ),
        "positive_reference_views_expected": bool(
            np.all(positive.views[:, 0, 0] == 0)
            and np.all(positive.views[:, 1, 0] == Present80.inverse_sbox_layer(0))
        ),
        "negative_reference_views_expected": bool(
            np.all(negative.views[:, 0, 0] == 0)
            and np.all(negative.views[:, 1, 0] == Present80.inverse_sbox_layer(0))
        ),
        "split_fixture_keys_distinct": len(expected_keys) == len(SPLIT_KEY_OFFSETS),
        "feature_length_matches_protocol": len(positive.features)
        == integral_input_bits(config.multiset_count),
    }


def evaluate_fixed_parity_baselines(
    dataset: DifferentialDataset,
    *,
    multiset_count: int,
) -> dict[str, Any]:
    labels = np.asarray(dataset.labels, dtype=np.uint8)
    scores = fixed_parity_weight_scores(
        dataset.features,
        multiset_count=multiset_count,
    )
    result: dict[str, Any] = {}
    for name, values in scores.items():
        best_accuracy, threshold = best_threshold_accuracy_and_threshold(
            labels,
            values,
        )
        result[name] = {
            "auc": binary_auc(labels, values),
            "best_accuracy": best_accuracy,
            "best_threshold": threshold,
            "positive_mean": float(values[labels == 1].mean()),
            "negative_mean": float(values[labels == 0].mean()),
        }
    return result


def evaluate_untrained_candidate_baseline(
    config: HighRoundIntegralExperimentConfig,
    dataset: DifferentialDataset,
    *,
    model_seed: int,
) -> dict[str, float]:
    torch.manual_seed(model_seed)
    model = _build_model(config, "present_integral_structured_residual")
    metrics = evaluate_binary_classifier(
        model,
        dataset,
        batch_size=config.batch_size,
        device=config.device,
    )
    return {
        "auc": metrics["auc"],
        "best_accuracy": metrics["calibrated_accuracy"],
        "accuracy": metrics["accuracy"],
        "loss": metrics["loss"],
        "model_seed": float(model_seed),
    }


def adjudicate_high_round_integral(
    config: HighRoundIntegralExperimentConfig,
    *,
    rows: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    fixed_baselines: dict[str, Any],
) -> dict[str, Any]:
    by_role = {str(row["role"]): row for row in rows}
    required_roles = {"anchor", "candidate", "linear", "control"}
    finite_model_metrics = all(
        np.isfinite(float(row[key]))
        for row in rows
        for key in ("test_accuracy", "test_auc", "test_calibrated_accuracy")
    )
    finite_fixed_metrics = all(
        np.isfinite(float(metrics[key]))
        for metrics in fixed_baselines.values()
        for key in ("auc", "best_accuracy")
    )
    readiness_checks = {
        "dataset_summary_passed": dataset_summary["status"] == "pass",
        "four_model_roles_present": set(by_role) == required_roles,
        "all_model_metrics_finite": finite_model_metrics,
        "all_fixed_baseline_metrics_finite": finite_fixed_metrics,
        "strict_negative_definition": dataset_summary["negative_definition"]
        == NEGATIVE_MODE,
        "disk_backed_cache_complete": bool(
            dataset_summary["all_caches_complete"]
            and dataset_summary["all_splits_disk_backed"]
        ),
        "reference_c0_views_expected": bool(
            dataset_summary["all_reference_c0_views_expected"]
        ),
        "key_splits_disjoint": bool(
            dataset_summary["key_splits_disjoint_by_construction"]
        ),
        "protocol_fixture_passed": all(dataset_summary["protocol_fixture"].values()),
        "shuffled_control_uses_shuffled_train_and_validation": bool(
            by_role.get("control", {}).get("fit_train_labels_shuffled")
            and by_role.get("control", {}).get("fit_validation_labels_shuffled")
        ),
    }
    anchor_accuracy = float(by_role["anchor"]["test_accuracy"])
    anchor_auc = float(by_role["anchor"]["test_auc"])
    candidate_accuracy = float(by_role["candidate"]["test_accuracy"])
    candidate_auc = float(by_role["candidate"]["test_auc"])
    shuffled_auc = float(by_role["control"]["test_auc"])
    shuffled_fit_validation_auc = float(by_role["control"]["fit_validation_auc"])
    fixed_total = fixed_baselines["negative_total_parity_weight"]
    untrained_candidate = fixed_baselines["untrained_structured_candidate"]
    untrained_candidate_auc = float(untrained_candidate["auc"])
    architecture_prior_auc = max(
        0.5 + abs(untrained_candidate_auc - 0.5),
        0.5 + abs(shuffled_auc - 0.5),
    )
    fixed_parity_oriented = {
        name: 0.5 + abs(float(metrics["auc"]) - 0.5)
        for name, metrics in fixed_baselines.items()
        if name.startswith("negative_")
    }
    strongest_fixed_parity_name = max(
        fixed_parity_oriented,
        key=fixed_parity_oriented.__getitem__,
    )
    strongest_fixed_parity_auc = fixed_parity_oriented[
        strongest_fixed_parity_name
    ]
    diagnostic_checks = {
        "r5_anchor_accuracy_at_least_0_90": config.rounds != 5
        or anchor_accuracy >= 0.90,
        "shuffled_fit_validation_auc_within_0_07_of_chance": abs(
            shuffled_fit_validation_auc - 0.5
        )
        <= 0.07,
        "r5_candidate_beats_architecture_prior_by_0_10": config.rounds != 5
        or candidate_auc - architecture_prior_auc >= 0.10,
        "r8_metrics_finite": config.rounds != 8
        or all(
            np.isfinite(value)
            for value in (
                anchor_accuracy,
                anchor_auc,
                candidate_accuracy,
                candidate_auc,
            )
        ),
    }
    readiness_passed = all(readiness_checks.values())
    diagnostic_passed = all(diagnostic_checks.values())
    bridge_plan_checks = {
        "rounds_is_8": config.rounds == 8,
        "train_total_rows_is_262144": config.train_rows == 262144,
        "validation_total_rows_is_32768": config.validation_rows == 32768,
        "test_total_rows_is_65536": config.test_rows == 65536,
        "multiset_count_is_2": config.multiset_count == 2,
        "epochs_is_5": config.epochs == 5,
        "batch_size_is_128": config.batch_size == 128,
        "base_channels_is_16": config.base_channels == 16,
        "head_bits_is_256": config.head_bits == 256,
        "block_count_is_2": config.block_count == 2,
        "dropout_is_0_1": bool(np.isclose(config.dropout, 0.1)),
        "learning_rate_is_1e_3": bool(
            np.isclose(config.learning_rate, 1e-3)
        ),
        "weight_decay_is_1e_5": bool(np.isclose(config.weight_decay, 1e-5)),
        "seed_is_frozen_bridge_seed": config.seed in {0, 1},
        "remote_cuda_device_requested": config.device.startswith("cuda"),
    }
    bridge_signal_checks = {
        "shuffled_fit_validation_auc_within_0_03_of_chance": abs(
            shuffled_fit_validation_auc - 0.5
        )
        <= 0.03,
        "candidate_test_auc_at_least_0_53": candidate_auc >= 0.53,
        "candidate_beats_architecture_prior_by_0_01": candidate_auc
        - architecture_prior_auc
        >= 0.01,
        "candidate_beats_strongest_oriented_fixed_parity_by_0_01": candidate_auc
        - strongest_fixed_parity_auc
        >= 0.01,
    }
    if config.gate_mode == "readiness":
        status = "pass" if readiness_passed else "fail"
        decision = (
            "innovation2_high_round_integral_readiness_passed"
            if readiness_passed
            else "innovation2_high_round_integral_readiness_invalid"
        )
        next_action = (
            "Run the frozen 8192/2048/4096-total-row r5/r7/r8 local diagnostic ladder."
            if readiness_passed
            else "Repair the exact failing cache, fixture, split, or control check before any diagnostic."
        )
    elif config.gate_mode == "bridge" and not all(bridge_plan_checks.values()):
        status = "fail"
        decision = "innovation2_high_round_integral_bridge_plan_mismatch"
        next_action = (
            "Repair the frozen bridge configuration before remote launch or "
            "result interpretation; do not compare a mismatched run to the bridge gate."
        )
    elif config.gate_mode == "bridge" and (
        not readiness_passed
        or not bridge_signal_checks[
            "shuffled_fit_validation_auc_within_0_03_of_chance"
        ]
    ):
        status = "fail"
        decision = "innovation2_high_round_integral_bridge_invalid_control"
        next_action = (
            "Audit cache, split, labels, and shuffled-fit validation behavior; "
            "do not increase data or epochs."
        )
    elif config.gate_mode == "bridge" and all(bridge_signal_checks.values()):
        status = "pass"
        decision = "innovation2_high_round_integral_bridge_advance"
        next_action = (
            "Prepare the 2^21-total-row, 50-epoch paper-reference run and an "
            "independent seed; keep r9 and GIFT stopped."
        )
    elif config.gate_mode == "bridge":
        status = "hold"
        decision = "innovation2_high_round_integral_bridge_stop"
        next_action = (
            "Stop mechanical sample scaling and audit the paper's missing Nf, "
            "two-multiset join, block count, and learning-rate schedule; do not run r9 or GIFT."
        )
    elif (
        not readiness_passed
        or not diagnostic_checks["shuffled_fit_validation_auc_within_0_07_of_chance"]
    ):
        status = "fail"
        decision = "innovation2_high_round_integral_invalid_control"
        next_action = "Audit label, cache, split, or shuffled-control leakage; do not increase data or epochs."
    elif config.rounds == 5 and not diagnostic_passed:
        status = "hold"
        decision = "innovation2_high_round_integral_r5_protocol_mismatch"
        next_action = "Resolve bit order, tensor concatenation, training schedule, or architecture mismatch before r7/r8 scale."
    else:
        status = "pass"
        decision = "innovation2_high_round_integral_local_round_complete"
        next_action = "Combine the r5/r7/r8 diagnostic gates; remote scale is allowed only if r5 calibrates and the round slope is plausible."
    return {
        "status": status,
        "decision": decision,
        "run_id": config.run_id,
        "gate_mode": config.gate_mode,
        "rounds": config.rounds,
        "readiness_checks": readiness_checks,
        "diagnostic_checks": diagnostic_checks,
        "bridge_plan_checks": bridge_plan_checks,
        "bridge_signal_checks": bridge_signal_checks,
        "metrics": {
            "anchor_test_accuracy": anchor_accuracy,
            "anchor_test_auc": anchor_auc,
            "candidate_test_accuracy": candidate_accuracy,
            "candidate_test_auc": candidate_auc,
            "candidate_anchor_accuracy_delta": candidate_accuracy - anchor_accuracy,
            "candidate_anchor_auc_delta": candidate_auc - anchor_auc,
            "linear_test_auc": float(by_role["linear"]["test_auc"]),
            "shuffled_test_auc": shuffled_auc,
            "shuffled_fit_validation_auc": shuffled_fit_validation_auc,
            "untrained_candidate_test_auc": untrained_candidate_auc,
            "architecture_prior_oriented_auc": architecture_prior_auc,
            "candidate_architecture_prior_auc_delta": candidate_auc
            - architecture_prior_auc,
            "fixed_total_parity_auc": float(fixed_total["auc"]),
            "fixed_total_parity_best_accuracy": float(fixed_total["best_accuracy"]),
            "strongest_oriented_fixed_parity_name": strongest_fixed_parity_name,
            "strongest_oriented_fixed_parity_auc": strongest_fixed_parity_auc,
            "candidate_strongest_fixed_parity_auc_delta": candidate_auc
            - strongest_fixed_parity_auc,
        },
        "paper_alignment": {
            "headline_multisets": 2,
            "paper_headline_r8_accuracy": 0.5732,
            "paper_epochs": 50,
            "paper_batch_size": 2000,
            "paper_training_total_rows": 1 << 21,
            "paper_validation_total_rows": 1 << 17,
            "paper_test_total_rows": 1 << 17,
            "paper_repetitions_best_selected": 10,
            "paper_tensor_concat_assumption": "spatial_axis_1",
            "exact_reproduction": False,
            "missing_from_paper": [
                "Nf",
                "dropout_rate",
                "exact_mbconv_dense_block_count",
                "adam_initial_learning_rate",
                "learning_rate_min_max",
            ],
        },
        "next_action": next_action,
        "claim_scope": (
            f"{'remote data-scarcity bridge' if config.gate_mode == 'bridge' else 'local'} "
            f"PRESENT-80 r{config.rounds} Wu/Guo paper-family protocol "
            f"{config.gate_mode}; not exact reproduction and not paper-scale"
        ),
    }


def _fit_model_row(
    config: HighRoundIntegralExperimentConfig,
    *,
    role: str,
    model_name: str,
    shuffle_labels: bool,
    model_seed: int,
    train_dataset: DiskDifferentialDataset,
    validation_dataset: DiskDifferentialDataset,
    test_dataset: DiskDifferentialDataset,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    fit_dataset: DifferentialDataset = train_dataset
    fit_validation_dataset: DifferentialDataset = validation_dataset
    if shuffle_labels:
        shuffled_train = np.random.default_rng(model_seed + 77).permutation(
            np.asarray(train_dataset.labels)
        )
        fit_dataset = DiskDifferentialDataset(
            features=train_dataset.features,
            labels=shuffled_train.astype(np.uint8, copy=False),
            metadata={**train_dataset.metadata, "fit_train_labels_shuffled": True},
            cache_dir=train_dataset.cache_dir,
        )
        shuffled_validation = np.random.default_rng(model_seed + 88).permutation(
            np.asarray(validation_dataset.labels)
        )
        fit_validation_dataset = DiskDifferentialDataset(
            features=validation_dataset.features,
            labels=shuffled_validation.astype(np.uint8, copy=False),
            metadata={
                **validation_dataset.metadata,
                "fit_validation_labels_shuffled": True,
            },
            cache_dir=validation_dataset.cache_dir,
        )

    torch.manual_seed(model_seed)
    model = _build_model(config, model_name)
    _emit(
        progress_callback,
        "high_round_model_start",
        role=role,
        model=model_name,
        model_seed=model_seed,
        fit_train_labels_shuffled=shuffle_labels,
        fit_validation_labels_shuffled=shuffle_labels,
    )
    training_config = TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=model_seed,
        device=config.device,
        optimizer="adam",
        weight_decay=config.weight_decay,
        lr_scheduler="none",
        checkpoint_metric="val_auc",
        restore_best_checkpoint=True,
        loss="mse",
        train_eval_interval=1,
    )

    def model_progress(event: str, payload: dict[str, Any]) -> None:
        _emit(
            progress_callback,
            event,
            role=role,
            model=model_name,
            **payload,
        )

    training_result = train_binary_classifier(
        model,
        fit_dataset,
        fit_validation_dataset,
        training_config,
        progress_callback=model_progress,
    )
    true_train_metrics = evaluate_binary_classifier(
        model,
        train_dataset,
        batch_size=config.batch_size,
        device=config.device,
    )
    true_validation_metrics = evaluate_binary_classifier(
        model,
        validation_dataset,
        batch_size=config.batch_size,
        device=config.device,
    )
    test_metrics = evaluate_binary_classifier(
        model,
        test_dataset,
        batch_size=config.batch_size,
        device=config.device,
    )
    execution_scope = (
        "remote data-scarcity bridge"
        if config.gate_mode == "bridge"
        else f"local {config.gate_mode}"
    )
    row: dict[str, Any] = {
        "run_id": config.run_id,
        "cipher": "PRESENT-80",
        "structure": "SPN",
        "rounds": config.rounds,
        "seed": config.seed,
        "model_seed": model_seed,
        "role": role,
        "task": "innovation2_present_high_round_integral_multiset",
        "model": model_name,
        "selected_model": model_name,
        "samples_per_class": config.train_rows // 2,
        "train_total_rows": config.train_rows,
        "validation": {
            "total_rows": config.validation_rows,
            "samples_per_class": config.validation_rows // 2,
        },
        "test": {
            "total_rows": config.test_rows,
            "samples_per_class": config.test_rows // 2,
        },
        "pairs_per_sample": config.multiset_count * 16,
        "multisets_per_sample": config.multiset_count,
        "texts_per_multiset": 16,
        "input_bits": integral_input_bits(config.multiset_count),
        "input_view": "wu_guo_invp_invs_cj_xor_c0",
        "negative_mode": NEGATIVE_MODE,
        "key_sampling": "one unique PRESENT-80 master key per sample",
        "fit_train_labels_shuffled": shuffle_labels,
        "fit_validation_labels_shuffled": shuffle_labels,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "loss": "mse",
        "optimizer": "adam",
        "history": training_result.history,
        "train_loss": true_train_metrics["loss"],
        "train_accuracy": true_train_metrics["accuracy"],
        "train_auc": true_train_metrics["auc"],
        "val_loss": true_validation_metrics["loss"],
        "val_accuracy": true_validation_metrics["accuracy"],
        "val_auc": true_validation_metrics["auc"],
        "val_calibrated_accuracy": true_validation_metrics["calibrated_accuracy"],
        "fit_validation_loss": training_result.final_metrics["loss"],
        "fit_validation_accuracy": training_result.final_metrics["accuracy"],
        "fit_validation_auc": training_result.final_metrics["auc"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_calibrated_accuracy": test_metrics["calibrated_accuracy"],
        "training": training_result.metadata,
        "paper_tensor_concat_assumption": "spatial_axis_1",
        "claim_scope": (
            f"PRESENT-80 r{config.rounds} {execution_scope}; "
            "paper-family protocol only, not exact reproduction and not paper-scale"
        ),
    }
    _emit(
        progress_callback,
        "high_round_model_done",
        role=role,
        model=model_name,
        test_accuracy=row["test_accuracy"],
        test_auc=row["test_auc"],
    )
    return row


def _build_model(
    config: HighRoundIntegralExperimentConfig,
    model_name: str,
) -> nn.Module:
    common = {
        "multiset_count": config.multiset_count,
        "base_channels": config.base_channels,
        "head_bits": config.head_bits,
        "block_count": config.block_count,
        "dropout": config.dropout,
    }
    if model_name == "wu_guo_paper_family_mbconv":
        return PresentIntegralPaperMbconvAnchor(**common)
    if model_name.startswith("present_integral_structured_residual"):
        return PresentIntegralStructuredResidualCandidate(**common)
    if model_name == "same_input_flat_linear":
        return PresentIntegralFlatLinear(multiset_count=config.multiset_count)
    raise ValueError(f"unsupported model: {model_name}")


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)


__all__ = [
    "HighRoundIntegralExperimentConfig",
    "adjudicate_high_round_integral",
    "evaluate_fixed_parity_baselines",
    "evaluate_untrained_candidate_baseline",
    "protocol_fixture",
    "run_high_round_integral_experiment",
    "summarize_high_round_splits",
]
