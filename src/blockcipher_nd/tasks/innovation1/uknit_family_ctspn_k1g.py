from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1f import (
    CANDIDATE_MODEL,
    CONTROL_CONDITIONS,
    EXPECTED_BATCH_SIZE,
    EXPECTED_CONTROL_ROWS as K1F_EXPECTED_CONTROL_ROWS,
    EXPECTED_TRAINING_ROWS as K1F_EXPECTED_TRAINING_ROWS,
    RUN_ID as K1F_RUN_ID,
    build_k1f_control,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_ctspn_cell_path_hypergraph_same_key_attribution_k1g_20260728"
K1F_DECISION = "innovation1_uknit_family_ctspn_k1f_hypergraph_not_supported"
EXPECTED_SPLITS = ("train_seen", "same_key_fresh", "cross_key_validation")
EXPECTED_RESULT_ROWS = 72
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_HOLDOUT_ROWS = 2048
SAME_KEY_SEED_OFFSET = 20_000
MARGIN = 0.005
UKNIT_AUC_FLOOR = 0.520
REPLAY_AUC_TOLERANCE = 5e-6

ModelBuilder = Callable[..., torch.nn.Module]


def validate_k1g_source(
    *,
    tasks: Sequence[Mapping[str, Any]],
    source_gate: Mapping[str, Any],
    source_results: Sequence[Mapping[str, Any]],
    source_controls: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    source_preflight: Mapping[str, Any],
    plan_path: Path,
) -> dict[str, bool]:
    task_map = _task_map(tasks)
    result_map = _source_result_map(source_results)
    control_map = _source_control_map(source_controls)
    checkpoint_map = _checkpoint_map(checkpoint_manifest)
    expected_tasks = _expected_task_keys()
    expected_controls = {
        (cipher, seed, condition)
        for cipher, seed in expected_tasks
        for condition in CONTROL_CONDITIONS
    }
    return {
        "source_k1f_clean_hold": (
            source_gate.get("run_id") == K1F_RUN_ID
            and source_gate.get("status") == "hold"
            and source_gate.get("decision") == K1F_DECISION
            and bool(source_gate.get("protocol_checks"))
            and all(source_gate.get("protocol_checks", {}).values())
        ),
        "four_frozen_tasks_complete": (
            len(tasks) == K1F_EXPECTED_TRAINING_ROWS and set(task_map) == expected_tasks
        ),
        "four_source_results_complete": (
            len(source_results) == K1F_EXPECTED_TRAINING_ROWS
            and set(result_map) == expected_tasks
        ),
        "twenty_four_source_controls_complete": (
            len(source_controls) == K1F_EXPECTED_CONTROL_ROWS
            and set(control_map) == expected_controls
        ),
        "four_checkpoint_bindings_complete": (
            checkpoint_manifest.get("run_id") == K1F_RUN_ID
            and checkpoint_manifest.get("status") == "pass"
            and len(checkpoint_manifest.get("entries", []))
            == K1F_EXPECTED_TRAINING_ROWS
            and set(checkpoint_map) == expected_tasks
        ),
        "source_plan_sha256_matches": (
            source_preflight.get("run_id") == K1F_RUN_ID
            and source_preflight.get("plan_sha256") == file_sha256(plan_path)
        ),
        "source_cache_root_declared": bool(source_preflight.get("source_cache_root"))
        and Path(str(source_preflight.get("source_cache_root"))).is_dir(),
        "source_protocol_frozen": _source_protocol_frozen(task_map, result_map),
        "source_controls_frozen": all(
            row.get("strict_state_dict_load") is True
            and row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in source_controls
        ),
        "checkpoint_paths_and_hashes_match": _checkpoint_bindings_match(
            result_map, control_map, checkpoint_map
        ),
    }


def evaluate_k1g(
    *,
    tasks: Sequence[Mapping[str, Any]],
    source_results: Sequence[Mapping[str, Any]],
    source_controls: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DifferentialDataset],
    device: str = "cpu",
    model_builder: ModelBuilder = build_k1f_control,
) -> list[dict[str, Any]]:
    task_map = _task_map(tasks)
    result_map = _source_result_map(source_results)
    source_control_map = _source_control_map(source_controls)
    checkpoint_map = _checkpoint_map(checkpoint_manifest)
    expected_datasets = {
        (cipher, seed, split)
        for cipher, seed in _expected_task_keys()
        for split in EXPECTED_SPLITS
    }
    if set(datasets) != expected_datasets:
        raise ValueError("K1-G requires all train, same-key, and cross-key datasets")

    rows: list[dict[str, Any]] = []
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            key = (cipher, seed)
            task = task_map[key]
            source = result_map[key]
            checkpoint = checkpoint_map[key]
            checkpoint_path = Path(str(source["training"]["checkpoint_output"]))
            checkpoint_sha256 = file_sha256(checkpoint_path)
            if checkpoint_sha256 != checkpoint.get("sha256"):
                raise ValueError(f"K1-G checkpoint hash mismatch: {checkpoint_path}")
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state_dict = payload["state_dict"]
            state_sha256 = tensor_mapping_sha256(state_dict)
            train_dataset = datasets[(cipher, seed, "train_seen")]
            same_key_dataset = datasets[(cipher, seed, "same_key_fresh")]
            same_key_overlap = dataset_row_overlap_count(
                train_dataset, same_key_dataset
            )

            for split in EXPECTED_SPLITS:
                dataset = datasets[(cipher, seed, split)]
                _validate_dataset(dataset, split=split)
                dataset_sha256 = differential_dataset_sha256(dataset)
                probabilities: dict[str, np.ndarray] = {}
                models: dict[str, torch.nn.Module] = {}
                for condition in CONTROL_CONDITIONS:
                    model = model_builder(
                        task=task,
                        condition=condition,
                        input_bits=int(dataset.features.shape[1]),
                    )
                    model.load_state_dict(state_dict, strict=True)
                    if tensor_mapping_sha256(model.state_dict()) != state_sha256:
                        raise ValueError("K1-G strict load changed learned state")
                    models[condition] = model
                    probabilities[condition] = predict_binary_probabilities(
                        model,
                        dataset,
                        batch_size=EXPECTED_BATCH_SIZE,
                        device=device,
                    )
                labels = np.asarray(dataset.labels, dtype=np.float32)
                aucs = {
                    condition: binary_auc(labels, probabilities[condition])
                    for condition in CONTROL_CONDITIONS
                }
                reference = probabilities["correct_ordered"]
                for condition in CONTROL_CONDITIONS:
                    current = probabilities[condition]
                    source_control = source_control_map.get(
                        (cipher, seed, condition), {}
                    )
                    model = models[condition]
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "source_run_id": K1F_RUN_ID,
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "condition": condition,
                            "rows": int(dataset.features.shape[0]),
                            "auc": aucs[condition],
                            "correct_minus_condition_auc": (
                                0.0
                                if condition == "correct_ordered"
                                else aucs["correct_ordered"] - aucs[condition]
                            ),
                            "max_abs_probability_delta_from_correct": float(
                                np.max(np.abs(reference - current))
                            ),
                            "mean_abs_probability_delta_from_correct": float(
                                np.mean(np.abs(reference - current))
                            ),
                            "dataset_sha256": dataset_sha256,
                            "dataset_seed": _dataset_seed(seed, split),
                            "key_scope": (
                                "validation_key"
                                if split == "cross_key_validation"
                                else "train_key"
                            ),
                            "same_key_train_overlap_rows": (
                                same_key_overlap if split == "same_key_fresh" else None
                            ),
                            "source_validation_dataset_sha256": (
                                source_control.get("dataset_sha256")
                                if split == "cross_key_validation"
                                else None
                            ),
                            "source_validation_auc": (
                                source_control.get("auc")
                                if split == "cross_key_validation"
                                else None
                            ),
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": checkpoint_sha256,
                            "expected_checkpoint_sha256": checkpoint.get("sha256"),
                            "state_dict_sha256": state_sha256,
                            "incidence_mode": model.incidence_mode,
                            "routing_sha256": model.cell_path_routing_sha256,
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def adjudicate_k1g(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    grouped = _result_map(rows)
    expected_keys = {
        (cipher, seed, split, condition)
        for cipher in EXPECTED_CIPHERS
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in CONTROL_CONDITIONS
    }
    replay_deltas = [
        abs(
            float(row.get("auc", math.nan))
            - float(row.get("source_validation_auc", math.nan))
        )
        for row in rows
        if row.get("split") == "cross_key_validation"
    ]
    max_replay_delta = max(replay_deltas) if replay_deltas else math.inf
    protocol_checks = {
        **source_checks,
        "seventy_two_rows_complete": (
            len(rows) == EXPECTED_RESULT_ROWS and set(grouped) == expected_keys
        ),
        "run_and_source_ids_exact": all(
            row.get("run_id") == RUN_ID and row.get("source_run_id") == K1F_RUN_ID
            for row in rows
        ),
        "split_row_counts_exact": all(
            int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train_seen"
                else EXPECTED_HOLDOUT_ROWS
            )
            for row in rows
        ),
        "dataset_seed_contract_exact": all(
            int(row.get("dataset_seed", -1))
            == _dataset_seed(int(row.get("seed", -1)), str(row.get("split")))
            for row in rows
        ),
        "key_scope_contract_exact": all(
            row.get("key_scope")
            == (
                "validation_key"
                if row.get("split") == "cross_key_validation"
                else "train_key"
            )
            for row in rows
        ),
        "same_dataset_per_seed_split": _same_dataset_per_seed_split(grouped),
        "three_datasets_distinct": all(
            len(
                {
                    grouped[(cipher, seed, split, "correct_ordered")]["dataset_sha256"]
                    for split in EXPECTED_SPLITS
                }
            )
            == len(EXPECTED_SPLITS)
            for cipher, seed in _expected_task_keys()
        ),
        "same_key_fresh_has_zero_train_overlap": all(
            grouped[(cipher, seed, "same_key_fresh", condition)].get(
                "same_key_train_overlap_rows"
            )
            == 0
            for cipher, seed in _expected_task_keys()
            for condition in CONTROL_CONDITIONS
        ),
        "same_checkpoint_and_state_per_seed": _same_checkpoint_per_seed(grouped),
        "checkpoint_hashes_match_manifest": all(
            row.get("checkpoint_sha256") == row.get("expected_checkpoint_sha256")
            for row in rows
        ),
        "cross_key_validation_replays_k1f": all(
            row.get("split") != "cross_key_validation"
            or (
                row.get("dataset_sha256") == row.get("source_validation_dataset_sha256")
                and abs(
                    float(row.get("auc", math.nan))
                    - float(row.get("source_validation_auc", math.nan))
                )
                <= REPLAY_AUC_TOLERANCE
            )
            for row in rows
        ),
        "incidence_control_isolated": all(
            grouped[(cipher, seed, split, "incidence_shuffled")].get("incidence_mode")
            == "shuffled"
            and grouped[(cipher, seed, split, "correct_ordered")].get("incidence_mode")
            == "true"
            and grouped[(cipher, seed, split, "incidence_shuffled")].get(
                "routing_sha256"
            )
            != grouped[(cipher, seed, split, "correct_ordered")].get("routing_sha256")
            for cipher, seed in _expected_task_keys()
            for split in EXPECTED_SPLITS
        ),
        "strict_load_and_zero_training": all(
            row.get("strict_state_dict_load") is True
            and row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in rows
        ),
        "finite_metrics": all(_row_finite(row) for row in rows),
    }
    seed_results = {
        cipher: {
            str(seed): {
                split: _split_result(grouped, cipher, seed, split)
                for split in EXPECTED_SPLITS
            }
            for seed in EXPECTED_SEEDS
        }
        for cipher in EXPECTED_CIPHERS
    }
    attribution_summary = {
        cipher: {
            split: all(
                seed_results[cipher][str(seed)][split]["attribution_pass"]
                for seed in EXPECTED_SEEDS
            )
            for split in EXPECTED_SPLITS
        }
        for cipher in EXPECTED_CIPHERS
    }
    uknit_floor = {
        split: all(
            seed_results["uknit64"][str(seed)][split]["correct_ordered_auc"]
            >= UKNIT_AUC_FLOOR
            for seed in EXPECTED_SEEDS
        )
        for split in EXPECTED_SPLITS
    }
    protocol_valid = all(protocol_checks.values())
    train_pass = attribution_summary["uknit64"]["train_seen"]
    same_key_attribution = attribution_summary["uknit64"]["same_key_fresh"]
    same_key_pass = same_key_attribution and uknit_floor["same_key_fresh"]
    cross_key_pass = (
        attribution_summary["uknit64"]["cross_key_validation"]
        and uknit_floor["cross_key_validation"]
    )
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1g_protocol_invalid"
        next_action = (
            "repair only the failed cache, key, overlap, checkpoint, or replay "
            "binding and rerun K1-G unchanged"
        )
    elif train_pass and same_key_pass and not cross_key_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1g_"
            "key_specific_hypergraph_signal_confirmed"
        )
        next_action = (
            "retain shared-cell routing and preregister one same-budget "
            "difference-only input bottleneck that removes absolute ciphertext/key cues"
        )
    elif train_pass and not same_key_attribution:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1g_"
            "sample_specific_hypergraph_attribution_overfit_confirmed"
        )
        next_action = (
            "close this learned two-transition hypergraph and preregister one "
            "exact operator-tied latent propagation model at the same budget"
        )
    elif train_pass and same_key_attribution and not uknit_floor["same_key_fresh"]:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1g_"
            "same_key_relation_generalizes_but_signal_weak"
        )
        next_action = (
            "retain only the relation-attribution mechanism evidence; close this "
            "high-capacity predictor and preregister one constrained exact "
            "operator-tied latent propagation model at the same budget"
        )
    elif not train_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1g_shared_cell_relation_underuse_confirmed"
        )
        next_action = (
            "close learned incidence conditioning and preregister one exact "
            "operator-tied latent propagation model at the same budget"
        )
    elif cross_key_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1g_"
            "hypergraph_generalization_supported_but_k1f_gate_held"
        )
        next_action = (
            "audit the remaining K1-F anchor/control failure before changing the model"
        )
    else:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1g_mixed_failure_confirmed"
        next_action = (
            "do not scale; localize the failed uKNIT seed/control before selecting "
            "another architecture"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "thresholds": {
            "control_margin": MARGIN,
            "uknit_auc_floor": UKNIT_AUC_FLOOR,
            "validation_replay_auc_tolerance": REPLAY_AUC_TOLERANCE,
        },
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "attribution_summary": attribution_summary,
        "uknit_auc_floor_summary": uknit_floor,
        "replay_diagnostics": {"max_cross_key_replay_auc_delta": max_replay_delta},
        "training_rows": 0,
        "optimizer_steps": 0,
        "claim_scope": (
            "zero-training same-checkpoint K1-F train-seen versus fresh-same-key "
            "versus original-cross-key attribution audit at 2048/1024-class local "
            "diagnostic scale; not formal scale, attack, SOTA, transfer, or ceiling evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote scale-up or extra training before the K1-G cause is resolved",
            "K2, MoE, DDT, trail, partial decryption, or cipher identity",
            "using Dialga or an average to hide a failed uKNIT seed",
        ],
    }


def dataset_row_overlap_count(
    left: DifferentialDataset, right: DifferentialDataset
) -> int:
    left_rows = _dataset_row_bytes(left)
    right_rows = _dataset_row_bytes(right)
    return len(left_rows.intersection(right_rows))


def _dataset_row_bytes(dataset: DifferentialDataset) -> set[bytes]:
    features = np.asarray(dataset.features, dtype=np.uint8)
    labels = np.asarray(dataset.labels, dtype=np.uint8).reshape(-1, 1)
    combined = np.concatenate((features, labels), axis=1)
    return {row.tobytes() for row in combined}


def _split_result(
    grouped: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    cipher: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    correct = float(grouped[(cipher, seed, split, "correct_ordered")]["auc"])
    controls = {
        condition: float(grouped[(cipher, seed, split, condition)]["auc"])
        for condition in CONTROL_CONDITIONS[1:]
    }
    margins = {condition: correct - auc for condition, auc in controls.items()}
    return {
        "correct_ordered_auc": correct,
        **{f"{condition}_auc": auc for condition, auc in controls.items()},
        **{
            f"correct_minus_{condition}": margin
            for condition, margin in margins.items()
        },
        "attribution_pass": all(margin >= MARGIN for margin in margins.values()),
    }


def _dataset_seed(seed: int, split: str) -> int:
    if split == "train_seen":
        return seed
    if split == "cross_key_validation":
        return seed + 10_000
    if split == "same_key_fresh":
        return seed + SAME_KEY_SEED_OFFSET
    raise ValueError(f"unknown K1-G split: {split}")


def _expected_task_keys() -> set[tuple[str, int]]:
    return {(cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS}


def _task_map(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for task in tasks:
        if task.get("model_key") != CANDIDATE_MODEL:
            continue
        key = (str(task.get("cipher_key")), int(task.get("seed", -1)))
        if key in result:
            raise ValueError(f"duplicate K1-G task: {key}")
        result[key] = task
    return result


def _source_result_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("cipher_key")), int(row.get("seed", -1)))
        if key in result:
            raise ValueError(f"duplicate K1-G source result: {key}")
        result[key] = row
    return result


def _source_control_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("cipher_key")),
            int(row.get("seed", -1)),
            str(row.get("condition")),
        )
        if key in result:
            raise ValueError(f"duplicate K1-G source control: {key}")
        result[key] = row
    return result


def _checkpoint_map(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in manifest.get("entries", []):
        key = (str(row.get("cipher_key")), int(row.get("seed", -1)))
        if key in result:
            raise ValueError(f"duplicate K1-G checkpoint: {key}")
        result[key] = row
    return result


def _result_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("cipher_key")),
            int(row.get("seed", -1)),
            str(row.get("split")),
            str(row.get("condition")),
        )
        if key in result:
            raise ValueError(f"duplicate K1-G result: {key}")
        result[key] = row
    return result


def _source_protocol_frozen(
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
    results: Mapping[tuple[str, int], Mapping[str, Any]],
) -> bool:
    if set(tasks) != _expected_task_keys() or set(results) != _expected_task_keys():
        return False
    return all(
        task.get("samples_per_class") == 2048
        and task.get("pairs_per_sample") == 4
        and task.get("negative_mode") == "encrypted_random_plaintexts"
        and task.get("target_epochs") == 10
        and task.get("train_key") != task.get("validation_key")
        and result.get("model") == CANDIDATE_MODEL
        and result.get("training", {}).get("train_rows") == EXPECTED_TRAIN_ROWS
        and result.get("training", {}).get("validation_rows") == EXPECTED_HOLDOUT_ROWS
        for key, task in tasks.items()
        for result in (results[key],)
    )


def _checkpoint_bindings_match(
    results: Mapping[tuple[str, int], Mapping[str, Any]],
    controls: Mapping[tuple[str, int, str], Mapping[str, Any]],
    checkpoints: Mapping[tuple[str, int], Mapping[str, Any]],
) -> bool:
    for key in _expected_task_keys():
        result = results.get(key, {})
        checkpoint = checkpoints.get(key, {})
        path = Path(str(result.get("training", {}).get("checkpoint_output", "")))
        if (
            not path.is_file()
            or checkpoint.get("path") != str(path)
            or checkpoint.get("sha256") != file_sha256(path)
        ):
            return False
        for condition in CONTROL_CONDITIONS:
            control = controls.get((*key, condition), {})
            if control.get("checkpoint_path") != str(path) or control.get(
                "checkpoint_sha256"
            ) != checkpoint.get("sha256"):
                return False
    return True


def _validate_dataset(dataset: DifferentialDataset, *, split: str) -> None:
    expected_rows = (
        EXPECTED_TRAIN_ROWS if split == "train_seen" else EXPECTED_HOLDOUT_ROWS
    )
    if int(dataset.features.shape[0]) != expected_rows:
        raise ValueError(f"K1-G {split} row count drifted")
    labels = np.asarray(dataset.labels, dtype=np.uint8)
    if int(labels.sum()) * 2 != expected_rows:
        raise ValueError(f"K1-G {split} labels are not balanced")


def _same_dataset_per_seed_split(
    grouped: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> bool:
    return all(
        len(
            {
                grouped[(cipher, seed, split, condition)].get("dataset_sha256")
                for condition in CONTROL_CONDITIONS
            }
        )
        == 1
        for cipher, seed in _expected_task_keys()
        for split in EXPECTED_SPLITS
    )


def _same_checkpoint_per_seed(
    grouped: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> bool:
    return all(
        len(
            {
                (
                    grouped[(cipher, seed, split, condition)].get("checkpoint_sha256"),
                    grouped[(cipher, seed, split, condition)].get("state_dict_sha256"),
                )
                for split in EXPECTED_SPLITS
                for condition in CONTROL_CONDITIONS
            }
        )
        == 1
        for cipher, seed in _expected_task_keys()
    )


def _row_finite(row: Mapping[str, Any]) -> bool:
    return all(
        math.isfinite(float(row.get(field, math.nan)))
        for field in (
            "auc",
            "correct_minus_condition_auc",
            "max_abs_probability_delta_from_correct",
            "mean_abs_probability_delta_from_correct",
        )
    )


__all__ = [
    "EXPECTED_RESULT_ROWS",
    "EXPECTED_SPLITS",
    "RUN_ID",
    "SAME_KEY_SEED_OFFSET",
    "adjudicate_k1g",
    "dataset_row_overlap_count",
    "evaluate_k1g",
    "validate_k1g_source",
]
