from __future__ import annotations

from collections import defaultdict
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
    CONTROL_CONDITIONS,
    EXPECTED_BATCH_SIZE,
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1d import (
    CANDIDATE_MODEL,
    RUN_ID as K1D_RUN_ID,
    build_k1d_control,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_ctspn_relative_path_train_validation_attribution_k1e_20260728"
K1D_DECISION = "innovation1_uknit_family_ctspn_k1d_relative_path_not_supported"
EXPECTED_SPLITS = ("train", "validation")
EXPECTED_RESULT_ROWS = 40
EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
MARGIN = 0.005
REPLAY_AUC_TOLERANCE = 5e-6

ModelBuilder = Callable[..., torch.nn.Module]


def validate_k1e_source(
    *,
    tasks: Sequence[Mapping[str, Any]],
    source_gate: Mapping[str, Any],
    source_results: Sequence[Mapping[str, Any]],
    source_controls: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    source_preflight: Mapping[str, Any],
    plan_path: Path,
    source_root: Path,
) -> dict[str, bool]:
    task_map = _task_map(tasks)
    training_map = _training_map(source_results)
    control_map = _source_control_map(source_controls)
    checkpoint_map = _checkpoint_map(checkpoint_manifest)
    expected_keys = _expected_task_keys()
    expected_controls = {
        (cipher, seed, condition)
        for cipher, seed in expected_keys
        for condition in CONTROL_CONDITIONS
    }
    checks = {
        "source_k1d_gate_exact": (
            source_gate.get("run_id") == K1D_RUN_ID
            and source_gate.get("status") == "hold"
            and source_gate.get("decision") == K1D_DECISION
        ),
        "source_k1d_protocol_clean": bool(source_gate.get("protocol_checks"))
        and all(source_gate.get("protocol_checks", {}).values()),
        "four_frozen_tasks_complete": len(tasks) == 4
        and set(task_map) == expected_keys,
        "four_source_results_complete": len(source_results) == 4
        and set(training_map) == expected_keys,
        "twenty_source_controls_complete": len(source_controls) == 20
        and set(control_map) == expected_controls,
        "four_selected_checkpoints_complete": (
            checkpoint_manifest.get("run_id") == K1D_RUN_ID
            and checkpoint_manifest.get("status") == "pass"
            and len(checkpoint_manifest.get("entries", [])) == 4
            and set(checkpoint_map) == expected_keys
        ),
        "source_plan_sha256_matches": (
            source_preflight.get("run_id") == K1D_RUN_ID
            and source_preflight.get("plan_sha256") == file_sha256(plan_path)
        ),
        "source_protocol_frozen": _source_protocol_frozen(
            task_map=task_map,
            training_map=training_map,
        ),
        "checkpoint_paths_and_hashes_match": _checkpoint_bindings_match(
            training_map=training_map,
            control_map=control_map,
            checkpoint_map=checkpoint_map,
            source_root=source_root,
        ),
        "source_controls_are_frozen": all(
            row.get("strict_state_dict_load") is True
            and row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in source_controls
        ),
        "source_cache_files_complete": _source_cache_files_complete(
            tasks=task_map,
            source_root=source_root,
        ),
    }
    return checks


def evaluate_k1e(
    *,
    tasks: Sequence[Mapping[str, Any]],
    source_results: Sequence[Mapping[str, Any]],
    source_controls: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DifferentialDataset],
    device: str = "cpu",
    model_builder: ModelBuilder = build_k1d_control,
) -> list[dict[str, Any]]:
    task_map = _task_map(tasks)
    training_map = _training_map(source_results)
    source_control_map = _source_control_map(source_controls)
    checkpoint_map = _checkpoint_map(checkpoint_manifest)
    expected_datasets = {
        (cipher, seed, split)
        for cipher, seed in _expected_task_keys()
        for split in EXPECTED_SPLITS
    }
    if set(datasets) != expected_datasets:
        raise ValueError("K1-E requires all four train and validation datasets")

    rows: list[dict[str, Any]] = []
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            key = (cipher, seed)
            task = task_map[key]
            source = training_map[key]
            checkpoint = checkpoint_map[key]
            checkpoint_path = Path(str(source["training"]["checkpoint_output"]))
            checkpoint_sha256 = file_sha256(checkpoint_path)
            if checkpoint_sha256 != checkpoint.get("sha256"):
                raise ValueError(f"K1-E checkpoint hash mismatch: {checkpoint_path}")
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state_dict = payload["state_dict"]
            state_sha256 = tensor_mapping_sha256(state_dict)

            for split in EXPECTED_SPLITS:
                dataset = datasets[(cipher, seed, split)]
                _validate_dataset(dataset, cipher=cipher, split=split)
                dataset_sha256 = differential_dataset_sha256(dataset)
                probabilities: dict[str, np.ndarray] = {}
                for condition in CONTROL_CONDITIONS:
                    model = model_builder(
                        task=task,
                        condition=condition,
                        input_bits=int(dataset.features.shape[1]),
                    )
                    model.load_state_dict(state_dict, strict=True)
                    if tensor_mapping_sha256(model.state_dict()) != state_sha256:
                        raise ValueError("K1-E strict load changed learned state")
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
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "source_run_id": K1D_RUN_ID,
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
                            "source_validation_dataset_sha256": (
                                source_control.get("dataset_sha256")
                                if split == "validation"
                                else None
                            ),
                            "source_validation_auc": (
                                source_control.get("auc")
                                if split == "validation"
                                else None
                            ),
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_sha256": checkpoint_sha256,
                            "expected_checkpoint_sha256": checkpoint.get("sha256"),
                            "state_dict_sha256": state_sha256,
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def adjudicate_k1e(
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
    validation_replay_deltas = [
        abs(
            float(row.get("auc", math.nan))
            - float(row.get("source_validation_auc", math.nan))
        )
        for row in rows
        if row.get("split") == "validation"
    ]
    max_validation_replay_auc_delta = (
        max(validation_replay_deltas) if validation_replay_deltas else math.inf
    )
    protocol_checks = {
        **source_checks,
        "forty_rows_complete": len(rows) == EXPECTED_RESULT_ROWS
        and set(grouped) == expected_keys,
        "run_and_source_ids_exact": all(
            row.get("run_id") == RUN_ID and row.get("source_run_id") == K1D_RUN_ID
            for row in rows
        ),
        "split_row_counts_exact": all(
            int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train"
                else EXPECTED_VALIDATION_ROWS
            )
            for row in rows
        ),
        "same_dataset_per_split_panel": _same_panel_value(
            grouped, "dataset_sha256", per_split=True
        ),
        "train_and_validation_datasets_distinct": all(
            grouped.get((cipher, seed, "train", "correct_ordered"), {}).get(
                "dataset_sha256"
            )
            != grouped.get((cipher, seed, "validation", "correct_ordered"), {}).get(
                "dataset_sha256"
            )
            for cipher in EXPECTED_CIPHERS
            for seed in EXPECTED_SEEDS
        ),
        "same_checkpoint_and_state_per_seed": _same_panel_value(
            grouped, "checkpoint_sha256", per_split=False
        )
        and _same_panel_value(grouped, "state_dict_sha256", per_split=False),
        "checkpoint_hashes_match_manifest": all(
            row.get("checkpoint_sha256") == row.get("expected_checkpoint_sha256")
            for row in rows
        ),
        "validation_replays_k1d_exactly": all(
            row.get("split") != "validation"
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
        "strict_load_and_zero_training": all(
            row.get("strict_state_dict_load") is True
            and row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in rows
        ),
        "finite_metrics": all(_row_finite(row) for row in rows),
    }

    seed_results: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    research_checks: dict[str, bool] = {}
    attribution_passes: dict[tuple[str, int, str], bool] = {}
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            split_results: dict[str, Any] = {}
            for split in EXPECTED_SPLITS:
                condition_aucs = {
                    condition: float(
                        grouped.get((cipher, seed, split, condition), {}).get(
                            "auc", math.nan
                        )
                    )
                    for condition in CONTROL_CONDITIONS
                }
                correct = condition_aucs["correct_ordered"]
                margins = {
                    condition: correct - condition_aucs[condition]
                    for condition in CONTROL_CONDITIONS[1:]
                }
                for condition, margin in margins.items():
                    research_checks[
                        f"{cipher}_seed{seed}_{split}_beats_{condition}"
                    ] = margin >= MARGIN
                attribution_pass = all(margin >= MARGIN for margin in margins.values())
                attribution_passes[(cipher, seed, split)] = attribution_pass
                split_results[split] = {
                    "correct_ordered_auc": correct,
                    **{
                        f"{condition}_auc": value
                        for condition, value in condition_aucs.items()
                        if condition != "correct_ordered"
                    },
                    **{
                        f"correct_minus_{condition}": margin
                        for condition, margin in margins.items()
                    },
                    "attribution_pass": attribution_pass,
                }
            seed_results[cipher][str(seed)] = split_results

    protocol_valid = all(protocol_checks.values())
    uknit_train_pass = all(
        attribution_passes.get(("uknit64", seed, "train"), False)
        for seed in EXPECTED_SEEDS
    )
    uknit_validation_pass = all(
        attribution_passes.get(("uknit64", seed, "validation"), False)
        for seed in EXPECTED_SEEDS
    )
    dialga_train_pass = all(
        attribution_passes.get(("dialga128", seed, "train"), False)
        for seed in EXPECTED_SEEDS
    )
    dialga_validation_pass = all(
        attribution_passes.get(("dialga128", seed, "validation"), False)
        for seed in EXPECTED_SEEDS
    )

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1e_protocol_invalid"
        next_action = "repair only the failed source, cache, checkpoint, or replay binding and rerun K1-E unchanged"
    elif uknit_train_pass and not uknit_validation_pass:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1e_split_specific_relative_path_overfit_confirmed"
        next_action = (
            "close anonymous relative-path pooling and build a same-budget "
            "permutation-equivariant cell/path hypergraph whose cell indices are routing only"
        )
    elif uknit_train_pass and uknit_validation_pass:
        status = "hold"
        decision = "innovation1_uknit_family_ctspn_k1e_source_replay_inconsistency"
        next_action = (
            "audit K1-D checkpoint selection and gate binding before another model"
        )
    else:
        status = "pass"
        decision = "innovation1_uknit_family_ctspn_k1e_anonymous_path_relation_collapse_confirmed"
        next_action = (
            "close anonymous path-set pooling and run zero-training readiness for "
            "permutation-equivariant cell/path hypergraph message passing"
        )

    return {
        "run_id": RUN_ID,
        "source_run_id": K1D_RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "seed_results": dict(seed_results),
        "attribution_summary": {
            "uknit_train_all_seeds": uknit_train_pass,
            "uknit_validation_all_seeds": uknit_validation_pass,
            "dialga_train_all_seeds": dialga_train_pass,
            "dialga_validation_all_seeds": dialga_validation_pass,
        },
        "thresholds": {
            "correct_topology_margin": MARGIN,
            "validation_replay_auc_tolerance": REPLAY_AUC_TOLERANCE,
        },
        "replay_diagnostics": {
            "validation_rows": len(validation_replay_deltas),
            "max_validation_replay_auc_delta": max_validation_replay_auc_delta,
        },
        "training_rows": 0,
        "optimizer_steps": 0,
        "claim_scope": (
            "zero-training same-checkpoint train-versus-validation topology attribution "
            "audit of completed uKNIT-BC r5 and Dialga-128 r4 K1-D 2048/class local "
            "diagnostic; not new training, formal scale, attack, SOTA, transfer, or ceiling evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote scale-up, extra samples, pairs, epochs, width or experts",
            "K2 S-box conditioning, MoE, DDT, trail or partial decryption",
            "claim a uKNIT ceiling from this local mechanism audit",
        ],
    }


def _task_map(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for task in tasks:
        if task.get("model_key") != CANDIDATE_MODEL:
            continue
        key = (str(task.get("cipher_key")), int(task.get("seed", -1)))
        if key in result:
            raise ValueError(f"duplicate K1-E task: {key}")
        result[key] = task
    return result


def _training_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("model") != CANDIDATE_MODEL:
            continue
        key = (str(row.get("cipher_key")), int(row.get("seed", -1)))
        if key in result:
            raise ValueError(f"duplicate K1-D source result: {key}")
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
            raise ValueError(f"duplicate K1-D source control: {key}")
        result[key] = row
    return result


def _checkpoint_map(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in manifest.get("entries", []):
        key = (str(entry.get("cipher_key")), int(entry.get("seed", -1)))
        if key in result:
            raise ValueError(f"duplicate K1-D checkpoint manifest entry: {key}")
        result[key] = entry
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
            raise ValueError(f"duplicate K1-E result row: {key}")
        result[key] = row
    return result


def _expected_task_keys() -> set[tuple[str, int]]:
    return {(cipher, seed) for cipher in EXPECTED_CIPHERS for seed in EXPECTED_SEEDS}


def _source_protocol_frozen(
    *,
    task_map: Mapping[tuple[str, int], Mapping[str, Any]],
    training_map: Mapping[tuple[str, int], Mapping[str, Any]],
) -> bool:
    if (
        set(task_map) != _expected_task_keys()
        or set(training_map) != _expected_task_keys()
    ):
        return False
    for (cipher, seed), task in task_map.items():
        source = training_map[(cipher, seed)]
        training = source.get("training", {})
        options = task.get("model_options", {})
        if (
            task.get("rounds") != (5 if cipher == "uknit64" else 4)
            or task.get("samples_per_class") != 2048
            or task.get("pairs_per_sample") != 4
            or task.get("negative_mode") != "encrypted_random_plaintexts"
            or task.get("sample_structure") != "independent_pairs"
            or task.get("target_epochs") != 10
            or options.get("runtime_round_start") != (3 if cipher == "uknit64" else 2)
            or options.get("runtime_rounds") != 2
            or options.get("processor_steps") != 2
            or source.get("seed") != seed
            or source.get("samples_per_class") != 2048
            or source.get("pairs_per_sample") != 4
            or training.get("batch_size") != EXPECTED_BATCH_SIZE
            or training.get("epochs") != 10
            or training.get("checkpoint_metric") != "val_auc"
            or training.get("selected_checkpoint") != "best"
            or training.get("train_rows") != EXPECTED_TRAIN_ROWS
            or training.get("validation_rows") != EXPECTED_VALIDATION_ROWS
            or training.get("train_dataset_storage") != "disk"
            or training.get("validation_dataset_storage") != "disk"
        ):
            return False
    return True


def _checkpoint_bindings_match(
    *,
    training_map: Mapping[tuple[str, int], Mapping[str, Any]],
    control_map: Mapping[tuple[str, int, str], Mapping[str, Any]],
    checkpoint_map: Mapping[tuple[str, int], Mapping[str, Any]],
    source_root: Path,
) -> bool:
    if (
        set(training_map) != _expected_task_keys()
        or set(checkpoint_map) != _expected_task_keys()
    ):
        return False
    root = source_root.resolve()
    for cipher, seed in _expected_task_keys():
        path = Path(
            str(
                training_map[(cipher, seed)]
                .get("training", {})
                .get("checkpoint_output", "")
            )
        )
        manifest = checkpoint_map[(cipher, seed)]
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError):
            return False
        if root not in resolved.parents:
            return False
        if (
            str(path) != str(manifest.get("path"))
            or manifest.get("model") != CANDIDATE_MODEL
            or manifest.get("selected_checkpoint") != "best"
            or file_sha256(path) != manifest.get("sha256")
        ):
            return False
        control_hashes = {
            control_map.get((cipher, seed, condition), {}).get("checkpoint_sha256")
            for condition in CONTROL_CONDITIONS
        }
        if control_hashes != {manifest.get("sha256")}:
            return False
    return True


def _source_cache_files_complete(
    *,
    tasks: Mapping[tuple[str, int], Mapping[str, Any]],
    source_root: Path,
) -> bool:
    cache_root = source_root / "cache"
    for (cipher, seed), task in tasks.items():
        rounds = int(task["rounds"])
        for split, split_seed in (("train", seed), ("validation", seed + 10_000)):
            parent = cache_root / cipher / f"r{rounds}" / split
            matches = list(parent.glob(f"seed-{split_seed}_*"))
            if len(matches) != 1:
                return False
            if not all(
                (matches[0] / name).is_file()
                for name in ("metadata.json", "features.npy", "labels.npy")
            ):
                return False
    return True


def _validate_dataset(dataset: DifferentialDataset, *, cipher: str, split: str) -> None:
    expected_rows = (
        EXPECTED_TRAIN_ROWS if split == "train" else EXPECTED_VALIDATION_ROWS
    )
    expected_bits = 512 if cipher == "uknit64" else 1024
    labels = np.asarray(dataset.labels)
    if dataset.features.shape != (expected_rows, expected_bits):
        raise ValueError(f"unexpected K1-E {cipher} {split} feature shape")
    if labels.shape != (expected_rows,):
        raise ValueError(f"unexpected K1-E {cipher} {split} label shape")
    if int(labels.sum()) != expected_rows // 2:
        raise ValueError(f"K1-E {cipher} {split} labels are not balanced")


def _same_panel_value(
    rows: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    field: str,
    *,
    per_split: bool,
) -> bool:
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            split_groups = EXPECTED_SPLITS if per_split else (None,)
            for selected_split in split_groups:
                values = {
                    row.get(field)
                    for (
                        row_cipher,
                        row_seed,
                        row_split,
                        _condition,
                    ), row in rows.items()
                    if row_cipher == cipher
                    and row_seed == seed
                    and (selected_split is None or row_split == selected_split)
                }
                if len(values) != 1 or None in values:
                    return False
    return True


def _row_finite(row: Mapping[str, Any]) -> bool:
    fields = (
        "auc",
        "correct_minus_condition_auc",
        "max_abs_probability_delta_from_correct",
        "mean_abs_probability_delta_from_correct",
    )
    return all(
        isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
        for field in fields
    )


__all__ = [
    "EXPECTED_RESULT_ROWS",
    "EXPECTED_SPLITS",
    "K1D_DECISION",
    "RUN_ID",
    "adjudicate_k1e",
    "evaluate_k1e",
    "validate_k1e_source",
]
