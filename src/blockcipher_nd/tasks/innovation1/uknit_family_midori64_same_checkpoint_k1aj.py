from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import load_bound_state
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    AUC_FLOOR,
    CONTROL_CONDITIONS,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    EXPECTED_TRAIN_ROWS,
    FRESH_SPLITS,
    INPUT_DIFFERENCE,
    NO_STRUCTURE_MARGIN,
    RUN_ID as K1AI_RUN_ID,
    SEMANTIC_MARGIN,
    build_k1ai_control,
    task_map,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_midori64_same_checkpoint_k1aj_replay_fix_20260729"
SOURCE_DECISION = (
    "innovation1_uknit_family_midori64_k1ai_"
    "signal_learned_structure_attribution_not_supported"
)
EXPECTED_SOURCE_DIGESTS = {
    "gate": "5f7eca268a26a9f3d3fdf746a0e9beae4552b156c1a832a7f81f02457d32803d",
    "validation": "a901d807da281762acbba30d960fc787dedd5df0981ed77499d09cf0589e370e",
    "checkpoint_manifest": (
        "1afc62124164e21340aac4c2ffe7450f462e341ae9a5be20b380265a104fb327"
    ),
    "controls": "f2e6a9ba34821f3acd1ccc787befb465ceca4e9f9f90ca58bbc62ca5d87092de",
    "dataset_manifest": (
        "5525a28f099a21bcca09aafbe05498f0f7951e22e171eaac6db055c174ff35bc"
    ),
}
SOURCE_REPLAY_TOLERANCE = 1e-7
PROBABILITY_DELTA_FLOOR = 1e-6
EXPECTED_ROWS = len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS) * len(CONTROL_CONDITIONS)


def source_binding_checks(
    *,
    gate: Mapping[str, Any],
    validation: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    source_controls: Sequence[Mapping[str, Any]],
    dataset_manifest: Sequence[Mapping[str, Any]],
    source_digests: Mapping[str, str],
) -> dict[str, bool]:
    correct_checkpoints = [
        row
        for row in checkpoint_manifest.get("entries", [])
        if row.get("condition") == "correct_structure"
    ]
    correct_controls = [
        row for row in source_controls if row.get("condition") == "correct_structure"
    ]
    expected_dataset_keys = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    return {
        "k1ai_source_digests_exact": dict(source_digests) == EXPECTED_SOURCE_DIGESTS,
        "k1ai_gate_exact_hold": (
            gate.get("run_id") == K1AI_RUN_ID
            and gate.get("status") == "hold"
            and gate.get("decision") == SOURCE_DECISION
            and gate.get("remote_scale") == "no"
            and not gate.get("failed_protocol_checks")
        ),
        "k1ai_validation_exact_pass": (
            validation.get("run_id") == K1AI_RUN_ID
            and validation.get("status") == "pass"
            and not validation.get("errors")
        ),
        "two_correct_best_checkpoint_entries": (
            checkpoint_manifest.get("run_id") == K1AI_RUN_ID
            and checkpoint_manifest.get("status") == "pass"
            and len(correct_checkpoints) == len(EXPECTED_SEEDS)
            and {int(row.get("seed", -1)) for row in correct_checkpoints}
            == set(EXPECTED_SEEDS)
            and all(
                row.get("selected_checkpoint") == "best"
                and row.get("model")
                == "runtime_spn_ct_k1aa_virtual_slot_histogram_true"
                and _sha256(row.get("sha256"))
                for row in correct_checkpoints
            )
        ),
        "six_correct_source_replay_rows": (
            len(correct_controls) == len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS)
            and {
                (int(row.get("seed", -1)), str(row.get("split")))
                for row in correct_controls
            }
            == expected_dataset_keys
            and all(
                row.get("run_id") == K1AI_RUN_ID
                and row.get("strict_state_dict_load") is True
                and row.get("training_performed") is False
                and int(row.get("optimizer_steps", -1)) == 0
                for row in correct_controls
            )
        ),
        "six_bound_dataset_rows": (
            len(dataset_manifest) == len(expected_dataset_keys)
            and {
                (int(row.get("seed", -1)), str(row.get("split")))
                for row in dataset_manifest
            }
            == expected_dataset_keys
            and all(
                int(row.get("cell", -1)) == 8
                and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
                and int(row.get("rounds", -1)) == 4
                and row.get("cache_payloads_present") is True
                for row in dataset_manifest
            )
        ),
    }


def evaluate_same_checkpoint_panel(
    *,
    tasks: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    source_controls: Sequence[Mapping[str, Any]],
    datasets: Mapping[tuple[int, str], DifferentialDataset],
    source_digests: Mapping[str, str],
    batch_size: int = 64,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    tasks_by_key = task_map(tasks)
    checkpoints = _correct_checkpoint_map(checkpoint_manifest)
    source_rows = _correct_source_map(source_controls)
    expected_datasets = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    if set(datasets) != expected_datasets:
        raise ValueError("K1-AJ requires six seed6/7 K1-AI datasets")

    result_rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        task = tasks_by_key[(seed, "correct_structure")]
        checkpoint_row = checkpoints[seed]
        checkpoint_path = Path(str(checkpoint_row["path"]))
        state, checkpoint_sha256 = load_bound_state(checkpoint_path, checkpoint_row)
        state_sha256 = tensor_mapping_sha256(state)
        for split in EXPECTED_SPLITS:
            dataset = datasets[(seed, split)]
            labels = np.asarray(dataset.labels, dtype=np.float32)
            dataset_sha256 = differential_dataset_sha256(dataset)
            models: dict[str, Any] = {}
            probabilities: dict[str, np.ndarray] = {}
            for condition in CONTROL_CONDITIONS:
                model = build_k1ai_control(
                    task=task,
                    condition=condition,
                    input_bits=int(dataset.features.shape[1]),
                )
                model.load_state_dict(state, strict=True)
                if tensor_mapping_sha256(model.state_dict()) != state_sha256:
                    raise ValueError("K1-AJ strict load changed the source state")
                models[condition] = model
                probabilities[condition] = predict_binary_probabilities(
                    model,
                    dataset,
                    batch_size=batch_size,
                    device=device,
                )
            exact_probabilities = probabilities["correct_structure"]
            source_row = source_rows[(seed, split)]
            for condition in CONTROL_CONDITIONS:
                values = probabilities[condition]
                delta = np.abs(exact_probabilities - values)
                model = models[condition]
                result_rows.append(
                    {
                        "run_id": RUN_ID,
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "cipher_key": "midori64",
                        "rounds": 4,
                        "auc": binary_auc(labels, values),
                        "source_correct_auc": float(source_row["auc"]),
                        "correct_minus_condition_auc": (
                            0.0
                            if condition == "correct_structure"
                            else binary_auc(labels, exact_probabilities)
                            - binary_auc(labels, values)
                        ),
                        "max_abs_probability_delta_from_correct": float(delta.max()),
                        "mean_abs_probability_delta_from_correct": float(delta.mean()),
                        "probability_sha256": hashlib.sha256(
                            values.astype(np.float32, copy=False).tobytes()
                        ).hexdigest(),
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_selected": checkpoint_row.get(
                            "selected_checkpoint"
                        ),
                        "checkpoint_reported_seed": checkpoint_row.get("seed"),
                        "state_dict_sha256": state_sha256,
                        "dataset_sha256": dataset_sha256,
                        "source_dataset_sha256": source_row.get("dataset_sha256"),
                        "source_checkpoint_sha256": source_row.get("checkpoint_sha256"),
                        "source_state_dict_sha256": source_row.get("state_dict_sha256"),
                        "source_decision": SOURCE_DECISION,
                        **{
                            f"source_{name}_sha256": digest
                            for name, digest in source_digests.items()
                        },
                        "runtime_structure_mode": model.runtime_structure_mode,
                        "runtime_structure_window_sha256": (
                            model.runtime_structure.window_sha256()
                        ),
                        "composition_sha256": model.composition_sha256,
                        "rows": int(dataset.features.shape[0]),
                        "input_bits": int(dataset.features.shape[1]),
                        "pairs_per_sample": int(dataset.metadata["pairs_per_sample"]),
                        "input_difference": int(dataset.metadata["input_difference"]),
                        "negative_mode": dataset.metadata["negative_mode"],
                        "sample_structure": dataset.metadata["sample_structure"],
                        "parameter_count": sum(
                            parameter.numel() for parameter in model.parameters()
                        ),
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "epochs": 0,
                    }
                )
    return result_rows


def adjudicate(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_checks: Mapping[str, bool],
    control_checks: Mapping[str, bool],
) -> dict[str, Any]:
    mapped = _row_map(rows)
    seed_results = {
        str(seed): {
            split: _split_result(mapped, seed, split) for split in EXPECTED_SPLITS
        }
        for seed in EXPECTED_SEEDS
    }
    complete = set(mapped) == {
        (seed, split, condition)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in CONTROL_CONDITIONS
    }
    protocol_checks = {
        **dict(source_checks),
        **dict(control_checks),
        "twenty_four_rows_complete": len(rows) == EXPECTED_ROWS and complete,
        "same_checkpoint_state_and_dataset_within_seed_split": (
            complete
            and all(
                len(
                    {
                        tuple(
                            mapped[(seed, split, condition)].get(field)
                            for field in (
                                "checkpoint_sha256",
                                "state_dict_sha256",
                                "dataset_sha256",
                            )
                        )
                        for condition in CONTROL_CONDITIONS
                    }
                )
                == 1
                for seed in EXPECTED_SEEDS
                for split in EXPECTED_SPLITS
            )
        ),
        "distinct_seed_checkpoints": complete
        and len(
            {
                mapped[(seed, "train_seen", "correct_structure")].get(
                    "checkpoint_sha256"
                )
                for seed in EXPECTED_SEEDS
            }
        )
        == len(EXPECTED_SEEDS),
        "correct_best_checkpoints_strictly_loaded": all(
            row.get("checkpoint_selected") == "best"
            and int(row.get("checkpoint_reported_seed", -1)) == int(row.get("seed", -2))
            and row.get("strict_state_dict_load") is True
            for row in rows
        ),
        "source_provenance_bound": all(
            row.get("source_decision") == SOURCE_DECISION
            and all(
                row.get(f"source_{name}_sha256") == digest
                for name, digest in EXPECTED_SOURCE_DIGESTS.items()
            )
            for row in rows
        ),
        "runtime_fingerprints_distinct": complete
        and all(
            len(
                {
                    mapped[(seed, split, condition)].get("composition_sha256")
                    for condition in CONTROL_CONDITIONS
                }
            )
            == len(CONTROL_CONDITIONS)
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "frozen_evaluation_geometry": all(
            row.get("cipher_key") == "midori64"
            and int(row.get("rounds", -1)) == 4
            and int(row.get("rows", -1))
            == (
                EXPECTED_TRAIN_ROWS
                if row.get("split") == "train_seen"
                else EXPECTED_HOLDOUT_ROWS
            )
            and int(row.get("input_bits", -1)) == 512
            and int(row.get("pairs_per_sample", -1)) == 4
            and int(row.get("input_difference", -1)) == INPUT_DIFFERENCE
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("sample_structure") == "independent_pairs"
            and int(row.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT
            for row in rows
        ),
        "correct_auc_reproduces_source": complete
        and all(
            abs(
                float(mapped[(seed, split, "correct_structure")]["auc"])
                - float(
                    mapped[(seed, split, "correct_structure")]["source_correct_auc"]
                )
            )
            <= SOURCE_REPLAY_TOLERANCE
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "source_dataset_checkpoint_state_replayed": all(
            row.get("dataset_sha256") == row.get("source_dataset_sha256")
            and row.get("checkpoint_sha256") == row.get("source_checkpoint_sha256")
            and row.get("state_dict_sha256") == row.get("source_state_dict_sha256")
            for row in rows
        ),
        "finite_metrics": all(
            _finite(row.get(field))
            for row in rows
            for field in (
                "auc",
                "source_correct_auc",
                "correct_minus_condition_auc",
                "max_abs_probability_delta_from_correct",
                "mean_abs_probability_delta_from_correct",
            )
        ),
        "inference_only": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and int(row.get("epochs", -1)) == 0
            for row in rows
        ),
    }
    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        for split in FRESH_SPLITS:
            result = seed_results[str(seed)][split]
            prefix = f"seed{seed}_{split}"
            research_checks[f"{prefix}_correct_auc_floor"] = (
                result["correct_auc"] >= AUC_FLOOR
            )
            research_checks[f"{prefix}_correct_beats_wrong_sbox"] = (
                result["correct_minus_wrong_sbox"] >= SEMANTIC_MARGIN
            )
            research_checks[f"{prefix}_correct_beats_corrupted_linear"] = (
                result["correct_minus_corrupted_linear"] >= SEMANTIC_MARGIN
            )
            research_checks[f"{prefix}_correct_beats_no_structure"] = (
                result["correct_minus_no_structure"] >= NO_STRUCTURE_MARGIN
            )
            for condition in ("wrong_sbox", "corrupted_linear", "no_structure"):
                research_checks[f"{prefix}_{condition}_changes_predictions"] = (
                    result[f"{condition}_max_probability_delta"]
                    > PROBABILITY_DELTA_FLOOR
                )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    sbox_pass = all(
        research_checks[f"seed{seed}_{split}_correct_beats_wrong_sbox"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    diffusion_pass = all(
        research_checks[f"seed{seed}_{split}_correct_beats_corrupted_linear"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    signal_pass = all(
        research_checks[f"seed{seed}_{split}_correct_auc_floor"]
        and research_checks[f"seed{seed}_{split}_correct_beats_no_structure"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_midori64_k1aj_protocol_invalid"
        next_action = (
            "repair only the failed K1-AJ source, checkpoint, dataset, runtime, "
            "or replay binding and rerun unchanged"
        )
    elif sbox_pass and diffusion_pass and signal_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_midori64_k1aj_"
            "same_checkpoint_semantic_use_supported"
        )
        next_action = (
            "keep the K1-AA representation and test one same-budget paired-"
            "initialization semantic-contrast objective against the independently "
            "trained wrong-S-box shortcut"
        )
    elif diffusion_pass and signal_pass and not sbox_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1aj_"
            "diffusion_causal_sbox_discrimination_failed"
        )
        next_action = (
            "replace only the compact invariant histogram readout with one bounded "
            "cell-conditional S-box-transition residual at the same data, pair, "
            "seed, epoch, and geometry budget before any scale"
        )
    elif signal_pass and sbox_pass and not diffusion_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1aj_"
            "sbox_causal_linear_discrimination_failed"
        )
        next_action = (
            "audit the same-checkpoint edge and histogram branches separately before "
            "changing the architecture or data budget"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1aj_"
            "structure_independent_path_dominates"
        )
        next_action = (
            "run one zero-training base, edge-residual, and histogram-residual branch "
            "ablation on the identical checkpoints and datasets before redesign"
        )

    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "seed_results": seed_results,
        "thresholds": {
            "correct_auc": AUC_FLOOR,
            "correct_minus_wrong_sbox": SEMANTIC_MARGIN,
            "correct_minus_corrupted_linear": SEMANTIC_MARGIN,
            "correct_minus_no_structure": NO_STRUCTURE_MARGIN,
            "max_probability_delta": PROBABILITY_DELTA_FLOOR,
            "source_replay_tolerance": SOURCE_REPLAY_TOLERANCE,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed zero-training same-checkpoint Midori64 r4 K1-AA four-"
            "structure three-split causal audit; not formal scale, attack, SOTA, "
            "family transfer, or arbitrary-SPN evidence"
        ),
        "blocked_actions": [
            "remote scale",
            "new data, pairs, epochs, seeds, rounds, differences, or model families",
            "MoE, trail/DDT inputs, or claiming arbitrary-SPN semantic understanding",
        ],
    }


def _correct_checkpoint_map(
    manifest: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    mapped = {
        int(row["seed"]): row
        for row in manifest.get("entries", [])
        if row.get("condition") == "correct_structure"
    }
    if set(mapped) != set(EXPECTED_SEEDS):
        raise ValueError("K1-AJ correct checkpoint manifest is incomplete")
    return mapped


def _correct_source_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped = {
        (int(row["seed"]), str(row["split"])): row
        for row in rows
        if row.get("condition") == "correct_structure"
    }
    expected = {(seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS}
    if set(mapped) != expected:
        raise ValueError("K1-AJ correct source replay rows are incomplete")
    return mapped


def _row_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["split"]), str(row["condition"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-AJ row: {key}")
        mapped[key] = row
    return mapped


def _split_result(
    rows: Mapping[tuple[int, str, str], Mapping[str, Any]],
    seed: int,
    split: str,
) -> dict[str, float]:
    values = {
        condition: rows[(seed, split, condition)] for condition in CONTROL_CONDITIONS
    }
    correct = float(values["correct_structure"]["auc"])
    result: dict[str, float] = {"correct_auc": correct}
    for condition in ("wrong_sbox", "corrupted_linear", "no_structure"):
        result[f"{condition}_auc"] = float(values[condition]["auc"])
        result[f"correct_minus_{condition}"] = correct - float(values[condition]["auc"])
        result[f"{condition}_max_probability_delta"] = float(
            values[condition]["max_abs_probability_delta_from_correct"]
        )
        result[f"{condition}_mean_probability_delta"] = float(
            values[condition]["mean_abs_probability_delta_from_correct"]
        )
    return result


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "EXPECTED_ROWS",
    "EXPECTED_SOURCE_DIGESTS",
    "PROBABILITY_DELTA_FLOOR",
    "RUN_ID",
    "SOURCE_DECISION",
    "SOURCE_REPLAY_TOLERANCE",
    "adjudicate",
    "evaluate_same_checkpoint_panel",
    "source_binding_checks",
]
