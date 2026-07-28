from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import load_bound_state
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    AUC_FLOOR,
    EXPECTED_HOLDOUT_ROWS,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    EXPECTED_TRAIN_ROWS,
    FRESH_SPLITS,
    INPUT_DIFFERENCE,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_sbox_transition_k1ak import (
    EXPECTED_PARAMETER_COUNT,
    RUN_ID as K1AK_RUN_ID,
    build_k1ak_control,
    checkpoint_map,
    task_map,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


RUN_ID = "i1_uknit_family_midori64_transition_causal_k1al_20260729"
SOURCE_DECISION = (
    "innovation1_uknit_family_midori64_k1ak_"
    "sbox_transition_discrimination_failed"
)
EXPECTED_SOURCE_DIGESTS = {
    "gate": "a8cd9de68a7b4e43a4c8f0793e31cbf8ce87f090c35be6f6821cab282e927f8f",
    "validation": "2d64a4e27b39a65fda5b44b217226fabb78a954d843573b47abbe34e0070e419",
    "checkpoint_manifest": (
        "048906c4e9288f9795453d15b4fd5ba476ba54247b22296b5bc745517cabd2f7"
    ),
    "controls": "3b667435eb6c91dfb1c828953e834e9556dedf16c5054b4e70ded1d598e6e04e",
    "dataset_manifest": (
        "5525a28f099a21bcca09aafbe05498f0f7951e22e171eaac6db055c174ff35bc"
    ),
}
EXPECTED_CORRECT_CHECKPOINTS = {
    6: "ac5364cb2b45d6e5f5dad189b582bfccedc18d29a06626d1ab3d349f12f44ed4",
    7: "29d1d8918ed2f6fd0c5345c87cf4b6efe66682540116f8340dc6d3c785996018",
}
AUDIT_CONDITIONS = (
    "correct_runtime",
    "wrong_sbox_same_checkpoint",
    "transition_branch_off_same_checkpoint",
)
SOURCE_REPLAY_TOLERANCE = 1e-7
SEMANTIC_MARGIN = 0.005
BRANCH_MARGIN = 0.005
PROBABILITY_DELTA_FLOOR = 1e-6
EXPECTED_ROWS = len(EXPECTED_SEEDS) * len(EXPECTED_SPLITS) * len(AUDIT_CONDITIONS)


class TransitionBranchOffWrapper(nn.Module):
    """Disable only the K1-AK transition projection during this forward call."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        def zero_transition(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            return torch.zeros_like(output)

        handle = self.model.backbone.transition_projection.register_forward_hook(
            zero_transition
        )
        try:
            return self.model(features)
        finally:
            handle.remove()


def source_binding_checks(
    *,
    gate: Mapping[str, Any],
    validation: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    source_controls: Sequence[Mapping[str, Any]],
    dataset_manifest: Sequence[Mapping[str, Any]],
    source_digests: Mapping[str, str],
) -> dict[str, bool]:
    correct_checkpoints = {
        int(row.get("seed", -1)): row
        for row in checkpoint_manifest.get("entries", [])
        if row.get("condition") == "correct_structure"
    }
    correct_controls = [
        row for row in source_controls if row.get("condition") == "correct_structure"
    ]
    expected_dataset_keys = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    return {
        "k1ak_source_digests_exact": dict(source_digests)
        == EXPECTED_SOURCE_DIGESTS,
        "k1ak_gate_exact_hold": (
            gate.get("run_id") == K1AK_RUN_ID
            and gate.get("status") == "hold"
            and gate.get("decision") == SOURCE_DECISION
            and gate.get("remote_scale") == "no"
            and not gate.get("failed_protocol_checks")
        ),
        "k1ak_validation_exact_pass": (
            validation.get("run_id") == K1AK_RUN_ID
            and validation.get("status") == "pass"
            and not validation.get("errors")
        ),
        "two_exact_correct_best_checkpoints": (
            checkpoint_manifest.get("run_id") == K1AK_RUN_ID
            and checkpoint_manifest.get("status") == "pass"
            and set(correct_checkpoints) == set(EXPECTED_SEEDS)
            and all(
                row.get("selected_checkpoint") == "best"
                and row.get("model")
                == "runtime_spn_ct_k1ak_sbox_transition_true"
                and row.get("sha256") == EXPECTED_CORRECT_CHECKPOINTS[seed]
                for seed, row in correct_checkpoints.items()
            )
        ),
        "six_correct_source_replay_rows": (
            len(correct_controls) == len(expected_dataset_keys)
            and {
                (int(row.get("seed", -1)), str(row.get("split")))
                for row in correct_controls
            }
            == expected_dataset_keys
            and all(
                row.get("run_id") == K1AK_RUN_ID
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


def evaluate_transition_causal_panel(
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
    checkpoints = {
        seed: row
        for (seed, condition), row in checkpoint_map(checkpoint_manifest).items()
        if condition == "correct_structure"
    }
    source_rows = {
        (int(row["seed"]), str(row["split"])): row
        for row in source_controls
        if row.get("condition") == "correct_structure"
    }
    expected_datasets = {
        (seed, split) for seed in EXPECTED_SEEDS for split in EXPECTED_SPLITS
    }
    if set(checkpoints) != set(EXPECTED_SEEDS):
        raise ValueError("K1-AL requires both K1-AK correct checkpoints")
    if set(source_rows) != expected_datasets or set(datasets) != expected_datasets:
        raise ValueError("K1-AL requires all six K1-AK correct replay datasets")

    result_rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        task = tasks_by_key[(seed, "correct_structure")]
        checkpoint_row = checkpoints[seed]
        checkpoint_path = Path(str(checkpoint_row["path"]))
        state, checkpoint_sha256 = load_bound_state(checkpoint_path, checkpoint_row)
        state_sha256 = tensor_mapping_sha256(state)

        correct_model = build_k1ak_control(task=task, condition="correct_structure")
        wrong_model = build_k1ak_control(task=task, condition="wrong_sbox")
        branch_off_model = build_k1ak_control(
            task=task,
            condition="correct_structure",
        )
        base_models = {
            "correct_runtime": correct_model,
            "wrong_sbox_same_checkpoint": wrong_model,
            "transition_branch_off_same_checkpoint": branch_off_model,
        }
        for model in base_models.values():
            model.load_state_dict(state, strict=True)
            if tensor_mapping_sha256(model.state_dict()) != state_sha256:
                raise ValueError("K1-AL strict load changed the K1-AK source state")
        audit_models: dict[str, nn.Module] = {
            "correct_runtime": correct_model,
            "wrong_sbox_same_checkpoint": wrong_model,
            "transition_branch_off_same_checkpoint": TransitionBranchOffWrapper(
                branch_off_model
            ),
        }

        for split in EXPECTED_SPLITS:
            dataset = datasets[(seed, split)]
            labels = np.asarray(dataset.labels, dtype=np.float32)
            dataset_sha256 = differential_dataset_sha256(dataset)
            probabilities = {
                condition: predict_binary_probabilities(
                    model,
                    dataset,
                    batch_size=batch_size,
                    device=device,
                )
                for condition, model in audit_models.items()
            }
            correct_probabilities = probabilities["correct_runtime"]
            source_row = source_rows[(seed, split)]
            for condition in AUDIT_CONDITIONS:
                model = base_models[condition]
                values = probabilities[condition]
                delta = np.abs(correct_probabilities - values)
                state_after_sha256 = tensor_mapping_sha256(model.state_dict())
                auc = binary_auc(labels, values)
                result_rows.append(
                    {
                        "run_id": RUN_ID,
                        "source_run_id": K1AK_RUN_ID,
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "cipher_key": "midori64",
                        "rounds": 4,
                        "auc": auc,
                        "source_correct_auc": float(source_row["auc"]),
                        "correct_minus_condition_auc": (
                            0.0
                            if condition == "correct_runtime"
                            else binary_auc(labels, correct_probabilities) - auc
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
                        "state_dict_sha256": state_after_sha256,
                        "source_state_dict_sha256": source_row.get(
                            "state_dict_sha256"
                        ),
                        "branch_off_state_preserved": state_after_sha256
                        == state_sha256,
                        "dataset_sha256": dataset_sha256,
                        "source_dataset_sha256": source_row.get("dataset_sha256"),
                        "source_checkpoint_sha256": source_row.get(
                            "checkpoint_sha256"
                        ),
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
                        "sbox_transition_semantics_sha256": (
                            model.sbox_transition_semantics_sha256
                        ),
                        "transition_branch_enabled": condition
                        != "transition_branch_off_same_checkpoint",
                        "intervention": (
                            "zero_transition_projection_output"
                            if condition == "transition_branch_off_same_checkpoint"
                            else "runtime_only"
                        ),
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
    expected_keys = {
        (seed, split, condition)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
        for condition in AUDIT_CONDITIONS
    }
    complete = set(mapped) == expected_keys
    seed_results = {
        str(seed): {
            split: _split_result(mapped, seed, split) for split in EXPECTED_SPLITS
        }
        for seed in EXPECTED_SEEDS
    }
    protocol_checks = {
        **dict(source_checks),
        **dict(control_checks),
        "eighteen_rows_complete": len(rows) == EXPECTED_ROWS and complete,
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
                        for condition in AUDIT_CONDITIONS
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
                mapped[(seed, "train_seen", "correct_runtime")].get(
                    "checkpoint_sha256"
                )
                for seed in EXPECTED_SEEDS
            }
        )
        == len(EXPECTED_SEEDS),
        "correct_best_checkpoints_strictly_loaded": all(
            row.get("checkpoint_selected") == "best"
            and int(row.get("checkpoint_reported_seed", -1))
            == int(row.get("seed", -2))
            and row.get("strict_state_dict_load") is True
            and row.get("checkpoint_sha256")
            == EXPECTED_CORRECT_CHECKPOINTS.get(int(row.get("seed", -1)))
            for row in rows
        ),
        "source_provenance_bound": all(
            row.get("source_run_id") == K1AK_RUN_ID
            and row.get("source_decision") == SOURCE_DECISION
            and all(
                row.get(f"source_{name}_sha256") == digest
                for name, digest in EXPECTED_SOURCE_DIGESTS.items()
            )
            for row in rows
        ),
        "runtime_interventions_exact": complete
        and all(
            mapped[(seed, split, "correct_runtime")].get("composition_sha256")
            == mapped[
                (seed, split, "transition_branch_off_same_checkpoint")
            ].get("composition_sha256")
            and mapped[(seed, split, "correct_runtime")].get(
                "sbox_transition_semantics_sha256"
            )
            == mapped[
                (seed, split, "transition_branch_off_same_checkpoint")
            ].get("sbox_transition_semantics_sha256")
            and mapped[(seed, split, "correct_runtime")].get("composition_sha256")
            != mapped[(seed, split, "wrong_sbox_same_checkpoint")].get(
                "composition_sha256"
            )
            and mapped[(seed, split, "correct_runtime")].get(
                "sbox_transition_semantics_sha256"
            )
            != mapped[(seed, split, "wrong_sbox_same_checkpoint")].get(
                "sbox_transition_semantics_sha256"
            )
            and mapped[(seed, split, "correct_runtime")].get(
                "transition_branch_enabled"
            )
            is True
            and mapped[(seed, split, "wrong_sbox_same_checkpoint")].get(
                "transition_branch_enabled"
            )
            is True
            and mapped[
                (seed, split, "transition_branch_off_same_checkpoint")
            ].get("transition_branch_enabled")
            is False
            for seed in EXPECTED_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "correct_auc_reproduces_source": complete
        and all(
            abs(
                float(mapped[(seed, split, "correct_runtime")]["auc"])
                - float(
                    mapped[(seed, split, "correct_runtime")]["source_correct_auc"]
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
        "branch_off_preserves_source_state": all(
            row.get("branch_off_state_preserved") is True for row in rows
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
                result["correct_runtime_auc"] >= AUC_FLOOR
            )
            research_checks[f"{prefix}_correct_beats_wrong_sbox"] = (
                result["correct_minus_wrong_sbox"] >= SEMANTIC_MARGIN
            )
            research_checks[f"{prefix}_correct_beats_transition_off"] = (
                result["correct_minus_transition_branch_off"] >= BRANCH_MARGIN
            )
            research_checks[f"{prefix}_wrong_sbox_changes_predictions"] = (
                result["wrong_sbox_max_probability_delta"]
                > PROBABILITY_DELTA_FLOOR
            )
            research_checks[f"{prefix}_transition_off_changes_predictions"] = (
                result["transition_branch_off_max_probability_delta"]
                > PROBABILITY_DELTA_FLOOR
            )

    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    sbox_pass = all(
        research_checks[f"seed{seed}_{split}_correct_beats_wrong_sbox"]
        and research_checks[f"seed{seed}_{split}_wrong_sbox_changes_predictions"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )
    branch_pass = all(
        research_checks[f"seed{seed}_{split}_correct_beats_transition_off"]
        and research_checks[f"seed{seed}_{split}_transition_off_changes_predictions"]
        for seed in EXPECTED_SEEDS
        for split in FRESH_SPLITS
    )

    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_midori64_k1al_protocol_invalid"
        next_action = (
            "repair only the failed K1-AL source, state, dataset, runtime, or "
            "replay binding and rerun unchanged"
        )
    elif sbox_pass and branch_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_midori64_k1al_"
            "transition_and_sbox_causal_use_supported"
        )
        next_action = (
            "retain the K1-AK representation and test one same-budget paired "
            "semantic-contrast objective against the independently trained "
            "wrong-S-box substitute"
        )
    elif branch_pass:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1al_"
            "transition_causal_sbox_identification_failed"
        )
        next_action = (
            "discard the non-identifying transition representation and redesign "
            "its S-box semantic comparison before another trained model"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_midori64_k1al_"
            "transition_branch_causal_use_failed"
        )
        next_action = (
            "discard the K1-AK transition readout and run one zero-training base "
            "versus edge-path audit to locate the source of its apparent gain"
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
            "correct_minus_transition_branch_off": BRANCH_MARGIN,
            "max_probability_delta": PROBABILITY_DELTA_FLOOR,
            "source_replay_tolerance": SOURCE_REPLAY_TOLERANCE,
        },
        "next_action": next_action,
        "claim_scope": (
            "zero-training two-seed Midori64 r4 K1-AK same-checkpoint causal "
            "audit; not formal scale, attack, SOTA, family-transfer, arbitrary-"
            "SPN, or ceiling evidence"
        ),
        "blocked_actions": [
            "remote scale or family transfer from K1-AL",
            "more pairs, samples, epochs, seeds, positions, rounds, width, or MoE",
            "DDT/trail inputs or averaging failed seeds/splits",
        ],
    }


def _row_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["split"]), str(row["condition"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-AL row: {key}")
        mapped[key] = row
    return mapped


def _split_result(
    rows: Mapping[tuple[int, str, str], Mapping[str, Any]],
    seed: int,
    split: str,
) -> dict[str, float]:
    correct = rows[(seed, split, "correct_runtime")]
    wrong = rows[(seed, split, "wrong_sbox_same_checkpoint")]
    branch_off = rows[(seed, split, "transition_branch_off_same_checkpoint")]
    correct_auc = float(correct["auc"])
    wrong_auc = float(wrong["auc"])
    branch_off_auc = float(branch_off["auc"])
    return {
        "correct_runtime_auc": correct_auc,
        "wrong_sbox_auc": wrong_auc,
        "transition_branch_off_auc": branch_off_auc,
        "correct_minus_wrong_sbox": correct_auc - wrong_auc,
        "correct_minus_transition_branch_off": correct_auc - branch_off_auc,
        "wrong_sbox_max_probability_delta": float(
            wrong["max_abs_probability_delta_from_correct"]
        ),
        "transition_branch_off_max_probability_delta": float(
            branch_off["max_abs_probability_delta_from_correct"]
        ),
    }


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "AUDIT_CONDITIONS",
    "BRANCH_MARGIN",
    "EXPECTED_CORRECT_CHECKPOINTS",
    "EXPECTED_ROWS",
    "EXPECTED_SOURCE_DIGESTS",
    "PROBABILITY_DELTA_FLOOR",
    "RUN_ID",
    "SEMANTIC_MARGIN",
    "SOURCE_DECISION",
    "TransitionBranchOffWrapper",
    "adjudicate",
    "evaluate_transition_causal_panel",
    "source_binding_checks",
]
