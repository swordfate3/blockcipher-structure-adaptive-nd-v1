from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


EXPECTED_SEEDS = (0, 1)
EXPECTED_ROLES = ("candidate", "anchor")
EXPECTED_STRUCTURES = ("correct", "shuffled")
CANDIDATE_MARGIN_FLOOR = 0.005
ANCHOR_AUC_DELTA_CEILING = 1e-12
ANCHOR_PROBABILITY_DELTA_CEILING = 1e-6


def evaluate_same_checkpoint_pair(
    *,
    seed: int,
    source_role: str,
    model_options: dict[str, Any],
    checkpoint_path: Path,
    dataset: DifferentialDataset,
    correct_structure: RuntimeSpnStructure,
    shuffled_structure: RuntimeSpnStructure,
    checkpoint_sha256: str,
    feature_sha256: str,
    label_sha256: str,
    descriptor_sha256: str,
    batch_size: int = 256,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    if source_role not in EXPECTED_ROLES:
        raise ValueError(f"unsupported source role: {source_role}")
    model = build_model(
        "runtime_spn_e4_equivariant_true",
        input_bits=int(dataset.features.shape[1]),
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=model_options,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("source checkpoint must contain a state_dict")
    model.load_state_dict(payload["state_dict"], strict=True)
    parameter_count = sum(value.numel() for value in model.parameters())

    probabilities: dict[str, np.ndarray] = {}
    for structure_mode, runtime_structure in (
        ("correct", correct_structure),
        ("shuffled", shuffled_structure),
    ):
        model.runtime_structure = runtime_structure
        probabilities[structure_mode] = predict_binary_probabilities(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
        )
    probability_delta = float(
        np.max(np.abs(probabilities["correct"] - probabilities["shuffled"]))
    )
    labels = np.asarray(dataset.labels, dtype=np.float32)
    aucs = {
        mode: binary_auc(labels, values) for mode, values in probabilities.items()
    }
    auc_margin = aucs["correct"] - aucs["shuffled"]
    checkpoint_metadata = payload.get("metadata", {})

    return [
        {
            "seed": seed,
            "source_role": source_role,
            "structure_mode": structure_mode,
            "auc": aucs[structure_mode],
            "correct_minus_shuffled_auc": auc_margin,
            "pair_max_abs_probability_delta": probability_delta,
            "mean_probability": float(probabilities[structure_mode].mean()),
            "probability_sha256": hashlib.sha256(
                probabilities[structure_mode].tobytes()
            ).hexdigest(),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_selected": checkpoint_metadata.get("selected_checkpoint"),
            "feature_sha256": feature_sha256,
            "label_sha256": label_sha256,
            "descriptor_sha256": descriptor_sha256,
            "descriptor_round_start": 2,
            "descriptor_loaded_rounds": 2,
            "samples_total": int(len(dataset.labels)),
            "input_bits": int(dataset.features.shape[1]),
            "parameter_count": parameter_count,
            "training_performed": False,
            "sbox_context_mode": model_options.get("sbox_context_mode"),
        }
        for structure_mode in EXPECTED_STRUCTURES
    ]


def adjudicate_uknit_same_checkpoint_counterfactual(
    *,
    run_id: str,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    grouped: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(int(row.get("seed", -1)), str(row.get("source_role")))][
            str(row.get("structure_mode"))
        ].append(row)

    complete_pairs = all(
        len(grouped[(seed, role)].get(mode, ())) == 1
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
        for mode in EXPECTED_STRUCTURES
    )
    pair_results = {
        f"seed{seed}_{role}": _pair_result(grouped[(seed, role)])
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
    }
    protocol_checks = {
        "eight_rows_complete": len(rows) == 8,
        "two_seed_two_role_two_structure_panel": complete_pairs
        and set(grouped)
        == {(seed, role) for seed in EXPECTED_SEEDS for role in EXPECTED_ROLES},
        "same_checkpoint_within_pair": complete_pairs
        and _same_pair_field(grouped, "checkpoint_sha256"),
        "same_features_within_pair": complete_pairs
        and _same_pair_field(grouped, "feature_sha256"),
        "same_labels_within_pair": complete_pairs
        and _same_pair_field(grouped, "label_sha256"),
        "selected_best_checkpoints": all(
            row.get("checkpoint_selected") == "best" for row in rows
        ),
        "exact_descriptor_window": len(rows) == 8
        and len({row.get("descriptor_sha256") for row in rows}) == 1
        and all(
            row.get("descriptor_round_start") == 2
            and row.get("descriptor_loaded_rounds") == 2
            for row in rows
        ),
        "finite_metrics": all(
            _finite(row.get("auc"))
            and _finite(row.get("mean_probability"))
            and _finite(row.get("pair_max_abs_probability_delta"))
            for row in rows
        ),
        "frozen_validation_size": all(
            row.get("samples_total") == 2048 and row.get("input_bits") == 512
            for row in rows
        ),
        "equal_parameter_geometry": len(rows) == 8
        and {row.get("parameter_count") for row in rows} == {442466},
        "no_training_performed": all(
            row.get("training_performed") is False for row in rows
        ),
        "role_context_contract": all(
            row.get("sbox_context_mode")
            == ("late_cell" if row.get("source_role") == "candidate" else "late_pair")
            for row in rows
        ),
        "sha256_fields_present": all(
            _is_sha256(row.get(field))
            for row in rows
            for field in (
                "checkpoint_sha256",
                "feature_sha256",
                "label_sha256",
                "descriptor_sha256",
                "probability_sha256",
            )
        ),
    }
    research_checks = {}
    for seed in EXPECTED_SEEDS:
        candidate = pair_results[f"seed{seed}_candidate"]
        anchor = pair_results[f"seed{seed}_anchor"]
        research_checks[f"seed{seed}_candidate_margin_at_least_0p005"] = bool(
            candidate["correct_minus_shuffled_auc"] is not None
            and candidate["correct_minus_shuffled_auc"] >= CANDIDATE_MARGIN_FLOOR
        )
        research_checks[f"seed{seed}_anchor_auc_invariant"] = bool(
            anchor["absolute_auc_delta"] is not None
            and anchor["absolute_auc_delta"] <= ANCHOR_AUC_DELTA_CEILING
        )
        research_checks[f"seed{seed}_anchor_probabilities_invariant"] = bool(
            anchor["max_abs_probability_delta"] is not None
            and anchor["max_abs_probability_delta"]
            <= ANCHOR_PROBABILITY_DELTA_CEILING
        )

    protocol_passed = all(protocol_checks.values())
    research_passed = all(research_checks.values())
    anchor_passed = all(
        research_checks[f"seed{seed}_{check}"]
        for seed in EXPECTED_SEEDS
        for check in ("anchor_auc_invariant", "anchor_probabilities_invariant")
    )
    if not protocol_passed:
        status = "fail"
        decision = "innovation1_uknit_same_checkpoint_protocol_invalid"
        next_action = "repair U2-A without changing sources, data, or thresholds"
    elif research_passed:
        status = "pass"
        decision = "innovation1_uknit_same_checkpoint_ownership_supported"
        next_action = (
            "preregister same-budget paired structure-swap training; do not scale data"
        )
    elif not anchor_passed:
        status = "hold"
        decision = "innovation1_uknit_same_checkpoint_invariance_control_failed"
        next_action = "debug the structure-swap audit before making a model decision"
    else:
        status = "hold"
        decision = "innovation1_uknit_additive_late_cell_ownership_not_used"
        next_action = (
            "design one local edge-conditioned or gated S-box/topology interaction "
            "against the frozen late_pair and shuffled-assignment controls"
        )

    return {
        "run_id": run_id,
        "task": "innovation1_uknit_runtime_e4_sbox_assignment_u2a",
        "cipher": "uKNIT-BC",
        "status": status,
        "decision": decision,
        "thresholds": {
            "candidate_margin": CANDIDATE_MARGIN_FLOOR,
            "anchor_auc_delta": ANCHOR_AUC_DELTA_CEILING,
            "anchor_probability_delta": ANCHOR_PROBABILITY_DELTA_CEILING,
        },
        "pair_results": pair_results,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "claim_scope": (
            "uKNIT-BC U1 same-checkpoint inference-time S-box-assignment audit; "
            "no training, scale, attack, cross-cipher, or breakthrough claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "increase U1 samples or epochs",
            "launch remote GPU",
            "add DDT, trail, or partial-decryption features",
            "claim stable S-box ownership superiority or a uKNIT attack",
        ],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_result(group: dict[str, list[dict[str, Any]]]) -> dict[str, float | None]:
    correct_rows = group.get("correct", ())
    shuffled_rows = group.get("shuffled", ())
    correct_auc = (
        float(correct_rows[0]["auc"])
        if len(correct_rows) == 1 and _finite(correct_rows[0].get("auc"))
        else None
    )
    shuffled_auc = (
        float(shuffled_rows[0]["auc"])
        if len(shuffled_rows) == 1 and _finite(shuffled_rows[0].get("auc"))
        else None
    )
    max_delta = (
        float(correct_rows[0]["pair_max_abs_probability_delta"])
        if len(correct_rows) == 1
        and _finite(correct_rows[0].get("pair_max_abs_probability_delta"))
        else None
    )
    return {
        "correct_auc": correct_auc,
        "shuffled_auc": shuffled_auc,
        "correct_minus_shuffled_auc": (
            correct_auc - shuffled_auc
            if correct_auc is not None and shuffled_auc is not None
            else None
        ),
        "absolute_auc_delta": (
            abs(correct_auc - shuffled_auc)
            if correct_auc is not None and shuffled_auc is not None
            else None
        ),
        "max_abs_probability_delta": max_delta,
    }


def _same_pair_field(
    grouped: dict[tuple[int, str], dict[str, list[dict[str, Any]]]],
    field: str,
) -> bool:
    return all(
        len(
            {
                rows[0].get(field)
                for rows in grouped[(seed, role)].values()
                if len(rows) == 1
            }
        )
        == 1
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
    )


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


__all__ = [
    "adjudicate_uknit_same_checkpoint_counterfactual",
    "evaluate_same_checkpoint_pair",
    "file_sha256",
]
