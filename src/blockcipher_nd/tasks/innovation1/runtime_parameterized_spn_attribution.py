from __future__ import annotations

import math
from typing import Any


EXPECTED_RUNTIME_MODELS = {
    "GIFT-64": {
        "true": "gift64_runtime_spn_true",
        "corrupted": "gift64_runtime_spn_corrupted",
        "independent": "gift64_runtime_spn_independent",
    },
    "PRESENT-80": {
        "true": "present_runtime_spn_true",
        "corrupted": "present_runtime_spn_corrupted",
        "independent": "present_runtime_spn_independent",
    },
}

EXPECTED_ANCHOR_MODELS = {
    "GIFT-64": "gift_cross_spn_typed_cell_true",
    "PRESENT-80": "present_cross_spn_typed_cell_true",
}

ANCHOR_TOLERANCE = 0.005
CONTROL_MARGIN = 0.005


def adjudicate_runtime_spn_r1(
    *,
    run_id: str,
    cipher: str,
    runtime_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    r0_gate: dict[str, Any],
) -> dict[str, Any]:
    if cipher not in EXPECTED_RUNTIME_MODELS:
        raise ValueError(f"unsupported RTG1 R1 cipher: {cipher}")
    runtime_by_role = _runtime_rows_by_role(cipher, runtime_rows)
    anchor = _anchor_row(cipher, anchor_rows)
    protocol_checks = _protocol_checks(
        cipher,
        runtime_by_role,
        anchor,
        r0_gate,
    )
    aucs = {
        role: float(row["metrics"]["auc"])
        for role, row in runtime_by_role.items()
    }
    aucs["anchor"] = float(anchor["metrics"]["auc"])
    margins = {
        "true_minus_anchor": aucs["true"] - aucs["anchor"],
        "true_minus_corrupted": aucs["true"] - aucs["corrupted"],
        "true_minus_independent": aucs["true"] - aucs["independent"],
    }
    research_checks = {
        "true_within_anchor_tolerance": margins["true_minus_anchor"]
        >= -ANCHOR_TOLERANCE,
        "true_exceeds_corrupted": margins["true_minus_corrupted"]
        >= CONTROL_MARGIN,
        "true_exceeds_independent": margins["true_minus_independent"]
        >= CONTROL_MARGIN,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_r1_protocol_invalid"
        next_action = "repair the invalid R1 protocol or artifacts before interpretation"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_runtime_spn_r1_seed0_supported"
        next_action = (
            "run the same cipher and the second SPN at the frozen seed0 budget; "
            "only a two-cipher pass may authorize seed1"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_r1_seed0_not_supported"
        next_action = (
            "replace only the bit-to-pair global pooling with a runtime cell-token "
            "interaction stage that preserves E4 per-pair aligned cell relations; "
            "rerun a small local GIFT calibration before any 8192/class repeat"
        )
    return {
        "run_id": run_id,
        "cipher": cipher,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "thresholds": {
            "anchor_tolerance": ANCHOR_TOLERANCE,
            "control_margin": CONTROL_MARGIN,
        },
        "claim_scope": (
            f"{cipher} seed0 8192/class local diagnostic only; not multi-seed, "
            "multi-cipher, formal, or paper-scale evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "run seed1 before a seed0 pass",
            "run the heavier second-cipher matrix after a decisive first-cipher miss",
            "remote scale-up",
            "claim stable topology superiority",
        ],
    }


def _runtime_rows_by_role(
    cipher: str,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected = EXPECTED_RUNTIME_MODELS[cipher]
    by_model = {str(row.get("model")): row for row in rows}
    missing = [model for model in expected.values() if model not in by_model]
    if missing:
        raise ValueError(f"missing RTG1 runtime result rows: {missing}")
    return {role: by_model[model] for role, model in expected.items()}


def _anchor_row(cipher: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = EXPECTED_ANCHOR_MODELS[cipher]
    matches = [row for row in rows if row.get("model") == expected]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {expected} anchor row, found {len(matches)}"
        )
    return matches[0]


def _protocol_checks(
    cipher: str,
    runtime: dict[str, dict[str, Any]],
    anchor: dict[str, Any],
    r0_gate: dict[str, Any],
) -> dict[str, bool]:
    rows = list(runtime.values())
    static_fields = (
        "cipher",
        "rounds",
        "seed",
        "samples_per_class",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "train_key",
        "validation_key",
    )
    training_fields = (
        "epochs",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
    )
    all_rows = rows + [anchor]
    return {
        "r0_readiness_passed": r0_gate.get("status") == "pass"
        and r0_gate.get("decision") == "innovation1_runtime_spn_r0_readiness_passed",
        "runtime_model_set_complete": {
            str(row.get("model")) for row in rows
        }
        == set(EXPECTED_RUNTIME_MODELS[cipher].values()),
        "cipher_and_seed_match": all(
            row.get("cipher") == cipher and row.get("seed") == 0 for row in all_rows
        ),
        "same_data_protocol_as_anchor": all(
            all(row.get(field) == anchor.get(field) for field in static_fields)
            for row in rows
        ),
        "same_training_protocol_as_anchor": all(
            all(
                row.get("training", {}).get(field)
                == anchor.get("training", {}).get(field)
                for field in training_fields
            )
            for row in rows
        ),
        "frozen_r1_scale": all(
            row.get("samples_per_class") == 8192
            and row.get("training", {}).get("train_rows") == 16384
            and row.get("training", {}).get("validation_rows") == 8192
            and row.get("training", {}).get("epochs") == 10
            for row in rows
        ),
        "encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in all_rows
        ),
        "equal_runtime_parameter_count": len(
            {int(row.get("parameter_count", -1)) for row in rows}
        )
        == 1,
        "finite_auc_metrics": all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in all_rows
        ),
        "best_validation_checkpoints_restored": all(
            row.get("training", {}).get("checkpoint_metric") == "val_auc"
            and row.get("training", {}).get("restore_best_checkpoint") is True
            and row.get("training", {}).get("selected_checkpoint") == "best"
            for row in all_rows
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in rows
        ),
    }


__all__ = [
    "ANCHOR_TOLERANCE",
    "CONTROL_MARGIN",
    "adjudicate_runtime_spn_r1",
]
