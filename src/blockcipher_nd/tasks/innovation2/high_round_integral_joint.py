from __future__ import annotations

import math
from statistics import mean
from typing import Any


EXPECTED_ROLES = {"anchor", "candidate", "linear", "control"}
PASS_DECISION = "innovation2_high_round_integral_bridge_advance"

PROTOCOL_FIELDS = (
    "cipher",
    "task",
    "rounds",
    "train_total_rows",
    "validation",
    "test",
    "multisets_per_sample",
    "texts_per_multiset",
    "input_bits",
    "input_view",
    "negative_mode",
    "key_sampling",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "loss",
    "optimizer",
    "paper_tensor_concat_assumption",
)


def adjudicate_joint_high_round_integral(
    *,
    run_id: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(sources) != 2:
        raise ValueError("joint bridge adjudication requires exactly two sources")

    rows: list[dict[str, Any]] = []
    source_protocols: list[dict[str, Any]] = []
    source_validity: list[bool] = []
    for source in sources:
        gate = source.get("gate")
        result_rows = source.get("rows")
        artifact_root = str(source.get("artifact_root", ""))
        if not isinstance(gate, dict) or not isinstance(result_rows, list):
            raise ValueError("each joint source requires a gate and result rows")
        if not result_rows or not all(isinstance(row, dict) for row in result_rows):
            raise ValueError("each joint source requires non-empty JSONL rows")

        try:
            seeds = {int(row["seed"]) for row in result_rows}
        except (KeyError, TypeError, ValueError):
            seeds = set()
        source_seed_valid = len(seeds) == 1
        seed = seeds.pop() if source_seed_valid else -1
        by_role = {str(row.get("role")): row for row in result_rows}
        roles_match = set(by_role) == EXPECTED_ROLES and len(by_role) == len(
            result_rows
        )
        if not roles_match:
            raise ValueError("source result roles must be anchor/candidate/linear/control")

        candidate = by_role["candidate"]
        linear = by_role["linear"]
        control = by_role["control"]
        metrics = gate.get("metrics")
        plan_checks = gate.get("bridge_plan_checks")
        readiness_checks = gate.get("readiness_checks")
        signal_checks = gate.get("bridge_signal_checks")
        readjudication = gate.get("readjudication")
        if not all(
            isinstance(value, dict)
            for value in (
                metrics,
                plan_checks,
                readiness_checks,
                signal_checks,
                readjudication,
            )
        ):
            raise ValueError("source gate is missing bridge or readjudication checks")

        candidate_auc = float(candidate["test_auc"])
        linear_auc = float(linear["test_auc"])
        shuffled_auc = float(control["test_auc"])
        shuffled_fit_auc = float(control["fit_validation_auc"])
        prior_auc = float(metrics["architecture_prior_oriented_auc"])
        parity_auc = float(metrics["strongest_oriented_fixed_parity_auc"])
        numeric_values = (
            candidate_auc,
            linear_auc,
            shuffled_auc,
            shuffled_fit_auc,
            prior_auc,
            parity_auc,
        )
        source_revision_valid = bool(
            readjudication.get("source_revision_matches_expected")
        )
        anchor_exclusion_valid = bool(
            readjudication.get("anchor_layout_invalidated")
        ) == (seed == 0)
        valid = all(
            (
                gate.get("gate_mode") == "bridge",
                int(gate.get("rounds", -1)) == 8,
                str(gate.get("run_id", "")) == str(candidate.get("run_id", "")),
                all(bool(value) for value in plan_checks.values()),
                all(bool(value) for value in readiness_checks.values()),
                source_revision_valid,
                anchor_exclusion_valid,
                source_seed_valid,
                all(math.isfinite(value) for value in numeric_values),
            )
        )
        source_validity.append(valid)
        source_protocols.append({field: candidate.get(field) for field in PROTOCOL_FIELDS})
        rows.append(
            {
                "run_id": run_id,
                "task": "innovation2_present_high_round_integral_joint_bridge",
                "cipher": "PRESENT-80",
                "rounds": 8,
                "seed": seed,
                "source_run_id": str(gate.get("run_id", "")),
                "source_artifact_root": artifact_root,
                "source_status": str(gate.get("status", "")),
                "source_decision": str(gate.get("decision", "")),
                "source_valid": valid,
                "source_seed_valid": source_seed_valid,
                "source_revision_matches_expected": source_revision_valid,
                "anchor_layout_invalidated": bool(
                    readjudication.get("anchor_layout_invalidated")
                ),
                "candidate_test_accuracy": float(candidate["test_accuracy"]),
                "candidate_test_auc": candidate_auc,
                "linear_test_auc": linear_auc,
                "shuffled_test_auc": shuffled_auc,
                "shuffled_fit_validation_auc": shuffled_fit_auc,
                "architecture_prior_oriented_auc": prior_auc,
                "strongest_oriented_fixed_parity_auc": parity_auc,
                "candidate_linear_auc_delta": candidate_auc - linear_auc,
                "candidate_architecture_prior_auc_delta": candidate_auc - prior_auc,
                "candidate_strongest_fixed_parity_auc_delta": candidate_auc
                - parity_auc,
                "candidate_auc_gate_passed": candidate_auc >= 0.53,
                "architecture_prior_margin_gate_passed": candidate_auc - prior_auc
                >= 0.01,
                "fixed_parity_margin_gate_passed": candidate_auc - parity_auc
                >= 0.01,
                "shuffled_fit_gate_passed": abs(shuffled_fit_auc - 0.5) <= 0.03,
            }
        )

    rows.sort(key=lambda row: int(row["seed"]))
    exact_seed_pair = [int(row["seed"]) for row in rows] == [0, 1]
    protocol_matches = source_protocols[0] == source_protocols[1]
    both_sources_valid = all(source_validity)
    both_source_gates_pass = all(
        row["source_status"] == "pass" and row["source_decision"] == PASS_DECISION
        for row in rows
    )
    signal_checks = {
        "both_candidate_test_auc_at_least_0_53": all(
            bool(row["candidate_auc_gate_passed"]) for row in rows
        ),
        "both_candidate_beat_architecture_prior_by_0_01": all(
            bool(row["architecture_prior_margin_gate_passed"]) for row in rows
        ),
        "both_candidate_beat_fixed_parity_by_0_01": all(
            bool(row["fixed_parity_margin_gate_passed"]) for row in rows
        ),
        "both_shuffled_fit_validation_auc_within_0_03": all(
            bool(row["shuffled_fit_gate_passed"]) for row in rows
        ),
        "both_source_gates_pass": both_source_gates_pass,
    }
    validity_checks = {
        "exact_seed0_seed1_pair": exact_seed_pair,
        "same_frozen_protocol_except_seed": protocol_matches,
        "both_sources_valid_and_revision_matched": both_sources_valid,
    }

    if not all(validity_checks.values()):
        status = "fail"
        decision = "innovation2_high_round_integral_two_seed_bridge_invalid"
        next_action = (
            "Reject the joint interpretation and repair the exact seed, protocol, "
            "source-revision, role, or artifact mismatch before any new training."
        )
    elif all(signal_checks.values()):
        status = "pass"
        decision = "innovation2_high_round_integral_two_seed_bridge_confirmed"
        next_action = (
            "Freeze the 2^21-total-train-row, 50-epoch paper-reference approximation "
            "with repaired anchor layout and explicit assumptions for unpublished "
            "Nf, dropout, block count, and learning-rate parameters; keep r9 and "
            "other ciphers stopped until that reference gate completes."
        )
    else:
        status = "hold"
        decision = "innovation2_high_round_integral_two_seed_bridge_not_confirmed"
        next_action = (
            "Stop mechanical scale-up. Audit seed sensitivity, checkpoint dynamics, "
            "and the exact failed candidate/control margin before considering 2^21, "
            "r9, GIFT, or AES."
        )

    metric_names = (
        "candidate_test_accuracy",
        "candidate_test_auc",
        "linear_test_auc",
        "shuffled_test_auc",
        "shuffled_fit_validation_auc",
        "architecture_prior_oriented_auc",
        "strongest_oriented_fixed_parity_auc",
        "candidate_linear_auc_delta",
        "candidate_architecture_prior_auc_delta",
        "candidate_strongest_fixed_parity_auc_delta",
    )
    metrics = {
        f"{name}_{suffix}": statistic
        for name in metric_names
        for suffix, statistic in (
            ("min", min(float(row[name]) for row in rows)),
            ("mean", mean(float(row[name]) for row in rows)),
            ("max", max(float(row[name]) for row in rows)),
        )
    }
    gate = {
        "status": status,
        "decision": decision,
        "run_id": run_id,
        "gate_mode": "high-round-integral-bridge-joint",
        "source_run_ids": [str(row["source_run_id"]) for row in rows],
        "validity_checks": validity_checks,
        "signal_checks": signal_checks,
        "thresholds": {
            "candidate_test_auc_min": 0.53,
            "candidate_architecture_prior_auc_delta_min": 0.01,
            "candidate_fixed_parity_auc_delta_min": 0.01,
            "shuffled_fit_validation_auc_abs_delta_from_chance_max": 0.03,
        },
        "metrics": metrics,
        "training_performed": False,
        "next_action": next_action,
        "claim_scope": (
            "two-seed remote PRESENT-80 r8 high-round integral-neural bridge at "
            "262144 total training rows (approximately 131072/class) per seed; "
            "not paper-scale, not exact reproduction, not a deterministic integral "
            "proof, and not a breakthrough claim"
        ),
    }
    return {"rows": rows, "gate": gate}


__all__ = ["adjudicate_joint_high_round_integral"]
