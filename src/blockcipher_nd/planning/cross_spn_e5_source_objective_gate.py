from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    load_score_artifact,
)
from blockcipher_nd.planning.cross_spn_target_adaptation_gate import (
    paired_stratified_bootstrap_auc_differences,
    write_paired_score_csv_gz,
)
from blockcipher_nd.planning.result_alignment import (
    validate_result_plan_alignment,
)
from blockcipher_nd.training.metrics import binary_auc


E5_MODEL_ROLES = {
    "scratch": "gift_cross_spn_typed_cell_e5_scratch",
    "off_transfer": "gift_cross_spn_typed_cell_e5_from_present_off",
    "candidate_transfer": (
        "gift_cross_spn_typed_cell_e5_from_present_true_shuffled"
    ),
    "placebo_transfer": (
        "gift_cross_spn_typed_cell_e5_from_present_shuffled_placebo"
    ),
}
E5_SOURCE_MODELS = {
    "off_transfer": "present_cross_spn_typed_cell_e5_off",
    "candidate_transfer": "present_cross_spn_typed_cell_e5_true_shuffled",
    "placebo_transfer": "present_cross_spn_typed_cell_e5_shuffled_placebo",
}
E5_CONTROL_ROLES = ("off_transfer", "placebo_transfer", "scratch")
E5_MARGIN_THRESHOLD = 0.004
E5_BOOTSTRAP_REPLICATES = 10_000
E5_BOOTSTRAP_SEED = 20260715


def gate_cross_spn_e5_source_objective(
    *,
    plan_path: Path,
    results_path: Path,
    score_artifact_paths: dict[str, Path],
    expected_target_seed: int,
    expected_source_seed: int = 0,
    paired_scores_output: Path | None = None,
    bootstrap_replicates: int = E5_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = E5_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    errors: list[str] = []
    if expected_target_seed not in {2, 3}:
        errors.append("expected_target_seed must be 2 or 3")
    if expected_source_seed != 0:
        errors.append("E5-R0 Phase 1A requires source seed0")
    if bootstrap_replicates != E5_BOOTSTRAP_REPLICATES:
        errors.append("E5-R0 requires 10000 bootstrap replicates")
    if bootstrap_seed != E5_BOOTSTRAP_SEED:
        errors.append("E5-R0 requires bootstrap seed 20260715")
    if set(score_artifact_paths) != set(E5_MODEL_ROLES):
        errors.append("score artifact roles do not match the frozen E5 roles")

    rows, read_errors = _read_jsonl(results_path)
    errors.extend(read_errors)
    alignment = validate_result_plan_alignment(
        plan_path,
        results_path,
        expected_rows=4,
    )
    errors.extend(alignment["errors"])
    by_model = {row.get("selected_model"): row for row in rows}
    if set(by_model) != set(E5_MODEL_ROLES.values()):
        errors.append("result models do not match the frozen E5 roles")
    else:
        errors.extend(
            _result_errors(
                by_model,
                expected_target_seed=expected_target_seed,
                expected_source_seed=expected_source_seed,
            )
        )

    artifacts, artifact_errors = _load_artifacts(
        score_artifact_paths,
        by_model=by_model,
        expected_target_seed=expected_target_seed,
    )
    errors.extend(artifact_errors)
    if errors:
        return _invalid(errors, alignment=alignment)

    if paired_scores_output is not None:
        write_paired_score_csv_gz(
            paired_scores_output,
            artifacts,
            target_seed=expected_target_seed,
        )
    adjudication = adjudicate_e5_score_artifacts(
        artifacts,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "status": "pass",
        "decision": adjudication["decision"],
        "errors": [],
        "expected_source_seed": expected_source_seed,
        "expected_target_seed": expected_target_seed,
        "samples_per_class": 8192,
        "validation_samples_per_class": 4096,
        "target_epochs": 1,
        "models": E5_MODEL_ROLES,
        "alignment": alignment,
        "score_rows": len(artifacts["scratch"].labels),
        "score_pairing": "identical validation labels and sample_ids",
        "paired_scores_output": (
            str(paired_scores_output) if paired_scores_output is not None else None
        ),
        **adjudication,
        "research_decision_applied": True,
        "claim_scope": (
            "E5-R0 local 8192/class source-objective target-adaptation "
            "diagnostic; not medium, formal, paper-scale, SOTA, or breakthrough evidence"
        ),
        "next_action": (
            "run_source_seed1_confirmation_on_same_target_seeds"
            if adjudication["decision"] == "e5_r0_target_seed_gate_pass"
            else "stop_e5_r0_no_source_seed1_or_remote_scale"
        ),
        "stopped_actions": (
            []
            if adjudication["decision"] == "e5_r0_target_seed_gate_pass"
            else [
                "source_seed1_confirmation",
                "65536_per_class_remote_medium",
                "262144_per_class",
                "1000000_per_class",
                "auxiliary_scale_tuning",
                "extra_target_epochs",
            ]
        ),
    }


def adjudicate_e5_score_artifacts(
    artifacts: dict[str, EnsembleScoreArtifact],
    *,
    bootstrap_replicates: int = E5_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = E5_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    aucs = {
        role: binary_auc(artifact.labels, artifact.probabilities)
        for role, artifact in artifacts.items()
    }
    candidate_auc = aucs["candidate_transfer"]
    margins = {
        role: candidate_auc - aucs[role] for role in E5_CONTROL_ROLES
    }
    bootstrap = paired_stratified_bootstrap_auc_differences(
        artifacts["candidate_transfer"].labels,
        {role: artifact.probabilities for role, artifact in artifacts.items()},
        candidate_role="candidate_transfer",
        control_roles=E5_CONTROL_ROLES,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    point_pass = {
        role: margin >= E5_MARGIN_THRESHOLD for role, margin in margins.items()
    }
    ci_pass = {
        role: bootstrap["comparisons"][role]["ci_lower"] > 0.0
        for role in E5_CONTROL_ROLES
    }
    gate_pass = all(point_pass.values()) and all(ci_pass.values())
    return {
        "decision": (
            "e5_r0_target_seed_gate_pass"
            if gate_pass
            else "e5_r0_source_objective_rejected"
        ),
        "aucs": aucs,
        "margins": margins,
        "thresholds": {
            "candidate_margin_each_control": E5_MARGIN_THRESHOLD,
            "paired_ci_lower_strictly_greater_than": 0.0,
        },
        "point_pass": point_pass,
        "ci_pass": ci_pass,
        "bootstrap": bootstrap,
        "gate_pass": gate_pass,
    }


def gate_cross_spn_e5_source_objective_joint(
    seed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    by_seed: dict[int, dict[str, Any]] = {}
    for report in seed_reports:
        seed = report.get("expected_target_seed")
        if type(seed) is not int or seed in by_seed:
            errors.append(f"invalid or duplicate target seed: {seed!r}")
            continue
        by_seed[seed] = report
        if report.get("status") != "pass" or report.get("errors") != []:
            errors.append(f"target seed {seed} is not valid pass evidence")
        if report.get("expected_source_seed") != 0:
            errors.append(f"target seed {seed} did not use source seed0")
        if report.get("research_decision_applied") is not True:
            errors.append(f"target seed {seed} did not apply its decision")
    if set(by_seed) != {2, 3}:
        errors.append("joint gate requires target seeds 2 and 3")
    if errors:
        return _invalid(errors)

    gate_pass = all(report.get("gate_pass") is True for report in seed_reports)
    decision = (
        "e5_r0_source_seed0_two_target_gate_pass"
        if gate_pass
        else "e5_r0_source_objective_rejected"
    )
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "expected_source_seed": 0,
        "expected_target_seeds": [2, 3],
        "per_seed": {str(seed): by_seed[seed] for seed in (2, 3)},
        "gate_pass": gate_pass,
        "research_decision_applied": True,
        "claim_scope": (
            "two-target-seed E5-R0 local 8192/class diagnostic with one fixed "
            "source seed; not source-seed-robust, medium, formal, or paper-scale evidence"
        ),
        "next_action": (
            "run_source_seed1_confirmation_on_same_target_seeds"
            if gate_pass
            else "stop_e5_r0_no_source_seed1_or_remote_scale"
        ),
        "stopped_actions": (
            []
            if gate_pass
            else [
                "source_seed1_confirmation",
                "65536_per_class_remote_medium",
                "262144_per_class",
                "1000000_per_class",
                "auxiliary_scale_tuning",
                "extra_target_epochs",
            ]
        ),
    }


def _result_errors(
    by_model: dict[Any, dict[str, Any]],
    *,
    expected_target_seed: int,
    expected_source_seed: int,
) -> list[str]:
    errors: list[str] = []
    parameter_counts: set[Any] = set()
    checkpoint_paths: set[str] = set()
    for role, model in E5_MODEL_ROLES.items():
        row = by_model[model]
        label = f"role={role}"
        expected = {
            "cipher": "GIFT-64",
            "cipher_key": "gift64",
            "rounds": 6,
            "seed": expected_target_seed,
            "samples_per_class": 8192,
            "pairs_per_sample": 4,
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
        }
        for field, value in expected.items():
            if row.get(field) != value:
                errors.append(
                    f"{label} {field} expected={value!r} actual={row.get(field)!r}"
                )
        training = row.get("training")
        if not isinstance(training, dict):
            errors.append(f"{label} training must be an object")
            continue
        if training.get("epochs") != 1 or training.get("selected_checkpoint") != "best":
            errors.append(f"{label} must use one epoch and restored-best checkpoint")
        if training.get("checkpoint_metric") != "val_auc":
            errors.append(f"{label} checkpoint metric must be val_auc")
        if training.get("validation_rows") != 8192:
            errors.append(f"{label} validation rows must equal 8192")
        if any(float(item.get("train_auxiliary_loss", 0.0)) != 0.0 for item in row.get("history", [])):
            errors.append(f"{label} target auxiliary loss must be disabled")
        checkpoint = training.get("checkpoint_output")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
            errors.append(f"{label} checkpoint is missing")
        else:
            checkpoint_paths.add(str(Path(checkpoint).resolve()))
        parameter_counts.add(row.get("parameter_count"))
        initialization = row.get("initialization")
        if not isinstance(initialization, dict):
            errors.append(f"{label} initialization must be an object")
            continue
        if role == "scratch":
            if initialization.get("kind") != "scratch":
                errors.append("scratch role must use scratch initialization")
        else:
            expected_source_model = E5_SOURCE_MODELS[role]
            if initialization.get("kind") != "checkpoint":
                errors.append(f"{label} must use checkpoint initialization")
            if initialization.get("strict_state_dict_load") is not True:
                errors.append(f"{label} strict state load must be true")
            if initialization.get("source_model") != expected_source_model:
                errors.append(f"{label} source model mismatch")
            if initialization.get("source_seed") != expected_source_seed:
                errors.append(f"{label} source seed mismatch")
            if initialization.get("source_samples_per_class") != 8192:
                errors.append(f"{label} source sample budget mismatch")
            if initialization.get("source_epochs") != 10:
                errors.append(f"{label} source epoch budget mismatch")
    if parameter_counts != {196003}:
        errors.append(f"parameter counts must all equal 196003: {parameter_counts}")
    if len(checkpoint_paths) != 4:
        errors.append("target checkpoints must be four distinct files")
    return errors


def _load_artifacts(
    paths: dict[str, Path],
    *,
    by_model: dict[Any, dict[str, Any]],
    expected_target_seed: int,
) -> tuple[dict[str, EnsembleScoreArtifact], list[str]]:
    artifacts: dict[str, EnsembleScoreArtifact] = {}
    errors: list[str] = []
    for role, path in paths.items():
        try:
            artifact = load_score_artifact(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"role={role} score artifact unreadable: {exc}")
            continue
        artifacts[role] = artifact
        metadata = artifact.metadata
        model = E5_MODEL_ROLES[role]
        if metadata.get("model_key") != model:
            errors.append(f"role={role} score model mismatch")
        if metadata.get("seed") != expected_target_seed:
            errors.append(f"role={role} score seed mismatch")
        if metadata.get("score_split") != "validation":
            errors.append(f"role={role} score split must be validation")
        if metadata.get("validation_samples_per_class") != 4096:
            errors.append(f"role={role} validation score budget mismatch")
        row = by_model.get(model, {})
        expected_checkpoint = (row.get("training") or {}).get("checkpoint_output")
        actual_checkpoint = metadata.get("checkpoint_path")
        if not isinstance(actual_checkpoint, str) or not isinstance(expected_checkpoint, str):
            errors.append(f"role={role} checkpoint path metadata missing")
        elif Path(actual_checkpoint).resolve() != Path(expected_checkpoint).resolve():
            errors.append(f"role={role} checkpoint path mismatch")
        score_auc = binary_auc(artifact.labels, artifact.probabilities)
        result_auc = (row.get("metrics") or {}).get("auc")
        if not isinstance(result_auc, (int, float)) or not np.isclose(
            score_auc,
            float(result_auc),
            rtol=0.0,
            atol=1e-12,
        ):
            errors.append(
                f"role={role} score/result AUC mismatch "
                f"score={score_auc!r} result={result_auc!r}"
            )
    if len(artifacts) == len(E5_MODEL_ROLES):
        first = artifacts["scratch"]
        for role, artifact in artifacts.items():
            if not np.array_equal(first.labels, artifact.labels):
                errors.append(f"role={role} labels are not paired")
            if not np.array_equal(first.sample_ids, artifact.sample_ids):
                errors.append(f"role={role} sample_ids are not paired")
    return artifacts, errors


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [], [f"results unreadable: {exc}"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"results invalid JSON line={index}: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"results row line={index} must be an object")
            continue
        rows.append(row)
    return rows, errors


def _invalid(
    errors: list[str],
    *,
    alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "invalid_e5_r0_evidence",
        "errors": errors,
        "alignment": alignment,
        "research_decision_applied": False,
        "claim_scope": "invalid E5-R0 evidence",
        "next_action": "repair_protocol_or_artifacts_before_interpretation",
        "stopped_actions": [
            "source_seed1_confirmation",
            "65536_per_class_remote_medium",
        ],
    }


__all__ = [
    "E5_BOOTSTRAP_REPLICATES",
    "E5_BOOTSTRAP_SEED",
    "E5_CONTROL_ROLES",
    "E5_MARGIN_THRESHOLD",
    "E5_MODEL_ROLES",
    "adjudicate_e5_score_artifacts",
    "gate_cross_spn_e5_source_objective",
    "gate_cross_spn_e5_source_objective_joint",
]
