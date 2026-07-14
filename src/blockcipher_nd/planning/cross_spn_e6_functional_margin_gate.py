from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    load_score_artifact,
)
from blockcipher_nd.planning.cross_spn_e5_source_objective_gate import (
    E5_BOOTSTRAP_REPLICATES,
    E5_BOOTSTRAP_SEED,
    adjudicate_e5_score_artifacts,
)
from blockcipher_nd.planning.cross_spn_target_adaptation_gate import (
    write_paired_score_csv_gz,
)
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment
from blockcipher_nd.training.metrics import binary_auc


E6_MODEL_ROLES = {
    "scratch": "gift_cross_spn_typed_cell_e6_scratch",
    "off_transfer": "gift_cross_spn_typed_cell_e6_from_present_off",
    "candidate_transfer": (
        "gift_cross_spn_typed_cell_e6_from_present_functional_margin"
    ),
    "placebo_transfer": (
        "gift_cross_spn_typed_cell_e6_from_present_shuffled_placebo"
    ),
}
E6_SOURCE_MODELS = {
    "off_transfer": "present_cross_spn_typed_cell_e5_off",
    "candidate_transfer": "present_cross_spn_typed_cell_e6_functional_margin",
    "placebo_transfer": "present_cross_spn_typed_cell_e6_shuffled_placebo",
}


def gate_cross_spn_e6_functional_margin(
    *,
    plan_path: Path,
    results_path: Path,
    score_artifact_paths: dict[str, Path],
    expected_target_seed: int,
    paired_scores_output: Path | None = None,
    bootstrap_replicates: int = E5_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = E5_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    errors: list[str] = []
    if expected_target_seed not in {2, 3}:
        errors.append("expected_target_seed must be 2 or 3")
    if bootstrap_replicates != E5_BOOTSTRAP_REPLICATES:
        errors.append("E6-R0 requires 10000 bootstrap replicates")
    if bootstrap_seed != E5_BOOTSTRAP_SEED:
        errors.append("E6-R0 requires bootstrap seed 20260715")
    if set(score_artifact_paths) != set(E6_MODEL_ROLES):
        errors.append("score artifact roles do not match frozen E6 roles")

    rows, read_errors = _read_jsonl(results_path)
    errors.extend(read_errors)
    alignment = validate_result_plan_alignment(
        plan_path,
        results_path,
        expected_rows=4,
    )
    errors.extend(alignment["errors"])
    by_model = {row.get("selected_model"): row for row in rows}
    if set(by_model) != set(E6_MODEL_ROLES.values()):
        errors.append("result models do not match frozen E6 roles")
    else:
        errors.extend(_result_errors(by_model, expected_target_seed))

    artifacts, artifact_errors = _load_artifacts(
        score_artifact_paths,
        by_model=by_model,
        expected_target_seed=expected_target_seed,
    )
    errors.extend(artifact_errors)
    if errors:
        return _invalid(errors, alignment)

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
    gate_pass = bool(adjudication["gate_pass"])
    adjudication["decision"] = (
        "e6_r0_target_seed_gate_pass"
        if gate_pass
        else "e6_r0_functional_margin_rejected"
    )
    return {
        "status": "pass",
        "errors": [],
        "expected_source_seed": 0,
        "expected_target_seed": expected_target_seed,
        "samples_per_class": 8192,
        "validation_samples_per_class": 4096,
        "target_epochs": 1,
        "models": E6_MODEL_ROLES,
        "alignment": alignment,
        "score_rows": len(artifacts["scratch"].labels),
        "score_pairing": "identical validation labels and sample_ids",
        "paired_scores_output": (
            str(paired_scores_output) if paired_scores_output is not None else None
        ),
        **adjudication,
        "research_decision_applied": True,
        "claim_scope": (
            "E6-R0 local 8192/class functional-margin target-adaptation "
            "diagnostic; not medium, formal, paper-scale, SOTA, or breakthrough evidence"
        ),
        "next_action": (
            "run_source_seed1_confirmation_on_same_target_seeds"
            if gate_pass
            else "stop_e6_r0_no_source_seed1_or_remote_scale"
        ),
        "stopped_actions": [] if gate_pass else _stopped_actions(),
    }


def gate_cross_spn_e6_functional_margin_joint(
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
        return _invalid(errors, None)

    gate_pass = all(report.get("gate_pass") is True for report in seed_reports)
    return {
        "status": "pass",
        "decision": (
            "e6_r0_source_seed0_two_target_gate_pass"
            if gate_pass
            else "e6_r0_functional_margin_rejected"
        ),
        "errors": [],
        "expected_source_seed": 0,
        "expected_target_seeds": [2, 3],
        "per_seed": {str(seed): by_seed[seed] for seed in (2, 3)},
        "gate_pass": gate_pass,
        "research_decision_applied": True,
        "claim_scope": (
            "two-target-seed E6-R0 local 8192/class diagnostic with one "
            "source seed; not source-seed-robust, medium, formal, or paper-scale evidence"
        ),
        "next_action": (
            "run_source_seed1_confirmation_on_same_target_seeds"
            if gate_pass
            else "stop_e6_r0_no_source_seed1_or_remote_scale"
        ),
        "stopped_actions": [] if gate_pass else _stopped_actions(),
    }


def _result_errors(
    by_model: dict[Any, dict[str, Any]],
    expected_target_seed: int,
) -> list[str]:
    errors: list[str] = []
    checkpoint_paths: set[str] = set()
    parameter_counts: set[Any] = set()
    for role, model in E6_MODEL_ROLES.items():
        row = by_model[model]
        expected = {
            "cipher": "GIFT-64",
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
                    f"role={role} {field} expected={value!r} actual={row.get(field)!r}"
                )
        training = row.get("training")
        if not isinstance(training, dict):
            errors.append(f"role={role} training metadata missing")
            continue
        if training.get("epochs") != 1 or training.get("selected_checkpoint") != "best":
            errors.append(f"role={role} must restore best after exactly one epoch")
        if training.get("validation_rows") != 8192:
            errors.append(f"role={role} validation rows must equal 8192")
        if any(float(item.get("train_auxiliary_loss", 0.0)) != 0.0 for item in row.get("history", [])):
            errors.append(f"role={role} target auxiliary loss must be zero")
        checkpoint = training.get("checkpoint_output")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
            errors.append(f"role={role} target checkpoint missing")
        else:
            checkpoint_paths.add(str(Path(checkpoint).resolve()))
        parameter_counts.add(row.get("parameter_count"))
        initialization = row.get("initialization")
        if not isinstance(initialization, dict):
            errors.append(f"role={role} initialization missing")
            continue
        if role == "scratch":
            if initialization.get("kind") != "scratch":
                errors.append("scratch role must use scratch initialization")
        else:
            if initialization.get("kind") != "checkpoint":
                errors.append(f"role={role} must load checkpoint")
            if initialization.get("strict_state_dict_load") is not True:
                errors.append(f"role={role} strict state load must pass")
            if initialization.get("source_model") != E6_SOURCE_MODELS[role]:
                errors.append(f"role={role} source model mismatch")
            if initialization.get("source_seed") != 0:
                errors.append(f"role={role} source seed mismatch")
            if initialization.get("source_samples_per_class") != 8192:
                errors.append(f"role={role} source sample budget mismatch")
            if initialization.get("source_epochs") != 10:
                errors.append(f"role={role} source epoch budget mismatch")
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
        model = E6_MODEL_ROLES[role]
        metadata = artifact.metadata
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
            errors.append(f"role={role} score/result AUC mismatch")
    if len(artifacts) == len(E6_MODEL_ROLES):
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
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"results invalid JSON line={index}: {exc.msg}")
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            errors.append(f"results row line={index} must be an object")
    return rows, errors


def _stopped_actions() -> list[str]:
    return [
        "source_seed1_confirmation",
        "65536_per_class_remote_medium",
        "262144_per_class",
        "1000000_per_class",
        "functional_margin_tuning",
        "extra_target_epochs",
    ]


def _invalid(
    errors: list[str],
    alignment: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "invalid_e6_r0_evidence",
        "errors": errors,
        "alignment": alignment,
        "research_decision_applied": False,
        "claim_scope": "invalid E6-R0 evidence",
        "next_action": "repair_protocol_or_artifacts_before_interpretation",
        "stopped_actions": [
            "source_seed1_confirmation",
            "65536_per_class_remote_medium",
        ],
    }


__all__ = [
    "E6_MODEL_ROLES",
    "E6_SOURCE_MODELS",
    "gate_cross_spn_e6_functional_margin",
    "gate_cross_spn_e6_functional_margin_joint",
]
