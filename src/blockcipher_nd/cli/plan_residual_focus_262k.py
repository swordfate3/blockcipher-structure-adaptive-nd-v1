from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.plan_bucket_residual_262k import (
    DEFAULT_POSTPROCESS_STATUS,
    DEFAULT_SEED_PLAN_PREFIX,
    READY_DECISION,
    _command,
    _feature_export_command,
    _is_windows_path,
    _read_models_json,
    _seed_from_run_id,
    _seed_warnings,
    _span_command,
)


DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_action_plan.json")
DEFAULT_ARTIFACT_ROOT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k")
RAW117_PREFIXES = (
    "aux_depth_cell_",
    "aux_depth_word_",
    "aux_word_global_",
    "primary_depth_trailword_",
)
RESIDUAL_PREFIXES = ("aux_depth_word_", "aux_word_")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the 262144/class PRESENT r8 residual-focused correction gate "
            "after trail-position score artifacts are retrieved."
        )
    )
    parser.add_argument("--postprocess-status", type=Path, default=DEFAULT_POSTPROCESS_STATUS)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def plan_residual_focus_262k(
    *,
    postprocess_status: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    if not postprocess_status.exists():
        return _pending_report(
            postprocess_status=postprocess_status,
            artifact_root=artifact_root,
            reason="postprocess_status_missing",
            missing=[str(postprocess_status)],
            source_status=None,
            source_decision=None,
        )
    source = json.loads(postprocess_status.read_text(encoding="utf-8"))
    source_status = str(source.get("status", "unknown"))
    source_decision = str(source.get("decision", ""))
    if source_status != "pass":
        return _pending_report(
            postprocess_status=postprocess_status,
            artifact_root=artifact_root,
            reason="trail_position_262k_postprocess_not_ready",
            missing=_missing_from_postprocess(source),
            source_status=source_status,
            source_decision=source_decision,
        )

    seed_plans = []
    errors: list[str] = []
    for index, run in enumerate(source.get("runs", [])):
        if not isinstance(run, dict):
            errors.append(f"run_{index}_not_object")
            continue
        try:
            seed_plans.append(_seed_plan(run, artifact_root=artifact_root))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        return {
            "status": "fail",
            "decision": "hold_residual_focus_262k_plan_inputs_missing",
            "should_run": False,
            "postprocess_status": str(postprocess_status),
            "source_status": source_status,
            "source_decision": source_decision,
            "artifact_root": str(artifact_root),
            "errors": errors,
            "claim_scope": _claim_scope(),
        }

    return {
        "status": "pass",
        "decision": "residual_focus_262k_action_plan_ready",
        "should_run": True,
        "postprocess_status": str(postprocess_status),
        "source_status": source_status,
        "source_decision": source_decision,
        "source_gate_assessment": _source_gate_assessment(source_decision),
        "artifact_root": str(artifact_root),
        "expected_score_rows": int(source.get("expected_score_rows", 262144)),
        "base": "logit_mean(trail_position, raw117_matched)",
        "raw117_feature_prefixes": list(RAW117_PREFIXES),
        "residual_feature_prefixes": list(RESIDUAL_PREFIXES),
        "candidates": ["focus05", "focus10"],
        "controls": ["uniform_no_focus", "focus10_label_shuffle"],
        "seeds": seed_plans,
        "commands": [command for seed in seed_plans for command in seed["commands"]],
        "control_commands": [command for seed in seed_plans for command in seed["control_commands"]],
        "next_action": (
            "Run these commands only after the 262k trail-position postprocess remains pass; "
            "compare global metrics and train-derived hard residual slice metrics before any "
            "remote-scale claim. If source_gate_assessment is mixed, use this only as a "
            "residual-diagnostic follow-up, not as promotion of the trail-position scale gate."
        ),
        "claim_scope": _claim_scope(),
    }


def _pending_report(
    *,
    postprocess_status: Path,
    artifact_root: Path,
    reason: str,
    missing: list[str],
    source_status: str | None,
    source_decision: str | None,
) -> dict[str, Any]:
    return {
        "status": "pending",
        "decision": "wait_for_trail_position_262k_score_artifacts",
        "should_run": False,
        "reason": reason,
        "postprocess_status": str(postprocess_status),
        "source_status": source_status,
        "source_decision": source_decision,
        "artifact_root": str(artifact_root),
        "missing": missing,
        "next_action": "Let the local tmux watchers retrieve and verify the 262k score artifacts first.",
        "claim_scope": _claim_scope(),
    }


def _missing_from_postprocess(source: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for run in source.get("runs", []):
        if isinstance(run, dict):
            missing.extend(str(path) for path in run.get("missing_score_files", []))
            if int(run.get("train_rows", 0)) < int(source.get("expected_matrix_rows", 2)):
                missing.append(str(run.get("train_matrix", "")))
    return sorted({item for item in missing if item})


def _seed_plan(run: dict[str, Any], *, artifact_root: Path) -> dict[str, Any]:
    run_root = Path(str(run.get("run_root", "")))
    if not run_root:
        raise ValueError("run_root_missing")
    run_id = str(run.get("run_id") or run_root.name)
    seed = _seed_from_run_id(run_id)
    eval_plan = Path(f"{DEFAULT_SEED_PLAN_PREFIX}{seed}.csv")
    validation_trail_scores = run_root / "score_artifacts" / "trail_position"
    trail_models = _read_models_json(validation_trail_scores / "models.json")
    remote_checkpoint_path = str(trail_models.get("checkpoint_path", ""))
    if not remote_checkpoint_path:
        raise ValueError(f"{run_id}: trail_position checkpoint_path_missing")
    checkpoint_path = _prefer_retrieved_checkpoint(run_root, remote_checkpoint_path)
    remote_checkpoint_reference = _is_windows_path(checkpoint_path)

    seed_root = artifact_root / f"seed{seed}"
    paths = {
        "dataset_cache_root": seed_root / "dataset_cache",
        "train_feature_dir": seed_root / "train_trail_position_stats_features",
        "validation_feature_dir": seed_root / "validation_trail_position_stats_features",
        "train_span_blocks": seed_root / "train_span_blocks",
        "validation_span_blocks": seed_root / "validation_span_blocks",
        "train_span_summary": seed_root / "train_span_summary_features",
        "validation_span_summary": seed_root / "validation_span_summary_features",
        "train_trail_scores": seed_root / "train_trail_position_scores",
        "train_raw117_scores": seed_root / "train_raw117_scores",
        "validation_raw117_scores": seed_root / "validation_raw117_scores",
        "raw117_report": seed_root / "raw117_report.json",
        "uniform_train_scores": seed_root / "residual_uniform_train_scores",
        "uniform_validation_scores": seed_root / "residual_uniform_validation_scores",
        "uniform_report": seed_root / "residual_uniform_report.json",
        "uniform_slice_eval": seed_root / "residual_uniform_slice_eval.json",
        "focus05_train_scores": seed_root / "residual_focus05_train_scores",
        "focus05_validation_scores": seed_root / "residual_focus05_validation_scores",
        "focus05_report": seed_root / "residual_focus05_report.json",
        "focus05_slice_eval": seed_root / "residual_focus05_slice_eval.json",
        "focus10_train_scores": seed_root / "residual_focus10_train_scores",
        "focus10_validation_scores": seed_root / "residual_focus10_validation_scores",
        "focus10_report": seed_root / "residual_focus10_report.json",
        "focus10_slice_eval": seed_root / "residual_focus10_slice_eval.json",
        "focus10_shuffle_train_scores": seed_root / "residual_focus10_labelshuffle_train_scores",
        "focus10_shuffle_validation_scores": seed_root / "residual_focus10_labelshuffle_validation_scores",
        "focus10_shuffle_report": seed_root / "residual_focus10_labelshuffle_report.json",
        "focus10_shuffle_slice_eval": seed_root / "residual_focus10_labelshuffle_slice_eval.json",
    }
    commands = [
        _feature_export_command(eval_plan, seed, "train", paths["train_feature_dir"], None, paths),
        _feature_export_command(eval_plan, seed, "validation", paths["validation_feature_dir"], validation_trail_scores, paths),
        _span_command(paths["train_feature_dir"], paths["train_span_blocks"], paths["train_span_summary"]),
        _span_command(paths["validation_feature_dir"], paths["validation_span_blocks"], paths["validation_span_summary"]),
        _checkpoint_score_command(checkpoint_path, eval_plan, seed, paths),
        _raw117_command(paths, seed),
        _residual_command(paths, validation_trail_scores, seed, label="focus05", focus_fraction=0.05),
        _slice_eval_command(paths, validation_trail_scores, corrected_key="focus05", focus_fraction=0.05),
        _residual_command(paths, validation_trail_scores, seed, label="focus10", focus_fraction=0.10),
        _slice_eval_command(paths, validation_trail_scores, corrected_key="focus10", focus_fraction=0.10),
    ]
    control_commands = [
        _residual_command(paths, validation_trail_scores, seed, label="uniform", focus_fraction=0.0),
        _slice_eval_command(paths, validation_trail_scores, corrected_key="uniform", focus_fraction=0.10),
        _residual_command(
            paths,
            validation_trail_scores,
            seed,
            label="focus10_shuffle",
            focus_fraction=0.10,
            suffix=["--shuffle-train-labels", "--shuffle-seed", str(9700 + seed)],
        ),
        _slice_eval_command(paths, validation_trail_scores, corrected_key="focus10_shuffle", focus_fraction=0.10),
    ]
    return {
        "seed": seed,
        "run_id": run_id,
        "run_root": str(run_root),
        "eval_plan": str(eval_plan),
        "eval_row_index": 1,
        "validation_trail_position_scores": str(validation_trail_scores),
        "train_trail_position_checkpoint": checkpoint_path,
        "remote_train_trail_position_checkpoint": remote_checkpoint_path,
        "remote_checkpoint_reference": remote_checkpoint_reference,
        "warnings": _seed_warnings(remote_checkpoint_reference),
        "artifact_root": str(seed_root),
        "commands": commands,
        "control_commands": control_commands,
        "planned_outputs": {
            key: str(paths[key])
            for key in (
                "raw117_report",
                "focus05_report",
                "focus05_slice_eval",
                "focus10_report",
                "focus10_slice_eval",
                "uniform_report",
                "uniform_slice_eval",
                "focus10_shuffle_report",
                "focus10_shuffle_slice_eval",
            )
        },
    }


def _prefer_retrieved_checkpoint(run_root: Path, checkpoint_path: str) -> str:
    if not _is_windows_path(checkpoint_path):
        return checkpoint_path
    checkpoint_name = checkpoint_path.replace("\\", "/").split("/")[-1]
    local_checkpoint = run_root / "checkpoints" / checkpoint_name
    if local_checkpoint.exists():
        return str(local_checkpoint)
    return checkpoint_path


def _checkpoint_score_command(
    checkpoint_path: str,
    eval_plan: Path,
    seed: int,
    paths: dict[str, Path],
) -> str:
    return _command(
        [
            "UV_CACHE_DIR=/tmp/uv-cache",
            "uv",
            "run",
            "scripts/export-checkpoint-scores",
            "--checkpoint",
            checkpoint_path,
            "--eval-plan",
            str(eval_plan),
            "--eval-row-index",
            "1",
            "--split",
            "train",
            "--model-key",
            "present_trail_position_stats_pairset",
            "--hidden-bits",
            "32",
            "--model-options",
            '{"activation":"gelu","norm":"layernorm","stats_hidden_bits":64,"trail_depth":4,"trail_words_per_depth":9}',
            "--expert-family",
            "trail_position",
            "--candidate-status",
            "weak_positive",
            "--dataset-cache-root",
            str(paths["dataset_cache_root"] / "train_scores"),
            "--progress-output",
            str(paths["dataset_cache_root"] / f"seed{seed}_train_score_export_progress.jsonl"),
            "--output-dir",
            str(paths["train_trail_scores"]),
        ]
    )


def _raw117_command(paths: dict[str, Path], seed: int) -> str:
    parts = [
        "UV_CACHE_DIR=/tmp/uv-cache",
        "uv",
        "run",
        "scripts/fit-compressed-feature-expert",
        "--train-feature-dir",
        str(paths["train_span_summary"]),
        "--validation-feature-dir",
        str(paths["validation_span_summary"]),
        "--output-train-dir",
        str(paths["train_raw117_scores"]),
        "--output-validation-dir",
        str(paths["validation_raw117_scores"]),
        "--output-report",
        str(paths["raw117_report"]),
        "--run-id",
        f"i1_present_r8_residual_focus_262k_seed{seed}_raw117",
        "--steps",
        "2000",
        "--learning-rate",
        "0.05",
        "--l2",
        "0.001",
    ]
    for prefix in RAW117_PREFIXES:
        parts.extend(["--include-feature-prefix", prefix])
    return _command(parts)


def _residual_command(
    paths: dict[str, Path],
    validation_trail_scores: Path,
    seed: int,
    *,
    label: str,
    focus_fraction: float,
    suffix: list[str] | None = None,
) -> str:
    parts = [
        "UV_CACHE_DIR=/tmp/uv-cache",
        "uv",
        "run",
        "scripts/fit-residual-correction-feature-expert",
        "--train-feature-dir",
        str(paths["train_span_summary"]),
        "--validation-feature-dir",
        str(paths["validation_span_summary"]),
        "--train-base-artifacts",
        str(paths["train_trail_scores"]),
        str(paths["train_raw117_scores"]),
        "--validation-base-artifacts",
        str(validation_trail_scores),
        str(paths["validation_raw117_scores"]),
        "--output-train-dir",
        str(paths[f"{label}_train_scores"]),
        "--output-validation-dir",
        str(paths[f"{label}_validation_scores"]),
        "--output-report",
        str(paths[f"{label}_report"]),
        "--run-id",
        f"i1_present_r8_residual_focus_262k_seed{seed}_{label}",
        "--bucket-count",
        "0",
        "--steps",
        "1000",
        "--learning-rate",
        "0.05",
        "--l2",
        "0.001",
        "--residual-focus-background-weight",
        "0.1",
        "--candidate-status",
        "residual_focus_262k_candidate",
    ]
    if focus_fraction > 0.0:
        parts.extend(["--residual-focus-fraction", _format_fraction(focus_fraction)])
    for prefix in RESIDUAL_PREFIXES:
        parts.extend(["--include-feature-prefix", prefix])
    if suffix:
        parts.extend(suffix)
    return _command(parts)


def _slice_eval_command(
    paths: dict[str, Path],
    validation_trail_scores: Path,
    *,
    corrected_key: str,
    focus_fraction: float,
) -> str:
    return _command(
        [
            "UV_CACHE_DIR=/tmp/uv-cache",
            "uv",
            "run",
            "scripts/evaluate-residual-slice-correction",
            "--train-base-artifacts",
            str(paths["train_trail_scores"]),
            str(paths["train_raw117_scores"]),
            "--validation-base-artifacts",
            str(validation_trail_scores),
            str(paths["validation_raw117_scores"]),
            "--validation-corrected-artifact",
            str(paths[f"{corrected_key}_validation_scores"]),
            "--focus-fraction",
            _format_fraction(focus_fraction),
            "--output",
            str(paths[f"{corrected_key}_slice_eval"]),
        ]
    )


def _format_fraction(value: float) -> str:
    return f"{value:g}"


def _source_gate_assessment(source_decision: str) -> str:
    if source_decision == READY_DECISION:
        return "trail_position_gate_support_all_runs"
    return "score_artifacts_ready_but_trail_position_gate_not_promoted"


def _claim_scope() -> str:
    return (
        "262144/class residual-focused correction action planning only; does not run "
        "training, does not SSH-poll, does not prove a breakthrough, and remains "
        "medium diagnostic SPN/PRESENT evidence until completed and compared."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = plan_residual_focus_262k(
        postprocess_status=args.postprocess_status,
        artifact_root=args.artifact_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
