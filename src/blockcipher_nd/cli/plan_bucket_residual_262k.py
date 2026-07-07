from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any


READY_DECISION = "support_trail_position_score_residual_all_runs"
DEFAULT_POSTPROCESS_STATUS = Path(
    "outputs/remote_results/i1_present_r8_trail_position_beamstats_262k_postprocess_status.json"
)
DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_bucket_residual_262k_action_plan.json")
DEFAULT_ARTIFACT_ROOT = Path("outputs/local_audits/i1_present_r8_bucket_residual_262k")
DEFAULT_SEED_PLAN_PREFIX = "configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_262k_seed"
FEATURE_PREFIXES = (
    "primary_depth_trailword_",
    "aux_depth_cell_",
    "aux_depth_word_",
    "aux_word_global_",
)
_GATE_INPUT_PATH_KEYS = (
    "bucket_report",
    "two_score_ensemble",
    "three_score_ensemble",
    "bucket_shuffle_report",
    "bucket_trainshuffle_report",
    "bucket_trainshuffle_ensemble",
    "bucket_valshuffle_report",
    "bucket_valshuffle_ensemble",
    "nobucket_report",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the 262144/class PRESENT r8 bucket-conditioned residual expert "
            "migration after trail-position score artifacts are retrieved."
        )
    )
    parser.add_argument("--postprocess-status", type=Path, default=DEFAULT_POSTPROCESS_STATUS)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def plan_bucket_residual_262k(
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
    if source_status != "pass" or source_decision != READY_DECISION:
        missing = _missing_from_postprocess(source)
        return _pending_report(
            postprocess_status=postprocess_status,
            artifact_root=artifact_root,
            reason="trail_position_262k_postprocess_not_ready",
            missing=missing,
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
            "decision": "hold_bucket_residual_262k_plan_inputs_missing",
            "should_run": False,
            "postprocess_status": str(postprocess_status),
            "source_status": source_status,
            "source_decision": source_decision,
            "artifact_root": str(artifact_root),
            "errors": errors,
            "claim_scope": _claim_scope(),
        }

    gate_output = artifact_root / "bucket_residual_controls_gate.json"
    gate_command = _gate_command(seed_plans, gate_output)
    return {
        "status": "pass",
        "decision": "bucket_residual_262k_action_plan_ready",
        "should_run": True,
        "postprocess_status": str(postprocess_status),
        "source_status": source_status,
        "source_decision": source_decision,
        "artifact_root": str(artifact_root),
        "expected_score_rows": int(source.get("expected_score_rows", 262144)),
        "required_inputs": [
            "train feature matrix for each seed",
            "validation feature matrix aligned to retrieved trail-position score artifact",
            "train trail-position score artifact exported from the retrieved checkpoint",
            "train raw117 score artifact fit from train compressed-span features",
            "validation trail-position score artifact retrieved by watcher",
            "validation raw117 score artifact scored on held-out validation features",
        ],
        "feature_prefixes": list(FEATURE_PREFIXES),
        "seeds": seed_plans,
        "commands": [command for seed in seed_plans for command in seed["commands"]],
        "control_commands": [command for seed in seed_plans for command in seed["control_commands"]],
        "gate_output": str(gate_output),
        "gate_command": gate_command,
        "next_action": (
            "Run these commands only after the 262k trail-position postprocess remains pass; "
            "then run the control gate before comparing trail+raw117 against trail+raw117+bucket "
            "at the same 262144/class scale."
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
    checkpoint_path = str(trail_models.get("checkpoint_path", ""))
    if not checkpoint_path:
        raise ValueError(f"{run_id}: trail_position checkpoint_path_missing")
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
        "bucket_train_scores": seed_root / "bucket_raw117_logitgap_train_scores",
        "bucket_validation_scores": seed_root / "bucket_raw117_logitgap_validation_scores",
        "bucket_report": seed_root / "bucket_raw117_logitgap_report.json",
        "two_score_ensemble": seed_root / "trail_raw117_two_score_ensemble.json",
        "three_score_ensemble": seed_root / "trail_raw117_bucket_three_score_ensemble.json",
        "bucket_shuffle_train_scores": seed_root / "bucket_raw117_logitgap_shuffle_labels_train_scores",
        "bucket_shuffle_validation_scores": seed_root / "bucket_raw117_logitgap_shuffle_labels_validation_scores",
        "nobucket_train_scores": seed_root / "raw117_nobucket_l2_0p0003_train_scores",
        "nobucket_validation_scores": seed_root / "raw117_nobucket_l2_0p0003_validation_scores",
        "nobucket_report": seed_root / "raw117_nobucket_l2_0p0003_report.json",
        "bucket_trainshuffle_train_scores": seed_root / "bucket_raw117_logitgap_trainbucket_shuffle_train_scores",
        "bucket_trainshuffle_validation_scores": seed_root / (
            "bucket_raw117_logitgap_trainbucket_shuffle_validation_scores"
        ),
        "bucket_valshuffle_train_scores": seed_root / "bucket_raw117_logitgap_valbucket_shuffle_train_scores",
        "bucket_valshuffle_validation_scores": seed_root / "bucket_raw117_logitgap_valbucket_shuffle_validation_scores",
        "bucket_shuffle_report": seed_root / "bucket_raw117_logitgap_shuffle_labels_report.json",
        "bucket_trainshuffle_report": seed_root / "bucket_raw117_logitgap_trainbucket_shuffle_report.json",
        "bucket_valshuffle_report": seed_root / "bucket_raw117_logitgap_valbucket_shuffle_report.json",
        "bucket_trainshuffle_ensemble": seed_root / "trail_raw117_bucket_trainshuffle_three_score_ensemble.json",
        "bucket_valshuffle_ensemble": seed_root / "trail_raw117_bucket_valshuffle_three_score_ensemble.json",
    }
    commands = [
        _feature_export_command(eval_plan, seed, "train", paths["train_feature_dir"], None, paths),
        _feature_export_command(eval_plan, seed, "validation", paths["validation_feature_dir"], validation_trail_scores, paths),
        _span_command(paths["train_feature_dir"], paths["train_span_blocks"], paths["train_span_summary"]),
        _span_command(paths["validation_feature_dir"], paths["validation_span_blocks"], paths["validation_span_summary"]),
        _checkpoint_score_command(
            checkpoint_path=checkpoint_path,
            eval_plan=eval_plan,
            seed=seed,
            output_dir=paths["train_trail_scores"],
            cache_root=paths["dataset_cache_root"],
        ),
        _raw117_command(paths, seed),
        _bucket_command(paths, validation_trail_scores, seed),
        _ensemble_command([validation_trail_scores, paths["validation_raw117_scores"]], paths["two_score_ensemble"]),
        _ensemble_command(
            [validation_trail_scores, paths["validation_raw117_scores"], paths["bucket_validation_scores"]],
            paths["three_score_ensemble"],
        ),
    ]
    control_commands = [
        _bucket_command(
            paths,
            validation_trail_scores,
            seed,
            suffix=["--shuffle-train-labels", "--shuffle-seed", str(9300 + seed)],
            output_train_dir=paths["bucket_shuffle_train_scores"],
            output_validation_dir=paths["bucket_shuffle_validation_scores"],
            output_report=paths["bucket_shuffle_report"],
        ),
        _bucket_command(
            paths,
            validation_trail_scores,
            seed,
            suffix=["--shuffle-train-bucket-values", "--shuffle-seed", str(9400 + seed)],
            output_train_dir=paths["bucket_trainshuffle_train_scores"],
            output_validation_dir=paths["bucket_trainshuffle_validation_scores"],
            output_report=paths["bucket_trainshuffle_report"],
        ),
        _ensemble_command(
            [
                validation_trail_scores,
                paths["validation_raw117_scores"],
                paths["bucket_trainshuffle_validation_scores"],
            ],
            paths["bucket_trainshuffle_ensemble"],
        ),
        _bucket_command(
            paths,
            validation_trail_scores,
            seed,
            suffix=["--shuffle-validation-bucket-values", "--shuffle-seed", str(9500 + seed)],
            output_train_dir=paths["bucket_valshuffle_train_scores"],
            output_validation_dir=paths["bucket_valshuffle_validation_scores"],
            output_report=paths["bucket_valshuffle_report"],
        ),
        _ensemble_command(
            [validation_trail_scores, paths["validation_raw117_scores"], paths["bucket_valshuffle_validation_scores"]],
            paths["bucket_valshuffle_ensemble"],
        ),
        _nobucket_command(paths, seed),
    ]
    return {
        "seed": seed,
        "run_id": run_id,
        "run_root": str(run_root),
        "eval_plan": str(eval_plan),
        "eval_row_index": 1,
        "validation_trail_position_scores": str(validation_trail_scores),
        "train_trail_position_checkpoint": checkpoint_path,
        "remote_checkpoint_reference": remote_checkpoint_reference,
        "warnings": _seed_warnings(remote_checkpoint_reference),
        "artifact_root": str(seed_root),
        "commands": commands,
        "control_commands": control_commands,
        "gate_inputs": {name: str(paths[name]) for name in _GATE_INPUT_PATH_KEYS},
    }


def _read_models_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"{path}: missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _seed_from_run_id(run_id: str) -> int:
    match = re.search(r"seed(\d+)", run_id)
    if not match:
        raise ValueError(f"{run_id}: seed_not_found")
    return int(match.group(1))


def _is_windows_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:\\", value))


def _seed_warnings(remote_checkpoint_reference: bool) -> list[str]:
    if not remote_checkpoint_reference:
        return []
    return [
        (
            "train trail-position score export references a remote Windows checkpoint path; "
            "run that export where the checkpoint exists or retrieve the checkpoint/score artifact "
            "before executing downstream local bucket commands"
        )
    ]


def _feature_export_command(
    eval_plan: Path,
    seed: int,
    split: str,
    output_dir: Path,
    reference_artifact: Path | None,
    paths: dict[str, Path],
) -> str:
    parts = [
        "UV_CACHE_DIR=/tmp/uv-cache",
        "uv",
        "run",
        "scripts/export-bit-sensitivity-features",
        "--eval-plan",
        str(eval_plan),
        "--eval-row-index",
        "1",
        "--split",
        split,
        "--feature-view",
        "trail_position_stats",
        "--dataset-cache-root",
        str(paths["dataset_cache_root"] / split),
        "--progress-output",
        str(paths["dataset_cache_root"] / f"seed{seed}_{split}_feature_export_progress.jsonl"),
    ]
    if reference_artifact is not None:
        parts.extend(["--reference-artifact", str(reference_artifact)])
    parts.extend(["--output-dir", str(output_dir)])
    return _command(parts)


def _span_command(feature_dir: Path, output_dir: Path, summary_dir: Path) -> str:
    return _command(
        [
            "UV_CACHE_DIR=/tmp/uv-cache",
            "uv",
            "run",
            "scripts/export-compressed-span-blocks",
            "--feature-dir",
            str(feature_dir),
            "--output-dir",
            str(output_dir),
            "--output-summary-feature-dir",
            str(summary_dir),
        ]
    )


def _checkpoint_score_command(
    *,
    checkpoint_path: str,
    eval_plan: Path,
    seed: int,
    output_dir: Path,
    cache_root: Path,
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
            str(cache_root / "train_scores"),
            "--progress-output",
            str(cache_root / f"seed{seed}_train_score_export_progress.jsonl"),
            "--output-dir",
            str(output_dir),
        ]
    )


def _raw117_command(paths: dict[str, Path], seed: int) -> str:
    return _compressed_feature_command(
        train_feature_dir=paths["train_span_summary"],
        validation_feature_dir=paths["validation_span_summary"],
        output_train_dir=paths["train_raw117_scores"],
        output_validation_dir=paths["validation_raw117_scores"],
        output_report=paths["raw117_report"],
        run_id=f"i1_present_r8_bucket_residual_262k_seed{seed}_raw117",
        steps=2000,
        l2=0.001,
    )


def _nobucket_command(paths: dict[str, Path], seed: int) -> str:
    return _compressed_feature_command(
        train_feature_dir=paths["train_span_summary"],
        validation_feature_dir=paths["validation_span_summary"],
        output_train_dir=paths["nobucket_train_scores"],
        output_validation_dir=paths["nobucket_validation_scores"],
        output_report=paths["nobucket_report"],
        run_id=f"i1_present_r8_bucket_residual_262k_seed{seed}_nobucket_l2_0p0003",
        steps=1000,
        l2=0.0003,
    )


def _compressed_feature_command(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    output_train_dir: Path,
    output_validation_dir: Path,
    output_report: Path,
    run_id: str,
    steps: int,
    l2: float,
) -> str:
    parts = [
        "UV_CACHE_DIR=/tmp/uv-cache",
        "uv",
        "run",
        "scripts/fit-compressed-feature-expert",
        "--train-feature-dir",
        str(train_feature_dir),
        "--validation-feature-dir",
        str(validation_feature_dir),
        "--output-train-dir",
        str(output_train_dir),
        "--output-validation-dir",
        str(output_validation_dir),
        "--output-report",
        str(output_report),
        "--run-id",
        run_id,
        "--steps",
        str(steps),
        "--learning-rate",
        "0.05",
        "--l2",
        str(l2),
    ]
    for prefix in FEATURE_PREFIXES:
        parts.extend(["--include-feature-prefix", prefix])
    return _command(parts)


def _bucket_command(
    paths: dict[str, Path],
    validation_trail_scores: Path,
    seed: int,
    *,
    suffix: list[str] | None = None,
    output_train_dir: Path | None = None,
    output_validation_dir: Path | None = None,
    output_report: Path | None = None,
) -> str:
    parts = [
        "UV_CACHE_DIR=/tmp/uv-cache",
        "uv",
        "run",
        "scripts/fit-bucket-conditioned-feature-expert",
        "--train-feature-dir",
        str(paths["train_span_summary"]),
        "--validation-feature-dir",
        str(paths["validation_span_summary"]),
        "--train-bucket-artifacts",
        str(paths["train_trail_scores"]),
        str(paths["train_raw117_scores"]),
        "--validation-bucket-artifacts",
        str(validation_trail_scores),
        str(paths["validation_raw117_scores"]),
        "--output-train-dir",
        str(output_train_dir or paths["bucket_train_scores"]),
        "--output-validation-dir",
        str(output_validation_dir or paths["bucket_validation_scores"]),
        "--output-report",
        str(output_report or paths["bucket_report"]),
        "--run-id",
        f"i1_present_r8_bucket_residual_262k_seed{seed}",
        "--bucket-feature",
        "logit_gap_abs",
        "--bucket-count",
        "5",
        "--steps",
        "1000",
        "--learning-rate",
        "0.05",
        "--l2",
        "0.0003",
    ]
    for prefix in FEATURE_PREFIXES:
        parts.extend(["--include-feature-prefix", prefix])
    if suffix:
        parts.extend(suffix)
    return _command(parts)


def _ensemble_command(artifacts: list[Path], output: Path) -> str:
    return _command(
        [
            "UV_CACHE_DIR=/tmp/uv-cache",
            "uv",
            "run",
            "scripts/evaluate-neural-ensemble",
            "--artifacts",
            *(str(path) for path in artifacts),
            "--output",
            str(output),
        ]
    )


def _gate_command(seed_plans: list[dict[str, Any]], output: Path) -> str:
    parts = [
        "UV_CACHE_DIR=/tmp/uv-cache",
        "uv",
        "run",
        "scripts/gate-bucket-residual-controls",
    ]
    flag_by_key = {
        "bucket_report": "--candidate-report",
        "two_score_ensemble": "--two-score-ensemble",
        "three_score_ensemble": "--three-score-ensemble",
        "bucket_shuffle_report": "--shuffle-label-report",
        "bucket_trainshuffle_report": "--train-bucket-shuffle-report",
        "bucket_trainshuffle_ensemble": "--train-bucket-shuffle-ensemble",
        "bucket_valshuffle_report": "--validation-bucket-shuffle-report",
        "bucket_valshuffle_ensemble": "--validation-bucket-shuffle-ensemble",
        "nobucket_report": "--no-bucket-report",
    }
    for seed in seed_plans:
        gate_inputs = seed["gate_inputs"]
        for key, flag in flag_by_key.items():
            parts.extend([flag, str(gate_inputs[key])])
    parts.extend(["--output", str(output)])
    return _command(parts)


def _command(parts: list[str]) -> str:
    return " ".join(_quote_part(part) for part in parts)


def _quote_part(part: str) -> str:
    if part == "UV_CACHE_DIR=/tmp/uv-cache":
        return part
    return shlex.quote(str(part))


def _claim_scope() -> str:
    return (
        "262144/class V16 bucket-conditioned residual action planning only; "
        "does not run training, does not SSH-poll, does not prove a breakthrough, "
        "and remains medium diagnostic SPN/PRESENT evidence until completed and compared."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = plan_bucket_residual_262k(
        postprocess_status=args.postprocess_status,
        artifact_root=args.artifact_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
