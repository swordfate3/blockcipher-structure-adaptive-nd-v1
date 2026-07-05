from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize local neural ensemble run artifacts.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def neural_ensemble_status(run_root: Path, *, expected_rows: int = 3) -> dict[str, Any]:
    train_results = run_root / "results" / "train_matrix.jsonl"
    ensemble_summary = run_root / "results" / "neural_ensemble_summary.json"
    progress = run_root / "logs" / "train_matrix_progress.jsonl"
    checkpoint_dir = run_root / "checkpoints"
    score_artifacts_dir = run_root / "score_artifacts"

    train_rows = _read_jsonl(train_results) if train_results.exists() else []
    checkpoints = sorted(path.name for path in checkpoint_dir.glob("row*.pt")) if checkpoint_dir.exists() else []
    score_artifacts = _score_artifact_status(score_artifacts_dir)
    latest_progress = _latest_jsonl(progress)
    missing_artifacts = _missing_artifacts(
        expected_rows=expected_rows,
        train_rows=len(train_rows),
        checkpoint_count=len(checkpoints),
        score_artifacts=score_artifacts,
        ensemble_summary_ready=ensemble_summary.exists(),
    )
    failed = bool(list((run_root / "logs").glob("*failed.marker"))) if (run_root / "logs").exists() else False

    if failed:
        status = "failed"
    elif not missing_artifacts:
        status = "ready"
    else:
        status = "running"

    return {
        "status": status,
        "run_root": str(run_root),
        "expected_rows": expected_rows,
        "train_rows": len(train_rows),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "score_artifacts": score_artifacts,
        "score_artifacts_ready": all(score_artifacts.values()) and bool(score_artifacts),
        "ensemble_summary_ready": ensemble_summary.exists(),
        "latest_progress": latest_progress,
        "missing_artifacts": missing_artifacts,
    }


def _score_artifact_status(score_artifacts_dir: Path) -> dict[str, bool]:
    names = ["zhang_wang", "invp_only", "ddt_graph"]
    return {
        name: (score_artifacts_dir / name / "models.json").exists()
        for name in names
    }


def _missing_artifacts(
    *,
    expected_rows: int,
    train_rows: int,
    checkpoint_count: int,
    score_artifacts: dict[str, bool],
    ensemble_summary_ready: bool,
) -> list[str]:
    missing: list[str] = []
    if train_rows < expected_rows:
        missing.append(f"train_matrix rows {train_rows}/{expected_rows}")
    if checkpoint_count < expected_rows:
        missing.append(f"checkpoints {checkpoint_count}/{expected_rows}")
    for name, ready in score_artifacts.items():
        if not ready:
            missing.append(f"score_artifacts/{name}/models.json")
    if not ensemble_summary_ready:
        missing.append("neural_ensemble_summary.json")
    return missing


def _latest_jsonl(path: Path) -> dict[str, Any] | None:
    rows = _read_jsonl(path) if path.exists() else []
    return rows[-1] if rows else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = neural_ensemble_status(args.run_root, expected_rows=args.expected_rows)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
