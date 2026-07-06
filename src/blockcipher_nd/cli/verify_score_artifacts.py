from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    load_score_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify aligned frozen neural score artifacts before ensemble analysis."
    )
    parser.add_argument("--artifacts", nargs="+", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument(
        "--require-model",
        action="append",
        default=[],
        metavar="MODEL_KEY:EXPERT_FAMILY:CANDIDATE_STATUS",
        help="Required model metadata tuple. May be repeated.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def verify_score_artifacts(
    artifact_dirs: list[Path],
    *,
    expected_rows: int,
    required_models: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    artifacts: list[EnsembleScoreArtifact] = []
    models: list[dict[str, Any]] = []

    for artifact_dir in artifact_dirs:
        try:
            artifact = load_score_artifact(artifact_dir)
        except Exception as exc:  # noqa: BLE001 - verifier reports artifact load failures.
            errors.append(f"load_failed:{artifact_dir}:{type(exc).__name__}:{exc}")
            continue
        artifacts.append(artifact)
        models.append(_model_summary(artifact_dir, artifact))
        _validate_lengths(artifact_dir, artifact, errors)

    rows = len(artifacts[0].labels) if artifacts else 0
    if rows != expected_rows:
        errors.append(f"row_count_mismatch:{rows}!={expected_rows}")

    labels_aligned = _arrays_aligned([artifact.labels for artifact in artifacts])
    sample_ids_aligned = _arrays_aligned([artifact.sample_ids for artifact in artifacts])
    if artifacts and not labels_aligned:
        errors.append("labels_mismatch")
    if artifacts and not sample_ids_aligned:
        errors.append("sample_ids_mismatch")

    required = [_parse_required_model(value, errors) for value in required_models]
    required = [item for item in required if item is not None]
    present = {
        (
            str(artifact.metadata.get("model_key", "")),
            str(artifact.metadata.get("expert_family", "")),
            str(artifact.metadata.get("candidate_status", "")),
        )
        for artifact in artifacts
    }
    for model_key, expert_family, candidate_status in required:
        if (model_key, expert_family, candidate_status) not in present:
            errors.append(f"missing_required_model:{model_key}:{expert_family}:{candidate_status}")

    return {
        "status": "pass" if not errors else "fail",
        "artifact_count": len(artifacts),
        "artifact_dirs": [str(path) for path in artifact_dirs],
        "rows": rows,
        "expected_rows": expected_rows,
        "alignment": {
            "labels": labels_aligned,
            "sample_ids": sample_ids_aligned,
        },
        "models": models,
        "required_models": [
            {
                "model_key": model_key,
                "expert_family": expert_family,
                "candidate_status": candidate_status,
            }
            for model_key, expert_family, candidate_status in required
        ],
        "errors": errors,
    }


def _validate_lengths(
    artifact_dir: Path,
    artifact: EnsembleScoreArtifact,
    errors: list[str],
) -> None:
    lengths = {
        "labels": len(artifact.labels),
        "probabilities": len(artifact.probabilities),
        "logits": len(artifact.logits),
        "sample_ids": len(artifact.sample_ids),
    }
    if len(set(lengths.values())) != 1:
        errors.append(f"array_length_mismatch:{artifact_dir}:{lengths}")


def _arrays_aligned(arrays: list[np.ndarray]) -> bool:
    if len(arrays) < 2:
        return True
    first = arrays[0]
    return all(np.array_equal(first, array) for array in arrays[1:])


def _parse_required_model(value: str, errors: list[str]) -> tuple[str, str, str] | None:
    parts = value.split(":")
    if len(parts) != 3 or not all(part.strip() for part in parts):
        errors.append(f"invalid_required_model:{value}")
        return None
    return (parts[0].strip(), parts[1].strip(), parts[2].strip())


def _model_summary(artifact_dir: Path, artifact: EnsembleScoreArtifact) -> dict[str, Any]:
    metadata = artifact.metadata
    return {
        "artifact_dir": str(artifact_dir),
        "model_key": str(metadata.get("model_key", "")),
        "expert_family": str(metadata.get("expert_family", "")),
        "candidate_status": str(metadata.get("candidate_status", "")),
        "rows": len(artifact.labels),
        "metadata": {
            key: metadata.get(key)
            for key in (
                "cipher",
                "cipher_key",
                "rounds",
                "negative_mode",
                "sample_structure",
                "feature_encoding",
                "validation_samples_per_class",
                "pairs_per_sample",
            )
            if key in metadata
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = verify_score_artifacts(
        args.artifacts,
        expected_rows=args.expected_rows,
        required_models=args.require_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
