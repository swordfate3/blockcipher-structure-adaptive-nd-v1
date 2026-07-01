from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = (
    "cipher_key",
    "model",
    "rounds",
    "seed",
    "samples_per_class",
    "pairs_per_sample",
    "feature_encoding",
    "negative_mode",
    "sample_structure",
    "validation_key",
)


@dataclass(frozen=True)
class QualityGateReport:
    status: str
    rows_compared: int
    max_auc_drop: float
    max_calibrated_accuracy_drop: float
    failures: list[str]

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_result_files(
    baseline_path: Path,
    candidate_path: Path,
    *,
    max_auc_drop: float = 0.002,
    max_calibrated_accuracy_drop: float = 0.002,
) -> QualityGateReport:
    baseline_rows = read_jsonl(baseline_path)
    candidate_rows = read_jsonl(candidate_path)
    failures: list[str] = []
    if len(baseline_rows) != len(candidate_rows):
        failures.append(
            f"row count mismatch: baseline={len(baseline_rows)} candidate={len(candidate_rows)}"
        )
    rows_compared = min(len(baseline_rows), len(candidate_rows))
    for index in range(rows_compared):
        baseline = baseline_rows[index]
        candidate = candidate_rows[index]
        baseline_identity = identity(baseline)
        candidate_identity = identity(candidate)
        if baseline_identity != candidate_identity:
            failures.append(
                f"row {index} identity mismatch: "
                f"baseline={baseline_identity} candidate={candidate_identity}"
            )
            continue
        auc_drop = metric(baseline, "auc") - metric(candidate, "auc")
        if auc_drop > max_auc_drop:
            failures.append(
                f"row {index} auc drop {auc_drop:.6f} exceeds {max_auc_drop:.6f}"
            )
        calibrated_drop = metric(baseline, "calibrated_accuracy") - metric(
            candidate,
            "calibrated_accuracy",
        )
        if calibrated_drop > max_calibrated_accuracy_drop:
            failures.append(
                f"row {index} calibrated_accuracy drop {calibrated_drop:.6f} "
                f"exceeds {max_calibrated_accuracy_drop:.6f}"
            )
    return QualityGateReport(
        status="passed" if not failures else "failed",
        rows_compared=rows_compared,
        max_auc_drop=max_auc_drop,
        max_calibrated_accuracy_drop=max_calibrated_accuracy_drop,
        failures=failures,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def identity(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in IDENTITY_FIELDS}


def metric(row: dict[str, Any], name: str) -> float:
    return float(row["metrics"][name])
