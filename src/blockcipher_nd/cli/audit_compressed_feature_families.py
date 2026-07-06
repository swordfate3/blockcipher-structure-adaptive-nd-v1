from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.fit_compressed_feature_expert import fit_compressed_feature_expert


DEFAULT_SPAN_FAMILIES = [
    "depth_word_cell_span",
    "depth_cell_span",
    "word_span",
    "depth_word_span",
    "cell_span",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit compressed SPN feature families with union, single-family, and leave-one-out fits."
    )
    parser.add_argument("--train-feature-dir", required=True, type=Path)
    parser.add_argument("--validation-feature-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--min-positive-auc", type=float, default=0.99)
    return parser.parse_args(argv)


def audit_compressed_feature_families(
    *,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    families: list[str],
    steps: int = 800,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    standardize: bool = True,
    min_positive_auc: float = 0.99,
) -> dict[str, Any]:
    family_list = _unique_nonempty(families) or list(DEFAULT_SPAN_FAMILIES)
    if min_positive_auc < 0.0 or min_positive_auc > 1.0:
        raise ValueError("min_positive_auc must be in [0, 1]")

    union_row = _fit_row(
        mode="union",
        train_feature_dir=train_feature_dir,
        validation_feature_dir=validation_feature_dir,
        include_families=family_list,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
    )
    rows = [union_row]
    for family in family_list:
        rows.append(
            _fit_row(
                mode="single_family",
                train_feature_dir=train_feature_dir,
                validation_feature_dir=validation_feature_dir,
                include_families=[family],
                steps=steps,
                learning_rate=learning_rate,
                l2=l2,
                standardize=standardize,
                family=family,
            )
        )
    if len(family_list) > 1:
        for family in family_list:
            rows.append(
                _fit_row(
                    mode="leave_one_out",
                    train_feature_dir=train_feature_dir,
                    validation_feature_dir=validation_feature_dir,
                    include_families=[candidate for candidate in family_list if candidate != family],
                    steps=steps,
                    learning_rate=learning_rate,
                    l2=l2,
                    standardize=standardize,
                    left_out_family=family,
                )
            )

    decision = (
        "span_family_attribution_local_positive"
        if float(union_row["validation_metrics"]["auc"]) >= min_positive_auc
        else "span_family_attribution_hold"
    )
    return {
        "status": "pass",
        "decision": decision,
        "families": family_list,
        "train_feature_dir": str(train_feature_dir),
        "validation_feature_dir": str(validation_feature_dir),
        "fit": {
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "standardize": bool(standardize),
        },
        "thresholds": {"min_positive_auc": float(min_positive_auc)},
        "union_row": union_row,
        "rows": rows,
        "claim_scope": (
            "compressed SPN feature-family attribution diagnostic only; each fit uses train labels only, "
            "scores held-out validation features, and is not remote or formal SPN/PRESENT evidence"
        ),
    }


def _fit_row(
    *,
    mode: str,
    train_feature_dir: Path,
    validation_feature_dir: Path,
    include_families: list[str],
    steps: int,
    learning_rate: float,
    l2: float,
    standardize: bool,
    family: str | None = None,
    left_out_family: str | None = None,
) -> dict[str, Any]:
    _, _, report = fit_compressed_feature_expert(
        train_feature_dir=train_feature_dir,
        validation_feature_dir=validation_feature_dir,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        standardize=standardize,
        model_key=f"compressed_{mode}_family_logistic_expert",
        expert_family="compressed_spn_structural_family_attribution",
        candidate_status=f"{mode}_family_attribution",
        include_feature_families=include_families,
    )
    row = {
        "mode": mode,
        "include_feature_families": report["feature_selection"]["include_feature_families"],
        "feature_count": int(report["feature_count"]),
        "original_feature_count": int(report["feature_selection"]["original_feature_count"]),
        "train_metrics": report["train_metrics"],
        "validation_metrics": report["validation_metrics"],
    }
    if family is not None:
        row["family"] = family
    if left_out_family is not None:
        row["left_out_family"] = left_out_family
    return row


def _unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_compressed_feature_families(
        train_feature_dir=args.train_feature_dir,
        validation_feature_dir=args.validation_feature_dir,
        families=args.family,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        standardize=not args.no_standardize,
        min_positive_auc=args.min_positive_auc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
