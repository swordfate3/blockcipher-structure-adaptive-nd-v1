from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_axis_spectrum_summary.json")
KNOWN_PREFIXES = (
    ("aux_depth_word", "aux_depth_word_"),
    ("aux_depth_cell", "aux_depth_cell_"),
    ("aux_word", "aux_word_"),
    ("aux_cell", "aux_cell_"),
    ("primary_depth_trailword", "primary_depth_trailword_"),
    ("primary_depth_cell", "primary_depth_cell_"),
)
PRIMARY_PREFIXES = ("primary_",)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize train-only residual bucket axis-spectrum reports into "
            "stable SPN feature-family source candidates."
        )
    )
    parser.add_argument("--spectrum-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--min-report-support", type=int, default=2)
    parser.add_argument("--top-groups", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def summarize_residual_axis_spectrum(
    *,
    spectrum_reports: list[Path],
    min_report_support: int = 2,
    top_groups: int = 8,
) -> dict[str, Any]:
    if not spectrum_reports:
        raise ValueError("at least one spectrum report is required")
    if min_report_support <= 0:
        raise ValueError("min_report_support must be positive")
    if top_groups <= 0:
        raise ValueError("top_groups must be positive")

    reports = [_load_train_report(path) for path in spectrum_reports]
    group_stats: dict[str, dict[str, Any]] = {}
    targets: list[str] = []
    for report_index, report in enumerate(reports):
        target = str(report.get("target", ""))
        if target and target not in targets:
            targets.append(target)
        seen_in_report: set[str] = set()
        for group in _global_groups(report):
            _update_group(group_stats, group, report_index=report_index, source="global", target=target)
            seen_in_report.add(str(group.get("group", "")))
        for group in _bucket_groups(report):
            _update_group(group_stats, group, report_index=report_index, source="bucket", target=target)
            seen_in_report.add(str(group.get("group", "")))
        for group_name in seen_in_report:
            if group_name:
                group_stats[group_name]["report_indices"].add(report_index)

    ranked = sorted(
        (_finalize_group(group, stats) for group, stats in group_stats.items()),
        key=_rank_key,
        reverse=True,
    )
    stable = [row for row in ranked if int(row["report_support_count"]) >= min_report_support]
    selected = stable[:top_groups]
    prefixes = _recommended_prefixes(selected)
    return {
        "status": "pass" if selected else "hold",
        "decision": (
            "residual_axis_spectrum_stable_groups_selected"
            if selected
            else "hold_residual_axis_spectrum_no_stable_groups"
        ),
        "spectrum_reports": [str(path) for path in spectrum_reports],
        "spectrum_count": int(len(reports)),
        "targets": targets,
        "min_report_support": int(min_report_support),
        "top_groups": int(top_groups),
        "selected_groups": [str(row["group"]) for row in selected],
        "recommended_feature_prefixes": prefixes,
        "stable_groups": selected,
        "all_groups": ranked,
        "claim_scope": (
            "train-only axis-spectrum source selection diagnostic; does not train, "
            "does not inspect held-out validation labels for structure selection, "
            "does not launch remote work, and does not prove SPN/PRESENT evidence"
        ),
    }


def _load_train_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    feature_dir = str(payload.get("feature_dir", ""))
    if "validation" in Path(feature_dir).parts or "validation" in Path(feature_dir).name:
        raise ValueError(f"{path}: expected train-only spectrum report, got feature_dir={feature_dir}")
    if "train" not in Path(feature_dir).parts and "train" not in Path(feature_dir).name:
        raise ValueError(f"{path}: expected train-only spectrum report, got feature_dir={feature_dir}")
    return payload


def _global_groups(report: dict[str, Any]) -> list[dict[str, Any]]:
    groups = report.get("global_top_groups", [])
    if not isinstance(groups, list):
        return []
    return [group for group in groups if isinstance(group, dict) and group.get("group")]


def _bucket_groups(report: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for bucket in report.get("bucket_reports", []):
        if not isinstance(bucket, dict):
            continue
        top_groups = bucket.get("top_groups", [])
        if isinstance(top_groups, list):
            groups.extend(group for group in top_groups if isinstance(group, dict) and group.get("group"))
    return groups


def _update_group(
    group_stats: dict[str, dict[str, Any]],
    group: dict[str, Any],
    *,
    report_index: int,
    source: str,
    target: str,
) -> None:
    name = str(group.get("group", ""))
    stats = group_stats.setdefault(
        name,
        {
            "report_indices": set(),
            "global_top_count": 0,
            "bucket_top_count": 0,
            "target_scores": [],
            "target_scores_by_target": {},
            "target_report_indices": {},
        },
    )
    if source == "global":
        stats["global_top_count"] += 1
    elif source == "bucket":
        stats["bucket_top_count"] += 1
    stats["report_indices"].add(report_index)
    score = group.get("target_score")
    if score is not None:
        score_float = float(score)
        stats["target_scores"].append(score_float)
        if target:
            stats["target_scores_by_target"].setdefault(target, []).append(score_float)
            stats["target_report_indices"].setdefault(target, set()).add(report_index)


def _finalize_group(group: str, stats: dict[str, Any]) -> dict[str, Any]:
    scores = [float(score) for score in stats["target_scores"]]
    target_summary = _target_summary(stats)
    preferred_target = _preferred_target(target_summary)
    return {
        "group": group,
        "feature_prefix": _feature_prefix(group),
        "residual_source_priority": "secondary_overlap" if _is_primary_group(group) else "candidate",
        "report_support_count": int(len(stats["report_indices"])),
        "global_top_count": int(stats["global_top_count"]),
        "bucket_top_count": int(stats["bucket_top_count"]),
        "mean_target_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "max_target_score": float(max(scores)) if scores else 0.0,
        "target_summary": target_summary,
        "preferred_target": preferred_target,
        "preferred_target_support_count": int(
            target_summary.get(preferred_target, {}).get("report_support_count", 0)
        ),
        "preferred_target_mean_score": float(
            target_summary.get(preferred_target, {}).get("mean_target_score", 0.0)
        ),
    }


def _rank_key(row: dict[str, Any]) -> tuple[int, int, float, int, int, str]:
    return (
        0 if row["residual_source_priority"] == "secondary_overlap" else 1,
        1 if row["preferred_target"] == "residual_loss" else 0,
        float(row["preferred_target_mean_score"]),
        int(row["preferred_target_support_count"]),
        int(row["global_top_count"]),
        str(row["group"]),
    )


def _target_summary(stats: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    scores_by_target = stats["target_scores_by_target"]
    report_indices_by_target = stats["target_report_indices"]
    for target, scores in scores_by_target.items():
        score_values = [float(score) for score in scores]
        summary[str(target)] = {
            "report_support_count": int(len(report_indices_by_target.get(target, set()))),
            "mean_target_score": float(sum(score_values) / len(score_values)) if score_values else 0.0,
            "max_target_score": float(max(score_values)) if score_values else 0.0,
        }
    return summary


def _preferred_target(target_summary: dict[str, dict[str, float | int]]) -> str:
    if "residual_loss" in target_summary:
        return "residual_loss"
    if not target_summary:
        return ""
    return max(
        target_summary,
        key=lambda target: (
            int(target_summary[target]["report_support_count"]),
            float(target_summary[target]["mean_target_score"]),
            target,
        ),
    )


def _recommended_prefixes(groups: list[dict[str, Any]]) -> list[str]:
    prefixes: list[str] = []
    for group in groups:
        prefix = str(group.get("feature_prefix", ""))
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


def _feature_prefix(group: str) -> str:
    for stem, prefix in KNOWN_PREFIXES:
        if group.startswith(stem):
            return prefix
    return ""


def _is_primary_group(group: str) -> bool:
    return group.startswith(PRIMARY_PREFIXES)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = summarize_residual_axis_spectrum(
        spectrum_reports=args.spectrum_reports,
        min_report_support=args.min_report_support,
        top_groups=args.top_groups,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
