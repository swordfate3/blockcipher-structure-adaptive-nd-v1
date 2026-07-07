from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_state_token_residual_control_summary.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize local PRESENT r8 state-token residual correction controls. "
            "This reads existing JSON reports only and does not train or launch remote work."
        )
    )
    parser.add_argument("--candidate-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--drop-coordinate-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--coordinate-shuffle-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--label-shuffle-reports", nargs="+", required=True, type=Path)
    parser.add_argument("--min-base-delta", type=float, default=0.0)
    parser.add_argument("--min-control-delta", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def summarize_state_token_residual_controls(
    *,
    candidate_reports: list[Path],
    drop_coordinate_reports: list[Path],
    coordinate_shuffle_reports: list[Path],
    label_shuffle_reports: list[Path],
    min_base_delta: float = 0.0,
    min_control_delta: float = 0.0,
) -> dict[str, Any]:
    candidate = _load_report_group(candidate_reports, group_name="candidate")
    drop_coordinate = _load_report_group(drop_coordinate_reports, group_name="drop_coordinate")
    coordinate_shuffle = _load_report_group(coordinate_shuffle_reports, group_name="coordinate_shuffle")
    label_shuffle = _load_report_group(label_shuffle_reports, group_name="label_shuffle")
    seed_set = set(candidate) | set(drop_coordinate) | set(coordinate_shuffle) | set(label_shuffle)
    missing = _missing_group_rows(
        seeds=seed_set,
        groups={
            "candidate": candidate,
            "drop_coordinate": drop_coordinate,
            "coordinate_shuffle": coordinate_shuffle,
            "label_shuffle": label_shuffle,
        },
    )
    if missing:
        return {
            "status": "pending",
            "decision": "wait_for_state_token_control_reports",
            "seed_count": len(seed_set),
            "missing_report_rows": missing,
            "missing_report_row_count": len(missing),
            "next_action": {
                "branch": "retrieve_or_generate_state_token_control_reports",
                "should_launch_remote": False,
            },
            "claim_scope": _claim_scope(),
        }

    seed_reports = []
    for seed in sorted(seed_set):
        seed_report = _seed_report(
            seed=seed,
            candidate=candidate[seed],
            drop_coordinate=drop_coordinate[seed],
            coordinate_shuffle=coordinate_shuffle[seed],
            label_shuffle=label_shuffle[seed],
            min_base_delta=min_base_delta,
            min_control_delta=min_control_delta,
        )
        seed_reports.append(seed_report)

    failing = [row for row in seed_reports if not row["supported"]]
    failing_control_event_count = sum(len(row["failing_controls"]) for row in seed_reports)
    passed = not failing and bool(seed_reports)
    return {
        "status": "pass" if passed else "hold",
        "decision": (
            "state_token_residual_controls_pass"
            if passed
            else "hold_state_token_coordinate_controls"
        ),
        "seed_count": len(seed_reports),
        "seed_reports": seed_reports,
        "failing_seed_count": len(failing),
        "failing_control_event_count": failing_control_event_count,
        "min_base_delta": float(min_base_delta),
        "min_control_delta": float(min_control_delta),
        "next_action": {
            "branch": (
                "eligible_for_local_residual_pool_screen"
                if passed
                else "do_not_promote_state_token_coordinate_route"
            ),
            "should_launch_remote": False,
        },
        "claim_scope": _claim_scope(),
    }


def _seed_report(
    *,
    seed: int,
    candidate: dict[str, Any],
    drop_coordinate: dict[str, Any],
    coordinate_shuffle: dict[str, Any],
    label_shuffle: dict[str, Any],
    min_base_delta: float,
    min_control_delta: float,
) -> dict[str, Any]:
    candidate_auc = _validation_auc(candidate)
    base_auc = _base_auc(candidate)
    drop_auc = _validation_auc(drop_coordinate)
    coordshuffle_auc = _validation_auc(coordinate_shuffle)
    labelshuffle_auc = _validation_auc(label_shuffle)
    delta_vs_base = float(candidate_auc - base_auc)
    delta_vs_drop = float(candidate_auc - drop_auc)
    delta_vs_coordshuffle = float(candidate_auc - coordshuffle_auc)
    delta_vs_labelshuffle = float(candidate_auc - labelshuffle_auc)
    failing_controls: list[str] = []
    if delta_vs_base <= min_base_delta:
        failing_controls.append("base_logit_mean")
    if delta_vs_drop <= min_control_delta:
        failing_controls.append("drop_coordinate")
    if delta_vs_coordshuffle <= min_control_delta:
        failing_controls.append("coordinate_shuffle")
    if delta_vs_labelshuffle <= min_control_delta:
        failing_controls.append("label_shuffle")
    return {
        "seed": seed,
        "supported": not failing_controls,
        "failing_controls": failing_controls,
        "candidate_validation_auc": candidate_auc,
        "base_logit_mean_validation_auc": base_auc,
        "drop_coordinate_validation_auc": drop_auc,
        "coordinate_shuffle_validation_auc": coordshuffle_auc,
        "label_shuffle_validation_auc": labelshuffle_auc,
        "candidate_delta_vs_base_auc": delta_vs_base,
        "candidate_delta_vs_drop_coordinate_auc": delta_vs_drop,
        "candidate_delta_vs_coordinate_shuffle_auc": delta_vs_coordshuffle,
        "candidate_delta_vs_label_shuffle_auc": delta_vs_labelshuffle,
        "candidate_report": str(candidate["_path"]),
        "drop_coordinate_report": str(drop_coordinate["_path"]),
        "coordinate_shuffle_report": str(coordinate_shuffle["_path"]),
        "label_shuffle_report": str(label_shuffle["_path"]),
    }


def _load_report_group(paths: list[Path], *, group_name: str) -> dict[int, dict[str, Any]]:
    reports: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = _load_json(path)
        seed = _seed(payload, path=path)
        if seed in reports:
            raise ValueError(f"{group_name}: duplicate seed {seed}")
        payload["_path"] = str(path)
        reports[seed] = payload
    return reports


def _missing_group_rows(*, seeds: set[int], groups: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for seed in sorted(seeds):
        for group_name, reports in sorted(groups.items()):
            if seed not in reports:
                rows.append({"seed": seed, "group": group_name})
    return rows


def _seed(payload: dict[str, Any], *, path: Path) -> int:
    value = payload.get("seed")
    if value is None and isinstance(payload.get("fit"), dict):
        value = payload["fit"].get("seed")
    if value is None:
        raise ValueError(f"{path}: missing seed or fit.seed")
    return int(value)


def _validation_auc(payload: dict[str, Any]) -> float:
    return float(payload["validation_metrics"]["auc"])


def _base_auc(payload: dict[str, Any]) -> float:
    return float(payload["validation_base_logit_mean_metrics"]["auc"])


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _claim_scope() -> str:
    return (
        "local state-token residual control summary only; reads completed JSON reports, "
        "does not train, SSH, launch remote jobs, change labels or negative mode, or make "
        "a medium/formal SPN/PRESENT claim"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = summarize_state_token_residual_controls(
        candidate_reports=args.candidate_reports,
        drop_coordinate_reports=args.drop_coordinate_reports,
        coordinate_shuffle_reports=args.coordinate_shuffle_reports,
        label_shuffle_reports=args.label_shuffle_reports,
        min_base_delta=args.min_base_delta,
        min_control_delta=args.min_control_delta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
