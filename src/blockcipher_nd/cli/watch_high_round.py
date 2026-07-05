from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.advance_high_round import advance_high_round


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch local high-round SPN/PRESENT result artifacts and advance "
            "postprocess/arbitration when ready. This never SSH-polls or launches remote training."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/remote_results"),
        help="Local remote-result artifact root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/remote_results/high_round_auto_advance_report.json"),
        help="Where to write the latest watcher report.",
    )
    parser.add_argument(
        "--arbitration-output",
        type=Path,
        default=Path("outputs/remote_results/high_round_next_action_arbitration.json"),
        help="Where to write arbitration when enough high-round summaries exist.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=840.0,
        help="Delay between local artifact checks while the status is waiting.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional maximum iterations; useful for smoke tests and one-shot tmux checks.",
    )
    parser.add_argument(
        "--no-update-plan-doc",
        action="store_true",
        help="Do not update docs/experiments during postprocess. Intended for dry runs.",
    )
    return parser.parse_args(argv)


def watch_high_round(
    *,
    root: Path = Path("outputs/remote_results"),
    output: Path = Path("outputs/remote_results/high_round_auto_advance_report.json"),
    arbitration_output: Path = Path("outputs/remote_results/high_round_next_action_arbitration.json"),
    interval_seconds: float = 840.0,
    max_iterations: int | None = None,
    update_plan_docs: bool = True,
) -> dict[str, Any]:
    iteration = 0
    last_report: dict[str, Any] | None = None
    while True:
        iteration += 1
        report = advance_high_round(
            root=root,
            arbitration_output=arbitration_output,
            update_plan_docs=update_plan_docs,
        )
        report = {
            **report,
            "watch_iteration": iteration,
            "watch_interval_seconds": interval_seconds,
            "watch_max_iterations": max_iterations,
            "watch_policy": "local_artifacts_only_no_ssh_no_remote_launch",
        }
        _write_report(output, report)
        last_report = report

        status = str(report.get("status") or "")
        if status != "waiting":
            return report
        if max_iterations is not None and iteration >= max_iterations:
            return report
        time.sleep(max(0.0, interval_seconds))


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = watch_high_round(
        root=args.root,
        output=args.output,
        arbitration_output=args.arbitration_output,
        interval_seconds=args.interval_seconds,
        max_iterations=args.max_iterations,
        update_plan_docs=not args.no_update_plan_doc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
