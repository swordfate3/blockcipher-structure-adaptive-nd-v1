from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Markdown decision report from trail-position postprocess JSON."
    )
    parser.add_argument("--postprocess", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def render_trail_position_report(report: dict[str, Any]) -> str:
    lines = [
        "# PRESENT r8 Trail-Position Decision Report",
        "",
        "## Summary",
        "",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Decision: `{report.get('decision', 'unknown')}`",
        f"- Action: `{report.get('action', 'unknown')}`",
        f"- Ready runs: `{report.get('ready_run_count', 0)}`",
        f"- Pending runs: `{report.get('pending_run_count', 0)}`",
        f"- Failed runs: `{report.get('failed_run_count', 0)}`",
        f"- Expected score rows: `{report.get('expected_score_rows', 'unknown')}`",
        f"- Expected matrix rows: `{report.get('expected_matrix_rows', 'unknown')}`",
        "",
        "## Claim Scope",
        "",
        str(report.get("claim_scope", "No claim scope recorded.")),
        "",
        "## Runs",
        "",
        "| Run | Status | Train rows | Decision | AUC margin vs global | Missing score files |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for run in report.get("runs", []):
        analysis = run.get("analysis") if isinstance(run.get("analysis"), dict) else {}
        margins = (
            analysis.get("margins_vs_global_control", {})
            if isinstance(analysis.get("margins_vs_global_control"), dict)
            else {}
        )
        margin = _format_number(margins.get("auc"))
        decision = analysis.get("decision", run.get("reason", ""))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{run.get('run_id', 'unknown')}`",
                    f"`{run.get('status', 'unknown')}`",
                    f"`{run.get('train_rows', 'unknown')}`",
                    f"`{decision}`",
                    f"`{margin}`",
                    f"`{len(run.get('missing_score_files', []))}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- `pending` means no AUC or score-overlap claim is allowed yet.",
            "- `262144/class` is medium diagnostic evidence, not formal SPN/PRESENT evidence.",
            "- A passing report can justify a next scale gate, not a breakthrough claim.",
            "- A diverse ensemble claim still requires a separate non-neighbor expert and aligned score artifacts.",
        ]
    )
    if report.get("status") == "pass":
        lines.extend(
            [
                "",
                "## Next Gate",
                "",
                "- Update the experiment record with per-seed metrics and artifact paths.",
                "- Compare both seeds against the same-input global control and overlap gate.",
                "- Prepare a `>=1000000/class` multi-seed plan only if the medium diagnostic gate passes.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Next Gate",
                "",
                "- Keep watcher-managed retrieval active if the report is pending.",
                "- Inspect local verifier output before interpreting any failed score artifacts.",
                "- Do not start a new SPN/PRESENT branch while active 262k watchers are healthy.",
            ]
        )
    return "\n".join(lines) + "\n"


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = json.loads(args.postprocess.read_text(encoding="utf-8"))
    markdown = render_trail_position_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
