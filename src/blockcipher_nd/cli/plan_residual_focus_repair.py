from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_gate.json")
DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_repair_plan.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan local-only residual-focus repair branches from a failed PRESENT r8 "
            "262144/class residual-focus gate or Pool 3 hold plan."
        )
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def plan_residual_focus_repair(*, summary: Path) -> dict[str, Any]:
    payload = _load_json(summary)
    status = str(payload.get("status", ""))
    decision = str(payload.get("decision", ""))
    hints = [str(hint) for hint in payload.get("repair_hints", [])]

    if status == "pending":
        return {
            "status": "pending",
            "decision": "wait_for_residual_focus_gate_before_repair_plan",
            "source_summary": str(summary),
            "source_decision": decision,
            "repair_hints": hints,
            "repair_branches": [],
            "missing_outputs": [str(path) for path in payload.get("missing_outputs", [])],
            "should_launch_remote": False,
            "claim_scope": "local repair planning guard only; residual-focus gate is not complete",
        }
    if status not in {"fail", "hold"} and not hints:
        return {
            "status": "not_needed",
            "decision": "residual_focus_repair_not_required",
            "source_summary": str(summary),
            "source_decision": decision,
            "repair_hints": [],
            "repair_branches": [],
            "should_launch_remote": False,
            "claim_scope": "local repair planning guard only; no failed residual-focus repair hint was present",
        }

    branches = _repair_branches(hints, decision=decision)
    return {
        "status": "ready",
        "decision": "repair_residual_focus_before_pool_or_scaleup",
        "source_summary": str(summary),
        "source_decision": decision,
        "repair_hints": hints,
        "errors": [str(error) for error in payload.get("errors", [])],
        "primary_repair_branch": branches[0]["branch"] if branches else "inspect_residual_focus_gate_errors",
        "repair_branches": branches,
        "forbidden_actions": [
            "launch_residual_guided_pool3",
            "scale_residual_focus_to_1m",
            "claim_residual_focus_candidate",
        ],
        "should_launch_remote": False,
        "claim_scope": (
            "local repair planning guard only; failed/held residual-focus evidence must be "
            "repaired before Pool 3, 1M/class scale-up, or any SPN/PRESENT claim"
        ),
    }


def _repair_branches(hints: list[str], *, decision: str = "") -> list[dict[str, Any]]:
    branch_by_hint = {
        "candidate_not_better_than_uniform_control": {
            "branch": "separate_focus_from_uniform_residual_objective",
            "question": (
                "Can a residual-focused objective beat the uniform residual control "
                "when the comparison is made on the same hard slice?"
            ),
            "next_local_check": "fit residual correction with focus-vs-uniform attribution controls",
            "success_gate": "candidate hard-slice residual-loss drop is strictly better than uniform across seeds",
        },
        "label_shuffle_control_failed": {
            "branch": "repair_label_shuffle_attribution_control",
            "question": "Is the label-shuffle artifact/control generation invalidating the residual signal?",
            "next_local_check": "audit label-shuffle score artifacts and rerun shuffle controls before promotion",
            "success_gate": "label-shuffle correction worsens focus residual loss while true labels improve it",
        },
        "focus_candidate_metric_failed": {
            "branch": "rescan_residual_feature_family",
            "question": "Is aux_depth_word_ + aux_word_ the wrong residual source at this scale?",
            "next_local_check": "rerun axis-spectrum source selection on retrieved train artifacts",
            "success_gate": "new frozen feature family passes train-derived hard-slice controls",
        },
    }
    branches = [branch_by_hint[hint] for hint in hints if hint in branch_by_hint]
    if branches:
        return branches
    if decision == "residual_guided_pool3_fixed_fusion_mixed_or_controlled":
        return [
            {
                "branch": "repair_residual_guided_pool3_controls",
                "question": "Does the residual-focus expert add value after fixed fusion controls?",
                "next_local_check": "inspect Pool 3 seed reports and compare against uniform/label-shuffle fusions",
                "success_gate": "residual-focus fusion strictly beats trail+raw117 and both residual controls",
            }
        ]
    return [
        {
            "branch": "inspect_residual_focus_gate_errors",
            "question": "Which residual-focus invariant failed?",
            "next_local_check": "inspect gate errors and add a dedicated repair hint before launching more work",
            "success_gate": "a specific repair branch is selected and documented",
        }
    ]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = plan_residual_focus_repair(summary=args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
