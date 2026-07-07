from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_STATUS = Path("outputs/local_audits/i1_present_r8_residual_focus_status.json")
DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_state_token_residual_expert_plan.json")
ROUTE = "present_r8_state_token_residual_expert"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the next PRESENT r8 state-token residual expert without launching "
            "remote work. The planner is gated by the current residual-focus status."
        )
    )
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--control-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def plan_state_token_residual_expert(*, status_path: Path, control_summary_path: Path | None = None) -> dict[str, Any]:
    status = _load_json(status_path)
    control_summary = _load_json_or_empty(control_summary_path)
    gate_status = str(status.get("gate_status", status.get("status", "")))
    gate_decision = str(status.get("gate_decision", status.get("decision", "")))
    repair_hints = [str(hint) for hint in status.get("repair_hints", [])]

    base: dict[str, Any] = {
        "route": ROUTE,
        "source_status": str(status_path),
        "activation_gate": {
            "gate_status": gate_status,
            "gate_decision": gate_decision,
            "next_action_branch": _next_action_branch(status),
        },
        "state_token_control_summary": str(control_summary_path or ""),
        "state_token_control_status": str(control_summary.get("status", "missing" if not control_summary else "")),
        "state_token_control_decision": str(control_summary.get("decision", "")),
        "state_token_failing_seed_count": int(control_summary.get("failing_seed_count", 0)),
        "state_token_failing_control_event_count": int(control_summary.get("failing_control_event_count", 0)),
        "candidate": _candidate_summary(),
        "required_controls": _required_controls(),
        "forbidden_actions": [
            "launch_state_token_remote",
            "scale_state_token_to_1m",
            "claim_state_token_candidate",
            "change_labels_or_negative_mode",
        ],
        "should_launch_remote": False,
        "claim_scope": (
            "local architecture planning only; no SPN/PRESENT medium or formal claim "
            "until residual-focus artifacts are retrieved, controls pass, and a "
            "plan-aligned run is completed"
        ),
    }

    if gate_status in {"fail", "hold"} or gate_decision.startswith("hold_"):
        return {
            **base,
            "status": "hold",
            "decision": "repair_residual_focus_before_state_token_expert",
            "repair_hints": repair_hints,
            "allowed_local_actions": ["repair_residual_focus_source_or_objective"],
        }

    if gate_status == "pass":
        if control_summary.get("status") == "hold":
            return {
                **base,
                "status": "hold",
                "decision": "repair_state_token_controls_before_pool",
                "allowed_local_actions": ["repair_state_token_coordinate_or_value_only_controls"],
            }
        return {
            **base,
            "status": "ready",
            "decision": "state_token_residual_expert_local_plan_ready",
            "allowed_local_actions": [
                "write_state_token_model_smoke_test",
                "implement_local_state_token_smoke_only",
                "compare_against_same_input_controls",
            ],
            "next_local_checks": _next_local_checks(),
        }

    return {
        **base,
        "status": "pending",
        "decision": "wait_for_residual_focus_outputs_before_state_token_expert",
        "missing_output_count": int(status.get("missing_output_count", 0)),
        "allowed_local_actions": [
            "finalize_state_token_experiment_plan",
            "prepare_local_smoke_tests_only",
        ],
    }


def _candidate_summary() -> dict[str, Any]:
    return {
        "model_family": "state_token_residual_graph",
        "objective": "frozen_base_residual_correction",
        "base_scores": [
            "trail_position_anchor",
            "matched_raw117_compressed_spn_structural_expert",
        ],
        "primary_tokens": [
            "depth_word_cell_span",
            "depth_cell_span",
            "word_span",
            "depth_word_span",
            "cell_span",
        ],
        "token_coordinates": [
            "stat_family",
            "trail_depth",
            "trail_word",
            "cell_index",
        ],
        "design_intent": (
            "preserve SPN depth/word/cell coordinates and learn a small residual "
            "correction over hard or uncertain slices instead of averaging another "
            "near-neighbor classifier"
        ),
    }


def _required_controls() -> list[str]:
    return [
        "same_input_global_control",
        "uniform_residual_control",
        "label_shuffle_control",
        "token_coordinate_shuffle_control",
        "token_coordinate_drop_control",
        "train_only_selection_control",
    ]


def _next_local_checks() -> list[dict[str, Any]]:
    return [
        {
            "name": "state_token_forward_smoke",
            "purpose": "prove the tokenized span-stat model accepts the planned feature shape",
            "remote": False,
        },
        {
            "name": "local_2048class_residual_slice_screen",
            "purpose": "compare state-token correction against frozen trail+raw117 base and controls",
            "remote": False,
        },
        {
            "name": "same_protocol_control_gate",
            "purpose": "require control pass before any 262144/class or 1M/class promotion",
            "remote": False,
        },
    ]


def _next_action_branch(status: dict[str, Any]) -> str:
    next_action = status.get("next_action", {})
    if isinstance(next_action, dict):
        return str(next_action.get("branch", ""))
    return ""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _load_json_or_empty(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return _load_json(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = plan_state_token_residual_expert(
        status_path=args.status,
        control_summary_path=args.control_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
