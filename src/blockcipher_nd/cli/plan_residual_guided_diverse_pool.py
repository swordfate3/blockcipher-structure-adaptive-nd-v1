from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESIDUAL_FOCUS_GATE = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_gate.json")
DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_plan.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the PRESENT r8 residual-guided diverse expert pool after the "
            "262144/class residual-focus hard-slice gate completes."
        )
    )
    parser.add_argument("--residual-focus-gate", type=Path, default=DEFAULT_RESIDUAL_FOCUS_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def plan_residual_guided_diverse_pool(*, residual_focus_gate: Path) -> dict[str, Any]:
    gate = _load_json(residual_focus_gate)
    status = str(gate.get("status", ""))
    decision = str(gate.get("decision", ""))
    passing_candidates = [str(candidate) for candidate in gate.get("passing_candidates", [])]
    if status == "pass" and decision == "keep_residual_focus_262k_hard_slice_candidate" and passing_candidates:
        selected = _select_candidate(gate, passing_candidates)
        return {
            "status": "pass",
            "decision": "residual_guided_diverse_pool_ready",
            "should_run_pool": True,
            "residual_focus_gate": str(residual_focus_gate),
            "residual_focus_decision": decision,
            "selected_residual_candidate": selected,
            "expert_families": [
                "trail_position_anchor",
                "compressed_span_structural",
                "residual_focus_aux_word",
            ],
            "control_families": [
                "global_trail_position_control",
                "uniform_residual_control",
                "labelshuffle_residual_control",
            ],
            "planned_fixed_fusions": [
                "best_single",
                "trail_position + raw117",
                "trail_position + raw117 + residual_focus",
                "trail_position + raw117 + uniform_residual_control",
                "trail_position + raw117 + labelshuffle_residual_control",
            ],
            "next_action": {
                "branch": "instantiate_residual_guided_pool3_fixed_fusion",
                "should_launch_remote": False,
            },
            "claim_scope": (
                "application-level medium diagnostic evidence for residual-guided diverse "
                "expert combination; not raw single-sample SOTA, not formal SPN/PRESENT "
                "evidence, and not a PRESENT r8 breakthrough claim"
            ),
        }
    if status == "fail" or decision == "hold_residual_focus_262k_controls_failed":
        return {
            "status": "hold",
            "decision": "repair_residual_focus_before_pool",
            "should_run_pool": False,
            "residual_focus_gate": str(residual_focus_gate),
            "residual_focus_decision": decision,
            "errors": [str(error) for error in gate.get("errors", [])],
            "repair_hints": [str(hint) for hint in gate.get("repair_hints", [])],
            "next_action": {
                "branch": _gate_next_branch(gate, default="repair_residual_focus_controls_before_scaleup"),
                "should_launch_remote": False,
            },
            "claim_scope": (
                "planning guard only; failed residual-focus controls must be repaired before "
                "any residual-guided diverse expert pool is instantiated"
            ),
        }
    return {
        "status": "pending",
        "decision": "wait_for_residual_focus_gate",
        "should_run_pool": False,
        "residual_focus_gate": str(residual_focus_gate),
        "residual_focus_decision": decision,
        "missing_outputs": [str(path) for path in gate.get("missing_outputs", [])],
        "next_action": {
            "branch": _gate_next_branch(gate, default="finish_residual_focus_262k_outputs"),
            "should_launch_remote": False,
        },
        "claim_scope": (
            "planning guard only; residual-guided diverse expert pool is not available until "
            "the residual-focus 262144/class gate passes"
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _select_candidate(gate: dict[str, Any], passing_candidates: list[str]) -> str:
    if "focus10" in passing_candidates and "focus05" in passing_candidates:
        focus10_margin = _float_or_none(gate.get("min_focus10_vs_uniform_loss_margin"))
        focus05_margin = _float_or_none(gate.get("min_focus05_vs_uniform_loss_margin"))
        if focus10_margin is not None and focus05_margin is not None:
            return "focus10" if focus10_margin <= focus05_margin else "focus05"
    if "focus10" in passing_candidates:
        return "focus10"
    return passing_candidates[0]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _gate_next_branch(gate: dict[str, Any], *, default: str) -> str:
    next_action = gate.get("next_action", {})
    if isinstance(next_action, dict) and next_action.get("branch"):
        return str(next_action["branch"])
    return default


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = plan_residual_guided_diverse_pool(residual_focus_gate=args.residual_focus_gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
