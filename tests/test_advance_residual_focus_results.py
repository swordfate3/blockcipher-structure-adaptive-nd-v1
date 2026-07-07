from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.advance_residual_focus_results import main as advance_main


def test_advance_residual_focus_results_waits_when_outputs_missing(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=False)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = advance_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate-output",
            str(gate),
            "--pool-output",
            str(pool),
            "--status-output",
            str(status_output),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pending"
    assert report["decision"] == "wait_for_residual_focus_outputs"
    assert report["ran_gate"] is False
    assert report["ran_pool_planner"] is False
    assert not pool.exists()


def test_advance_residual_focus_results_runs_gate_and_pool_when_outputs_ready(tmp_path):
    action_plan = _write_action_plan(tmp_path, create_outputs=True)
    gate = tmp_path / "gate.json"
    pool = tmp_path / "pool.json"
    status_output = tmp_path / "status.json"
    output = tmp_path / "advance.json"
    gate.write_text(
        json.dumps({"status": "pending", "decision": "wait_for_residual_focus_262k_outputs"}),
        encoding="utf-8",
    )

    status = advance_main(
        [
            "--action-plan",
            str(action_plan),
            "--gate-output",
            str(gate),
            "--pool-output",
            str(pool),
            "--status-output",
            str(status_output),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    gate_report = json.loads(gate.read_text(encoding="utf-8"))
    pool_report = json.loads(pool.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "residual_guided_pool_ready"
    assert report["ran_gate"] is True
    assert report["ran_pool_planner"] is True
    assert gate_report["decision"] == "keep_residual_focus_262k_hard_slice_candidate"
    assert pool_report["decision"] == "residual_guided_diverse_pool_ready"
    assert pool_report["selected_residual_candidate"] == "focus10"


def _write_action_plan(tmp_path: Path, *, create_outputs: bool) -> Path:
    seeds = []
    for seed in [0, 1]:
        seed_root = tmp_path / f"seed{seed}"
        outputs = {
            "uniform_slice_eval": seed_root / "uniform_slice_eval.json",
            "focus10_shuffle_slice_eval": seed_root / "focus10_shuffle_slice_eval.json",
            "focus05_slice_eval": seed_root / "focus05_slice_eval.json",
            "focus10_slice_eval": seed_root / "focus10_slice_eval.json",
        }
        if create_outputs:
            seed_root.mkdir(parents=True, exist_ok=True)
            _write_slice(outputs["uniform_slice_eval"], loss_delta=-0.001, auc_delta=0.001)
            _write_slice(outputs["focus10_shuffle_slice_eval"], loss_delta=0.03, auc_delta=-0.1)
            _write_slice(outputs["focus05_slice_eval"], loss_delta=-0.01, auc_delta=0.002)
            _write_slice(outputs["focus10_slice_eval"], loss_delta=-0.02, auc_delta=0.003)
        seeds.append(
            {
                "seed": seed,
                "planned_outputs": {key: str(path) for key, path in outputs.items()},
            }
        )
    action_plan = tmp_path / "action_plan.json"
    action_plan.write_text(
        json.dumps(
            {
                "status": "pass",
                "source_decision": "hold_trail_position_score_residual_mixed_runs",
                "source_gate_assessment": "score_artifacts_ready_but_trail_position_gate_not_promoted",
                "seeds": seeds,
            }
        ),
        encoding="utf-8",
    )
    return action_plan


def _write_slice(path: Path, *, loss_delta: float, auc_delta: float) -> None:
    path.write_text(
        json.dumps(
            {
                "focus": {"mode": "train_derived_base_residual_loss_threshold"},
                "validation_focus_metrics": {"rows": 16},
                "validation_focus_delta": {
                    "auc": auc_delta,
                    "residual_loss_mean": loss_delta,
                },
            }
        ),
        encoding="utf-8",
    )
