from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np

from blockcipher_nd.cli.gate_cross_spn_e5_source_objective_joint import (
    render_joint_gate_svg,
    write_joint_summary_csv,
)
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact
from blockcipher_nd.planning.cross_spn_e5_source_objective_gate import (
    E5_MODEL_ROLES,
    adjudicate_e5_score_artifacts,
    gate_cross_spn_e5_source_objective_joint,
)
from blockcipher_nd.planning.matrix import build_tasks


def _artifacts(*, candidate_shift: float) -> dict[str, EnsembleScoreArtifact]:
    labels = np.array([0.0] * 128 + [1.0] * 128, dtype=np.float32)
    index = np.arange(len(labels), dtype=np.float32)
    noise = ((index * 17.0) % 29.0) / 100.0
    base = 0.40 + 0.12 * labels + noise
    scores = {
        "scratch": base,
        "off_transfer": base + 0.02 * labels,
        "placebo_transfer": base + 0.01 * labels,
        "candidate_transfer": base + candidate_shift * labels,
    }
    return {
        role: EnsembleScoreArtifact(
            labels=labels,
            probabilities=values.astype(np.float32),
            logits=values.astype(np.float32),
            sample_ids=np.array([str(item) for item in range(len(labels))]),
            metadata={"model_key": E5_MODEL_ROLES[role]},
        )
        for role, values in scores.items()
    }


def test_e5_score_adjudication_rejects_candidate_below_off_anchor() -> None:
    report = adjudicate_e5_score_artifacts(
        _artifacts(candidate_shift=0.015),
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )

    assert report["decision"] == "e5_r0_source_objective_rejected"
    assert report["gate_pass"] is False
    assert report["margins"]["off_transfer"] < 0.0


def test_e5_score_adjudication_can_pass_all_controls() -> None:
    report = adjudicate_e5_score_artifacts(
        _artifacts(candidate_shift=0.20),
        bootstrap_replicates=500,
        bootstrap_seed=7,
    )

    assert report["decision"] == "e5_r0_target_seed_gate_pass"
    assert report["gate_pass"] is True
    assert all(report["point_pass"].values())
    assert all(report["ci_pass"].values())


def test_e5_joint_gate_stops_when_either_target_seed_fails() -> None:
    reports = [
        {
            "status": "pass",
            "errors": [],
            "expected_source_seed": 0,
            "expected_target_seed": 2,
            "research_decision_applied": True,
            "gate_pass": True,
        },
        {
            "status": "pass",
            "errors": [],
            "expected_source_seed": 0,
            "expected_target_seed": 3,
            "research_decision_applied": True,
            "gate_pass": False,
        },
    ]

    joint = gate_cross_spn_e5_source_objective_joint(reports)

    assert joint["status"] == "pass"
    assert joint["decision"] == "e5_r0_source_objective_rejected"
    assert joint["next_action"] == "stop_e5_r0_no_source_seed1_or_remote_scale"
    assert "65536_per_class_remote_medium" in joint["stopped_actions"]


def test_e5_joint_gate_writes_readable_summary_artifacts(tmp_path: Path) -> None:
    seed_reports = []
    for seed, candidate_shift in ((2, 0.015), (3, 0.025)):
        adjudication = adjudicate_e5_score_artifacts(
            _artifacts(candidate_shift=candidate_shift),
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )
        seed_reports.append(
            {
                "status": "pass",
                "errors": [],
                "expected_source_seed": 0,
                "expected_target_seed": seed,
                "research_decision_applied": True,
                **adjudication,
            }
        )
    report = gate_cross_spn_e5_source_objective_joint(seed_reports)
    csv_path = tmp_path / "summary.csv"
    svg_path = tmp_path / "curves.svg"

    write_joint_summary_csv(report, csv_path)
    render_joint_gate_svg(report, svg_path)

    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 7
    assert "candidate_auc" in csv_lines[0]
    root = ElementTree.parse(svg_path).getroot()
    visible_text = " ".join(
        text.strip()
        for element in root.iter()
        if element.tag.endswith("text")
        for text in element.itertext()
        if text.strip()
    )
    assert "创新1 E5-R0：源拓扑反事实辅助目标的迁移门控" in visible_text
    assert "拒绝 E5-R0" in visible_text


def test_e5_phase1a_plans_hold_source_seed_and_cross_target_seeds() -> None:
    root = Path("configs/experiment/innovation1")
    for target_seed in (2, 3):
        tasks = build_tasks(
            SimpleNamespace(
                plan=str(
                    root
                    / (
                        "innovation1_spn_gift64_cross_spn_e5_target_8192_"
                        f"source_seed0_target_seed{target_seed}.csv"
                    )
                ),
                feature_encoding="ciphertext_pair_bits",
                pairs_per_sample=1,
                difference_profile=None,
                difference_member=0,
                key_rotation_interval=0,
                sample_structure="independent_pairs",
                integral_active_nibble=0,
            )
        )

        assert [task["model_key"] for task in tasks] == list(E5_MODEL_ROLES.values())
        assert {task["seed"] for task in tasks} == {target_seed}
        assert {task["samples_per_class"] for task in tasks} == {8192}
        assert {task["pairs_per_sample"] for task in tasks} == {4}
