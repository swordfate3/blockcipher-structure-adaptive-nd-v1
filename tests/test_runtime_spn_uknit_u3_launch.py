from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blockcipher_nd.cli.check_runtime_spn_uknit_u3_launch import main as gate_main
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_medium import (
    RTG3A_RUN_STEM,
    adjudicate_runtime_spn_skinny_medium_joint,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_uknit_u3_launch import (
    RTG3_JOINT_RUN_ID,
    RTG3_SEED0_RUN_ID,
    U3_PLAN,
    U3_PLAN_SHA256,
    U3_READINESS_RUN_ID,
    adjudicate_runtime_spn_uknit_u3_launch,
    build_runtime_spn_uknit_u3_launch_gate,
)


ROOT = Path(__file__).resolve().parents[1]
READINESS_ROOT = ROOT / (
    "outputs/local_readiness/"
    "i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_"
    "readiness_20260725"
)


def _seed_gate(seed: int, *, supported: bool = True) -> dict[str, object]:
    aucs = (
        {"true": 0.65, "corrupted": 0.62, "independent": 0.61}
        if supported
        else {"true": 0.65, "corrupted": 0.649, "independent": 0.648}
    )
    margins = {
        "true_minus_corrupted": aucs["true"] - aucs["corrupted"],
        "true_minus_independent": aucs["true"] - aucs["independent"],
    }
    status = "pass" if supported else "hold"
    return {
        "run_id": f"{RTG3A_RUN_STEM}_seed{seed}_20260725",
        "phase": "rtg3a",
        "seed": seed,
        "samples_per_class": 1_000_000,
        "train_rows": 2_000_000,
        "validation_rows": 1_000_000,
        "status": status,
        "decision": (
            f"innovation1_rtg3a_skinny_formal_seed{seed}_supported"
            if supported
            else "innovation1_rtg3a_skinny_formal_not_supported"
        ),
        "protocol_checks": {"frozen_contract": True},
        "research_checks": {
            "true_auc_at_least_0p55": True,
            "true_exceeds_corrupted_by_0p005": supported,
            "true_exceeds_independent_by_0p005": supported,
        },
        "thresholds": {"true_auc": 0.55, "control_margin": 0.005},
        "aucs": aucs,
        "margins": margins,
    }


def _joint_gate(*, seed1_supported: bool = True) -> dict[str, object]:
    return adjudicate_runtime_spn_skinny_medium_joint(
        run_id=RTG3_JOINT_RUN_ID,
        gates=[_seed_gate(0), _seed_gate(1, supported=seed1_supported)],
        phase="rtg3a",
    )


def _readiness_gate() -> dict[str, object]:
    return {
        "run_id": U3_READINESS_RUN_ID,
        "task": "innovation1_runtime_spn_recurrent_window_readiness",
        "status": "pass",
        "decision": "innovation1_runtime_spn_recurrent_window_readiness_passed",
        "protocol_checks": {"frozen_contract": True},
    }


def _common(*, joint: dict[str, object] | None = None) -> dict[str, object]:
    readiness = _readiness_gate()
    joint_gate = joint if joint is not None else _joint_gate()
    return {
        "seed0_gate": _seed_gate(0),
        "joint_present": True,
        "joint_gate": joint_gate,
        "joint_validation": {
            "run_id": RTG3_JOINT_RUN_ID,
            "status": "pass",
            "checks": joint_gate["protocol_checks"],
        },
        "joint_recomputed_exact": True,
        "joint_sources_verified": True,
        "seed0_linked_to_joint": True,
        "readiness_gate": readiness,
        "readiness_validation": {
            "run_id": U3_READINESS_RUN_ID,
            "status": "pass",
            "checks": readiness["protocol_checks"],
            "manifest_rows": 10,
            "expected_rows": 10,
            "training_performed": False,
            "plan": U3_PLAN.as_posix(),
        },
        "readiness_recomputed_exact": True,
        "readiness_manifest_rows": 10,
        "plan_path_exact": True,
        "plan_sha256": U3_PLAN_SHA256,
    }


def test_u3_authorization_requires_verified_two_seed_rtg3_support() -> None:
    gate = adjudicate_runtime_spn_uknit_u3_launch(**_common())

    assert gate["status"] == "pass"
    assert gate["decision"] == ("innovation1_runtime_spn_uknit_u3_execution_authorized")
    assert gate["execution_authorized"] is True
    assert all(gate["seed0_checks"].values())
    assert all(gate["joint_checks"].values())
    assert all(gate["readiness_checks"].values())


def test_u3_waits_when_seed0_passed_but_joint_evidence_is_missing() -> None:
    common = _common()
    common.update(
        joint_present=False,
        joint_gate={},
        joint_validation={},
        joint_recomputed_exact=False,
        joint_sources_verified=False,
        seed0_linked_to_joint=False,
    )

    gate = adjudicate_runtime_spn_uknit_u3_launch(**common)

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation1_runtime_spn_uknit_u3_waiting_for_rtg3_joint"
    )
    assert gate["execution_authorized"] is False


def test_u3_stops_when_seed0_does_not_support_route() -> None:
    common = _common()
    common["seed0_gate"] = _seed_gate(0, supported=False)
    common.update(
        joint_present=False,
        joint_gate={},
        joint_validation={},
        joint_recomputed_exact=False,
        joint_sources_verified=False,
        seed0_linked_to_joint=False,
    )

    gate = adjudicate_runtime_spn_uknit_u3_launch(**common)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_runtime_spn_uknit_u3_not_authorized"
    assert gate["execution_authorized"] is False


def test_u3_stops_when_two_seed_joint_gate_holds() -> None:
    joint = _joint_gate(seed1_supported=False)
    common = _common(joint=joint)
    common["joint_validation"]["checks"] = joint["protocol_checks"]

    gate = adjudicate_runtime_spn_uknit_u3_launch(**common)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_runtime_spn_uknit_u3_not_authorized"
    assert gate["execution_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_group", "failed_check"),
    [
        ("seed0_metric", "seed0_checks", "seed0_research_contract_consistent"),
        ("joint_recompute", "joint_checks", "joint_gate_recomputed_exact"),
        ("joint_sha", "joint_checks", "seed0_gate_linked_by_sha256"),
        ("readiness_recompute", "readiness_checks", "readiness_gate_recomputed_exact"),
        ("plan_sha", "readiness_checks", "u3_plan_sha256_exact"),
    ],
)
def test_u3_authorization_fails_closed_on_evidence_drift(
    mutation: str,
    failed_group: str,
    failed_check: str,
) -> None:
    common = _common()
    if mutation == "seed0_metric":
        common["seed0_gate"]["margins"]["true_minus_corrupted"] = 0.5
    elif mutation == "joint_recompute":
        common["joint_recomputed_exact"] = False
    elif mutation == "joint_sha":
        common["seed0_linked_to_joint"] = False
    elif mutation == "readiness_recompute":
        common["readiness_recomputed_exact"] = False
    else:
        common["plan_sha256"] = "0" * 64

    gate = adjudicate_runtime_spn_uknit_u3_launch(**common)

    assert gate["status"] == "fail"
    assert gate["execution_authorized"] is False
    assert gate[failed_group][failed_check] is False


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rtg3_evidence(tmp_path: Path) -> tuple[Path, Path]:
    seed0_root = tmp_path / "seed0"
    seed0_path = seed0_root / "gate.local.json"
    seed1_path = tmp_path / "seed1" / "gate.local.json"
    _write_json(seed0_path, _seed_gate(0))
    _write_json(seed1_path, _seed_gate(1))

    joint = _joint_gate()
    joint_root = tmp_path / "joint"
    _write_json(joint_root / "gate.json", joint)
    _write_json(
        joint_root / "validation.json",
        {
            "run_id": RTG3_JOINT_RUN_ID,
            "status": "pass",
            "checks": joint["protocol_checks"],
            "sources": [
                {"path": str(seed0_path), "sha256": _sha256(seed0_path)},
                {"path": str(seed1_path), "sha256": _sha256(seed1_path)},
            ],
        },
    )
    return seed0_root, joint_root


def test_u3_build_gate_replays_current_readiness_and_joint_sources(
    tmp_path: Path,
) -> None:
    seed0_root, joint_root = _write_rtg3_evidence(tmp_path)

    gate = build_runtime_spn_uknit_u3_launch_gate(
        seed0_root=seed0_root,
        rtg3_joint_root=joint_root,
        readiness_root=READINESS_ROOT,
        plan=ROOT / U3_PLAN,
        repository=ROOT,
    )

    assert gate["status"] == "pass"
    assert gate["execution_authorized"] is True
    assert gate["readiness_checks"]["readiness_gate_recomputed_exact"] is True
    assert gate["joint_checks"]["joint_gate_recomputed_exact"] is True
    assert gate["joint_checks"]["seed0_gate_linked_by_sha256"] is True


def test_u3_cli_writes_auditable_authorization_artifacts(tmp_path: Path) -> None:
    seed0_root, joint_root = _write_rtg3_evidence(tmp_path)
    output_root = tmp_path / "authorization"

    status = gate_main(
        [
            "--seed0-root",
            str(seed0_root),
            "--rtg3-joint-root",
            str(joint_root),
            "--readiness-root",
            str(READINESS_ROOT),
            "--plan",
            str(ROOT / U3_PLAN),
            "--repository",
            str(ROOT),
            "--output-root",
            str(output_root),
        ]
    )

    assert status == 0
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    progress = [
        json.loads(line)
        for line in (output_root / "progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert gate["execution_authorized"] is True
    assert summary["training_performed"] is False
    assert summary["execution_authorized"] is True
    assert progress[-1]["event"] == "u3_authorization_gate_done"


def test_u3_plan_identity_and_script_are_frozen() -> None:
    plan = ROOT / U3_PLAN
    script = ROOT / "scripts/check-runtime-spn-uknit-u3-launch"

    assert _sha256(plan) == U3_PLAN_SHA256
    assert script.exists()
    assert RTG3_SEED0_RUN_ID == f"{RTG3A_RUN_STEM}_seed0_20260725"
