from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_h1_relation_activity_pooling as a5,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_h1_relation_activity_pooling import (
    adjudicate_h1_relation_activity_pooling,
    load_and_validate_h1_relation_activity_pooling_config,
    run_h1_relation_activity_pooling,
    run_h1_relation_activity_pooling_readiness,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    EXPECTED_SEEDS,
    EXPECTED_SOURCES,
    RelationModeRuntimeE4,
    _plain_spec,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_h1_relation_activity_pooling_a5_2048_seed0_seed1.json"
)


def test_frozen_h1_a5_config_is_valid_without_readiness_artifact() -> None:
    config = load_and_validate_h1_relation_activity_pooling_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )

    assert config["candidate"]["relation_activity_pooling_mode"] == "correct"
    assert config["candidate"]["pooling_controls"] == ["uniform", "shuffled"]
    assert config["gate"]["pooling_control_margin"] == 0.005


def test_readiness_rejects_unidentifiable_target_and_parameter_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_and_validate_h1_relation_activity_pooling_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    h1_config = json.loads((ROOT / config["source"]["h1_config_path"]).read_text())
    checkpoint_state = RelationModeRuntimeE4(
        _plain_spec(h1_config["model"]),
        "true",
    ).state_dict()
    monkeypatch.setattr(
        a5.torch,
        "load",
        lambda *_args, **_kwargs: {"state_dict": checkpoint_state},
    )

    unidentifiable = run_h1_relation_activity_pooling_readiness(
        config=config,
        project_root=ROOT,
    )

    assert unidentifiable["status"] == "fail"
    assert unidentifiable["checks"]["target_pooling_controls_identifiable"] is False

    monkeypatch.setattr(a5, "HOLDOUT_CIPHER", "uknit64")
    readiness = run_h1_relation_activity_pooling_readiness(
        config=config,
        project_root=ROOT,
    )

    assert readiness["status"] == "pass"
    assert readiness["checks"]["one_to_one_logits_bit_exact"] is True
    assert readiness["checks"]["both_a3_checkpoints_load_strictly"] is True
    assert readiness["target_rows_loaded"] == 0

    drifted = copy.deepcopy(config)
    drifted["candidate"]["expected_parameter_count"] = 1
    failed = run_h1_relation_activity_pooling_readiness(
        config=drifted,
        project_root=ROOT,
    )

    assert failed["status"] == "fail"
    assert failed["checks"]["parameter_counts_exact"] is False


def test_gate_passes_when_all_frozen_requirements_hold() -> None:
    gate = adjudicate_h1_relation_activity_pooling(_payload())

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("relation_activity_pooling_supported")
    assert gate["full_pass"] is True


def test_gate_retains_only_partial_evidence() -> None:
    payload = _payload()
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        payload["candidate_source_auc"][key]["skinny64"] = 0.501
        payload["source_pooling_control_auc"][key]["uniform"]["skinny64"] = 0.500
        payload["source_pooling_control_auc"][key]["shuffled"]["skinny64"] = 0.500

    gate = adjudicate_h1_relation_activity_pooling(payload)

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("relation_activity_pooling_partial")
    assert gate["partial"] is True


def test_gate_closes_unsupported_pooling_primitive() -> None:
    payload = _payload()
    payload["candidate_target_auc"]["1"]["candidate_correct"] = 0.51

    gate = adjudicate_h1_relation_activity_pooling(payload)

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("relation_activity_pooling_not_supported")


def test_gate_fails_closed_on_invalid_protocol() -> None:
    payload = _payload()
    payload["validation"]["status"] = "fail"

    gate = adjudicate_h1_relation_activity_pooling(payload)

    assert gate["status"] == "invalid"
    assert gate["decision"].endswith("relation_activity_pooling_invalid")


def test_same_checkpoint_pooling_controls_have_zero_optimizer_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_and_validate_h1_relation_activity_pooling_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    h1_config = json.loads((ROOT / config["source"]["h1_config_path"]).read_text())
    candidate_state = RelationModeRuntimeE4(
        a5._pooling_spec(h1_config["model"], "correct"),
        "true",
    ).state_dict()
    base = _payload()
    base.update(
        {
            "rows": [],
            "history": [],
            "gradient_scales": [
                {"seed": seed, "conflict_projections": 1} for seed in EXPECTED_SEEDS
            ],
            "validation": {"status": "pass", "checks": {}},
        }
    )
    monkeypatch.setattr(a5, "run_h1_gradient_equalization", lambda **_kwargs: base)
    monkeypatch.setattr(a5, "load_and_validate_holdout_config", lambda _path: h1_config)
    monkeypatch.setattr(
        a5, "_load_structures", lambda _config: {"rectangle80": object()}
    )
    monkeypatch.setattr(a5, "_load_source_tasks", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        a5, "_load_target_validation", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        a5,
        "evaluate_runtime_spn_joint",
        lambda *_args, **_kwargs: {name: {"auc": 0.6} for name in EXPECTED_SOURCES},
    )
    monkeypatch.setattr(a5, "_evaluate_target", lambda *_args, **_kwargs: {"auc": 0.6})
    monkeypatch.setattr(
        a5.torch,
        "load",
        lambda *_args, **_kwargs: {"state_dict": candidate_state},
    )

    def read_json(path: Path) -> dict[str, object]:
        if path.name in {"source-metrics.json", "target-metrics.json"}:
            return {"0": {}, "1": {}}
        return {"status": "pass", "decision": "expected"}

    monkeypatch.setattr(a5, "_read_json", read_json)
    output_root = tmp_path / "a5"
    payload = run_h1_relation_activity_pooling(
        config=config,
        config_path=CONFIG,
        output_root=output_root,
        project_root=ROOT,
    )

    expected_rows = len(EXPECTED_SEEDS) * 2 * (len(EXPECTED_SOURCES) + 1)
    assert len(payload["control_rows"]) == expected_rows
    assert {row["optimizer_steps"] for row in payload["control_rows"]} == {0}
    for seed in EXPECTED_SEEDS:
        paths = {
            row["checkpoint"] for row in payload["control_rows"] if row["seed"] == seed
        }
        assert paths == {str(output_root / "checkpoints" / f"seed{seed}-candidate.pt")}


def test_existing_result_is_revalidated_as_protocol_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(
        load_and_validate_h1_relation_activity_pooling_config(
            CONFIG,
            project_root=ROOT,
            require_readiness=False,
        )
    )
    config["source"]["readiness_gate_path"] = "readiness/gate.json"
    config["source"]["a3_output_root"] = "a3"
    output_root = tmp_path / "run"
    output_root.mkdir()
    _write_json(
        tmp_path / "readiness/gate.json",
        {
            "checks": {"target_pooling_controls_identifiable": False},
        },
    )
    _write_json(
        output_root / "validation.json",
        {"status": "pass", "checks": {"old_protocol": True}},
    )
    _write_json(
        output_root / "gate.json",
        {
            "run_id": config["run_id"],
            "status": "hold",
            "decision": (
                "innovation1_runtime_spn_h1_relation_activity_pooling_not_supported"
            ),
            "claim_scope": "diagnostic",
        },
    )
    for path in (
        output_root / "source-metrics.json",
        output_root / "target-metrics.json",
        output_root / "source-pooling-controls.json",
        output_root / "target-pooling-controls.json",
        tmp_path / "a3/source-metrics.json",
        tmp_path / "a3/target-metrics.json",
    ):
        _write_json(path, {})
    monkeypatch.setattr(
        a5,
        "render_h1_relation_activity_pooling_svg",
        lambda *_args, **_kwargs: None,
    )

    gate = a5.revalidate_existing_h1_relation_activity_pooling(
        config=config,
        output_root=output_root,
        project_root=tmp_path,
    )

    assert gate["status"] == "invalid"
    assert gate["decision"].endswith("relation_activity_pooling_invalid")
    assert gate["supersedes_decision"].endswith(
        "relation_activity_pooling_not_supported"
    )
    repeated = a5.revalidate_existing_h1_relation_activity_pooling(
        config=config,
        output_root=output_root,
        project_root=tmp_path,
    )
    assert repeated["supersedes_decision"] == gate["supersedes_decision"]
    validation = json.loads((output_root / "validation.json").read_text())
    assert validation["status"] == "fail"
    assert validation["checks"]["target_pooling_controls_identifiable"] is False


def _payload() -> dict[str, object]:
    config = load_and_validate_h1_relation_activity_pooling_config(
        CONFIG,
        project_root=ROOT,
        require_readiness=False,
    )
    candidate_source = {
        str(seed): {
            "gift64": 0.53,
            "skinny64": 0.55,
            "uknit64": 0.56,
            "dialga128": 0.93,
        }
        for seed in EXPECTED_SEEDS
    }
    pooling_controls = {
        str(seed): {
            mode: {
                "gift64": 0.52,
                "skinny64": 0.53,
                "uknit64": 0.54,
                "dialga128": 0.92,
            }
            for mode in ("uniform", "shuffled")
        }
        for seed in EXPECTED_SEEDS
    }
    return {
        "config": config,
        "candidate_source_auc": candidate_source,
        "a3_source_auc": {
            str(seed): {
                "gift64": 0.52,
                "skinny64": 0.49,
                "uknit64": 0.54,
                "dialga128": 0.93,
            }
            for seed in EXPECTED_SEEDS
        },
        "anchor_source_auc": {
            str(seed): {
                "gift64": 0.52,
                "skinny64": 0.535,
                "uknit64": 0.54,
                "dialga128": 0.93,
            }
            for seed in EXPECTED_SEEDS
        },
        "source_pooling_control_auc": pooling_controls,
        "candidate_target_auc": {
            str(seed): {
                "candidate_correct": 0.68,
                "candidate_corrupted_target": 0.65,
                "candidate_no_topology_target": 0.64,
            }
            for seed in EXPECTED_SEEDS
        },
        "a3_target_auc": {
            str(seed): {"candidate_correct": 0.69} for seed in EXPECTED_SEEDS
        },
        "target_pooling_control_auc": {
            str(seed): {"uniform": 0.66, "shuffled": 0.65} for seed in EXPECTED_SEEDS
        },
        "conflict_projections_by_seed": {str(seed): 2 for seed in EXPECTED_SEEDS},
        "validation": {"status": "pass"},
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
