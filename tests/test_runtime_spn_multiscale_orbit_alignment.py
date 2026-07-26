from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.tasks.innovation1.runtime_spn_multiscale_orbit_alignment import (
    RUN_ID,
    build_aligned_structure_panel,
    load_and_validate_alignment_config,
    run_multiscale_orbit_alignment_audit,
    write_audit_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/experiment/innovation1/"
    "innovation1_runtime_spn_multiscale_orbit_protocol_alignment_c4p_20260726.json"
)


def test_frozen_c4p_config_uses_exact_two_transition_protocol_windows() -> None:
    config = load_and_validate_alignment_config(CONFIG)
    panel, metadata = build_aligned_structure_panel(config, project_root=ROOT)

    assert config["run_id"] == RUN_ID
    assert config["audit"] == {
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
    }
    assert config["orbit"]["depths"] == [0, 1, 2, 4, 8]
    assert config["orbit"]["semantics"] == (
        "periodic_topology_operator_power_not_literal_round_state"
    )
    assert set(panel) == {
        "present",
        "gift",
        "rectangle",
        "skinny",
        "uknit",
        "dialga",
    }
    assert all(structure.rounds == 2 for structure in panel.values())
    assert metadata["uknit"]["round_start"] == 3
    assert metadata["dialga"]["round_start"] == 2
    assert all(
        torch.equal(panel[name].linear_matrices[0], panel[name].linear_matrices[1])
        for name in ("present", "gift", "rectangle", "skinny")
    )
    assert all(
        not torch.equal(panel[name].linear_matrices[0], panel[name].linear_matrices[1])
        for name in ("uknit", "dialga")
    )


def test_c4p_audit_is_protocol_valid_and_preserves_zero_training() -> None:
    config = load_and_validate_alignment_config(CONFIG)
    payload = run_multiscale_orbit_alignment_audit(
        config,
        project_root=ROOT,
    )

    assert len(payload["results"]) == 6
    assert payload["validation"]["status"] == "pass"
    assert payload["validation"]["errors"] == []
    assert all(payload["validation"]["checks"].values())
    assert payload["gate"]["training_rows"] == 0
    assert payload["gate"]["optimizer_steps"] == 0
    assert payload["gate"]["remote"] is False
    assert payload["gate"]["status"] in {"pass", "hold"}
    assert payload["gate"]["decision"] in {
        "innovation1_runtime_spn_multiscale_orbit_protocol_alignment_supported",
        "innovation1_runtime_spn_multiscale_orbit_protocol_alignment_not_supported",
    }
    assert all(
        row["orbit_semantics"]
        == "periodic_topology_operator_power_not_literal_round_state"
        for row in payload["results"]
    )


def test_c4p_artifact_writer_emits_no_visualization(tmp_path: Path) -> None:
    config = load_and_validate_alignment_config(CONFIG)
    payload = run_multiscale_orbit_alignment_audit(
        config,
        project_root=ROOT,
    )

    write_audit_artifacts(payload, tmp_path)

    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 6
    assert json.loads((tmp_path / "validation.json").read_text())["status"] == "pass"
    assert json.loads((tmp_path / "gate.json").read_text())["run_id"] == RUN_ID
    assert json.loads((tmp_path / "summary.json").read_text())["training_rows"] == 0
    assert not (tmp_path / "curves.svg").exists()
