from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.tasks.innovation1.runtime_spn_multiscale_orbit import (
    RUN_ID,
    build_structure_panel,
    exact_inverse_orbit,
    load_and_validate_config,
    run_multiscale_orbit_audit,
    support_jaccard_distance,
    write_audit_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/experiment/innovation1/"
    "innovation1_runtime_spn_multiscale_orbit_basis_c4_20260726.json"
)


def test_frozen_c4_config_and_structure_panel() -> None:
    config = load_and_validate_config(CONFIG)
    panel = build_structure_panel(config)

    assert config["run_id"] == RUN_ID
    assert config["audit"] == {
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
    }
    assert config["orbit"]["depths"] == [0, 1, 2, 4, 8]
    assert set(panel) == {
        "present",
        "gift",
        "rectangle",
        "skinny",
        "uknit",
        "dialga",
    }
    assert {name: value.block_bits for name, value in panel.items()} == {
        "present": 64,
        "gift": 64,
        "rectangle": 64,
        "skinny": 64,
        "uknit": 64,
        "dialga": 128,
    }
    assert panel["uknit"].rounds == 10
    assert panel["dialga"].rounds == 4


def test_exact_inverse_orbit_follows_reverse_window_and_cycles() -> None:
    swap = torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)
    upper = torch.tensor([[1, 1], [0, 1]], dtype=torch.uint8)

    orbit = exact_inverse_orbit(
        torch.stack((upper, swap)),
        depths=(0, 1, 2, 4),
    )

    identity = torch.eye(2, dtype=torch.uint8)
    depth_one = swap
    depth_two = torch.remainder(upper.to(torch.int16) @ swap.to(torch.int16), 2).to(
        torch.uint8
    )
    depth_four = torch.remainder(
        depth_two.to(torch.int16) @ depth_two.to(torch.int16), 2
    ).to(torch.uint8)
    assert torch.equal(orbit[0], identity)
    assert torch.equal(orbit[1], depth_one)
    assert torch.equal(orbit[2], depth_two)
    assert torch.equal(orbit[3], depth_four)


def test_support_jaccard_distance_has_exact_control_semantics() -> None:
    identity = torch.eye(4, dtype=torch.uint8).repeat(2, 1, 1)
    changed = identity.clone()
    changed[1, 0] ^= changed[1, 1]

    assert support_jaccard_distance(identity, identity) == 0.0
    assert 0.0 < support_jaccard_distance(identity, changed) <= 1.0


def test_full_c4_audit_preserves_zero_training_protocol() -> None:
    config = load_and_validate_config(CONFIG)

    payload = run_multiscale_orbit_audit(config)

    assert len(payload["results"]) == 6
    assert payload["validation"]["status"] == "pass"
    assert payload["validation"]["errors"] == []
    assert all(payload["validation"]["checks"].values())
    assert (
        payload["validation"]["manifest_sha256"]
        == payload["validation"]["repeated_manifest_sha256"]
    )
    assert payload["gate"]["training_rows"] == 0
    assert payload["gate"]["optimizer_steps"] == 0
    assert payload["gate"]["remote"] is False
    assert payload["gate"]["status"] == "pass"
    assert (
        payload["gate"]["decision"]
        == "innovation1_runtime_spn_multiscale_orbit_basis_feasible"
    )
    assert all(payload["gate"]["research_checks"].values())


def test_artifact_writer_emits_no_visualization(tmp_path: Path) -> None:
    payload = {
        "results": [{"run_id": RUN_ID, "cipher": "present"}],
        "validation": {"run_id": RUN_ID, "status": "pass", "errors": []},
        "gate": {
            "run_id": RUN_ID,
            "status": "hold",
            "decision": "innovation1_runtime_spn_multiscale_orbit_basis_not_ready",
        },
        "summary": {
            "run_id": RUN_ID,
            "training_rows": 0,
            "optimizer_steps": 0,
        },
    }

    write_audit_artifacts(payload, tmp_path)

    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path / "validation.json").read_text())["status"] == "pass"
    assert json.loads((tmp_path / "gate.json").read_text())["status"] == "hold"
    assert json.loads((tmp_path / "summary.json").read_text())["training_rows"] == 0
    assert not (tmp_path / "curves.svg").exists()
