from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli import audit_innovation2_topology_geometry as cli
from blockcipher_nd.tasks.innovation2 import integral_topology_geometry as topology
from blockcipher_nd.tasks.innovation2.integral_topology_geometry import (
    TopologyGeometryConfig,
    adjudicate_topology_geometry_audit,
    present_p_bit,
    topology_geometries,
)


def test_topology_geometries_are_twelve_unique_p_orbit_sets() -> None:
    geometries = topology_geometries()

    assert len(geometries) == 12
    assert len(set(geometries.values())) == 12
    assert geometries["block00_p0"] == tuple(range(16))
    assert geometries["block12_p0"] == tuple(range(48, 64))
    assert geometries["block00_p1"] == tuple(
        sorted(present_p_bit(bit) for bit in range(16))
    )
    assert geometries["block00_p0"] == tuple(
        sorted(present_p_bit(bit) for bit in geometries["block00_p2"])
    )


def test_topology_gate_requires_four_signatures_and_six_structures() -> None:
    config = TopologyGeometryConfig(run_id="test")
    readiness = {"ok": True}

    passed = adjudicate_topology_geometry_audit(
        config,
        [],
        readiness,
        distinct_signatures=4,
        nontrivial_structures=6,
    )
    held = adjudicate_topology_geometry_audit(
        config,
        [],
        readiness,
        distinct_signatures=3,
        nontrivial_structures=12,
    )

    assert passed["decision"] == (
        "innovation2_topology_geometry_kernel_diversity_ready"
    )
    assert held["decision"] == (
        "innovation2_topology_geometry_kernel_diversity_insufficient"
    )


def test_topology_runner_returns_complete_result(monkeypatch) -> None:
    constrained = {0, 4, 12, 16, 48, 20, 28, 52, 60}
    orthogonal_rows = [1 << bit for bit in range(64) if bit not in constrained]
    orthogonal_rows.extend(
        [
            (1 << 4) | (1 << 12),
            (1 << 16) | (1 << 48),
            (1 << 20) | (1 << 28),
            (1 << 20) | (1 << 52),
            (1 << 20) | (1 << 60),
        ]
    )
    words = np.asarray(
        orthogonal_rows + orthogonal_rows + orthogonal_rows[:8],
        dtype=np.uint64,
    )
    monkeypatch.setattr(
        topology,
        "_collect_xor_words",
        lambda structure, keys, **kwargs: words.copy(),
    )
    monkeypatch.setattr(
        topology,
        "scalar_bit_integral_output_xor",
        lambda structure, rounds, key: int(words[0]),
    )

    result = topology.run_topology_geometry_audit(
        TopologyGeometryConfig(run_id="topology-runner")
    )

    assert len(result["rows"]) == 12
    assert result["gate"]["status"] == "hold"
    assert all(result["gate"]["readiness_checks"].values())
    assert result["metadata"]["task"] == (
        "innovation2_present_r7_topology_geometry_kernel_diversity"
    )


def test_topology_cli_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    rows = [
        {
            "run_id": "test",
            "geometry_id": f"block{start:02d}_p{power}",
            "base_start_nibble": start,
            "p_power": power,
            "joint_kernel_dimension": 4 if start == 12 else 0,
            "joint_basis_signature": "paper" if start == 12 else "",
        }
        for start in (0, 4, 8, 12)
        for power in (0, 1, 2)
    ]
    gate = {
        "run_id": "test",
        "status": "hold",
        "decision": "innovation2_topology_geometry_kernel_diversity_insufficient",
        "distinct_joint_kernel_signatures": 2,
        "nontrivial_joint_kernel_structures": 3,
        "next_action": {"training": False, "remote_scale": False},
    }
    monkeypatch.setattr(
        cli,
        "run_topology_geometry_audit",
        lambda config, progress_callback: {
            "rows": rows,
            "basis_rows": [{"run_id": "test", "geometry_id": "block12_p0"}],
            "gate": gate,
            "metadata": {"run_id": "test"},
        },
    )

    assert cli.main(["--run-id", "test", "--output-root", str(tmp_path)]) == 0

    assert {
        "results.jsonl",
        "kernel_basis.csv",
        "progress.jsonl",
        "gate.json",
        "metadata.json",
        "curves.svg",
    }.issubset(path.name for path in tmp_path.iterdir())
    assert json.loads((tmp_path / "gate.json").read_text())["status"] == "hold"
    svg = (tmp_path / "curves.svg").read_text(encoding="utf-8")
    assert "PRESENT P-layer 拓扑活动几何" in svg
    assert "不是神经训练" in svg
