from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.tasks.innovation1.runtime_spn_topology_diversity import (
    RUN_ID,
    build_real_transition_panel,
    cell_relabel_matrix,
    gf2_rank,
    lift_transition,
    load_and_validate_config,
    mutate_invertible_matrix,
    run_topology_diversity_audit,
    topology_features,
    write_audit_artifacts,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_topology_diversity import (
    _stable_seed,
    _topology_seed_material_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/experiment/innovation1/"
    "innovation1_runtime_spn_source_topology_diversity_d1_20260726.json"
)


def test_frozen_d1_config_and_real_transition_panel() -> None:
    config = load_and_validate_config(CONFIG)
    panel = build_real_transition_panel(config)

    assert config["run_id"] == RUN_ID
    assert config["audit"] == {
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
    }
    assert {row.cipher for row in panel} == {
        "present",
        "gift",
        "rectangle",
        "skinny",
        "uknit",
        "dialga",
    }
    assert len(panel) == 18
    assert sum(row.cipher == "uknit" for row in panel) == 10
    assert sum(row.cipher == "dialga" for row in panel) == 4
    assert {row.block_bits for row in panel} == {64, 128}
    assert all(gf2_rank(row.matrix) == row.block_bits for row in panel)


def test_elementary_mutation_and_width_lift_remain_invertible() -> None:
    matrix = torch.eye(8, dtype=torch.uint8)
    membership = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    roles = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    mutated = mutate_invertible_matrix(
        matrix,
        mutation_count=8,
        seed=3,
        half_width=None,
    )
    lifted, lifted_membership, lifted_roles = lift_transition(
        matrix,
        membership,
        roles,
        factor=2,
    )
    lifted_mutated = mutate_invertible_matrix(
        lifted,
        mutation_count=8,
        seed=3,
        half_width=8,
    )

    assert gf2_rank(mutated) == 8
    assert not torch.equal(mutated, matrix)
    assert lifted.shape == (16, 16)
    assert lifted_membership.tolist() == membership.tolist() + [2, 2, 2, 2, 3, 3, 3, 3]
    assert lifted_roles.tolist() == roles.tolist() * 2
    assert gf2_rank(lifted_mutated) == 16
    assert bool(torch.any(lifted_mutated[:8, 8:])) or bool(
        torch.any(lifted_mutated[8:, :8])
    )


def test_generator_seed_depends_on_topology_not_cipher_identity() -> None:
    matrix = torch.eye(8, dtype=torch.uint8)
    membership = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    roles = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    source_sha256 = _topology_seed_material_sha256(matrix, membership, roles)
    first = _stable_seed(source_sha256, 8, 4, 0)
    second = _stable_seed(source_sha256, 8, 4, 0)
    changed_topology = matrix.clone()
    changed_topology[0] ^= changed_topology[1]
    changed_sha256 = _topology_seed_material_sha256(
        changed_topology,
        membership,
        roles,
    )

    assert first == second
    assert source_sha256 != changed_sha256
    assert first != _stable_seed(changed_sha256, 8, 4, 0)


def test_cell_relabel_control_preserves_topology_features() -> None:
    matrix = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=torch.uint8,
    )
    membership = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    roles = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    relabeled = cell_relabel_matrix(
        matrix,
        membership,
        roles,
        cell_permutation=(1, 0),
    )

    source = topology_features(matrix, membership, roles, power_exponents=(0, 1, 2, 3))
    control = topology_features(
        relabeled,
        membership,
        roles,
        power_exponents=(0, 1, 2, 3),
    )

    assert gf2_rank(relabeled) == 8
    assert not torch.equal(relabeled, matrix)
    assert source.signature == control.signature
    assert torch.equal(source.vector, control.vector)


def test_artifact_writer_uses_zero_training_contract(tmp_path: Path) -> None:
    payload = {
        "candidates": [{"candidate_id": "candidate-0"}],
        "results": [{"holdout_cipher": "present"}],
        "validation": {"run_id": RUN_ID, "status": "pass", "errors": []},
        "gate": {
            "run_id": RUN_ID,
            "status": "hold",
            "decision": "innovation1_runtime_spn_source_topology_diversity_not_ready",
        },
        "summary": {"run_id": RUN_ID, "training_rows": 0, "optimizer_steps": 0},
    }

    write_audit_artifacts(payload, tmp_path)

    assert len((tmp_path / "candidates.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path / "validation.json").read_text())["status"] == "pass"
    assert json.loads((tmp_path / "gate.json").read_text())["status"] == "hold"
    assert json.loads((tmp_path / "summary.json").read_text())["training_rows"] == 0
    assert not (tmp_path / "curves.svg").exists()


def test_full_d1_audit_preserves_frozen_protocol() -> None:
    config = load_and_validate_config(CONFIG)

    payload = run_topology_diversity_audit(config)

    assert len(payload["candidates"]) == 448
    assert len(payload["results"]) == 18
    assert payload["validation"]["status"] == "pass"
    assert payload["validation"]["errors"] == []
    assert (
        payload["validation"]["manifest_sha256"]
        == payload["validation"]["repeated_manifest_sha256"]
    )
    assert all(payload["validation"]["checks"].values())
    assert payload["gate"]["training_rows"] == 0
    assert payload["gate"]["optimizer_steps"] == 0
    assert payload["gate"]["remote"] is False
