from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_experiment import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_descriptor_audit import (
    GRADIENT_VIEWS,
    adjudicate_primitive_descriptor_audit,
    build_descriptor_profiles,
    find_descriptor_collisions,
    load_and_validate_audit_config,
    pairwise_gradient_cosines,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_primitive_descriptor_gradient_audit_seed0_seed1.json"
)


def test_frozen_descriptor_audit_config_is_valid() -> None:
    config = load_and_validate_audit_config(CONFIG, project_root=ROOT)

    assert config["audit"]["rows_per_cipher"] == 4096
    assert config["audit"]["split"] == "train"
    assert tuple(config["audit"]["gradient_views"]) == GRADIENT_VIEWS


def test_five_cipher_coarse_router_has_expected_structural_collisions() -> None:
    source = load_and_validate_audit_config(CONFIG, project_root=ROOT)
    source_config = __import__("json").loads(
        (ROOT / source["source"]["config_path"]).read_text(encoding="utf-8")
    )
    structures = {
        item["name"]: load_runtime_spn_descriptor(
            item["runtime_structure_path"],
            rounds=2,
            round_start=item["runtime_round_start"],
        ).structure
        for item in source_config["protocols"]
    }
    profiles = build_descriptor_profiles(structures)
    collisions = find_descriptor_collisions(profiles)
    task_sets = {frozenset(collision["tasks"]) for collision in collisions}

    assert frozenset(("gift64", "rectangle80")) in task_sets
    assert frozenset(("uknit64", "dialga128")) in task_sets
    assert profiles["skinny64"]["normalized_route_signature"] not in {
        profiles["gift64"]["normalized_route_signature"],
        profiles["uknit64"]["normalized_route_signature"],
    }


def test_pairwise_cosines_preserve_inactive_adapter_as_null() -> None:
    gradients = {}
    for seed in (0, 1):
        gradients[str(seed)] = {}
        for index, task in enumerate(EXPECTED_CIPHERS):
            gradients[str(seed)][task] = {
                view: torch.tensor([1.0, float(index + 1)]) for view in GRADIENT_VIEWS
            }
        gradients[str(seed)]["gift64"]["multi_source_adapter"] = torch.zeros(2)
    rows = pairwise_gradient_cosines(gradients)
    inactive = [
        row
        for row in rows
        if row["seed"] == 0
        and row["view"] == "multi_source_adapter"
        and "gift64" in {row["task_a"], row["task_b"]}
    ]

    assert len(rows) == 80
    assert all(row["cosine"] is None for row in inactive)


def test_gate_prioritizes_descriptor_when_same_route_gradients_conflict() -> None:
    config = load_and_validate_audit_config(CONFIG, project_root=ROOT)
    cosine_rows = []
    for seed in (0, 1):
        for view in GRADIENT_VIEWS:
            for index, task_a in enumerate(EXPECTED_CIPHERS):
                for task_b in EXPECTED_CIPHERS[index + 1 :]:
                    cosine = 0.2
                    if view == "fan_in_1_adapter" and {
                        task_a,
                        task_b,
                    } == {"gift64", "rectangle80"}:
                        cosine = -0.2
                    cosine_rows.append(
                        {
                            "seed": seed,
                            "view": view,
                            "task_a": task_a,
                            "task_b": task_b,
                            "cosine": cosine,
                        }
                    )
    payload = {
        "config": config,
        "descriptor_collisions": [
            {
                "normalized_route_signature": "1:0|1:0",
                "tasks": ["gift64", "rectangle80"],
                "distinct_window_fingerprints": 2,
            },
            {
                "normalized_route_signature": "0:1|0:1",
                "tasks": ["dialga128", "uknit64"],
                "distinct_window_fingerprints": 2,
            },
        ],
        "gradient_cosines": cosine_rows,
        "validation": {"status": "pass"},
    }

    gate = adjudicate_primitive_descriptor_audit(payload)

    assert gate["status"] == "pass"
    assert gate["descriptor_refinement_priority"] is True
    assert gate["shared_gradient_conflict"] is False
    assert gate["decision"].endswith("descriptor_refinement_priority")
