from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.tasks.innovation1.runtime_spn_h1_source_gradient_audit import (
    GRADIENT_VIEWS,
    adjudicate_h1_source_gradient_audit,
    load_and_validate_h1_source_gradient_config,
    pairwise_gradient_cosines,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    EXPECTED_SOURCES,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_h1_source_gradient_alignment_a1_seed0_seed1.json"
)


def test_frozen_h1_source_gradient_config_is_valid() -> None:
    config = load_and_validate_h1_source_gradient_config(CONFIG, project_root=ROOT)

    assert config["audit"]["rows_per_cipher"] == 4096
    assert tuple(config["audit"]["source_ciphers"]) == EXPECTED_SOURCES
    assert tuple(config["audit"]["gradient_views"]) == GRADIENT_VIEWS
    assert config["source"]["expected_failing_seeds"] == [1]


def test_pairwise_gradient_cosines_cover_all_views_and_pairs() -> None:
    gradients = {
        str(seed): {
            task: {
                view: torch.tensor([1.0, float(index + seed + 1)])
                for view in GRADIENT_VIEWS
            }
            for index, task in enumerate(EXPECTED_SOURCES)
        }
        for seed in (0, 1)
    }

    rows = pairwise_gradient_cosines(gradients)

    assert len(rows) == 36
    assert all(row["cosine"] is not None for row in rows)


def test_gate_opens_gradient_normalization_for_failing_seed_imbalance() -> None:
    payload = _payload(largest_seed1_share=0.55, seed1_conflict=-0.2)

    gate = adjudicate_h1_source_gradient_audit(payload)

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("source_gradient_imbalance_supported")
    assert gate["failing_seed_gradient_imbalance"] is True


def test_gate_uses_representation_audit_without_imbalance_or_conflict() -> None:
    payload = _payload(largest_seed1_share=0.28, seed1_conflict=0.1)

    gate = adjudicate_h1_source_gradient_audit(payload)

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("representation_alignment_priority")
    assert gate["failing_seed_gradient_imbalance"] is False
    assert gate["failing_seed_gradient_conflict"] is False


def test_invalid_source_evidence_fails_closed() -> None:
    payload = _payload(largest_seed1_share=0.55, seed1_conflict=-0.2)
    payload["validation"] = {"status": "fail"}

    gate = adjudicate_h1_source_gradient_audit(payload)

    assert gate["status"] == "invalid"
    assert gate["decision"].endswith("source_gradient_audit_invalid")


def _payload(
    *,
    largest_seed1_share: float,
    seed1_conflict: float,
) -> dict[str, object]:
    config = load_and_validate_h1_source_gradient_config(CONFIG, project_root=ROOT)
    gradient_norms = []
    for seed in (0, 1):
        shares = [0.25, 0.25, 0.25, 0.25]
        norms = [1.0, 1.0, 1.0, 1.0]
        if seed == 1:
            shares = [largest_seed1_share, 0.15, 0.15, 0.15]
            norms = [3.0 if largest_seed1_share >= 0.5 else 1.1, 1.0, 1.0, 1.0]
        for task, share, norm in zip(EXPECTED_SOURCES, shares, norms, strict=True):
            gradient_norms.append(
                {
                    "seed": seed,
                    "task": task,
                    "view": "representation_backbone",
                    "l2_norm": norm,
                    "norm_share": share,
                    "finite": True,
                }
            )
    cosines = []
    for seed in (0, 1):
        for index, task_a in enumerate(EXPECTED_SOURCES):
            for task_b in EXPECTED_SOURCES[index + 1 :]:
                cosine = 0.1
                if seed == 1 and {task_a, task_b} == {"gift64", "dialga128"}:
                    cosine = seed1_conflict
                cosines.append(
                    {
                        "seed": seed,
                        "view": "representation_backbone",
                        "task_a": task_a,
                        "task_b": task_b,
                        "cosine": cosine,
                    }
                )
    source_auc = []
    for seed in (0, 1):
        aucs = (0.48, 0.53, 0.48, 0.94) if seed == 1 else (0.54, 0.54, 0.51, 0.94)
        source_auc.extend(
            {"seed": seed, "task": task, "validation_auc": auc}
            for task, auc in zip(EXPECTED_SOURCES, aucs, strict=True)
        )
    return {
        "config": config,
        "gradient_norms": gradient_norms,
        "gradient_cosines": cosines,
        "source_auc": source_auc,
        "validation": {"status": "pass"},
    }
