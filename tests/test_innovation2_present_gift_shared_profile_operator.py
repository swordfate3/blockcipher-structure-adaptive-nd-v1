from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from blockcipher_nd.cli.plot_innovation2_present_gift_shared_profile_operator import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_gift_shared_profile_operator_readiness import (
    SharedProfileReadinessConfig,
    _epoch_schedule,
    adjudicate_shared_profile_readiness,
    inverse_player,
    make_shared_profile_model,
    measure_shared_profile_contract,
    relation_players,
)


def _sources(structures: int = 4) -> dict[str, object]:
    rng = np.random.default_rng(structures)
    targets = rng.integers(0, 2, size=(structures, 64), dtype=np.int8)
    return {
        "prefix_features": rng.normal(size=(structures, 64, 13)).astype(np.float64),
        "profile_targets": targets,
        "profile_observed": np.ones((structures, 64), dtype=np.bool_),
        "matched_rows": [
            {
                "structure_index": structure,
                "output_bit": output,
                "label": int(targets[structure, output]),
                "split": "train" if structure < structures - 1 else "validation",
            }
            for structure in range(structures)
            for output in range(64)
        ],
    }


def test_shared_operator_accepts_runtime_topologies_without_cipher_state() -> None:
    config = SharedProfileReadinessConfig(run_id="e85-test")
    model = make_shared_profile_model(config, "true", dropout=0.0)
    features = torch.randn(3, 64, 13)
    players = relation_players("true")

    present = model(features, inverse_player(players["present"]))
    gift = model(features, inverse_player(players["gift"]))

    assert present.shape == (3, 64)
    assert float(torch.max(torch.abs(present - gift)).detach()) > 1e-6
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_795
    assert not any("player" in name or "cipher" in name for name in model.state_dict())


def test_shared_operator_rejects_non_permutation_runtime_topology() -> None:
    model = make_shared_profile_model(
        SharedProfileReadinessConfig(run_id="e85-test"), "true"
    )
    with pytest.raises(ValueError, match="permutation"):
        model(torch.randn(2, 64, 13), torch.zeros(64, dtype=torch.long))


def test_e85_contract_is_fair_dynamic_and_equivariant() -> None:
    contract = measure_shared_profile_contract(
        SharedProfileReadinessConfig(run_id="e85-test"),
        {"present": _sources(), "gift": _sources()},
    )

    assert set(contract["parameter_counts"].values()) == {4_795}
    assert contract["initial_parameter_max_abs_difference"] == 0.0
    assert contract["runtime_topology_state_absent"] is True
    assert min(contract["topology_logit_max_abs_differences"].values()) > 1e-6
    assert max(contract["cell_relabel_max_abs_errors"].values()) <= 1e-6
    assert contract["true_players_are_distinct"] is True


def _contract() -> dict[str, object]:
    return {
        "output_shapes": {"present": [4, 64], "gift": [4, 64]},
        "masked_loss_explicit_max_abs_errors": {"present": 0.0, "gift": 0.0},
        "parameter_counts": {mode: 4_795 for mode in ("independent", "true", "corrupted")},
        "parameter_counts_match": True,
        "initial_parameter_max_abs_difference": 0.0,
        "runtime_topology_state_absent": True,
        "cipher_specific_named_state_absent": True,
        "topology_logit_max_abs_differences": {"present": 0.1, "gift": 0.2},
        "cell_relabel_max_abs_errors": {"present": 1e-7, "gift": 1e-7},
        "true_players_are_distinct": True,
        "corrupted_players_are_distinct_from_true": True,
        "all_players_are_permutations": True,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
    }


def _matrix(true_present: float = 0.82, true_gift: float = 0.78) -> dict[str, object]:
    values = {
        "independent": (0.68, 0.61),
        "corrupted": (0.74, 0.70),
        "true": (true_present, true_gift),
    }
    rows = []
    for mode, (present_auc, gift_auc) in values.items():
        row = {
            "relation_mode": mode,
            "epochs_completed": 2,
            "macro_validation_auc": (present_auc + gift_auc) / 2,
        }
        for cipher, auc in (("present", present_auc), ("gift", gift_auc)):
            for split in ("train", "validation"):
                row[f"{cipher}_{split}_auc"] = auc
                row[f"{cipher}_{split}_accuracy"] = 0.70
                row[f"{cipher}_{split}_loss"] = 0.55
        rows.append(row)
    audits = {
        mode: {
            "epoch_batch_counts": [
                {"present": 7, "gift": 14},
                {"present": 7, "gift": 14},
            ],
            "total_updates": 42,
        }
        for mode in values
    }
    return {"rows": rows, "history": [], "schedule_audits": audits}


def test_e85_gate_requires_both_ciphers_to_beat_controls() -> None:
    gate = adjudicate_shared_profile_readiness(
        SharedProfileReadinessConfig(run_id="e85-test"),
        {
            "source_valid": True,
            "present_r3_shape_is_96x64x13": True,
            "gift_r3_shape_is_192x64x13": True,
        },
        _contract(),
        _matrix(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_shared_profile_operator_readiness_passed"

    held = adjudicate_shared_profile_readiness(
        SharedProfileReadinessConfig(run_id="e85-test"),
        {
            "source_valid": True,
            "present_r3_shape_is_96x64x13": True,
            "gift_r3_shape_is_192x64x13": True,
        },
        _contract(),
        _matrix(true_gift=0.70),
    )
    assert held["status"] == "hold"


def test_e85_epoch_schedule_matches_combined_anchor_budget() -> None:
    schedule = _epoch_schedule(
        {
            "present": {"train": list(range(50))},
            "gift": {"train": list(range(110))},
        },
        batch_size=8,
        seed=85,
    )

    assert len(schedule) == 21
    assert sum(cipher == "present" for cipher, _ in schedule) == 7
    assert sum(cipher == "gift" for cipher, _ in schedule) == 14
    assert sorted(index for cipher, batch in schedule if cipher == "present" for index in batch) == list(range(50))
    assert sorted(index for cipher, batch in schedule if cipher == "gift" for index in batch) == list(range(110))


def test_plot_writes_clear_chinese_e85_svg(tmp_path: Path) -> None:
    gate = adjudicate_shared_profile_readiness(
        SharedProfileReadinessConfig(run_id="e85-test"),
        {
            "source_valid": True,
            "present_r3_shape_is_96x64x13": True,
            "gift_r3_shape_is_192x64x13": True,
        },
        _contract(),
        _matrix(),
    )
    summary = tmp_path / "summary.json"
    output = tmp_path / "curves.svg"
    summary.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary), "--output", str(output)]) == 0
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E85" in svg
    assert "一套共享神经算子" in svg
    assert "不是零样本迁移" in svg


def test_e85_protocol_is_frozen() -> None:
    with pytest.raises(ValueError, match="frozen"):
        SharedProfileReadinessConfig(run_id="e85-test", hidden_dim=64)
