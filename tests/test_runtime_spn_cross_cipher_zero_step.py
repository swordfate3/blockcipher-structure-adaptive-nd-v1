from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from blockcipher_nd.cli import audit_runtime_spn_cross_cipher_zero_step as cli
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_zero_step import (
    EXPECTED_CONDITIONS,
    FROZEN_MODEL_OPTIONS,
    TARGET_MODELS,
    adjudicate_zero_step_panel,
    validate_target_result_rows,
)


def _row(seed: int, condition: str, auc: float) -> dict[str, object]:
    target_mode = {
        "true_source_true_target": "true",
        "corrupted_source_true_target": "true",
        "true_source_corrupted_target": "corrupted",
        "true_source_no_target": "independent",
    }[condition]
    source_role = "corrupted" if condition == "corrupted_source_true_target" else "true"
    checkpoint_hash = (
        ("c" if seed == 0 else "d") * 64
        if source_role == "corrupted"
        else ("a" if seed == 0 else "b") * 64
    )
    structure_hash = {
        "true": "1" * 64,
        "corrupted": "2" * 64,
        "independent": "1" * 64,
    }[target_mode]
    relation_mode = "independent" if target_mode == "independent" else "true"
    intervention_hash = {
        "true": "8" * 64,
        "corrupted": "9" * 64,
        "independent": "0" * 64,
    }[target_mode]
    probability_delta = 0.0 if condition == "true_source_true_target" else 0.01
    return {
        "seed": seed,
        "condition": condition,
        "source_role": source_role,
        "target_mode": target_mode,
        "target_relation_mode": relation_mode,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_selected": "best",
        "target_structure_sha256": structure_hash,
        "target_intervention_sha256": intervention_hash,
        "auc": auc,
        "max_abs_probability_delta_from_candidate": probability_delta,
        "mean_probability": 0.5,
        "probability_sha256": f"{seed}{condition}".encode().hex().ljust(64, "0")[:64],
        "feature_sha256": ("4" if seed == 0 else "5") * 64,
        "label_sha256": ("6" if seed == 0 else "7") * 64,
        "metadata_sha256": ("8" if seed == 0 else "9") * 64,
        "target_results_sha256": ("e" if seed == 0 else "f") * 64,
        "target_validation_key": 0x1111111111111111,
        "source_protocol_verified": True,
        "target_dataset_metadata_verified": True,
        "samples_total": 2048,
        "input_bits": 512,
        "pairs_per_sample": 4,
        "source_cipher": "GIFT-64",
        "source_rounds": 6,
        "target_cipher": "SKINNY-64/64",
        "target_rounds": 7,
        "target_difference": 0x2000,
        "negative_mode": "encrypted_random_plaintexts",
        "model_options": FROZEN_MODEL_OPTIONS,
        "parameter_count": 442466,
        "strict_state_dict_load": True,
        "training_performed": False,
    }


def _passing_rows() -> list[dict[str, object]]:
    aucs = {
        "true_source_true_target": 0.56,
        "corrupted_source_true_target": 0.54,
        "true_source_corrupted_target": 0.53,
        "true_source_no_target": 0.52,
    }
    return [
        _row(seed, condition, aucs[condition])
        for seed in (0, 1)
        for condition in EXPECTED_CONDITIONS
    ]


def test_zero_step_gate_passes_complete_two_seed_panel() -> None:
    gate = adjudicate_zero_step_panel(run_id="x1", rows=_passing_rows())

    assert gate["status"] == "pass"
    assert (
        gate["decision"] == "innovation1_runtime_spn_zero_step_topology_use_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["sensitivity_checks"].values())
    assert all(gate["research_checks"].values())


def test_zero_step_gate_holds_when_auc_margin_misses() -> None:
    rows = _passing_rows()
    rows[1]["auc"] = 0.558

    gate = adjudicate_zero_step_panel(run_id="x1-hold", rows=rows)

    assert gate["status"] == "hold"
    assert (
        gate["decision"]
        == "innovation1_runtime_spn_topology_sensitive_not_zero_step_discriminative"
    )
    assert gate["research_checks"]["seed0_beats_source_by_0p005"] is False


def test_zero_step_gate_fails_when_intervention_is_inactive() -> None:
    rows = deepcopy(_passing_rows())
    rows[2]["max_abs_probability_delta_from_candidate"] = 0.0

    gate = adjudicate_zero_step_panel(run_id="x1-fail", rows=rows)

    assert gate["status"] == "fail"
    assert (
        gate["decision"]
        == "innovation1_runtime_spn_zero_step_protocol_or_intervention_invalid"
    )
    assert gate["sensitivity_checks"]["seed0_target_probabilities_change"] is False


def test_target_result_protocol_rejects_wrong_validation_key() -> None:
    rows = []
    for model in TARGET_MODELS.values():
        rows.append(
            {
                "model": model,
                "cipher": "SKINNY-64/64",
                "rounds": 7,
                "seed": 0,
                "pairs_per_sample": 4,
                "input_difference": 0x2000,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "structure": "SPN",
                "validation_key": 0x2222222222222222,
                "training": {
                    "input_bits": 512,
                    "validation_rows": 2048,
                    "model_options": FROZEN_MODEL_OPTIONS,
                },
                "validation": {"samples_total": 2048, "samples_per_class": 1024},
            }
        )

    with pytest.raises(ValueError, match="target attribution protocol"):
        validate_target_result_rows(rows, 0)


def test_gift_state_dict_strictly_loads_all_skinny_adapters() -> None:
    source = build_model(
        "gift64_runtime_e4_equivariant_true",
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=FROZEN_MODEL_OPTIONS,
    )
    source_state = source.state_dict()

    for target_model in TARGET_MODELS.values():
        target = build_model(
            target_model,
            input_bits=512,
            hidden_bits=64,
            pair_bits=128,
            structure="SPN",
            model_options=FROZEN_MODEL_OPTIONS,
        )
        target.load_state_dict(source_state, strict=True)
        assert sum(parameter.numel() for parameter in target.parameters()) == 442466


def test_zero_step_cli_writes_complete_artifacts(tmp_path: Path, monkeypatch) -> None:
    rows = _passing_rows()

    def fake_evaluate_seed(_args, seed: int):
        return [row for row in rows if row["seed"] == seed]

    monkeypatch.setattr(cli, "_evaluate_seed", fake_evaluate_seed)
    assert cli.main(["--output-root", str(tmp_path), "--run-id", "cli-x1"]) == 0

    expected = {
        "results.jsonl",
        "validation.json",
        "gate.json",
        "summary.json",
        "progress.jsonl",
        "curves.svg",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    gate = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 8
    svg = (tmp_path / "curves.svg").read_text(encoding="utf-8")
    assert "GIFT 权重零训练切换到 SKINNY" in svg
    assert "不是目标适配" in svg


def test_zero_step_cache_loader_uses_exact_seed_directory(tmp_path: Path) -> None:
    cache_dir = (
        tmp_path / "cache" / "skinny64" / "r7" / "validation" / "seed-10000_fixture"
    )
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / "features.npy", np.zeros((8, 512), dtype=np.uint8))
    np.save(cache_dir / "labels.npy", np.array([0, 1] * 4, dtype=np.uint8))
    (cache_dir / "metadata.json").write_text('{"seed": 10000}\n')

    dataset, feature_path, label_path, metadata_path = cli._load_target_dataset(
        tmp_path, 0
    )

    assert dataset.cache_dir == cache_dir
    assert feature_path == cache_dir / "features.npy"
    assert label_path == cache_dir / "labels.npy"
    assert metadata_path == cache_dir / "metadata.json"
