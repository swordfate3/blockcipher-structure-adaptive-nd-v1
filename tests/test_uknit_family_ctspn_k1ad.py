from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1ad import render_k1ad_svg
from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ad import (
    _validate_dataset,
    adjudicate,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _rows(*, margins: tuple[float, float], deltas: tuple[float, float] = (0.2, 0.3)) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, margin in enumerate(margins):
        exact_auc = 0.999 - seed * 0.001
        common = {
            "run_id": "i1_uknit_family_ctspn_dialga_same_checkpoint_k1ad_20260729",
            "seed": seed,
            "cipher_key": "dialga128",
            "rounds": 4,
            "source_exact_auc": exact_auc,
            "checkpoint_path": f"checkpoint-seed{seed}.pt",
            "checkpoint_sha256": SHA_A if seed == 0 else SHA_B,
            "checkpoint_selected": "best",
            "checkpoint_reported_seed": seed,
            "checkpoint_best_metric": exact_auc,
            "state_dict_sha256": SHA_C,
            "feature_sha256": SHA_A,
            "label_sha256": SHA_B,
            "metadata_sha256": SHA_C,
            "cache_dir": f"cache-seed{seed}",
            "source_results_sha256": SHA_A,
            "source_gate_sha256": SHA_B,
            "source_progress_sha256": SHA_C,
            "source_decision": "innovation1_uknit_family_ctspn_k1ac_semantic_attribution_failed",
            "samples_total": 2048,
            "input_bits": 4096,
            "pair_bits": 256,
            "pairs_per_sample": 16,
            "input_difference": 0x40,
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
            "validation_seed": 10000 + seed,
            "parameter_count": 214316,
            "strict_state_dict_load": True,
            "training_performed": False,
            "optimizer_steps": 0,
            "mean_abs_probability_delta_from_exact": 0.0,
            "max_abs_probability_delta_from_exact": 0.0,
        }
        rows.append(
            {
                **common,
                "condition": "exact",
                "auc": exact_auc,
                "exact_minus_condition_auc": 0.0,
                "probability_sha256": SHA_A,
                "runtime_structure_window_sha256": SHA_A,
            }
        )
        rows.append(
            {
                **common,
                "condition": "wrong_sbox",
                "auc": exact_auc - margin,
                "exact_minus_condition_auc": margin,
                "probability_sha256": SHA_B,
                "runtime_structure_window_sha256": SHA_B,
                "max_abs_probability_delta_from_exact": deltas[seed],
                "mean_abs_probability_delta_from_exact": deltas[seed] / 10.0,
            }
        )
    return rows


def test_k1ad_gate_passes_only_with_per_seed_same_checkpoint_margin() -> None:
    gate = adjudicate(_rows(margins=(0.020, 0.015)))
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_uknit_family_ctspn_k1ad_functional_sbox_use_supported"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_k1ad_gate_holds_when_predictions_change_without_auc_margin() -> None:
    gate = adjudicate(_rows(margins=(0.0001, -0.0002)))
    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_uknit_family_ctspn_k1ad_discriminative_sbox_use_failed"
    assert gate["research_checks"]["seed0_prediction_changes"] is True
    assert gate["research_checks"]["seed1_prediction_changes"] is True
    assert gate["research_checks"]["seed0_exact_beats_wrong_sbox_by_0p010"] is False
    assert gate["research_checks"]["seed1_exact_beats_wrong_sbox_by_0p010"] is False


def test_k1ad_gate_rejects_cross_condition_checkpoint_drift() -> None:
    rows = _rows(margins=(0.020, 0.015))
    mutated = deepcopy(rows)
    mutated[1]["checkpoint_sha256"] = "d" * 64
    gate = adjudicate(mutated)
    assert gate["status"] == "invalid"
    assert gate["protocol_checks"]["same_checkpoint_and_state_within_seed"] is False


def test_k1ad_gate_rejects_training_or_cache_drift() -> None:
    rows = _rows(margins=(0.020, 0.015))
    mutated = deepcopy(rows)
    mutated[3]["optimizer_steps"] = 1
    mutated[3]["input_bits"] = 1024
    gate = adjudicate(mutated)
    assert gate["status"] == "invalid"
    assert gate["protocol_checks"]["inference_only"] is False
    assert gate["protocol_checks"]["frozen_validation_geometry"] is False


def test_k1ad_plot_uses_plain_chinese_explanation(tmp_path: Path) -> None:
    gate = adjudicate(_rows(margins=(0.0001, -0.0002)))
    output = tmp_path / "curves.svg"
    report = render_k1ad_svg(gate, output)
    text = output.read_text(encoding="utf-8")
    assert report["status"] == "rendered_pending_visual_qa"
    assert "同一组权重是否真正依赖" in text
    assert "全程零训练" in text
    assert "基础路径与结构直方图残差" in text


def test_k1ad_accepts_the_real_cache_metadata_schema(tmp_path: Path) -> None:
    dataset = DiskDifferentialDataset(
        features=np.zeros((2048, 4096), dtype=np.uint8),
        labels=np.zeros((2048,), dtype=np.uint8),
        metadata={
            "cipher": "Dialga-128",
            "rounds": 4,
            "seed": 10000,
            "pair_bits": 256,
            "pairs_per_sample": 16,
            "input_difference": 0x40,
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
        },
        cache_dir=tmp_path,
    )
    _validate_dataset(dataset, 0)
