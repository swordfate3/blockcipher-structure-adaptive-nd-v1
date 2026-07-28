from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1s import render_k1s_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1r import read_tasks
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import build_k1n_control
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import task_map
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1s import (
    EXPECTED_CHECKPOINT_SHAS,
    EXPECTED_FEATURE_DIMS,
    EXPECTED_SOURCE_DIGESTS,
    SCORER_MODES,
    TAPS,
    adjudicate_k1s,
    extract_k1s_batch_taps,
    source_binding_checks,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_cell11_neural_attribution_k1r_2048_seed3_seed4.csv"
)


def test_k1s_hooks_capture_frozen_path_without_changing_logits() -> None:
    task = task_map(read_tasks(PLAN))[(3, "exact_composition")]
    model = build_k1n_control(
        task=task,
        condition="exact_composition",
        input_bits=512,
    )
    features = torch.randint(
        0,
        2,
        (5, 512),
        generator=torch.Generator().manual_seed(20260728),
    ).float()

    taps, replayed = extract_k1s_batch_taps(features, model.eval())

    assert replayed is True
    assert {name: values.shape for name, values in taps.items()} == {
        tap: (5, EXPECTED_FEATURE_DIMS[tap]) for tap in TAPS
    }


def test_k1s_source_gate_requires_exact_sources_and_two_exact_checkpoints() -> None:
    checks = source_binding_checks(
        source_digests=EXPECTED_SOURCE_DIGESTS,
        k1q_gate={
            "run_id": (
                "i1_uknit_family_ctspn_difference_position_discovery_"
                "k1q_seed2_confirm_seed3_seed4_20260728"
            ),
            "status": "pass",
            "decision": (
                "innovation1_uknit_family_ctspn_k1q_confirmed_"
                "r5_difference_position_supported"
            ),
            "confirmed_cells": [11, 0],
            "protocol_checks": {"complete": True},
        },
        k1q_validation={
            "run_id": (
                "i1_uknit_family_ctspn_difference_position_discovery_"
                "k1q_seed2_confirm_seed3_seed4_20260728"
            ),
            "status": "pass",
            "errors": [],
        },
        k1r_gate={
            "run_id": (
                "i1_uknit_family_ctspn_cell11_neural_attribution_"
                "k1r_2048_seed3_seed4_20260728"
            ),
            "status": "hold",
            "decision": (
                "innovation1_uknit_family_ctspn_k1r_"
                "cell11_neural_signal_not_supported"
            ),
            "protocol_checks": {"complete": True},
        },
        k1r_validation={
            "run_id": (
                "i1_uknit_family_ctspn_cell11_neural_attribution_"
                "k1r_2048_seed3_seed4_20260728"
            ),
            "status": "pass",
            "errors": [],
        },
        dataset_manifest=source_manifest_rows(),
        checkpoint_entries=[
            {
                "seed": seed,
                "condition": "exact_composition",
                "sha256": EXPECTED_CHECKPOINT_SHAS[seed],
                "selected_checkpoint": "best",
            }
            for seed in (3, 4)
        ],
    )
    assert all(checks.values())

    wrong = dict(EXPECTED_SOURCE_DIGESTS)
    wrong["k1r_gate"] = "0" * 64
    failed = source_binding_checks(
        source_digests=wrong,
        k1q_gate={},
        k1q_validation={},
        k1r_gate={},
        k1r_validation={},
        dataset_manifest=[],
        checkpoint_entries=[],
    )
    assert failed["source_artifact_digests_exact"] is False


def test_k1s_gate_identifies_invariant_pool_bottleneck() -> None:
    gate = adjudicate_k1s(
        feature_rows=synthetic_feature_rows(),
        scorer_rows=synthetic_scorer_rows(),
        result_rows=synthetic_result_rows(),
        source_checks={"source": True},
        k1r_logit_rows=synthetic_logit_rows(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("invariant_cell_pool_bottleneck_supported")
    assert gate["tap_accessible_on_all_fresh_splits"][TAPS[2]] is True
    assert gate["tap_accessible_on_all_fresh_splits"][TAPS[3]] is False
    assert all(gate["protocol_checks"].values())


def test_k1s_plot_explains_taps_in_chinese(tmp_path: Path) -> None:
    gate = adjudicate_k1s(
        feature_rows=synthetic_feature_rows(),
        scorer_rows=synthetic_scorer_rows(),
        result_rows=synthetic_result_rows(),
        source_checks={"source": True},
        k1r_logit_rows=synthetic_logit_rows(),
    )
    output = tmp_path / "curves.svg"

    report = render_k1s_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "uKNIT 第5轮强信号在神经网络哪一层消失" in svg
    assert "T2 拓扑更新量" in svg
    assert "T3 cell无序池化后表示" in svg
    assert "标签打乱" in svg


def source_manifest_rows() -> list[dict[str, object]]:
    return [
        {
            "run_id": (
                "i1_uknit_family_ctspn_difference_position_discovery_"
                "k1q_seed2_confirm_seed3_seed4_20260728"
            ),
            "phase": "confirmation",
            "cell": 11,
            "seed": seed,
            "split": split,
            "input_difference": 0x0000400000000000,
            "rows": 4096 if split == "train_seen" else 2048,
            "cache_payloads_present": True,
        }
        for seed in (3, 4)
        for split in ("train_seen", "same_key_fresh", "cross_key_validation")
    ]


def synthetic_feature_rows() -> list[dict[str, object]]:
    rows = []
    for seed in (3, 4):
        for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
            for tap in TAPS:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "tap": tap,
                        "rows": 4096 if split == "train_seen" else 2048,
                        "feature_dim": EXPECTED_FEATURE_DIMS[tap],
                        "feature_sha256": "feature",
                        "source_feature_sha256": (
                            "feature" if tap == TAPS[0] else None
                        ),
                        "finite": True,
                        "ordinary_logits_bit_exact": True,
                        "state_dict_unchanged": True,
                        "state_dict_sha256_before": "state",
                        "state_dict_sha256_after": "state",
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "epochs": 0,
                    }
                )
    return rows


def synthetic_scorer_rows() -> list[dict[str, object]]:
    rows = []
    for seed in (3, 4):
        for tap in TAPS:
            for mode in SCORER_MODES:
                rows.append(
                    {
                        "seed": seed,
                        "tap": tap,
                        "mode": mode,
                        "feature_dim": EXPECTED_FEATURE_DIMS[tap],
                        "scorer_sha256": "scorer",
                        "source_scorer_sha256": (
                            "scorer" if tap == TAPS[0] else None
                        ),
                        "class_counts_preserved": True,
                        "label_assignment_changed": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "epochs": 0,
                    }
                )
    return rows


def synthetic_result_rows() -> list[dict[str, object]]:
    interpreted_auc = {
        TAPS[0]: 0.82,
        TAPS[1]: 0.71,
        TAPS[2]: 0.68,
        TAPS[3]: 0.51,
    }
    rows = []
    for seed in (3, 4):
        for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
            for tap in TAPS:
                for mode in SCORER_MODES:
                    auc = interpreted_auc[tap] if mode == "interpreted" else 0.50
                    rows.append(
                        {
                            "seed": seed,
                            "split": split,
                            "tap": tap,
                            "mode": mode,
                            "rows": 4096 if split == "train_seen" else 2048,
                            "auc": auc,
                            "source_auc": auc if tap == TAPS[0] else None,
                            "score_mean": 0.0,
                            "score_std": 1.0,
                            "feature_sha256": "feature",
                            "source_feature_sha256": (
                                "feature" if tap == TAPS[0] else None
                            ),
                            "scorer_sha256": "scorer",
                            "source_scorer_sha256": (
                                "scorer" if tap == TAPS[0] else None
                            ),
                            "pairs_per_sample": 4,
                            "negative_mode": "encrypted_random_plaintexts",
                            "fit_split": "train_seen",
                            "training_performed": False,
                            "optimizer_steps": 0,
                            "epochs": 0,
                        }
                    )
    return rows


def synthetic_logit_rows() -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "split": split,
            "condition": "exact_composition",
            "auc": 0.51,
        }
        for seed in (3, 4)
        for split in ("train_seen", "same_key_fresh", "cross_key_validation")
    ]
