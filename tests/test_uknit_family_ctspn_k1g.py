from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1 import render_ctspn_k1g_svg
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1f import (
    CONTROL_CONDITIONS,
    RUN_ID as K1F_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1g import (
    EXPECTED_RESULT_ROWS,
    RUN_ID,
    adjudicate_k1g,
    dataset_row_overlap_count,
)


def test_k1g_key_specific_signal_requires_fresh_same_key_generalization() -> None:
    gate = adjudicate_k1g(
        rows=_rows(
            train_margin=0.020,
            same_key_margin=0.015,
            cross_key_margin=-0.010,
            same_key_auc=0.56,
        ),
        source_checks={"source_valid": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_uknit_family_ctspn_k1g_key_specific_hypergraph_signal_confirmed"
    )
    assert gate["attribution_summary"]["uknit64"]["same_key_fresh"] is True
    assert gate["attribution_summary"]["uknit64"]["cross_key_validation"] is False
    assert "difference-only" in gate["next_action"]


def test_k1g_same_key_failure_confirms_sample_specific_overfit() -> None:
    gate = adjudicate_k1g(
        rows=_rows(
            train_margin=0.020,
            same_key_margin=-0.010,
            cross_key_margin=-0.015,
            same_key_auc=0.51,
        ),
        source_checks={"source_valid": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_uknit_family_ctspn_k1g_"
        "sample_specific_hypergraph_attribution_overfit_confirmed"
    )
    assert "operator-tied" in gate["next_action"]


def test_k1g_separates_generalized_relation_from_weak_absolute_signal() -> None:
    gate = adjudicate_k1g(
        rows=_rows(
            train_margin=0.020,
            same_key_margin=0.015,
            cross_key_margin=-0.010,
            same_key_auc=0.51,
        ),
        source_checks={"source_valid": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_uknit_family_ctspn_k1g_"
        "same_key_relation_generalizes_but_signal_weak"
    )
    assert gate["attribution_summary"]["uknit64"]["same_key_fresh"] is True
    assert gate["uknit_auc_floor_summary"]["same_key_fresh"] is False


def test_k1g_training_relation_failure_closes_learned_incidence() -> None:
    gate = adjudicate_k1g(
        rows=_rows(
            train_margin=-0.010,
            same_key_margin=-0.010,
            cross_key_margin=-0.010,
            same_key_auc=0.51,
        ),
        source_checks={"source_valid": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_uknit_family_ctspn_k1g_shared_cell_relation_underuse_confirmed"
    )


def test_k1g_fails_closed_when_cross_key_panel_does_not_replay_k1f() -> None:
    rows = _rows(
        train_margin=0.020,
        same_key_margin=0.015,
        cross_key_margin=-0.010,
        same_key_auc=0.56,
    )
    target = next(
        row
        for row in rows
        if row["cipher_key"] == "uknit64"
        and row["seed"] == 0
        and row["split"] == "cross_key_validation"
        and row["condition"] == "correct_ordered"
    )
    target["source_validation_auc"] = float(target["auc"]) + 0.001

    gate = adjudicate_k1g(rows=rows, source_checks={"source_valid": True})

    assert gate["status"] == "invalid"
    assert gate["protocol_checks"]["cross_key_validation_replays_k1f"] is False


def test_k1g_row_overlap_counts_feature_and_label_identity() -> None:
    left = DifferentialDataset(
        features=np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
        labels=np.asarray([1, 0], dtype=np.uint8),
        metadata={},
    )
    right = DifferentialDataset(
        features=np.asarray([[0, 1], [1, 1]], dtype=np.uint8),
        labels=np.asarray([1, 0], dtype=np.uint8),
        metadata={},
    )
    different_label = DifferentialDataset(
        features=np.asarray([[0, 1]], dtype=np.uint8),
        labels=np.asarray([0], dtype=np.uint8),
        metadata={},
    )

    assert dataset_row_overlap_count(left, right) == 1
    assert dataset_row_overlap_count(left, different_label) == 0


def test_k1g_chinese_plot_explains_sample_and_key_separation(
    tmp_path: Path,
) -> None:
    gate = adjudicate_k1g(
        rows=_rows(
            train_margin=0.020,
            same_key_margin=0.015,
            cross_key_margin=-0.010,
            same_key_auc=0.56,
        ),
        source_checks={"source_valid": True},
    )
    output = tmp_path / "curves.svg"

    render_ctspn_k1g_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "失败来自记住样本，还是只适用于训练密钥" in svg
    assert "同一密钥的新明文" in svg
    assert "更换固定 key" in svg
    assert "共享-cell关系打乱" in svg


def _rows(
    *,
    train_margin: float,
    same_key_margin: float,
    cross_key_margin: float,
    same_key_auc: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cipher in ("uknit64", "dialga128"):
        for seed in (0, 1):
            checkpoint = f"checkpoint-{cipher}-{seed}"
            state = f"state-{cipher}-{seed}"
            routing = f"routing-{cipher}-{seed}"
            for split, correct_auc, margin, dataset_seed, key_scope in (
                ("train_seen", 0.72, train_margin, seed, "train_key"),
                (
                    "same_key_fresh",
                    same_key_auc if cipher == "uknit64" else 0.95,
                    same_key_margin,
                    seed + 20_000,
                    "train_key",
                ),
                (
                    "cross_key_validation",
                    0.51 if cipher == "uknit64" else 0.95,
                    cross_key_margin,
                    seed + 10_000,
                    "validation_key",
                ),
            ):
                dataset = f"dataset-{cipher}-{seed}-{split}"
                for condition in CONTROL_CONDITIONS:
                    auc = (
                        correct_auc
                        if condition == "correct_ordered"
                        else correct_auc - margin
                    )
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "source_run_id": K1F_RUN_ID,
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "condition": condition,
                            "rows": 4096 if split == "train_seen" else 2048,
                            "auc": auc,
                            "correct_minus_condition_auc": (
                                0.0 if condition == "correct_ordered" else margin
                            ),
                            "max_abs_probability_delta_from_correct": (
                                0.0 if condition == "correct_ordered" else 0.1
                            ),
                            "mean_abs_probability_delta_from_correct": (
                                0.0 if condition == "correct_ordered" else 0.01
                            ),
                            "dataset_sha256": dataset,
                            "dataset_seed": dataset_seed,
                            "key_scope": key_scope,
                            "same_key_train_overlap_rows": (
                                0 if split == "same_key_fresh" else None
                            ),
                            "source_validation_dataset_sha256": (
                                dataset if split == "cross_key_validation" else None
                            ),
                            "source_validation_auc": (
                                auc if split == "cross_key_validation" else None
                            ),
                            "checkpoint_sha256": checkpoint,
                            "expected_checkpoint_sha256": checkpoint,
                            "state_dict_sha256": state,
                            "incidence_mode": (
                                "shuffled"
                                if condition == "incidence_shuffled"
                                else "true"
                            ),
                            "routing_sha256": (
                                f"{routing}-shuffled"
                                if condition == "incidence_shuffled"
                                else routing
                            ),
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    assert len(rows) == EXPECTED_RESULT_ROWS
    return rows
