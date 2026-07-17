from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_small_spn_exact_labels import (
    render_small_spn_exact_label_svg,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_matched_contrast import (
    render_matched_contrast_svg,
)
from blockcipher_nd.tasks.innovation2 import small_spn_exact_labels as small


def test_frozen_variants_structures_and_masks_are_unique_bijections() -> None:
    config = small.SmallSpnAuditConfig(run_id="audit")
    variants = small.make_variants(config)
    structures = small.make_structures()
    masks = small.make_output_masks()
    assert len(variants) == 16
    assert len({(variant.sbox, variant.player) for variant in variants}) == 16
    assert all(sorted(variant.sbox) == list(range(16)) for variant in variants)
    assert all(sorted(variant.player) == list(range(16)) for variant in variants)
    assert len(structures) == 14
    assert {structure.dimension for structure in structures} == {4, 8, 12}
    assert len(masks) == len(set(masks)) == 64
    assert all(mask != 0 for mask in masks)


def test_scalar_and_vectorized_small_spn_match() -> None:
    assert small.scalar_vectorized_fixture_matches() is True


def test_cache_resumes_and_labels_recompute_from_parity(tmp_path: Path) -> None:
    config = small.SmallSpnAuditConfig(
        run_id="cache",
        mode="smoke",
        sbox_variants=1,
        player_variants=1,
        rounds=(1,),
        keys=4,
    )
    first = small.run_cached_exact_labels(config, cache_root=tmp_path)
    resumed = small.run_cached_exact_labels(config, cache_root=tmp_path)
    assert first["generated_blocks"] == 14
    assert resumed["generated_blocks"] == 0
    assert first["completed"].all()
    result = small.evaluate_exact_labels(
        config, first, resume_generated_blocks=resumed["generated_blocks"]
    )
    assert result["gate"]["decision"] == (
        "innovation2_small_spn_exact_label_readiness_passed"
    )
    assert all(result["gate"]["readiness_checks"].values())


def test_gate_separates_ready_shortcut_and_narrow() -> None:
    config = small.SmallSpnAuditConfig(run_id="audit")
    readiness = {"protocol": True}

    def metrics(*, positive: int, negative: int, signatures: int, fraction: float, auc: float):
        splits = {
            name: {"positive": 300, "negative": 300}
            for name in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
        }
        baselines = {
            name: {"strongest_auc": auc}
            for name in ("unseen_sbox", "unseen_player", "dual_unseen")
        }
        return {
            "positive_labels": positive,
            "negative_labels": negative,
            "distinct_label_signatures": signatures,
            "cipher_variable_cell_fraction": fraction,
            "split_metrics": splits,
            "marginal_baselines": baselines,
        }

    ready = small.adjudicate_exact_labels(
        config,
        readiness,
        metrics(positive=6000, negative=6000, signatures=80, fraction=0.2, auc=0.7),
    )
    assert ready["decision"] == "innovation2_small_spn_exact_label_family_ready"
    shortcut = small.adjudicate_exact_labels(
        config,
        readiness,
        metrics(positive=6000, negative=6000, signatures=80, fraction=0.2, auc=0.9),
    )
    assert shortcut["decision"] == (
        "innovation2_small_spn_exact_label_shortcut_dominated"
    )
    narrow = small.adjudicate_exact_labels(
        config,
        readiness,
        metrics(positive=100, negative=6000, signatures=80, fraction=0.2, auc=0.7),
    )
    assert narrow["decision"] == "innovation2_small_spn_exact_label_too_narrow"


def test_binary_auc_handles_ties() -> None:
    target = np.asarray([0, 0, 1, 1], dtype=np.bool_)
    assert small._binary_auc(target, np.asarray([0.0, 0.0, 1.0, 1.0])) == 1.0
    assert small._binary_auc(target, np.ones(4)) == 0.5


def test_matched_selection_depends_only_on_train_labels() -> None:
    rng = np.random.default_rng(44)
    labels = rng.integers(0, 2, size=(16, 4, 14, 64), dtype=np.uint8).astype(np.bool_)
    split = small._split_indices(
        small.make_variants(small.SmallSpnAuditConfig(run_id="selection"))
    )
    selected = small.select_train_contrast_cells(labels, split["train"])
    modified = labels.copy()
    heldout = np.concatenate(
        [split["unseen_sbox"], split["unseen_player"], split["dual_unseen"]]
    )
    modified[heldout] = ~modified[heldout]
    assert np.array_equal(
        selected,
        small.select_train_contrast_cells(modified, split["train"]),
    )


def test_matched_plot_compares_raw_and_selected_shortcuts(tmp_path: Path) -> None:
    split_metrics = {
        "train": {"positive": 2700, "negative": 2600},
        "unseen_sbox": {"positive": 1000, "negative": 700},
        "unseen_player": {"positive": 900, "negative": 800},
        "dual_unseen": {"positive": 350, "negative": 220},
    }
    marginal = {
        name: {
            "global": 0.5,
            "mask_only": 0.55,
            "round_mask": 0.7,
            "structure_mask": 0.6,
            "round_structure_mask": 0.72,
            "strongest_auc": 0.72,
        }
        for name in ("unseen_sbox", "unseen_player", "dual_unseen")
    }
    summary = {
        "split_metrics": split_metrics,
        "marginal_baselines": marginal,
        "gate": {
            "decision": "innovation2_small_spn_matched_contrast_ready",
            "metrics": {
                "selected_base_cells": 589,
                "selected_total_label_rows": 9424,
                "distinct_topology_label_patterns": 336,
                "raw_strongest_marginal_auc": {
                    "unseen_sbox": 0.98,
                    "unseen_player": 0.98,
                    "dual_unseen": 0.98,
                },
            },
        },
    }
    output = tmp_path / "matched.svg"
    render_matched_contrast_svg(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E32b" in svg
    assert "train-only选择前后的组外捷径" in svg
    assert "不是实际密码高轮结果" in svg


def test_plot_contains_scope_width_and_shortcut_panels(tmp_path: Path) -> None:
    split_metrics = {
        name: {"positive": 1000, "negative": 1000}
        for name in ("train", "unseen_sbox", "unseen_player", "dual_unseen")
    }
    marginal = {
        name: {
            "global": 0.5,
            "mask_only": 0.55,
            "round_mask": 0.6,
            "structure_mask": 0.62,
            "round_structure_mask": 0.7,
            "strongest_auc": 0.7,
        }
        for name in ("unseen_sbox", "unseen_player", "dual_unseen")
    }
    summary = {
        "split_metrics": split_metrics,
        "marginal_baselines": marginal,
        "gate": {
            "decision": "innovation2_small_spn_exact_label_family_ready",
            "metrics": {
                "total_labels": 8000,
                "positive_labels": 4000,
                "negative_labels": 4000,
                "distinct_label_signatures": 100,
                "cipher_variable_cell_fraction": 0.2,
            },
        },
    }
    output = tmp_path / "curves.svg"
    render_small_spn_exact_label_svg(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E32" in svg
    assert "ID边际基线" in svg
    assert "不是实际密码高轮结果" in svg
