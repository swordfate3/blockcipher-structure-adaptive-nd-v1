from __future__ import annotations

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_context_group_split_readiness import (
    ContextGroupSplitConfig,
    adjudicate_context_group_split,
    grouped_ridge_predictions,
    make_class_complete_group_folds,
    make_group_folds,
    run_context_group_split_audit,
)


def test_group_folds_are_deterministic_and_balanced() -> None:
    contexts = make_group_folds(tuple(range(16)), folds=4, seed=6101)
    masks = make_group_folds(tuple(range(32)), folds=4, seed=6201)

    assert contexts == make_group_folds(tuple(range(16)), folds=4, seed=6101)
    assert sorted(list(contexts.values()).count(fold) for fold in range(4)) == [
        4,
        4,
        4,
        4,
    ]
    assert sorted(list(masks.values()).count(fold) for fold in range(4)) == [
        8,
        8,
        8,
        8,
    ]


def test_dual_group_predictions_assign_every_row_once() -> None:
    context_ids = [context_id for context_id in range(16) for _ in range(32)]
    mask_ids = [mask_id for _ in range(16) for mask_id in range(32)]
    labels = np.asarray(
        [(context_id + mask_id) % 2 for context_id, mask_id in zip(context_ids, mask_ids)],
        dtype=np.float64,
    )
    design = np.ones((len(labels), 3), dtype=np.float64)
    design[:, 1] = np.asarray(context_ids) % 2
    design[:, 2] = np.asarray(mask_ids) % 2

    predictions, diagnostics = grouped_ridge_predictions(
        design,
        labels,
        context_ids=context_ids,
        mask_ids=mask_ids,
        context_folds=make_group_folds(tuple(range(16)), folds=4, seed=6101),
        mask_folds=make_group_folds(tuple(range(32)), folds=4, seed=6201),
        mode="dual",
        alpha=1.0,
    )

    assert predictions.shape == labels.shape
    assert diagnostics["all_rows_assigned_once"] is True
    assert diagnostics["minimum_assignments"] == 1
    assert diagnostics["maximum_assignments"] == 1


def test_class_complete_fold_search_handles_e17b_incidence_patterns() -> None:
    context_ids = [context_id for context_id in range(16) for _ in range(32)]
    mask_ids = [mask_id for _ in range(16) for mask_id in range(32)]
    labels = np.asarray(
        [
            int(
                context_id
                in (
                    {5, 10, 12, 14}
                    if mask_id < 16
                    else {7, 12, 13, 14}
                )
            )
            for context_id, mask_id in zip(context_ids, mask_ids, strict=True)
        ],
        dtype=np.float64,
    )

    context_folds, mask_folds, attempts = make_class_complete_group_folds(
        labels,
        context_ids=context_ids,
        mask_ids=mask_ids,
        folds=4,
        seed=6401,
    )
    _, diagnostics = grouped_ridge_predictions(
        np.ones((len(labels), 1), dtype=np.float64),
        labels,
        context_ids=context_ids,
        mask_ids=mask_ids,
        context_folds=context_folds,
        mask_folds=mask_folds,
        mode="dual",
        alpha=1.0,
    )

    assert attempts >= 1
    assert diagnostics["all_train_test_splits_have_both_classes"] is True


def test_group_split_gate_holds_for_generalizing_shortcut() -> None:
    config = ContextGroupSplitConfig(run_id="test")
    rows = [
        {
            "baseline": key,
            "accuracy": 0.75,
            "brier": 0.2,
            "auc": auc,
            "directional_auc": max(auc, 1.0 - auc),
        }
        for key, auc in (
            ("context_disjoint_bitwise", 0.80),
            ("mask_disjoint_bitwise", 0.60),
            ("dual_disjoint_bitwise", 0.60),
            ("label_shuffle_dual_disjoint", 0.50),
        )
    ]

    gate = adjudicate_context_group_split(config, rows, {"ok": True})

    assert gate["status"] == "hold"
    assert gate["shortcut_checks"][
        "context_disjoint_directional_auc_below_0p75"
    ] is False


def test_group_split_runner_returns_complete_result() -> None:
    label_rows = [
        {
            "context_id": str(context_id),
            "context_value": str(context_id * 17),
            "mask_index": str(mask_id),
            "mask_value": str(1 << (mask_id % 64)),
            "balanced_label": str((context_id + mask_id) % 2),
        }
        for context_id in range(16)
        for mask_id in range(32)
    ]
    result = run_context_group_split_audit(
        ContextGroupSplitConfig(run_id="runner"),
        source_gate={
            "run_id": "source",
            "status": "hold",
            "decision": (
                "innovation2_equal_prevalence_context_label_shortcut_dominated"
            ),
        },
        source_metadata={
            "task": "innovation2_equal_prevalence_context_mask_label_readiness",
            "training_performed": False,
            "contexts": 16,
            "candidate_masks": 32,
            "label_rows": 512,
        },
        source_label_rows=label_rows,
    )

    assert len(result["rows"]) == 5
    assert result["metadata"]["rows"] == 512
    assert result["metadata"]["diagnostics"]["dual_disjoint"][
        "all_rows_assigned_once"
    ] is True
