from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1w import (
    build_k1w_control,
    read_tasks,
    task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1x import (
    K1W_CHECKPOINTS,
    K1W_CHECKPOINT_DIGESTS,
    PLAN,
    TRAIN_CACHES,
    VALIDATION_CACHES,
    load_cache,
    load_state,
    read_json,
)
from blockcipher_nd.training.data import make_loader
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_ctspn_compact_branch_interference_k1z_20260728"
K1X_ROOT = ROOT / (
    "outputs/local_audit/"
    "i1_uknit_family_ctspn_compact_optimization_geometry_k1x_20260728"
)
EXPECTED_SEEDS = (3, 4)
ALPHAS = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
EXPECTED_GRID_ROWS = len(EXPECTED_SEEDS) * len(ALPHAS)
EXPECTED_CONFIRMATION_ROWS = len(EXPECTED_SEEDS) * 4
EXPECTED_BATCH_SIZE = 64
REPLAY_TOLERANCE = 1e-6
SEMANTIC_MARGIN = 0.010
ANCHOR_TOLERANCE = 0.020
ANCHOR_AUCS = {3: 0.5654244422912598, 4: 0.5940475463867188}
SOURCE_PATHS = {
    "k1x_gate": K1X_ROOT / "gate.json",
    "k1x_results": K1X_ROOT / "results.jsonl",
    "k1x_gradients": K1X_ROOT / "gradients.jsonl",
}
SOURCE_DIGESTS = {
    "k1x_gate": "ceae8bca25b0b3a9af034d02898d1233c491b4865f8a28e7cebfb1489f17b0d9",
    "k1x_results": "42dae07903565af850470a3b9f1c0d600d5f0e5256a35a4bc6257edcc7988a58",
    "k1x_gradients": "a00a0099d01049909ca6a07242915e4612aefe48aff1996647d0426fb8670a8d",
}


def source_binding_checks() -> dict[str, bool]:
    checks = {
        f"{name}_sha256_exact": path.is_file()
        and file_sha256(path) == SOURCE_DIGESTS[name]
        for name, path in SOURCE_PATHS.items()
    }
    checks.update(
        {
            f"k1w_seed{seed}_checkpoint_sha256_exact": path.is_file()
            and file_sha256(path) == K1W_CHECKPOINT_DIGESTS[seed]
            for seed, path in K1W_CHECKPOINTS.items()
        }
    )
    checks.update(
        {
            f"seed{seed}_source_caches_exist": all(
                (cache / name).is_file()
                for cache in (TRAIN_CACHES[seed], VALIDATION_CACHES[seed])
                for name in ("features.npy", "labels.npy", "metadata.json")
            )
            for seed in EXPECTED_SEEDS
        }
    )
    if checks and all(checks.values()):
        gate = read_json(SOURCE_PATHS["k1x_gate"])
        checks["k1x_hold_decision_exact"] = (
            gate.get("status") == "hold"
            and gate.get("decision")
            == "innovation1_uknit_family_ctspn_k1x_optimization_geometry_not_sufficient"
            and gate.get("remote_scale") == "no"
        )
    return checks


def run_audit(*, device: str = "cpu") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = task_map(read_tasks(PLAN))
    k1x_gate = read_json(SOURCE_PATHS["k1x_gate"])
    grid_rows: list[dict[str, Any]] = []
    confirmation_rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        task = tasks[("uknit64", seed, "compact_exact")]
        state = load_state(K1W_CHECKPOINTS[seed])
        exact = build_k1w_control(
            task=task,
            condition="compact_exact",
            input_bits=512,
        )
        wrong = build_k1w_control(
            task=task,
            condition="compact_wrong_sbox",
            input_bits=512,
        )
        exact.load_state_dict(state, strict=True)
        wrong.load_state_dict(state, strict=True)
        source_state_sha = tensor_mapping_sha256(state)
        exact_state_before = tensor_mapping_sha256(exact.state_dict())
        wrong_state_before = tensor_mapping_sha256(wrong.state_dict())
        if exact_state_before != source_state_sha or wrong_state_before != source_state_sha:
            raise ValueError("K1-Z strict state load changed the learned tensors")

        train_metrics = evaluate_alpha_grid(
            exact,
            load_cache(TRAIN_CACHES[seed]),
            alphas=ALPHAS,
            device=device,
        )
        selected_alpha = select_alpha(train_metrics)
        for alpha in ALPHAS:
            grid_rows.append(
                {
                    "run_id": RUN_ID,
                    "cipher_key": "uknit64",
                    "rounds": 5,
                    "seed": seed,
                    "split": "train_seen_discovery",
                    "semantics": "exact_sbox",
                    "alpha": alpha,
                    "selected": alpha == selected_alpha,
                    "selection_rule": "max_train_auc_then_min_abs_alpha_minus_1_then_alpha",
                    **train_metrics[alpha],
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )

        validation = load_cache(VALIDATION_CACHES[seed])
        exact_metrics = evaluate_alpha_grid(
            exact,
            validation,
            alphas=(0.0, 1.0, selected_alpha),
            device=device,
        )
        wrong_metrics = evaluate_alpha_grid(
            wrong,
            validation,
            alphas=(selected_alpha,),
            device=device,
        )
        source_auc = float(
            k1x_gate["seed_results"][str(seed)]["k1w_exact_auc"]
        )
        common = {
            "run_id": RUN_ID,
            "cipher_key": "uknit64",
            "rounds": 5,
            "seed": seed,
            "split": "cross_key_validation_confirmation",
            "selected_alpha": selected_alpha,
            "source_alpha1_auc": source_auc,
            "strict_state_dict_load": True,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        confirmation_rows.extend(
            (
                {
                    **common,
                    "condition": "exact_selected",
                    "semantics": "exact_sbox",
                    "alpha": selected_alpha,
                    **exact_metrics[selected_alpha],
                },
                {
                    **common,
                    "condition": "wrong_sbox_selected",
                    "semantics": "wrong_sbox_same_checkpoint",
                    "alpha": selected_alpha,
                    **wrong_metrics[selected_alpha],
                },
                {
                    **common,
                    "condition": "exact_alpha0",
                    "semantics": "exact_sbox",
                    "alpha": 0.0,
                    **exact_metrics[0.0],
                },
                {
                    **common,
                    "condition": "exact_alpha1",
                    "semantics": "exact_sbox",
                    "alpha": 1.0,
                    **exact_metrics[1.0],
                },
            )
        )
        exact_state_after = tensor_mapping_sha256(exact.state_dict())
        wrong_state_after = tensor_mapping_sha256(wrong.state_dict())
        for row in confirmation_rows[-4:]:
            row["state_sha256"] = source_state_sha
            row["state_unchanged"] = (
                exact_state_before == exact_state_after == source_state_sha
                and wrong_state_before == wrong_state_after == source_state_sha
            )
    return grid_rows, confirmation_rows


def evaluate_alpha_grid(
    model: torch.nn.Module,
    dataset: Any,
    *,
    alphas: Sequence[float],
    device: str,
) -> dict[float, dict[str, float]]:
    selected = torch.device(device)
    model = model.to(selected).eval()
    chunks: dict[float, list[np.ndarray]] = {float(alpha): [] for alpha in alphas}
    with torch.inference_mode():
        for features, _labels in make_loader(
            dataset,
            batch_size=EXPECTED_BATCH_SIZE,
            shuffle=False,
        ):
            features = features.to(selected)
            base, histogram = branch_embeddings(model, features)
            gate = torch.tanh(model.backbone.histogram_gate)
            for alpha in chunks:
                logits = model.backbone.base.classifier(
                    base + alpha * gate * histogram
                ).squeeze(1)
                chunks[alpha].append(torch.sigmoid(logits).cpu().numpy())
    labels = np.asarray(dataset.labels, dtype=np.float32)
    results: dict[float, dict[str, float]] = {}
    for alpha, values in chunks.items():
        probabilities = np.concatenate(values).astype(np.float32, copy=False)
        results[alpha] = {
            "auc": binary_auc(labels, probabilities),
            "accuracy": float(((probabilities >= 0.5) == labels).mean()),
        }
    return results


def branch_embeddings(
    model: torch.nn.Module,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime = features.reshape(
        features.shape[0],
        -1,
        2,
        model.runtime_structure.block_bits,
    ).flip(-1)
    base = model.backbone.base.encode(runtime, model.runtime_structure)
    edge = model.backbone.edge_residual_embedding(
        runtime,
        model.runtime_structure,
        apply_sboxes=model.apply_sboxes,
    )
    base = base + torch.tanh(model.backbone.residual_gate) * torch.tanh(edge)
    histogram = model.backbone.histogram_embedding(
        runtime,
        model.runtime_structure,
        apply_sboxes=model.apply_sboxes,
    )
    return base, torch.tanh(histogram.repeat(1, 3))


def select_alpha(metrics: Mapping[float, Mapping[str, float]]) -> float:
    expected = set(ALPHAS)
    if set(metrics) != expected:
        raise ValueError("K1-Z alpha grid is incomplete")
    return min(
        ALPHAS,
        key=lambda alpha: (
            -float(metrics[alpha]["auc"]),
            abs(alpha - 1.0),
            alpha,
        ),
    )


def adjudicate(
    *,
    grid_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
    source_checks: Mapping[str, bool],
) -> dict[str, Any]:
    grid = grid_map(grid_rows)
    confirmation = confirmation_map(confirmation_rows)
    expected_grid = {(seed, alpha) for seed in EXPECTED_SEEDS for alpha in ALPHAS}
    expected_confirmation = {
        (seed, condition)
        for seed in EXPECTED_SEEDS
        for condition in (
            "exact_selected",
            "wrong_sbox_selected",
            "exact_alpha0",
            "exact_alpha1",
        )
    }
    selection_checks: dict[int, bool] = {}
    for seed in EXPECTED_SEEDS:
        metrics = {
            alpha: {"auc": float(grid[(seed, alpha)]["auc"])}
            for alpha in ALPHAS
            if (seed, alpha) in grid
        }
        selected_rows = [
            row
            for row in grid_rows
            if int(row.get("seed", -1)) == seed and row.get("selected") is True
        ]
        selected = (
            float(selected_rows[0]["alpha"]) if len(selected_rows) == 1 else math.nan
        )
        confirmation_selected = confirmation.get((seed, "exact_selected"), {}).get(
            "selected_alpha"
        )
        try:
            confirmation_alpha = float(confirmation_selected)
        except (TypeError, ValueError):
            confirmation_alpha = math.nan
        selection_checks[seed] = (
            set(metrics) == set(ALPHAS)
            and selected == select_alpha(metrics)
            and confirmation_alpha == selected
        )
    replay_errors = {
        str(seed): abs(
            float(confirmation.get((seed, "exact_alpha1"), {}).get("auc", math.nan))
            - float(
                confirmation.get((seed, "exact_alpha1"), {}).get(
                    "source_alpha1_auc", math.nan
                )
            )
        )
        for seed in EXPECTED_SEEDS
    }
    protocol_checks = {
        **dict(source_checks),
        "twenty_four_grid_rows_exact": len(grid_rows) == EXPECTED_GRID_ROWS
        and set(grid) == expected_grid,
        "eight_confirmation_rows_exact": len(confirmation_rows)
        == EXPECTED_CONFIRMATION_ROWS
        and set(confirmation) == expected_confirmation,
        "train_only_alpha_selection_exact": all(selection_checks.values()),
        "alpha1_source_auc_replay": max(replay_errors.values(), default=math.inf)
        <= REPLAY_TOLERANCE,
        "finite_metrics": bool(grid_rows) and bool(confirmation_rows)
        and all(
            math.isfinite(float(row.get(metric, math.nan)))
            for row in (*grid_rows, *confirmation_rows)
            for metric in ("auc", "accuracy")
        ),
        "strict_state_unchanged_zero_steps": all(
            row.get("state_unchanged") is True
            and row.get("strict_state_dict_load") is True
            and row.get("training_performed") is False
            and row.get("optimizer_steps") == 0
            for row in confirmation_rows
        ),
    }
    research_checks: dict[str, bool] = {}
    seed_results: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        exact = confirmation.get((seed, "exact_selected"), {})
        wrong = confirmation.get((seed, "wrong_sbox_selected"), {})
        exact_auc = float(exact.get("auc", math.nan))
        wrong_auc = float(wrong.get("auc", math.nan))
        threshold = ANCHOR_AUCS[seed] - ANCHOR_TOLERANCE
        research_checks[f"seed{seed}_selected_alpha_retains_anchor"] = (
            exact_auc >= threshold
        )
        research_checks[f"seed{seed}_selected_alpha_beats_wrong_sbox"] = (
            exact_auc - wrong_auc >= SEMANTIC_MARGIN
        )
        seed_results[str(seed)] = {
            "selected_alpha": exact.get("selected_alpha"),
            "train_selected_auc": next(
                (
                    float(row["auc"])
                    for row in grid_rows
                    if int(row["seed"]) == seed and row.get("selected") is True
                ),
                math.nan,
            ),
            "validation_selected_exact_auc": exact_auc,
            "validation_selected_wrong_sbox_auc": wrong_auc,
            "validation_exact_minus_wrong_sbox": exact_auc - wrong_auc,
            "validation_alpha0_auc": confirmation.get(
                (seed, "exact_alpha0"), {}
            ).get("auc"),
            "validation_alpha1_auc": confirmation.get(
                (seed, "exact_alpha1"), {}
            ).get("auc"),
            "anchor_auc": ANCHOR_AUCS[seed],
            "retention_threshold": threshold,
            "selected_minus_anchor": exact_auc - ANCHOR_AUCS[seed],
        }
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    research_valid = bool(research_checks) and all(research_checks.values())
    if not protocol_valid:
        status = "invalid"
        decision = "innovation1_uknit_family_ctspn_k1z_protocol_invalid"
        next_action = "repair only the source, cached-forward, selection, replay, or state-restoration failure"
    elif research_valid:
        status = "pass"
        decision = (
            "innovation1_uknit_family_ctspn_k1z_recoverable_signal_fusion_failure"
        )
        next_action = (
            "compare one explicit late-logit fusion design against unchanged K1-W at the same local budget"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_ctspn_k1z_inference_rescaling_insufficient_optimization_unresolved"
        )
        next_action = (
            "preregister K1-Y changing only the compact projection weight learning-rate multiplier to 16x"
        )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "remote_scale": "no",
        "protocol_checks": protocol_checks,
        "failed_protocol_checks": sorted(
            name for name, passed in protocol_checks.items() if not passed
        ),
        "research_checks": research_checks,
        "failed_research_checks": sorted(
            name for name, passed in research_checks.items() if not passed
        ),
        "source_replay_errors": replay_errors,
        "seed_results": seed_results,
        "thresholds": {
            "anchor_tolerance": ANCHOR_TOLERANCE,
            "semantic_margin": SEMANTIC_MARGIN,
        },
        "next_action": next_action,
        "claim_scope": (
            "two-seed local zero-training train-selected/cross-key-confirmed branch-scale audit; "
            "not formal scale, attack, SOTA, learned calibration, transfer, or family ceiling"
        ),
        "blocked_actions": [
            "remote scale, sixteen pairs, or projection learning-rate tuning",
            "validation-selected alpha or averaging seeds",
            "changing data, epochs, labels, negatives, differences, or cipher identity",
        ],
    }


def grid_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, float], Mapping[str, Any]]:
    mapped: dict[tuple[int, float], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), float(row["alpha"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-Z grid row: {key}")
        mapped[key] = row
    return mapped


def confirmation_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    mapped: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["condition"]))
        if key in mapped:
            raise ValueError(f"duplicate K1-Z confirmation row: {key}")
        mapped[key] = row
    return mapped


def comparison_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"seed": int(seed), **dict(result)}
        for seed, result in sorted(gate.get("seed_results", {}).items())
    ]


__all__ = [
    "ALPHAS",
    "EXPECTED_CONFIRMATION_ROWS",
    "EXPECTED_GRID_ROWS",
    "RUN_ID",
    "adjudicate",
    "branch_embeddings",
    "comparison_rows",
    "evaluate_alpha_grid",
    "run_audit",
    "select_alpha",
    "source_binding_checks",
]
