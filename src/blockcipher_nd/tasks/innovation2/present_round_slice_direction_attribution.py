from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.present_balance_profile_operator import (
    PresentBalanceProfileOperator,
    PresentBalanceProfileOperatorSpec,
)
from blockcipher_nd.tasks.innovation2.present_balance_profile_operator_readiness import (
    _present_player,
    load_profile_operator_sources,
    validate_profile_operator_sources,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    fit_train_only_ridge,
)
from blockcipher_nd.training.metrics import binary_auc


ROUND_SLICES = {"r1": slice(0, 13), "r2": slice(13, 26), "r3": slice(26, 39)}
SEED0_RUN_ID = "i2_present_r4_prefix_guided_profile_operator_attribution_seed0_20260718"
SEED1_RUN_ID = "i2_present_r4_prefix_guided_profile_operator_seed1_20260718"
SEED0_DECISION = "innovation2_present_profile_operator_neural_gain_attributed"
SEED1_DECISION = "innovation2_present_profile_operator_two_seed_confirmed"
EXPECTED_FULL_RIDGE_AUC = 0.7936111111111112


@dataclass(frozen=True)
class RoundSliceDirectionConfig:
    run_id: str
    ridge_lambda: float = 1e-3
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.ridge_lambda != 1e-3 or self.device != "cpu":
            raise ValueError("E72 protocol is frozen")


def load_round_slice_sources(
    profile_root: Path,
    atlas_root: Path,
    seed0_root: Path,
    seed1_root: Path,
) -> dict[str, Any]:
    profile = load_profile_operator_sources(profile_root, atlas_root)
    seed0_gate = json.loads((seed0_root / "gate.json").read_text(encoding="utf-8"))
    seed1_gate = json.loads((seed1_root / "gate.json").read_text(encoding="utf-8"))
    checkpoint_paths = {
        0: seed0_root / "checkpoints" / "profile_true_seed0.pt",
        1: seed1_root / "checkpoints" / "profile_true_seed1.pt",
    }
    return {
        "profile": profile,
        "seed0_gate": seed0_gate,
        "seed1_gate": seed1_gate,
        "checkpoint_paths": checkpoint_paths,
        "hashes": {
            "seed0_gate": _sha256(seed0_root / "gate.json"),
            "seed1_gate": _sha256(seed1_root / "gate.json"),
            "seed0_checkpoint": _sha256(checkpoint_paths[0]),
            "seed1_checkpoint": _sha256(checkpoint_paths[1]),
        },
    }


def validate_round_slice_sources(sources: dict[str, Any]) -> dict[str, bool | float]:
    profile_checks = validate_profile_operator_sources(sources["profile"], strict=True)
    seed0_expected = next(
        row["validation_auc"]
        for row in sources["seed0_gate"]["metrics"]["rows"]
        if row["relation_mode"] == "true"
    )
    seed1_expected = sources["seed1_gate"]["metrics"]["seed_metrics"]["seed1"][
        "true_auc"
    ]
    return {
        **profile_checks,
        "seed0_run_id_matches": sources["seed0_gate"].get("run_id") == SEED0_RUN_ID,
        "seed0_decision_matches": sources["seed0_gate"].get("decision")
        == SEED0_DECISION,
        "seed0_status_pass": sources["seed0_gate"].get("status") == "pass",
        "seed1_run_id_matches": sources["seed1_gate"].get("run_id") == SEED1_RUN_ID,
        "seed1_decision_matches": sources["seed1_gate"].get("decision")
        == SEED1_DECISION,
        "seed1_status_pass": sources["seed1_gate"].get("status") == "pass",
        "all_hashes_present": all(len(value) == 64 for value in sources["hashes"].values()),
        "seed0_expected_auc": float(seed0_expected),
        "seed1_expected_auc": float(seed1_expected),
    }


def evaluate_round_slice_direction(
    config: RoundSliceDirectionConfig,
    sources: dict[str, Any],
) -> dict[str, Any]:
    profile = sources["profile"]
    rows = profile["matched_rows"]
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    train_x = _row_features(profile["prefix_features"], train_rows)
    validation_x = _row_features(profile["prefix_features"], validation_rows)
    train_y = np.asarray([row["label"] for row in train_rows], dtype=np.float64)
    validation_y = np.asarray(
        [row["label"] for row in validation_rows], dtype=np.float64
    )
    ridge_reports: dict[str, float] = {}
    for round_name, columns in ROUND_SLICES.items():
        fitted = fit_train_only_ridge(
            train_x[:, columns],
            train_y,
            validation_x[:, columns],
            config.ridge_lambda,
        )
        ridge_reports[round_name] = _auc(validation_y, fitted["validation_scores"])
    full_ridge = fit_train_only_ridge(
        train_x, train_y, validation_x, config.ridge_lambda
    )
    full_ridge_auc = _auc(validation_y, full_ridge["validation_scores"])

    train_indices = sorted(
        {row["structure_index"] for row in train_rows}
    )
    validation_indices = sorted(
        {row["structure_index"] for row in validation_rows}
    )
    train_mean = profile["prefix_features"][train_indices].mean(axis=0)
    checkpoint_reports: dict[str, Any] = {}
    for seed in (0, 1):
        model = _load_true_model(sources["checkpoint_paths"][seed])
        intact_auc = _checkpoint_auc(
            model, profile, validation_indices, profile["prefix_features"]
        )
        ablated_auc = {}
        drops = {}
        for round_name, columns in ROUND_SLICES.items():
            neutralized = profile["prefix_features"].copy()
            neutralized[:, :, columns] = train_mean[None, :, columns]
            ablated_auc[round_name] = _checkpoint_auc(
                model, profile, validation_indices, neutralized
            )
            drops[round_name] = intact_auc - ablated_auc[round_name]
        checkpoint_reports[f"seed{seed}"] = {
            "intact_auc": intact_auc,
            "neutralized_auc": ablated_auc,
            "ablation_drop": drops,
        }
    return {
        "ridge_auc": ridge_reports,
        "full_ridge_auc": full_ridge_auc,
        "checkpoints": checkpoint_reports,
        "train_mean_source_structures": len(train_indices),
        "validation_structures": len(validation_indices),
    }


def adjudicate_round_slice_direction(
    config: RoundSliceDirectionConfig,
    source_checks: dict[str, bool | float],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    ridge = evaluation["ridge_auc"]
    checkpoint_reports = evaluation["checkpoints"]
    ridge_order = sorted(ridge, key=ridge.get, reverse=True)
    seed_orders = {
        seed: sorted(
            report["ablation_drop"],
            key=report["ablation_drop"].get,
            reverse=True,
        )
        for seed, report in checkpoint_reports.items()
    }
    dominant_round = ridge_order[0]
    consensus = all(order[0] == dominant_round for order in seed_orders.values())
    expected = {
        "seed0": float(source_checks["seed0_expected_auc"]),
        "seed1": float(source_checks["seed1_expected_auc"]),
    }
    protocol_checks = {
        **{
            key: value
            for key, value in source_checks.items()
            if isinstance(value, bool)
        },
        "full_ridge_replays_e65": math.isclose(
            evaluation["full_ridge_auc"],
            EXPECTED_FULL_RIDGE_AUC,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "seed0_checkpoint_auc_replays": math.isclose(
            checkpoint_reports["seed0"]["intact_auc"],
            expected["seed0"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "seed1_checkpoint_auc_replays": math.isclose(
            checkpoint_reports["seed1"]["intact_auc"],
            expected["seed1"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for value in (
                list(ridge.values())
                + [evaluation["full_ridge_auc"]]
                + [
                    metric
                    for report in checkpoint_reports.values()
                    for family in ("neutralized_auc", "ablation_drop")
                    for metric in report[family].values()
                ]
            )
        ),
    }
    direction_checks = {
        "ridge_and_both_seeds_share_dominant_round": consensus,
        "ridge_top_minus_second_at_least_0p03": ridge[ridge_order[0]]
        - ridge[ridge_order[1]]
        >= 0.03,
        **{
            f"{seed}_dominant_drop_at_least_0p03": report["ablation_drop"].get(
                dominant_round, -math.inf
            )
            >= 0.03
            for seed, report in checkpoint_reports.items()
        },
        **{
            f"{seed}_top_minus_second_drop_at_least_0p02": report[
                "ablation_drop"
            ][order[0]]
            - report["ablation_drop"][order[1]]
            >= 0.02
            for seed, (report, order) in (
                (seed, (checkpoint_reports[seed], seed_orders[seed]))
                for seed in checkpoint_reports
            )
        },
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_round_slice_protocol_invalid"
        action = "repair E65/E67/E68 replay or slice-neutralization protocol"
    elif all(direction_checks.values()) and dominant_round == "r1":
        status = "pass"
        decision = "innovation2_present_early_round_skip_candidate_ready"
        action = "pre-register a same-budget early-round skip/gated residual readiness"
    else:
        status = "hold"
        decision = "innovation2_present_round_direction_not_confirmed"
        action = "retain E68 and stop the round-recurrent branch"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "direction_checks": direction_checks,
        "metrics": {
            **evaluation,
            "ridge_order": ridge_order,
            "checkpoint_orders": seed_orders,
            "dominant_round": dominant_round,
            "consensus": consensus,
        },
        "claim_scope": (
            "no-training round-slice attribution for two confirmed PRESENT-80 r4 "
            "8-bit-cube unit-profile checkpoints; no new neural gain, high-round, "
            "cross-dimension, attack, remote-scale, or SOTA claim"
        ),
        "next_action": {"action": action, "neural_readiness": status == "pass", "remote_scale": False},
    }


def result_rows_for_round_slice(
    config: RoundSliceDirectionConfig, gate: dict[str, Any]
) -> list[dict[str, Any]]:
    metrics = gate["metrics"]
    return [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_round_slice_direction_attribution",
            "round": round_name,
            "ridge_auc": metrics["ridge_auc"][round_name],
            "seed0_ablation_drop": metrics["checkpoints"]["seed0"][
                "ablation_drop"
            ][round_name],
            "seed1_ablation_drop": metrics["checkpoints"]["seed1"][
                "ablation_drop"
            ][round_name],
            "status": gate["status"],
            "decision": gate["decision"],
            "training_performed": False,
        }
        for round_name in ROUND_SLICES
    ]


def _row_features(features: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [features[row["structure_index"], row["output_bit"]] for row in rows],
        dtype=np.float64,
    )


def _load_true_model(checkpoint: Path) -> PresentBalanceProfileOperator:
    model = PresentBalanceProfileOperator(
        PresentBalanceProfileOperatorSpec(
            input_dim=39,
            hidden_dim=32,
            steps=2,
            dropout=0.10,
            relation_mode="true",
        ),
        torch.from_numpy(_present_player()),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _checkpoint_auc(
    model: PresentBalanceProfileOperator,
    profile: dict[str, Any],
    indices: list[int],
    features: np.ndarray,
) -> float:
    selected_features = torch.from_numpy(features[indices].astype(np.float32))
    targets = torch.from_numpy(profile["profile_targets"][indices].astype(np.float32))
    observed = torch.from_numpy(profile["profile_observed"][indices])
    with torch.no_grad():
        logits = model(selected_features)
    return _auc(targets[observed].numpy(), logits[observed].numpy())


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


def serializable_config(config: RoundSliceDirectionConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ROUND_SLICES",
    "RoundSliceDirectionConfig",
    "adjudicate_round_slice_direction",
    "evaluate_round_slice_direction",
    "load_round_slice_sources",
    "result_rows_for_round_slice",
    "serializable_config",
    "validate_round_slice_sources",
]
