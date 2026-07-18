from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_certificate_guided_pair_residual import (
    E45_PREFIX_AUC,
    build_prefix_ridge_bundle,
    load_e50_sources,
    measure_cgpr_contract,
    train_cgpr_matrix,
    validate_e50_sources,
)


E50_RUN_ID = "i2_present_r4_cgpr_readiness_seed0_20260718"
E50_DECISION = "innovation2_present_cgpr_readiness_passed"


@dataclass(frozen=True)
class CgprAttributionConfig:
    run_id: str
    mode: str = "full"
    epochs: int = 30
    batch_size: int = 32
    hidden_dim: int = 16
    path_rank: int = 2
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    residual_bound: float = 0.25
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.mode != "full"
            or self.epochs != 30
            or self.batch_size != 32
            or self.hidden_dim != 16
            or self.path_rank != 2
            or self.dropout != 0.10
            or self.residual_bound != 0.25
            or self.seed != 0
            or self.device != "cpu"
        ):
            raise ValueError("E51 full protocol is frozen")


def load_e51_sources(
    atlas_root: Path,
    e44_root: Path,
    e45_root: Path,
    e49_root: Path,
    e50_root: Path,
) -> dict[str, Any]:
    sources = load_e50_sources(atlas_root, e44_root, e45_root, e49_root)
    sources["e50_gate"] = json.loads(
        (e50_root / "gate.json").read_text(encoding="utf-8")
    )
    sources["e51_hashes"] = {
        name: _sha256(e50_root / name) for name in ("gate.json", "results.jsonl")
    }
    return sources


def validate_e51_sources(sources: dict[str, Any]) -> dict[str, bool]:
    e50_sources = validate_e50_sources(sources)
    e50_gate = sources["e50_gate"]
    return {
        **{f"e50_source_{key}": value for key, value in e50_sources.items()},
        "e50_run_id_matches": e50_gate.get("run_id") == E50_RUN_ID,
        "e50_decision_matches": e50_gate.get("decision") == E50_DECISION,
        "e50_status_pass": e50_gate.get("status") == "pass",
        "e50_ridge_auc_matches": math.isclose(
            float(e50_gate["metrics"]["ridge_validation_auc"]),
            E45_PREFIX_AUC,
            abs_tol=1e-12,
        ),
        "e50_hashes_present": all(
            len(value) == 64 for value in sources["e51_hashes"].values()
        ),
    }


def train_e51_matrix(
    config: CgprAttributionConfig,
    data: dict[str, Any],
    ridge: dict[str, Any],
) -> dict[str, Any]:
    matrix = train_cgpr_matrix(config, data, ridge)
    for row in matrix["rows"]:
        row["task"] = "innovation2_present_cgpr_attribution"
    return matrix


def adjudicate_e51(
    config: CgprAttributionConfig,
    source_checks: dict[str, bool],
    contract: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    trained = [row for row in matrix["rows"] if row["training_performed"]]
    prefix = next(row for row in trained if row["residual_mode"] == "prefix")
    true = next(
        row
        for row in trained
        if row["residual_mode"] == "pair" and row["topology_mode"] == "true"
    )
    corrupted = next(
        row for row in trained if row["topology_mode"] == "corrupted"
    )
    true_auc = float(true["validation_auc"])
    prefix_auc = float(prefix["validation_auc"])
    corrupted_auc = float(corrupted["validation_auc"])
    protocol_checks = {
        **source_checks,
        "expected_four_rows_present": len(matrix["rows"]) == 4,
        "three_residual_rows_present": len(trained) == 3,
        "ridge_validation_auc_reproduced": math.isclose(
            contract["ridge_validation_auc"], E45_PREFIX_AUC, abs_tol=1e-12
        ),
        "prefix_shape_is_1036x39": contract["prefix_shape"] == [1036, 39],
        "train_standardization_only": contract["train_standardization_only"],
        "zero_residual_matches_ridge": max(
            contract["zero_residual_prefix_max_abs_error"],
            contract["zero_residual_true_max_abs_error"],
            contract["zero_residual_corrupted_max_abs_error"],
        )
        <= 1e-7,
        "ridge_buffers_frozen": contract["ridge_buffers_require_grad_false"]
        and all(float(row["ridge_weight_max_delta"]) == 0.0 for row in trained),
        "true_corrupted_pair_embedding_delta_at_least_1e_5": contract[
            "true_corrupted_pair_embedding_max_abs_difference"
        ]
        >= 1e-5,
        "residual_parameter_relative_spread_at_most_0p01": contract[
            "parameter_relative_spread"
        ]
        <= 0.01,
        "forward_loss_gradients_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "forbidden_buffers_absent": contract["forbidden_buffers_absent"],
        "all_trained_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in trained
            for key in ("train_auc", "validation_auc", "train_loss", "validation_loss")
        ),
        "all_rows_completed_30_epochs": all(
            sum(history["row_id"] == row["row_id"] for history in matrix["history"])
            == config.epochs
            for row in trained
        ),
    }
    candidate_checks = {
        "true_cgpr_auc_at_least_0p70": true_auc >= 0.70,
        "true_cgpr_minus_ridge_at_least_0p02": true_auc - E45_PREFIX_AUC >= 0.02,
    }
    residual_checks = {
        "true_cgpr_minus_prefix_residual_at_least_0p02": true_auc - prefix_auc
        >= 0.02,
    }
    topology_checks = {
        "true_cgpr_minus_corrupted_at_least_0p03": true_auc - corrupted_auc
        >= 0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_cgpr_attribution_protocol_invalid"
        action = "repair source, ridge, zero-equivalence, model, control, or training protocol"
    elif not all(candidate_checks.values()):
        status = "hold"
        decision = "innovation2_present_cgpr_candidate_not_ready"
        action = "stop CGPR and the E43 r4 neural architecture enumeration"
    elif not all(residual_checks.values()):
        status = "hold"
        decision = "innovation2_present_cgpr_pair_residual_not_attributed"
        action = "retain prefix-only residual as a control and stop the pair-state branch"
    elif not all(topology_checks.values()):
        status = "hold"
        decision = "innovation2_present_cgpr_topology_not_attributed"
        action = "withdraw P-layer attribution and stop the pair-state branch"
    else:
        status = "pass"
        decision = "innovation2_present_cgpr_topology_attributed"
        action = "run the frozen E51 matrix with seed1 locally"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "candidate_checks": candidate_checks,
        "residual_attribution_checks": residual_checks,
        "topology_attribution_checks": topology_checks,
        "metrics": {
            "ridge_validation_auc": E45_PREFIX_AUC,
            "prefix_residual_validation_auc": prefix_auc,
            "true_cgpr_validation_auc": true_auc,
            "corrupted_cgpr_validation_auc": corrupted_auc,
            "true_minus_ridge": true_auc - E45_PREFIX_AUC,
            "true_minus_prefix_residual": true_auc - prefix_auc,
            "true_minus_corrupted": true_auc - corrupted_auc,
            "parameter_counts": contract["parameter_counts"],
            "zero_residual_max_abs_error": max(
                contract["zero_residual_prefix_max_abs_error"],
                contract["zero_residual_true_max_abs_error"],
                contract["zero_residual_corrupted_max_abs_error"],
            ),
        },
        "claim_scope": (
            "local seed0 candidate, residual, and topology attribution for CGPR "
            "on the E43 real PRESENT-80 r4 strict benchmark; not a high-round "
            "integral distinguisher, new attack, remote-scale result, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "seed1": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: CgprAttributionConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CgprAttributionConfig",
    "adjudicate_e51",
    "build_prefix_ridge_bundle",
    "load_e51_sources",
    "measure_cgpr_contract",
    "serializable_config",
    "train_e51_matrix",
    "validate_e51_sources",
]
