from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_mspn_readiness import (
    E44_PARAMETER_ANCHOR,
    E44_TRIANGLE_AUC,
    E45_PREFIX_AUC,
    load_readiness_sources,
    measure_mspn_contract,
    train_mspn_row,
    validate_readiness_sources,
)


E46_RUN_ID = "i2_present_r4_mspn_readiness_smoke_seed0_20260718"
E46_DECISION = "innovation2_present_mspn_readiness_passed"


@dataclass(frozen=True)
class MspnAttributionConfig:
    run_id: str
    mode: str = "full"
    epochs: int = 30
    batch_size: int = 32
    hidden_dim: int = 32
    degree_channels: int = 9
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode != "full":
            raise ValueError("E47 only defines full mode")
        if (
            self.epochs != 30
            or self.batch_size != 32
            or self.hidden_dim != 32
            or self.degree_channels != 9
            or self.dropout != 0.10
            or self.seed != 0
            or self.device != "cpu"
        ):
            raise ValueError("E47 full protocol is frozen")


def load_attribution_sources(
    atlas_root: Path,
    e44_root: Path,
    e45_root: Path,
    e46_root: Path,
) -> dict[str, Any]:
    sources = load_readiness_sources(atlas_root, e44_root, e45_root)
    sources["e46_gate"] = json.loads(
        (e46_root / "gate.json").read_text(encoding="utf-8")
    )
    sources["e46_hashes"] = {
        name: _sha256(e46_root / name) for name in ("gate.json", "results.jsonl")
    }
    return sources


def validate_attribution_sources(sources: dict[str, Any]) -> dict[str, bool]:
    readiness = validate_readiness_sources(sources)
    return {
        **{f"readiness_{key}": value for key, value in readiness.items()},
        "e46_run_id_matches": sources["e46_gate"].get("run_id") == E46_RUN_ID,
        "e46_decision_matches": sources["e46_gate"].get("decision") == E46_DECISION,
        "e46_status_pass": sources["e46_gate"].get("status") == "pass",
        "e46_hashes_present": all(
            len(value) == 64 for value in sources["e46_hashes"].values()
        ),
    }


def train_attribution_matrix(
    config: MspnAttributionConfig, data: dict[str, Any]
) -> dict[str, Any]:
    rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_mspn_attribution",
            "row_id": "e45_anf_prefix_ridge_anchor",
            "model_name": "anf_prefix_ridge",
            "topology_mode": "deterministic_prefix",
            "label_mode": "true",
            "seed": 0,
            "best_epoch": 0,
            "train_auc": None,
            "validation_auc": E45_PREFIX_AUC,
            "parameter_count": 0,
            "training_performed": False,
        },
        {
            "run_id": config.run_id,
            "task": "innovation2_present_mspn_attribution",
            "row_id": "e44_triangle_anchor",
            "model_name": "present_pair_triangle",
            "topology_mode": "true",
            "label_mode": "true",
            "seed": 0,
            "best_epoch": 27,
            "train_auc": 0.6113625,
            "validation_auc": E44_TRIANGLE_AUC,
            "parameter_count": E44_PARAMETER_ANCHOR,
            "training_performed": False,
        },
    ]
    history: list[dict[str, Any]] = []
    for topology_mode, label_mode in (
        ("true", "true"),
        ("corrupted", "true"),
        ("true", "shuffled"),
    ):
        output = train_mspn_row(
            config, data, topology_mode=topology_mode, label_mode=label_mode
        )
        output["result"]["task"] = "innovation2_present_mspn_attribution"
        rows.append(output["result"])
        history.extend(output["history"])
    return {"rows": rows, "history": history}


def adjudicate_e47(
    config: MspnAttributionConfig,
    source_checks: dict[str, bool],
    contract: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    trained = [row for row in matrix["rows"] if row["training_performed"]]
    true = next(
        row
        for row in trained
        if row["topology_mode"] == "true" and row["label_mode"] == "true"
    )
    corrupted = next(row for row in trained if row["topology_mode"] == "corrupted")
    shuffled = next(row for row in trained if row["label_mode"] == "shuffled")
    true_auc = float(true["validation_auc"])
    corrupted_auc = float(corrupted["validation_auc"])
    shuffle_auc = float(shuffled["validation_auc"])
    protocol_checks = {
        **source_checks,
        "expected_five_rows_present": len(matrix["rows"]) == 5,
        "three_mspn_rows_present": len(trained) == 3,
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
        "initial_state_shape_is_8x64x32": contract["initial_state_shape"]
        == [8, 64, 32],
        "cell_relabel_error_at_most_1e_6": contract[
            "cell_relabeling_max_abs_logit_error"
        ]
        <= 1e-6,
        "true_corrupted_initial_delta_at_least_1e_5": contract[
            "true_corrupted_max_abs_logit_difference"
        ]
        >= 1e-5,
        "parameter_ratio_in_0p5_2p0": 0.5
        <= contract["parameter_ratio_to_e44"]
        <= 2.0,
        "forward_loss_gradients_finite": contract["logits_finite"]
        and contract["loss_finite"]
        and contract["gradients_finite"],
        "precomputed_certificate_features_absent": contract[
            "precomputed_certificate_feature_buffers_absent"
        ],
        "label_shuffle_auc_in_0p35_0p65": 0.35 <= shuffle_auc <= 0.65,
    }
    candidate_checks = {
        "mspn_true_auc_at_least_0p62": true_auc >= 0.62,
        "mspn_true_minus_e44_at_least_0p04": true_auc - E44_TRIANGLE_AUC >= 0.04,
        "mspn_true_minus_prefix_at_least_minus_0p04": true_auc - E45_PREFIX_AUC
        >= -0.04,
        "mspn_true_minus_shuffle_at_least_0p05": true_auc - shuffle_auc >= 0.05,
    }
    attribution_checks = {
        "mspn_true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_mspn_attribution_protocol_invalid"
        action = "repair source, MSPN, control, metric, or training protocol"
    elif not all(candidate_checks.values()):
        status = "hold"
        decision = "innovation2_present_mspn_candidate_not_ready"
        action = "stop MSPN scaling and audit compressed support-state information loss"
    elif not all(attribution_checks.values()):
        status = "hold"
        decision = "innovation2_present_mspn_topology_not_attributed"
        action = "separate ANF term combination from P-layer transport in a deterministic ablation"
    else:
        status = "pass"
        decision = "innovation2_present_mspn_topology_attributed"
        action = "run the frozen E47 matrix with seed1 locally"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "candidate_checks": candidate_checks,
        "attribution_checks": attribution_checks,
        "metrics": {
            "mspn_true_validation_auc": true_auc,
            "mspn_corrupted_validation_auc": corrupted_auc,
            "mspn_shuffle_validation_auc": shuffle_auc,
            "e44_triangle_validation_auc": E44_TRIANGLE_AUC,
            "e45_prefix_validation_auc": E45_PREFIX_AUC,
            "mspn_true_minus_e44": true_auc - E44_TRIANGLE_AUC,
            "mspn_true_minus_prefix": true_auc - E45_PREFIX_AUC,
            "mspn_true_minus_corrupted": true_auc - corrupted_auc,
            "mspn_true_minus_shuffle": true_auc - shuffle_auc,
            "parameter_count": contract["parameter_count"],
        },
        "claim_scope": (
            "local seed0 candidate and topology attribution for MSPN on the E43 "
            "real PRESENT-80 r4 strict matched benchmark; not a high-round integral "
            "distinguisher, remote-scale result, new attack, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "seed1": status == "pass",
            "remote_scale": False,
        },
    }


def serializable_config(config: MspnAttributionConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
