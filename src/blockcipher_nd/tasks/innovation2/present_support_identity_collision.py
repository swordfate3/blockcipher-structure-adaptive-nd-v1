from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    fit_train_only_ridge,
)
from blockcipher_nd.tasks.innovation2.present_pair_state_neural_attribution import (
    load_e43_source,
    validate_e43_source,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    possible_active_monomials,
)
from blockcipher_nd.training.metrics import binary_auc


E45_RUN_ID = "i2_present_r4_certificate_complexity_attribution_20260718"
E45_DECISION = "innovation2_present_mspn_route_ready"
E47_RUN_ID = "i2_present_r4_mspn_neural_attribution_seed0_20260718"
E47_DECISION = "innovation2_present_mspn_candidate_not_ready"
SKETCH_SEED = 48001
RIDGE_LAMBDA = 1e-3
SKETCH_WIDTHS = (16, 32, 64)


@dataclass(frozen=True)
class SupportIdentityAuditConfig:
    run_id: str
    mode: str = "audit"
    sketch_seed: int = SKETCH_SEED
    ridge_lambda: float = RIDGE_LAMBDA

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.sketch_seed <= 0 or self.ridge_lambda <= 0:
            raise ValueError("sketch seed and ridge lambda must be positive")
        if self.mode == "audit" and (
            self.sketch_seed != SKETCH_SEED
            or self.ridge_lambda != RIDGE_LAMBDA
        ):
            raise ValueError("E48 audit protocol is frozen")


def load_e48_sources(
    atlas_root: Path, e45_root: Path, e47_root: Path
) -> dict[str, Any]:
    atlas = load_e43_source(atlas_root)
    e45_gate = _read_json(e45_root / "gate.json")
    e47_gate = _read_json(e47_root / "gate.json")
    return {
        "atlas": atlas,
        "e45_gate": e45_gate,
        "e47_gate": e47_gate,
        "source_hashes": {
            "e45_gate.json": _sha256(e45_root / "gate.json"),
            "e45_results.jsonl": _sha256(e45_root / "results.jsonl"),
            "e47_gate.json": _sha256(e47_root / "gate.json"),
            "e47_results.jsonl": _sha256(e47_root / "results.jsonl"),
        },
    }


def validate_e48_sources(sources: dict[str, Any], *, strict: bool) -> dict[str, bool]:
    atlas_checks = validate_e43_source(sources["atlas"], strict=strict)
    e45 = sources["e45_gate"]
    e47 = sources["e47_gate"]
    return {
        **{f"atlas_{key}": value for key, value in atlas_checks.items()},
        "e45_run_id_matches": e45.get("run_id") == E45_RUN_ID,
        "e45_decision_matches": e45.get("decision") == E45_DECISION,
        "e45_status_pass": e45.get("status") == "pass",
        "e47_run_id_matches": e47.get("run_id") == E47_RUN_ID,
        "e47_decision_matches": e47.get("decision") == E47_DECISION,
        "e47_status_hold": e47.get("status") == "hold",
        "e47_true_auc_matches": math.isclose(
            float(e47["metrics"]["mspn_true_validation_auc"]),
            0.5186727951738006,
            abs_tol=1e-12,
        ),
        "source_hashes_present": all(
            len(value) == 64 for value in sources["source_hashes"].values()
        ),
    }


def build_support_identity_table(
    config: SupportIdentityAuditConfig, sources: dict[str, Any]
) -> dict[str, Any]:
    atlas = sources["atlas"]
    rows = atlas["rows"]
    structures = atlas["structures"]
    true_player = atlas["players"][0]
    corrupted_player = topology_players(atlas["players"], "corrupted")[0]
    structure_indices = sorted({row["structure_index"] for row in rows})
    true_supports = {
        structure_index: {
            rounds: possible_active_monomials(
                tuple(structures[structure_index]["active_bits"]),
                rounds,
                player=true_player,
            )
            for rounds in (1, 2, 3)
        }
        for structure_index in structure_indices
    }
    corrupted_supports = {
        structure_index: {
            rounds: possible_active_monomials(
                tuple(structures[structure_index]["active_bits"]),
                rounds,
                player=corrupted_player,
            )
            for rounds in (1, 2, 3)
        }
        for structure_index in structure_indices
    }
    local_permutations = {
        structure_index: np.random.default_rng(
            config.sketch_seed + 1000 + structure_index
        ).permutation(8)
        for structure_index in structure_indices
    }
    exact = np.zeros((len(rows), 3 * 256), dtype=np.float64)
    permuted = np.zeros_like(exact)
    corrupted = np.zeros_like(exact)
    degree = np.zeros((len(rows), 3 * 9), dtype=np.float64)
    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        structure_index = int(row["structure_index"])
        mask_index = int(row["mask_index"])
        selected = np.flatnonzero(atlas["output_mask_bits"][mask_index])
        exact[row_index] = support_identity_vector(
            selected, true_supports[structure_index]
        )
        corrupted[row_index] = support_identity_vector(
            selected, corrupted_supports[structure_index]
        )
        permuted[row_index] = permute_identity_vector(
            exact[row_index], local_permutations[structure_index]
        )
        degree[row_index] = degree_only_vector(exact[row_index])
        output_rows.append(
            {
                "split": row["split"],
                "structure_index": structure_index,
                "mask_index": mask_index,
                "label": int(row["label"]),
                "exact_identity_sha256": hashlib.sha256(
                    np.rint(exact[row_index] * 4).astype(np.uint8).tobytes()
                ).hexdigest(),
            }
        )
    rng = np.random.default_rng(config.sketch_seed)
    rademacher = rng.choice(
        np.asarray([-1.0, 1.0]), size=(3 * 256, max(SKETCH_WIDTHS))
    )
    binary_projection = (rademacher > 0).astype(np.uint8)
    sketches: dict[str, np.ndarray] = {}
    binary_sketches: dict[str, np.ndarray] = {}
    for width in SKETCH_WIDTHS:
        sketches[f"sketch{width}"] = (
            exact @ rademacher[:, :width] / math.sqrt(exact.shape[1])
        )
        incidence = (exact > 0).astype(np.uint8)
        binary_sketches[f"sketch{width}"] = (
            incidence @ binary_projection[:, :width]
        ) % 2
    sketches["permuted_sketch64"] = (
        permuted @ rademacher[:, :64] / math.sqrt(exact.shape[1])
    )
    sketches["corrupted_sketch64"] = (
        corrupted @ rademacher[:, :64] / math.sqrt(exact.shape[1])
    )
    return {
        "rows": output_rows,
        "degree": degree,
        "exact": exact,
        "permuted": permuted,
        "corrupted": corrupted,
        "sketches": sketches,
        "binary_sketches": binary_sketches,
        "true_player": true_player,
        "corrupted_player": corrupted_player,
        "rademacher_sha256": hashlib.sha256(rademacher.tobytes()).hexdigest(),
        "binary_projection_sha256": hashlib.sha256(
            binary_projection.tobytes()
        ).hexdigest(),
    }


def support_identity_vector(
    selected: np.ndarray,
    supports_by_round: dict[int, tuple[frozenset[int], ...]],
) -> np.ndarray:
    output = np.zeros(3 * 256, dtype=np.float64)
    denominator = max(1, len(selected))
    for round_index, rounds in enumerate((1, 2, 3)):
        for bit in selected:
            for monomial in supports_by_round[rounds][int(bit)]:
                output[round_index * 256 + monomial] += 1.0 / denominator
    return output


def degree_only_vector(identity: np.ndarray) -> np.ndarray:
    output = np.zeros(3 * 9, dtype=np.float64)
    degrees = np.asarray([mask.bit_count() for mask in range(256)], dtype=np.int64)
    for round_index in range(3):
        block = identity[round_index * 256 : (round_index + 1) * 256]
        for degree in range(9):
            output[round_index * 9 + degree] = float(block[degrees == degree].sum())
    return output


def permute_identity_vector(identity: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    output = np.zeros_like(identity)
    mapping = np.asarray(
        [
            sum(
                1 << int(permutation[bit])
                for bit in range(8)
                if monomial & (1 << bit)
            )
            for monomial in range(256)
        ],
        dtype=np.int64,
    )
    for round_index in range(3):
        source = identity[round_index * 256 : (round_index + 1) * 256]
        output[round_index * 256 + mapping] = source
    return output


def evaluate_support_identity(
    config: SupportIdentityAuditConfig, table: dict[str, Any]
) -> dict[str, Any]:
    split = np.asarray([row["split"] for row in table["rows"]])
    labels = np.asarray([row["label"] for row in table["rows"]], dtype=np.float64)
    train = split == "train"
    validation = split == "validation"
    feature_matrices = {
        "degree_only": table["degree"],
        "exact_identity": table["exact"],
        **table["sketches"],
    }
    reports: dict[str, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    for family, matrix in feature_matrices.items():
        fitted = fit_train_only_ridge(
            matrix[train], labels[train], matrix[validation], config.ridge_lambda
        )
        report = {
            "feature_count": matrix.shape[1],
            "train_auc": _safe_auc(labels[train], fitted["train_scores"]),
            "validation_auc": _safe_auc(
                labels[validation], fitted["validation_scores"]
            ),
            "ridge_lambda": config.ridge_lambda,
            "train_standardization_only": True,
        }
        reports[family] = report
        result_rows.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_present_support_identity_collision",
                "feature_family": family,
                **report,
                "training_performed": False,
            }
        )
    collision_inputs = {
        "degree_only": np.rint(table["degree"] * 4).astype(np.uint16),
        "exact_identity": np.rint(table["exact"] * 4).astype(np.uint8),
        **table["binary_sketches"],
    }
    collisions = {
        family: collision_metrics(matrix, labels)
        for family, matrix in collision_inputs.items()
    }
    collision_rows = [
        {"signature_family": family, **metrics}
        for family, metrics in collisions.items()
    ]
    return {
        "reports": reports,
        "result_rows": result_rows,
        "collisions": collisions,
        "collision_rows": collision_rows,
    }


def collision_metrics(matrix: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    groups: dict[bytes, list[int]] = {}
    for row_index, row in enumerate(matrix):
        signature = np.asarray(row).tobytes()
        groups.setdefault(signature, []).append(row_index)
    conflicting_groups = 0
    conflicting_rows = 0
    for indices in groups.values():
        group_labels = {int(labels[index]) for index in indices}
        if len(group_labels) > 1:
            conflicting_groups += 1
            conflicting_rows += len(indices)
    return {
        "rows": len(matrix),
        "unique_signatures": len(groups),
        "conflicting_signatures": conflicting_groups,
        "conflicting_rows": conflicting_rows,
        "conflicting_row_rate": conflicting_rows / len(matrix),
    }


def adjudicate_e48(
    config: SupportIdentityAuditConfig,
    source_checks: dict[str, bool],
    table: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    reports = evaluation["reports"]
    collisions = evaluation["collisions"]
    degree_auc = reports["degree_only"]["validation_auc"]
    exact_auc = reports["exact_identity"]["validation_auc"]
    sketch_auc = reports["sketch64"]["validation_auc"]
    permuted_auc = reports["permuted_sketch64"]["validation_auc"]
    corrupted_auc = reports["corrupted_sketch64"]["validation_auc"]
    degree_conflict = collisions["degree_only"]["conflicting_row_rate"]
    sketch_conflict = collisions["sketch64"]["conflicting_row_rate"]
    protocol_checks = {
        **source_checks,
        "all_feature_matrices_finite": all(
            bool(np.isfinite(matrix).all())
            for matrix in (
                table["degree"],
                table["exact"],
                *table["sketches"].values(),
            )
        ),
        "all_auc_finite": all(
            math.isfinite(float(report["validation_auc"]))
            for report in reports.values()
        ),
        "all_use_train_standardization": all(
            report["train_standardization_only"] for report in reports.values()
        ),
        "primary_sketch_width_is_64": reports["sketch64"]["feature_count"] == 64,
        "true_corrupted_players_are_distinct": not np.array_equal(
            table["true_player"], table["corrupted_player"]
        ),
        "projection_hashes_present": len(table["rademacher_sha256"]) == 64
        and len(table["binary_projection_sha256"]) == 64,
        "collision_rows_cover_all_samples": all(
            metrics["rows"] == len(table["rows"])
            for metrics in collisions.values()
        ),
    }
    sketch_checks = {
        "sketch64_auc_at_least_0p62": sketch_auc >= 0.62,
        "sketch64_minus_degree_at_least_0p03": sketch_auc - degree_auc >= 0.03,
        "sketch64_minus_permuted_at_least_0p03": sketch_auc - permuted_auc >= 0.03,
        "sketch64_minus_corrupted_at_least_0p03": sketch_auc - corrupted_auc >= 0.03,
        "sketch64_conflict_reduction_at_least_0p20": degree_conflict
        - sketch_conflict
        >= 0.20,
    }
    exact_checks = {
        "exact_identity_auc_at_least_0p62": exact_auc >= 0.62,
        "exact_identity_minus_degree_at_least_0p03": exact_auc - degree_auc >= 0.03,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_support_identity_protocol_invalid"
        action = "repair source, support, projection, collision, ridge, or metric protocol"
    elif all(sketch_checks.values()):
        status = "pass"
        decision = "innovation2_present_identity_sketch_route_ready"
        action = "prepare Identity-Sketch Monomial Propagator readiness smoke"
    elif all(exact_checks.values()):
        status = "pass"
        decision = "innovation2_present_exact_monomial_token_route_ready"
        action = "prepare sparse Monomial Token Set Transformer readiness smoke"
    else:
        status = "hold"
        decision = "innovation2_present_support_identity_not_supported"
        action = (
            "stop identity-network route and audit intermediate degree-spectrum "
            "distillation before any new certificate network"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "identity_sketch_checks": sketch_checks,
        "exact_identity_checks": exact_checks,
        "metrics": {
            "degree_validation_auc": degree_auc,
            "exact_identity_validation_auc": exact_auc,
            "sketch16_validation_auc": reports["sketch16"]["validation_auc"],
            "sketch32_validation_auc": reports["sketch32"]["validation_auc"],
            "sketch64_validation_auc": sketch_auc,
            "permuted_sketch64_validation_auc": permuted_auc,
            "corrupted_sketch64_validation_auc": corrupted_auc,
            "degree_conflicting_row_rate": degree_conflict,
            "exact_conflicting_row_rate": collisions["exact_identity"][
                "conflicting_row_rate"
            ],
            "sketch64_conflicting_row_rate": sketch_conflict,
            "sketch64_minus_degree": sketch_auc - degree_auc,
            "sketch64_minus_permuted": sketch_auc - permuted_auc,
            "sketch64_minus_corrupted": sketch_auc - corrupted_auc,
            "selected_route": (
                "identity_sketch"
                if all(sketch_checks.values())
                else "exact_monomial_tokens"
                if all(exact_checks.values())
                else None
            ),
        },
        "claim_scope": (
            "support-identity collision and deterministic architecture routing on "
            "the E43 real PRESENT-80 r4 strict benchmark; not a neural result, "
            "high-round distinguisher, new attack, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "network_smoke": status == "pass",
            "remote_scale": False,
        },
    }


def export_feature_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row_index, row in enumerate(table["rows"]):
        record = dict(row)
        for column, value in enumerate(table["degree"][row_index]):
            record[f"degree_{column:02d}"] = float(value)
        for family, matrix in table["sketches"].items():
            for column, value in enumerate(matrix[row_index]):
                record[f"{family}_{column:02d}"] = float(value)
        output.append(record)
    return output


def serializable_config(config: SupportIdentityAuditConfig) -> dict[str, Any]:
    return asdict(config)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
