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
from blockcipher_nd.tasks.innovation2.present_pair_state_neural_attribution import (
    load_e43_source,
    validate_e43_source,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    possible_active_monomials,
)
from blockcipher_nd.training.metrics import binary_auc


SOURCE_NEURAL_RUN_ID = "i2_present_r4_pair_state_neural_attribution_seed0_20260718"
SOURCE_NEURAL_DECISION = "innovation2_present_pair_state_candidate_not_ready"
E44_TRIANGLE_AUC = 0.5619793162884229
RIDGE_LAMBDA = 1e-3


@dataclass(frozen=True)
class CertificateAttributionConfig:
    run_id: str
    mode: str = "audit"
    ridge_lambda: float = RIDGE_LAMBDA

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        if self.mode == "audit" and self.ridge_lambda != RIDGE_LAMBDA:
            raise ValueError("E45 audit ridge lambda is frozen")


def load_sources(atlas_root: Path, neural_root: Path) -> dict[str, Any]:
    atlas = load_e43_source(atlas_root)
    neural_gate = json.loads((neural_root / "gate.json").read_text(encoding="utf-8"))
    neural_results = [
        json.loads(line)
        for line in (neural_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "atlas": atlas,
        "neural_gate": neural_gate,
        "neural_results": neural_results,
        "neural_hashes": {
            name: _sha256(neural_root / name)
            for name in ("gate.json", "results.jsonl")
        },
    }


def validate_sources(sources: dict[str, Any], *, strict: bool) -> dict[str, bool]:
    atlas_checks = validate_e43_source(sources["atlas"], strict=strict)
    neural_gate = sources["neural_gate"]
    neural_results = sources["neural_results"]
    triangle = next(
        (
            row
            for row in neural_results
            if row.get("row_id") == "pair_triangle_true_seed0"
        ),
        None,
    )
    return {
        **{f"atlas_{key}": value for key, value in atlas_checks.items()},
        "neural_run_id_matches": neural_gate.get("run_id") == SOURCE_NEURAL_RUN_ID,
        "neural_decision_matches": neural_gate.get("decision")
        == SOURCE_NEURAL_DECISION,
        "neural_status_hold": neural_gate.get("status") == "hold",
        "triangle_result_present": triangle is not None,
        "triangle_auc_matches_gate": triangle is not None
        and math.isclose(
            float(triangle["validation_auc"]),
            float(neural_gate["metrics"]["best_true_validation_auc"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "source_hashes_present": all(
            len(value) == 64
            for value in (
                *sources["atlas"]["source_hashes"].values(),
                *sources["neural_hashes"].values(),
            )
        ),
    }


def build_feature_table(sources: dict[str, Any]) -> dict[str, Any]:
    atlas = sources["atlas"]
    structures = atlas["structures"]
    masks = atlas["masks"]
    rows = atlas["rows"]
    true_player = atlas["players"][0]
    corrupted_player = topology_players(atlas["players"], "corrupted")[0]
    support_cache = {
        structure_index: {
            rounds: possible_active_monomials(
                tuple(structures[structure_index]["active_bits"]), rounds
            )
            for rounds in (1, 2, 3, 4)
        }
        for structure_index in sorted({row["structure_index"] for row in rows})
    }
    feature_names = {
        "static_set": static_feature_names(),
        "true_topology": topology_feature_names(),
        "corrupted_topology": topology_feature_names(),
        "anf_prefix": prefix_feature_names(),
        "final_oracle": ("negative_full_cube_candidate_count",),
    }
    matrices = {
        family: np.zeros((len(rows), len(names)), dtype=np.float64)
        for family, names in feature_names.items()
    }
    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        structure_index = int(row["structure_index"])
        mask_index = int(row["mask_index"])
        active = np.flatnonzero(atlas["structure_active"][structure_index]).astype(
            np.int64
        )
        selected = np.flatnonzero(atlas["output_mask_bits"][mask_index]).astype(
            np.int64
        )
        matrices["static_set"][row_index] = static_set_features(active, selected)
        matrices["true_topology"][row_index] = topology_reachability_features(
            active, selected, true_player
        )
        matrices["corrupted_topology"][row_index] = topology_reachability_features(
            active, selected, corrupted_player
        )
        matrices["anf_prefix"][row_index] = anf_prefix_features(
            selected, support_cache[structure_index]
        )
        full_cube = (1 << len(active)) - 1
        full_count = sum(
            full_cube in support_cache[structure_index][4][int(bit)]
            for bit in selected
        )
        matrices["final_oracle"][row_index, 0] = -float(full_count)
        output_rows.append(
            {
                "split": row["split"],
                "structure_index": structure_index,
                "mask_index": mask_index,
                "label": int(row["label"]),
            }
        )
    return {
        "rows": output_rows,
        "matrices": matrices,
        "feature_names": feature_names,
        "true_player": true_player,
        "corrupted_player": corrupted_player,
    }


def evaluate_feature_families(
    config: CertificateAttributionConfig, table: dict[str, Any]
) -> dict[str, Any]:
    split = np.asarray([row["split"] for row in table["rows"]])
    labels = np.asarray([row["label"] for row in table["rows"]], dtype=np.float64)
    train = split == "train"
    validation = split == "validation"
    reports: dict[str, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    for family in ("static_set", "true_topology", "corrupted_topology", "anf_prefix"):
        matrix = table["matrices"][family]
        fitted = fit_train_only_ridge(
            matrix[train], labels[train], matrix[validation], config.ridge_lambda
        )
        train_auc = _safe_auc(labels[train], fitted["train_scores"])
        validation_auc = _safe_auc(labels[validation], fitted["validation_scores"])
        single_aucs = [
            max(
                _safe_auc(labels[validation], matrix[validation, column]),
                1.0 - _safe_auc(labels[validation], matrix[validation, column]),
            )
            for column in range(matrix.shape[1])
        ]
        best_index = int(np.argmax(single_aucs))
        report = {
            "feature_count": matrix.shape[1],
            "train_auc": train_auc,
            "validation_auc": validation_auc,
            "strongest_single_feature_auc": single_aucs[best_index],
            "strongest_single_feature": table["feature_names"][family][best_index],
            "train_standardization_only": True,
            "ridge_lambda": config.ridge_lambda,
        }
        reports[family] = report
        result_rows.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_present_certificate_complexity_attribution",
                "feature_family": family,
                **report,
                "oracle": False,
                "training_performed": False,
            }
        )
    oracle_scores = table["matrices"]["final_oracle"][:, 0]
    oracle = {
        "feature_count": 1,
        "train_auc": _safe_auc(labels[train], oracle_scores[train]),
        "validation_auc": _safe_auc(labels[validation], oracle_scores[validation]),
        "strongest_single_feature_auc": _safe_auc(
            labels[validation], oracle_scores[validation]
        ),
        "strongest_single_feature": "negative_full_cube_candidate_count",
        "train_standardization_only": False,
        "ridge_lambda": None,
    }
    reports["final_oracle"] = oracle
    result_rows.append(
        {
            "run_id": config.run_id,
            "task": "innovation2_present_certificate_complexity_attribution",
            "feature_family": "final_oracle",
            **oracle,
            "oracle": True,
            "training_performed": False,
        }
    )
    return {"reports": reports, "result_rows": result_rows}


def adjudicate_e45(
    config: CertificateAttributionConfig,
    source_checks: dict[str, bool],
    table: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    reports = evaluation["reports"]
    static_auc = reports["static_set"]["validation_auc"]
    true_auc = reports["true_topology"]["validation_auc"]
    corrupted_auc = reports["corrupted_topology"]["validation_auc"]
    prefix_auc = reports["anf_prefix"]["validation_auc"]
    oracle_auc = reports["final_oracle"]["validation_auc"]
    matrices = table["matrices"]
    protocol_checks = {
        **source_checks,
        "all_feature_matrices_finite": all(
            bool(np.isfinite(matrix).all()) for matrix in matrices.values()
        ),
        "true_corrupted_topology_dimensions_match": matrices["true_topology"].shape
        == matrices["corrupted_topology"].shape,
        "true_corrupted_players_are_distinct": not np.array_equal(
            table["true_player"], table["corrupted_player"]
        ),
        "all_nonoracle_use_train_standardization": all(
            reports[family]["train_standardization_only"]
            for family in (
                "static_set",
                "true_topology",
                "corrupted_topology",
                "anf_prefix",
            )
        ),
        "oracle_auc_is_one": math.isclose(oracle_auc, 1.0, abs_tol=1e-12),
        "all_auc_finite": all(
            math.isfinite(float(report["validation_auc"]))
            for report in reports.values()
        ),
    }
    route_checks = {
        "certificate_prefix_route": {
            "prefix_auc_at_least_0p60": prefix_auc >= 0.60,
            "prefix_minus_static_at_least_0p03": prefix_auc - static_auc >= 0.03,
            "prefix_minus_true_topology_at_least_0p02": prefix_auc - true_auc >= 0.02,
        },
        "topology_route": {
            "true_topology_auc_at_least_0p60": true_auc >= 0.60,
            "true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03,
            "true_at_least_prefix_minus_0p01": true_auc >= prefix_auc - 0.01,
        },
        "static_route": {
            "static_auc_at_least_0p60": static_auc >= 0.60,
            "static_within_0p01_of_best_nonoracle": static_auc
            >= max(prefix_auc, true_auc) - 0.01,
        },
    }
    passing = [route for route, checks in route_checks.items() if all(checks.values())]
    score_by_route = {
        "certificate_prefix_route": prefix_auc,
        "topology_route": true_auc,
        "static_route": static_auc,
    }
    selected_route = _select_route(passing, score_by_route)
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_certificate_attribution_protocol_invalid"
        action = "repair source, feature, standardization, oracle, or metric protocol"
    elif selected_route == "certificate_prefix_route":
        status = "pass"
        decision = "innovation2_present_mspn_route_ready"
        action = "prepare E46 Monomial Support Propagation Network readiness smoke"
    elif selected_route == "topology_route":
        status = "pass"
        decision = "innovation2_present_query_nbfnet_route_ready"
        action = "prepare E46 query-conditioned NBFNet-style readiness smoke"
    elif selected_route == "static_route":
        status = "hold"
        decision = "innovation2_present_static_set_route_dominant"
        action = "pause topology models and build a same-budget set-interaction anchor"
    else:
        status = "hold"
        decision = "innovation2_present_certificate_attribution_unresolved"
        action = "add non-leaking certificate-state supervision or audit unknown boundaries"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "route_checks": route_checks,
        "metrics": {
            "static_validation_auc": static_auc,
            "true_topology_validation_auc": true_auc,
            "corrupted_topology_validation_auc": corrupted_auc,
            "anf_prefix_validation_auc": prefix_auc,
            "final_oracle_validation_auc": oracle_auc,
            "e44_triangle_validation_auc": E44_TRIANGLE_AUC,
            "true_minus_corrupted_topology": true_auc - corrupted_auc,
            "prefix_minus_static": prefix_auc - static_auc,
            "prefix_minus_true_topology": prefix_auc - true_auc,
            "selected_route": selected_route,
        },
        "claim_scope": (
            "deterministic feature attribution and neural-architecture routing on "
            "the E43/E44 real PRESENT-80 r4 matched benchmark; not a neural result, "
            "new integral distinguisher, high-round conclusion, or SOTA attack"
        ),
        "next_action": {
            "action": action,
            "network_smoke": status == "pass",
            "remote_scale": False,
        },
    }


def static_feature_names() -> tuple[str, ...]:
    return (
        "mask_weight",
        "active_cell_count",
        "mask_cell_count",
        "bit_overlap",
        "cell_overlap",
        "same_cell_cross_pairs",
        "active_bit_mean",
        "active_bit_std",
        "mask_bit_mean",
        "mask_bit_std",
        "active_span",
        "mask_span",
        *(f"active_lane_{lane}" for lane in range(4)),
        *(f"mask_lane_{lane}" for lane in range(4)),
    )


def static_set_features(active: np.ndarray, selected: np.ndarray) -> np.ndarray:
    active_cells = np.unique(active // 4)
    mask_cells = np.unique(selected // 4)
    values = [
        len(selected) / 4.0,
        len(active_cells) / 8.0,
        len(mask_cells) / 4.0,
        len(np.intersect1d(active, selected)) / 4.0,
        len(np.intersect1d(active_cells, mask_cells)) / 4.0,
        sum((left // 4) == (right // 4) for left in active for right in selected)
        / 32.0,
        float(np.mean(active)) / 63.0,
        float(np.std(active)) / 32.0,
        float(np.mean(selected)) / 63.0,
        float(np.std(selected)) / 32.0,
        float(np.ptp(active)) / 63.0,
        float(np.ptp(selected)) / 63.0,
    ]
    values.extend(float(np.sum(active % 4 == lane)) / 8.0 for lane in range(4))
    values.extend(
        float(np.sum(selected % 4 == lane)) / max(1, len(selected)) for lane in range(4)
    )
    return np.asarray(values, dtype=np.float64)


def topology_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for step in range(1, 5):
        names.extend(
            (
                f"step{step}_bit_overlap",
                f"step{step}_cell_overlap",
                f"step{step}_reachable_bits",
                f"step{step}_reachable_cells",
            )
        )
    names.extend(("first_bit_hit_step", "cumulative_bit_overlap"))
    return tuple(names)


def topology_reachability_features(
    active: np.ndarray, selected: np.ndarray, player: np.ndarray
) -> np.ndarray:
    reachable = np.zeros(64, dtype=np.bool_)
    reachable[active] = True
    selected_cells = np.unique(selected // 4)
    values: list[float] = []
    bit_hits: list[int] = []
    for _ in range(4):
        active_cells = np.flatnonzero(
            np.add.reduceat(reachable.astype(np.int8), np.arange(0, 64, 4)) > 0
        )
        expanded = np.zeros(64, dtype=np.bool_)
        for cell in active_cells:
            expanded[4 * cell : 4 * cell + 4] = True
        propagated = np.zeros(64, dtype=np.bool_)
        propagated[player[np.flatnonzero(expanded)]] = True
        reachable = propagated
        hit = int(np.sum(reachable[selected]))
        bit_hits.append(hit)
        reachable_cells = np.unique(np.flatnonzero(reachable) // 4)
        values.extend(
            (
                hit / max(1, len(selected)),
                len(np.intersect1d(reachable_cells, selected_cells))
                / max(1, len(selected_cells)),
                float(np.sum(reachable)) / 64.0,
                len(reachable_cells) / 16.0,
            )
        )
    first_hit = next((index + 1 for index, value in enumerate(bit_hits) if value), 0)
    values.extend((first_hit / 4.0, sum(bit_hits) / max(1, 4 * len(selected))))
    return np.asarray(values, dtype=np.float64)


def prefix_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for rounds in (1, 2, 3):
        names.extend(
            (
                f"r{rounds}_support_mean",
                f"r{rounds}_support_max",
                f"r{rounds}_support_sum",
                f"r{rounds}_support_union",
                *(f"r{rounds}_degree_{degree}" for degree in range(9)),
            )
        )
    return tuple(names)


def anf_prefix_features(
    selected: np.ndarray, supports_by_round: dict[int, tuple[frozenset[int], ...]]
) -> np.ndarray:
    values: list[float] = []
    for rounds in (1, 2, 3):
        supports = [supports_by_round[rounds][int(bit)] for bit in selected]
        sizes = np.asarray([len(support) for support in supports], dtype=np.float64)
        union = set().union(*supports)
        values.extend(
            (
                float(np.mean(sizes)) / 256.0,
                float(np.max(sizes)) / 256.0,
                float(np.sum(sizes)) / (256.0 * max(1, len(sizes))),
                len(union) / 256.0,
            )
        )
        degree_counts = np.bincount(
            [monomial.bit_count() for monomial in union], minlength=9
        ).astype(np.float64)
        degree_counts /= max(1.0, float(degree_counts.sum()))
        values.extend(degree_counts.tolist())
    return np.asarray(values, dtype=np.float64)


def fit_train_only_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    ridge_lambda: float,
) -> dict[str, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train = (train_x - mean) / scale
    validation = (validation_x - mean) / scale
    train_design = np.column_stack((np.ones(len(train)), train))
    validation_design = np.column_stack((np.ones(len(validation)), validation))
    penalty = np.eye(train_design.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0
    system = train_design.T @ train_design + penalty
    target = train_design.T @ train_y
    try:
        weights = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(system, target, rcond=None)[0]
    return {
        "train_scores": train_design @ weights,
        "validation_scores": validation_design @ weights,
        "weights": weights,
        "mean": mean,
        "scale": scale,
    }


def serializable_config(config: CertificateAttributionConfig) -> dict[str, Any]:
    return asdict(config)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


def _select_route(
    passing: list[str], score_by_route: dict[str, float]
) -> str | None:
    if not passing:
        return None
    complexity = {
        "static_route": 0,
        "topology_route": 1,
        "certificate_prefix_route": 2,
    }
    best_score = max(score_by_route[route] for route in passing)
    near_best = [
        route for route in passing if best_score - score_by_route[route] < 0.01
    ]
    return min(near_best, key=lambda route: complexity[route])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
