from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.present_active_dimension_zero_shot_transfer import (
    compatible_prefix_features,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    fit_train_only_ridge,
)
from blockcipher_nd.tasks.innovation2.rectangle80_unit_balance_profile_readiness import (
    rectangle80_variable_supports,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i2_rectangle80_r4_nested_cube_relation_mechanism_20260719"
SOURCE_RUN_ID = "i2_rectangle80_r4_nested_cube_monotonic_readiness_20260719"
SOURCE_DECISION = "innovation2_rectangle80_nested_cube_monotonic_labels_ready"
SOURCE_HASHES = {
    "gate.json": "323c7bada04941e79dc932ab1c562cab977b0288039632557cf1556b41c9f4f9",
    "chains.json": "f027b920d0a8bacfe1f2e5b3a4707444cb5b92af3b6f85896be2593bebe4c8bc",
    "matched_nested_contrast.csv": "49ded50d4d5c48adf610112223d9fa34d6db85b78ca3bab4ffa4aacba03abb3b",
    "atlas.jsonl": "2fa8706232edccca5e0bd45bd362dbbc075660b55b58731a4ea155f306afbef3",
}
DIMENSIONS = (7, 8, 9)
RELATION_MODES = (
    "independent_dimension",
    "true_nesting",
    "shuffled_nesting",
    "wrong_superset",
    "true_unconstrained",
)
PROJECTED_MODES = {"true_nesting", "shuffled_nesting", "wrong_superset"}
CHAIN_COUNT = 192
OUTPUT_BITS = 64
PREFIX_DIM = 13
FEATURE_COUNT = 44
COEFFICIENT_COUNT = 45
MATCHED_ROWS = 9_556
TRAIN_ROWS = 7_216
VALIDATION_ROWS = 2_340


@dataclass(frozen=True)
class NestedCubeRelationConfig:
    run_id: str = RUN_ID
    ridge_lambda: float = 1e-3
    rounds: int = 4
    chain_count: int = CHAIN_COUNT
    output_bits: int = OUTPUT_BITS

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if (
            self.ridge_lambda != 1e-3
            or self.rounds != 4
            or self.chain_count != CHAIN_COUNT
            or self.output_bits != OUTPUT_BITS
        ):
            raise ValueError("E95 protocol is frozen")


def load_e94_sources(source_root: Path) -> dict[str, Any]:
    paths = {name: source_root / name for name in SOURCE_HASHES}
    paths["visual_qa_passed.marker"] = source_root / "visual_qa_passed.marker"
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E94 source is missing: {', '.join(missing)}")

    gate = json.loads(paths["gate.json"].read_text(encoding="utf-8"))
    chains = json.loads(paths["chains.json"].read_text(encoding="utf-8"))[
        "chains"
    ]
    with paths["matched_nested_contrast.csv"].open(
        encoding="utf-8", newline=""
    ) as handle:
        matched_rows = []
        for row in csv.DictReader(handle):
            matched_rows.append(
                {
                    "chain_index": int(row["chain_index"]),
                    "chain_id": row["chain_id"],
                    "dimension": int(row["dimension"]),
                    "output_bit": int(row["output_bit"]),
                    "label": int(row["label"]),
                    "split": row["split"],
                }
            )
    hashes = {
        name: hashlib.sha256(paths[name].read_bytes()).hexdigest()
        for name in SOURCE_HASHES
    }
    internal_groups = (
        "protocol_checks",
        "monotonic_checks",
        "width_checks",
        "shortcut_checks",
    )
    train_chains = {
        row["chain_index"] for row in matched_rows if row["split"] == "train"
    }
    validation_chains = {
        row["chain_index"]
        for row in matched_rows
        if row["split"] == "validation"
    }
    checks = {
        "e94_run_id_matches": gate.get("run_id") == SOURCE_RUN_ID,
        "e94_status_is_pass": gate.get("status") == "pass",
        "e94_decision_matches": gate.get("decision") == SOURCE_DECISION,
        "e94_internal_checks_pass": all(
            bool(gate.get(group)) and all(gate[group].values())
            for group in internal_groups
        ),
        "e94_hashes_match": hashes == SOURCE_HASHES,
        "e94_visual_qa_passed": "status=pass"
        in paths["visual_qa_passed.marker"].read_text(encoding="utf-8"),
        "chain_count_is_192": len(chains) == CHAIN_COUNT,
        "matched_rows_are_9556": len(matched_rows) == MATCHED_ROWS,
        "train_rows_are_7216": sum(
            row["split"] == "train" for row in matched_rows
        )
        == TRAIN_ROWS,
        "validation_rows_are_2340": sum(
            row["split"] == "validation" for row in matched_rows
        )
        == VALIDATION_ROWS,
        "matched_edges_unique": len(
            {
                (row["chain_index"], row["dimension"], row["output_bit"])
                for row in matched_rows
            }
        )
        == len(matched_rows),
        "train_validation_chains_disjoint": not train_chains.intersection(
            validation_chains
        ),
    }
    return {
        "gate": gate,
        "chains": chains,
        "matched_rows": matched_rows,
        "hashes": hashes,
        "checks": checks,
    }


def build_prefix_tensor(chains: list[dict[str, Any]]) -> np.ndarray:
    prefix = np.empty(
        (len(chains), len(DIMENSIONS), OUTPUT_BITS, PREFIX_DIM),
        dtype=np.float64,
    )
    for chain in chains:
        chain_index = int(chain["index"])
        for dimension_index, dimension in enumerate(DIMENSIONS):
            active_bits = tuple(int(bit) for bit in chain[f"active_bits_{dimension}"])
            supports = {
                rounds: rectangle80_variable_supports(active_bits, rounds)
                for rounds in (1, 2, 3)
            }
            all_rounds = compatible_prefix_features(supports, dimension)
            prefix[chain_index, dimension_index] = all_rounds[:, 26:39]
    return prefix


def make_relation_maps(chains: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = {
        split: sorted(
            int(chain["index"])
            for chain in chains
            if chain["split"] == split
        )
        for split in ("train", "validation")
    }
    shuffled: dict[int, int] = {}
    wrong: dict[int, int] = {}
    for indices in by_split.values():
        for position, chain_index in enumerate(indices):
            shuffled[chain_index] = indices[(position + 1) % len(indices)]
            target = chains[chain_index]
            for offset in range(1, len(indices)):
                candidate_index = indices[(position + offset) % len(indices)]
                candidate = chains[candidate_index]
                if _all_adjacent_relations_wrong(target, candidate):
                    wrong[chain_index] = candidate_index
                    break
            if chain_index not in wrong:
                raise ValueError("could not construct a split-preserving wrong-superset map")
    identity = {int(chain["index"]): int(chain["index"]) for chain in chains}
    return {
        "true_nesting": identity,
        "shuffled_nesting": shuffled,
        "wrong_superset": wrong,
        "split_indices": by_split,
    }


def validate_relation_maps(
    chains: list[dict[str, Any]], maps: dict[str, Any]
) -> dict[str, bool | float | int]:
    split_by_index = {int(chain["index"]): chain["split"] for chain in chains}
    shuffled = maps["shuffled_nesting"]
    wrong = maps["wrong_superset"]
    identical_targets = sum(
        shuffled[target] == wrong[target] for target in shuffled
    )
    wrong_relations = sum(
        _wrong_relation_count(chains[target], chains[candidate])
        for target, candidate in wrong.items()
    )
    return {
        "true_map_is_identity": all(
            target == candidate
            for target, candidate in maps["true_nesting"].items()
        ),
        "shuffled_is_derangement": all(
            target != candidate for target, candidate in shuffled.items()
        ),
        "wrong_is_derangement": all(
            target != candidate for target, candidate in wrong.items()
        ),
        "shuffled_preserves_split": all(
            split_by_index[target] == split_by_index[candidate]
            for target, candidate in shuffled.items()
        ),
        "wrong_preserves_split": all(
            split_by_index[target] == split_by_index[candidate]
            for target, candidate in wrong.items()
        ),
        "wrong_relation_checks": 4 * len(chains),
        "wrong_relation_violations": wrong_relations,
        "wrong_relation_violation_rate": wrong_relations / (4 * len(chains)),
        "wrong_all_four_relations_false": wrong_relations == 4 * len(chains),
        "shuffled_wrong_identical_targets": identical_targets,
        "shuffled_wrong_identical_rate": identical_targets / len(chains),
    }


def build_relation_features(
    prefix: np.ndarray,
    relation_map: dict[int, int] | None,
) -> np.ndarray:
    if prefix.shape != (CHAIN_COUNT, 3, OUTPUT_BITS, PREFIX_DIM):
        raise ValueError("E95 prefix tensor must be 192x3x64x13")
    features = np.zeros(
        (CHAIN_COUNT, 3, OUTPUT_BITS, FEATURE_COUNT), dtype=np.float64
    )
    for chain_index in range(CHAIN_COUNT):
        context_index = (
            chain_index if relation_map is None else int(relation_map[chain_index])
        )
        for dimension_index in range(3):
            features[chain_index, dimension_index, :, :PREFIX_DIM] = prefix[
                chain_index, dimension_index
            ]
            if relation_map is not None and dimension_index > 0:
                features[
                    chain_index,
                    dimension_index,
                    :,
                    PREFIX_DIM : 2 * PREFIX_DIM,
                ] = prefix[context_index, dimension_index - 1]
                features[chain_index, dimension_index, :, 3 * PREFIX_DIM] = 1.0
            if relation_map is not None and dimension_index < 2:
                features[
                    chain_index,
                    dimension_index,
                    :,
                    2 * PREFIX_DIM : 3 * PREFIX_DIM,
                ] = prefix[context_index, dimension_index + 1]
                features[chain_index, dimension_index, :, 3 * PREFIX_DIM + 1] = 1.0
            features[
                chain_index,
                dimension_index,
                :,
                3 * PREFIX_DIM + 2 + dimension_index,
            ] = 1.0
    return features


def isotonic_project_triplets(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (CHAIN_COUNT, 3, OUTPUT_BITS):
        raise ValueError("E95 scores must have shape 192x3x64")
    projected = np.empty_like(values)
    for chain_index in range(CHAIN_COUNT):
        for output_bit in range(OUTPUT_BITS):
            projected[chain_index, :, output_bit] = _pava_three(
                values[chain_index, :, output_bit]
            )
    return projected


def evaluate_relation_modes(
    config: NestedCubeRelationConfig,
    sources: dict[str, Any],
    prefix: np.ndarray,
    maps: dict[str, Any],
) -> dict[str, Any]:
    matched_rows = sources["matched_rows"]
    flat_indices = np.asarray(
        [
            (
                row["chain_index"] * 3 * OUTPUT_BITS
                + DIMENSIONS.index(row["dimension"]) * OUTPUT_BITS
                + row["output_bit"]
            )
            for row in matched_rows
        ],
        dtype=np.int64,
    )
    labels = np.asarray([row["label"] for row in matched_rows], dtype=np.float64)
    train = np.asarray([row["split"] == "train" for row in matched_rows])
    validation = ~train
    feature_tensors = {
        "independent_dimension": build_relation_features(prefix, None),
        "true_nesting": build_relation_features(prefix, maps["true_nesting"]),
        "shuffled_nesting": build_relation_features(
            prefix, maps["shuffled_nesting"]
        ),
        "wrong_superset": build_relation_features(prefix, maps["wrong_superset"]),
    }
    feature_tensors["true_unconstrained"] = feature_tensors["true_nesting"]
    reports: dict[str, Any] = {}
    score_tensors: dict[str, np.ndarray] = {}
    for mode in RELATION_MODES:
        tensor = feature_tensors[mode]
        flat = tensor.reshape(-1, FEATURE_COUNT)
        fitted = fit_train_only_ridge(
            flat[flat_indices[train]],
            labels[train],
            flat,
            config.ridge_lambda,
        )
        raw_scores = fitted["validation_scores"].reshape(
            CHAIN_COUNT, 3, OUTPUT_BITS
        )
        final_scores = (
            isotonic_project_triplets(raw_scores)
            if mode in PROJECTED_MODES
            else raw_scores
        )
        score_tensors[mode] = final_scores
        selected_scores = final_scores.reshape(-1)[flat_indices]
        reports[mode] = {
            "relation_mode": mode,
            "feature_count": FEATURE_COUNT,
            "coefficient_count": int(len(fitted["weights"])),
            "ridge_lambda": config.ridge_lambda,
            "train_standardization_only": True,
            "isotonic_projection": mode in PROJECTED_MODES,
            "train_auc": _safe_auc(labels[train], selected_scores[train]),
            "validation_auc": _safe_auc(
                labels[validation], selected_scores[validation]
            ),
            "raw_monotonic_violations": monotonic_violation_count(raw_scores),
            "final_monotonic_violations": monotonic_violation_count(final_scores),
        }
    return {
        "reports": reports,
        "score_tensors": score_tensors,
        "matched_rows": len(matched_rows),
        "train_rows": int(np.sum(train)),
        "validation_rows": int(np.sum(validation)),
        "train_chains": len(
            {row["chain_index"] for row in matched_rows if row["split"] == "train"}
        ),
        "validation_chains": len(
            {
                row["chain_index"]
                for row in matched_rows
                if row["split"] == "validation"
            }
        ),
    }


def evaluate_relation_gate(
    config: NestedCubeRelationConfig,
    sources: dict[str, Any],
    prefix: np.ndarray,
    maps: dict[str, Any],
    relation_results: dict[str, Any],
) -> dict[str, Any]:
    reports = relation_results["reports"]
    true = reports["true_nesting"]
    relation_checks = validate_relation_maps(sources["chains"], maps)
    margins = {
        "true_minus_independent": true["validation_auc"]
        - reports["independent_dimension"]["validation_auc"],
        "true_minus_shuffled": true["validation_auc"]
        - reports["shuffled_nesting"]["validation_auc"],
        "true_minus_wrong_superset": true["validation_auc"]
        - reports["wrong_superset"]["validation_auc"],
        "true_minus_unconstrained": true["validation_auc"]
        - reports["true_unconstrained"]["validation_auc"],
        "true_train_minus_validation": true["train_auc"]
        - true["validation_auc"],
    }
    protocol_checks = {
        **sources["checks"],
        "prefix_shape_matches": prefix.shape
        == (CHAIN_COUNT, 3, OUTPUT_BITS, PREFIX_DIM),
        "prefix_features_finite": bool(np.isfinite(prefix).all()),
        "relation_maps_valid": all(
            bool(value)
            for key, value in relation_checks.items()
            if key
            in {
                "true_map_is_identity",
                "shuffled_is_derangement",
                "wrong_is_derangement",
                "shuffled_preserves_split",
                "wrong_preserves_split",
                "wrong_all_four_relations_false",
            }
        ),
        "all_modes_present": set(reports) == set(RELATION_MODES),
        "all_modes_have_44_features": all(
            row["feature_count"] == FEATURE_COUNT for row in reports.values()
        ),
        "all_modes_have_45_coefficients": all(
            row["coefficient_count"] == COEFFICIENT_COUNT
            for row in reports.values()
        ),
        "projected_modes_have_zero_final_violations": all(
            reports[mode]["final_monotonic_violations"] == 0
            for mode in PROJECTED_MODES
        ),
        "matched_row_counts_replay": relation_results["matched_rows"]
        == MATCHED_ROWS
        and relation_results["train_rows"] == TRAIN_ROWS
        and relation_results["validation_rows"] == VALIDATION_ROWS,
        "chain_split_counts_replay": relation_results["train_chains"] == 144
        and relation_results["validation_chains"] == 48,
    }
    quality_checks = {
        "true_validation_auc_at_least_0p70": true["validation_auc"] >= 0.70,
        "true_train_validation_gap_at_most_0p15": margins[
            "true_train_minus_validation"
        ]
        <= 0.15,
    }
    attribution_checks = {
        "true_minus_independent_at_least_0p03": margins[
            "true_minus_independent"
        ]
        >= 0.03,
        "true_minus_shuffled_at_least_0p03": margins["true_minus_shuffled"]
        >= 0.03,
        "true_minus_wrong_superset_at_least_0p03": margins[
            "true_minus_wrong_superset"
        ]
        >= 0.03,
        "true_not_worse_than_unconstrained_by_0p01": margins[
            "true_minus_unconstrained"
        ]
        >= -0.01,
    }
    status, decision, action = adjudicate_relation_checks(
        protocol_checks, quality_checks, attribution_checks
    )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "quality_checks": quality_checks,
        "attribution_checks": attribution_checks,
        "metrics": {
            "reports": reports,
            "margins": margins,
            "relation_checks": relation_checks,
            "prefix_shape": list(prefix.shape),
            "matched_rows": relation_results["matched_rows"],
            "train_rows": relation_results["train_rows"],
            "validation_rows": relation_results["validation_rows"],
        },
        "claim_scope": (
            "deterministic train-only ridge/isotonic attribution of RECTANGLE-80 "
            "r4 true nested-cube relations; no neural gain, third-SPN formal "
            "confirmation, high-round distinguisher, attack, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
            "neural_readiness": status == "pass",
        },
    }


def adjudicate_relation_checks(
    protocol_checks: dict[str, bool],
    quality_checks: dict[str, bool],
    attribution_checks: dict[str, bool],
) -> tuple[str, str, str]:
    if not all(protocol_checks.values()):
        return (
            "fail",
            "innovation2_rectangle80_nested_cube_relation_protocol_invalid",
            "repair E94 replay, relation maps, feature capacity, split, or projection",
        )
    if not all(quality_checks.values()) or not all(attribution_checks.values()):
        return (
            "hold",
            "innovation2_rectangle80_nested_cube_relation_not_attributed",
            "close the current nested-cube neural route without architecture tuning",
        )
    return (
        "pass",
        "innovation2_rectangle80_nested_cube_relation_mechanism_ready",
        "run a two-epoch local Monotone Cube-Lattice Operator readiness matrix",
    )


def result_rows_for_relation_gate(
    config: NestedCubeRelationConfig, gate: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": config.run_id,
            "task": "innovation2_rectangle80_nested_cube_relation_mechanism",
            "cipher": "RECTANGLE-80",
            "rounds": config.rounds,
            "relation_mode": mode,
            "training_performed": False,
            "status": gate["status"],
            "decision": gate["decision"],
            **gate["metrics"]["reports"][mode],
        }
        for mode in RELATION_MODES
    ]


def monotonic_violation_count(scores: np.ndarray, tolerance: float = 1e-12) -> int:
    values = np.asarray(scores, dtype=np.float64)
    return int(
        np.sum(values[:, 0] > values[:, 1] + tolerance)
        + np.sum(values[:, 1] > values[:, 2] + tolerance)
    )


def serializable_config(config: NestedCubeRelationConfig) -> dict[str, Any]:
    return asdict(config)


def _all_adjacent_relations_wrong(
    target: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    return _wrong_relation_count(target, candidate) == 4


def _wrong_relation_count(
    target: dict[str, Any], candidate: dict[str, Any]
) -> int:
    target_7 = set(target["active_bits_7"])
    target_8 = set(target["active_bits_8"])
    target_9 = set(target["active_bits_9"])
    candidate_7 = set(candidate["active_bits_7"])
    candidate_8 = set(candidate["active_bits_8"])
    candidate_9 = set(candidate["active_bits_9"])
    return sum(
        (
            not candidate_7.issubset(target_8),
            not target_7.issubset(candidate_8),
            not candidate_8.issubset(target_9),
            not target_8.issubset(candidate_9),
        )
    )


def _pava_three(values: np.ndarray) -> np.ndarray:
    blocks: list[tuple[int, int, float, int]] = []
    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        blocks.append((index, index + 1, float(value), 1))
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left[2] / left[3] <= right[2] / right[3]:
                break
            blocks[-2:] = [
                (left[0], right[1], left[2] + right[2], left[3] + right[3])
            ]
    projected = np.empty(3, dtype=np.float64)
    for start, stop, total, weight in blocks:
        projected[start:stop] = total / weight
    return projected


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


__all__ = [
    "CHAIN_COUNT",
    "COEFFICIENT_COUNT",
    "DIMENSIONS",
    "FEATURE_COUNT",
    "NestedCubeRelationConfig",
    "PREFIX_DIM",
    "PROJECTED_MODES",
    "RELATION_MODES",
    "RUN_ID",
    "SOURCE_DECISION",
    "SOURCE_HASHES",
    "SOURCE_RUN_ID",
    "adjudicate_relation_checks",
    "build_prefix_tensor",
    "build_relation_features",
    "evaluate_relation_gate",
    "evaluate_relation_modes",
    "isotonic_project_triplets",
    "load_e94_sources",
    "make_relation_maps",
    "monotonic_violation_count",
    "result_rows_for_relation_gate",
    "serializable_config",
    "validate_relation_maps",
]
