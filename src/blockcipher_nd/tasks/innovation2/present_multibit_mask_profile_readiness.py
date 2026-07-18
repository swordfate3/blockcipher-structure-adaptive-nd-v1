from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    anf_prefix_features,
    fit_train_only_ridge,
    static_set_features,
    topology_reachability_features,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    ActiveStructure,
    LinearOutputMask,
    build_checkerboard_benchmark,
    checkerboard_balance,
    marginal_baselines,
    possible_active_monomials,
)
from blockcipher_nd.training.metrics import binary_auc


SOURCE_RUN_ID = "i2_present_r4_universal_balance_atlas_20260718"
SOURCE_DECISION = "innovation2_present_universal_balance_atlas_ready"
MULTIBIT_FAMILIES = (
    "nibble",
    "player_pair",
    "same_nibble_pair",
    "adjacent_nibble_pair",
)
AUDIT_ATTEMPTS = 64


@dataclass(frozen=True)
class MultibitMaskProfileConfig:
    run_id: str
    mode: str = "audit"
    attempts: int = AUDIT_ATTEMPTS
    ridge_lambda: float = 1e-3

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.attempts <= 0 or self.ridge_lambda <= 0:
            raise ValueError("attempts and ridge_lambda must be positive")
        if self.mode == "audit" and (
            self.attempts != AUDIT_ATTEMPTS or self.ridge_lambda != 1e-3
        ):
            raise ValueError("E69 audit protocol is frozen")


def load_e43_multibit_source(source_root: Path) -> dict[str, Any]:
    gate = json.loads((source_root / "gate.json").read_text(encoding="utf-8"))
    structures_payload = json.loads(
        (source_root / "structures.json").read_text(encoding="utf-8")
    )
    masks_payload = json.loads((source_root / "masks.json").read_text(encoding="utf-8"))
    structures = tuple(
        ActiveStructure(
            index=int(row["index"]),
            structure_id=str(row["structure_id"]),
            role=str(row["role"]),
            active_bits=tuple(int(bit) for bit in row["active_bits"]),
        )
        for row in structures_payload["structures"]
    )
    masks = tuple(
        LinearOutputMask(
            index=int(row["index"]),
            mask_id=str(row["mask_id"]),
            family=str(row["family"]),
            value=int(str(row["mask_hex"]), 16),
        )
        for row in masks_payload["masks"]
    )
    labels = np.full((len(structures), len(masks)), -1, dtype=np.int8)
    seen: set[tuple[int, int]] = set()
    with (source_root / "atlas.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            edge = (int(row["structure_index"]), int(row["mask_index"]))
            if edge in seen:
                raise ValueError("duplicate edge in E43 atlas")
            seen.add(edge)
            labels[edge] = -1 if row["label"] is None else int(row["label"])
    return {
        "gate": gate,
        "structures": structures,
        "masks": masks,
        "labels": labels,
        "raw_edges": len(seen),
        "source_hashes": {
            name: _sha256(source_root / name)
            for name in ("gate.json", "structures.json", "masks.json", "atlas.jsonl")
        },
    }


def validate_multibit_source(source: dict[str, Any], *, strict: bool) -> dict[str, bool]:
    masks = source["masks"]
    families = {mask.family for mask in masks[64:]}
    checks = {
        "source_run_id_matches": source["gate"].get("run_id") == SOURCE_RUN_ID,
        "source_decision_ready": source["gate"].get("decision") == SOURCE_DECISION,
        "source_status_pass": source["gate"].get("status") == "pass",
        "source_hashes_present": all(
            len(value) == 64 for value in source["source_hashes"].values()
        ),
        "label_shape_is_96x300": source["labels"].shape == (96, 300),
        "labels_are_ternary": bool(np.isin(source["labels"], (-1, 0, 1)).all()),
        "first_64_masks_are_units": all(
            mask.index == index and mask.family == "unit" and mask.value == 1 << index
            for index, mask in enumerate(masks[:64])
        ),
        "nonunit_families_match": families == set(MULTIBIT_FAMILIES),
        "all_nonunit_masks_are_multibit": all(
            mask.value.bit_count() > 1 for mask in masks[64:]
        ),
    }
    if strict:
        checks.update(
            {
                "structure_count_is_96": len(source["structures"]) == 96,
                "mask_count_is_300": len(masks) == 300,
                "nonunit_mask_count_is_236": len(masks[64:]) == 236,
                "raw_edges_are_28800": source["raw_edges"] == 96 * 300,
            }
        )
    return checks


def build_multibit_benchmark(
    source: dict[str, Any], *, attempts: int
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    family_metrics: dict[str, dict[str, Any]] = {}
    family_benchmarks: dict[str, dict[str, Any]] = {}
    for family in MULTIBIT_FAMILIES:
        original_indices = [
            mask.index for mask in source["masks"] if mask.family == family
        ]
        local_masks = tuple(
            LinearOutputMask(
                index=local_index,
                mask_id=source["masks"][original_index].mask_id,
                family=family,
                value=source["masks"][original_index].value,
            )
            for local_index, original_index in enumerate(original_indices)
        )
        local = build_checkerboard_benchmark(
            labels=source["labels"][:, original_indices],
            structures=source["structures"],
            masks=local_masks,
            attempts=attempts,
        )
        mapped_rows = []
        for row in local["rows"]:
            mapped = dict(row)
            mapped["mask_index"] = original_indices[int(row["mask_index"])]
            mapped["mask_family"] = family
            mapped_rows.append(mapped)
        all_rows.extend(mapped_rows)
        family_metrics[family] = local["split_metrics"]
        family_benchmarks[family] = {
            "rows": mapped_rows,
            "marginal_baselines": marginal_baselines(
                mapped_rows_for_split(mapped_rows, "train"),
                mapped_rows_for_split(mapped_rows, "validation"),
                source["structures"],
                source["masks"],
            ),
        }
    combined = {
        "rows": all_rows,
        "split_metrics": _combined_split_metrics(all_rows),
        "family_metrics": family_metrics,
        "balance": checkerboard_balance(all_rows),
        "marginal_baselines": marginal_baselines(
            mapped_rows_for_split(all_rows, "train"),
            mapped_rows_for_split(all_rows, "validation"),
            source["structures"],
            source["masks"],
        ),
        "family_benchmarks": family_benchmarks,
    }
    return combined


def mapped_rows_for_split(
    rows: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


def decompose_multibit_labels(
    source: dict[str, Any], benchmark: dict[str, Any]
) -> dict[str, Any]:
    unit_labels = source["labels"][:, :64]
    rows = []
    for row in benchmark["rows"]:
        structure = int(row["structure_index"])
        mask = source["masks"][int(row["mask_index"])]
        bits = np.asarray(mask.bits, dtype=np.int64)
        components = unit_labels[structure, bits]
        all_positive = bool(np.all(components == 1))
        rows.append(
            {
                **row,
                "all_component_units_positive": all_positive,
                "component_positive_count": int(np.sum(components == 1)),
                "component_negative_count": int(np.sum(components == 0)),
                "component_unknown_count": int(np.sum(components < 0)),
                "nontrivial_positive": int(row["label"]) == 1 and not all_positive,
            }
        )
    reports: dict[str, dict[str, float | int]] = {}
    for name, selected in (
        ("combined", rows),
        *(
            (family, [row for row in rows if row["mask_family"] == family])
            for family in MULTIBIT_FAMILIES
        ),
    ):
        train = [row for row in selected if row["split"] == "train"]
        validation = [row for row in selected if row["split"] == "validation"]
        positive = [row for row in selected if int(row["label"]) == 1]
        validation_positive = [
            row for row in validation if int(row["label"]) == 1
        ]
        family_mask_indices = [
            mask.index
            for mask in source["masks"]
            if mask.index >= 64 and (name == "combined" or mask.family == name)
        ]
        raw_positive = 0
        raw_nontrivial_positive = 0
        for structure in range(source["labels"].shape[0]):
            for mask_index in family_mask_indices:
                if int(source["labels"][structure, mask_index]) != 1:
                    continue
                raw_positive += 1
                bits = np.asarray(source["masks"][mask_index].bits, dtype=np.int64)
                raw_nontrivial_positive += int(
                    not np.all(unit_labels[structure, bits] == 1)
                )
        reports[name] = {
            "train_nontrivial_positive": sum(
                bool(row["nontrivial_positive"]) for row in train
            ),
            "validation_nontrivial_positive": sum(
                bool(row["nontrivial_positive"]) for row in validation
            ),
            "nontrivial_positive_fraction": (
                sum(bool(row["nontrivial_positive"]) for row in positive)
                / len(positive)
                if positive
                else 0.0
            ),
            "raw_positive": raw_positive,
            "raw_nontrivial_positive": raw_nontrivial_positive,
            "raw_nontrivial_positive_fraction": (
                raw_nontrivial_positive / raw_positive if raw_positive else 0.0
            ),
            "validation_componentwise_auc": _safe_auc(
                np.asarray([row["label"] for row in validation], dtype=np.float64),
                np.asarray(
                    [row["all_component_units_positive"] for row in validation],
                    dtype=np.float64,
                ),
            ),
            "validation_positive": len(validation_positive),
        }
    return {"rows": rows, "reports": reports}


def build_multibit_feature_table(
    source: dict[str, Any], benchmark: dict[str, Any]
) -> dict[str, Any]:
    rows = benchmark["rows"]
    structure_indices = sorted({int(row["structure_index"]) for row in rows})
    support_cache = {
        structure_index: {
            rounds: possible_active_monomials(
                source["structures"][structure_index].active_bits, rounds
            )
            for rounds in (1, 2, 3)
        }
        for structure_index in structure_indices
    }
    true_player = np.asarray(
        [(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)], dtype=np.int64
    )
    corrupted_player = topology_players(true_player[None, :], "corrupted")[0]
    matrices = {
        "static_set": np.zeros((len(rows), 20), dtype=np.float64),
        "corrupted_topology": np.zeros((len(rows), 18), dtype=np.float64),
        "true_topology": np.zeros((len(rows), 18), dtype=np.float64),
        "anf_prefix": np.zeros((len(rows), 39), dtype=np.float64),
    }
    table_rows = []
    for row_index, row in enumerate(rows):
        structure_index = int(row["structure_index"])
        active = np.asarray(
            source["structures"][structure_index].active_bits, dtype=np.int64
        )
        selected = np.asarray(
            source["masks"][int(row["mask_index"])].bits, dtype=np.int64
        )
        matrices["static_set"][row_index] = static_set_features(active, selected)
        matrices["corrupted_topology"][row_index] = topology_reachability_features(
            active, selected, corrupted_player
        )
        matrices["true_topology"][row_index] = topology_reachability_features(
            active, selected, true_player
        )
        matrices["anf_prefix"][row_index] = anf_prefix_features(
            selected, support_cache[structure_index]
        )
        table_rows.append(
            {
                "split": row["split"],
                "mask_family": row["mask_family"],
                "structure_index": structure_index,
                "mask_index": int(row["mask_index"]),
                "label": int(row["label"]),
            }
        )
    return {"rows": table_rows, "matrices": matrices}


def evaluate_multibit_features(
    config: MultibitMaskProfileConfig, table: dict[str, Any]
) -> dict[str, dict[str, float | int]]:
    reports = {}
    for scope in ("combined", *MULTIBIT_FAMILIES):
        selected = np.asarray(
            [scope == "combined" or row["mask_family"] == scope for row in table["rows"]]
        )
        split = np.asarray([row["split"] for row in table["rows"]])
        labels = np.asarray([row["label"] for row in table["rows"]], dtype=np.float64)
        train = selected & (split == "train")
        validation = selected & (split == "validation")
        scope_reports = {}
        for family, matrix in table["matrices"].items():
            if (
                int(np.sum(train)) == 0
                or int(np.sum(validation)) == 0
                or len(np.unique(labels[train])) < 2
                or len(np.unique(labels[validation])) < 2
            ):
                scope_reports[family] = {
                    "feature_count": int(matrix.shape[1]),
                    "train_auc": 0.5,
                    "validation_auc": 0.5,
                }
            else:
                fitted = fit_train_only_ridge(
                    matrix[train],
                    labels[train],
                    matrix[validation],
                    config.ridge_lambda,
                )
                scope_reports[family] = {
                    "feature_count": int(matrix.shape[1]),
                    "train_auc": _safe_auc(labels[train], fitted["train_scores"]),
                    "validation_auc": _safe_auc(
                        labels[validation], fitted["validation_scores"]
                    ),
                }
        reports[scope] = scope_reports
    return reports


def adjudicate_multibit_profile(
    config: MultibitMaskProfileConfig,
    source_checks: dict[str, bool],
    benchmark: dict[str, Any],
    decomposition: dict[str, Any],
    table: dict[str, Any],
    feature_reports: dict[str, dict[str, dict[str, float | int]]],
) -> dict[str, Any]:
    combined = benchmark["split_metrics"]
    family_metrics = benchmark["family_metrics"]
    reports = decomposition["reports"]
    train_rows = mapped_rows_for_split(benchmark["rows"], "train")
    validation_rows = mapped_rows_for_split(benchmark["rows"], "validation")
    train_structures = {int(row["structure_index"]) for row in train_rows}
    validation_structures = {int(row["structure_index"]) for row in validation_rows}
    protocol_checks = {
        **source_checks,
        "feature_matrices_finite": all(
            bool(np.isfinite(matrix).all()) for matrix in table["matrices"].values()
        ),
        "decomposition_rows_match": len(decomposition["rows"])
        == len(benchmark["rows"]),
    }
    width_checks = {
        "combined_train_each_class_at_least_250": combined["train"]["positive"]
        >= 250
        and combined["train"]["negative"] >= 250,
        "combined_validation_each_class_at_least_80": combined["validation"][
            "positive"
        ]
        >= 80
        and combined["validation"]["negative"] >= 80,
        "combined_train_structures_at_least_48": len(train_structures) >= 48,
        "combined_validation_structures_at_least_16": len(validation_structures)
        >= 16,
        "each_family_train_each_class_at_least_40": all(
            family_metrics[family]["train"]["positive"] >= 40
            and family_metrics[family]["train"]["negative"] >= 40
            for family in MULTIBIT_FAMILIES
        ),
        "each_family_validation_each_class_at_least_12": all(
            family_metrics[family]["validation"]["positive"] >= 12
            and family_metrics[family]["validation"]["negative"] >= 12
            for family in MULTIBIT_FAMILIES
        ),
        "validation_masks_seen_in_train": all(
            {
                int(row["mask_index"])
                for row in benchmark["family_benchmarks"][family]["rows"]
                if row["split"] == "validation"
            }
            <= {
                int(row["mask_index"])
                for row in benchmark["family_benchmarks"][family]["rows"]
                if row["split"] == "train"
            }
            for family in MULTIBIT_FAMILIES
        ),
    }
    shortcut_checks = {
        "duplicate_edges_zero": benchmark["balance"]["duplicate_edges"] == 0,
        "each_structure_balanced": benchmark["balance"][
            "maximum_structure_class_delta"
        ]
        == 0,
        "each_mask_balanced": benchmark["balance"]["maximum_mask_class_delta"]
        == 0,
        "combined_marginal_auc_at_most_0p55": benchmark["marginal_baselines"][
            "strongest_auc"
        ]
        <= 0.55,
        "each_family_marginal_auc_at_most_0p55": all(
            benchmark["family_benchmarks"][family]["marginal_baselines"][
                "strongest_auc"
            ]
            <= 0.55
            for family in MULTIBIT_FAMILIES
        ),
    }
    nontrivial_checks = {
        "combined_raw_nontrivial_positive_fraction_at_least_0p10": reports[
            "combined"
        ]["raw_nontrivial_positive_fraction"]
        >= 0.10,
        "each_family_raw_nontrivial_positive_at_least_8": all(
            reports[family]["raw_nontrivial_positive"] >= 8
            for family in MULTIBIT_FAMILIES
        ),
        "componentwise_validation_auc_at_most_0p80": reports["combined"][
            "validation_componentwise_auc"
        ]
        <= 0.80,
    }
    combined_features = feature_reports["combined"]
    signal_checks = {
        "prefix_or_true_topology_auc_at_least_0p60": max(
            float(combined_features["anf_prefix"]["validation_auc"]),
            float(combined_features["true_topology"]["validation_auc"]),
        )
        >= 0.60,
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_multibit_profile_protocol_invalid"
        action = "repair E43 replay, matching, decomposition, or feature protocol"
    elif not all(nontrivial_checks.values()):
        status = "hold"
        decision = "innovation2_present_multibit_profile_componentwise_dominated"
        action = "stop decoder training; current positives are unit-status compositions"
    elif not all(width_checks.values()):
        status = "hold"
        decision = "innovation2_present_multibit_profile_too_narrow"
        action = "stop the multi-bit mask-query route; family width is insufficient"
    elif not all(shortcut_checks.values()):
        status = "hold"
        decision = "innovation2_present_multibit_profile_marginal_dominated"
        action = "redesign matching before any mask-query decoder"
    elif not all(signal_checks.values()):
        status = "hold"
        decision = "innovation2_present_multibit_profile_signal_not_ready"
        action = "stop decoder training; no nontrivial deterministic signal"
    else:
        status = "pass"
        decision = "innovation2_present_multibit_mask_query_ready"
        action = "prepare a lightweight mask-query decoder readiness plan"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "width_checks": width_checks,
        "shortcut_checks": shortcut_checks,
        "nontrivial_checks": nontrivial_checks,
        "signal_checks": signal_checks,
        "metrics": {
            "split_metrics": combined,
            "family_metrics": family_metrics,
            "marginal_baselines": benchmark["marginal_baselines"],
            "decomposition_reports": reports,
            "feature_reports": feature_reports,
        },
        "claim_scope": (
            "exact nontriviality, shortcut, width, and deterministic-signal audit "
            "for PRESENT-80 r4 strict multi-bit linear-mask labels; no neural "
            "training, high-round, new-attack, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "mask_query_decoder": status == "pass",
            "remote_scale": False,
        },
    }


def result_rows_for_multibit(
    config: MultibitMaskProfileConfig,
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for scope in ("combined", *MULTIBIT_FAMILIES):
        decomposition = gate["metrics"]["decomposition_reports"][scope]
        features = gate["metrics"]["feature_reports"][scope]
        rows.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_present_multibit_mask_profile_readiness",
                "scope": scope,
                **decomposition,
                "static_validation_auc": features["static_set"]["validation_auc"],
                "true_topology_validation_auc": features["true_topology"][
                    "validation_auc"
                ],
                "corrupted_topology_validation_auc": features[
                    "corrupted_topology"
                ]["validation_auc"],
                "anf_prefix_validation_auc": features["anf_prefix"][
                    "validation_auc"
                ],
                "status": gate["status"],
                "decision": gate["decision"],
                "training_performed": False,
            }
        )
    return rows


def _combined_split_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    metrics = {}
    for split in ("train", "validation"):
        selected = mapped_rows_for_split(rows, split)
        metrics[split] = {
            "rows": len(selected),
            "positive": sum(int(row["label"]) == 1 for row in selected),
            "negative": sum(int(row["label"]) == 0 for row in selected),
            "structures": len({int(row["structure_index"]) for row in selected}),
            "masks": len({int(row["mask_index"]) for row in selected}),
        }
    return metrics


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


def serializable_config(config: MultibitMaskProfileConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "MULTIBIT_FAMILIES",
    "MultibitMaskProfileConfig",
    "adjudicate_multibit_profile",
    "build_multibit_benchmark",
    "build_multibit_feature_table",
    "decompose_multibit_labels",
    "evaluate_multibit_features",
    "load_e43_multibit_source",
    "result_rows_for_multibit",
    "serializable_config",
    "validate_multibit_source",
]
