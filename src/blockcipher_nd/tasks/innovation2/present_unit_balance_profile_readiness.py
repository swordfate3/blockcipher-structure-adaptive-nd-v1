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
    RIDGE_LAMBDA,
    anf_prefix_features,
    fit_train_only_ridge,
    static_set_features,
    topology_reachability_features,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    ActiveStructure,
    LinearOutputMask,
    build_checkerboard_benchmark,
    possible_active_monomials,
)
from blockcipher_nd.training.metrics import binary_auc


SOURCE_RUN_ID = "i2_present_r4_universal_balance_atlas_20260718"
SOURCE_DECISION = "innovation2_present_universal_balance_atlas_ready"
PROFILE_OUTPUTS = 64
AUDIT_ATTEMPTS = 64


@dataclass(frozen=True)
class UnitBalanceProfileConfig:
    run_id: str
    mode: str = "audit"
    attempts: int = AUDIT_ATTEMPTS
    ridge_lambda: float = RIDGE_LAMBDA

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.attempts <= 0 or self.ridge_lambda <= 0:
            raise ValueError("attempts and ridge_lambda must be positive")
        if self.mode == "audit" and (
            self.attempts != AUDIT_ATTEMPTS or self.ridge_lambda != RIDGE_LAMBDA
        ):
            raise ValueError("E65 audit protocol is frozen")


def load_e43_unit_source(source_root: Path) -> dict[str, Any]:
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
    all_masks = tuple(
        LinearOutputMask(
            index=int(row["index"]),
            mask_id=str(row["mask_id"]),
            family=str(row["family"]),
            value=int(str(row["mask_hex"]), 16),
        )
        for row in masks_payload["masks"]
    )
    unit_masks = all_masks[:PROFILE_OUTPUTS]
    labels = np.full((len(structures), PROFILE_OUTPUTS), -1, dtype=np.int8)
    seen: set[tuple[int, int]] = set()
    raw_unit_rows = 0
    with (source_root / "atlas.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mask_index = int(row["mask_index"])
            if mask_index >= PROFILE_OUTPUTS:
                continue
            edge = (int(row["structure_index"]), mask_index)
            if edge in seen:
                raise ValueError("duplicate unit edge in E43 atlas")
            seen.add(edge)
            value = -1 if row["label"] is None else int(row["label"])
            labels[edge] = value
            raw_unit_rows += 1
    return {
        "gate": gate,
        "structures": structures,
        "all_masks": all_masks,
        "unit_masks": unit_masks,
        "labels": labels,
        "raw_unit_rows": raw_unit_rows,
        "duplicate_unit_edges": raw_unit_rows - len(seen),
        "source_hashes": {
            name: _sha256(source_root / name)
            for name in ("gate.json", "structures.json", "masks.json", "atlas.jsonl")
        },
    }


def validate_unit_source(source: dict[str, Any], *, strict: bool) -> dict[str, bool]:
    structures = source["structures"]
    masks = source["all_masks"]
    unit_masks = source["unit_masks"]
    labels = np.asarray(source["labels"])
    checks = {
        "source_run_id_matches": source["gate"].get("run_id") == SOURCE_RUN_ID,
        "source_decision_ready": source["gate"].get("decision") == SOURCE_DECISION,
        "source_status_pass": source["gate"].get("status") == "pass",
        "source_hashes_present": all(
            len(value) == 64 for value in source["source_hashes"].values()
        ),
        "unit_masks_are_first_64_singletons": len(unit_masks) == PROFILE_OUTPUTS
        and all(
            mask.index == index
            and mask.family == "unit"
            and mask.value == 1 << index
            for index, mask in enumerate(unit_masks)
        ),
        "unit_label_shape_is_96x64": labels.shape == (96, PROFILE_OUTPUTS),
        "unit_labels_are_ternary": bool(np.isin(labels, (-1, 0, 1)).all()),
        "unit_edges_unique": source["duplicate_unit_edges"] == 0,
    }
    if strict:
        checks.update(
            {
                "structure_count_is_96": len(structures) == 96,
                "mask_count_is_300": len(masks) == 300,
                "raw_unit_rows_are_6144": source["raw_unit_rows"] == 96 * 64,
            }
        )
    return checks


def build_unit_profile_benchmark(
    source: dict[str, Any], *, attempts: int
) -> dict[str, Any]:
    matched = build_checkerboard_benchmark(
        labels=np.asarray(source["labels"]),
        structures=source["structures"],
        masks=source["unit_masks"],
        attempts=attempts,
    )
    targets, observed = assemble_profile_targets(
        matched["rows"], len(source["structures"]), PROFILE_OUTPUTS
    )
    matched["profile_targets"] = targets
    matched["profile_observed"] = observed
    return matched


def assemble_profile_targets(
    rows: list[dict[str, Any]], structure_count: int, output_count: int
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.full((structure_count, output_count), -1, dtype=np.int8)
    observed = np.zeros((structure_count, output_count), dtype=np.bool_)
    for row in rows:
        edge = (int(row["structure_index"]), int(row["mask_index"]))
        if observed[edge]:
            raise ValueError("profile rows contain a duplicate edge")
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError("profile targets must be binary on observed edges")
        targets[edge] = label
        observed[edge] = True
    return targets, observed


def build_profile_feature_table(
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
    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        structure_index = int(row["structure_index"])
        output_bit = int(row["mask_index"])
        active = np.asarray(
            source["structures"][structure_index].active_bits, dtype=np.int64
        )
        selected = np.asarray([output_bit], dtype=np.int64)
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
        output_rows.append(
            {
                "split": row["split"],
                "structure_index": structure_index,
                "output_bit": output_bit,
                "label": int(row["label"]),
            }
        )
    return {
        "rows": output_rows,
        "matrices": matrices,
        "true_player": true_player,
        "corrupted_player": corrupted_player,
    }


def evaluate_profile_features(
    config: UnitBalanceProfileConfig, table: dict[str, Any]
) -> dict[str, dict[str, float | int | bool]]:
    split = np.asarray([row["split"] for row in table["rows"]])
    labels = np.asarray([row["label"] for row in table["rows"]], dtype=np.float64)
    train = split == "train"
    validation = split == "validation"
    reports: dict[str, dict[str, float | int | bool]] = {}
    for family, matrix in table["matrices"].items():
        fitted = fit_train_only_ridge(
            matrix[train], labels[train], matrix[validation], config.ridge_lambda
        )
        reports[family] = {
            "feature_count": int(matrix.shape[1]),
            "train_auc": _safe_auc(labels[train], fitted["train_scores"]),
            "validation_auc": _safe_auc(
                labels[validation], fitted["validation_scores"]
            ),
            "train_standardization_only": True,
        }
    return reports


def adjudicate_unit_profile(
    config: UnitBalanceProfileConfig,
    source_checks: dict[str, bool],
    benchmark: dict[str, Any],
    table: dict[str, Any],
    reports: dict[str, dict[str, float | int | bool]],
) -> dict[str, Any]:
    split_metrics = benchmark["split_metrics"]
    train_rows = [row for row in benchmark["rows"] if row["split"] == "train"]
    validation_rows = [
        row for row in benchmark["rows"] if row["split"] == "validation"
    ]
    train_outputs = {int(row["mask_index"]) for row in train_rows}
    validation_outputs = {int(row["mask_index"]) for row in validation_rows}
    train_structures = {int(row["structure_index"]) for row in train_rows}
    validation_structures = {
        int(row["structure_index"]) for row in validation_rows
    }
    true_auc = float(reports["true_topology"]["validation_auc"])
    corrupted_auc = float(reports["corrupted_topology"]["validation_auc"])
    prefix_auc = float(reports["anf_prefix"]["validation_auc"])
    protocol_checks = {
        **source_checks,
        "profile_shape_is_96x64": benchmark["profile_targets"].shape
        == (96, PROFILE_OUTPUTS),
        "observed_shape_is_96x64": benchmark["profile_observed"].shape
        == (96, PROFILE_OUTPUTS),
        "observed_targets_are_binary": bool(
            np.isin(
                benchmark["profile_targets"][benchmark["profile_observed"]],
                (0, 1),
            ).all()
        ),
        "unobserved_targets_are_minus_one": bool(
            np.all(
                benchmark["profile_targets"][~benchmark["profile_observed"]] == -1
            )
        ),
        "feature_matrices_finite": all(
            bool(np.isfinite(matrix).all()) for matrix in table["matrices"].values()
        ),
        "true_corrupted_players_distinct": not np.array_equal(
            table["true_player"], table["corrupted_player"]
        ),
        "all_ridge_use_train_standardization": all(
            bool(report["train_standardization_only"])
            for report in reports.values()
        ),
    }
    width_checks = {
        "train_each_class_at_least_150": split_metrics["train"]["positive"] >= 150
        and split_metrics["train"]["negative"] >= 150,
        "validation_each_class_at_least_50": split_metrics["validation"][
            "positive"
        ]
        >= 50
        and split_metrics["validation"]["negative"] >= 50,
        "train_structures_at_least_48": len(train_structures) >= 48,
        "validation_structures_at_least_16": len(validation_structures) >= 16,
        "train_outputs_at_least_24": len(train_outputs) >= 24,
        "validation_outputs_at_least_16": len(validation_outputs) >= 16,
        "validation_outputs_seen_in_train": validation_outputs <= train_outputs,
        "train_validation_structures_disjoint": train_structures.isdisjoint(
            validation_structures
        ),
    }
    shortcut_checks = {
        "duplicate_edges_zero": benchmark["balance"]["duplicate_edges"] == 0,
        "each_structure_balanced": benchmark["balance"][
            "maximum_structure_class_delta"
        ]
        == 0,
        "each_output_balanced": benchmark["balance"]["maximum_mask_class_delta"]
        == 0,
        "strongest_marginal_auc_at_most_0p55": benchmark["marginal_baselines"][
            "strongest_auc"
        ]
        <= 0.55,
    }
    route_checks = {
        "topology_profile_route": {
            "true_topology_auc_at_least_0p60": true_auc >= 0.60,
            "true_minus_corrupted_at_least_0p03": true_auc - corrupted_auc >= 0.03,
            "true_at_least_prefix_minus_0p02": true_auc >= prefix_auc - 0.02,
        },
        "prefix_guided_profile_route": {
            "prefix_auc_at_least_0p60": prefix_auc >= 0.60,
            "prefix_minus_true_at_least_0p02": prefix_auc - true_auc >= 0.02,
        },
    }
    passing = [name for name, checks in route_checks.items() if all(checks.values())]
    selected_route = max(
        passing,
        key=lambda name: true_auc if name == "topology_profile_route" else prefix_auc,
        default=None,
    )
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_present_unit_balance_profile_protocol_invalid"
        action = "repair E43 replay, profile relayout, feature, or split protocol"
    elif not all(width_checks.values()):
        status = "hold"
        decision = "innovation2_present_unit_balance_profile_too_narrow"
        action = "stop the joint profile route; unit outputs are too narrow"
    elif not all(shortcut_checks.values()):
        status = "hold"
        decision = "innovation2_present_unit_balance_profile_shortcut_dominated"
        action = "redesign profile matching before neural training"
    elif selected_route == "topology_profile_route":
        status = "pass"
        decision = "innovation2_present_unit_balance_profile_topology_ready"
        action = "prepare E66 shared-round nodewise topology operator readiness"
    elif selected_route == "prefix_guided_profile_route":
        status = "pass"
        decision = "innovation2_present_unit_balance_profile_prefix_ready"
        action = "prepare E66 prefix-guided shared-round nodewise operator readiness"
    else:
        status = "hold"
        decision = "innovation2_present_unit_balance_profile_signal_not_ready"
        action = "stop the joint profile route; no nontrivial deterministic signal"
    metrics = {
        "split_metrics": split_metrics,
        "train_structures": len(train_structures),
        "validation_structures": len(validation_structures),
        "train_outputs": len(train_outputs),
        "validation_outputs": len(validation_outputs),
        "shared_outputs": len(train_outputs & validation_outputs),
        "observed_edges": int(np.sum(benchmark["profile_observed"])),
        "marginal_baselines": benchmark["marginal_baselines"],
        "feature_reports": reports,
        "true_minus_corrupted_topology": true_auc - corrupted_auc,
        "prefix_minus_true_topology": prefix_auc - true_auc,
        "selected_route": selected_route,
    }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "width_checks": width_checks,
        "shortcut_checks": shortcut_checks,
        "route_checks": route_checks,
        "metrics": metrics,
        "claim_scope": (
            "PRESENT-80 r4 strict unit-output balance labels relaid as a masked "
            "64-coordinate profile with deterministic architecture routing; no "
            "neural training, high-round distinguisher, new attack, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "network_readiness": status == "pass",
            "remote_scale": False,
        },
    }


def result_rows_for_profile(
    config: UnitBalanceProfileConfig,
    gate: dict[str, Any],
    reports: dict[str, dict[str, float | int | bool]],
) -> list[dict[str, Any]]:
    rows = []
    for family, report in reports.items():
        rows.append(
            {
                "run_id": config.run_id,
                "task": "innovation2_present_unit_balance_profile_readiness",
                "feature_family": family,
                **report,
                "status": gate["status"],
                "decision": gate["decision"],
                "training_performed": False,
            }
        )
    return rows


def serializable_config(config: UnitBalanceProfileConfig) -> dict[str, Any]:
    return asdict(config)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(binary_auc(labels.astype(np.float32), scores.astype(np.float64)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "UnitBalanceProfileConfig",
    "adjudicate_unit_profile",
    "assemble_profile_targets",
    "build_profile_feature_table",
    "build_unit_profile_benchmark",
    "evaluate_profile_features",
    "load_e43_unit_source",
    "result_rows_for_profile",
    "serializable_config",
    "validate_unit_source",
]
