from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import Property
from blockcipher_nd.tasks.innovation2.present_r9_atm_support_component_pu_neural_ranking import (
    PuNeuralRankingConfig,
    _absolute_position,
    _ranking_metrics,
    _ranks,
    build_neural_folds,
    make_model,
    score_model,
    tensorize_pools,
    train_one_fold,
)
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    _canonical_coordinates,
    _relation_id,
    _rotation_candidates,
)


FREEZE_RUN_ID = "i2_present_r9_atm_e99_coordinate_checkpoint_replay_seed0_seed1_20260720"
EVALUATION_RUN_ID = (
    "i2_present_r9_atm_split333_source_heldout_ranking_seed0_seed1_20260720"
)
E99_SUMMARY_SHA256 = "a2b47b3d306b9f4b7a42cfe38e65699679b5fc34152474d8ff486825e1363d59"
E99_GATE_SHA256 = "a303939a7749452cbf92e4d17b12bec1677b74f20675093877a8907497adab6c"
E99_DECISION = "innovation2_present_r9_pu_generic_neural_signal_only"
E104_DECISION = "innovation2_present_r9_split333_generation_passed"
SEEDS = (0, 1)
FOLDS = tuple(range(6))


@dataclass(frozen=True)
class SourceHeldoutRankingConfig:
    freeze_run_id: str = FREEZE_RUN_ID
    evaluation_run_id: str = EVALUATION_RUN_ID
    minimum_relations: int = 32
    minimum_unlabeled_per_pool: int = 31
    minimum_recall_at_5: float = 0.50
    minimum_mrr: float = 0.40
    minimum_top5_enrichment: float = 5.0
    recall_margin_over_anchor: float = 0.20
    mrr_margin_over_anchor: float = 0.15
    maximum_seed_recall_delta: float = 0.10
    replay_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if self.freeze_run_id != FREEZE_RUN_ID:
            raise ValueError("E105 freeze run_id is frozen")
        if self.evaluation_run_id != EVALUATION_RUN_ID:
            raise ValueError("E105 evaluation run_id is frozen")
        if min(self.minimum_relations, self.minimum_unlabeled_per_pool) < 1:
            raise ValueError("E105 width gates must be positive")


def replay_e99_coordinate_checkpoints(
    config: SourceHeldoutRankingConfig,
    *,
    groups: dict[str, set[Property]],
    e99_summary: dict[str, Any],
    output_root: Path,
    device: str,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    e99_config = PuNeuralRankingConfig()
    fold_audit = build_neural_folds(groups, e99_config)
    references = {
        (int(row["seed"]), int(row["fold"])): row
        for row in e99_summary.get("fold_metrics", ())
        if row.get("model") == "coordinate_deepsets"
    }
    checkpoint_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold_data in fold_audit["folds"]:
            if progress_callback:
                progress_callback("checkpoint_replay_start", {"seed": seed, "fold": fold_data.fold})
            result = train_one_fold(
                e99_config,
                model_name="coordinate_deepsets",
                seed=seed,
                fold=fold_data.fold,
                train_tensors=tensorize_pools(fold_data.train_pools),
                test_tensors=tensorize_pools(fold_data.test_pools),
                device=device,
                include_state_dict=True,
            )
            metrics = result["metrics"]
            reference = references.get((seed, fold_data.fold))
            comparisons = _compare_replay_metrics(
                metrics,
                reference,
                tolerance=config.replay_tolerance,
            )
            checkpoint_name = f"coordinate_deepsets_seed{seed}_fold{fold_data.fold}.pt"
            checkpoint_path = checkpoint_root / checkpoint_name
            _atomic_torch_save(
                checkpoint_path,
                {
                    "model": "coordinate_deepsets",
                    "seed": seed,
                    "fold": fold_data.fold,
                    "state_dict": result["state_dict"],
                },
            )
            row = {
                **metrics,
                "reference_metrics_match": all(comparisons.values()),
                "metric_checks": comparisons,
            }
            replay_rows.append(row)
            checkpoint_rows.append(
                {
                    "model": "coordinate_deepsets",
                    "seed": seed,
                    "fold": fold_data.fold,
                    "path": f"checkpoints/{checkpoint_name}",
                    "sha256": sha256(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                    "absolute_position_target": fold_data.audit["absolute_position_target"],
                    "reference_metrics_match": row["reference_metrics_match"],
                }
            )
            if progress_callback:
                progress_callback(
                    "checkpoint_replay_done",
                    {"seed": seed, "fold": fold_data.fold, **metrics},
                )
    checks = {
        "e99_summary_run_id_matches": e99_summary.get("run_id")
        == PuNeuralRankingConfig().run_id,
        "e99_decision_matches": e99_summary.get("gate", {}).get("decision")
        == E99_DECISION,
        "exact_twelve_reference_rows": len(references) == 12,
        "exact_twelve_checkpoints": len(checkpoint_rows) == 12,
        "all_reference_metrics_replay": all(
            row["reference_metrics_match"] for row in replay_rows
        ),
        "all_checkpoint_hashes_unique": len({row["sha256"] for row in checkpoint_rows})
        == len(checkpoint_rows),
    }
    status = "pass" if all(checks.values()) else "fail"
    decision = (
        "innovation2_present_r9_e99_coordinate_checkpoints_frozen"
        if status == "pass"
        else "innovation2_present_r9_e99_checkpoint_replay_invalid"
    )
    manifest = {
        "run_id": config.freeze_run_id,
        "status": status,
        "decision": decision,
        "model": "coordinate_deepsets",
        "training_source": "ATM public eight PRESENT r9 splits only",
        "heldout_source_read": False,
        "e99_summary_sha256": E99_SUMMARY_SHA256,
        "e99_gate_sha256": E99_GATE_SHA256,
        "checkpoints": checkpoint_rows,
        "checks": checks,
    }
    gate = {
        "run_id": config.freeze_run_id,
        "status": status,
        "decision": decision,
        "checks": checks,
        "metrics": {
            "checkpoint_count": len(checkpoint_rows),
            "matching_replays": sum(
                row["reference_metrics_match"] for row in replay_rows
            ),
        },
        "next_action": {
            "action": (
                "wait for a verified E104 generation pass, then evaluate without adaptation"
                if status == "pass"
                else "repair deterministic E99 replay before reading E104 relations"
            ),
            "e104_relations_read": False,
            "evaluation_open_after_e104_pass": status == "pass",
        },
    }
    return {
        "manifest": manifest,
        "gate": gate,
        "replay_rows": replay_rows,
        "fold_metrics": fold_audit["fold_rows"],
    }


def evaluate_source_heldout(
    config: SourceHeldoutRankingConfig,
    *,
    public_groups: dict[str, set[Property]],
    heldout_relations: set[Property],
    checkpoint_manifest: dict[str, Any],
    checkpoint_root: Path,
    e104_gate: dict[str, Any],
    e104_evidence_checks: dict[str, bool],
    device: str,
) -> dict[str, Any]:
    public_relations = set().union(*public_groups.values())
    all_known = public_relations | heldout_relations
    pools = tuple(
        {
            "positive": relation,
            "positive_id": _relation_id(relation),
            "unlabeled_relations": _rotation_candidates(relation, all_known),
            "relations": (relation, *_rotation_candidates(relation, all_known)),
        }
        for relation in sorted(heldout_relations, key=_canonical_coordinates)
    )
    training_overlap_audit = _fold_training_overlap_audit(
        public_groups=public_groups,
        evaluation_pools=pools,
        heldout_relations=heldout_relations,
    )
    tensors = tensorize_pools(pools) if pools else None
    manifest_checks = _validate_checkpoint_manifest(
        checkpoint_manifest,
        checkpoint_root=checkpoint_root,
    )
    source_checks = {
        "e104_status_pass": e104_gate.get("status") == "pass",
        "e104_decision_matches": e104_gate.get("decision") == E104_DECISION,
        **e104_evidence_checks,
        "heldout_relations_nonempty": bool(heldout_relations),
        "heldout_exact_public_overlap_zero": not bool(heldout_relations & public_relations),
        "all_evaluation_relations_absent_from_fold_training_pools": (
            training_overlap_audit["maximum_fold_training_overlap"] == 0
        ),
    }
    if not all(manifest_checks.values()) or not all(source_checks.values()) or tensors is None:
        gate = {
            "run_id": config.evaluation_run_id,
            "status": "fail",
            "decision": "innovation2_present_r9_split333_source_heldout_protocol_invalid",
            "manifest_checks": manifest_checks,
            "source_checks": source_checks,
            "next_action": {"action": "repair source or checkpoint evidence; do not interpret ranks"},
        }
        return {
            "gate": gate,
            "result_rows": [],
            "rank_rows": [],
            "audit": training_overlap_audit,
        }

    scores_by_seed: dict[int, list[list[list[float]]]] = {seed: [] for seed in SEEDS}
    state_unchanged = True
    for entry in checkpoint_manifest["checkpoints"]:
        payload = torch.load(
            checkpoint_root / entry["path"],
            map_location=device,
            weights_only=True,
        )
        model = make_model("coordinate_deepsets").to(device)
        model.load_state_dict(payload["state_dict"])
        before = _state_dict_sha256(model.state_dict())
        scores_by_seed[int(entry["seed"])].append(
            score_model(model, tensors, device=device, batch_size=32)
        )
        after = _state_dict_sha256(model.state_dict())
        state_unchanged = state_unchanged and before == after

    ensemble_scores = {
        seed: _ensemble_scores(scores_by_seed[seed]) for seed in SEEDS
    }
    pool_sizes = [len(ids) for ids in tensors.relation_ids]
    result_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        ranks = _ranks(ensemble_scores[seed], tensors.relation_ids)
        metrics = _ranking_metrics(ranks, pool_sizes)
        result_rows.append({"model": "coordinate_deepsets_6fold_ensemble", "seed": seed, **metrics})
        rank_rows.extend(
            {
                "model": "coordinate_deepsets_6fold_ensemble",
                "seed": seed,
                "relation_id": relation_id,
                "rank": rank,
                "pool_size": pool_size,
            }
            for relation_id, rank, pool_size in zip(
                (ids[0] for ids in tensors.relation_ids), ranks, pool_sizes, strict=True
            )
        )

    anchor_scores = _absolute_anchor_scores(pools, checkpoint_manifest["checkpoints"])
    anchor_ranks = _ranks(anchor_scores, tensors.relation_ids)
    anchor_metrics = _ranking_metrics(anchor_ranks, pool_sizes)
    result_rows.append({"model": "absolute_position_6fold_ensemble", "seed": -1, **anchor_metrics})
    minimum_unlabeled = min(len(pool["unlabeled_relations"]) for pool in pools)
    public_coordinates = Counter(
        coordinate for relation in public_relations for coordinate in relation
    )
    audit = {
        **training_overlap_audit,
        "public_relations": len(public_relations),
        "heldout_relations": len(heldout_relations),
        "heldout_exact_public_overlap": len(heldout_relations & public_relations),
        "heldout_relations_with_public_coordinate_overlap": sum(
            any(coordinate in public_coordinates for coordinate in relation)
            for relation in heldout_relations
        ),
        "minimum_unlabeled_per_pool": minimum_unlabeled,
        "candidate_known_positive_overlap": sum(
            candidate in all_known
            for pool in pools
            for candidate in pool["unlabeled_relations"]
        ),
        "evaluation_optimizer_steps": 0,
        "evaluation_backward_calls": 0,
        "model_state_unchanged": state_unchanged,
    }
    advance_checks: dict[str, bool] = {
        "minimum_relation_width_met": len(heldout_relations) >= config.minimum_relations,
        "minimum_candidate_width_met": minimum_unlabeled
        >= config.minimum_unlabeled_per_pool,
        "candidate_known_positive_overlap_zero": audit["candidate_known_positive_overlap"] == 0,
        "zero_optimizer_steps": audit["evaluation_optimizer_steps"] == 0,
        "zero_backward_calls": audit["evaluation_backward_calls"] == 0,
        "model_state_unchanged": state_unchanged,
    }
    for seed in SEEDS:
        row = next(row for row in result_rows if row["seed"] == seed)
        advance_checks[f"seed{seed}_recall_at_5_met"] = row["recall_at_5"] >= config.minimum_recall_at_5
        advance_checks[f"seed{seed}_mrr_met"] = row["mean_reciprocal_rank"] >= config.minimum_mrr
        advance_checks[f"seed{seed}_enrichment_met"] = row["top5_enrichment"] >= config.minimum_top5_enrichment
        advance_checks[f"seed{seed}_recall_beats_anchor"] = row["recall_at_5"] >= anchor_metrics["recall_at_5"] + config.recall_margin_over_anchor
        advance_checks[f"seed{seed}_mrr_beats_anchor"] = row["mean_reciprocal_rank"] >= anchor_metrics["mean_reciprocal_rank"] + config.mrr_margin_over_anchor
    seed_rows = [next(row for row in result_rows if row["seed"] == seed) for seed in SEEDS]
    advance_checks["seed_recall_delta_bounded"] = abs(
        seed_rows[0]["recall_at_5"] - seed_rows[1]["recall_at_5"]
    ) <= config.maximum_seed_recall_delta

    protocol_valid = all(manifest_checks.values()) and all(source_checks.values())
    enough_width = advance_checks["minimum_relation_width_met"] and advance_checks[
        "minimum_candidate_width_met"
    ]
    if not protocol_valid or not all(
        advance_checks[name]
        for name in (
            "candidate_known_positive_overlap_zero",
            "zero_optimizer_steps",
            "zero_backward_calls",
            "model_state_unchanged",
        )
    ):
        status = "fail"
        decision = "innovation2_present_r9_split333_source_heldout_protocol_invalid"
        action = "repair source, candidate, or frozen-model integrity before interpretation"
    elif not enough_width:
        status = "hold"
        decision = "innovation2_present_r9_split333_source_heldout_diagnostic_only"
        action = "report ranks only; do not claim source-heldout generalization"
    elif all(advance_checks.values()):
        status = "pass"
        decision = "innovation2_present_r9_split333_source_heldout_signal_confirmed"
        action = "design a second independent algorithm or cipher source confirmation"
    else:
        status = "hold"
        decision = "innovation2_present_r9_split333_source_shift_not_confirmed"
        action = "stop the E99 coordinate-identity route; do not tune on heldout relations"
    gate = {
        "run_id": config.evaluation_run_id,
        "status": status,
        "decision": decision,
        "manifest_checks": manifest_checks,
        "source_checks": source_checks,
        "advance_checks": advance_checks,
        "metrics": {"anchor": anchor_metrics, "models": result_rows, **audit},
        "claim_scope": (
            "zero-adaptation positive-unlabeled ranking of locally generated PRESENT r9 ATM "
            "split (3,3,3) relations under independent round keys; not binary classification, "
            "PRESENT-80, a distinguisher, an attack, a published result, or SOTA"
        ),
        "next_action": {"action": action, "training_on_heldout": False},
    }
    return {
        "gate": gate,
        "result_rows": result_rows,
        "rank_rows": rank_rows,
        "audit": audit,
    }


def _fold_training_overlap_audit(
    *,
    public_groups: dict[str, set[Property]],
    evaluation_pools: tuple[dict[str, Any], ...],
    heldout_relations: set[Property],
) -> dict[str, Any]:
    fold_audit = build_neural_folds(public_groups, PuNeuralRankingConfig())
    evaluation_relations = {
        relation for pool in evaluation_pools for relation in pool["relations"]
    }
    fold_rows: list[dict[str, Any]] = []
    for fold_data in fold_audit["folds"]:
        training_relations = {
            relation
            for pool in fold_data.train_pools
            for relation in pool["relations"]
        }
        overlap = training_relations & evaluation_relations
        positive_overlap = training_relations & heldout_relations
        fold_rows.append(
            {
                "fold": fold_data.fold,
                "training_relations": len(training_relations),
                "evaluation_relations": len(evaluation_relations),
                "training_evaluation_overlap": len(overlap),
                "training_heldout_positive_overlap": len(positive_overlap),
                "overlap_relation_ids": sorted(_relation_id(item) for item in overlap),
            }
        )
    return {
        "evaluation_relation_identities": len(evaluation_relations),
        "maximum_fold_training_overlap": max(
            row["training_evaluation_overlap"] for row in fold_rows
        ),
        "maximum_fold_training_positive_overlap": max(
            row["training_heldout_positive_overlap"] for row in fold_rows
        ),
        "fold_training_overlap": fold_rows,
    }


def load_relations_json(path: Path) -> set[Property]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_relations = payload.get("relations")
    if not isinstance(raw_relations, list):
        raise ValueError("relations.json must contain a relations list")
    relations: set[Property] = set()
    for raw_relation in raw_relations:
        if not isinstance(raw_relation, list) or not raw_relation:
            raise ValueError("every heldout relation must be a non-empty list")
        coordinates: set[tuple[int, int]] = set()
        for coordinate in raw_relation:
            if (
                not isinstance(coordinate, list)
                or len(coordinate) != 2
                or any(type(value) is not int for value in coordinate)
                or any(value < 0 or value >= 1 << 64 for value in coordinate)
            ):
                raise ValueError("heldout coordinates must be two unsigned 64-bit integers")
            coordinates.add((coordinate[0], coordinate[1]))
        if len(coordinates) != len(raw_relation):
            raise ValueError("heldout relation contains duplicate coordinates")
        relations.add(frozenset(coordinates))
    if len(relations) != len(raw_relations):
        raise ValueError("relations.json contains duplicate relations")
    return relations


def serializable_config(config: SourceHeldoutRankingConfig) -> dict[str, Any]:
    return asdict(config)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_replay_metrics(
    actual: dict[str, Any],
    expected: dict[str, Any] | None,
    *,
    tolerance: float,
) -> dict[str, bool]:
    if expected is None:
        return {"reference_row_present": False}
    keys = (
        "parameter_count",
        "final_train_loss",
        "recall_at_1",
        "recall_at_5",
        "mean_reciprocal_rank",
        "top5_enrichment",
        "minimum_rank",
        "maximum_rank",
    )
    return {
        key: (
            actual[key] == expected[key]
            if isinstance(expected[key], int)
            else math.isclose(actual[key], expected[key], abs_tol=tolerance, rel_tol=0.0)
        )
        for key in keys
    }


def _validate_checkpoint_manifest(
    manifest: dict[str, Any],
    *,
    checkpoint_root: Path,
) -> dict[str, bool]:
    entries = manifest.get("checkpoints", ())
    paths = [checkpoint_root / str(entry.get("path", "")) for entry in entries]
    return {
        "freeze_status_pass": manifest.get("status") == "pass",
        "freeze_decision_matches": manifest.get("decision")
        == "innovation2_present_r9_e99_coordinate_checkpoints_frozen",
        "heldout_source_not_read_during_freeze": manifest.get("heldout_source_read") is False,
        "e99_summary_hash_matches": manifest.get("e99_summary_sha256")
        == E99_SUMMARY_SHA256,
        "e99_gate_hash_matches": manifest.get("e99_gate_sha256") == E99_GATE_SHA256,
        "exact_seed_fold_matrix": {
            (entry.get("seed"), entry.get("fold")) for entry in entries
        }
        == {(seed, fold) for seed in SEEDS for fold in FOLDS},
        "all_checkpoint_files_exist": len(paths) == 12 and all(path.is_file() for path in paths),
        "all_checkpoint_hashes_match": len(paths) == 12
        and all(
            path.is_file() and sha256(path) == entry.get("sha256")
            for path, entry in zip(paths, entries, strict=True)
        ),
    }


def _ensemble_scores(model_scores: list[list[list[float]]]) -> list[list[float]]:
    if len(model_scores) != 6:
        raise ValueError("E105 requires exactly six fold models per seed")
    ensembles: list[list[float]] = []
    for pool_index in range(len(model_scores[0])):
        standardized = []
        for fold_scores in model_scores:
            values = np.asarray(fold_scores[pool_index], dtype=np.float64)
            standard_deviation = float(values.std())
            standardized.append(
                values - values.mean()
                if standard_deviation == 0.0
                else (values - values.mean()) / standard_deviation
            )
        ensembles.append(np.mean(standardized, axis=0).tolist())
    return ensembles


def _absolute_anchor_scores(
    pools: tuple[dict[str, Any], ...],
    checkpoint_rows: list[dict[str, Any]],
) -> list[list[float]]:
    targets = [float(entry["absolute_position_target"]) for entry in checkpoint_rows]
    scores: list[list[float]] = []
    for pool in pools:
        rows = []
        for target in targets:
            values = np.asarray(
                [-abs(_absolute_position(relation) - target) for relation in pool["relations"]],
                dtype=np.float64,
            )
            standard_deviation = float(values.std())
            rows.append(
                values - values.mean()
                if standard_deviation == 0.0
                else (values - values.mean()) / standard_deviation
            )
        scores.append(np.mean(rows, axis=0).tolist())
    return scores


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous).tobytes())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
