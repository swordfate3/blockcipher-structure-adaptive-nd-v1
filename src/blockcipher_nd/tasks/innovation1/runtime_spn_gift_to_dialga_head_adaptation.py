from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.evaluation.runtime_spn_representation import (
    extract_runtime_e4_representation,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_head_adaptation import (
    model_backbone_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_zero_step import (
    FROZEN_MODEL_OPTIONS as SOURCE_MODEL_OPTIONS,
    PARAMETER_COUNT,
    _intervention_sha256,
    _source_row,
    _structure_sha256,
    _validate_source_row,
    file_sha256,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d1 import (
    adjudicate_runtime_spn_dialga_d1,
)
from blockcipher_nd.training import TrainingConfig, train_binary_classifier


EXPECTED_SEEDS = (0, 1)
EXPECTED_ROLES = (
    "true_source_true_target",
    "corrupted_source_true_target",
    "true_source_corrupted_target",
    "random_source_true_target",
)
SOURCE_MODELS = {
    "true": "gift64_runtime_e4_equivariant_true",
    "corrupted": "gift64_runtime_e4_equivariant_corrupted",
}
TARGET_MODELS = {
    "true": "runtime_spn_e4_equivariant_true",
    "corrupted": "runtime_spn_e4_equivariant_corrupted",
}
ROLE_SPECS = {
    "true_source_true_target": ("true", "true"),
    "corrupted_source_true_target": ("corrupted", "true"),
    "true_source_corrupted_target": ("true", "corrupted"),
    "random_source_true_target": ("random", "true"),
}
TARGET_BASE_MODEL_OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/dialga128.json",
    "runtime_round_start": 2,
    "runtime_rounds": 2,
    "processor_steps": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "sbox_context_mode": "edge_gate",
    "cell_input_mode": "state_triplet",
    "round_window_mode": "recurrent_window",
    "runtime_structure_window_control": "full",
}
TARGET_CORRUPTION_SEED = 20_260_725
TARGET_VALIDATION_KEY = int("11" * 32, 16)
TRAINABLE_PARAMETER_COUNT = 198_401
REPRESENTATION_WIDTH = 384
HEAD_INITIALIZATION_SEED = 25_071_101
RANDOM_BACKBONE_SEED = 25_071_201
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
AUC_FLOOR = 0.55
MARGIN_FLOOR = 0.005
FULL_TARGET_ANCHOR_AUCS = {
    0: 0.9584169387817383,
    1: 0.95867919921875,
}
D1_RUN_ID = "i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725"
D1_DECISION = "innovation1_dialga_runtime_e4_d1_two_seed_supported"


def target_model_options(target_mode: str) -> dict[str, Any]:
    if target_mode not in TARGET_MODELS:
        raise ValueError(f"unsupported X3 target mode: {target_mode}")
    options = dict(TARGET_BASE_MODEL_OPTIONS)
    if target_mode == "corrupted":
        options["topology_corruption_seed"] = TARGET_CORRUPTION_SEED
    return options


def build_target_model(target_mode: str) -> torch.nn.Module:
    return build_model(
        TARGET_MODELS[target_mode],
        input_bits=1024,
        hidden_bits=64,
        pair_bits=256,
        structure="SPN",
        model_options=target_model_options(target_mode),
    )


def deterministic_classifier_state() -> dict[str, torch.Tensor]:
    with torch.random.fork_rng():
        torch.manual_seed(HEAD_INITIALIZATION_SEED)
        model = build_target_model("true")
    return _clone_state(model.backbone.classifier.state_dict())


def prepare_adaptation_model(
    *,
    seed: int,
    source_role: str,
    target_mode: str,
    source_state_dicts: Mapping[str, Mapping[str, torch.Tensor]],
    classifier_state: Mapping[str, torch.Tensor],
) -> torch.nn.Module:
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unsupported X3 seed: {seed}")
    if source_role not in {"true", "corrupted", "random"}:
        raise ValueError(f"unsupported X3 source role: {source_role}")
    if target_mode not in TARGET_MODELS:
        raise ValueError(f"unsupported X3 target mode: {target_mode}")

    if source_role == "random":
        with torch.random.fork_rng():
            torch.manual_seed(RANDOM_BACKBONE_SEED + seed)
            model = build_target_model(target_mode)
    else:
        model = build_target_model(target_mode)
        try:
            source_state = source_state_dicts[source_role]
        except KeyError as exc:
            raise ValueError(f"missing {source_role} source state") from exc
        model.load_state_dict(source_state, strict=True)

    model.backbone.classifier.load_state_dict(classifier_state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.backbone.classifier.parameters():
        parameter.requires_grad_(True)
    _validate_parameter_ownership(model)
    return model


def validate_source_evidence(
    roots: tuple[Path, Path],
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    payloads: dict[int, dict[str, dict[str, Any]]] = {}
    evidence: dict[str, Any] = {}
    for seed, root in zip(EXPECTED_SEEDS, roots, strict=True):
        results_path = root / "results.jsonl"
        rows = _read_jsonl(results_path)
        source_rows = {
            role: _source_row(rows, seed, model)
            for role, model in SOURCE_MODELS.items()
        }
        seed_payloads: dict[str, dict[str, Any]] = {}
        seed_evidence: dict[str, Any] = {
            "root": str(root),
            "results_sha256": file_sha256(results_path),
            "roles": {},
        }
        for role, row in source_rows.items():
            _validate_source_row(row, seed)
            if row.get("training", {}).get("model_options") != SOURCE_MODEL_OPTIONS:
                raise ValueError("X3 source model options changed")
            checkpoint_path = Path(row["training"]["checkpoint_output"])
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(checkpoint, dict) or not isinstance(
                checkpoint.get("state_dict"), dict
            ):
                raise ValueError("X3 source checkpoint must contain a state_dict")
            if checkpoint.get("metadata", {}).get("selected_checkpoint") != "best":
                raise ValueError("X3 source checkpoint is not restored-best evidence")
            seed_payloads[role] = checkpoint
            seed_evidence["roles"][role] = {
                "model": row["model"],
                "auc": float(row["metrics"]["auc"]),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "state_dict_sha256": tensor_mapping_sha256(checkpoint["state_dict"]),
                "selected_checkpoint": "best",
            }
        if (
            seed_evidence["roles"]["true"]["checkpoint_sha256"]
            == seed_evidence["roles"]["corrupted"]["checkpoint_sha256"]
        ):
            raise ValueError("X3 true and corrupted source checkpoints are identical")
        payloads[seed] = seed_payloads
        evidence[f"seed{seed}"] = seed_evidence
    return payloads, evidence


def validate_d1_evidence(
    target_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results_path = target_root / "results.jsonl"
    gate_path = target_root / "gate.json"
    validation_path = target_root / "validation.json"
    rows = _read_jsonl(results_path)
    persisted_gate = _read_json(gate_path)
    persisted_validation = _read_json(validation_path)
    replayed_gate = adjudicate_runtime_spn_dialga_d1(run_id=D1_RUN_ID, rows=rows)
    if replayed_gate != persisted_gate:
        raise ValueError("X3 D1 gate does not replay byte-semantically")
    if (
        persisted_gate.get("status") != "pass"
        or persisted_gate.get("decision") != D1_DECISION
        or not all(persisted_gate.get("protocol_checks", {}).values())
        or not all(persisted_gate.get("research_checks", {}).values())
    ):
        raise ValueError("X3 requires the complete passing D1 gate")
    if (
        persisted_validation.get("run_id") != D1_RUN_ID
        or persisted_validation.get("status") != "pass"
        or persisted_validation.get("checks") != persisted_gate["protocol_checks"]
    ):
        raise ValueError("X3 D1 validation does not match the replayed gate")
    for seed, expected_auc in FULL_TARGET_ANCHOR_AUCS.items():
        matches = [
            row
            for row in rows
            if row.get("seed") == seed and row.get("model") == TARGET_MODELS["true"]
        ]
        if len(matches) != 1 or float(matches[0]["metrics"]["auc"]) != expected_auc:
            raise ValueError("X3 D1 full-target anchor changed")
    return rows, {
        "root": str(target_root),
        "results_sha256": file_sha256(results_path),
        "gate_sha256": file_sha256(gate_path),
        "validation_sha256": file_sha256(validation_path),
        "run_id": D1_RUN_ID,
        "status": "pass",
        "decision": D1_DECISION,
        "gate_replay_exact": True,
        "validation_replay_exact": True,
    }


def audit_strict_load_matrix(
    source_payloads: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for source_role in SOURCE_MODELS:
            source_state = source_payloads[seed][source_role]["state_dict"]
            for target_mode in TARGET_MODELS:
                model = build_target_model(target_mode)
                model.load_state_dict(source_state, strict=True)
                audits.append(
                    {
                        "seed": seed,
                        "source_role": source_role,
                        "target_mode": target_mode,
                        "strict_load": True,
                        "parameter_count": _parameter_count(model),
                        "state_dict_key_count": len(model.state_dict()),
                        "source_state_dict_sha256": tensor_mapping_sha256(source_state),
                    }
                )
    return audits


def audit_role_readiness(
    *,
    seed: int,
    role: str,
    source_state_dicts: Mapping[str, Mapping[str, torch.Tensor]],
    classifier_state: Mapping[str, torch.Tensor],
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    source_role, target_mode = ROLE_SPECS[role]
    model = prepare_adaptation_model(
        seed=seed,
        source_role=source_role,
        target_mode=target_mode,
        source_state_dicts=source_state_dicts,
        classifier_state=classifier_state,
    )
    model.train()
    initial_backbone_sha256 = model_backbone_sha256(model)
    initial_classifier_sha256 = tensor_mapping_sha256(
        model.backbone.classifier.state_dict()
    )
    batch = extract_runtime_e4_representation(model, features)
    representation = batch.representation
    logits = batch.logits
    representation_finite = bool(torch.isfinite(representation).all().item())
    representation_nonconstant = bool(
        representation.numel() > 1 and torch.var(representation).item() > 0.0
    )
    logits_finite = bool(torch.isfinite(logits).all().item())
    logits_nonconstant = bool(logits.numel() > 1 and torch.var(logits).item() > 0.0)

    optimizer = torch.optim.Adam(
        model.backbone.classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(
        torch.sigmoid(model(features).squeeze(1)),
        labels,
    )
    loss.backward()
    classifier_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.classifier.")
    ]
    frozen_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.classifier.")
    ]
    classifier_gradients_finite = bool(
        classifier_gradients
        and all(
            gradient is not None and torch.isfinite(gradient).all().item()
            for gradient in classifier_gradients
        )
    )
    classifier_gradient_l1 = float(
        sum(
            gradient.detach().abs().sum().item()
            for gradient in classifier_gradients
            if gradient is not None
        )
    )
    frozen_gradients_absent = all(gradient is None for gradient in frozen_gradients)
    optimizer.step()
    final_backbone_sha256 = model_backbone_sha256(model)
    final_classifier_sha256 = tensor_mapping_sha256(
        model.backbone.classifier.state_dict()
    )
    runtime_structure_sha256 = _structure_sha256(model.runtime_structure)
    return {
        "seed": seed,
        "role": role,
        "source_role": source_role,
        "target_mode": target_mode,
        "strict_state_dict_load": (True if source_role in SOURCE_MODELS else None),
        "parameter_count": _parameter_count(model),
        "trainable_parameter_count": _trainable_parameter_count(model),
        "trainable_parameter_names": _trainable_parameter_names(model),
        "classifier_initial_sha256": initial_classifier_sha256,
        "classifier_final_sha256": final_classifier_sha256,
        "backbone_initial_sha256": initial_backbone_sha256,
        "backbone_final_sha256": final_backbone_sha256,
        "runtime_structure_sha256": runtime_structure_sha256,
        "runtime_intervention_sha256": _intervention_sha256(
            runtime_structure_sha256,
            str(model.relation_mode),
        ),
        "target_relation_mode": str(model.relation_mode),
        "representation_shape": list(representation.shape),
        "representation_finite": representation_finite,
        "representation_nonconstant": representation_nonconstant,
        "representation_std": float(representation.std().item()),
        "logits_finite": logits_finite,
        "logits_nonconstant": logits_nonconstant,
        "classifier_gradients_finite": classifier_gradients_finite,
        "classifier_gradient_l1": classifier_gradient_l1,
        "frozen_gradients_absent": frozen_gradients_absent,
        "loss_finite": math.isfinite(float(loss.item())),
        "backbone_unchanged_after_step": (
            initial_backbone_sha256 == final_backbone_sha256
        ),
        "classifier_changed_after_step": (
            initial_classifier_sha256 != final_classifier_sha256
        ),
    }


def adjudicate_readiness(
    *,
    run_id: str,
    role_audits: Iterable[dict[str, Any]],
    source_evidence: dict[str, Any],
    d1_evidence: dict[str, Any],
    cache_evidence: dict[str, Any],
    strict_load_audits: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    role_audits = list(role_audits)
    strict_load_audits = list(strict_load_audits)
    grouped = {(row.get("seed"), row.get("role")): row for row in role_audits}
    expected = {(seed, role) for seed in EXPECTED_SEEDS for role in EXPECTED_ROLES}
    classifier_hashes = {row.get("classifier_initial_sha256") for row in role_audits}
    source_hashes_distinct = all(
        source_evidence[f"seed{seed}"]["roles"]["true"]["checkpoint_sha256"]
        != source_evidence[f"seed{seed}"]["roles"]["corrupted"]["checkpoint_sha256"]
        for seed in EXPECTED_SEEDS
    )
    checks = {
        "source_evidence_complete": set(source_evidence) == {"seed0", "seed1"}
        and source_hashes_distinct,
        "d1_gate_and_validation_replay_exact": (
            d1_evidence.get("gate_replay_exact") is True
            and d1_evidence.get("validation_replay_exact") is True
            and d1_evidence.get("decision") == D1_DECISION
        ),
        "four_exact_cache_leaves": cache_evidence.get("leaf_count") == 4,
        "cache_geometry_exact": cache_evidence.get("geometry_exact") is True,
        "cache_files_unchanged": cache_evidence.get("unchanged") is True,
        "eight_role_audits_complete": len(role_audits) == 8
        and set(grouped) == expected,
        "strict_cross_cipher_state_load": all(
            row.get("strict_load") is True
            and row.get("parameter_count") == PARAMETER_COUNT
            and row.get("state_dict_key_count") == 54
            and _is_sha256(row.get("source_state_dict_sha256"))
            for row in strict_load_audits
        )
        and len(strict_load_audits) == 8
        and {
            (row.get("seed"), row.get("source_role"), row.get("target_mode"))
            for row in strict_load_audits
        }
        == {
            (seed, source_role, target_mode)
            for seed in EXPECTED_SEEDS
            for source_role in SOURCE_MODELS
            for target_mode in TARGET_MODELS
        },
        "parameter_ownership_exact": all(
            row.get("parameter_count") == PARAMETER_COUNT
            and row.get("trainable_parameter_count") == TRAINABLE_PARAMETER_COUNT
            and row.get("trainable_parameter_names")
            and all(
                str(name).startswith("backbone.classifier.")
                for name in row.get("trainable_parameter_names", ())
            )
            for row in role_audits
        ),
        "common_classifier_initialization": len(classifier_hashes) == 1
        and all(_is_sha256(value) for value in classifier_hashes),
        "target_structures_exact": all(
            _readiness_target_structure_exact(grouped, seed) for seed in EXPECTED_SEEDS
        ),
        "representations_finite_nonconstant_384": all(
            row.get("representation_shape", [None, None])[1] == REPRESENTATION_WIDTH
            and row.get("representation_finite") is True
            and row.get("representation_nonconstant") is True
            and row.get("logits_finite") is True
            and row.get("logits_nonconstant") is True
            for row in role_audits
        ),
        "classifier_only_finite_nonzero_gradients": all(
            row.get("classifier_gradients_finite") is True
            and float(row.get("classifier_gradient_l1", 0.0)) > 0.0
            and row.get("frozen_gradients_absent") is True
            and row.get("loss_finite") is True
            for row in role_audits
        ),
        "disposable_step_respects_frozen_boundary": all(
            row.get("backbone_unchanged_after_step") is True
            and row.get("classifier_changed_after_step") is True
            for row in role_audits
        ),
        "sha256_evidence_complete": all(
            _is_sha256(row.get(field))
            for row in role_audits
            for field in (
                "classifier_initial_sha256",
                "classifier_final_sha256",
                "backbone_initial_sha256",
                "backbone_final_sha256",
                "runtime_structure_sha256",
                "runtime_intervention_sha256",
            )
        ),
    }
    passed = all(checks.values())
    return {
        "run_id": run_id,
        "task": "innovation1_runtime_spn_gift_to_dialga_x3_readiness",
        "status": "pass" if passed else "fail",
        "decision": (
            "runtime_spn_gift_to_dialga_x3_readiness_supported"
            if passed
            else "runtime_spn_gift_to_dialga_x3_readiness_not_supported"
        ),
        "checks": checks,
        "source_evidence": source_evidence,
        "d1_evidence": d1_evidence,
        "cache_evidence": cache_evidence,
        "role_audits": role_audits,
        "strict_load_audits": strict_load_audits,
        "next_action": (
            "run the frozen two-seed four-role five-epoch X3 matrix"
            if passed
            else "stop before training; do not add compatibility layers or regenerate data"
        ),
    }


def train_adaptation_seed(
    *,
    seed: int,
    source_payloads: Mapping[str, Mapping[str, Any]],
    source_evidence: Mapping[str, Any],
    target_rows: list[dict[str, Any]],
    train_dataset: DifferentialDataset,
    validation_dataset: DifferentialDataset,
    train_paths: Mapping[str, Path],
    validation_paths: Mapping[str, Path],
    readiness_sha256: str,
    checkpoint_dir: Path,
    device: str = "cpu",
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unsupported X3 seed: {seed}")
    _validate_target_rows(target_rows, seed)
    _validate_target_datasets(seed, train_dataset, validation_dataset)
    _validate_dataset_paths(train_paths, validation_paths)
    classifier_state = deterministic_classifier_state()
    classifier_initial_sha256 = tensor_mapping_sha256(classifier_state)
    source_state_dicts = {
        role: payload["state_dict"] for role, payload in source_payloads.items()
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, role in enumerate(EXPECTED_ROLES, start=1):
        source_role, target_mode = ROLE_SPECS[role]
        model = prepare_adaptation_model(
            seed=seed,
            source_role=source_role,
            target_mode=target_mode,
            source_state_dicts=source_state_dicts,
            classifier_state=classifier_state,
        )
        initial_backbone_sha256 = model_backbone_sha256(model)
        initial_head_sha256 = tensor_mapping_sha256(
            model.backbone.classifier.state_dict()
        )
        if initial_head_sha256 != classifier_initial_sha256:
            raise ValueError("classifier initialization changed across X3 roles")
        checkpoint_path = checkpoint_dir / f"seed{seed}_{role}.pt"

        def emit(event: str, payload: dict[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback(
                    event,
                    {"seed": seed, "role": role, "row_index": index, **payload},
                )

        result = train_binary_classifier(
            model,
            train_dataset,
            validation_dataset,
            TrainingConfig(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                seed=seed,
                device=device,
                optimizer="adam",
                weight_decay=WEIGHT_DECAY,
                lr_scheduler="none",
                checkpoint_metric="val_auc",
                restore_best_checkpoint=True,
                loss="mse",
                train_eval_interval=1,
                checkpoint_output=checkpoint_path,
            ),
            progress_callback=emit,
        )
        checkpoint_payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint_replay_verified = bool(
            isinstance(checkpoint_payload, dict)
            and isinstance(checkpoint_payload.get("state_dict"), dict)
            and tensor_mapping_sha256(checkpoint_payload["state_dict"])
            == tensor_mapping_sha256(model.state_dict())
            and checkpoint_payload.get("final_metrics") == result.final_metrics
            and checkpoint_payload.get("metadata", {}).get("selected_checkpoint")
            == "best"
        )
        if not checkpoint_replay_verified:
            raise ValueError("X3 best checkpoint replay does not match restored model")
        final_backbone_sha256 = model_backbone_sha256(model)
        final_head_sha256 = tensor_mapping_sha256(
            model.backbone.classifier.state_dict()
        )
        source_record = (
            source_evidence["roles"][source_role] if source_role != "random" else None
        )
        runtime_structure_sha256 = _structure_sha256(model.runtime_structure)
        rows.append(
            {
                "seed": seed,
                "role": role,
                "source_role": source_role,
                "target_mode": target_mode,
                "source_checkpoint_path": (
                    source_record["checkpoint_path"] if source_record else None
                ),
                "source_checkpoint_sha256": (
                    source_record["checkpoint_sha256"] if source_record else None
                ),
                "source_state_dict_sha256": (
                    source_record["state_dict_sha256"] if source_record else None
                ),
                "source_selected_checkpoint": (
                    source_record["selected_checkpoint"] if source_record else None
                ),
                "runtime_structure_sha256": runtime_structure_sha256,
                "runtime_intervention_sha256": _intervention_sha256(
                    runtime_structure_sha256,
                    str(model.relation_mode),
                ),
                "target_relation_mode": str(model.relation_mode),
                "classifier_initial_sha256": initial_head_sha256,
                "classifier_final_sha256": final_head_sha256,
                "backbone_initial_sha256": initial_backbone_sha256,
                "backbone_final_sha256": final_backbone_sha256,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "checkpoint_replay_verified": checkpoint_replay_verified,
                "readiness_sha256": readiness_sha256,
                "target_cache_tree_sha256_before": None,
                "target_cache_tree_sha256_after": None,
                "target_cache_unchanged": None,
                "parameter_count": _parameter_count(model),
                "trainable_parameter_count": _trainable_parameter_count(model),
                "trainable_parameter_names": _trainable_parameter_names(model),
                "auc": float(result.final_metrics["auc"]),
                "accuracy": float(result.final_metrics["accuracy"]),
                "loss": float(result.final_metrics["loss"]),
                "history": result.history,
                "training": result.metadata,
                "full_target_anchor_auc": FULL_TARGET_ANCHOR_AUCS[seed],
                "candidate_minus_full_target_anchor_auc": None,
                "train_feature_sha256": file_sha256(train_paths["features"]),
                "train_label_sha256": file_sha256(train_paths["labels"]),
                "train_metadata_sha256": file_sha256(train_paths["metadata"]),
                "validation_feature_sha256": file_sha256(validation_paths["features"]),
                "validation_label_sha256": file_sha256(validation_paths["labels"]),
                "validation_metadata_sha256": file_sha256(validation_paths["metadata"]),
                "source_cipher": "GIFT-64",
                "source_rounds": 6,
                "target_cipher": "Dialga-128",
                "target_rounds": 4,
                "target_difference": 0x40,
                "target_train_key": 0,
                "target_validation_key": TARGET_VALIDATION_KEY,
                "train_rows": 4096,
                "validation_rows": 2048,
                "pairs_per_sample": 4,
                "input_bits": 1024,
                "pair_bits": 256,
                "negative_mode": "encrypted_random_plaintexts",
                "source_model_options": SOURCE_MODEL_OPTIONS,
                "target_model_options": target_model_options(target_mode),
                "strict_state_dict_load": (
                    True if source_role in SOURCE_MODELS else None
                ),
                "backbone_frozen": True,
            }
        )

    candidate_auc = next(
        row["auc"] for row in rows if row["role"] == "true_source_true_target"
    )
    for row in rows:
        row["candidate_minus_full_target_anchor_auc"] = (
            candidate_auc - FULL_TARGET_ANCHOR_AUCS[seed]
        )
    return rows


def adjudicate_head_adaptation(
    *,
    run_id: str,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row.get("seed", -1))][str(row.get("role"))].append(row)
    complete = all(
        len(grouped[seed].get(role, ())) == 1
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
    )
    seed_results = {str(seed): _seed_result(grouped[seed]) for seed in EXPECTED_SEEDS}
    protocol_checks = {
        "eight_rows_complete": len(rows) == 8 and complete,
        "two_seed_four_role_panel": complete and set(grouped) == set(EXPECTED_SEEDS),
        "same_data_within_seed": complete
        and all(
            all(
                len({grouped[seed][role][0].get(field) for role in EXPECTED_ROLES}) == 1
                for field in (
                    "train_feature_sha256",
                    "train_label_sha256",
                    "train_metadata_sha256",
                    "validation_feature_sha256",
                    "validation_label_sha256",
                    "validation_metadata_sha256",
                )
            )
            for seed in EXPECTED_SEEDS
        ),
        "same_classifier_initialization_all_roles": complete
        and len({row.get("classifier_initial_sha256") for row in rows}) == 1,
        "source_checkpoint_attribution_exact": complete
        and all(
            _source_checkpoint_control_exact(grouped[seed]) for seed in EXPECTED_SEEDS
        ),
        "target_structure_attribution_exact": complete
        and all(
            _target_structure_control_exact(grouped[seed]) for seed in EXPECTED_SEEDS
        ),
        "frozen_backbone_unchanged": all(
            row.get("backbone_frozen") is True
            and row.get("backbone_initial_sha256") == row.get("backbone_final_sha256")
            for row in rows
        ),
        "classifier_updated": all(
            row.get("classifier_initial_sha256") != row.get("classifier_final_sha256")
            for row in rows
        ),
        "parameter_ownership_exact": all(
            row.get("parameter_count") == PARAMETER_COUNT
            and row.get("trainable_parameter_count") == TRAINABLE_PARAMETER_COUNT
            and all(
                str(name).startswith("backbone.classifier.")
                for name in row.get("trainable_parameter_names", ())
            )
            for row in rows
        ),
        "five_epoch_best_checkpoint_training": all(
            len(row.get("history", ())) == EPOCHS
            and row.get("training", {}).get("epochs") == EPOCHS
            and row.get("training", {}).get("epochs_ran") == EPOCHS
            and row.get("training", {}).get("batch_size") == BATCH_SIZE
            and row.get("training", {}).get("optimizer") == "adam"
            and row.get("training", {}).get("learning_rate") == LEARNING_RATE
            and row.get("training", {}).get("weight_decay") == WEIGHT_DECAY
            and row.get("training", {}).get("loss") == "mse"
            and row.get("training", {}).get("checkpoint_metric") == "val_auc"
            and row.get("training", {}).get("selected_checkpoint") == "best"
            and row.get("checkpoint_replay_verified") is True
            and (
                (
                    row.get("source_role") in SOURCE_MODELS
                    and row.get("strict_state_dict_load") is True
                )
                or (
                    row.get("source_role") == "random"
                    and row.get("strict_state_dict_load") is None
                )
            )
            for row in rows
        ),
        "frozen_target_protocol": all(
            row.get("source_cipher") == "GIFT-64"
            and row.get("source_rounds") == 6
            and row.get("target_cipher") == "Dialga-128"
            and row.get("target_rounds") == 4
            and row.get("target_difference") == 0x40
            and row.get("target_train_key") == 0
            and row.get("target_validation_key") == TARGET_VALIDATION_KEY
            and row.get("train_rows") == 4096
            and row.get("validation_rows") == 2048
            and row.get("pairs_per_sample") == 4
            and row.get("input_bits") == 1024
            and row.get("pair_bits") == 256
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("source_model_options") == SOURCE_MODEL_OPTIONS
            and row.get("target_model_options")
            == target_model_options(str(row.get("target_mode")))
            for row in rows
        ),
        "full_target_anchor_exact": all(
            row.get("full_target_anchor_auc")
            == FULL_TARGET_ANCHOR_AUCS.get(row.get("seed"))
            for row in rows
        ),
        "readiness_provenance_exact": len({row.get("readiness_sha256") for row in rows})
        == 1
        and all(_is_sha256(row.get("readiness_sha256")) for row in rows),
        "target_cache_unchanged": all(
            row.get("target_cache_unchanged") is True
            and row.get("target_cache_tree_sha256_before")
            == row.get("target_cache_tree_sha256_after")
            and _is_sha256(row.get("target_cache_tree_sha256_before"))
            for row in rows
        ),
        "finite_metrics": all(
            _finite(row.get(field))
            for row in rows
            for field in ("auc", "accuracy", "loss")
        ),
        "sha256_evidence_present": all(
            _is_sha256(row.get(field))
            for row in rows
            for field in (
                "runtime_structure_sha256",
                "runtime_intervention_sha256",
                "classifier_initial_sha256",
                "classifier_final_sha256",
                "backbone_initial_sha256",
                "backbone_final_sha256",
                "checkpoint_sha256",
                "train_feature_sha256",
                "train_label_sha256",
                "train_metadata_sha256",
                "validation_feature_sha256",
                "validation_label_sha256",
                "validation_metadata_sha256",
            )
        ),
    }

    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        result = seed_results[str(seed)]
        research_checks[f"seed{seed}_candidate_auc_at_least_0p55"] = bool(
            result["candidate_auc"] is not None and result["candidate_auc"] >= AUC_FLOOR
        )
        for control in ("source", "target", "random"):
            research_checks[f"seed{seed}_beats_{control}_by_0p005"] = bool(
                result[f"candidate_minus_{control}_auc"] is not None
                and result[f"candidate_minus_{control}_auc"] >= MARGIN_FLOOR
            )

    if not all(protocol_checks.values()):
        status = "fail"
        decision = "runtime_spn_gift_to_dialga_x3_protocol_invalid"
        next_action = (
            "repair evidence only without changing X3 data, roles, or thresholds"
        )
    elif all(research_checks.values()):
        status = "pass"
        decision = "runtime_spn_gift_to_dialga_x3_shared_backbone_supported"
        next_action = (
            "synthesize X2 and X3 without new training to bound the supported "
            "cross-cipher Runtime-E4 claim"
        )
    else:
        status = "hold"
        decision = "runtime_spn_gift_to_dialga_x3_signal_not_supported"
        next_action = (
            "stop X3 scaling and retain GIFT-to-SKINNY X2 as the current "
            "cross-cipher boundary"
        )
    return {
        "run_id": run_id,
        "task": "innovation1_runtime_spn_gift_to_dialga_frozen_backbone_x3",
        "status": status,
        "decision": decision,
        "thresholds": {"candidate_auc": AUC_FLOOR, "auc_margin": MARGIN_FLOOR},
        "seed_results": seed_results,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "claim_scope": (
            "small GIFT-to-Dialga cross-block-size frozen-backbone target-head "
            "diagnostic; no zero-shot, formal-scale, universal-SPN, attack, SOTA, "
            "or breakthrough claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "unfreeze or resize the Runtime-E4 backbone inside X3",
            "change target data, keys, negatives, epochs, roles, or thresholds",
            "launch remote, medium, or formal X3 scaling",
            "resume Runtime-E5 as a rescue",
        ],
    }


def cache_tree_snapshot(cache_root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(cache_root)): {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": file_sha256(path),
        }
        for path in sorted(cache_root.rglob("*"))
        if path.is_file()
    }


def cache_evidence(
    *,
    paths_by_seed: Mapping[int, Mapping[str, Mapping[str, Path]]],
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    leaves = {
        str(paths["metadata"].parent)
        for seed_paths in paths_by_seed.values()
        for paths in seed_paths.values()
    }
    return {
        "leaf_count": len(leaves),
        "leaves": sorted(leaves),
        "geometry_exact": len(leaves) == 4,
        "file_count_before": len(before),
        "file_count_after": len(after),
        "tree_sha256_before": _snapshot_sha256(before),
        "tree_sha256_after": _snapshot_sha256(after),
        "unchanged": before == after,
        "generation_events": 0,
    }


def _validate_target_rows(rows: list[dict[str, Any]], seed: int) -> None:
    matches = [row for row in rows if row.get("seed") == seed]
    if len(matches) != 3 or {row.get("model") for row in matches} != {
        "runtime_spn_e4_equivariant_true",
        "runtime_spn_e4_equivariant_corrupted",
        "runtime_spn_e4_equivariant_independent",
    }:
        raise ValueError("X3 target D1 rows are incomplete")
    for row in matches:
        if not (
            row.get("cipher") == "Dialga-128"
            and row.get("cipher_key") == "dialga128"
            and row.get("rounds") == 4
            and row.get("pairs_per_sample") == 4
            and row.get("input_difference") == 0x40
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("sample_structure") == "independent_pairs"
            and row.get("validation_key") == TARGET_VALIDATION_KEY
            and row.get("training", {}).get("input_bits") == 1024
            and row.get("training", {}).get("pair_bits") == 256
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
        ):
            raise ValueError("X3 target D1 protocol changed")


def _validate_target_datasets(
    seed: int,
    train_dataset: DifferentialDataset,
    validation_dataset: DifferentialDataset,
) -> None:
    common = {
        "cipher": "Dialga-128",
        "rounds": 4,
        "input_difference": 0x40,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "input_bits": 1024,
        "structure": "SPN",
    }
    for dataset, expected_seed, total, per_class in (
        (train_dataset, seed, 4096, 2048),
        (validation_dataset, 10_000 + seed, 2048, 1024),
    ):
        metadata = dataset.metadata
        if any(metadata.get(field) != value for field, value in common.items()):
            raise ValueError("X3 Dialga target dataset protocol changed")
        if not (
            metadata.get("seed") == expected_seed
            and metadata.get("samples_total") == total
            and metadata.get("samples_per_class") == per_class
            and metadata.get("positive_rows") == per_class
            and metadata.get("negative_rows") == per_class
            and dataset.features.shape == (total, 1024)
            and dataset.labels.shape == (total,)
        ):
            raise ValueError("X3 Dialga target dataset geometry changed")


def _validate_dataset_paths(
    train_paths: Mapping[str, Path], validation_paths: Mapping[str, Path]
) -> None:
    for paths in (train_paths, validation_paths):
        if set(paths) != {"features", "labels", "metadata"}:
            raise ValueError(
                "X3 dataset paths must identify features, labels, metadata"
            )
        if any(not path.is_file() for path in paths.values()):
            raise ValueError("X3 dataset evidence path is missing")


def _readiness_target_structure_exact(
    grouped: Mapping[tuple[Any, Any], dict[str, Any]], seed: int
) -> bool:
    try:
        candidate = grouped[(seed, "true_source_true_target")]
        source_control = grouped[(seed, "corrupted_source_true_target")]
        target_control = grouped[(seed, "true_source_corrupted_target")]
        random_control = grouped[(seed, "random_source_true_target")]
    except KeyError:
        return False
    true_hashes = {
        row.get("runtime_structure_sha256")
        for row in (candidate, source_control, random_control)
    }
    return bool(
        len(true_hashes) == 1
        and target_control.get("runtime_structure_sha256") not in true_hashes
        and candidate.get("target_mode") == "true"
        and source_control.get("target_mode") == "true"
        and random_control.get("target_mode") == "true"
        and target_control.get("target_mode") == "corrupted"
    )


def _source_checkpoint_control_exact(
    group: Mapping[str, list[dict[str, Any]]],
) -> bool:
    try:
        candidate = group["true_source_true_target"][0]
        source_control = group["corrupted_source_true_target"][0]
        target_control = group["true_source_corrupted_target"][0]
        random_control = group["random_source_true_target"][0]
    except (KeyError, IndexError):
        return False
    return bool(
        _is_sha256(candidate.get("source_checkpoint_sha256"))
        and candidate.get("source_checkpoint_sha256")
        == target_control.get("source_checkpoint_sha256")
        and candidate.get("source_checkpoint_sha256")
        != source_control.get("source_checkpoint_sha256")
        and _is_sha256(source_control.get("source_checkpoint_sha256"))
        and candidate.get("source_selected_checkpoint") == "best"
        and source_control.get("source_selected_checkpoint") == "best"
        and target_control.get("source_selected_checkpoint") == "best"
        and random_control.get("source_checkpoint_sha256") is None
        and random_control.get("source_selected_checkpoint") is None
    )


def _target_structure_control_exact(
    group: Mapping[str, list[dict[str, Any]]],
) -> bool:
    try:
        candidate = group["true_source_true_target"][0]
        source_control = group["corrupted_source_true_target"][0]
        target_control = group["true_source_corrupted_target"][0]
        random_control = group["random_source_true_target"][0]
    except (KeyError, IndexError):
        return False
    true_hashes = {
        row.get("runtime_structure_sha256")
        for row in (candidate, source_control, random_control)
    }
    return bool(
        len(true_hashes) == 1
        and target_control.get("runtime_structure_sha256") not in true_hashes
        and candidate.get("target_mode") == "true"
        and source_control.get("target_mode") == "true"
        and random_control.get("target_mode") == "true"
        and target_control.get("target_mode") == "corrupted"
        and all(
            row.get("target_relation_mode") == "true"
            for row in (candidate, source_control, target_control, random_control)
        )
    )


def _seed_result(group: Mapping[str, list[dict[str, Any]]]) -> dict[str, float | None]:
    def value(role: str, field: str = "auc") -> float | None:
        records = group.get(role, ())
        if len(records) != 1:
            return None
        raw = records[0].get(field)
        return float(raw) if _finite(raw) else None

    candidate = value("true_source_true_target")
    source = value("corrupted_source_true_target")
    target = value("true_source_corrupted_target")
    random = value("random_source_true_target")
    anchor = value("true_source_true_target", "full_target_anchor_auc")
    return {
        "candidate_auc": candidate,
        "corrupted_source_auc": source,
        "corrupted_target_auc": target,
        "random_frozen_auc": random,
        "full_target_anchor_auc": anchor,
        "candidate_minus_source_auc": _difference(candidate, source),
        "candidate_minus_target_auc": _difference(candidate, target),
        "candidate_minus_random_auc": _difference(candidate, random),
        "candidate_minus_full_target_anchor_auc": _difference(candidate, anchor),
    }


def _snapshot_sha256(snapshot: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for path, metadata in sorted(snapshot.items()):
        digest.update(path.encode("utf-8"))
        digest.update(str(metadata["size"]).encode("ascii"))
        digest.update(str(metadata["mtime_ns"]).encode("ascii"))
        digest.update(str(metadata["sha256"]).encode("ascii"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]


def _validate_parameter_ownership(model: torch.nn.Module) -> None:
    if _parameter_count(model) != PARAMETER_COUNT:
        raise ValueError("X3 total parameter count changed")
    if _trainable_parameter_count(model) != TRAINABLE_PARAMETER_COUNT:
        raise ValueError("X3 trainable classifier parameter count changed")
    names = _trainable_parameter_names(model)
    if not names or any(not name.startswith("backbone.classifier.") for name in names):
        raise ValueError("X3 may train only backbone.classifier parameters")


def _clone_state(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in values.items()}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "EXPECTED_ROLES",
    "EXPECTED_SEEDS",
    "FULL_TARGET_ANCHOR_AUCS",
    "TRAINABLE_PARAMETER_COUNT",
    "adjudicate_head_adaptation",
    "adjudicate_readiness",
    "audit_role_readiness",
    "audit_strict_load_matrix",
    "build_target_model",
    "cache_evidence",
    "cache_tree_snapshot",
    "deterministic_classifier_state",
    "prepare_adaptation_model",
    "target_model_options",
    "train_adaptation_seed",
    "validate_d1_evidence",
    "validate_source_evidence",
]
