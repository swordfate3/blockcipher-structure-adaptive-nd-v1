from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_zero_step import (
    FROZEN_MODEL_OPTIONS,
    PARAMETER_COUNT,
    TARGET_MODELS,
    TARGET_VALIDATION_KEY,
    _intervention_sha256,
    _source_row,
    _structure_sha256,
    _validate_source_row,
    file_sha256,
    validate_target_result_rows,
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
ROLE_SPECS = {
    "true_source_true_target": ("true", "true"),
    "corrupted_source_true_target": ("corrupted", "true"),
    "true_source_corrupted_target": ("true", "corrupted"),
    "random_source_true_target": ("random", "true"),
}
TRAINABLE_PARAMETER_COUNT = 198_401
HEAD_INITIALIZATION_SEED = 24_071_101
RANDOM_BACKBONE_SEED = 24_071_201
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
AUC_FLOOR = 0.55
MARGIN_FLOOR = 0.005
FULL_TARGET_ANCHOR_AUCS = {
    0: 0.6127333641052246,
    1: 0.6145439147949219,
}


def deterministic_classifier_state() -> dict[str, torch.Tensor]:
    with torch.random.fork_rng():
        torch.manual_seed(HEAD_INITIALIZATION_SEED)
        model = _build_target_model("true")
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
        raise ValueError(f"unsupported X2 seed: {seed}")
    if source_role not in {"true", "corrupted", "random"}:
        raise ValueError(f"unsupported X2 source role: {source_role}")
    if target_mode not in {"true", "corrupted"}:
        raise ValueError(f"unsupported X2 target mode: {target_mode}")

    if source_role == "random":
        with torch.random.fork_rng():
            torch.manual_seed(RANDOM_BACKBONE_SEED + seed)
            model = _build_target_model(target_mode)
    else:
        model = _build_target_model(target_mode)
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


def train_adaptation_seed(
    *,
    seed: int,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    train_dataset: DifferentialDataset,
    validation_dataset: DifferentialDataset,
    train_paths: Mapping[str, Path],
    validation_paths: Mapping[str, Path],
    checkpoint_dir: Path,
    device: str = "cpu",
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unsupported X2 seed: {seed}")
    validate_target_result_rows(target_rows, seed)
    _validate_target_datasets(seed, train_dataset, validation_dataset)
    _validate_dataset_paths(train_paths, validation_paths)

    source_records = {
        role: _source_row(source_rows, seed, model)
        for role, model in SOURCE_MODELS.items()
    }
    source_payloads: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, Path] = {}
    for role, row in source_records.items():
        _validate_source_row(row, seed)
        checkpoint_path = Path(row["training"]["checkpoint_output"])
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("state_dict"), dict
        ):
            raise ValueError("source checkpoint must contain a state_dict")
        source_payloads[role] = payload
        source_paths[role] = checkpoint_path

    classifier_state = deterministic_classifier_state()
    classifier_initial_sha256 = tensor_mapping_sha256(classifier_state)
    full_target_anchor_auc = _full_target_anchor_auc(target_rows, seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, role in enumerate(EXPECTED_ROLES, start=1):
        source_role, target_mode = ROLE_SPECS[role]
        source_state_dicts = {
            key: value["state_dict"] for key, value in source_payloads.items()
        }
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
            raise ValueError("classifier initialization changed across X2 roles")

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
            raise ValueError("X2 best checkpoint replay does not match restored model")
        final_backbone_sha256 = model_backbone_sha256(model)
        final_head_sha256 = tensor_mapping_sha256(
            model.backbone.classifier.state_dict()
        )
        source_checkpoint_path = (
            source_paths[source_role] if source_role != "random" else None
        )
        source_checkpoint_sha256 = (
            file_sha256(source_checkpoint_path)
            if source_checkpoint_path is not None
            else None
        )
        runtime_structure_sha256 = _structure_sha256(model.runtime_structure)
        rows.append(
            {
                "seed": seed,
                "role": role,
                "source_role": source_role,
                "target_mode": target_mode,
                "source_checkpoint_path": (
                    str(source_checkpoint_path)
                    if source_checkpoint_path is not None
                    else None
                ),
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "source_selected_checkpoint": (
                    source_payloads[source_role]
                    .get("metadata", {})
                    .get("selected_checkpoint")
                    if source_role != "random"
                    else None
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
                "parameter_count": _parameter_count(model),
                "trainable_parameter_count": _trainable_parameter_count(model),
                "trainable_parameter_names": _trainable_parameter_names(model),
                "auc": float(result.final_metrics["auc"]),
                "accuracy": float(result.final_metrics["accuracy"]),
                "loss": float(result.final_metrics["loss"]),
                "history": result.history,
                "training": result.metadata,
                "full_target_anchor_auc": full_target_anchor_auc,
                "candidate_minus_full_target_anchor_auc": None,
                "train_feature_sha256": file_sha256(train_paths["features"]),
                "train_label_sha256": file_sha256(train_paths["labels"]),
                "train_metadata_sha256": file_sha256(train_paths["metadata"]),
                "validation_feature_sha256": file_sha256(validation_paths["features"]),
                "validation_label_sha256": file_sha256(validation_paths["labels"]),
                "validation_metadata_sha256": file_sha256(validation_paths["metadata"]),
                "source_cipher": "GIFT-64",
                "source_rounds": 6,
                "target_cipher": "SKINNY-64/64",
                "target_rounds": 7,
                "target_difference": 0x2000,
                "target_train_key": 0,
                "target_validation_key": TARGET_VALIDATION_KEY,
                "train_rows": 4096,
                "validation_rows": 2048,
                "pairs_per_sample": 4,
                "input_bits": 512,
                "negative_mode": "encrypted_random_plaintexts",
                "model_options": FROZEN_MODEL_OPTIONS,
                "strict_state_dict_load": True,
                "backbone_frozen": True,
            }
        )

    candidate_auc = next(
        row["auc"] for row in rows if row["role"] == "true_source_true_target"
    )
    for row in rows:
        row["candidate_minus_full_target_anchor_auc"] = (
            candidate_auc - full_target_anchor_auc
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
            and row.get("strict_state_dict_load") is True
            for row in rows
        ),
        "frozen_target_protocol": all(
            row.get("source_cipher") == "GIFT-64"
            and row.get("source_rounds") == 6
            and row.get("target_cipher") == "SKINNY-64/64"
            and row.get("target_rounds") == 7
            and row.get("target_difference") == 0x2000
            and row.get("target_train_key") == 0
            and row.get("target_validation_key") == TARGET_VALIDATION_KEY
            and row.get("train_rows") == 4096
            and row.get("validation_rows") == 2048
            and row.get("pairs_per_sample") == 4
            and row.get("input_bits") == 512
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("model_options") == FROZEN_MODEL_OPTIONS
            for row in rows
        ),
        "full_target_anchor_exact": all(
            row.get("full_target_anchor_auc")
            == FULL_TARGET_ANCHOR_AUCS.get(row.get("seed"))
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
        decision = "runtime_spn_target_head_protocol_invalid"
        next_action = (
            "repair evidence only without changing X2 data, roles, or thresholds"
        )
    elif all(research_checks.values()):
        status = "pass"
        decision = "runtime_spn_frozen_backbone_target_head_supported"
        next_action = (
            "compare frozen-backbone target-head adaptation against formal SKINNY "
            "scale in a separate route audit; do not launch medium adaptation automatically"
        )
    else:
        status = "hold"
        decision = "runtime_spn_target_head_signal_unstable"
        next_action = "stop X2 scaling and retain the need for cipher-specific full-target training"

    return {
        "run_id": run_id,
        "task": "innovation1_runtime_spn_frozen_backbone_target_head_x2",
        "status": status,
        "decision": decision,
        "thresholds": {"candidate_auc": AUC_FLOOR, "auc_margin": MARGIN_FLOOR},
        "seed_results": seed_results,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "claim_scope": (
            "small GIFT-to-SKINNY frozen-backbone target-head diagnostic; no zero-shot, "
            "formal-scale, universal-SPN, attack, SOTA, or breakthrough claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "unfreeze the Runtime-E4 backbone inside X2",
            "change target data, keys, negatives, epochs, roles, or thresholds",
            "launch a medium or formal adaptation run without a separate plan",
        ],
    }


def model_backbone_sha256(model: torch.nn.Module) -> str:
    return tensor_mapping_sha256(
        {
            name: tensor
            for name, tensor in model.state_dict().items()
            if not name.startswith("backbone.classifier.")
        }
    )


def tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _build_target_model(target_mode: str) -> torch.nn.Module:
    return build_model(
        TARGET_MODELS[target_mode],
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=FROZEN_MODEL_OPTIONS,
    )


def _validate_parameter_ownership(model: torch.nn.Module) -> None:
    if _parameter_count(model) != PARAMETER_COUNT:
        raise ValueError("X2 total parameter count changed")
    if _trainable_parameter_count(model) != TRAINABLE_PARAMETER_COUNT:
        raise ValueError("X2 trainable classifier parameter count changed")
    names = _trainable_parameter_names(model)
    if not names or any(not name.startswith("backbone.classifier.") for name in names):
        raise ValueError("X2 may train only backbone.classifier parameters")


def _validate_target_datasets(
    seed: int,
    train_dataset: DifferentialDataset,
    validation_dataset: DifferentialDataset,
) -> None:
    expected_common = {
        "cipher": "SKINNY-64/64",
        "rounds": 7,
        "input_difference": 0x2000,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "input_bits": 512,
        "structure": "SPN",
    }
    expected_split = (
        (train_dataset, seed, 4096, 2048),
        (validation_dataset, 10000 + seed, 2048, 1024),
    )
    for dataset, expected_seed, total, per_class in expected_split:
        metadata = dataset.metadata
        if any(
            metadata.get(field) != value for field, value in expected_common.items()
        ):
            raise ValueError("X2 target dataset protocol changed")
        if (
            metadata.get("seed") != expected_seed
            or metadata.get("samples_total") != total
            or metadata.get("samples_per_class") != per_class
            or metadata.get("positive_rows") != per_class
            or metadata.get("negative_rows") != per_class
            or dataset.features.shape != (total, 512)
            or dataset.labels.shape != (total,)
        ):
            raise ValueError("X2 target dataset split geometry changed")


def _validate_dataset_paths(
    train_paths: Mapping[str, Path], validation_paths: Mapping[str, Path]
) -> None:
    for paths in (train_paths, validation_paths):
        if set(paths) != {"features", "labels", "metadata"}:
            raise ValueError(
                "X2 dataset paths must identify features, labels, metadata"
            )
        if any(not path.is_file() for path in paths.values()):
            raise ValueError("X2 dataset evidence path is missing")


def _full_target_anchor_auc(target_rows: list[dict[str, Any]], seed: int) -> float:
    matches = [
        row
        for row in target_rows
        if row.get("model") == TARGET_MODELS["true"] and row.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError("expected one full-target SKINNY anchor row")
    auc = float(matches[0].get("metrics", {}).get("auc"))
    if auc != FULL_TARGET_ANCHOR_AUCS[seed]:
        raise ValueError("full-target SKINNY anchor AUC changed")
    return auc


def _source_checkpoint_control_exact(group: Mapping[str, list[dict[str, Any]]]) -> bool:
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


def _target_structure_control_exact(group: Mapping[str, list[dict[str, Any]]]) -> bool:
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
        and candidate.get("target_mode") == "true"
        and source_control.get("target_mode") == "true"
        and random_control.get("target_mode") == "true"
        and target_control.get("target_mode") == "corrupted"
        and target_control.get("runtime_structure_sha256") not in true_hashes
        and all(
            row.get("target_relation_mode") == "true"
            for row in (candidate, source_control, target_control, random_control)
        )
    )


def _seed_result(group: Mapping[str, list[dict[str, Any]]]) -> dict[str, float | None]:
    def value(role: str, field: str = "auc") -> float | None:
        rows = group.get(role, ())
        if len(rows) != 1:
            return None
        raw = rows[0].get(field)
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
    "adjudicate_head_adaptation",
    "deterministic_classifier_state",
    "model_backbone_sha256",
    "prepare_adaptation_model",
    "tensor_mapping_sha256",
    "train_adaptation_seed",
]
