from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.evaluation.runtime_spn_representation import (
    FrozenRuntimeE4HeadAdapter,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_head_adaptation import (
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_rectangle_attribution import (
    INPUT_DIFFERENCE as TARGET_INPUT_DIFFERENCE,
)
from blockcipher_nd.training import TrainingConfig, train_binary_classifier


RUN_ID = "i1_skinny_formal_to_rectangle_frozen_representation_x3a_2048_seed0_20260725"
EXPECTED_ROLES = (
    "true_source_true_target",
    "corrupted_source_true_target",
    "true_source_corrupted_target",
    "random_source_true_target",
)
ROLE_SPECS = {
    "true_source_true_target": ("true", "true"),
    "corrupted_source_true_target": ("corrupted", "true"),
    "true_source_corrupted_target": ("true", "corrupted"),
    "random_source_true_target": ("random", "true"),
}
SOURCE_MODELS = {
    "true": "skinny64_runtime_e4_equivariant_true",
    "corrupted": "skinny64_runtime_e4_equivariant_corrupted",
}
TARGET_MODELS = {
    "true": "runtime_spn_e4_equivariant_true",
    "corrupted": "runtime_spn_e4_equivariant_corrupted",
}
TARGET_MODEL_OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/rectangle64.json",
    "runtime_rounds": 2,
    "processor_steps": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "sbox_context_mode": "late_pair",
}
SOURCE_CHECKPOINT_SHA256S = {
    "true": "edb4b37a74eb876164a14a8f4924607e6c31616b616810e9d43c32b13e816cc1",
    "corrupted": "797217c85b84edd507a66c9675c1752a65dec13478fa3b574ef95ae99e325f42",
}
SOURCE_AUCS = {
    "true": 0.653191631304,
    "corrupted": 0.607162432806,
}
TARGET_ANCHOR_AUC = 0.791468620300293
HEAD_INITIALIZATION_SEED = 25_070_301
RANDOM_EXTRACTOR_SEED = 25_070_302
EXTRACTOR_PARAMETER_COUNT = 442_466
TARGET_HEAD_PARAMETER_COUNT = 198_401
TOTAL_PARAMETER_COUNT = EXTRACTOR_PARAMETER_COUNT + TARGET_HEAD_PARAMETER_COUNT
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
AUC_FLOOR = 0.55
MARGIN_FLOOR = 0.005


def prepare_transfer_model(
    *,
    source_role: str,
    target_mode: str,
    source_state_dicts: Mapping[str, Mapping[str, torch.Tensor]],
) -> FrozenRuntimeE4HeadAdapter:
    if source_role not in {"true", "corrupted", "random"}:
        raise ValueError(f"unsupported X3-A source role: {source_role}")
    if target_mode not in TARGET_MODELS:
        raise ValueError(f"unsupported X3-A target mode: {target_mode}")

    if source_role == "random":
        with torch.random.fork_rng():
            torch.manual_seed(RANDOM_EXTRACTOR_SEED)
            extractor = _build_target_extractor(target_mode)
    else:
        extractor = _build_target_extractor(target_mode)
        try:
            source_state = source_state_dicts[source_role]
        except KeyError as exc:
            raise ValueError(f"missing {source_role} source state") from exc
        extractor.load_state_dict(source_state, strict=True)

    model = FrozenRuntimeE4HeadAdapter(
        extractor,
        deterministic_target_head(),
    )
    _validate_parameter_ownership(model)
    return model


def deterministic_target_head() -> torch.nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(HEAD_INITIALIZATION_SEED)
        template = _build_target_extractor("true")
    return deepcopy(template.backbone.classifier)


def train_transfer_panel(
    *,
    source_rows: list[dict[str, Any]],
    source_checkpoint_paths: Mapping[str, Path],
    target_rows: list[dict[str, Any]],
    train_dataset: DifferentialDataset,
    validation_dataset: DifferentialDataset,
    train_paths: Mapping[str, Path],
    validation_paths: Mapping[str, Path],
    checkpoint_dir: Path,
    device: str = "cpu",
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    _validate_source_rows(source_rows)
    target_anchor = _validate_target_rows(target_rows)
    _validate_target_datasets(train_dataset, validation_dataset)
    _validate_dataset_paths(train_paths, validation_paths)

    source_payloads: dict[str, dict[str, Any]] = {}
    for role in ("true", "corrupted"):
        try:
            checkpoint_path = source_checkpoint_paths[role]
        except KeyError as exc:
            raise ValueError(f"missing {role} source checkpoint path") from exc
        if not checkpoint_path.is_file():
            raise ValueError(f"missing {role} source checkpoint")
        if file_sha256(checkpoint_path) != SOURCE_CHECKPOINT_SHA256S[role]:
            raise ValueError(f"{role} source checkpoint SHA-256 changed")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("state_dict"), dict
        ):
            raise ValueError("source checkpoint must contain a state_dict")
        if payload.get("metadata", {}).get("selected_checkpoint") != "best":
            raise ValueError("source checkpoint must be the selected best checkpoint")
        source_payloads[role] = payload

    source_state_dicts = {
        role: payload["state_dict"] for role, payload in source_payloads.items()
    }
    head_initial_state = deterministic_target_head().state_dict()
    head_initial_sha256 = tensor_mapping_sha256(head_initial_state)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for row_index, role in enumerate(EXPECTED_ROLES, start=1):
        source_role, target_mode = ROLE_SPECS[role]
        model = prepare_transfer_model(
            source_role=source_role,
            target_mode=target_mode,
            source_state_dicts=source_state_dicts,
        )
        extractor_initial = tensor_mapping_sha256(model.feature_extractor.state_dict())
        source_classifier_initial = tensor_mapping_sha256(
            model.feature_extractor.backbone.classifier.state_dict()
        )
        target_head_initial = tensor_mapping_sha256(model.target_head.state_dict())
        if target_head_initial != head_initial_sha256:
            raise ValueError("target-head initialization changed across roles")
        checkpoint_path = checkpoint_dir / f"seed0_{role}.pt"

        def emit(event: str, payload: dict[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback(
                    event,
                    {"seed": 0, "role": role, "row_index": row_index, **payload},
                )

        result = train_binary_classifier(
            model,
            train_dataset,
            validation_dataset,
            TrainingConfig(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                seed=0,
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
            raise ValueError("X3-A best checkpoint replay does not match model")

        extractor_final = tensor_mapping_sha256(model.feature_extractor.state_dict())
        source_classifier_final = tensor_mapping_sha256(
            model.feature_extractor.backbone.classifier.state_dict()
        )
        target_head_final = tensor_mapping_sha256(model.target_head.state_dict())
        source_checkpoint_path = (
            source_checkpoint_paths[source_role] if source_role != "random" else None
        )
        rows.append(
            {
                "seed": 0,
                "role": role,
                "source_role": source_role,
                "target_mode": target_mode,
                "source_checkpoint_path": (
                    str(source_checkpoint_path)
                    if source_checkpoint_path is not None
                    else None
                ),
                "source_checkpoint_sha256": (
                    SOURCE_CHECKPOINT_SHA256S[source_role]
                    if source_role != "random"
                    else None
                ),
                "source_selected_checkpoint": (
                    "best" if source_role != "random" else None
                ),
                "source_auc": (
                    SOURCE_AUCS[source_role] if source_role != "random" else None
                ),
                "strict_source_state_dict_load": True,
                "target_structure_sha256": model.runtime_structure.window_sha256(),
                "target_relation_mode": model.relation_mode,
                "target_head_initial_sha256": target_head_initial,
                "target_head_final_sha256": target_head_final,
                "feature_extractor_initial_sha256": extractor_initial,
                "feature_extractor_final_sha256": extractor_final,
                "source_classifier_initial_sha256": source_classifier_initial,
                "source_classifier_final_sha256": source_classifier_final,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "checkpoint_replay_verified": checkpoint_replay_verified,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "trainable_parameter_names": [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ],
                "adapter_mode": model.adapter_mode,
                "feature_extractor_frozen": model.feature_extractor_frozen,
                "source_classifier_preserved": model.source_classifier_preserved,
                "runtime_structure_descriptor_name": getattr(
                    model, "runtime_structure_descriptor_name", None
                ),
                "runtime_structure_descriptor_sha256": getattr(
                    model, "runtime_structure_descriptor_sha256", None
                ),
                "runtime_structure_window_sha256": getattr(
                    model, "runtime_structure_window_sha256", None
                ),
                "runtime_structure_mode": getattr(
                    model, "runtime_structure_mode", None
                ),
                "auc": float(result.final_metrics["auc"]),
                "accuracy": float(result.final_metrics["accuracy"]),
                "loss": float(result.final_metrics["loss"]),
                "history": result.history,
                "training": result.metadata,
                "full_target_anchor_auc": target_anchor,
                "candidate_minus_full_target_anchor_auc": None,
                "train_feature_sha256": file_sha256(train_paths["features"]),
                "train_label_sha256": file_sha256(train_paths["labels"]),
                "train_metadata_sha256": file_sha256(train_paths["metadata"]),
                "validation_feature_sha256": file_sha256(validation_paths["features"]),
                "validation_label_sha256": file_sha256(validation_paths["labels"]),
                "validation_metadata_sha256": file_sha256(validation_paths["metadata"]),
                "source_cipher": "SKINNY-64/64",
                "source_rounds": 7,
                "source_samples_per_class": 1_000_000,
                "target_cipher": "RECTANGLE-80",
                "target_rounds": 6,
                "target_difference": TARGET_INPUT_DIFFERENCE,
                "target_train_key": 0,
                "target_validation_key": 0x11111111111111111111,
                "train_rows": 4096,
                "validation_rows": 2048,
                "pairs_per_sample": 4,
                "input_bits": 512,
                "negative_mode": "encrypted_random_plaintexts",
                "model_options": TARGET_MODEL_OPTIONS,
            }
        )

    candidate_auc = next(
        row["auc"] for row in rows if row["role"] == "true_source_true_target"
    )
    for row in rows:
        row["candidate_minus_full_target_anchor_auc"] = candidate_auc - target_anchor
    return rows


def adjudicate_transfer_panel(
    *,
    run_id: str,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    grouped = {str(row.get("role")): row for row in rows}
    complete = len(rows) == 4 and set(grouped) == set(EXPECTED_ROLES)
    candidate = grouped.get("true_source_true_target", {})
    corrupted_source = grouped.get("corrupted_source_true_target", {})
    corrupted_target = grouped.get("true_source_corrupted_target", {})
    random_source = grouped.get("random_source_true_target", {})
    all_rows = tuple(grouped.values())

    protocol_checks = {
        "four_role_panel_complete": complete,
        "same_target_data_all_roles": complete
        and all(
            len({row.get(field) for row in all_rows}) == 1
            for field in (
                "train_feature_sha256",
                "train_label_sha256",
                "train_metadata_sha256",
                "validation_feature_sha256",
                "validation_label_sha256",
                "validation_metadata_sha256",
            )
        ),
        "same_target_head_initialization": complete
        and len({row.get("target_head_initial_sha256") for row in all_rows}) == 1,
        "source_checkpoint_attribution_exact": complete
        and candidate.get("source_checkpoint_sha256")
        == SOURCE_CHECKPOINT_SHA256S["true"]
        and corrupted_target.get("source_checkpoint_sha256")
        == SOURCE_CHECKPOINT_SHA256S["true"]
        and corrupted_source.get("source_checkpoint_sha256")
        == SOURCE_CHECKPOINT_SHA256S["corrupted"]
        and random_source.get("source_checkpoint_sha256") is None,
        "target_structure_attribution_exact": complete
        and len(
            {
                candidate.get("target_structure_sha256"),
                corrupted_source.get("target_structure_sha256"),
                random_source.get("target_structure_sha256"),
            }
        )
        == 1
        and corrupted_target.get("target_structure_sha256")
        != candidate.get("target_structure_sha256"),
        "formal_skinny_source_authority": complete
        and candidate.get("source_auc") == SOURCE_AUCS["true"]
        and corrupted_target.get("source_auc") == SOURCE_AUCS["true"]
        and corrupted_source.get("source_auc") == SOURCE_AUCS["corrupted"]
        and all(
            row.get("source_samples_per_class") == 1_000_000
            for row in (candidate, corrupted_source, corrupted_target)
        ),
        "rectangle_rct1_anchor_exact": complete
        and all(
            row.get("full_target_anchor_auc") == TARGET_ANCHOR_AUC for row in all_rows
        ),
        "feature_extractors_frozen": complete
        and all(
            row.get("feature_extractor_frozen") is True
            and row.get("feature_extractor_initial_sha256")
            == row.get("feature_extractor_final_sha256")
            for row in all_rows
        ),
        "source_classifiers_preserved": complete
        and all(
            row.get("source_classifier_preserved") is True
            and row.get("source_classifier_initial_sha256")
            == row.get("source_classifier_final_sha256")
            for row in all_rows
        ),
        "only_independent_target_head_trained": complete
        and all(
            row.get("parameter_count") == TOTAL_PARAMETER_COUNT
            and row.get("trainable_parameter_count") == TARGET_HEAD_PARAMETER_COUNT
            and row.get("target_head_initial_sha256")
            != row.get("target_head_final_sha256")
            and row.get("trainable_parameter_names")
            and all(
                str(name).startswith("target_head.")
                for name in row.get("trainable_parameter_names", ())
            )
            for row in all_rows
        ),
        "checkpoint_replay_complete": complete
        and all(
            row.get("checkpoint_replay_verified") is True
            and _is_sha256(row.get("checkpoint_sha256"))
            for row in all_rows
        ),
        "frozen_target_protocol": complete
        and all(
            row.get("seed") == 0
            and row.get("target_cipher") == "RECTANGLE-80"
            and row.get("target_rounds") == 6
            and row.get("train_rows") == 4096
            and row.get("validation_rows") == 2048
            and row.get("pairs_per_sample") == 4
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("training", {}).get("epochs") == EPOCHS
            and row.get("training", {}).get("selected_checkpoint") == "best"
            and row.get("adapter_mode") == "frozen_runtime_e4_target_head"
            for row in all_rows
        ),
        "finite_auc_metrics": complete
        and all(_finite(row.get("auc")) for row in all_rows),
    }

    candidate_auc = _float_or_none(candidate.get("auc"))
    source_auc = _float_or_none(corrupted_source.get("auc"))
    target_auc = _float_or_none(corrupted_target.get("auc"))
    random_auc = _float_or_none(random_source.get("auc"))
    margins = {
        "candidate_minus_corrupted_source": _difference(candidate_auc, source_auc),
        "candidate_minus_corrupted_target": _difference(candidate_auc, target_auc),
        "candidate_minus_random_source": _difference(candidate_auc, random_auc),
        "candidate_minus_full_target_anchor": _difference(
            candidate_auc, TARGET_ANCHOR_AUC
        ),
    }
    research_checks = {
        "candidate_auc_at_least_0p55": bool(
            candidate_auc is not None and candidate_auc >= AUC_FLOOR
        ),
        "candidate_beats_corrupted_source_by_0p005": _margin_passes(
            margins["candidate_minus_corrupted_source"]
        ),
        "candidate_beats_corrupted_target_by_0p005": _margin_passes(
            margins["candidate_minus_corrupted_target"]
        ),
        "candidate_beats_random_source_by_0p005": _margin_passes(
            margins["candidate_minus_random_source"]
        ),
    }

    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_skinny_rectangle_transfer_protocol_invalid"
        next_action = "repair X3-A evidence without changing data, roles, or thresholds"
    elif all(research_checks.values()):
        status = "pass"
        decision = (
            "innovation1_skinny_rectangle_frozen_representation_readiness_supported"
        )
        next_action = (
            "wait for RECTANGLE RCT2; if its same-protocol medium anchor passes, "
            "prepare one seed0 medium transfer confirmation without changing X3-A roles"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_skinny_rectangle_frozen_representation_readiness_not_supported"
        )
        next_action = (
            "stop SKINNY-to-RECTANGLE frozen transfer scaling; retain the reusable "
            "adapter but require end-to-end target training"
        )

    return {
        "run_id": run_id,
        "task": "innovation1_skinny_formal_to_rectangle_frozen_representation_x3a",
        "status": status,
        "decision": decision,
        "thresholds": {"candidate_auc": AUC_FLOOR, "auc_margin": MARGIN_FLOOR},
        "aucs": {
            "candidate": candidate_auc,
            "corrupted_source": source_auc,
            "corrupted_target": target_auc,
            "random_source": random_auc,
            "full_target_anchor": TARGET_ANCHOR_AUC,
        },
        "margins": margins,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "claim_scope": (
            "single-seed local 2048/class SKINNY-formal-to-RECTANGLE frozen-"
            "representation readiness only; not medium/formal transfer, universal-SPN, "
            "paper reproduction, attack, SOTA, or breakthrough evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "launch medium transfer before RECTANGLE RCT2 passes",
            "unfreeze the RuntimeE4 feature extractor inside X3-A",
            "change target data, roles, epochs, negatives, or thresholds after seeing results",
            "claim universal SPN transfer from one source-target pair and one seed",
        ],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_target_extractor(target_mode: str) -> torch.nn.Module:
    return build_model(
        TARGET_MODELS[target_mode],
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=TARGET_MODEL_OPTIONS,
    )


def _validate_parameter_ownership(model: FrozenRuntimeE4HeadAdapter) -> None:
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != TOTAL_PARAMETER_COUNT
    ):
        raise ValueError("X3-A total parameter count changed")
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if (
        sum(parameter.numel() for parameter in trainable.values())
        != TARGET_HEAD_PARAMETER_COUNT
    ):
        raise ValueError("X3-A target-head parameter count changed")
    if not trainable or any(not name.startswith("target_head.") for name in trainable):
        raise ValueError("X3-A may train only target_head parameters")


def _validate_source_rows(rows: list[dict[str, Any]]) -> None:
    for role, model in SOURCE_MODELS.items():
        matches = [
            row for row in rows if row.get("model") == model and row.get("seed") == 0
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one formal SKINNY {role} source row")
        row = matches[0]
        if (
            row.get("samples_per_class") != 1_000_000
            or row.get("training", {}).get("selected_checkpoint") != "best"
            or float(row.get("metrics", {}).get("auc", math.nan)) != SOURCE_AUCS[role]
        ):
            raise ValueError(f"formal SKINNY {role} source authority changed")


def _validate_target_rows(rows: list[dict[str, Any]]) -> float:
    matches = [
        row
        for row in rows
        if row.get("model") == TARGET_MODELS["true"] and row.get("seed") == 0
    ]
    if len(matches) != 1:
        raise ValueError("expected one RECTANGLE RCT1 seed0 target anchor")
    row = matches[0]
    auc = float(row.get("metrics", {}).get("auc", math.nan))
    if (
        row.get("samples_per_class") != 2048
        or row.get("rounds") != 6
        or auc != TARGET_ANCHOR_AUC
    ):
        raise ValueError("RECTANGLE RCT1 seed0 target anchor changed")
    return auc


def _validate_target_datasets(
    train_dataset: DifferentialDataset,
    validation_dataset: DifferentialDataset,
) -> None:
    expected_common = {
        "cipher": "RECTANGLE-80",
        "rounds": 6,
        "input_difference": TARGET_INPUT_DIFFERENCE,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "input_bits": 512,
        "structure": "SPN",
    }
    for dataset, seed, total, per_class in (
        (train_dataset, 0, 4096, 2048),
        (validation_dataset, 10000, 2048, 1024),
    ):
        metadata = dataset.metadata
        if any(
            metadata.get(field) != value for field, value in expected_common.items()
        ):
            raise ValueError("X3-A target dataset protocol changed")
        if (
            metadata.get("seed") != seed
            or metadata.get("samples_total") != total
            or metadata.get("samples_per_class") != per_class
            or metadata.get("positive_rows") != per_class
            or metadata.get("negative_rows") != per_class
            or dataset.features.shape != (total, 512)
            or dataset.labels.shape != (total,)
        ):
            raise ValueError("X3-A target dataset split geometry changed")


def _validate_dataset_paths(
    train_paths: Mapping[str, Path],
    validation_paths: Mapping[str, Path],
) -> None:
    for paths in (train_paths, validation_paths):
        if set(paths) != {"features", "labels", "metadata"}:
            raise ValueError(
                "X3-A dataset paths must identify features, labels, metadata"
            )
        if any(not path.is_file() for path in paths.values()):
            raise ValueError("X3-A dataset evidence path is missing")


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _margin_passes(value: float | None) -> bool:
    return bool(value is not None and value >= MARGIN_FLOOR)


def _float_or_none(value: Any) -> float | None:
    return float(value) if _finite(value) else None


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
    "RUN_ID",
    "adjudicate_transfer_panel",
    "deterministic_target_head",
    "prepare_transfer_model",
    "train_transfer_panel",
]
