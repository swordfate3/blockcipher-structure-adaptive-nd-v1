from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.data.differential import (
    DifferentialDataset,
    DiskDifferentialDataset,
)
from blockcipher_nd.evaluation.runtime_spn_representation import (
    extract_runtime_e4_representation,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_skinny_rectangle_transfer as transfer,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_head_adaptation import (
    tensor_mapping_sha256,
)
from blockcipher_nd.training import TrainingConfig, train_binary_classifier


RUN_ID = "i1_skinny_rectangle_runtime_e4_linear_probe_x4_2048_seed0_seed1_20260725"
TARGET_SEEDS = (0, 1)
EXPECTED_ROLES = transfer.EXPECTED_ROLES
ROLE_SPECS = transfer.ROLE_SPECS
REPRESENTATION_WIDTH = 384
PROBE_PARAMETER_COUNT = REPRESENTATION_WIDTH + 1
PROBE_INITIALIZATION_SEED = 25_070_401
EPOCHS = 100
BATCH_SIZE = 256
EXTRACTION_BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
TRAIN_EVAL_INTERVAL = 10
AUC_FLOOR = 0.55
MARGIN_FLOOR = 0.005
MAX_CANDIDATE_AUC_DRIFT = 0.05
CACHE_SCHEMA_VERSION = 1


def deterministic_linear_probe(seed: int) -> torch.nn.Linear:
    if seed not in TARGET_SEEDS:
        raise ValueError(f"unsupported X4 target seed: {seed}")
    with torch.random.fork_rng():
        torch.manual_seed(PROBE_INITIALIZATION_SEED + seed)
        probe = torch.nn.Linear(REPRESENTATION_WIDTH, 1)
    if (
        sum(parameter.numel() for parameter in probe.parameters())
        != PROBE_PARAMETER_COUNT
    ):
        raise ValueError("X4 linear-probe parameter count changed")
    return probe


def load_source_state_dicts(
    *,
    source_rows: list[dict[str, Any]],
    source_checkpoint_paths: Mapping[str, Path],
) -> dict[str, Mapping[str, torch.Tensor]]:
    transfer._validate_source_rows(source_rows)
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    for role in ("true", "corrupted"):
        try:
            path = source_checkpoint_paths[role]
        except KeyError as exc:
            raise ValueError(f"missing {role} source checkpoint path") from exc
        if not path.is_file():
            raise ValueError(f"missing {role} source checkpoint")
        if transfer.file_sha256(path) != transfer.SOURCE_CHECKPOINT_SHA256S[role]:
            raise ValueError(f"{role} source checkpoint SHA-256 changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("state_dict"), dict
        ):
            raise ValueError("source checkpoint must contain a state_dict")
        if payload.get("metadata", {}).get("selected_checkpoint") != "best":
            raise ValueError("source checkpoint must be the selected best checkpoint")
        states[role] = payload["state_dict"]
    return states


def extract_representation_cache(
    *,
    extractor: torch.nn.Module,
    raw_dataset: DifferentialDataset,
    raw_paths: Mapping[str, Path],
    cache_dir: Path,
    seed: int,
    split: str,
    role: str,
    source_role: str,
    target_mode: str,
    source_checkpoint_sha256: str | None,
    device: str = "cpu",
    batch_size: int = EXTRACTION_BATCH_SIZE,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[DiskDifferentialDataset, dict[str, Any], bool]:
    if seed not in TARGET_SEEDS:
        raise ValueError(f"unsupported X4 target seed: {seed}")
    if split not in {"train", "validation"}:
        raise ValueError(f"unsupported X4 split: {split}")
    if role not in EXPECTED_ROLES or ROLE_SPECS[role] != (source_role, target_mode):
        raise ValueError("X4 role attribution changed")
    if set(raw_paths) != {"features", "labels", "metadata"}:
        raise ValueError("X4 raw paths must identify features, labels, metadata")
    if any(not path.is_file() for path in raw_paths.values()):
        raise ValueError("X4 raw dataset evidence path is missing")
    if batch_size <= 0:
        raise ValueError("X4 extraction batch size must be positive")

    rows = int(raw_dataset.labels.shape[0])
    expected_rows = 4096 if split == "train" else 2048
    expected_raw_width = 512
    if raw_dataset.features.shape != (expected_rows, expected_raw_width):
        raise ValueError("X4 raw feature geometry changed")
    if raw_dataset.labels.shape != (expected_rows,):
        raise ValueError("X4 raw label geometry changed")
    if rows != expected_rows:
        raise ValueError("X4 raw row count changed")

    extractor.eval()
    extractor_sha256 = tensor_mapping_sha256(extractor.state_dict())
    source_classifier_sha256 = tensor_mapping_sha256(
        extractor.backbone.classifier.state_dict()
    )
    target_structure_sha256 = extractor.runtime_structure.window_sha256()
    expected_metadata = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "seed": seed,
        "split": split,
        "role": role,
        "source_role": source_role,
        "target_mode": target_mode,
        "rows": rows,
        "representation_width": REPRESENTATION_WIDTH,
        "representation_dtype": "float32",
        "label_dtype": "uint8",
        "extraction_batch_size": batch_size,
        "extractor_sha256": extractor_sha256,
        "source_classifier_sha256": source_classifier_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "target_structure_sha256": target_structure_sha256,
        "raw_feature_sha256": transfer.file_sha256(raw_paths["features"]),
        "raw_label_sha256": transfer.file_sha256(raw_paths["labels"]),
        "raw_metadata_sha256": transfer.file_sha256(raw_paths["metadata"]),
    }

    if cache_dir.exists():
        dataset, metadata = _load_representation_cache(
            cache_dir,
            expected_metadata=expected_metadata,
        )
        return dataset, metadata, True

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{cache_dir.name}.partial-", dir=cache_dir.parent)
    )
    try:
        representation_path = partial / "representations.npy"
        label_path = partial / "labels.npy"
        metadata_path = partial / "metadata.json"
        representations = np.lib.format.open_memmap(
            representation_path,
            mode="w+",
            dtype=np.float32,
            shape=(rows, REPRESENTATION_WIDTH),
        )
        selected_device = torch.device(device)
        extractor.to(selected_device)
        extractor.eval()
        with torch.no_grad():
            for start in range(0, rows, batch_size):
                stop = min(rows, start + batch_size)
                features = torch.as_tensor(
                    np.asarray(raw_dataset.features[start:stop]).copy(),
                    dtype=torch.float32,
                    device=selected_device,
                )
                batch = extract_runtime_e4_representation(extractor, features)
                representation = batch.representation.detach().cpu().numpy()
                if representation.shape != (stop - start, REPRESENTATION_WIDTH):
                    raise ValueError("X4 RuntimeE4 representation geometry changed")
                representations[start:stop] = representation.astype(
                    np.float32,
                    copy=False,
                )
                if progress_callback is not None:
                    progress_callback(
                        "representation_batch",
                        {
                            "seed": seed,
                            "split": split,
                            "role": role,
                            "rows_done": stop,
                            "rows_total": rows,
                        },
                    )
        representations.flush()
        del representations
        np.save(label_path, np.asarray(raw_dataset.labels, dtype=np.uint8))
        extractor.to("cpu")
        extractor.eval()
        if tensor_mapping_sha256(extractor.state_dict()) != extractor_sha256:
            raise ValueError("X4 extractor changed during representation extraction")
        if (
            tensor_mapping_sha256(extractor.backbone.classifier.state_dict())
            != source_classifier_sha256
        ):
            raise ValueError(
                "X4 source classifier changed during representation extraction"
            )
        metadata = {
            **expected_metadata,
            "representation_sha256": transfer.file_sha256(representation_path),
            "cached_label_sha256": transfer.file_sha256(label_path),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, cache_dir)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    dataset, metadata = _load_representation_cache(
        cache_dir,
        expected_metadata=expected_metadata,
    )
    return dataset, metadata, False


def train_linear_probe_panel(
    *,
    source_rows: list[dict[str, Any]],
    source_checkpoint_paths: Mapping[str, Path],
    target_rows: list[dict[str, Any]],
    target_datasets: Mapping[int, Mapping[str, DifferentialDataset]],
    target_paths: Mapping[int, Mapping[str, Mapping[str, Path]]],
    representation_cache_root: Path,
    checkpoint_dir: Path,
    device: str = "cpu",
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    source_state_dicts = load_source_state_dicts(
        source_rows=source_rows,
        source_checkpoint_paths=source_checkpoint_paths,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for seed in TARGET_SEEDS:
        train_dataset, validation_dataset = _seed_datasets(target_datasets, seed)
        train_paths, validation_paths = _seed_paths(target_paths, seed)
        target_anchor_auc = transfer._validate_target_rows(
            target_rows, target_seed=seed
        )
        transfer._validate_target_datasets(
            train_dataset,
            validation_dataset,
            target_seed=seed,
            validation_seed=seed + 10_000,
        )
        transfer._validate_dataset_paths(train_paths, validation_paths)
        expected_probe_initial_sha256 = tensor_mapping_sha256(
            deterministic_linear_probe(seed).state_dict()
        )

        for row_index, role in enumerate(EXPECTED_ROLES, start=1):
            source_role, target_mode = ROLE_SPECS[role]
            adapter = transfer.prepare_transfer_model(
                source_role=source_role,
                target_mode=target_mode,
                source_state_dicts=source_state_dicts,
            )
            extractor = adapter.feature_extractor
            extractor_initial = tensor_mapping_sha256(extractor.state_dict())
            source_classifier_initial = tensor_mapping_sha256(
                extractor.backbone.classifier.state_dict()
            )
            source_checkpoint_sha256 = (
                transfer.SOURCE_CHECKPOINT_SHA256S[source_role]
                if source_role != "random"
                else None
            )

            def emit(event: str, payload: dict[str, Any]) -> None:
                if progress_callback is not None:
                    progress_callback(
                        event,
                        {
                            "seed": seed,
                            "role": role,
                            "row_index": row_index,
                            "row_total": len(EXPECTED_ROLES),
                            **payload,
                        },
                    )

            representation_datasets: dict[str, DiskDifferentialDataset] = {}
            representation_metadata: dict[str, dict[str, Any]] = {}
            reuse_verified: dict[str, bool] = {}
            for split, raw_dataset, raw_split_paths in (
                ("train", train_dataset, train_paths),
                ("validation", validation_dataset, validation_paths),
            ):
                cache_dir = representation_cache_root / f"seed{seed}" / role / split
                dataset, metadata, _ = extract_representation_cache(
                    extractor=extractor,
                    raw_dataset=raw_dataset,
                    raw_paths=raw_split_paths,
                    cache_dir=cache_dir,
                    seed=seed,
                    split=split,
                    role=role,
                    source_role=source_role,
                    target_mode=target_mode,
                    source_checkpoint_sha256=source_checkpoint_sha256,
                    device=device,
                    progress_callback=emit,
                )
                replay_dataset, replay_metadata, reused = extract_representation_cache(
                    extractor=extractor,
                    raw_dataset=raw_dataset,
                    raw_paths=raw_split_paths,
                    cache_dir=cache_dir,
                    seed=seed,
                    split=split,
                    role=role,
                    source_role=source_role,
                    target_mode=target_mode,
                    source_checkpoint_sha256=source_checkpoint_sha256,
                    device=device,
                )
                if (
                    replay_metadata != metadata
                    or replay_dataset.cache_dir != dataset.cache_dir
                ):
                    raise ValueError("X4 representation cache replay changed identity")
                representation_datasets[split] = dataset
                representation_metadata[split] = metadata
                reuse_verified[split] = reused

            extractor_final = tensor_mapping_sha256(extractor.state_dict())
            source_classifier_final = tensor_mapping_sha256(
                extractor.backbone.classifier.state_dict()
            )
            if extractor_final != extractor_initial:
                raise ValueError("X4 extractor changed before probe training")
            if source_classifier_final != source_classifier_initial:
                raise ValueError("X4 source classifier changed before probe training")

            probe = deterministic_linear_probe(seed)
            probe_initial_sha256 = tensor_mapping_sha256(probe.state_dict())
            if probe_initial_sha256 != expected_probe_initial_sha256:
                raise ValueError("X4 probe initialization changed within a seed")
            checkpoint_path = checkpoint_dir / f"seed{seed}_{role}.pt"
            result = train_binary_classifier(
                probe,
                representation_datasets["train"],
                representation_datasets["validation"],
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
                    train_eval_interval=TRAIN_EVAL_INTERVAL,
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
                == tensor_mapping_sha256(probe.state_dict())
                and checkpoint_payload.get("final_metrics") == result.final_metrics
                and checkpoint_payload.get("metadata", {}).get("selected_checkpoint")
                == "best"
            )
            if not checkpoint_replay_verified:
                raise ValueError("X4 best checkpoint replay does not match probe")
            probe_final_sha256 = tensor_mapping_sha256(probe.state_dict())
            extractor_post_training = tensor_mapping_sha256(extractor.state_dict())
            classifier_post_training = tensor_mapping_sha256(
                extractor.backbone.classifier.state_dict()
            )
            if extractor_post_training != extractor_initial:
                raise ValueError("X4 extractor changed during probe training")
            if classifier_post_training != source_classifier_initial:
                raise ValueError("X4 source classifier changed during probe training")

            rows.append(
                {
                    "seed": seed,
                    "role": role,
                    "source_role": source_role,
                    "target_mode": target_mode,
                    "source_checkpoint_path": (
                        str(source_checkpoint_paths[source_role])
                        if source_role != "random"
                        else None
                    ),
                    "source_checkpoint_sha256": source_checkpoint_sha256,
                    "source_selected_checkpoint": (
                        "best" if source_role != "random" else None
                    ),
                    "source_auc": (
                        transfer.SOURCE_AUCS[source_role]
                        if source_role != "random"
                        else None
                    ),
                    "strict_source_state_dict_load": True,
                    "target_structure_sha256": extractor.runtime_structure.window_sha256(),
                    "target_relation_mode": extractor.relation_mode,
                    "feature_extractor_initial_sha256": extractor_initial,
                    "feature_extractor_final_sha256": extractor_post_training,
                    "source_classifier_initial_sha256": source_classifier_initial,
                    "source_classifier_final_sha256": classifier_post_training,
                    "probe_initial_sha256": probe_initial_sha256,
                    "probe_final_sha256": probe_final_sha256,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": transfer.file_sha256(checkpoint_path),
                    "checkpoint_replay_verified": checkpoint_replay_verified,
                    "parameter_count": PROBE_PARAMETER_COUNT,
                    "trainable_parameter_count": PROBE_PARAMETER_COUNT,
                    "trainable_parameter_names": [
                        name
                        for name, parameter in probe.named_parameters()
                        if parameter.requires_grad
                    ],
                    "adapter_mode": "frozen_runtime_e4_linear_probe",
                    "feature_extractor_frozen": True,
                    "source_classifier_preserved": True,
                    "representation_width": REPRESENTATION_WIDTH,
                    "train_representation_cache": str(
                        representation_datasets["train"].cache_dir
                    ),
                    "validation_representation_cache": str(
                        representation_datasets["validation"].cache_dir
                    ),
                    "train_representation_sha256": representation_metadata["train"][
                        "representation_sha256"
                    ],
                    "validation_representation_sha256": representation_metadata[
                        "validation"
                    ]["representation_sha256"],
                    "train_representation_metadata_sha256": transfer.file_sha256(
                        representation_datasets["train"].cache_dir / "metadata.json"
                    ),
                    "validation_representation_metadata_sha256": transfer.file_sha256(
                        representation_datasets["validation"].cache_dir
                        / "metadata.json"
                    ),
                    "train_representation_cache_reuse_verified": reuse_verified[
                        "train"
                    ],
                    "validation_representation_cache_reuse_verified": reuse_verified[
                        "validation"
                    ],
                    "raw_train_feature_sha256": representation_metadata["train"][
                        "raw_feature_sha256"
                    ],
                    "raw_train_label_sha256": representation_metadata["train"][
                        "raw_label_sha256"
                    ],
                    "raw_train_metadata_sha256": representation_metadata["train"][
                        "raw_metadata_sha256"
                    ],
                    "raw_validation_feature_sha256": representation_metadata[
                        "validation"
                    ]["raw_feature_sha256"],
                    "raw_validation_label_sha256": representation_metadata[
                        "validation"
                    ]["raw_label_sha256"],
                    "raw_validation_metadata_sha256": representation_metadata[
                        "validation"
                    ]["raw_metadata_sha256"],
                    "auc": float(result.final_metrics["auc"]),
                    "accuracy": float(result.final_metrics["accuracy"]),
                    "loss": float(result.final_metrics["loss"]),
                    "history": result.history,
                    "training": result.metadata,
                    "full_target_anchor_auc": target_anchor_auc,
                    "source_cipher": "SKINNY-64/64",
                    "source_seed": 0,
                    "source_rounds": 7,
                    "source_samples_per_class": 1_000_000,
                    "target_cipher": "RECTANGLE-80",
                    "target_rounds": 6,
                    "target_difference": transfer.TARGET_INPUT_DIFFERENCE,
                    "target_train_seed": seed,
                    "target_validation_seed": seed + 10_000,
                    "train_rows": 4096,
                    "validation_rows": 2048,
                    "pairs_per_sample": 4,
                    "input_bits": 512,
                    "negative_mode": "encrypted_random_plaintexts",
                    "model_options": transfer.TARGET_MODEL_OPTIONS,
                }
            )
    return rows


def adjudicate_linear_probe_panel(
    *,
    run_id: str,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    grouped = {(int(row.get("seed", -1)), str(row.get("role"))): row for row in rows}
    expected_keys = {(seed, role) for seed in TARGET_SEEDS for role in EXPECTED_ROLES}
    complete = len(rows) == len(expected_keys) and set(grouped) == expected_keys

    protocol_checks = {
        "eight_role_panel_complete": complete,
        "same_target_data_within_each_seed": complete
        and all(
            all(
                len({grouped[(seed, role)].get(field) for role in EXPECTED_ROLES}) == 1
                for field in (
                    "raw_train_feature_sha256",
                    "raw_train_label_sha256",
                    "raw_train_metadata_sha256",
                    "raw_validation_feature_sha256",
                    "raw_validation_label_sha256",
                    "raw_validation_metadata_sha256",
                )
            )
            for seed in TARGET_SEEDS
        ),
        "different_target_data_across_seeds": complete
        and grouped[(0, EXPECTED_ROLES[0])].get("raw_train_feature_sha256")
        != grouped[(1, EXPECTED_ROLES[0])].get("raw_train_feature_sha256")
        and grouped[(0, EXPECTED_ROLES[0])].get("raw_validation_feature_sha256")
        != grouped[(1, EXPECTED_ROLES[0])].get("raw_validation_feature_sha256"),
        "same_probe_initialization_within_each_seed": complete
        and all(
            len(
                {
                    grouped[(seed, role)].get("probe_initial_sha256")
                    for role in EXPECTED_ROLES
                }
            )
            == 1
            for seed in TARGET_SEEDS
        ),
        "probe_initialization_differs_across_seeds": complete
        and grouped[(0, EXPECTED_ROLES[0])].get("probe_initial_sha256")
        != grouped[(1, EXPECTED_ROLES[0])].get("probe_initial_sha256"),
        "source_checkpoint_attribution_exact": complete
        and all(
            grouped[(seed, "true_source_true_target")].get("source_checkpoint_sha256")
            == transfer.SOURCE_CHECKPOINT_SHA256S["true"]
            and grouped[(seed, "true_source_corrupted_target")].get(
                "source_checkpoint_sha256"
            )
            == transfer.SOURCE_CHECKPOINT_SHA256S["true"]
            and grouped[(seed, "corrupted_source_true_target")].get(
                "source_checkpoint_sha256"
            )
            == transfer.SOURCE_CHECKPOINT_SHA256S["corrupted"]
            and grouped[(seed, "random_source_true_target")].get(
                "source_checkpoint_sha256"
            )
            is None
            for seed in TARGET_SEEDS
        ),
        "target_structure_attribution_exact": complete
        and all(
            len(
                {
                    grouped[(seed, role)].get("target_structure_sha256")
                    for role in (
                        "true_source_true_target",
                        "corrupted_source_true_target",
                        "random_source_true_target",
                    )
                }
            )
            == 1
            and grouped[(seed, "true_source_corrupted_target")].get(
                "target_structure_sha256"
            )
            != grouped[(seed, "true_source_true_target")].get("target_structure_sha256")
            for seed in TARGET_SEEDS
        ),
        "feature_extractors_and_source_classifiers_frozen": complete
        and all(
            row.get("feature_extractor_frozen") is True
            and row.get("source_classifier_preserved") is True
            and row.get("feature_extractor_initial_sha256")
            == row.get("feature_extractor_final_sha256")
            and row.get("source_classifier_initial_sha256")
            == row.get("source_classifier_final_sha256")
            for row in rows
        ),
        "only_385_parameter_linear_probe_trained": complete
        and all(
            row.get("parameter_count") == PROBE_PARAMETER_COUNT
            and row.get("trainable_parameter_count") == PROBE_PARAMETER_COUNT
            and set(row.get("trainable_parameter_names", ())) == {"weight", "bias"}
            and row.get("probe_initial_sha256") != row.get("probe_final_sha256")
            and row.get("adapter_mode") == "frozen_runtime_e4_linear_probe"
            and row.get("representation_width") == REPRESENTATION_WIDTH
            for row in rows
        ),
        "representation_cache_identity_and_reuse_complete": complete
        and all(
            row.get("train_representation_cache_reuse_verified") is True
            and row.get("validation_representation_cache_reuse_verified") is True
            and _is_sha256(row.get("train_representation_sha256"))
            and _is_sha256(row.get("validation_representation_sha256"))
            and _is_sha256(row.get("train_representation_metadata_sha256"))
            and _is_sha256(row.get("validation_representation_metadata_sha256"))
            for row in rows
        ),
        "checkpoint_replay_complete": complete
        and all(
            row.get("checkpoint_replay_verified") is True
            and _is_sha256(row.get("checkpoint_sha256"))
            for row in rows
        ),
        "frozen_probe_protocol": complete
        and all(
            row.get("seed") in TARGET_SEEDS
            and row.get("source_seed") == 0
            and row.get("target_train_seed") == row.get("seed")
            and row.get("target_validation_seed") == row.get("seed") + 10_000
            and row.get("target_cipher") == "RECTANGLE-80"
            and row.get("target_rounds") == 6
            and row.get("train_rows") == 4096
            and row.get("validation_rows") == 2048
            and row.get("pairs_per_sample") == 4
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and row.get("training", {}).get("epochs") == EPOCHS
            and row.get("training", {}).get("batch_size") == BATCH_SIZE
            and row.get("training", {}).get("learning_rate") == LEARNING_RATE
            and row.get("training", {}).get("weight_decay") == WEIGHT_DECAY
            and row.get("training", {}).get("selected_checkpoint") == "best"
            for row in rows
        ),
        "finite_auc_metrics": complete and all(_finite(row.get("auc")) for row in rows),
    }

    aucs: dict[str, dict[str, float | None]] = {}
    margins: dict[str, dict[str, float | None]] = {}
    research_checks: dict[str, dict[str, bool]] = {}
    candidate_aucs: dict[int, float | None] = {}
    for seed in TARGET_SEEDS:
        values = {
            name: _float_or_none(grouped.get((seed, role), {}).get("auc"))
            for name, role in (
                ("candidate", "true_source_true_target"),
                ("corrupted_source", "corrupted_source_true_target"),
                ("corrupted_target", "true_source_corrupted_target"),
                ("random_source", "random_source_true_target"),
            )
        }
        candidate_aucs[seed] = values["candidate"]
        seed_margins = {
            "candidate_minus_corrupted_source": _difference(
                values["candidate"], values["corrupted_source"]
            ),
            "candidate_minus_corrupted_target": _difference(
                values["candidate"], values["corrupted_target"]
            ),
            "candidate_minus_random_source": _difference(
                values["candidate"], values["random_source"]
            ),
        }
        aucs[f"seed{seed}"] = values
        margins[f"seed{seed}"] = seed_margins
        research_checks[f"seed{seed}"] = {
            "candidate_auc_at_least_0p55": bool(
                values["candidate"] is not None and values["candidate"] >= AUC_FLOOR
            ),
            "candidate_beats_corrupted_source_by_0p005": _margin_passes(
                seed_margins["candidate_minus_corrupted_source"]
            ),
            "candidate_beats_corrupted_target_by_0p005": _margin_passes(
                seed_margins["candidate_minus_corrupted_target"]
            ),
            "candidate_beats_random_source_by_0p005": _margin_passes(
                seed_margins["candidate_minus_random_source"]
            ),
        }

    candidate_drift = _absolute_difference(candidate_aucs[0], candidate_aucs[1])
    stability_checks = {
        "candidate_auc_drift_at_most_0p05": bool(
            candidate_drift is not None and candidate_drift <= MAX_CANDIDATE_AUC_DRIFT
        )
    }
    research_pass = all(
        all(checks.values()) for checks in research_checks.values()
    ) and all(stability_checks.values())

    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_skinny_rectangle_linear_probe_protocol_invalid"
        next_action = (
            "repair X4 cache, frozen-state, checkpoint, or protocol evidence without "
            "changing data, roles, epochs, or thresholds"
        )
    elif research_pass:
        status = "pass"
        decision = "innovation1_skinny_rectangle_linear_probe_accessibility_supported"
        next_action = (
            "retain direct linear accessibility as mechanism evidence; continue waiting "
            "for RECTANGLE RCT2 before any medium frozen-transfer confirmation"
        )
    else:
        status = "hold"
        decision = (
            "innovation1_skinny_rectangle_linear_probe_accessibility_not_supported"
        )
        next_action = (
            "retain X3 nonlinear-head transfer evidence but stop scaling the linear-probe "
            "route; describe transfer as requiring nonlinear target adaptation"
        )

    return {
        "run_id": run_id,
        "task": "innovation1_skinny_rectangle_runtime_e4_linear_probe_x4",
        "status": status,
        "decision": decision,
        "thresholds": {
            "candidate_auc": AUC_FLOOR,
            "auc_margin": MARGIN_FLOOR,
            "max_candidate_auc_drift": MAX_CANDIDATE_AUC_DRIFT,
        },
        "aucs": aucs,
        "margins": margins,
        "candidate_auc_drift": candidate_drift,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "stability_checks": stability_checks,
        "claim_scope": (
            "local 2048/class dual-seed SKINNY-formal-to-RECTANGLE frozen RuntimeE4 "
            "linear-readout attribution only; not medium/formal transfer, universal-SPN, "
            "paper reproduction, attack, SOTA, or breakthrough evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "launch medium transfer before RECTANGLE RCT2 passes",
            "unfreeze or fine-tune the RuntimeE4 extractor inside X4",
            "change target data, roles, epochs, negatives, or thresholds after results",
            "treat 100 linear-probe epochs as same-compute superiority over X3",
            "claim universal SPN transfer from one source-target pair",
        ],
    }


def verify_linear_probe_artifacts(
    *,
    output_root: Path,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    errors: list[str] = []
    checkpoint_entries: list[dict[str, Any]] = []
    cache_entries: list[dict[str, Any]] = []
    expected_checkpoint_names = {
        f"seed{seed}_{role}.pt" for seed in TARGET_SEEDS for role in EXPECTED_ROLES
    }
    checkpoint_dir = output_root / "checkpoints"
    actual_checkpoint_names = (
        {path.name for path in checkpoint_dir.iterdir() if path.is_file()}
        if checkpoint_dir.is_dir()
        else set()
    )
    if actual_checkpoint_names != expected_checkpoint_names:
        errors.append("checkpoint file set does not match the eight-role panel")

    expected_cache_dirs: set[Path] = set()
    for row in rows:
        seed = row.get("seed")
        role = row.get("role")
        checkpoint_path = Path(str(row.get("checkpoint_path", "")))
        checkpoint_checks: dict[str, bool] = {
            "path_owned_by_output": _path_is_within(checkpoint_path, output_root),
            "file_exists": checkpoint_path.is_file(),
        }
        payload: Any = None
        if checkpoint_checks["file_exists"]:
            checkpoint_checks["sha256"] = transfer.file_sha256(
                checkpoint_path
            ) == row.get("checkpoint_sha256")
            try:
                payload = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception:
                payload = None
        else:
            checkpoint_checks["sha256"] = False
        state = payload.get("state_dict") if isinstance(payload, dict) else None
        checkpoint_checks.update(
            {
                "payload_shape": isinstance(payload, dict) and isinstance(state, dict),
                "linear_state_geometry": isinstance(state, dict)
                and set(state) == {"weight", "bias"}
                and tuple(state["weight"].shape) == (1, REPRESENTATION_WIDTH)
                and tuple(state["bias"].shape) == (1,),
                "state_sha256": isinstance(state, dict)
                and tensor_mapping_sha256(state) == row.get("probe_final_sha256"),
                "final_metrics": isinstance(payload, dict)
                and all(
                    payload.get("final_metrics", {}).get(field) == row.get(field)
                    for field in ("accuracy", "auc", "loss")
                )
                and all(
                    _finite(payload.get("final_metrics", {}).get(field))
                    for field in (
                        "advantage",
                        "best_accuracy",
                        "calibrated_accuracy",
                        "calibrated_advantage",
                        "calibrated_threshold",
                    )
                ),
                "history": isinstance(payload, dict)
                and payload.get("history") == row.get("history"),
                "metadata": isinstance(payload, dict)
                and payload.get("metadata") == row.get("training"),
                "selected_best": isinstance(payload, dict)
                and payload.get("metadata", {}).get("selected_checkpoint") == "best",
            }
        )
        if not all(checkpoint_checks.values()):
            errors.append(f"checkpoint verification failed for seed{seed} {role}")
        checkpoint_entries.append(
            {
                "seed": seed,
                "role": role,
                "path": str(checkpoint_path),
                "checks": checkpoint_checks,
            }
        )

        for split, rows_field in (
            ("train", "train_rows"),
            ("validation", "validation_rows"),
        ):
            cache_dir = Path(str(row.get(f"{split}_representation_cache", "")))
            expected_cache_dirs.add(cache_dir)
            representation_path = cache_dir / "representations.npy"
            label_path = cache_dir / "labels.npy"
            metadata_path = cache_dir / "metadata.json"
            cache_checks: dict[str, bool] = {
                "path_owned_by_output": _path_is_within(cache_dir, output_root),
                "file_set_exact": cache_dir.is_dir()
                and {path.name for path in cache_dir.iterdir()}
                == {"representations.npy", "labels.npy", "metadata.json"},
                "metadata_sha256": metadata_path.is_file()
                and transfer.file_sha256(metadata_path)
                == row.get(f"{split}_representation_metadata_sha256"),
            }
            metadata: dict[str, Any] = {}
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    metadata = {}
            cache_checks.update(
                {
                    "metadata_identity": metadata.get("seed") == seed
                    and metadata.get("role") == role
                    and metadata.get("split") == split
                    and metadata.get("source_role") == row.get("source_role")
                    and metadata.get("target_mode") == row.get("target_mode")
                    and metadata.get("extractor_sha256")
                    == row.get("feature_extractor_initial_sha256")
                    and metadata.get("source_classifier_sha256")
                    == row.get("source_classifier_initial_sha256")
                    and metadata.get("target_structure_sha256")
                    == row.get("target_structure_sha256"),
                    "representation_sha256": representation_path.is_file()
                    and transfer.file_sha256(representation_path)
                    == row.get(f"{split}_representation_sha256")
                    == metadata.get("representation_sha256"),
                    "label_sha256": label_path.is_file()
                    and transfer.file_sha256(label_path)
                    == metadata.get("cached_label_sha256"),
                }
            )
            try:
                representations = np.load(representation_path, mmap_mode="r")
                labels = np.load(label_path, mmap_mode="r")
            except (OSError, ValueError):
                representations = None
                labels = None
            expected_rows = int(row.get(rows_field, -1))
            cache_checks.update(
                {
                    "representation_geometry": representations is not None
                    and representations.shape == (expected_rows, REPRESENTATION_WIDTH)
                    and representations.dtype == np.float32,
                    "label_geometry": labels is not None
                    and labels.shape == (expected_rows,)
                    and labels.dtype == np.uint8,
                }
            )
            if not all(cache_checks.values()):
                errors.append(
                    f"representation cache verification failed for seed{seed} "
                    f"{role} {split}"
                )
            cache_entries.append(
                {
                    "seed": seed,
                    "role": role,
                    "split": split,
                    "path": str(cache_dir),
                    "checks": cache_checks,
                }
            )

    actual_cache_dirs = {
        path.parent
        for path in (output_root / "representation_cache").glob(
            "seed*/*/*/metadata.json"
        )
    }
    cache_dir_set_exact = actual_cache_dirs == expected_cache_dirs
    if not cache_dir_set_exact:
        errors.append("representation cache directory set does not match the panel")
    result_rows_exact = len(rows) == len(TARGET_SEEDS) * len(EXPECTED_ROLES)
    if not result_rows_exact:
        errors.append("result row count does not match the eight-role panel")
    return {
        "run_id": RUN_ID,
        "status": "pass" if not errors else "fail",
        "result_rows_exact": result_rows_exact,
        "checkpoint_file_set_exact": actual_checkpoint_names
        == expected_checkpoint_names,
        "cache_directory_set_exact": cache_dir_set_exact,
        "checkpoint_entries": checkpoint_entries,
        "cache_entries": cache_entries,
        "errors": errors,
    }


def _load_representation_cache(
    cache_dir: Path,
    *,
    expected_metadata: Mapping[str, Any],
) -> tuple[DiskDifferentialDataset, dict[str, Any]]:
    expected_files = {"representations.npy", "labels.npy", "metadata.json"}
    if (
        not cache_dir.is_dir()
        or {path.name for path in cache_dir.iterdir()} != expected_files
    ):
        raise ValueError(
            "X4 representation cache is incomplete or has unexpected files"
        )
    representation_path = cache_dir / "representations.npy"
    label_path = cache_dir / "labels.npy"
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mismatches = [
        key
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "X4 representation cache parameters changed: " + ", ".join(mismatches)
        )
    if metadata.get("representation_sha256") != transfer.file_sha256(
        representation_path
    ):
        raise ValueError("X4 representation cache SHA-256 changed")
    if metadata.get("cached_label_sha256") != transfer.file_sha256(label_path):
        raise ValueError("X4 cached labels SHA-256 changed")
    representations = np.load(representation_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    expected_rows = int(expected_metadata["rows"])
    if representations.shape != (expected_rows, REPRESENTATION_WIDTH):
        raise ValueError("X4 cached representation geometry changed")
    if representations.dtype != np.float32:
        raise ValueError("X4 cached representation dtype changed")
    if labels.shape != (expected_rows,) or labels.dtype != np.uint8:
        raise ValueError("X4 cached label geometry or dtype changed")
    return (
        DiskDifferentialDataset(
            features=representations,
            labels=labels,
            metadata=dict(metadata),
            cache_dir=cache_dir,
        ),
        metadata,
    )


def _seed_datasets(
    datasets: Mapping[int, Mapping[str, DifferentialDataset]],
    seed: int,
) -> tuple[DifferentialDataset, DifferentialDataset]:
    try:
        splits = datasets[seed]
        return splits["train"], splits["validation"]
    except KeyError as exc:
        raise ValueError(f"missing X4 target dataset for seed {seed}") from exc


def _seed_paths(
    paths: Mapping[int, Mapping[str, Mapping[str, Path]]],
    seed: int,
) -> tuple[Mapping[str, Path], Mapping[str, Path]]:
    try:
        splits = paths[seed]
        return splits["train"], splits["validation"]
    except KeyError as exc:
        raise ValueError(f"missing X4 target paths for seed {seed}") from exc


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _absolute_difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else abs(left - right)


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


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


__all__ = [
    "EXPECTED_ROLES",
    "RUN_ID",
    "TARGET_SEEDS",
    "adjudicate_linear_probe_panel",
    "deterministic_linear_probe",
    "extract_representation_cache",
    "load_source_state_dicts",
    "train_linear_probe_panel",
    "verify_linear_probe_artifacts",
]
