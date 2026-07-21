from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    MODEL_NAMES,
    KimuraCiphertextLstm,
    ParameterMatchedOutputMlp,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i2_output_prediction_op10_present_r3_easy_bit_smoke_20260721"
PAPER_RUN_ID = "i2_output_prediction_op10_present_r3_easy_bit_confirm_20260721"
FRESH_CACHE_VERSION = 1
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class OutputBitDiscoveryConfig:
    run_id: str = RUN_ID
    mode: str = "smoke"
    seed: int = 0
    fresh_rows: int = 64
    fresh_chunk_rows: int = 32
    batch_size: int = 32
    candidate_limit: int = 8
    minimum_auc: float = 0.510
    minimum_accuracy_margin: float = 0.005
    minimum_shuffle_auc_margin: float = 0.005
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "fresh_confirmation"}:
            raise ValueError("mode must be smoke or fresh_confirmation")
        if min(
            self.fresh_rows,
            self.fresh_chunk_rows,
            self.batch_size,
            self.candidate_limit,
        ) <= 0:
            raise ValueError("row, chunk, batch, and candidate values must be positive")

    @classmethod
    def fresh_confirmation(
        cls,
        *,
        run_id: str = PAPER_RUN_ID,
        seed: int = 0,
        device: str = "cuda",
    ) -> OutputBitDiscoveryConfig:
        return cls(
            run_id=run_id,
            mode="fresh_confirmation",
            seed=seed,
            fresh_rows=1 << 16,
            fresh_chunk_rows=4096,
            batch_size=250,
            candidate_limit=8,
            device=device,
        )


def load_output_prediction_source(source_root: Path) -> dict[str, Any]:
    metadata = _read_json(source_root / "metadata.json")
    manifest = _read_json(source_root / "checkpoint_manifest.json")
    data_metadata = _read_json(source_root / "data" / "cache_metadata.json")
    source_config = metadata["config"]
    train_rows = int(source_config["train_rows"])
    test_rows = int(source_config["test_rows"])
    total_rows = train_rows + test_rows
    plaintexts = np.load(source_root / "data" / "plaintexts.npy", mmap_mode="r")
    features = np.load(source_root / "data" / "features.npy", mmap_mode="r")
    targets = np.load(source_root / "data" / "full_targets.npy", mmap_mode="r")
    manifest_by_model = {item["model"]: item for item in manifest}
    checkpoint_hashes_match = set(manifest_by_model) == set(MODEL_NAMES) and all(
        _sha256(source_root / manifest_by_model[name]["path"])
        == manifest_by_model[name]["sha256"]
        for name in MODEL_NAMES
    )
    checks = {
        "source_is_true_output_prediction": metadata.get("sample_classification")
        is False
        and metadata.get("target") == "64 MSB-first true ciphertext bits",
        "source_is_present_r3": metadata.get("cipher") == "PRESENT-80"
        and int(source_config["rounds"]) == 3,
        "source_cache_complete": data_metadata.get("status") == "complete"
        and int(data_metadata.get("completed_rows", -1)) == total_rows,
        "source_arrays_have_expected_shapes": plaintexts.shape == (total_rows,)
        and features.shape == (total_rows, 64)
        and targets.shape == (total_rows, 64),
        "three_checkpoint_hashes_match": checkpoint_hashes_match,
        "discovery_is_source_test_split": test_rows > 0,
    }
    return {
        "metadata": metadata,
        "source_config": source_config,
        "data_metadata": data_metadata,
        "manifest": manifest_by_model,
        "plaintexts": plaintexts,
        "discovery_features": features[train_rows:],
        "discovery_targets": targets[train_rows:],
        "train_rows": train_rows,
        "test_rows": test_rows,
        "secret_key": int(metadata["secret_key_hex"], 16),
        "checks": checks,
    }


def evaluate_output_bits(
    source_root: Path,
    source: dict[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
    *,
    split: str,
    batch_size: int,
    device: str,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        model = _load_model(
            source_root,
            source["source_config"],
            source["manifest"][model_name],
            device=device,
        )
        scores = _predict_raw(model, features, batch_size=batch_size, device=device)
        rows.extend(per_bit_metric_rows(model_name, split, scores, targets))
        del model
        del scores
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        if progress is not None:
            progress("split_model_evaluated", {"split": split, "model": model_name})
    return rows


def per_bit_metric_rows(
    model_name: str,
    split: str,
    raw_scores: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, Any]]:
    scores = np.asarray(raw_scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    if scores.shape != labels.shape or scores.ndim != 2 or scores.shape[1] != 64:
        raise ValueError("raw scores and targets must both have shape [rows, 64]")
    rows: list[dict[str, Any]] = []
    for msb_index in range(64):
        bit_scores = scores[:, msb_index]
        bit_labels = labels[:, msb_index]
        predictions = bit_scores >= 0.5
        prevalence = float(np.mean(bit_labels))
        majority = max(prevalence, 1.0 - prevalence)
        rounded = np.rint(bit_scores)
        valid_rounded = (rounded == 0.0) | (rounded == 1.0)
        accuracy = float(np.mean(predictions == bit_labels))
        rows.append(
            {
                "split": split,
                "model": model_name,
                "target": "single_true_ciphertext_output_bit",
                "sample_classification": False,
                "msb_index": msb_index,
                "integer_bit": 63 - msb_index,
                "nibble_msb_index": msb_index // 4,
                "bit_in_nibble_msb": msb_index % 4,
                "rows": len(bit_labels),
                "prevalence": prevalence,
                "threshold_accuracy": accuracy,
                "majority_accuracy": majority,
                "accuracy_minus_majority": accuracy - majority,
                "auc": float(binary_auc(bit_labels, bit_scores)),
                "mse": float(np.mean(np.square(bit_scores - bit_labels))),
                "invalid_numpy_rint_rate": float(1.0 - np.mean(valid_rounded)),
            }
        )
    return rows


def select_discovery_candidates(
    config: OutputBitDiscoveryConfig,
    discovery_rows: list[dict[str, Any]],
    *,
    source_run_id: str,
) -> dict[str, Any]:
    indexed = {
        (row["model"], int(row["msb_index"])): row for row in discovery_rows
    }
    ranked: list[dict[str, Any]] = []
    for msb_index in range(64):
        shuffled_row = indexed[("kimura_lstm_label_shuffle", msb_index)]
        model_options: list[dict[str, Any]] = []
        for model_name in (
            "kimura_lstm_true_output",
            "matched_mlp_true_output",
        ):
            true_row = indexed[(model_name, msb_index)]
            auc_margin = float(true_row["auc"]) - float(shuffled_row["auc"])
            selection_score = min(
                float(true_row["auc"]) - 0.5,
                float(true_row["accuracy_minus_majority"]),
                auc_margin,
            )
            model_options.append(
                {
                    "selector_model": model_name,
                    "eligible": (
                        float(true_row["auc"]) >= config.minimum_auc
                        and float(true_row["accuracy_minus_majority"])
                        >= config.minimum_accuracy_margin
                        and auc_margin >= config.minimum_shuffle_auc_margin
                    ),
                    "selection_score": selection_score,
                    "discovery_auc": float(true_row["auc"]),
                    "discovery_accuracy": float(true_row["threshold_accuracy"]),
                    "discovery_majority_accuracy": float(
                        true_row["majority_accuracy"]
                    ),
                    "discovery_accuracy_margin": float(
                        true_row["accuracy_minus_majority"]
                    ),
                    "discovery_auc_minus_shuffle": auc_margin,
                    "shuffle_control_scope": (
                        "architecture_matched"
                        if model_name == "kimura_lstm_true_output"
                        else "cross_architecture_negative_control"
                    ),
                }
            )
        selected = max(
            model_options,
            key=lambda row: (
                float(row["selection_score"]),
                float(row["discovery_auc"]),
                row["selector_model"] == "kimura_lstm_true_output",
            ),
        )
        ranked.append(
            {
                "msb_index": msb_index,
                "integer_bit": 63 - msb_index,
                "nibble_msb_index": msb_index // 4,
                "bit_in_nibble_msb": msb_index % 4,
                **selected,
                "discovery_shuffle_auc": float(shuffled_row["auc"]),
                "lstm_option": next(
                    row
                    for row in model_options
                    if row["selector_model"] == "kimura_lstm_true_output"
                ),
                "mlp_option": next(
                    row
                    for row in model_options
                    if row["selector_model"] == "matched_mlp_true_output"
                ),
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            -float(row["discovery_auc"]),
            int(row["msb_index"]),
        )
    )
    candidates = [row for row in ranked if row["eligible"]][
        : config.candidate_limit
    ]
    return {
        "run_id": config.run_id,
        "source_run_id": source_run_id,
        "selection_split": "op9_disjoint_test_discovery_only",
        "confirmation_split": "fresh_plaintexts_not_evaluated_when_frozen",
        "bit_order": "msb_first",
        "selector_models": [
            "kimura_lstm_true_output",
            "matched_mlp_true_output",
        ],
        "shuffle_control_model": "kimura_lstm_label_shuffle",
        "mlp_shuffle_control_boundary": (
            "The available shuffled LSTM is a cross-architecture negative control for "
            "MLP-selected bits; OP11 must add matched_mlp_label_shuffle before an "
            "architecture-attribution or cross-key claim."
        ),
        "candidate_limit": config.candidate_limit,
        "thresholds": {
            "minimum_auc": config.minimum_auc,
            "minimum_accuracy_margin": config.minimum_accuracy_margin,
            "minimum_shuffle_auc_margin": config.minimum_shuffle_auc_margin,
        },
        "candidate_msb_indices": [row["msb_index"] for row in candidates],
        "candidates": candidates,
        "all_64_discovery_ranking": ranked,
    }


def prepare_fresh_output_bit_data(
    config: OutputBitDiscoveryConfig,
    source: dict[str, Any],
    output_root: Path,
    *,
    candidate_sha256: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    data_root = output_root / "fresh_data"
    data_root.mkdir(parents=True, exist_ok=True)
    metadata_path = data_root / "cache_metadata.json"
    source_plaintexts = {int(value) for value in source["plaintexts"]}
    expected = {
        "cache_version": FRESH_CACHE_VERSION,
        "source_run_id": source["metadata"]["run_id"],
        "source_plaintext_rows": len(source_plaintexts),
        "source_plaintexts_sha256": _sha256_array(source["plaintexts"]),
        "candidate_sha256": candidate_sha256,
        "rounds": int(source["source_config"]["rounds"]),
        "seed": config.seed,
        "fresh_rows": config.fresh_rows,
        "bit_order": "msb_first",
        "secret_key_hex": f"{int(source['secret_key']):020x}",
    }
    paths = {
        "plaintexts": data_root / "plaintexts.npy",
        "features": data_root / "features.npy",
        "full_targets": data_root / "full_targets.npy",
    }
    metadata = _read_json(metadata_path) if metadata_path.exists() else None
    if metadata is not None:
        if {key: metadata.get(key) for key in expected} != expected:
            raise ValueError("existing fresh output-bit cache parameters do not match")
    completed_rows = int(metadata.get("completed_rows", 0)) if metadata else 0
    arrays_exist = all(path.exists() for path in paths.values())
    if completed_rows and not arrays_exist:
        raise ValueError("fresh cache metadata reports progress but arrays are missing")
    if arrays_exist:
        plaintexts = np.load(paths["plaintexts"], mmap_mode="r+")
        features = np.load(paths["features"], mmap_mode="r+")
        targets = np.load(paths["full_targets"], mmap_mode="r+")
        if (
            plaintexts.shape != (config.fresh_rows,)
            or features.shape != (config.fresh_rows, 64)
            or targets.shape != (config.fresh_rows, 64)
        ):
            raise ValueError("existing fresh output-bit cache has invalid shapes")
    else:
        plaintexts = open_memmap(
            paths["plaintexts"],
            mode="w+",
            dtype=np.uint64,
            shape=(config.fresh_rows,),
        )
        features = open_memmap(
            paths["features"],
            mode="w+",
            dtype=np.float32,
            shape=(config.fresh_rows, 64),
        )
        targets = open_memmap(
            paths["full_targets"],
            mode="w+",
            dtype=np.float32,
            shape=(config.fresh_rows, 64),
        )
        completed_rows = 0
    rng = np.random.default_rng(1_020_000 + config.seed)
    if metadata and metadata.get("rng_state"):
        rng.bit_generator.state = metadata["rng_state"]
    seen = source_plaintexts | {int(value) for value in plaintexts[:completed_rows]}
    cipher = Present80(
        rounds=int(source["source_config"]["rounds"]), key=int(source["secret_key"])
    )
    shifts = np.arange(63, -1, -1, dtype=np.uint64)
    _write_json(
        metadata_path,
        {
            **expected,
            "status": "generating" if completed_rows < config.fresh_rows else "complete",
            "completed_rows": completed_rows,
            "rng_state": rng.bit_generator.state,
        },
    )
    while completed_rows < config.fresh_rows:
        stop = min(config.fresh_rows, completed_rows + config.fresh_chunk_rows)
        values: list[int] = []
        while len(values) < stop - completed_rows:
            low = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            high = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            value = low | (high << 32)
            if value not in seen:
                seen.add(value)
                values.append(value)
        words = np.asarray(values, dtype=np.uint64)
        ciphertexts = np.asarray(
            [cipher.encrypt(int(word)) for word in words], dtype=np.uint64
        )
        plaintexts[completed_rows:stop] = words
        features[completed_rows:stop] = (
            (words[:, None] >> shifts[None, :]) & np.uint64(1)
        ).astype(np.float32)
        targets[completed_rows:stop] = (
            (ciphertexts[:, None] >> shifts[None, :]) & np.uint64(1)
        ).astype(np.float32)
        plaintexts.flush()
        features.flush()
        targets.flush()
        completed_rows = stop
        _write_json(
            metadata_path,
            {
                **expected,
                "status": "complete" if stop == config.fresh_rows else "generating",
                "completed_rows": completed_rows,
                "rng_state": rng.bit_generator.state,
            },
        )
        if progress is not None:
            progress(
                "fresh_cache_chunk",
                {"completed_rows": completed_rows, "total_rows": config.fresh_rows},
            )
    return {
        "plaintexts": plaintexts,
        "features": features,
        "full_targets": targets,
        "metadata": _read_json(metadata_path),
        "data_root": data_root,
    }


def validate_fresh_output_bit_data(
    config: OutputBitDiscoveryConfig,
    source: dict[str, Any],
    fresh: dict[str, Any],
    *,
    candidate_path: Path,
    candidate_sha256: str,
) -> dict[str, bool]:
    plaintexts = fresh["plaintexts"]
    features = fresh["features"]
    targets = fresh["full_targets"]
    source_plaintexts = {int(value) for value in source["plaintexts"]}
    fresh_plaintexts = {int(value) for value in plaintexts}
    cipher = Present80(
        rounds=int(source["source_config"]["rounds"]), key=int(source["secret_key"])
    )
    sample_indices = sorted({0, config.fresh_rows - 1})
    return {
        "candidate_artifact_hash_matches": _sha256(candidate_path)
        == candidate_sha256,
        "fresh_cache_is_complete": fresh["metadata"]["status"] == "complete"
        and int(fresh["metadata"]["completed_rows"]) == config.fresh_rows,
        "fresh_arrays_have_expected_shapes": plaintexts.shape
        == (config.fresh_rows,)
        and features.shape == (config.fresh_rows, 64)
        and targets.shape == (config.fresh_rows, 64),
        "fresh_plaintexts_are_unique": len(fresh_plaintexts) == config.fresh_rows,
        "fresh_plaintexts_are_source_disjoint": source_plaintexts.isdisjoint(
            fresh_plaintexts
        ),
        "fresh_features_are_msb_first": all(
            _bits_to_word(features[index]) == int(plaintexts[index])
            for index in sample_indices
        ),
        "fresh_targets_are_true_ciphertext_bits": all(
            _bits_to_word(targets[index]) == cipher.encrypt(int(plaintexts[index]))
            for index in sample_indices
        ),
        "labels_are_output_values_not_sample_classes": True,
    }


def adjudicate_output_bit_discovery(
    config: OutputBitDiscoveryConfig,
    source_checks: dict[str, bool],
    fresh_checks: dict[str, bool],
    discovery_rows: list[dict[str, Any]],
    fresh_rows: list[dict[str, Any]],
    candidates: dict[str, Any],
) -> dict[str, Any]:
    expected_rows = 64 * len(MODEL_NAMES)
    fresh_index = {
        (row["model"], int(row["msb_index"])): row for row in fresh_rows
    }
    confirmed: list[dict[str, Any]] = []
    for candidate in candidates["candidates"]:
        bit = int(candidate["msb_index"])
        selector_model = str(candidate["selector_model"])
        true_row = fresh_index[(selector_model, bit)]
        shuffled_row = fresh_index[("kimura_lstm_label_shuffle", bit)]
        auc_margin = float(true_row["auc"]) - float(shuffled_row["auc"])
        checks = {
            "fresh_auc_at_least_0_510": float(true_row["auc"])
            >= config.minimum_auc,
            "fresh_accuracy_margin_at_least_0_005": float(
                true_row["accuracy_minus_majority"]
            )
            >= config.minimum_accuracy_margin,
            "fresh_auc_minus_shuffle_at_least_0_005": auc_margin
            >= config.minimum_shuffle_auc_margin,
        }
        confirmed.append(
            {
                **candidate,
                "fresh_selector_model": selector_model,
                "fresh_auc": float(true_row["auc"]),
                "fresh_accuracy": float(true_row["threshold_accuracy"]),
                "fresh_majority_accuracy": float(true_row["majority_accuracy"]),
                "fresh_accuracy_margin": float(
                    true_row["accuracy_minus_majority"]
                ),
                "fresh_shuffle_auc": float(shuffled_row["auc"]),
                "fresh_auc_minus_shuffle": auc_margin,
                "confirmation_checks": checks,
                "confirmed": all(checks.values()),
            }
        )
    protocol_checks = {
        **source_checks,
        **fresh_checks,
        "discovery_has_64_bits_x_3_models": len(discovery_rows) == expected_rows,
        "fresh_has_64_bits_x_3_models": len(fresh_rows) == expected_rows,
        "all_metrics_are_finite": all(
            math.isfinite(float(row[field]))
            for row in discovery_rows + fresh_rows
            for field in (
                "threshold_accuracy",
                "majority_accuracy",
                "accuracy_minus_majority",
                "auc",
                "mse",
                "invalid_numpy_rint_rate",
            )
        ),
        "candidate_count_within_frozen_limit": len(candidates["candidates"])
        <= config.candidate_limit,
        "candidate_selection_uses_discovery_only": candidates["selection_split"]
        == "op9_disjoint_test_discovery_only",
        "candidate_selector_models_are_frozen": all(
            row["selector_model"]
            in {"kimura_lstm_true_output", "matched_mlp_true_output"}
            for row in candidates["candidates"]
        ),
    }
    confirmed_bits = [row for row in confirmed if row["confirmed"]]
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation2_output_bit_discovery_protocol_invalid"
        action = "repair only checkpoint, bit-order, split, cache, or candidate-freeze protocol"
        next_adjudication = "op10_protocol_repair"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_output_bit_discovery_local_smoke_passed"
        action = "run frozen OP10 on the completed OP9 remote checkpoints and fresh 2^16 plaintexts"
        next_adjudication = "op10_remote_fresh_bit_confirmation"
    elif confirmed_bits:
        status = "pass"
        decision = "innovation2_true_output_bits_fresh_confirmed"
        action = "train dedicated heads only for confirmed bits and repeat them under one independent fixed key"
        next_adjudication = "op11_dedicated_bit_head_independent_key"
    else:
        status = "hold"
        decision = "innovation2_no_true_output_bit_fresh_confirmed"
        action = "do not scale rounds or full-output training; audit a dedicated single-bit head only as a multitask-interference diagnostic"
        next_adjudication = "selected_bit_head_feasibility_reassessment"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "metrics": {
            "discovery_candidate_count": len(candidates["candidates"]),
            "fresh_confirmed_count": len(confirmed_bits),
            "fresh_confirmed_msb_indices": [
                int(row["msb_index"]) for row in confirmed_bits
            ],
        },
        "candidate_confirmation": confirmed,
        "claim_scope": (
            "local implementation smoke"
            if config.mode == "smoke"
            else "single-fixed-key PRESENT r3 selected true-output-bit confirmation"
        )
        + "; not full-ciphertext recovery, sample classification, cross-key evidence, r4 evidence, or SOTA",
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "sample_classification": False,
            "target": "selected_true_ciphertext_output_bits",
        },
    }


def build_bit_ranking(
    discovery_rows: list[dict[str, Any]],
    fresh_rows: list[dict[str, Any]],
    candidates: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    discovery_index = {
        (row["model"], int(row["msb_index"])): row for row in discovery_rows
    }
    fresh_index = {
        (row["model"], int(row["msb_index"])): row for row in fresh_rows
    }
    rank_by_bit = {
        int(row["msb_index"]): rank
        for rank, row in enumerate(candidates["all_64_discovery_ranking"], start=1)
    }
    selection_by_bit = {
        int(row["msb_index"]): row
        for row in candidates["all_64_discovery_ranking"]
    }
    candidate_bits = set(candidates["candidate_msb_indices"])
    confirmed_bits = set(gate["metrics"]["fresh_confirmed_msb_indices"])
    rows: list[dict[str, Any]] = []
    for bit in range(64):
        discovery_true = discovery_index[("kimura_lstm_true_output", bit)]
        discovery_shuffle = discovery_index[("kimura_lstm_label_shuffle", bit)]
        discovery_mlp = discovery_index[("matched_mlp_true_output", bit)]
        fresh_true = fresh_index[("kimura_lstm_true_output", bit)]
        fresh_shuffle = fresh_index[("kimura_lstm_label_shuffle", bit)]
        fresh_mlp = fresh_index[("matched_mlp_true_output", bit)]
        selection = selection_by_bit[bit]
        fresh_selector = fresh_index[(selection["selector_model"], bit)]
        rows.append(
            {
                "discovery_rank": rank_by_bit[bit],
                "msb_index": bit,
                "integer_bit": 63 - bit,
                "nibble_msb_index": bit // 4,
                "bit_in_nibble_msb": bit % 4,
                "selected_candidate": bit in candidate_bits,
                "fresh_confirmed": bit in confirmed_bits,
                "selector_model": selection["selector_model"],
                "shuffle_control_scope": selection["shuffle_control_scope"],
                "discovery_selector_auc": selection["discovery_auc"],
                "fresh_selector_auc": fresh_selector["auc"],
                "fresh_selector_accuracy_margin": fresh_selector[
                    "accuracy_minus_majority"
                ],
                "discovery_lstm_auc": discovery_true["auc"],
                "discovery_shuffle_auc": discovery_shuffle["auc"],
                "discovery_mlp_auc": discovery_mlp["auc"],
                "discovery_lstm_accuracy_margin": discovery_true[
                    "accuracy_minus_majority"
                ],
                "fresh_lstm_auc": fresh_true["auc"],
                "fresh_shuffle_auc": fresh_shuffle["auc"],
                "fresh_mlp_auc": fresh_mlp["auc"],
                "fresh_lstm_accuracy_margin": fresh_true[
                    "accuracy_minus_majority"
                ],
            }
        )
    rows.sort(key=lambda row: int(row["discovery_rank"]))
    return rows


def serializable_config(config: OutputBitDiscoveryConfig) -> dict[str, Any]:
    return asdict(config)


def _load_model(
    source_root: Path,
    source_config: dict[str, Any],
    checkpoint_item: dict[str, Any],
    *,
    device: str,
) -> nn.Module:
    model_name = checkpoint_item["model"]
    if model_name == "matched_mlp_true_output":
        model: nn.Module = ParameterMatchedOutputMlp(
            int(source_config["mlp_hidden_dim"])
        )
    else:
        model = KimuraCiphertextLstm(
            int(source_config["hidden_dim"]), int(source_config["layers"])
        )
    checkpoint = torch.load(
        source_root / checkpoint_item["path"], map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def _predict_raw(
    model: nn.Module,
    features: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.array(features, copy=True))),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _bits_to_word(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8):
        value = (value << 1) | int(bit)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    flat = np.asarray(array).reshape(-1)
    step = max(1, (1 << 20) // flat.dtype.itemsize)
    for start in range(0, len(flat), step):
        digest.update(np.ascontiguousarray(flat[start : start + step]).tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
