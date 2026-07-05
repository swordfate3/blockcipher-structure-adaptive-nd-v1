from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.features.spn_candidate_evidence import present_pair_candidate_evidence_layers
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.difference_profiles import difference_for_profile
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.features.registry import pair_bits_for_encoding
from blockcipher_nd.tasks.innovation1.spn_candidate.dataset import make_candidate_dataset
from blockcipher_nd.tasks.innovation1.protocols import OFFICIAL_ZHANG_WANG_CASE2_MCND
from blockcipher_nd.training.metrics import binary_auc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SPN feature-level positive/negative separation before expensive training."
    )
    parser.add_argument("--cipher", default="present80")
    parser.add_argument("--rounds", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=2048)
    parser.add_argument("--pairs-per-sample", type=int, default=16)
    parser.add_argument(
        "--feature-encodings",
        nargs="+",
        default=[
            "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits",
            "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
            "present_pair_xor_paligned_sboxddt_beam4deep3_cell_matrix_bits",
            "present_pair_xor_paligned_sboxddt_beam8deep4_cell_matrix_bits",
        ],
    )
    parser.add_argument("--difference-profile", default="present_zhang_wang2022_mcnd")
    parser.add_argument("--difference-member", type=int, default=0)
    parser.add_argument("--negative-mode", default="encrypted_random_plaintexts")
    parser.add_argument("--key-rotation-interval", type=int, default=1024)
    parser.add_argument("--sample-structure", default=OFFICIAL_ZHANG_WANG_CASE2_MCND)
    parser.add_argument("--train-key", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output", default="outputs/innovation1/spn_feature_separation_audit.json")
    parser.add_argument("--beamstats-attribution-plan", type=Path, default=None)
    parser.add_argument("--candidate-evidence-feature-probe-config", type=Path, default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--key-split", choices=["train", "validation"], default="validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.candidate_evidence_feature_probe_config is not None:
        config = json.loads(args.candidate_evidence_feature_probe_config.read_text(encoding="utf-8"))
        payload = candidate_evidence_feature_probe_from_config(
            config,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            key_split=args.key_split,
            top_k=args.top_k,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.beamstats_attribution_plan is not None:
        tasks = tasks_from_plan(
            args.beamstats_attribution_plan,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=1,
            difference_profile=None,
            difference_member=0,
        )
        if args.row_index < 0 or args.row_index >= len(tasks):
            raise ValueError(f"row-index {args.row_index} outside plan rows 0..{len(tasks) - 1}")
        payload = beamstats_attribution_from_task(
            tasks[args.row_index],
            samples_per_class=args.samples_per_class,
            seed=args.seed if args.seed is not None else (args.seeds[0] if args.seeds else None),
            key_split=args.key_split,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    rows = run_audit(args)
    payload = {
        "kind": "spn_feature_separation_audit",
        "config": {
            "cipher": args.cipher,
            "rounds": args.rounds,
            "seeds": args.seeds,
            "samples_per_class": args.samples_per_class,
            "pairs_per_sample": args.pairs_per_sample,
            "feature_encodings": args.feature_encodings,
            "difference_profile": args.difference_profile,
            "difference_member": args.difference_member,
            "negative_mode": args.negative_mode,
            "key_rotation_interval": args.key_rotation_interval,
            "sample_structure": args.sample_structure,
            "train_key": args.train_key,
            "top_k": args.top_k,
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} audit rows to {output}")
    return 0


def run_audit(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_difference = difference_for_profile(args.difference_profile, args.difference_member)
    for rounds in args.rounds:
        for feature_encoding in args.feature_encodings:
            for seed in args.seeds:
                cipher = build_cipher(args.cipher, rounds, key=args.train_key)
                dataset = make_differential_dataset(
                    DifferentialDatasetConfig(
                        cipher=cipher,
                        input_difference=input_difference,
                        samples_per_class=args.samples_per_class,
                        seed=seed,
                        shuffle=True,
                        feature_encoding=feature_encoding,
                        pairs_per_sample=args.pairs_per_sample,
                        negative_mode=args.negative_mode,
                        key_rotation_interval=args.key_rotation_interval,
                        sample_structure=args.sample_structure,
                    )
                )
                rows.append(
                    audit_dataset(
                        dataset.features.astype(np.float32),
                        dataset.labels.astype(np.uint8),
                        pair_bits=pair_bits_for_encoding(cipher.block_bits, feature_encoding),
                        block_bits=cipher.block_bits,
                        cipher_name=cipher.name,
                        rounds=rounds,
                        seed=seed,
                        feature_encoding=feature_encoding,
                        top_k=args.top_k,
                    )
                )
    return rows


def audit_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    pair_bits: int,
    block_bits: int,
    cipher_name: str,
    rounds: int,
    seed: int,
    feature_encoding: str,
    top_k: int,
) -> dict[str, Any]:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if features.shape[1] % pair_bits != 0:
        raise ValueError("feature width must be a multiple of pair_bits")
    pairs_per_sample = features.shape[1] // pair_bits
    words_per_pair = pair_bits // block_bits
    cells_per_word = block_bits // 4

    bit_scores = _feature_axis_scores(features, labels)
    word_activity = features.reshape(features.shape[0], pairs_per_sample, words_per_pair, block_bits).mean(axis=3)
    word_scores = _feature_axis_scores(word_activity.reshape(features.shape[0], -1), labels)
    cell_activity = features.reshape(features.shape[0], pairs_per_sample, words_per_pair, cells_per_word, 4).mean(axis=4)
    cell_scores = _feature_axis_scores(cell_activity.reshape(features.shape[0], -1), labels)
    pair_activity = features.reshape(features.shape[0], pairs_per_sample, pair_bits).mean(axis=2)
    pair_scores = _feature_axis_scores(pair_activity, labels)
    global_scores = _named_scalar_scores(
        {
            "global_bit_density": features.mean(axis=1),
            "pair_activity_mean": pair_activity.mean(axis=1),
            "pair_activity_std": pair_activity.std(axis=1),
            "word_activity_mean": word_activity.mean(axis=(1, 2)),
            "word_activity_std": word_activity.std(axis=(1, 2)),
            "cell_activity_mean": cell_activity.mean(axis=(1, 2, 3)),
            "cell_activity_std": cell_activity.std(axis=(1, 2, 3)),
            "first_last_pair_activity_delta": pair_activity[:, -1] - pair_activity[:, 0],
            "first_last_word_activity_delta": word_activity[:, -1, -1] - word_activity[:, 0, 0],
        },
        labels,
    )

    top_bit_indices = _top_indices(bit_scores["auc_advantage"], top_k)
    top_word_indices = _top_indices(word_scores["auc_advantage"], top_k)
    top_cell_indices = _top_indices(cell_scores["auc_advantage"], top_k)
    return {
        "cipher": cipher_name,
        "rounds": rounds,
        "seed": seed,
        "feature_encoding": feature_encoding,
        "samples": int(features.shape[0]),
        "samples_per_class": int(min((labels == 0).sum(), (labels == 1).sum())),
        "input_bits": int(features.shape[1]),
        "pair_bits": int(pair_bits),
        "pairs_per_sample": int(pairs_per_sample),
        "words_per_pair": int(words_per_pair),
        "best_bit_auc_advantage": float(bit_scores["auc_advantage"][top_bit_indices[0]]) if top_bit_indices else 0.0,
        "best_word_auc_advantage": float(word_scores["auc_advantage"][top_word_indices[0]]) if top_word_indices else 0.0,
        "best_cell_auc_advantage": float(cell_scores["auc_advantage"][top_cell_indices[0]]) if top_cell_indices else 0.0,
        "best_pair_auc_advantage": float(pair_scores["auc_advantage"].max()) if pair_scores["auc_advantage"].size else 0.0,
        "global_scores": global_scores,
        "top_bits": _top_feature_rows(bit_scores, top_bit_indices),
        "top_words": _top_feature_rows(word_scores, top_word_indices),
        "top_cells": _top_feature_rows(cell_scores, top_cell_indices),
    }


def candidate_evidence_feature_probe_from_config(
    config: dict[str, Any],
    *,
    samples_per_class: int | None = None,
    seed: int | None = None,
    key_split: str = "validation",
    top_k: int = 12,
) -> dict[str, Any]:
    if key_split not in {"train", "validation"}:
        raise ValueError("key_split must be train or validation")

    resolved = _candidate_probe_config(config, samples_per_class=samples_per_class, seed=seed, key_split=key_split)
    input_difference = difference_for_profile(resolved["difference_profile"], resolved["difference_member"])
    features, labels = make_candidate_dataset(
        rounds=resolved["rounds"],
        key=resolved["key"],
        input_difference=input_difference,
        seed=resolved["seed"],
        samples_per_class=resolved["samples_per_class"],
        pairs_per_sample=resolved["pairs_per_sample"],
        negative_mode=resolved["negative_mode"],
        sample_structure=resolved["sample_structure"],
        key_rotation_interval=resolved["key_rotation_interval"],
        beam_width=resolved["beam_width"],
        depth=resolved["depth"],
        feature_mode=resolved["feature_mode"],
        feature_cache_root=None,
        split=key_split,
    )
    labels = labels.astype(np.uint8, copy=False)
    axis_scores = _feature_axis_scores(features.astype(np.float64, copy=False), labels)
    top_indices = _top_indices(axis_scores["auc_advantage"], top_k)
    top_rows = _top_feature_rows(axis_scores, top_indices)
    feature_names = _candidate_feature_axis_names(
        feature_dim=features.shape[1],
        feature_mode=resolved["feature_mode"],
    )
    composite_scores = _oriented_zscore_composite(features, labels, top_indices)
    composite = _scalar_statistic_report(composite_scores, labels.astype(np.float32, copy=False))
    best_axis = top_rows[0] if top_rows else {
        "index": 0,
        "positive_mean": 0.0,
        "negative_mean": 0.0,
        "mean_delta": 0.0,
        "cohen_d": 0.0,
        "auc": 0.5,
        "auc_advantage": 0.0,
    }
    decision = (
        "candidate_evidence_lowdim_probe_positive"
        if max(float(best_axis["auc_advantage"]), composite["auc_advantage"]) >= 0.02
        else "candidate_evidence_lowdim_probe_weak_or_negative"
    )
    return {
        "status": "pass",
        "audit": "candidate_evidence_feature_probe",
        "rounds": resolved["rounds"],
        "samples_per_class": resolved["samples_per_class"],
        "seed": resolved["seed"],
        "key_split": key_split,
        "sample_structure": resolved["sample_structure"],
        "negative_mode": resolved["negative_mode"],
        "pairs_per_sample": resolved["pairs_per_sample"],
        "difference_profile": resolved["difference_profile"],
        "difference_member": resolved["difference_member"],
        "input_difference": input_difference,
        "beam_width": resolved["beam_width"],
        "depth": resolved["depth"],
        "feature_mode": resolved["feature_mode"],
        "feature_dim": int(features.shape[1]),
        "best_axis": {
            **best_axis,
            "name": feature_names[int(best_axis["index"])] if feature_names else f"axis_{best_axis['index']}",
        },
        "top_axes": {
            "feature_names": [feature_names[index] for index in top_indices],
            "rows": top_rows,
        },
        "composite": {
            **composite,
            "axis_indices": top_indices,
            "combiner": "top_axis_oriented_zscore_mean",
        },
        "decision": decision,
        "claim_scope": (
            "Local candidate-evidence feature probe only; not neural training, "
            "not scale evidence, and not a remote launch gate."
        ),
    }


def beamstats_attribution_from_task(
    task: dict[str, Any],
    *,
    samples_per_class: int,
    seed: int | None = None,
    key_split: str = "validation",
    beam_width: int = 4,
    depth: int = 3,
) -> dict[str, Any]:
    if task["feature_encoding"] != "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits":
        raise ValueError("beamstats attribution currently expects the r8 GPD-style beamstats feature")
    if key_split not in {"train", "validation"}:
        raise ValueError("key_split must be train or validation")

    key = task["validation_key"] if key_split == "validation" else task["train_key"]
    cipher = build_cipher(task["cipher_key"], task["rounds"], key=key)
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=task["input_difference"],
            samples_per_class=samples_per_class,
            seed=task["seed"] if seed is None else seed,
            shuffle=False,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=task["pairs_per_sample"],
            negative_mode=task["negative_mode"],
            key_rotation_interval=task["key_rotation_interval"],
            sample_structure=task["sample_structure"],
            integral_active_nibble=task["integral_active_nibble"],
            selected_bit_indices=task["selected_bit_indices"],
        )
    )
    pair_words = _ciphertext_pair_rows_to_words(
        dataset.features.astype(np.uint8, copy=False),
        pairs_per_sample=task["pairs_per_sample"],
        block_bits=cipher.block_bits,
    )
    scores = _beamstats_semantic_scores(
        pair_words,
        cipher=cipher,
        beam_width=beam_width,
        depth=depth,
    )
    labels = dataset.labels.astype(np.float32, copy=False)
    statistics = {name: _scalar_statistic_report(values, labels) for name, values in scores.items()}
    best_name, best_report = max(
        statistics.items(),
        key=lambda item: item[1]["auc_advantage"],
    )
    return {
        "status": "pass",
        "audit": "present_beamstats_semantic_attribution",
        "cipher_key": task["cipher_key"],
        "rounds": task["rounds"],
        "samples_per_class": samples_per_class,
        "seed": task["seed"] if seed is None else seed,
        "key_split": key_split,
        "sample_structure": task["sample_structure"],
        "negative_mode": task["negative_mode"],
        "feature_encoding": task["feature_encoding"],
        "pairs_per_sample": task["pairs_per_sample"],
        "input_difference": task["input_difference"],
        "beam_width": beam_width,
        "depth": depth,
        "statistics": statistics,
        "best_statistic": {
            "name": best_name,
            "auc": best_report["auc"],
            "auc_advantage": best_report["auc_advantage"],
            "best_threshold": best_report["best_threshold"],
        },
        "claim_scope": (
            "Local semantic attribution diagnostic only; not neural training, "
            "not scale evidence, and not a remote launch gate."
        ),
    }


def _feature_axis_scores(features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    positive = features[labels == 1]
    negative = features[labels == 0]
    pos_mean = positive.mean(axis=0)
    neg_mean = negative.mean(axis=0)
    pos_var = positive.var(axis=0)
    neg_var = negative.var(axis=0)
    pooled_std = np.sqrt((pos_var + neg_var) / 2.0)
    cohen_d = np.divide(pos_mean - neg_mean, pooled_std, out=np.zeros_like(pos_mean), where=pooled_std > 0)
    auc = np.array([binary_auc(labels, features[:, index]) for index in range(features.shape[1])], dtype=np.float64)
    return {
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "mean_delta": pos_mean - neg_mean,
        "cohen_d": cohen_d,
        "auc": auc,
        "auc_advantage": np.abs(auc - 0.5),
    }


def _candidate_probe_config(
    config: dict[str, Any],
    *,
    samples_per_class: int | None,
    seed: int | None,
    key_split: str,
) -> dict[str, Any]:
    key_field = "validation_key" if key_split == "validation" else "train_key"
    return {
        "rounds": int(config.get("rounds", 7)),
        "seed": int(config.get("seed", 0) if seed is None else seed),
        "samples_per_class": int(config.get("samples_per_class", 4096) if samples_per_class is None else samples_per_class),
        "pairs_per_sample": int(config.get("pairs_per_sample", 16)),
        "negative_mode": str(config.get("negative_mode", "encrypted_random_plaintexts")),
        "sample_structure": str(config.get("sample_structure", OFFICIAL_ZHANG_WANG_CASE2_MCND)),
        "difference_profile": str(config.get("difference_profile", "present_zhang_wang2022_mcnd")),
        "difference_member": int(config.get("difference_member", 0)),
        "key": _parse_int_like(config.get(key_field, config.get("train_key", 0))),
        "key_rotation_interval": int(config.get("key_rotation_interval", 0)),
        "beam_width": int(config.get("beam_width", 4)),
        "depth": int(config.get("depth", 3)),
        "feature_mode": str(config.get("feature_mode", "aggregate")),
    }


def _parse_int_like(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def _candidate_feature_axis_names(*, feature_dim: int, feature_mode: str) -> list[str]:
    return [f"{feature_mode}_axis_{index}" for index in range(feature_dim)]


def _oriented_zscore_composite(features: np.ndarray, labels: np.ndarray, axis_indices: list[int]) -> np.ndarray:
    if not axis_indices:
        return np.zeros(features.shape[0], dtype=np.float64)
    columns = []
    for index in axis_indices:
        scores = features[:, index].astype(np.float64, copy=False)
        auc = binary_auc(labels, scores)
        oriented = scores if auc >= 0.5 else -scores
        std = float(oriented.std())
        if std <= 0.0:
            columns.append(np.zeros_like(oriented, dtype=np.float64))
        else:
            columns.append((oriented - float(oriented.mean())) / std)
    return np.stack(columns, axis=1).mean(axis=1)


def _ciphertext_pair_rows_to_words(
    features: np.ndarray,
    *,
    pairs_per_sample: int,
    block_bits: int,
) -> np.ndarray:
    expected_bits = pairs_per_sample * block_bits * 2
    if features.ndim != 2 or features.shape[1] != expected_bits:
        raise ValueError("ciphertext_pair_bits rows have unexpected shape")
    bit_rows = features.reshape(features.shape[0], pairs_per_sample, 2, block_bits)
    weights = (1 << np.arange(block_bits - 1, -1, -1, dtype=np.uint64)).reshape(1, 1, 1, block_bits)
    return (bit_rows.astype(np.uint64) * weights).sum(axis=3)


def _beamstats_semantic_scores(
    pair_words: np.ndarray,
    *,
    cipher: Any,
    beam_width: int,
    depth: int,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[float]] = {
        "top_active_mean": [],
        "confidence_mean": [],
        "confidence_std": [],
        "margin_mean": [],
        "margin_std": [],
        "margin_positive_rate": [],
        "disagreement_nonzero_rate": [],
        "confidence_union_mean": [],
        "margin_union_mean": [],
        "score_mean": [],
        "score_max": [],
        "cumulative_mean": [],
        "cumulative_max": [],
        "active_mean": [],
        "active_max": [],
    }
    for sample_pairs in pair_words:
        collected = {name: [] for name in rows}
        for left, right in sample_pairs:
            layers = present_pair_candidate_evidence_layers(
                int(left),
                int(right),
                width=int(cipher.block_bits),
                cipher=cipher,
                beam_width=beam_width,
                depth=depth,
                source="structural_inverse",
            )
            for layer in layers:
                confidence_values = _nibbles(layer.confidence_word, cipher.block_bits)
                margin_values = _nibbles(layer.margin_word, cipher.block_bits)
                disagreement_values = _nibbles(layer.disagreement_word, cipher.block_bits)
                confidence_union_values = _nibbles(layer.confidence_union_word, cipher.block_bits)
                margin_union_values = _nibbles(layer.margin_union_word, cipher.block_bits)
                score_values = _nibbles(layer.score_word, 4 * beam_width)
                cumulative_values = _nibbles(layer.cumulative_word, 4 * beam_width)
                active_values = _nibbles(layer.active_word, 4 * beam_width)
                collected["top_active_mean"].append(_active_nibble_count(layer.top_word, cipher.block_bits))
                collected["confidence_mean"].append(float(np.mean(confidence_values)))
                collected["confidence_std"].append(float(np.std(confidence_values)))
                collected["margin_mean"].append(float(np.mean(margin_values)))
                collected["margin_std"].append(float(np.std(margin_values)))
                collected["margin_positive_rate"].append(_positive_rate(margin_values))
                collected["disagreement_nonzero_rate"].append(_positive_rate(disagreement_values))
                collected["confidence_union_mean"].append(float(np.mean(confidence_union_values)))
                collected["margin_union_mean"].append(float(np.mean(margin_union_values)))
                collected["score_mean"].append(float(np.mean(score_values)))
                collected["score_max"].append(float(max(score_values) if score_values else 0.0))
                collected["cumulative_mean"].append(float(np.mean(cumulative_values)))
                collected["cumulative_max"].append(float(max(cumulative_values) if cumulative_values else 0.0))
                collected["active_mean"].append(float(np.mean(active_values)))
                collected["active_max"].append(float(max(active_values) if active_values else 0.0))
        for name, values in collected.items():
            rows[name].append(float(np.mean(values)) if values else 0.0)
    return {name: np.array(values, dtype=np.float64) for name, values in rows.items()}


def _scalar_statistic_report(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    auc = binary_auc(labels, scores)
    return {
        "positive_mean": float(positive.mean()) if positive.size else 0.0,
        "negative_mean": float(negative.mean()) if negative.size else 0.0,
        "mean_delta": float(positive.mean() - negative.mean()) if positive.size and negative.size else 0.0,
        "auc": float(auc),
        "auc_advantage": float(abs(auc - 0.5)),
        "best_threshold": _best_threshold_report(scores, labels),
    }


def _best_threshold_report(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    best = {"accuracy": -1.0, "threshold": 0.0, "operator": ">="}
    bool_labels = labels == 1
    for threshold in np.unique(scores):
        for operator in ("<=", ">="):
            predicted = scores <= threshold if operator == "<=" else scores >= threshold
            accuracy = float((predicted == bool_labels).mean())
            if accuracy > best["accuracy"]:
                best = {
                    "accuracy": accuracy,
                    "threshold": float(threshold),
                    "operator": operator,
                }
    return best


def _nibbles(word: int, width: int) -> list[int]:
    return [int((word >> (4 * index)) & 0xF) for index in range(width // 4)]


def _positive_rate(values: list[int]) -> float:
    return float(sum(1 for value in values if value > 0) / len(values)) if values else 0.0


def _active_nibble_count(word: int, width: int) -> float:
    return float(sum(1 for value in _nibbles(word, width) if value > 0))


def _named_scalar_scores(named_scores: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, dict[str, float]]:
    rows = {}
    for name, scores in named_scores.items():
        scores = scores.astype(np.float64)
        positive = scores[labels == 1]
        negative = scores[labels == 0]
        pooled_std = np.sqrt((positive.var() + negative.var()) / 2.0)
        cohen_d = float((positive.mean() - negative.mean()) / pooled_std) if pooled_std > 0 else 0.0
        auc = binary_auc(labels, scores)
        rows[name] = {
            "positive_mean": float(positive.mean()),
            "negative_mean": float(negative.mean()),
            "mean_delta": float(positive.mean() - negative.mean()),
            "cohen_d": cohen_d,
            "auc": float(auc),
            "auc_advantage": float(abs(auc - 0.5)),
        }
    return rows


def _top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if top_k <= 0 or scores.size == 0:
        return []
    count = min(top_k, scores.size)
    return np.argsort(scores, kind="mergesort")[-count:][::-1].astype(int).tolist()


def _top_feature_rows(scores: dict[str, np.ndarray], indices: list[int]) -> list[dict[str, float | int]]:
    rows = []
    for index in indices:
        rows.append(
            {
                "index": int(index),
                "positive_mean": float(scores["positive_mean"][index]),
                "negative_mean": float(scores["negative_mean"][index]),
                "mean_delta": float(scores["mean_delta"][index]),
                "cohen_d": float(scores["cohen_d"][index]),
                "auc": float(scores["auc"][index]),
                "auc_advantage": float(scores["auc_advantage"][index]),
            }
        )
    return rows


if __name__ == "__main__":
    main()
