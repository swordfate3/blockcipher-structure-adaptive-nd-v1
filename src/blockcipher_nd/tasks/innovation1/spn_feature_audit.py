from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.features.encoders.present_sbox_ddt import parse_parameterized_present_sboxddt_encoding
from blockcipher_nd.models.structure.spn.present_pairset_global_stats_hybrid import (
    present_global_pairset_statistics,
    present_global_stats_feature_bits,
)
from blockcipher_nd.models.structure.spn.present_trail_position_stats import (
    PresentTrailPositionStatsPairSetDistinguisher,
)
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
    parser.add_argument("--trail-position-attribution-plan", type=Path, default=None)
    parser.add_argument("--trail-position-split-baseline-plan", type=Path, default=None)
    parser.add_argument("--candidate-evidence-feature-probe-config", type=Path, default=None)
    parser.add_argument("--sgp-stable-axis-config", type=Path, default=None)
    parser.add_argument("--sgp-grouped-axis-config", type=Path, default=None)
    parser.add_argument("--invp-global-stats-config", type=Path, default=None)
    parser.add_argument("--invp-group-distribution-config", type=Path, default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--key-split", choices=["train", "validation"], default="validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.sgp_stable_axis_config is not None:
        config = json.loads(args.sgp_stable_axis_config.read_text(encoding="utf-8"))
        payload = sgp_stable_axis_audit_from_config(
            config,
            samples_per_class=args.samples_per_class,
            top_k=args.top_k,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.sgp_grouped_axis_config is not None:
        config = json.loads(args.sgp_grouped_axis_config.read_text(encoding="utf-8"))
        payload = sgp_grouped_axis_audit_from_config(
            config,
            samples_per_class=args.samples_per_class,
            top_k=args.top_k,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.invp_global_stats_config is not None:
        config = json.loads(args.invp_global_stats_config.read_text(encoding="utf-8"))
        payload = invp_global_stats_audit_from_config(
            config,
            samples_per_class=args.samples_per_class,
            top_k=args.top_k,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.invp_group_distribution_config is not None:
        config = json.loads(args.invp_group_distribution_config.read_text(encoding="utf-8"))
        payload = invp_group_distribution_audit_from_config(
            config,
            samples_per_class=args.samples_per_class,
            top_k=args.top_k,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

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

    if args.trail_position_attribution_plan is not None:
        tasks = tasks_from_plan(
            args.trail_position_attribution_plan,
            feature_encoding=None,
            pairs_per_sample=None,
            difference_profile=None,
            difference_member=0,
        )
        if args.row_index < 0 or args.row_index >= len(tasks):
            raise ValueError(f"row-index {args.row_index} outside plan rows 0..{len(tasks) - 1}")
        payload = trail_position_attribution_from_task(
            tasks[args.row_index],
            samples_per_class=args.samples_per_class,
            seed=args.seed if args.seed is not None else (args.seeds[0] if args.seeds else None),
            key_split=args.key_split,
            top_k=args.top_k,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.trail_position_split_baseline_plan is not None:
        tasks = tasks_from_plan(
            args.trail_position_split_baseline_plan,
            feature_encoding=None,
            pairs_per_sample=None,
            difference_profile=None,
            difference_member=0,
        )
        if args.row_index < 0 or args.row_index >= len(tasks):
            raise ValueError(f"row-index {args.row_index} outside plan rows 0..{len(tasks) - 1}")
        payload = trail_position_split_baseline_from_task(
            tasks[args.row_index],
            samples_per_class=args.samples_per_class,
            seed=args.seed if args.seed is not None else (args.seeds[0] if args.seeds else None),
            top_k=args.top_k,
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


def sgp_stable_axis_audit_from_config(
    config: dict[str, Any],
    *,
    samples_per_class: int | None = None,
    top_k: int = 12,
) -> dict[str, Any]:
    resolved = _sgp_config(config, samples_per_class=samples_per_class, top_k=top_k)
    source_reports = []
    for source in resolved["feature_sources"]:
        probes = []
        for seed in resolved["seeds"]:
            for key_split in resolved["key_splits"]:
                probes.append(
                    _sgp_source_matrix(
                        source,
                        resolved,
                        seed=seed,
                        key_split=key_split,
                    )
                )
        controls = [
            _sgp_source_matrix(
                control,
                resolved,
                seed=resolved["seeds"][0],
                key_split=resolved["key_splits"][0],
            )
            for control in resolved["control_sources"]
            if int(control.get("feature_dim", -1)) in {-1, int(probes[0]["features"].shape[1])}
        ]
        source_reports.append(
            sgp_stable_axis_report_from_feature_matrices(
                probes,
                controls=controls,
                top_k=resolved["top_k"],
                stable_top_k=resolved["stable_top_k"],
                min_composite_auc=resolved["min_composite_auc"],
                min_topk_jaccard=resolved["min_topk_jaccard"],
                min_control_delta=resolved["min_control_delta"],
                source_name=str(source["name"]),
            )
        )
    best_report = max(
        source_reports,
        key=lambda item: (
            item["decision"] == "sgp_stable_axis_candidate",
            item["summary"]["probe_composite_auc_min"],
            item["control_gap"]["best_composite_auc_delta"],
        ),
    )
    return {
        "status": "pass",
        "audit": "sgp_stable_axis_audit",
        "decision": best_report["decision"],
        "best_source": best_report["source_name"],
        "config": {
            "rounds": resolved["rounds"],
            "samples_per_class": resolved["samples_per_class"],
            "seeds": resolved["seeds"],
            "key_splits": resolved["key_splits"],
            "pairs_per_sample": resolved["pairs_per_sample"],
            "negative_mode": resolved["negative_mode"],
            "sample_structure": resolved["sample_structure"],
            "difference_profile": resolved["difference_profile"],
            "difference_member": resolved["difference_member"],
            "top_k": resolved["top_k"],
            "stable_top_k": resolved["stable_top_k"],
            "min_composite_auc": resolved["min_composite_auc"],
            "min_topk_jaccard": resolved["min_topk_jaccard"],
            "min_control_delta": resolved["min_control_delta"],
        },
        "source_reports": source_reports,
        "candidate_masks": best_report["candidate_masks"],
        "claim_scope": (
            "Local SGP stable-axis audit only; not neural training, not scale "
            "evidence, not a remote launch gate, and not an ensemble result."
        ),
    }


def sgp_stable_axis_report_from_feature_matrices(
    probes: list[dict[str, Any]],
    *,
    controls: list[dict[str, Any]] | None = None,
    top_k: int = 12,
    stable_top_k: int = 32,
    min_composite_auc: float = 0.55,
    min_topk_jaccard: float = 0.35,
    min_control_delta: float = 0.01,
    source_name: str = "inline",
) -> dict[str, Any]:
    if len(probes) < 2:
        raise ValueError("SGP stable-axis audit requires at least two probes")
    feature_dim = int(np.asarray(probes[0]["features"]).shape[1])
    probe_reports = []
    top_sets: list[set[int]] = []
    axis_advantages = []
    for probe in probes:
        features, labels = _probe_features_and_labels(probe)
        if features.shape[1] != feature_dim:
            raise ValueError("all SGP probes for one source must share feature width")
        axis_scores = _feature_axis_scores(features, labels)
        top_indices = _top_indices(axis_scores["auc_advantage"], top_k)
        top_sets.append(set(top_indices))
        axis_advantages.append(axis_scores["auc_advantage"])
        probe_reports.append(
            {
                "name": str(probe.get("name", f"probe{len(probe_reports)}")),
                "samples_per_class": int(min((labels == 0).sum(), (labels == 1).sum())),
                "feature_dim": feature_dim,
                "top_axes": _top_feature_rows(axis_scores, top_indices),
            }
        )

    stable_axes = _stable_axes_from_probe_scores(
        top_sets=top_sets,
        axis_advantages=axis_advantages,
        stable_top_k=stable_top_k,
        feature_dim=feature_dim,
    )
    probe_composites = []
    for probe, report in zip(probes, probe_reports, strict=True):
        features, labels = _probe_features_and_labels(probe)
        composite_scores = _oriented_zscore_composite(features, labels, stable_axes)
        composite = _scalar_statistic_report(composite_scores, labels.astype(np.float32, copy=False))
        report["stable_axis_composite"] = composite
        probe_composites.append(composite)

    control_reports = []
    for control in controls or []:
        features, labels = _probe_features_and_labels(control)
        if features.shape[1] != feature_dim:
            continue
        composite_scores = _oriented_zscore_composite(features, labels, stable_axes)
        control_reports.append(
            {
                "name": str(control.get("name", f"control{len(control_reports)}")),
                "stable_axis_composite": _scalar_statistic_report(
                    composite_scores,
                    labels.astype(np.float32, copy=False),
                ),
            }
        )

    topk_jaccard_min = _topk_jaccard_min(top_sets)
    probe_composite_auc_min = min(float(item["auc"]) for item in probe_composites) if probe_composites else 0.5
    control_auc_max = (
        max(float(item["stable_axis_composite"]["auc"]) for item in control_reports) if control_reports else 0.5
    )
    control_delta = probe_composite_auc_min - control_auc_max
    decision = (
        "sgp_stable_axis_candidate"
        if probe_composite_auc_min >= min_composite_auc
        and topk_jaccard_min >= min_topk_jaccard
        and control_delta >= min_control_delta
        else "sgp_stable_axis_hold"
    )
    return {
        "audit": "sgp_stable_axis_audit",
        "source_name": source_name,
        "decision": decision,
        "summary": {
            "feature_dim": feature_dim,
            "probe_count": len(probe_reports),
            "control_count": len(control_reports),
            "probe_composite_auc_min": probe_composite_auc_min,
            "probe_composite_auc_mean": float(np.mean([item["auc"] for item in probe_composites])),
        },
        "stability": {
            "top_k": top_k,
            "stable_top_k": stable_top_k,
            "topk_jaccard_min": topk_jaccard_min,
            "stable_axes": stable_axes,
        },
        "control_gap": {
            "control_composite_auc_max": control_auc_max,
            "best_composite_auc_delta": control_delta,
            "min_control_delta": min_control_delta,
        },
        "candidate_masks": {f"sgp_top{len(stable_axes)}_stable": stable_axes},
        "probe_reports": probe_reports,
        "control_reports": control_reports,
        "claim_scope": (
            "Local SGP stable-axis audit only; not neural training, not scale evidence, "
            "not a remote launch gate, and not an ensemble result."
        ),
    }


def sgp_grouped_axis_audit_from_config(
    config: dict[str, Any],
    *,
    samples_per_class: int | None = None,
    top_k: int = 12,
) -> dict[str, Any]:
    resolved = _sgp_config(config, samples_per_class=samples_per_class, top_k=top_k)
    group_schemes = [str(scheme) for scheme in config.get("group_schemes", ["word_cell"])]
    reports = []
    for source in resolved["feature_sources"]:
        probes = []
        for seed in resolved["seeds"]:
            for key_split in resolved["key_splits"]:
                probes.append(
                    _sgp_source_matrix(
                        source,
                        resolved,
                        seed=seed,
                        key_split=key_split,
                    )
                )
        controls = [
            _sgp_source_matrix(
                control,
                resolved,
                seed=resolved["seeds"][0],
                key_split=resolved["key_splits"][0],
            )
            for control in resolved["control_sources"]
            if int(control.get("feature_dim", -1)) in {-1, int(probes[0]["features"].shape[1])}
        ]
        for group_scheme in group_schemes:
            reports.append(
                sgp_grouped_axis_report_from_feature_matrices(
                    probes,
                    controls=controls,
                    axis_groups=_axis_groups_for_sgp_source(
                        {
                            **source,
                            "pairs_per_sample": int(source.get("pairs_per_sample", resolved["pairs_per_sample"])),
                        },
                        feature_dim=int(probes[0]["features"].shape[1]),
                        group_scheme=group_scheme,
                    ),
                    top_k=resolved["top_k"],
                    stable_top_k=resolved["stable_top_k"],
                    min_composite_auc=resolved["min_composite_auc"],
                    min_topk_jaccard=resolved["min_topk_jaccard"],
                    min_control_delta=resolved["min_control_delta"],
                    max_selected_axis_fraction=resolved["max_selected_axis_fraction"],
                    source_name=str(source["name"]),
                    group_scheme=group_scheme,
                )
            )
    best_report = max(
        reports,
        key=lambda item: (
            item["decision"] == "sgp_grouped_axis_candidate",
            item["summary"]["probe_composite_auc_min"],
            item["control_gap"]["best_composite_auc_delta"],
        ),
    )
    return {
        "status": "pass",
        "audit": "sgp_grouped_axis_audit",
        "decision": best_report["decision"],
        "best_source": best_report["source_name"],
        "best_group_scheme": best_report["group_scheme"],
        "config": {
            "rounds": resolved["rounds"],
            "samples_per_class": resolved["samples_per_class"],
            "seeds": resolved["seeds"],
            "key_splits": resolved["key_splits"],
            "pairs_per_sample": resolved["pairs_per_sample"],
            "negative_mode": resolved["negative_mode"],
            "sample_structure": resolved["sample_structure"],
            "difference_profile": resolved["difference_profile"],
            "difference_member": resolved["difference_member"],
            "top_k": resolved["top_k"],
            "stable_top_k": resolved["stable_top_k"],
            "min_composite_auc": resolved["min_composite_auc"],
            "min_topk_jaccard": resolved["min_topk_jaccard"],
            "min_control_delta": resolved["min_control_delta"],
            "max_selected_axis_fraction": resolved["max_selected_axis_fraction"],
            "group_schemes": group_schemes,
        },
        "source_reports": reports,
        "candidate_masks": best_report["candidate_masks"],
        "claim_scope": (
            "Local SGP grouped-axis audit only; not neural training, not scale "
            "evidence, not a remote launch gate, and not an ensemble result."
        ),
    }


def sgp_grouped_axis_report_from_feature_matrices(
    probes: list[dict[str, Any]],
    *,
    axis_groups: list[str],
    controls: list[dict[str, Any]] | None = None,
    top_k: int = 12,
    stable_top_k: int = 8,
    min_composite_auc: float = 0.55,
    min_topk_jaccard: float = 0.35,
    min_control_delta: float = 0.01,
    max_selected_axis_fraction: float = 0.75,
    source_name: str = "inline",
    group_scheme: str = "word_cell",
) -> dict[str, Any]:
    if len(probes) < 2:
        raise ValueError("SGP grouped-axis audit requires at least two probes")
    feature_dim = int(np.asarray(probes[0]["features"]).shape[1])
    if len(axis_groups) != feature_dim:
        raise ValueError("axis_groups must match feature width")
    group_to_axes = _group_to_axes(axis_groups)
    group_ids = sorted(group_to_axes)
    probe_reports = []
    top_sets: list[set[str]] = []
    group_advantages = []
    for probe in probes:
        features, labels = _probe_features_and_labels(probe)
        if features.shape[1] != feature_dim:
            raise ValueError("all SGP probes for one source must share feature width")
        axis_scores = _feature_axis_scores(features, labels)
        group_scores = _group_axis_advantages(axis_scores["auc_advantage"], group_to_axes, group_ids)
        top_groups = _top_group_ids(group_scores, group_ids, top_k)
        top_sets.append(set(top_groups))
        group_advantages.append(group_scores)
        probe_reports.append(
            {
                "name": str(probe.get("name", f"probe{len(probe_reports)}")),
                "samples_per_class": int(min((labels == 0).sum(), (labels == 1).sum())),
                "feature_dim": feature_dim,
                "top_groups": [
                    {
                        "group": group_id,
                        "axis_count": len(group_to_axes[group_id]),
                        "auc_advantage_max": float(group_scores[index]),
                    }
                    for index, group_id in enumerate(group_ids)
                    if group_id in top_groups
                ],
            }
        )

    stable_groups = _stable_groups_from_probe_scores(
        top_sets=top_sets,
        group_advantages=group_advantages,
        group_ids=group_ids,
        stable_top_k=stable_top_k,
    )
    stable_axes = _axes_for_groups(stable_groups, group_to_axes)
    probe_composites = []
    for probe, report in zip(probes, probe_reports, strict=True):
        features, labels = _probe_features_and_labels(probe)
        composite_scores = _oriented_zscore_composite(features, labels, stable_axes)
        composite = _scalar_statistic_report(composite_scores, labels.astype(np.float32, copy=False))
        report["stable_group_composite"] = composite
        probe_composites.append(composite)

    control_reports = []
    for control in controls or []:
        features, labels = _probe_features_and_labels(control)
        if features.shape[1] != feature_dim:
            continue
        composite_scores = _oriented_zscore_composite(features, labels, stable_axes)
        control_reports.append(
            {
                "name": str(control.get("name", f"control{len(control_reports)}")),
                "stable_group_composite": _scalar_statistic_report(
                    composite_scores,
                    labels.astype(np.float32, copy=False),
                ),
            }
        )

    topk_jaccard_min = _group_jaccard_min(top_sets)
    probe_composite_auc_min = min(float(item["auc"]) for item in probe_composites) if probe_composites else 0.5
    control_auc_max = (
        max(float(item["stable_group_composite"]["auc"]) for item in control_reports) if control_reports else 0.5
    )
    control_delta = probe_composite_auc_min - control_auc_max
    selected_axis_fraction = float(len(stable_axes) / feature_dim) if feature_dim else 1.0
    decision = (
        "sgp_grouped_axis_candidate"
        if probe_composite_auc_min >= min_composite_auc
        and topk_jaccard_min >= min_topk_jaccard
        and control_delta >= min_control_delta
        and selected_axis_fraction <= max_selected_axis_fraction
        else "sgp_grouped_axis_hold"
    )
    mask_name = f"sgp_grouped_top{len(stable_groups)}_{group_scheme}"
    return {
        "audit": "sgp_grouped_axis_audit",
        "source_name": source_name,
        "group_scheme": group_scheme,
        "decision": decision,
        "summary": {
            "feature_dim": feature_dim,
            "group_count": len(group_ids),
            "probe_count": len(probe_reports),
            "control_count": len(control_reports),
            "probe_composite_auc_min": probe_composite_auc_min,
            "probe_composite_auc_mean": float(np.mean([item["auc"] for item in probe_composites])),
        },
        "stability": {
            "top_k": top_k,
            "stable_top_k": stable_top_k,
            "topk_jaccard_min": topk_jaccard_min,
            "stable_groups": stable_groups,
            "stable_axes": stable_axes,
        },
        "control_gap": {
            "control_composite_auc_max": control_auc_max,
            "best_composite_auc_delta": control_delta,
            "min_control_delta": min_control_delta,
        },
        "degeneracy": {
            "selected_axis_count": len(stable_axes),
            "selected_axis_fraction": selected_axis_fraction,
            "max_selected_axis_fraction": max_selected_axis_fraction,
        },
        "candidate_masks": {mask_name: stable_axes},
        "probe_reports": probe_reports,
        "control_reports": control_reports,
        "claim_scope": (
            "Local SGP grouped-axis audit only; not neural training, not scale evidence, "
            "not a remote launch gate, and not an ensemble result."
        ),
    }


def invp_global_stats_audit_from_config(
    config: dict[str, Any],
    *,
    samples_per_class: int | None = None,
    top_k: int = 12,
) -> dict[str, Any]:
    resolved = _sgp_config(config, samples_per_class=samples_per_class, top_k=top_k)
    reports = []
    for source in resolved["feature_sources"]:
        probes = []
        for seed in resolved["seeds"]:
            for key_split in resolved["key_splits"]:
                probes.append(
                    _sgp_source_matrix(
                        source,
                        resolved,
                        seed=seed,
                        key_split=key_split,
                    )
                )
        reports.append(
            invp_global_stats_report_from_feature_matrices(
                probes,
                pairs_per_sample=int(source.get("pairs_per_sample", resolved["pairs_per_sample"])),
                pair_bits=int(probes[0]["features"].shape[1])
                // int(source.get("pairs_per_sample", resolved["pairs_per_sample"])),
                top_k=resolved["top_k"],
                min_composite_auc=resolved["min_composite_auc"],
                min_topk_jaccard=resolved["min_topk_jaccard"],
                min_best_stat_auc=float(config.get("min_best_stat_auc", 0.55)),
                source_name=str(source["name"]),
            )
        )
    best_report = max(
        reports,
        key=lambda item: (
            item["decision"] == "invp_global_stats_candidate",
            item["summary"]["probe_composite_auc_min"],
            item["summary"]["best_stat_auc_min"],
        ),
    )
    return {
        "status": "pass",
        "audit": "invp_global_stats_audit",
        "decision": best_report["decision"],
        "best_source": best_report["source_name"],
        "config": {
            "rounds": resolved["rounds"],
            "samples_per_class": resolved["samples_per_class"],
            "seeds": resolved["seeds"],
            "key_splits": resolved["key_splits"],
            "pairs_per_sample": resolved["pairs_per_sample"],
            "negative_mode": resolved["negative_mode"],
            "sample_structure": resolved["sample_structure"],
            "difference_profile": resolved["difference_profile"],
            "difference_member": resolved["difference_member"],
            "top_k": resolved["top_k"],
            "min_composite_auc": resolved["min_composite_auc"],
            "min_topk_jaccard": resolved["min_topk_jaccard"],
            "min_best_stat_auc": float(config.get("min_best_stat_auc", 0.55)),
        },
        "source_reports": reports,
        "candidate_feature_names": best_report["candidate_feature_names"],
        "claim_scope": (
            "Local InvP global-statistics audit only; not neural training, not "
            "scale evidence, not a remote launch gate, and not an ensemble result."
        ),
    }


def invp_global_stats_report_from_feature_matrices(
    probes: list[dict[str, Any]],
    *,
    pairs_per_sample: int,
    pair_bits: int = 128,
    top_k: int = 16,
    min_composite_auc: float = 0.62,
    min_topk_jaccard: float = 0.35,
    min_best_stat_auc: float = 0.55,
    source_name: str = "inline",
) -> dict[str, Any]:
    if len(probes) < 2:
        raise ValueError("InvP global statistics audit requires at least two probes")
    if pair_bits % 64 != 0:
        raise ValueError("InvP global statistics require pair_bits to be a multiple of 64")
    words_per_pair = pair_bits // 64
    cells_per_word = 16
    stat_names = _present_global_stat_names(
        words_per_pair=words_per_pair,
        cells_per_word=cells_per_word,
        pairs_per_sample=pairs_per_sample,
    )
    stat_dim = present_global_stats_feature_bits(words_per_pair, cells_per_word, pairs_per_sample)
    if len(stat_names) != stat_dim:
        raise ValueError("InvP global statistic names do not match expected feature width")

    probe_reports = []
    top_sets: list[set[str]] = []
    stat_advantages = []
    stat_matrices = []
    for probe in probes:
        features, labels = _probe_features_and_labels(probe)
        if features.shape[1] != pairs_per_sample * pair_bits:
            raise ValueError("InvP global statistics probe width does not match pairs_per_sample * pair_bits")
        stat_matrix = _present_global_stats_matrix(
            features,
            pairs_per_sample=pairs_per_sample,
            words_per_pair=words_per_pair,
            cells_per_word=cells_per_word,
        )
        if stat_matrix.shape[1] != stat_dim:
            raise ValueError("InvP global statistics matrix width changed unexpectedly")
        stat_matrices.append(stat_matrix)
        stat_scores = _feature_axis_scores(stat_matrix, labels)
        top_indices = _top_indices(stat_scores["auc_advantage"], top_k)
        top_names = [stat_names[index] for index in top_indices]
        top_sets.append(set(top_names))
        stat_advantages.append(stat_scores["auc_advantage"])
        best_auc = max(float(max(auc, 1.0 - auc)) for auc in stat_scores["auc"])
        probe_reports.append(
            {
                "name": str(probe.get("name", f"probe{len(probe_reports)}")),
                "samples_per_class": int(min((labels == 0).sum(), (labels == 1).sum())),
                "input_bits": int(features.shape[1]),
                "stat_feature_dim": stat_dim,
                "best_stat_auc": best_auc,
                "top_statistics": [
                    {
                        "name": stat_names[index],
                        "index": int(index),
                        "auc": float(stat_scores["auc"][index]),
                        "auc_advantage": float(stat_scores["auc_advantage"][index]),
                        "positive_mean": float(stat_scores["positive_mean"][index]),
                        "negative_mean": float(stat_scores["negative_mean"][index]),
                    }
                    for index in top_indices
                ],
            }
        )

    stable_indices = _stable_axes_from_probe_scores(
        top_sets=[{stat_names.index(name) for name in top_set} for top_set in top_sets],
        axis_advantages=stat_advantages,
        stable_top_k=top_k,
        feature_dim=stat_dim,
    )
    candidate_feature_names = [stat_names[index] for index in stable_indices]
    probe_composites = []
    for probe, stat_matrix, report in zip(probes, stat_matrices, probe_reports, strict=True):
        _, labels = _probe_features_and_labels(probe)
        composite_scores = _oriented_zscore_composite(stat_matrix, labels, stable_indices)
        composite = _scalar_statistic_report(composite_scores, labels.astype(np.float32, copy=False))
        report["stable_stat_composite"] = composite
        probe_composites.append(composite)

    topk_jaccard_min = _group_jaccard_min(top_sets)
    probe_composite_auc_min = min(float(item["auc"]) for item in probe_composites) if probe_composites else 0.5
    best_stat_auc_min = min(float(item["best_stat_auc"]) for item in probe_reports) if probe_reports else 0.5
    decision = (
        "invp_global_stats_candidate"
        if probe_composite_auc_min >= min_composite_auc
        and topk_jaccard_min >= min_topk_jaccard
        and best_stat_auc_min >= min_best_stat_auc
        else "invp_global_stats_hold"
    )
    return {
        "audit": "invp_global_stats_audit",
        "source_name": source_name,
        "decision": decision,
        "summary": {
            "pairs_per_sample": pairs_per_sample,
            "pair_bits": pair_bits,
            "stat_feature_dim": stat_dim,
            "probe_count": len(probe_reports),
            "probe_composite_auc_min": probe_composite_auc_min,
            "probe_composite_auc_mean": float(np.mean([item["auc"] for item in probe_composites])),
            "best_stat_auc_min": best_stat_auc_min,
        },
        "stability": {
            "top_k": top_k,
            "topk_jaccard_min": topk_jaccard_min,
            "stable_feature_names": candidate_feature_names,
        },
        "gates": {
            "min_composite_auc": min_composite_auc,
            "min_topk_jaccard": min_topk_jaccard,
            "min_best_stat_auc": min_best_stat_auc,
        },
        "candidate_feature_names": candidate_feature_names,
        "probe_reports": probe_reports,
        "claim_scope": (
            "Local InvP global-statistics audit only; not neural training, not scale evidence, "
            "not a remote launch gate, and not an ensemble result."
        ),
    }


def invp_group_distribution_audit_from_config(
    config: dict[str, Any],
    *,
    samples_per_class: int | None = None,
    top_k: int = 12,
) -> dict[str, Any]:
    resolved = _sgp_config(config, samples_per_class=samples_per_class, top_k=top_k)
    group_scheme_names = [str(scheme) for scheme in config.get("group_schemes", ["cell", "word_bit_role"])]
    reports = []
    for source in resolved["feature_sources"]:
        probes = []
        for seed in resolved["seeds"]:
            for key_split in resolved["key_splits"]:
                probes.append(
                    _sgp_source_matrix(
                        source,
                        resolved,
                        seed=seed,
                        key_split=key_split,
                    )
                )
        pairs_per_sample = int(source.get("pairs_per_sample", resolved["pairs_per_sample"]))
        source_with_pairs = {**source, "pairs_per_sample": pairs_per_sample}
        group_schemes = {
            group_scheme: _axis_groups_for_sgp_source(
                source_with_pairs,
                feature_dim=int(probes[0]["features"].shape[1]),
                group_scheme=group_scheme,
            )
            for group_scheme in group_scheme_names
        }
        reports.append(
            invp_group_distribution_report_from_feature_matrices(
                probes,
                group_schemes=group_schemes,
                top_k=resolved["top_k"],
                min_composite_auc=resolved["min_composite_auc"],
                min_topk_jaccard=resolved["min_topk_jaccard"],
                min_best_stat_auc=float(config.get("min_best_stat_auc", 0.55)),
                source_name=str(source["name"]),
            )
        )
    best_report = max(
        reports,
        key=lambda item: (
            item["decision"] == "invp_group_distribution_candidate",
            item["summary"]["probe_composite_auc_min"],
            item["summary"]["best_stat_auc_min"],
        ),
    )
    return {
        "status": "pass",
        "audit": "invp_group_distribution_audit",
        "decision": best_report["decision"],
        "best_source": best_report["source_name"],
        "config": {
            "rounds": resolved["rounds"],
            "samples_per_class": resolved["samples_per_class"],
            "seeds": resolved["seeds"],
            "key_splits": resolved["key_splits"],
            "pairs_per_sample": resolved["pairs_per_sample"],
            "negative_mode": resolved["negative_mode"],
            "sample_structure": resolved["sample_structure"],
            "difference_profile": resolved["difference_profile"],
            "difference_member": resolved["difference_member"],
            "top_k": resolved["top_k"],
            "min_composite_auc": resolved["min_composite_auc"],
            "min_topk_jaccard": resolved["min_topk_jaccard"],
            "min_best_stat_auc": float(config.get("min_best_stat_auc", 0.55)),
            "group_schemes": group_scheme_names,
        },
        "source_reports": reports,
        "candidate_feature_names": best_report["candidate_feature_names"],
        "claim_scope": (
            "Local InvP group-distribution audit only; not neural training, not "
            "scale evidence, not a remote launch gate, and not an ensemble result."
        ),
    }


def invp_group_distribution_report_from_feature_matrices(
    probes: list[dict[str, Any]],
    *,
    group_schemes: dict[str, list[str]],
    top_k: int = 12,
    min_composite_auc: float = 0.62,
    min_topk_jaccard: float = 0.35,
    min_best_stat_auc: float = 0.55,
    source_name: str = "inline",
) -> dict[str, Any]:
    if len(probes) < 2:
        raise ValueError("InvP group distribution audit requires at least two probes")
    feature_dim = int(np.asarray(probes[0]["features"]).shape[1])
    for scheme, axis_groups in group_schemes.items():
        if len(axis_groups) != feature_dim:
            raise ValueError(f"group scheme {scheme!r} does not match feature width")

    probe_reports = []
    top_sets: list[set[str]] = []
    stat_matrices = []
    stat_names: list[str] | None = None
    stat_advantages = []
    for probe in probes:
        features, labels = _probe_features_and_labels(probe)
        if features.shape[1] != feature_dim:
            raise ValueError("all InvP group distribution probes must share feature width")
        named_scores = _group_distribution_named_scores(features, group_schemes)
        current_names = list(named_scores)
        if stat_names is None:
            stat_names = current_names
        elif stat_names != current_names:
            raise ValueError("group distribution statistics changed order across probes")
        stat_matrix = np.stack([named_scores[name] for name in current_names], axis=1)
        stat_matrices.append(stat_matrix)
        stat_scores = _feature_axis_scores(stat_matrix, labels)
        top_indices = _top_indices(stat_scores["auc_advantage"], top_k)
        top_names = [current_names[index] for index in top_indices]
        top_sets.append(set(top_names))
        stat_advantages.append(stat_scores["auc_advantage"])
        best_auc = max(float(max(auc, 1.0 - auc)) for auc in stat_scores["auc"])
        probe_reports.append(
            {
                "name": str(probe.get("name", f"probe{len(probe_reports)}")),
                "samples_per_class": int(min((labels == 0).sum(), (labels == 1).sum())),
                "input_bits": feature_dim,
                "stat_feature_dim": len(current_names),
                "best_stat_auc": best_auc,
                "top_statistics": [
                    {
                        "name": current_names[index],
                        "index": int(index),
                        "auc": float(stat_scores["auc"][index]),
                        "auc_advantage": float(stat_scores["auc_advantage"][index]),
                        "positive_mean": float(stat_scores["positive_mean"][index]),
                        "negative_mean": float(stat_scores["negative_mean"][index]),
                    }
                    for index in top_indices
                ],
            }
        )

    if stat_names is None:
        raise ValueError("no group distribution statistics were generated")
    stable_indices = _stable_axes_from_probe_scores(
        top_sets=[{stat_names.index(name) for name in top_set} for top_set in top_sets],
        axis_advantages=stat_advantages,
        stable_top_k=top_k,
        feature_dim=len(stat_names),
    )
    candidate_feature_names = [stat_names[index] for index in stable_indices]
    probe_composites = []
    for probe, stat_matrix, report in zip(probes, stat_matrices, probe_reports, strict=True):
        _, labels = _probe_features_and_labels(probe)
        composite_scores = _oriented_zscore_composite(stat_matrix, labels, stable_indices)
        composite = _scalar_statistic_report(composite_scores, labels.astype(np.float32, copy=False))
        report["stable_stat_composite"] = composite
        probe_composites.append(composite)

    topk_jaccard_min = _group_jaccard_min(top_sets)
    probe_composite_auc_min = min(float(item["auc"]) for item in probe_composites) if probe_composites else 0.5
    best_stat_auc_min = min(float(item["best_stat_auc"]) for item in probe_reports) if probe_reports else 0.5
    decision = (
        "invp_group_distribution_candidate"
        if probe_composite_auc_min >= min_composite_auc
        and topk_jaccard_min >= min_topk_jaccard
        and best_stat_auc_min >= min_best_stat_auc
        else "invp_group_distribution_hold"
    )
    return {
        "audit": "invp_group_distribution_audit",
        "source_name": source_name,
        "decision": decision,
        "summary": {
            "input_bits": feature_dim,
            "stat_feature_dim": len(stat_names),
            "probe_count": len(probe_reports),
            "probe_composite_auc_min": probe_composite_auc_min,
            "probe_composite_auc_mean": float(np.mean([item["auc"] for item in probe_composites])),
            "best_stat_auc_min": best_stat_auc_min,
        },
        "stability": {
            "top_k": top_k,
            "topk_jaccard_min": topk_jaccard_min,
            "stable_feature_names": candidate_feature_names,
        },
        "gates": {
            "min_composite_auc": min_composite_auc,
            "min_topk_jaccard": min_topk_jaccard,
            "min_best_stat_auc": min_best_stat_auc,
        },
        "candidate_feature_names": candidate_feature_names,
        "probe_reports": probe_reports,
        "claim_scope": (
            "Local InvP group-distribution audit only; not neural training, not scale evidence, "
            "not a remote launch gate, and not an ensemble result."
        ),
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


def trail_position_attribution_from_task(
    task: dict[str, Any],
    *,
    samples_per_class: int,
    seed: int | None = None,
    key_split: str = "validation",
    top_k: int = 16,
) -> dict[str, Any]:
    if key_split not in {"train", "validation"}:
        raise ValueError("key_split must be train or validation")

    matrix = _trail_position_stat_matrix_from_task(
        task,
        samples_per_class=samples_per_class,
        seed=seed,
        key_split=key_split,
    )
    stat_matrix = matrix["stat_matrix"]
    labels = matrix["labels"]
    stat_names = matrix["stat_names"]
    axis_scores = _feature_axis_scores(stat_matrix, labels)
    top_indices = _top_indices(axis_scores["auc_advantage"], top_k)
    top_rows = _top_feature_rows(axis_scores, top_indices)
    composite_scores = _oriented_zscore_composite(stat_matrix, labels, top_indices)
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
    best_name = stat_names[int(best_axis["index"])]
    return {
        "status": "pass",
        "audit": "present_trail_position_stats_attribution",
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
        "trail_depth": matrix["trail_depth"],
        "trail_words_per_depth": matrix["trail_words_per_depth"],
        "position_stat_dim": int(stat_matrix.shape[1]),
        "best_statistic": {
            **best_axis,
            "name": best_name,
        },
        "top_statistics": {
            "feature_names": [stat_names[index] for index in top_indices],
            "rows": top_rows,
        },
        "composite": {
            **composite,
            "axis_indices": top_indices,
            "feature_names": [stat_names[index] for index in top_indices],
            "combiner": "top_position_stat_oriented_zscore_mean",
        },
        "claim_scope": (
            "Local trail-position statistic attribution only; not neural training, "
            "not scale evidence, and not a remote launch gate."
        ),
    }


def trail_position_split_baseline_from_task(
    task: dict[str, Any],
    *,
    samples_per_class: int,
    seed: int | None = None,
    top_k: int = 16,
) -> dict[str, Any]:
    reference = _trail_position_stat_matrix_from_task(
        task,
        samples_per_class=samples_per_class,
        seed=seed,
        key_split="train",
    )
    evaluation = _trail_position_stat_matrix_from_task(
        task,
        samples_per_class=samples_per_class,
        seed=seed,
        key_split="validation",
    )
    if reference["stat_names"] != evaluation["stat_names"]:
        raise ValueError("train and validation trail-position statistic names differ")

    reference_scores = _feature_axis_scores(reference["stat_matrix"], reference["labels"])
    selected_indices = _top_indices(reference_scores["auc_advantage"], top_k)
    fit = _fit_oriented_zscore_composite(
        reference["stat_matrix"],
        reference["labels"],
        selected_indices,
    )
    reference_composite_scores = _apply_oriented_zscore_composite(
        reference["stat_matrix"],
        selected_indices,
        fit,
    )
    evaluation_composite_scores = _apply_oriented_zscore_composite(
        evaluation["stat_matrix"],
        selected_indices,
        fit,
    )
    selected_names = [reference["stat_names"][index] for index in selected_indices]
    combiner = "train_selected_position_stat_oriented_zscore_mean"
    return {
        "status": "pass",
        "audit": "present_trail_position_split_baseline",
        "cipher_key": task["cipher_key"],
        "rounds": task["rounds"],
        "samples_per_class": samples_per_class,
        "seed": task["seed"] if seed is None else seed,
        "sample_structure": task["sample_structure"],
        "negative_mode": task["negative_mode"],
        "feature_encoding": task["feature_encoding"],
        "pairs_per_sample": task["pairs_per_sample"],
        "input_difference": task["input_difference"],
        "trail_depth": reference["trail_depth"],
        "trail_words_per_depth": reference["trail_words_per_depth"],
        "position_stat_dim": int(reference["stat_matrix"].shape[1]),
        "selected_statistics": {
            "indices": selected_indices,
            "feature_names": selected_names,
            "fit_key_split": "train",
            "orientation": fit["orientation"],
            "mean": fit["mean"],
            "std": fit["std"],
        },
        "reference": _trail_position_split_report(
            reference,
            axis_scores=reference_scores,
            selected_indices=selected_indices,
            composite_scores=reference_composite_scores,
            combiner=combiner,
        ),
        "evaluation": _trail_position_split_report(
            evaluation,
            axis_scores=_feature_axis_scores(evaluation["stat_matrix"], evaluation["labels"]),
            selected_indices=selected_indices,
            composite_scores=evaluation_composite_scores,
            combiner=combiner,
        ),
        "claim_scope": (
            "Local train-selected deterministic position-statistics baseline only; "
            "not neural training, not scale evidence, and not a remote launch gate."
        ),
    }


def _trail_position_stat_matrix_from_task(
    task: dict[str, Any],
    *,
    samples_per_class: int,
    seed: int | None,
    key_split: str,
) -> dict[str, Any]:
    params = parse_parameterized_present_sboxddt_encoding(str(task["feature_encoding"]))
    if params is None or not bool(params["use_statistics"]):
        raise ValueError("trail-position attribution expects a parameterized PRESENT beamstats feature")
    if key_split not in {"train", "validation"}:
        raise ValueError("key_split must be train or validation")

    trail_depth = int(task.get("model_options", {}).get("trail_depth", params["depth"]))
    trail_words_per_depth = int(task.get("model_options", {}).get("trail_words_per_depth", 9))
    key = task["validation_key"] if key_split == "validation" else task["train_key"]
    cipher = build_cipher(task["cipher_key"], task["rounds"], key=key)
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=task["input_difference"],
            samples_per_class=samples_per_class,
            seed=task["seed"] if seed is None else seed,
            shuffle=False,
            feature_encoding=task["feature_encoding"],
            pairs_per_sample=task["pairs_per_sample"],
            negative_mode=task["negative_mode"],
            key_rotation_interval=task["key_rotation_interval"],
            sample_structure=task["sample_structure"],
            integral_active_nibble=task["integral_active_nibble"],
            selected_bit_indices=task["selected_bit_indices"],
        )
    )
    pair_bits = pair_bits_for_encoding(cipher.block_bits, task["feature_encoding"])
    model = PresentTrailPositionStatsPairSetDistinguisher(
        input_bits=int(dataset.features.shape[1]),
        pair_bits=pair_bits,
        base_channels=8,
        trail_depth=trail_depth,
        trail_words_per_depth=trail_words_per_depth,
        stats_hidden_bits=64,
    )
    with torch.no_grad():
        stat_matrix = (
            model._position_statistics(torch.from_numpy(dataset.features.astype(np.float32, copy=False)))
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
    stat_names = _trail_position_stat_names(
        words_per_pair=pair_bits // cipher.block_bits,
        cells_per_word=cipher.block_bits // 4,
        trail_depth=trail_depth,
        trail_words_per_depth=trail_words_per_depth,
    )
    if len(stat_names) != stat_matrix.shape[1]:
        raise ValueError("trail-position statistic names do not match matrix width")
    return {
        "key_split": key_split,
        "stat_matrix": stat_matrix,
        "labels": dataset.labels.astype(np.uint8, copy=False),
        "stat_names": stat_names,
        "trail_depth": trail_depth,
        "trail_words_per_depth": trail_words_per_depth,
    }


def _trail_position_split_report(
    matrix: dict[str, Any],
    *,
    axis_scores: dict[str, np.ndarray],
    selected_indices: list[int],
    composite_scores: np.ndarray,
    combiner: str,
) -> dict[str, Any]:
    top_rows = _top_feature_rows(axis_scores, selected_indices)
    best_axis = top_rows[0] if top_rows else {
        "index": 0,
        "positive_mean": 0.0,
        "negative_mean": 0.0,
        "mean_delta": 0.0,
        "cohen_d": 0.0,
        "auc": 0.5,
        "auc_advantage": 0.0,
    }
    return {
        "key_split": matrix["key_split"],
        "best_statistic": {
            **best_axis,
            "name": matrix["stat_names"][int(best_axis["index"])],
        },
        "top_statistics": {
            "feature_names": [matrix["stat_names"][index] for index in selected_indices],
            "rows": top_rows,
        },
        "composite": {
            **_scalar_statistic_report(composite_scores, matrix["labels"].astype(np.float32, copy=False)),
            "axis_indices": selected_indices,
            "feature_names": [matrix["stat_names"][index] for index in selected_indices],
            "combiner": combiner,
            "fit_key_split": "train",
        },
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


def _trail_position_stat_names(
    *,
    words_per_pair: int,
    cells_per_word: int,
    trail_depth: int,
    trail_words_per_depth: int,
) -> list[str]:
    prefix_words = words_per_pair - trail_depth * trail_words_per_depth
    if prefix_words <= 0:
        raise ValueError("trail-position names require prefix words before trail words")
    names: list[str] = []

    def add_word_cell(prefix: str, *, depth: int | None = None, trail_word: int | None = None) -> None:
        for word in range(words_per_pair if depth is None else 1):
            for cell in range(cells_per_word):
                if depth is None:
                    names.append(f"{prefix}_word{word}_cell{cell}")
                else:
                    names.append(f"{prefix}_depth{depth}_trailword{trail_word}_cell{cell}")

    add_word_cell("word_cell_mean")
    add_word_cell("word_cell_std")
    for family in ("word_mean", "word_std", "word_first_last", "word_span"):
        names.extend(f"{family}_word{word}" for word in range(words_per_pair))
    for family in ("cell_mean", "cell_std", "cell_first_last", "cell_span"):
        names.extend(f"{family}_cell{cell}" for cell in range(cells_per_word))
    for family in ("depth_word_cell_mean", "depth_word_cell_std", "depth_word_cell_span"):
        for depth in range(trail_depth):
            for trail_word in range(trail_words_per_depth):
                add_word_cell(family, depth=depth, trail_word=trail_word)
    for family in ("depth_cell_mean", "depth_cell_std", "depth_cell_first_last", "depth_cell_span"):
        for depth in range(trail_depth):
            names.extend(f"{family}_depth{depth}_cell{cell}" for cell in range(cells_per_word))
    for family in ("depth_word_mean", "depth_word_std", "depth_word_first_last", "depth_word_span"):
        for depth in range(trail_depth):
            names.extend(f"{family}_depth{depth}_trailword{word}" for word in range(trail_words_per_depth))
    for family in ("prefix_mean", "prefix_std"):
        for word in range(prefix_words):
            names.extend(f"{family}_word{word}_cell{cell}" for cell in range(cells_per_word))
    names.extend(
        [
            "global_pair_density_mean",
            "global_pair_density_std",
            "global_pair_density_first_last",
            "global_pair_density_span",
            "global_trail_density_mean",
            "global_trail_density_std",
            "global_trail_density_first_last",
            "global_trail_density_span",
            "global_low_cells_mean",
            "global_mid_cells_mean",
            "global_high_cells_mean",
            "global_even_odd_mean",
            "global_low_cells_first_last",
            "global_mid_cells_first_last",
            "global_high_cells_first_last",
            "global_even_odd_first_last",
        ]
    )
    return names


def _sgp_config(
    config: dict[str, Any],
    *,
    samples_per_class: int | None,
    top_k: int,
) -> dict[str, Any]:
    feature_sources = list(
        config.get(
            "feature_sources",
            [
                {
                    "name": "invp_delta",
                    "kind": "differential_feature",
                    "feature_encoding": "ciphertext_xor_spn_paligned_bits",
                }
            ],
        )
    )
    return {
        "cipher": str(config.get("cipher", "present80")),
        "rounds": int(config.get("rounds", 8)),
        "seeds": [int(seed) for seed in config.get("seeds", [0, 1])],
        "key_splits": [str(split) for split in config.get("key_splits", ["validation"])],
        "samples_per_class": int(config.get("samples_per_class", 2048) if samples_per_class is None else samples_per_class),
        "pairs_per_sample": int(config.get("pairs_per_sample", 16)),
        "negative_mode": str(config.get("negative_mode", "encrypted_random_plaintexts")),
        "sample_structure": str(config.get("sample_structure", OFFICIAL_ZHANG_WANG_CASE2_MCND)),
        "difference_profile": str(config.get("difference_profile", "present_zhang_wang2022_mcnd")),
        "difference_member": int(config.get("difference_member", 0)),
        "train_key": _parse_int_like(config.get("train_key", 0)),
        "validation_key": _parse_int_like(config.get("validation_key", 0x11111111111111111111)),
        "key_rotation_interval": int(config.get("key_rotation_interval", 0)),
        "top_k": int(top_k),
        "stable_top_k": int(config.get("stable_top_k", top_k)),
        "min_composite_auc": float(config.get("min_composite_auc", 0.55)),
        "min_topk_jaccard": float(config.get("min_topk_jaccard", 0.35)),
        "min_control_delta": float(config.get("min_control_delta", 0.01)),
        "max_selected_axis_fraction": float(config.get("max_selected_axis_fraction", 0.75)),
        "feature_sources": feature_sources,
        "control_sources": list(config.get("control_sources", [])),
    }


def _sgp_source_matrix(
    source: dict[str, Any],
    config: dict[str, Any],
    *,
    seed: int,
    key_split: str,
) -> dict[str, Any]:
    kind = str(source.get("kind", "differential_feature"))
    key = config["validation_key"] if key_split == "validation" else config["train_key"]
    input_difference = difference_for_profile(config["difference_profile"], config["difference_member"])
    name = f"{source.get('name', kind)}:seed{seed}:{key_split}"
    if kind == "candidate_evidence":
        features, labels = make_candidate_dataset(
            rounds=config["rounds"],
            key=key,
            input_difference=input_difference,
            seed=seed,
            samples_per_class=config["samples_per_class"],
            pairs_per_sample=int(source.get("pairs_per_sample", config["pairs_per_sample"])),
            negative_mode=config["negative_mode"],
            sample_structure=config["sample_structure"],
            key_rotation_interval=config["key_rotation_interval"],
            beam_width=int(source.get("beam_width", 4)),
            depth=int(source.get("depth", 3)),
            feature_mode=str(source.get("feature_mode", "aggregate")),
            feature_cache_root=None,
            split=key_split,
        )
        return {"name": name, "features": np.asarray(features, dtype=np.float32), "labels": labels}
    if kind != "differential_feature":
        raise ValueError(f"unsupported SGP source kind: {kind}")
    feature_encoding = str(source.get("feature_encoding", "ciphertext_xor_spn_paligned_bits"))
    cipher = build_cipher(config["cipher"], config["rounds"], key=key)
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=input_difference,
            samples_per_class=config["samples_per_class"],
            seed=seed,
            shuffle=True,
            feature_encoding=feature_encoding,
            pairs_per_sample=int(source.get("pairs_per_sample", config["pairs_per_sample"])),
            negative_mode=config["negative_mode"],
            key_rotation_interval=config["key_rotation_interval"],
            sample_structure=config["sample_structure"],
            selected_bit_indices=tuple(source.get("selected_bit_indices") or ()),
        )
    )
    return {
        "name": name,
        "features": dataset.features.astype(np.float32, copy=False),
        "labels": dataset.labels.astype(np.uint8, copy=False),
    }


def _probe_features_and_labels(probe: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(probe["features"], dtype=np.float64)
    labels = np.asarray(probe["labels"], dtype=np.uint8)
    if features.ndim != 2:
        raise ValueError("SGP probe features must be a 2D matrix")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("SGP probe labels must match feature rows")
    return features, labels


def _stable_axes_from_probe_scores(
    *,
    top_sets: list[set[int]],
    axis_advantages: list[np.ndarray],
    stable_top_k: int,
    feature_dim: int,
) -> list[int]:
    counts = np.zeros(feature_dim, dtype=np.int64)
    mean_advantage = np.zeros(feature_dim, dtype=np.float64)
    for top_set, advantages in zip(top_sets, axis_advantages, strict=True):
        for index in top_set:
            counts[index] += 1
        mean_advantage += advantages
    mean_advantage /= max(1, len(axis_advantages))
    order = sorted(
        range(feature_dim),
        key=lambda index: (int(counts[index]), float(mean_advantage[index]), -index),
        reverse=True,
    )
    return [int(index) for index in order[: min(stable_top_k, feature_dim)]]


def _topk_jaccard_min(top_sets: list[set[int]]) -> float:
    if len(top_sets) < 2:
        return 1.0
    values = []
    for left_index, left in enumerate(top_sets):
        for right in top_sets[left_index + 1 :]:
            union = left | right
            values.append(float(len(left & right) / len(union)) if union else 1.0)
    return min(values) if values else 1.0


def _present_global_stats_matrix(
    features: np.ndarray,
    *,
    pairs_per_sample: int,
    words_per_pair: int,
    cells_per_word: int,
) -> np.ndarray:
    tensor = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        stats = present_global_pairset_statistics(
            tensor,
            pairs_per_sample=pairs_per_sample,
            words_per_pair=words_per_pair,
            cells_per_word=cells_per_word,
            nibble_bits=4,
        )
    return stats.detach().cpu().numpy().astype(np.float64, copy=False)


def _present_global_stat_names(
    *,
    words_per_pair: int,
    cells_per_word: int,
    pairs_per_sample: int,
) -> list[str]:
    even_cell_count = cells_per_word // 2
    global_names = [
        "global_cell_activity_mean",
        "global_cell_activity_std",
        "global_word_activity_mean",
        "global_word_activity_std",
        "global_pair_activity_mean",
        "global_pair_activity_std",
        "global_pair_last_first_delta",
        "global_word_last_first_delta_mean",
        "global_cell_last_first_delta_mean",
        "global_even_odd_word_delta",
    ]
    names = (
        [f"word{index}_mean" for index in range(words_per_pair)]
        + [f"word{index}_std" for index in range(words_per_pair)]
        + [f"word{index}_last_first_delta" for index in range(words_per_pair)]
        + [f"word{index}_span" for index in range(words_per_pair)]
        + [f"cell{index}_mean" for index in range(cells_per_word)]
        + [f"cell{index}_std" for index in range(cells_per_word)]
        + [f"cell{index}_last_first_delta" for index in range(cells_per_word)]
        + [f"even_cell{index}_minus_odd_cell{index}" for index in range(even_cell_count)]
        + [f"pair{index}_activity" for index in range(pairs_per_sample)]
        + [f"pair{index}_centered_activity" for index in range(pairs_per_sample)]
        + [f"pair{index}_minus_first_activity" for index in range(pairs_per_sample)]
        + [f"last_minus_pair{index}_activity" for index in range(pairs_per_sample)]
        + global_names
        + [f"abs_{name}" for name in global_names]
    )
    expected = present_global_stats_feature_bits(words_per_pair, cells_per_word, pairs_per_sample)
    if len(names) != expected:
        raise ValueError(f"expected {expected} global statistic names, got {len(names)}")
    return names


def _group_distribution_named_scores(
    features: np.ndarray,
    group_schemes: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    named_scores: dict[str, np.ndarray] = {}
    for scheme, axis_groups in group_schemes.items():
        group_to_axes = _group_to_axes(axis_groups)
        group_ids = sorted(group_to_axes)
        group_activity = np.stack(
            [features[:, group_to_axes[group_id]].mean(axis=1) for group_id in group_ids],
            axis=1,
        )
        sorted_activity = np.sort(group_activity, axis=1)
        top2_count = min(2, group_activity.shape[1])
        top4_count = min(4, group_activity.shape[1])
        named_scores[f"{scheme}:activity_mean"] = group_activity.mean(axis=1)
        named_scores[f"{scheme}:activity_std"] = group_activity.std(axis=1)
        named_scores[f"{scheme}:activity_max"] = group_activity.max(axis=1)
        named_scores[f"{scheme}:top2_activity_mean"] = sorted_activity[:, -top2_count:].mean(axis=1)
        named_scores[f"{scheme}:top4_activity_mean"] = sorted_activity[:, -top4_count:].mean(axis=1)
        named_scores[f"{scheme}:bottom2_activity_mean"] = sorted_activity[:, :top2_count].mean(axis=1)
        named_scores[f"{scheme}:bottom4_activity_mean"] = sorted_activity[:, :top4_count].mean(axis=1)
        named_scores[f"{scheme}:activity_span"] = group_activity.max(axis=1) - group_activity.min(axis=1)
    return named_scores


def _axis_groups_for_sgp_source(
    source: dict[str, Any],
    *,
    feature_dim: int,
    group_scheme: str,
) -> list[str]:
    explicit = source.get("axis_groups")
    if explicit is not None:
        groups = [str(group) for group in explicit]
        if len(groups) != feature_dim:
            raise ValueError("explicit axis_groups must match feature width")
        return groups
    feature_encoding = str(source.get("feature_encoding", "ciphertext_xor_spn_paligned_bits"))
    base_dim = 128
    pairs_per_sample = int(source.get("pairs_per_sample", 1))
    if (
        feature_encoding != "ciphertext_xor_spn_paligned_bits"
        or feature_dim != base_dim * pairs_per_sample
        or feature_dim % base_dim != 0
    ):
        raise ValueError("automatic SGP grouped axes currently support ciphertext_xor_spn_paligned_bits repeated per pair")
    groups = []
    for axis_index in range(feature_dim):
        pair_index = axis_index // base_dim
        local_axis_index = axis_index % base_dim
        word_index = local_axis_index // 64
        word_name = "delta" if word_index == 0 else "invp_delta"
        offset = local_axis_index % 64
        lsb_bit_index = 63 - offset
        cell_index = lsb_bit_index // 4
        bit_role = lsb_bit_index % 4
        if group_scheme == "pair_word_cell":
            groups.append(f"pair{pair_index}:{word_name}:cell{cell_index}")
        elif group_scheme == "word_cell":
            groups.append(f"{word_name}:cell{cell_index}")
        elif group_scheme == "cell":
            groups.append(f"cell{cell_index}")
        elif group_scheme == "word_bit_role":
            groups.append(f"{word_name}:bit{bit_role}")
        elif group_scheme == "p_layer_orbit":
            groups.append(f"{word_name}:orbit{_present_p_layer_orbit_id(lsb_bit_index)}")
        else:
            raise ValueError(f"unsupported SGP group scheme: {group_scheme}")
    return groups


def _present_p_layer_orbit_id(lsb_bit_index: int) -> int:
    values = []
    current = int(lsb_bit_index)
    for _ in range(4):
        values.append(current)
        current = _present_p_layer_lsb_index(current)
        if current == lsb_bit_index:
            break
    return min(values)


def _present_p_layer_lsb_index(lsb_bit_index: int) -> int:
    if lsb_bit_index == 63:
        return 63
    return (16 * lsb_bit_index) % 63


def _group_to_axes(axis_groups: list[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for axis_index, group in enumerate(axis_groups):
        grouped.setdefault(str(group), []).append(axis_index)
    return grouped


def _group_axis_advantages(
    axis_advantages: np.ndarray,
    group_to_axes: dict[str, list[int]],
    group_ids: list[str],
) -> np.ndarray:
    return np.array(
        [float(np.max(axis_advantages[group_to_axes[group_id]])) for group_id in group_ids],
        dtype=np.float64,
    )


def _top_group_ids(group_scores: np.ndarray, group_ids: list[str], top_k: int) -> list[str]:
    order = sorted(
        range(len(group_ids)),
        key=lambda index: (float(group_scores[index]), group_ids[index]),
        reverse=True,
    )
    return [group_ids[index] for index in order[: min(top_k, len(group_ids))]]


def _stable_groups_from_probe_scores(
    *,
    top_sets: list[set[str]],
    group_advantages: list[np.ndarray],
    group_ids: list[str],
    stable_top_k: int,
) -> list[str]:
    counts = {group_id: 0 for group_id in group_ids}
    mean_advantage = {group_id: 0.0 for group_id in group_ids}
    for top_set, advantages in zip(top_sets, group_advantages, strict=True):
        for group_id in top_set:
            counts[group_id] += 1
        for group_id, advantage in zip(group_ids, advantages, strict=True):
            mean_advantage[group_id] += float(advantage)
    divisor = max(1, len(group_advantages))
    order = sorted(
        group_ids,
        key=lambda group_id: (counts[group_id], mean_advantage[group_id] / divisor, group_id),
        reverse=True,
    )
    return order[: min(stable_top_k, len(order))]


def _axes_for_groups(stable_groups: list[str], group_to_axes: dict[str, list[int]]) -> list[int]:
    axes = []
    for group in stable_groups:
        axes.extend(group_to_axes[group])
    return sorted(int(axis) for axis in axes)


def _group_jaccard_min(top_sets: list[set[str]]) -> float:
    if len(top_sets) < 2:
        return 1.0
    values = []
    for left_index, left in enumerate(top_sets):
        for right in top_sets[left_index + 1 :]:
            union = left | right
            values.append(float(len(left & right) / len(union)) if union else 1.0)
    return min(values) if values else 1.0


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


def _fit_oriented_zscore_composite(
    features: np.ndarray,
    labels: np.ndarray,
    axis_indices: list[int],
) -> dict[str, list[float]]:
    orientation = []
    means = []
    stds = []
    for index in axis_indices:
        scores = features[:, index].astype(np.float64, copy=False)
        sign = 1.0 if binary_auc(labels, scores) >= 0.5 else -1.0
        oriented = sign * scores
        std = float(oriented.std())
        orientation.append(sign)
        means.append(float(oriented.mean()))
        stds.append(std)
    return {"orientation": orientation, "mean": means, "std": stds}


def _apply_oriented_zscore_composite(
    features: np.ndarray,
    axis_indices: list[int],
    fit: dict[str, list[float]],
) -> np.ndarray:
    if not axis_indices:
        return np.zeros(features.shape[0], dtype=np.float64)
    columns = []
    for offset, index in enumerate(axis_indices):
        scores = features[:, index].astype(np.float64, copy=False)
        oriented = float(fit["orientation"][offset]) * scores
        std = float(fit["std"][offset])
        if std <= 0.0:
            columns.append(np.zeros_like(oriented, dtype=np.float64))
        else:
            columns.append((oriented - float(fit["mean"][offset])) / std)
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
