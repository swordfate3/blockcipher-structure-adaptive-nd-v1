from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.ciphers.spn.uknit import UknitBc, uknit_round_keys
from blockcipher_nd.cli.plot_uknit_r6_key_hypothesis_k1bp import render_k1bp_svg
from blockcipher_nd.data.differential import DifferentialDataset, DiskDifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    build_k1t_control,
)
from blockcipher_nd.tasks.innovation1.uknit_r6_key_hypothesis_k1bp import (
    CONFIRMATION_SEEDS,
    EXPECTED_CANDIDATES,
    EXPECTED_CONE_SOURCE_BITS,
    EXPECTED_EFFECTIVE_KEY_BITS,
    EXPECTED_SPLITS,
    FRESH_SPLITS,
    RUN_ID,
    adjudicate_k1bp,
    dependency_cones,
    evaluate_sparse_anchor,
    inverse_sbox_table,
    masked_r5_cell_values_from_r6,
    rank_sparse_hypotheses,
    round_key_runtime_bits,
    strip_last_round,
    true_effective_hypothesis,
)
from blockcipher_nd.training.metrics import binary_auc, predict_binary_probabilities


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_r6_last_round_key_hypothesis_k1bp_20260730.json"
)
EXPECTED_DIFFERENCE = 0x0000400000000000
EXPECTED_CHECKPOINT_DIGESTS = {
    3: "ff43fb8a9787b60ae02dd79509d5702e0d1605455b795ca0aba7d9dcf017f750",
    4: "c2709f21784a1e580caa0ae058be1e8b4cf6278cebd42bb053004683ca663c81",
}
EXPECTED_K1U_GATE_DIGEST = (
    "79a5f3652b8a6125af8c987cb8b1df075fc8e992e73cdb5dc61dedbfbdb6c3ed"
)
EXPECTED_K1U_PLAN_DIGEST = (
    "1cc74466314eaa94f36acec0bb7a3ea29f69e0c28c7f4601256dae2dec4bff65"
)
EXPECTED_DESCRIPTOR_DIGEST = (
    "b74f9cc28b5fc28637b179f45ded67dec1a3d5dca04ca2eccb176ec790fbefd2"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit bounded last-round key hypotheses for uKNIT r6."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    require_fresh_output_root(args.output_root)
    args.output_root.mkdir(parents=True)
    write_progress(args.output_root, "k1bp_preflight_start")

    descriptor_path = ROOT / str(config["runtime_structure_path"])
    r5_transition = load_runtime_spn_descriptor(
        descriptor_path, rounds=1, round_start=4
    ).structure
    r6_suffix = load_runtime_spn_descriptor(
        descriptor_path, rounds=1, round_start=5
    ).structure
    wrong_r5_transition = r5_transition.shuffled_sbox_assignments(20260730)

    r5_root = ROOT / str(config["source_r5_root"])
    r6_root = ROOT / str(config["source_r6_root"])
    window_root = ROOT / str(config["source_window_root"])
    k1u_root = ROOT / str(config["source_k1u_root"])
    r5_rows = selected_source_rows(
        r5_root / "dataset_manifest.jsonl", rounds=5, include_discovery=True
    )
    r6_rows = selected_source_rows(
        r6_root / "dataset_manifest.jsonl", rounds=6, include_discovery=False
    )
    r5_datasets, r5_keys, r5_manifest = load_source_datasets(r5_rows, rounds=5)
    r6_datasets, r6_keys, r6_manifest = load_source_datasets(r6_rows, rounds=6)
    source_manifest = [*r5_manifest, *r6_manifest]

    k1u_plan = ROOT / str(config["k1u_plan"])
    models, checkpoint_checks = load_k1u_models(
        k1u_plan=k1u_plan,
        k1u_root=k1u_root,
        device=args.device,
    )
    cones = dependency_cones(r5_transition)
    inverse_fixture_exact = verify_inverse_fixture(r6_suffix)
    source_checks = build_source_checks(
        config=config,
        descriptor_path=descriptor_path,
        k1u_plan=k1u_plan,
        r5_root=r5_root,
        r6_root=r6_root,
        window_root=window_root,
        k1u_root=k1u_root,
        r5_rows=r5_rows,
        r6_rows=r6_rows,
        r5_datasets=r5_datasets,
        r6_datasets=r6_datasets,
        r5_keys=r5_keys,
        r6_keys=r6_keys,
        source_manifest=source_manifest,
        cones=cones,
        inverse_fixture_exact=inverse_fixture_exact,
        checkpoint_checks=checkpoint_checks,
    )
    preflight = {
        "run_id": RUN_ID,
        "status": "pass" if all(source_checks.values()) else "fail",
        "execution_authorized": all(source_checks.values()),
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
        "remote_scale_authorized": False,
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "source_checks": source_checks,
        "failed_source_checks": sorted(
            name for name, passed in source_checks.items() if not passed
        ),
    }
    write_json(args.output_root / "preflight.json", preflight)
    write_jsonl(args.output_root / "source_cache_manifest.jsonl", source_manifest)
    write_json(
        args.output_root / "dependency_cones.json",
        {
            "run_id": RUN_ID,
            "full_model_required_key_bits": 64,
            "full_model_candidate_count": 1 << 64,
            "sparse_cones": [cone.as_dict() for cone in cones],
        },
    )
    if not preflight["execution_authorized"]:
        raise ValueError(f"K1-BP preflight failed: {preflight['failed_source_checks']}")
    write_progress(args.output_root, "k1bp_preflight_passed")

    discovery_rows, selected_cell = run_discovery(
        datasets={split: r5_datasets[(2, split)] for split in EXPECTED_SPLITS},
        transition=r5_transition,
    )
    write_jsonl(args.output_root / "discovery_results.jsonl", discovery_rows)
    selection = {
        "run_id": RUN_ID,
        "selection_seed": 2,
        "selected_cell": selected_cell,
        "criterion": "maximum minimum fresh AUC; ties use lower cell index",
        "selected_row": next(
            row for row in discovery_rows if row["target_cell"] == selected_cell
        ),
        "confirmation_results_read_before_selection": False,
    }
    write_json(args.output_root / "selection.json", selection)
    write_progress(
        args.output_root, "k1bp_discovery_done", selected_cell=selected_cell
    )

    full_oracle_rows = run_full_oracle(
        models=models,
        datasets=r6_datasets,
        master_keys=r6_keys,
        suffix=r6_suffix,
        wrong_count=int(config["wrong_full_key_controls"]),
        batch_size=args.batch_size,
        device=args.device,
    )
    write_jsonl(args.output_root / "full_oracle_results.jsonl", full_oracle_rows)
    write_progress(args.output_root, "k1bp_full_oracle_done")

    sparse_rows = run_sparse_ranking(
        selected_cell=selected_cell,
        r5_datasets=r5_datasets,
        r6_datasets=r6_datasets,
        r6_keys=r6_keys,
        r5_transition=r5_transition,
        wrong_r5_transition=wrong_r5_transition,
        r6_suffix=r6_suffix,
        candidate_batch_size=args.candidate_batch_size,
    )
    write_jsonl(args.output_root / "sparse_rank_results.jsonl", sparse_rows)
    write_progress(args.output_root, "k1bp_sparse_ranking_done")

    gate = adjudicate_k1bp(
        protocol_checks=source_checks,
        discovery_rows=discovery_rows,
        full_oracle_rows=full_oracle_rows,
        sparse_rank_rows=sparse_rows,
        selected_cell=selected_cell,
        thresholds=config["thresholds"],
    )
    finalize(
        args.output_root,
        gate=gate,
        discovery_rows=discovery_rows,
        full_oracle_rows=full_oracle_rows,
        sparse_rows=sparse_rows,
        selection=selection,
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def load_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    expected = {
        "run_id": RUN_ID,
        "cipher": "uknit64",
        "rounds": 6,
        "difference": "0x0000400000000000",
        "active_cell": 11,
        "active_bit_role": 1,
        "pairs_per_sample": 4,
        "discovery_seed": 2,
        "confirmation_seeds": [3, 4],
        "splits": list(EXPECTED_SPLITS),
        "train_samples_per_class": 2048,
        "holdout_samples_per_class": 1024,
        "negative_mode": "encrypted_random_plaintexts",
        "variance_floor": 1e-6,
        "full_model_required_key_bits": 64,
        "full_model_candidate_count": 1 << 64,
        "sparse_source_key_bits": EXPECTED_CONE_SOURCE_BITS,
        "sparse_effective_key_bits": EXPECTED_EFFECTIVE_KEY_BITS,
        "sparse_candidate_count": EXPECTED_CANDIDATES,
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
        "remote_scale_authorized": False,
    }
    mismatches = {
        name: {"expected": value, "observed": payload.get(name)}
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatches:
        raise ValueError(f"K1-BP frozen config mismatch: {mismatches}")
    return payload


def selected_source_rows(
    path: Path, *, rounds: int, include_discovery: bool
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    allowed_seeds = {2, *CONFIRMATION_SEEDS} if include_discovery else set(CONFIRMATION_SEEDS)
    selected = [
        row
        for row in rows
        if int(row.get("rounds", -1)) == rounds
        and int(row.get("cell", -1)) == 11
        and int(row.get("input_difference", -1)) == EXPECTED_DIFFERENCE
        and int(row.get("seed", -1)) in allowed_seeds
        and str(row.get("split")) in EXPECTED_SPLITS
        and (
            (int(row.get("seed", -1)) == 2 and row.get("phase") == "discovery")
            or (int(row.get("seed", -1)) in CONFIRMATION_SEEDS and row.get("phase") == "confirmation")
        )
    ]
    selected.sort(key=lambda row: (int(row["seed"]), EXPECTED_SPLITS.index(row["split"])))
    return selected


def load_source_datasets(
    rows: Sequence[Mapping[str, Any]], *, rounds: int
) -> tuple[
    dict[tuple[int, str], DiskDifferentialDataset],
    dict[tuple[int, str], int],
    list[dict[str, Any]],
]:
    datasets: dict[tuple[int, str], DiskDifferentialDataset] = {}
    keys: dict[tuple[int, str], int] = {}
    manifests: list[dict[str, Any]] = []
    for row in rows:
        cache_dir = ROOT / str(row["cache_dir"])
        dataset, digests = load_cache(cache_dir)
        key = (int(row["seed"]), str(row["split"]))
        if key in datasets:
            raise ValueError(f"duplicate K1-BP r{rounds} source cache: {key}")
        datasets[key] = dataset
        keys[key] = int(str(row["key_hex"]), 16)
        manifests.append(
            {
                "run_id": RUN_ID,
                "source_run_id": row.get("run_id"),
                "rounds": rounds,
                "seed": key[0],
                "split": key[1],
                "cache_dir": str(cache_dir.relative_to(ROOT)),
                "rows": len(dataset.labels),
                "key_hex": row.get("key_hex"),
                "input_difference": row.get("input_difference"),
                "pairs_per_sample": dataset.metadata.get("pairs_per_sample"),
                "negative_mode": dataset.metadata.get("negative_mode"),
                "source_dataset_sha256": row.get("dataset_sha256"),
                "dataset_sha256": differential_dataset_sha256(dataset),
                **digests,
            }
        )
    return datasets, keys, manifests


def load_cache(path: Path) -> tuple[DiskDifferentialDataset, dict[str, str]]:
    paths = {
        "features_sha256": path / "features.npy",
        "labels_sha256": path / "labels.npy",
        "metadata_sha256": path / "metadata.json",
    }
    if not all(item.is_file() for item in paths.values()):
        raise ValueError(f"incomplete K1-BP source cache: {path}")
    dataset = DiskDifferentialDataset(
        features=np.load(paths["features_sha256"], mmap_mode="r"),
        labels=np.load(paths["labels_sha256"], mmap_mode="r"),
        metadata=read_json(paths["metadata_sha256"]),
        cache_dir=path,
    )
    return dataset, {name: file_sha256(item) for name, item in paths.items()}


def load_k1u_models(
    *, k1u_plan: Path, k1u_root: Path, device: str
) -> tuple[dict[int, torch.nn.Module], dict[str, bool]]:
    tasks = tasks_from_plan(
        k1u_plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    invariant = {
        int(task["seed"]): task
        for task in tasks
        if task.get("model_key") == "runtime_spn_ct_k1t_position_histogram_invariant"
    }
    models: dict[int, torch.nn.Module] = {}
    checks: dict[str, bool] = {"two_k1u_invariant_tasks_exact": set(invariant) == {3, 4}}
    for seed in CONFIRMATION_SEEDS:
        checkpoint_path = k1u_root / (
            f"checkpoints/row{3 if seed == 3 else 6:04d}_"
            f"runtime_spn_ct_k1t_position_histogram_invariant_seed{seed}.pt"
        )
        checks[f"seed{seed}_checkpoint_sha256_exact"] = (
            checkpoint_path.is_file()
            and file_sha256(checkpoint_path) == EXPECTED_CHECKPOINT_DIGESTS[seed]
        )
        model = build_k1t_control(
            task=invariant[seed],
            condition="invariant_histogram_residual",
            input_bits=512,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise ValueError(f"K1-BP seed{seed} checkpoint has no state_dict")
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        models[seed] = model
        checks[f"seed{seed}_checkpoint_strict_load"] = True
    return models, checks


def run_discovery(
    *, datasets: Mapping[str, DifferentialDataset], transition: Any
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    for cell in range(16):
        scorer, aucs, cone, _linear_lookup, _inverse_table = evaluate_sparse_anchor(
            datasets=datasets,
            transition=transition,
            target_cell=cell,
            seed=2,
        )
        rows.append(
            {
                "run_id": RUN_ID,
                "phase": "discovery",
                "seed": 2,
                "target_cell": cell,
                "source_key_bits": cone.source_key_bits,
                "effective_key_bits": cone.effective_key_bits,
                "candidate_count": cone.candidate_count,
                "source_key_equivalence_size": cone.source_key_equivalence_size,
                "train_auc": aucs["train_seen"],
                "same_key_fresh_auc": aucs["same_key_fresh"],
                "cross_key_validation_auc": aucs["cross_key_validation"],
                "minimum_fresh_auc": min(
                    aucs["same_key_fresh"], aucs["cross_key_validation"]
                ),
                "scorer_sha256": scorer.sha256,
                "training_performed": False,
                "optimizer_steps": 0,
                "epochs": 0,
            }
        )
    selected = max(
        rows, key=lambda row: (float(row["minimum_fresh_auc"]), -int(row["target_cell"]))
    )
    return rows, int(selected["target_cell"])


def run_full_oracle(
    *,
    models: Mapping[int, torch.nn.Module],
    datasets: Mapping[tuple[int, str], DifferentialDataset],
    master_keys: Mapping[tuple[int, str], int],
    suffix: Any,
    wrong_count: int,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in CONFIRMATION_SEEDS:
        for split in FRESH_SPLITS:
            dataset = datasets[(seed, split)]
            master_key = master_keys[(seed, split)]
            true_round_key = uknit_round_keys(master_key)[5]
            candidates = wrong_full_key_panel(
                true_round_key, count=wrong_count, seed=seed * 100 + FRESH_SPLITS.index(split)
            )
            aucs: dict[str, float] = {}
            positive_means: dict[str, float] = {}
            for name, round_key in (("correct", true_round_key), *candidates):
                transformed = strip_last_round(
                    dataset.features,
                    last_transition=suffix,
                    round_key_bits=round_key_int_bits(round_key),
                )
                transformed_dataset = DifferentialDataset(
                    features=transformed,
                    labels=np.asarray(dataset.labels, dtype=np.uint8),
                    metadata={**dict(dataset.metadata), "rounds": 5},
                )
                probabilities = predict_binary_probabilities(
                    models[seed],
                    transformed_dataset,
                    batch_size=batch_size,
                    device=device,
                )
                aucs[name] = binary_auc(transformed_dataset.labels, probabilities)
                positive_means[name] = float(
                    probabilities[np.asarray(dataset.labels) == 1].mean()
                )
            wrong_aucs = {name: value for name, value in aucs.items() if name != "correct"}
            rows.append(
                {
                    "run_id": RUN_ID,
                    "seed": seed,
                    "split": split,
                    "rows": len(dataset.labels),
                    "pairs_per_sample": 4,
                    "required_key_bits": 64,
                    "candidate_count": 1 << 64,
                    "evaluated_wrong_controls": len(wrong_aucs),
                    "correct_key_auc": aucs["correct"],
                    "best_wrong_key_auc": max(wrong_aucs.values()),
                    "correct_minus_best_wrong_auc": aucs["correct"]
                    - max(wrong_aucs.values()),
                    "correct_positive_mean": positive_means["correct"],
                    "wrong_control_aucs": wrong_aucs,
                    "training_performed": False,
                    "optimizer_steps": 0,
                    "epochs": 0,
                    "oracle_only": True,
                    "attack_claim_allowed": False,
                }
            )
    return rows


def run_sparse_ranking(
    *,
    selected_cell: int,
    r5_datasets: Mapping[tuple[int, str], DifferentialDataset],
    r6_datasets: Mapping[tuple[int, str], DifferentialDataset],
    r6_keys: Mapping[tuple[int, str], int],
    r5_transition: Any,
    wrong_r5_transition: Any,
    r6_suffix: Any,
    candidate_batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in CONFIRMATION_SEEDS:
        source = {split: r5_datasets[(seed, split)] for split in EXPECTED_SPLITS}
        scorer, r5_aucs, cone, linear_lookup, exact_inverse_table = evaluate_sparse_anchor(
            datasets=source,
            transition=r5_transition,
            target_cell=selected_cell,
            seed=seed,
        )
        (
            shuffled_scorer,
            _shuffled_aucs,
            _cone,
            _linear_lookup,
            _inverse_table,
        ) = evaluate_sparse_anchor(
            datasets=source,
            transition=r5_transition,
            target_cell=selected_cell,
            shuffle_labels=True,
            seed=seed,
        )
        wrong_inverse_table = inverse_sbox_table(wrong_r5_transition, selected_cell)
        for split in FRESH_SPLITS:
            dataset = r6_datasets[(seed, split)]
            left, right = masked_r5_cell_values_from_r6(
                dataset,
                last_transition=r6_suffix,
                cone=cone,
                linear_lookup=linear_lookup,
            )
            true_guess = true_effective_hypothesis(
                r6_keys[(seed, split)], cone, linear_lookup
            )
            exact = rank_sparse_hypotheses(
                left_values=left,
                right_values=right,
                labels=dataset.labels,
                inverse_table=exact_inverse_table,
                scorer=scorer,
                true_hypothesis=true_guess,
                candidate_batch_size=candidate_batch_size,
            )
            wrong = rank_sparse_hypotheses(
                left_values=left,
                right_values=right,
                labels=dataset.labels,
                inverse_table=wrong_inverse_table,
                scorer=scorer,
                true_hypothesis=true_guess,
                candidate_batch_size=candidate_batch_size,
            )
            shuffled = rank_sparse_hypotheses(
                left_values=left,
                right_values=right,
                labels=dataset.labels,
                inverse_table=exact_inverse_table,
                scorer=shuffled_scorer,
                true_hypothesis=true_guess,
                candidate_batch_size=candidate_batch_size,
            )
            rows.append(
                {
                    "run_id": RUN_ID,
                    "seed": seed,
                    "split": split,
                    "target_cell": selected_cell,
                    "source_bits": list(cone.source_bits),
                    "source_cells": list(cone.source_cells),
                    "source_key_bits": cone.source_key_bits,
                    "effective_key_bits": cone.effective_key_bits,
                    "source_key_equivalence_size": cone.source_key_equivalence_size,
                    "r5_sparse_auc": r5_aucs[split],
                    **exact.as_dict(),
                    "wrong_sbox_true_rank": wrong.true_rank,
                    "wrong_sbox_correct_auc": wrong.correct_auc,
                    "label_shuffle_true_rank": shuffled.true_rank,
                    "label_shuffle_correct_auc": shuffled.correct_auc,
                    "negative_mode": "encrypted_random_plaintexts",
                    "training_performed": False,
                    "optimizer_steps": 0,
                    "epochs": 0,
                }
            )
    return rows


def build_source_checks(
    *,
    config: Mapping[str, Any],
    descriptor_path: Path,
    k1u_plan: Path,
    r5_root: Path,
    r6_root: Path,
    window_root: Path,
    k1u_root: Path,
    r5_rows: Sequence[Mapping[str, Any]],
    r6_rows: Sequence[Mapping[str, Any]],
    r5_datasets: Mapping[tuple[int, str], DifferentialDataset],
    r6_datasets: Mapping[tuple[int, str], DifferentialDataset],
    r5_keys: Mapping[tuple[int, str], int],
    r6_keys: Mapping[tuple[int, str], int],
    source_manifest: Sequence[Mapping[str, Any]],
    cones: Sequence[Any],
    inverse_fixture_exact: bool,
    checkpoint_checks: Mapping[str, bool],
) -> dict[str, bool]:
    expected_r5 = {(seed, split) for seed in (2, 3, 4) for split in EXPECTED_SPLITS}
    expected_r6 = {(seed, split) for seed in CONFIRMATION_SEEDS for split in EXPECTED_SPLITS}
    r5_gate = read_json(r5_root / "gate.json")
    r6_gate = read_json(r6_root / "gate.json")
    window_gate = read_json(window_root / "gate.json")
    k1u_gate = read_json(k1u_root / "gate.json")
    return {
        "r5_source_gate_exact": r5_gate.get("status") == "pass"
        and r5_gate.get("decision") == config["source_r5_decision"],
        "r6_source_gate_exact": r6_gate.get("status") == "hold"
        and r6_gate.get("decision") == config["source_r6_decision"],
        "window_source_gate_exact": window_gate.get("status") == "hold"
        and window_gate.get("decision") == config["source_window_decision"]
        and not window_gate.get("failed_protocol_checks"),
        "k1u_source_gate_exact": k1u_gate.get("decision") == config["source_k1u_decision"]
        and not k1u_gate.get("failed_protocol_checks"),
        "k1u_gate_sha256_exact": file_sha256(k1u_root / "gate.json")
        == EXPECTED_K1U_GATE_DIGEST,
        "k1u_plan_sha256_exact": file_sha256(k1u_plan) == EXPECTED_K1U_PLAN_DIGEST,
        "descriptor_sha256_exact": file_sha256(descriptor_path)
        == EXPECTED_DESCRIPTOR_DIGEST,
        "nine_r5_source_rows_exact": len(r5_rows) == 9
        and set(r5_datasets) == expected_r5,
        "six_r6_source_rows_exact": len(r6_rows) == 6
        and set(r6_datasets) == expected_r6,
        "source_cache_hashes_exact": all(
            row.get("source_dataset_sha256") == row.get("dataset_sha256")
            for row in source_manifest
        ),
        "source_protocol_exact": all(
            int(row.get("input_difference", -1)) == EXPECTED_DIFFERENCE
            and int(row.get("pairs_per_sample", -1)) == 4
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and int(row.get("rows", -1))
            == (4096 if row.get("split") == "train_seen" and row.get("seed") in (3, 4) else 2048 if row.get("seed") in (3, 4) else 2048 if row.get("split") == "train_seen" else 1024)
            for row in source_manifest
        ),
        "r5_r6_confirmation_keys_match": all(
            r5_keys[(seed, split)] == r6_keys[(seed, split)]
            for seed in CONFIRMATION_SEEDS
            for split in EXPECTED_SPLITS
        ),
        "correct_full_key_inverse_fixture_exact": inverse_fixture_exact,
        "full_model_requires_64_key_bits": int(config["full_model_required_key_bits"])
        == 64
        and int(config["full_model_candidate_count"]) == 1 << 64,
        "sixteen_sparse_cones_exact": len(cones) == 16
        and all(
            cone.source_key_bits == EXPECTED_CONE_SOURCE_BITS
            and cone.effective_key_bits == EXPECTED_EFFECTIVE_KEY_BITS
            and cone.candidate_count == EXPECTED_CANDIDATES
            for cone in cones
        ),
        "zero_training_protocol": config.get("training_authorized") is False
        and int(config.get("optimizer_steps_authorized", -1)) == 0,
        **dict(checkpoint_checks),
    }


def verify_inverse_fixture(suffix: Any) -> bool:
    key = 0x0123456789ABCDEFFEDCBA9876543210
    plaintexts = list(range(16))
    r5 = [UknitBc(rounds=5, key=key).encrypt(value) for value in plaintexts]
    r6 = [UknitBc(rounds=6, key=key).encrypt(value) for value in plaintexts]
    observed = strip_last_round(
        blocks_to_features(r6),
        last_transition=suffix,
        round_key_bits=round_key_runtime_bits(key),
    )
    return np.array_equal(observed, blocks_to_features(r5))


def wrong_full_key_panel(
    true_key: int, *, count: int, seed: int
) -> tuple[tuple[str, int], ...]:
    if count <= 0:
        raise ValueError("K1-BP wrong full-key panel must be nonempty")
    mask = (1 << 64) - 1
    rotated = ((true_key << 1) | (true_key >> 63)) & mask
    rng = np.random.default_rng(20260730 + seed)
    weight = true_key.bit_count()
    values: list[tuple[str, int]] = [("zero", 0), ("bit_rotated", rotated)]
    observed = {true_key, 0, rotated}
    while len(values) < count + 2:
        positions = rng.choice(64, size=weight, replace=False)
        value = sum(1 << int(position) for position in positions)
        if value in observed:
            continue
        observed.add(value)
        values.append((f"same_weight_{len(values) - 1:02d}", value))
    return tuple(values)


def round_key_int_bits(value: int) -> torch.Tensor:
    return torch.tensor([(value >> bit) & 1 for bit in range(64)], dtype=torch.float32)


def finalize(
    output_root: Path,
    *,
    gate: Mapping[str, Any],
    discovery_rows: Sequence[Mapping[str, Any]],
    full_oracle_rows: Sequence[Mapping[str, Any]],
    sparse_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> None:
    write_comparison_csv(
        output_root / "comparison.csv",
        full_oracle_rows=full_oracle_rows,
        sparse_rows=sparse_rows,
    )
    write_json(output_root / "gate.json", dict(gate))
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "discovery_rows": len(discovery_rows),
        "full_oracle_rows": len(full_oracle_rows),
        "sparse_rank_rows": len(sparse_rows),
        "training_rows": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    write_json(output_root / "validation.json", validation)
    write_json(
        output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "status": gate["status"],
            "decision": gate["decision"],
            "selected_cell": selection["selected_cell"],
            "full_model_required_key_bits": 64,
            "full_model_candidate_count": 1 << 64,
            "sparse_source_key_bits": EXPECTED_CONE_SOURCE_BITS,
            "sparse_effective_key_bits": EXPECTED_EFFECTIVE_KEY_BITS,
            "sparse_candidate_count": EXPECTED_CANDIDATES,
            "bounded_route_pass": gate["bounded_route_pass"],
            "evidence_tier": gate["evidence_tier"],
            "weak_signal_observed": gate["weak_signal_observed"],
            "weak_signal_confirmed": gate["weak_signal_confirmed"],
            "discovery_signal_tier": gate["discovery_signal_tier"],
            "discovery_minimum_fresh_auc": gate["discovery_minimum_fresh_auc"],
            "confirmation_signal_tier": gate["confirmation_signal_tier"],
            "confirmation_minimum_r5_auc": gate["confirmation_minimum_r5_auc"],
            "confirmation_minimum_r6_auc": gate["confirmation_minimum_r6_auc"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    plot_report = render_k1bp_svg(
        discovery_rows=discovery_rows,
        full_oracle_rows=full_oracle_rows,
        sparse_rows=sparse_rows,
        gate=gate,
        output=output_root / "curves.svg",
    )
    write_json(output_root / "plot_report.json", plot_report)
    write_progress(
        output_root,
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        selected_cell=selection["selected_cell"],
    )


def write_comparison_csv(
    path: Path,
    *,
    full_oracle_rows: Sequence[Mapping[str, Any]],
    sparse_rows: Sequence[Mapping[str, Any]],
) -> None:
    rows = [
        {
            "phase": "full_oracle",
            "seed": row["seed"],
            "split": row["split"],
            "correct_auc": row["correct_key_auc"],
            "control_auc": row["best_wrong_key_auc"],
            "margin": row["correct_minus_best_wrong_auc"],
            "true_rank": "",
            "wrong_sbox_rank": "",
            "label_shuffle_rank": "",
        }
        for row in full_oracle_rows
    ]
    rows.extend(
        {
            "phase": "sparse_12bit_rank",
            "seed": row["seed"],
            "split": row["split"],
            "correct_auc": row["correct_auc"],
            "control_auc": row["label_shuffle_correct_auc"],
            "margin": row["correct_minus_best_wrong"],
            "true_rank": row["true_rank"],
            "wrong_sbox_rank": row["wrong_sbox_true_rank"],
            "label_shuffle_rank": row["label_shuffle_true_rank"],
        }
        for row in sparse_rows
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def blocks_to_features(blocks: Sequence[int]) -> np.ndarray:
    if len(blocks) % 8:
        raise ValueError("K1-BP fixture needs complete four-pair samples")
    bits = np.asarray(
        [[(value >> bit) & 1 for bit in range(63, -1, -1)] for value in blocks],
        dtype=np.uint8,
    )
    return bits.reshape(-1, 512)


def require_fresh_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"K1-BP output root is not empty: {path}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_progress(output_root: Path, event: str, **payload: Any) -> None:
    path = output_root / "progress.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"run_id": RUN_ID, "event": event, **payload}, ensure_ascii=False, sort_keys=True)
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
