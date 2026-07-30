from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from blockcipher_nd.cli.plot_uknit_ctspn_r6_window_depth_k1bo import (
    render_k1bo_svg,
)
from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_window_depth_k1bo import (
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    RUN_ID,
    adjudicate_k1bo,
    evaluate_k1bo,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_uknit_r6_public_window_depth_k1bo_20260730.json"
)
EXPECTED_DIFFERENCE = 0x0000400000000000
SOURCE_VIEW = "exact_five_stage_position_histogram"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit two- versus three-transition public windows at uKNIT r6."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    require_fresh_output_root(args.output_root)
    args.output_root.mkdir(parents=True)
    write_progress(args.output_root, "k1bo_preflight_start")

    source_roots = {
        5: ROOT / str(config["source_r5_root"]),
        6: ROOT / str(config["source_r6_root"]),
    }
    source_gates = {
        rounds: read_json(root / "gate.json") for rounds, root in source_roots.items()
    }
    boundary_root = ROOT / str(config["source_boundary_root"])
    boundary_gate = read_json(boundary_root / "gate.json")
    source_rows = {
        rounds: selected_source_rows(root / "dataset_manifest.jsonl", rounds=rounds)
        for rounds, root in source_roots.items()
    }
    datasets, cache_manifest = load_source_datasets(source_rows)
    source_auc = load_source_auc(source_roots)

    descriptor_path = ROOT / str(config["runtime_structure_path"])
    r5_exact_two = load_runtime_spn_descriptor(
        descriptor_path,
        rounds=2,
        round_start=3,
    ).structure
    r6_exact_two = load_runtime_spn_descriptor(
        descriptor_path,
        rounds=2,
        round_start=4,
    ).structure
    r6_exact_three = load_runtime_spn_descriptor(
        descriptor_path,
        rounds=3,
        round_start=3,
    ).structure
    r6_wrong_three = r6_exact_three.shuffled_sbox_assignments(20260728)

    source_checks = build_source_checks(
        config=config,
        source_gates=source_gates,
        boundary_gate=boundary_gate,
        boundary_root=boundary_root,
        source_rows=source_rows,
        datasets=datasets,
        cache_manifest=cache_manifest,
        descriptor_path=descriptor_path,
        r6_exact_two=r6_exact_two,
        r6_exact_three=r6_exact_three,
        r6_wrong_three=r6_wrong_three,
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
        "descriptor": str(descriptor_path),
        "descriptor_sha256": file_sha256(descriptor_path),
        "source_checks": source_checks,
        "failed_source_checks": sorted(
            name for name, passed in source_checks.items() if not passed
        ),
    }
    write_json(args.output_root / "preflight.json", preflight)
    write_jsonl(args.output_root / "source_cache_manifest.jsonl", cache_manifest)
    if not preflight["execution_authorized"]:
        raise ValueError(f"K1-BO preflight failed: {preflight['failed_source_checks']}")
    write_progress(args.output_root, "k1bo_preflight_passed")

    feature_rows, scorer_rows, result_rows, prefix_checks = evaluate_k1bo(
        datasets=datasets,
        r5_exact_two=r5_exact_two,
        r6_exact_two=r6_exact_two,
        r6_exact_three=r6_exact_three,
        r6_wrong_three=r6_wrong_three,
        batch_size=args.batch_size,
    )
    write_progress(
        args.output_root,
        "k1bo_feature_evaluation_done",
        feature_rows=len(feature_rows),
        scorer_rows=len(scorer_rows),
        result_rows=len(result_rows),
    )
    gate = adjudicate_k1bo(
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
        source_auc=source_auc,
        source_checks=source_checks,
        prefix_checks=prefix_checks,
        thresholds=config["thresholds"],
    )
    finalize(
        args.output_root,
        gate=gate,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def load_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    expected = {
        "run_id": RUN_ID,
        "cipher": "uknit64",
        "difference": "0x0000400000000000",
        "active_cell": 11,
        "active_bit_role": 1,
        "pairs_per_sample": 4,
        "seeds": [3, 4],
        "splits": list(EXPECTED_SPLITS),
        "train_samples_per_class": 2048,
        "holdout_samples_per_class": 1024,
        "negative_mode": "encrypted_random_plaintexts",
        "variance_floor": 1e-6,
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
        raise ValueError(f"K1-BO frozen config mismatch: {mismatches}")
    return payload


def selected_source_rows(path: Path, *, rounds: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    selected = [
        row
        for row in rows
        if row.get("phase") == "confirmation"
        and int(row.get("rounds", -1)) == rounds
        and int(row.get("cell", -1)) == 11
        and int(row.get("input_difference", -1)) == EXPECTED_DIFFERENCE
        and int(row.get("seed", -1)) in EXPECTED_SEEDS
        and str(row.get("split")) in EXPECTED_SPLITS
    ]
    selected.sort(key=lambda row: (int(row["seed"]), EXPECTED_SPLITS.index(row["split"])))
    return selected


def load_source_datasets(
    source_rows: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[
    dict[tuple[int, int, str], DiskDifferentialDataset],
    list[dict[str, Any]],
]:
    datasets: dict[tuple[int, int, str], DiskDifferentialDataset] = {}
    manifests: list[dict[str, Any]] = []
    for rounds in (5, 6):
        for row in source_rows[rounds]:
            cache_dir = ROOT / str(row["cache_dir"])
            dataset, digests = load_cache(cache_dir)
            key = (rounds, int(row["seed"]), str(row["split"]))
            if key in datasets:
                raise ValueError(f"duplicate K1-BO source cache: {key}")
            datasets[key] = dataset
            manifests.append(
                {
                    "run_id": RUN_ID,
                    "rounds": rounds,
                    "seed": key[1],
                    "split": key[2],
                    "cache_dir": str(cache_dir.relative_to(ROOT)),
                    "source_run_id": row.get("run_id"),
                    "source_dataset_sha256": row.get("dataset_sha256"),
                    "dataset_sha256": differential_dataset_sha256(dataset),
                    "rows": len(dataset.labels),
                    "key_hex": row.get("key_hex"),
                    "input_difference": row.get("input_difference"),
                    "pairs_per_sample": dataset.metadata.get("pairs_per_sample"),
                    "negative_mode": dataset.metadata.get("negative_mode"),
                    "cache_payloads_present": True,
                    **digests,
                }
            )
    return datasets, manifests


def load_cache(path: Path) -> tuple[DiskDifferentialDataset, dict[str, str]]:
    paths = {
        "features_sha256": path / "features.npy",
        "labels_sha256": path / "labels.npy",
        "metadata_sha256": path / "metadata.json",
    }
    if not all(item.is_file() for item in paths.values()):
        raise ValueError(f"incomplete K1-BO source cache: {path}")
    dataset = DiskDifferentialDataset(
        features=np.load(paths["features_sha256"], mmap_mode="r"),
        labels=np.load(paths["labels_sha256"], mmap_mode="r"),
        metadata=read_json(paths["metadata_sha256"]),
        cache_dir=path,
    )
    return dataset, {name: file_sha256(item) for name, item in paths.items()}


def load_source_auc(
    source_roots: Mapping[int, Path],
) -> dict[tuple[int, int, str], float]:
    mapped: dict[tuple[int, int, str], float] = {}
    for rounds, root in source_roots.items():
        for row in read_jsonl(root / "results.jsonl"):
            if row.get("phase") != "confirmation":
                continue
            if int(row.get("cell", -1)) != 11 or row.get("view") != SOURCE_VIEW:
                continue
            seed = int(row.get("seed", -1))
            split = str(row.get("split"))
            if seed in EXPECTED_SEEDS and split in EXPECTED_SPLITS:
                mapped[(rounds, seed, split)] = float(row["auc"])
    return mapped


def build_source_checks(
    *,
    config: Mapping[str, Any],
    source_gates: Mapping[int, Mapping[str, Any]],
    boundary_gate: Mapping[str, Any],
    boundary_root: Path,
    source_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    datasets: Mapping[tuple[int, int, str], DiskDifferentialDataset],
    cache_manifest: Sequence[Mapping[str, Any]],
    descriptor_path: Path,
    r6_exact_two: Any,
    r6_exact_three: Any,
    r6_wrong_three: Any,
) -> dict[str, bool]:
    expected_keys = {
        (rounds, seed, split)
        for rounds in (5, 6)
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
    }
    paired_keys_match = all(
        next(
            row["key_hex"]
            for row in cache_manifest
            if row["rounds"] == 5 and row["seed"] == seed and row["split"] == split
        )
        == next(
            row["key_hex"]
            for row in cache_manifest
            if row["rounds"] == 6 and row["seed"] == seed and row["split"] == split
        )
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
    )
    return {
        "r5_source_gate_exact": source_gates[5].get("status") == "pass"
        and source_gates[5].get("decision") == config["source_r5_decision"],
        "r6_source_gate_exact": source_gates[6].get("status") == "hold"
        and source_gates[6].get("decision") == config["source_r6_decision"],
        "boundary_source_gate_exact": boundary_gate.get("status") == "hold"
        and boundary_gate.get("decision") == config["source_boundary_decision"]
        and not boundary_gate.get("failed_protocol_checks")
        and boundary_root == ROOT / str(config["source_boundary_root"]),
        "twelve_source_manifest_rows_exact": all(
            len(source_rows[rounds]) == 6 for rounds in (5, 6)
        ),
        "twelve_source_datasets_exact": set(datasets) == expected_keys
        and len(cache_manifest) == 12,
        "source_dataset_hashes_exact": all(
            row.get("source_dataset_sha256") == row.get("dataset_sha256")
            for row in cache_manifest
        ),
        "source_cache_payloads_complete": all(
            row.get("cache_payloads_present") is True for row in cache_manifest
        ),
        "source_protocol_exact": all(
            int(row.get("input_difference", -1)) == EXPECTED_DIFFERENCE
            and int(row.get("pairs_per_sample", -1)) == 4
            and row.get("negative_mode") == "encrypted_random_plaintexts"
            and int(row.get("rows", -1))
            == (4096 if row.get("split") == "train_seen" else 2048)
            for row in cache_manifest
        ),
        "r5_r6_seed_split_keys_match": paired_keys_match,
        "descriptor_present": descriptor_path.is_file(),
        "window_geometry_exact": r6_exact_two.rounds == 2
        and r6_exact_three.rounds == 3
        and r6_wrong_three.rounds == 3,
        "wrong_sbox_changes_only_semantics": np.array_equal(
            r6_exact_three.linear_matrices.numpy(),
            r6_wrong_three.linear_matrices.numpy(),
        )
        and not np.array_equal(
            r6_exact_three.sbox_truth_bits.numpy(),
            r6_wrong_three.sbox_truth_bits.numpy(),
        ),
    }


def finalize(
    output_root: Path,
    *,
    gate: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    scorer_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_jsonl(output_root / "feature_manifest.jsonl", feature_rows)
    write_jsonl(output_root / "scorer_manifest.jsonl", scorer_rows)
    write_jsonl(output_root / "results.jsonl", result_rows)
    write_comparison_csv(output_root / "comparison.csv", gate)
    write_json(output_root / "gate.json", dict(gate))
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "source_cache_rows": 12,
        "feature_rows": len(feature_rows),
        "scorer_rows": len(scorer_rows),
        "result_rows": len(result_rows),
        "training_rows": 0,
        "neural_parameter_count": 0,
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
            "passed_routes": gate["passed_routes"],
            "neural_training_authorized": gate["neural_training_authorized"],
            "remote_scale": gate["remote_scale"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
            "feature_rows": len(feature_rows),
            "scorer_rows": len(scorer_rows),
            "result_rows": len(result_rows),
        },
    )
    plot_report = render_k1bo_svg(gate, output_root / "curves.svg")
    write_json(output_root / "plot_report.json", plot_report)
    write_progress(
        output_root,
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        passed_routes=gate["passed_routes"],
    )


def write_comparison_csv(path: Path, gate: Mapping[str, Any]) -> None:
    rows = []
    for route, seed_rows in gate["route_results"].items():
        for seed, split_rows in seed_rows.items():
            for split, values in split_rows.items():
                rows.append({"route": route, "seed": seed, "split": split, **values})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_fresh_output_root(path: Path) -> None:
    if path.exists():
        raise ValueError(f"K1-BO output root already exists: {path}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_progress(root: Path, event: str, **payload: Any) -> None:
    path = root / "progress.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"run_id": RUN_ID, "event": event, **payload}, sort_keys=True)
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
