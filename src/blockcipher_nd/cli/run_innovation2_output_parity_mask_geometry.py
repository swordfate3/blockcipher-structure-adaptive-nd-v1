from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.tasks.innovation2.output_parity_mask_geometry import (
    ALIGNED_MASKS,
    CONTIGUOUS_MASKS,
    RUN_ID,
    adjudicate_mask_geometry,
    adjudicate_two_key_confirmation,
    build_mask_geometry_data,
    mask_positions,
    train_mask_geometry_matrix,
    validate_mask_geometry_contract,
)
from blockcipher_nd.tasks.innovation2.output_parity_prediction import (
    OutputParityPredictionConfig,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Innovation 2 OP2 ciphertext parity mask-geometry calibration."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--rounds", default=1, type=int)
    parser.add_argument("--experiment")
    parser.add_argument("--anchor-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = OutputParityPredictionConfig(
        run_id=args.run_id, seed=args.seed, rounds=args.rounds
    )
    experiment = args.experiment or (
        "op3_independent_key_confirmation"
        if args.anchor_root is not None
        else "op2_mask_geometry"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {
            "experiment": experiment,
            "training": True,
            "sample_classification": False,
            "rounds": config.rounds,
            "seed": config.seed,
        },
    )
    datasets = build_mask_geometry_data(config)
    protocol_checks = validate_mask_geometry_contract(config, datasets)
    _write_progress(progress, "data_ready", protocol_checks)
    training = train_mask_geometry_matrix(config, datasets)
    single_key_gate = adjudicate_mask_geometry(config, protocol_checks, training)
    anchor_gate: dict[str, Any] | None = None
    independence_checks: dict[str, bool] | None = None
    if args.anchor_root is not None:
        anchor_gate, independence_checks = _load_and_validate_anchor(
            args.anchor_root, config, datasets
        )
        gate = adjudicate_two_key_confirmation(
            config.run_id,
            anchor_gate,
            single_key_gate,
            independence_checks,
            rounds=config.rounds,
        )
    else:
        gate = single_key_gate
    for row in training["rows"]:
        row["experiment"] = experiment
        row["status"] = gate["status"]
        row["decision"] = gate["decision"]
    contiguous = datasets["contiguous"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_output_parity_prediction",
        "experiment": experiment,
        "config": serializable_config(config),
        "cipher": "PRESENT-80",
        "secret_key_hex": f"{int(contiguous['secret_key']):020x}",
        "key_protocol": "one fixed unknown key; disjoint plaintext splits",
        "only_changed_variable": (
            "PRESENT round count"
            if config.rounds > 1
            else "independent fixed secret key and disjoint plaintexts"
            if args.anchor_root is not None
            else "ciphertext output parity mask geometry"
        ),
        "anchor_root": str(args.anchor_root) if args.anchor_root is not None else None,
        "sample_classification": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    dataset_summary = {
        "train_rows": config.train_rows,
        "validation_rows": config.validation_rows,
        "test_rows": config.test_rows,
        "input_shape": [64],
        "full_output_shape": [64],
        "parity_output_shape": [16],
        "contiguous_masks": [f"{mask:016x}" for mask in CONTIGUOUS_MASKS],
        "aligned_masks": [f"{mask:016x}" for mask in ALIGNED_MASKS],
        "contiguous_test_prevalence": contiguous["test"]
        .parity_targets.mean(axis=0)
        .tolist(),
        "aligned_test_prevalence": datasets["aligned"]["test"]
        .parity_targets.mean(axis=0)
        .tolist(),
        "protocol_checks": protocol_checks,
        "independence_checks": independence_checks,
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "dataset_summary": dataset_summary,
        "protocol_checks": protocol_checks,
        "trained_rows": training["rows"],
        "derived_parity_metrics": training["derived_parity_metrics"],
        "single_key_gate": single_key_gate,
        "anchor_gate": anchor_gate,
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", training["rows"])
    _write_csv(args.output_root / "history.csv", training["history"])
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "dataset_summary.json", dataset_summary)
    _write_masks(args.output_root / "masks.csv")
    _write_arrays(args.output_root, datasets)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True)
    )
    return 1 if gate["status"] == "fail" else 0


def _load_and_validate_anchor(
    anchor_root: Path,
    config: OutputParityPredictionConfig,
    datasets: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    anchor_gate = json.loads((anchor_root / "gate.json").read_text(encoding="utf-8"))
    anchor_metadata = json.loads(
        (anchor_root / "metadata.json").read_text(encoding="utf-8")
    )
    anchor_plaintexts = np.concatenate(
        [
            np.load(anchor_root / "data" / split_name / "plaintexts.npy")
            for split_name in ("train", "validation", "test")
        ]
    )
    current_plaintexts = np.concatenate(
        [
            datasets["contiguous"][split_name].plaintexts
            for split_name in ("train", "validation", "test")
        ]
    )
    anchor_seed = int(anchor_metadata["config"]["seed"])
    anchor_key = int(str(anchor_metadata["secret_key_hex"]), 16)
    current_key = int(datasets["contiguous"]["secret_key"])
    checks = {
        "anchor_gate_and_metadata_run_ids_match": anchor_gate.get("run_id")
        == anchor_metadata.get("run_id"),
        "anchor_seed_is_zero": anchor_seed == 0,
        "current_seed_is_one": config.seed == 1,
        "anchor_and_current_rounds_match": int(anchor_metadata["config"]["rounds"])
        == config.rounds,
        "independent_secret_keys_differ": anchor_key != current_key,
        "cross_run_plaintexts_are_disjoint": np.intersect1d(
            anchor_plaintexts, current_plaintexts
        ).size
        == 0,
    }
    return anchor_gate, checks


def _write_arrays(output_root: Path, datasets: dict[str, dict[str, Any]]) -> None:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    contiguous = datasets["contiguous"]
    aligned = datasets["aligned"]
    for split_name in ("train", "validation", "test"):
        split_root = data_root / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        np.save(split_root / "plaintexts.npy", contiguous[split_name].plaintexts)
        np.save(split_root / "features.npy", contiguous[split_name].features)
        np.save(split_root / "full_targets.npy", contiguous[split_name].full_targets)
        np.save(
            split_root / "contiguous_parity_targets.npy",
            contiguous[split_name].parity_targets,
        )
        np.save(
            split_root / "aligned_parity_targets.npy",
            aligned[split_name].parity_targets,
        )


def _write_masks(path: Path) -> None:
    rows = []
    for geometry, masks in (
        ("contiguous_output_nibble", CONTIGUOUS_MASKS),
        ("last_round_sbox_p_layer_aligned", ALIGNED_MASKS),
    ):
        for index, mask in enumerate(masks):
            rows.append(
                {
                    "geometry": geometry,
                    "mask_index": index,
                    "mask_hex": f"{mask:016x}",
                    "output_bits_lsb_first": ",".join(
                        str(bit) for bit in mask_positions(mask)
                    ),
                    "weight": mask.bit_count(),
                }
            )
    _write_csv(path, rows)


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
