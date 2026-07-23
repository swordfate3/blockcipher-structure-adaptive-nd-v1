from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.speck32_output_prediction_data import (
    Speck32OutputPredictionDataConfig,
    prepare_speck32_output_prediction_data,
    serializable_config,
    validate_speck32_output_prediction_data,
)
from blockcipher_nd.tasks.innovation2.speck32_output_prediction_models import (
    jeong_anchor_protocols,
)
from blockcipher_nd.tasks.innovation2.speck32_output_prediction_training import (
    Speck32OutputPredictionTrainingConfig,
    adjudicate_speck32_arx1_a,
    serializable_training_config,
    train_speck32_arx1_a1,
    train_speck32_arx1_a2,
)
from blockcipher_nd.tasks.innovation2.speck32_rotation_carry_model import (
    rotation_carry_protocols,
)


READINESS_RUN_ID = "i2_output_prediction_arx1_speck32_r3_readiness_seed21_20260723"
FORMAL_RUN_ID = "i2_output_prediction_arx1a_speck32_r3_key21_20260723"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Innovation 2 ARX1 SPECK32/64 r3 full true-output A1/A2 matrix."
        )
    )
    parser.add_argument("--mode", choices=("readiness", "arx1_a"), default="readiness")
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "arx1_a":
        run_id = args.run_id or FORMAL_RUN_ID
        data_config = Speck32OutputPredictionDataConfig.arx1_a(run_id=run_id)
        training_config = Speck32OutputPredictionTrainingConfig.arx1_a(
            run_id=run_id,
            device=args.device or "cuda",
        )
    else:
        run_id = args.run_id or READINESS_RUN_ID
        data_config = Speck32OutputPredictionDataConfig(run_id=run_id)
        training_config = Speck32OutputPredictionTrainingConfig(
            run_id=run_id,
            device=args.device or "cpu",
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    progress(
        "run_start",
        {
            "run_id": run_id,
            "mode": args.mode,
            "training": True,
            "target": "full_32_bit_true_speck32_ciphertext_output",
            "sample_classification": False,
        },
    )
    source = prepare_speck32_output_prediction_data(
        data_config,
        args.output_root,
        progress=progress,
    )
    data_checks = validate_speck32_output_prediction_data(data_config, source)
    progress("source_data_validated", data_checks)
    if not all(data_checks.values()):
        raise ValueError(f"invalid ARX1 SPECK32 source data: {data_checks}")
    a1 = train_speck32_arx1_a1(
        training_config,
        source,
        args.output_root,
        progress=progress,
    )
    a2 = train_speck32_arx1_a2(
        training_config,
        source,
        args.output_root,
        progress=progress,
    )
    gate = adjudicate_speck32_arx1_a(training_config, data_checks, a1, a2)
    result_rows = _result_rows(a2, gate)
    history = [*a1["history"], *a2["new_history"]]
    protocols = {
        "runtime_models": {
            **jeong_anchor_protocols(),
            **rotation_carry_protocols(training_config.candidate_channels),
        },
        "formal_rotation_carry_reference": rotation_carry_protocols(400),
        "readiness_uses_reduced_candidate_channels": args.mode == "readiness",
    }
    metadata = {
        "run_id": run_id,
        "task": "innovation2_output_prediction",
        "experiment": "arx1_speck32_r3_full32_output",
        "mode": args.mode,
        "cipher": "SPECK32/64",
        "rounds": data_config.rounds,
        "data_config": serializable_config(data_config),
        "training_config": serializable_training_config(training_config),
        "key_protocol": (
            "one fixed unknown 64-bit key; disjoint train/test plaintexts; "
            "one independently trained model per key"
        ),
        "input": "32 MSB-first plaintext bits preserving x_msw/y_lsw words",
        "target": "32 MSB-first true SPECK32/64 ciphertext bits",
        "sample_classification": False,
        "source_manifest_sha256": a2["source_manifest"]["manifest_sha256"],
        "a1_bundle_sha256": a2["a1_bundle_sha256"],
        "a2_bundle_sha256": a2["bundle_sha256"],
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": run_id,
        "metadata": metadata,
        "data_checks": data_checks,
        "training_rows": a2["full_matrix_rows"],
        "checkpoint_manifest": a2["checkpoints"],
        "a2_fairness_checks": a2["fairness_checks"],
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    _write_csv(args.output_root / "history.csv", history)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "checkpoint_manifest.json", a2["checkpoints"])
    _write_json(args.output_root / "model_protocols.json", protocols)
    progress("run_done", {"status": gate["status"], "decision": gate["decision"]})
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)},
            sort_keys=True,
        )
    )
    return 1 if gate["status"] == "fail" else 0


def _result_rows(a2: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    groups = (
        ("training_summary", a2["full_matrix_rows"]),
        ("per_bit_metric", a2["full_matrix_per_bit_rows"]),
    )
    return [
        {
            **row,
            "row_type": row_type,
            "status": gate["status"],
            "decision": gate["decision"],
        }
        for row_type, rows in groups
        for row in rows
    ]


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
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
