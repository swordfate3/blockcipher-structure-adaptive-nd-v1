from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1o import render_k1o_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from blockcipher_nd.cli.run_uknit_family_ctspn_k1n import read_tasks
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    load_bound_datasets,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import (
    build_k1n_control,
    candidate_task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    EXPECTED_FEATURE_ROWS,
    EXPECTED_RESULT_ROWS,
    EXPECTED_SCORER_ROWS,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    RUN_ID,
    adjudicate_k1o,
    evaluate_k1o,
    validate_k1o_source,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact uKNIT r5 partial-state position histograms with a "
            "closed-form diagonal Fisher/LDA scorer and no neural training."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-O run_id must remain frozen as {RUN_ID}")
    if args.batch_size != 256:
        raise ValueError("K1-O feature batch size is frozen at 256")
    require_fresh_output_root(args.output_root)

    source_gate = read_json(args.source_root / "gate.json")
    source_validation = read_json(args.source_root / "validation.json")
    source_preflight = read_json(args.source_root / "preflight.json")
    dataset_manifest = read_jsonl(args.source_root / "dataset_manifest.jsonl")
    source_checks = validate_k1o_source(
        source_root=args.source_root,
        source_gate=source_gate,
        source_validation=source_validation,
        source_preflight=source_preflight,
        dataset_manifest=dataset_manifest,
        plan_path=args.plan,
    )
    if not all(source_checks.values()):
        print(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "status": "invalid",
                    "decision": "innovation1_uknit_family_ctspn_k1o_source_invalid",
                    "source_checks": source_checks,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 4

    args.output_root.mkdir(parents=True)
    write_json(
        args.output_root / "preflight.json",
        {
            "run_id": RUN_ID,
            "status": "pass",
            "execution_authorized": True,
            "training_authorized": False,
            "optimizer_steps_authorized": 0,
            "plan": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "source_root": str(args.source_root),
            "source_artifact_sha256": {
                name: file_sha256(args.source_root / name)
                for name in (
                    "gate.json",
                    "dataset_manifest.jsonl",
                    "validation.json",
                    "preflight.json",
                )
            },
            "source_checks": source_checks,
            "feature_batch_size": args.batch_size,
            "device": args.device,
            "training_rows": 0,
            "neural_parameter_count": 0,
            "optimizer_steps": 0,
            "epochs": 0,
        },
    )
    progress(args.output_root / "progress.jsonl", "k1o_preflight_passed")

    selected_manifest = [
        {
            **row,
            "source_manifest_run_id": row.get("run_id"),
            "run_id": RUN_ID,
        }
        for row in dataset_manifest
        if row.get("cipher_key") == "uknit64"
    ]
    write_jsonl(args.output_root / "dataset_manifest.jsonl", selected_manifest)
    all_datasets = load_bound_datasets(dataset_manifest)
    datasets = {
        (seed, split): all_datasets[("uknit64", seed, split)]
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
    }

    task_map = candidate_task_map(read_tasks(args.plan))
    exact_structures = {}
    wrong_sbox_structures = {}
    for seed in EXPECTED_SEEDS:
        task = task_map[("uknit64", seed)]
        input_bits = int(datasets[(seed, "train_seen")].features.shape[1])
        exact_structures[seed] = build_k1n_control(
            task=task,
            condition="exact_composition",
            input_bits=input_bits,
        ).runtime_structure
        wrong_sbox_structures[seed] = build_k1n_control(
            task=task,
            condition="wrong_sbox_semantics",
            input_bits=input_bits,
        ).runtime_structure

    progress(
        args.output_root / "progress.jsonl",
        "k1o_signal_audit_start",
        expected_feature_rows=EXPECTED_FEATURE_ROWS,
        expected_scorer_rows=EXPECTED_SCORER_ROWS,
        expected_result_rows=EXPECTED_RESULT_ROWS,
    )
    feature_rows, scorer_rows, result_rows = evaluate_k1o(
        datasets=datasets,
        exact_structures=exact_structures,
        wrong_sbox_structures=wrong_sbox_structures,
        batch_size=args.batch_size,
    )
    write_jsonl(args.output_root / "feature_manifest.jsonl", feature_rows)
    write_jsonl(args.output_root / "scorer_manifest.jsonl", scorer_rows)
    write_jsonl(args.output_root / "results.jsonl", result_rows)
    write_attribution_csv(args.output_root / "attribution.csv", result_rows)

    gate = adjudicate_k1o(
        result_rows=result_rows,
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        source_checks=source_checks,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "feature_rows": len(feature_rows),
        "expected_feature_rows": EXPECTED_FEATURE_ROWS,
        "scorer_rows": len(scorer_rows),
        "expected_scorer_rows": EXPECTED_SCORER_ROWS,
        "result_rows": len(result_rows),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "training_rows": 0,
        "neural_parameter_count": 0,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    write_json(
        args.output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "status": gate["status"],
            "decision": gate["decision"],
            "remote_scale": gate["remote_scale"],
            "seed_results": gate["seed_results"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
            "feature_rows": len(feature_rows),
            "scorer_rows": len(scorer_rows),
            "result_rows": len(result_rows),
            "training_rows": 0,
            "optimizer_steps": 0,
        },
    )
    plot_report = render_k1o_svg(gate, args.output_root / "curves.svg")
    write_json(args.output_root / "plot_report.json", plot_report)
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(result_rows),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "dataset_manifest.jsonl",
        "feature_manifest.jsonl",
        "scorer_manifest.jsonl",
        "results.jsonl",
        "progress.jsonl",
        "gate.json",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-O output root already contains run artifacts")


def write_attribution_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "cipher_key",
        "rounds",
        "seed",
        "split",
        "view",
        "rows",
        "auc",
        "zero_threshold_accuracy",
        "feature_dim",
        "dataset_sha256",
        "feature_sha256",
        "scorer_sha256",
        "fit_split",
        "fit_rows",
        "pairs_per_sample",
        "negative_mode",
        "variance_floor",
        "training_performed",
        "neural_parameter_count",
        "optimizer_steps",
        "epochs",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "write_attribution_csv"]
