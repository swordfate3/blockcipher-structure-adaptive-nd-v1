from __future__ import annotations

import argparse
import json
from pathlib import Path
from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)
from blockcipher_nd.cli.run_uknit_family_midori64_k1ai import (
    load_k1ai_datasets,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    build_control_checks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_same_checkpoint_k1aj import (
    EXPECTED_ROWS,
    EXPECTED_SOURCE_DIGESTS,
    RUN_ID,
    adjudicate,
    evaluate_same_checkpoint_panel,
    source_binding_checks,
)


PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_midori64_neural_attribution_"
    "k1ai_2048_seed6_seed7.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the K1-AI correct checkpoints under four same-state Midori64 "
            "runtime structure interventions."
        )
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--plan", default=PLAN, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--batch-size", default=64, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-AJ run_id must remain frozen as {RUN_ID}")
    require_fresh_output_root(args.output_root)

    source_paths = {
        "gate": args.source_root / "gate.json",
        "validation": args.source_root / "validation.json",
        "checkpoint_manifest": args.source_root / "checkpoint_manifest.json",
        "controls": args.source_root / "controls.jsonl",
        "dataset_manifest": args.source_root / "dataset_manifest.jsonl",
    }
    if not all(path.is_file() for path in source_paths.values()):
        raise ValueError("K1-AJ source run is incomplete")
    source_digests = {name: file_sha256(path) for name, path in source_paths.items()}
    source_gate = read_json(source_paths["gate"])
    source_validation = read_json(source_paths["validation"])
    checkpoint_manifest = read_json(source_paths["checkpoint_manifest"])
    source_controls = read_jsonl(source_paths["controls"])
    dataset_manifest = read_jsonl(source_paths["dataset_manifest"])
    source_checks = source_binding_checks(
        gate=source_gate,
        validation=source_validation,
        checkpoint_manifest=checkpoint_manifest,
        source_controls=source_controls,
        dataset_manifest=dataset_manifest,
        source_digests=source_digests,
    )
    tasks = read_tasks(args.plan)
    control_checks = build_control_checks(tasks)
    datasets = load_k1ai_datasets(dataset_manifest)
    source_checks["six_dataset_payload_digests_verified"] = len(datasets) == 6
    source_checks["source_artifact_hashes_live"] = (
        source_digests == EXPECTED_SOURCE_DIGESTS
    )
    if not all(source_checks.values()):
        raise ValueError(f"K1-AJ source binding failed: {source_checks}")
    if not all(control_checks.values()):
        raise ValueError(f"K1-AJ structure controls failed: {control_checks}")

    args.output_root.mkdir(parents=True)
    write_json(
        args.output_root / "preflight.json",
        {
            "run_id": RUN_ID,
            "status": "pass",
            "execution_authorized": True,
            "source_root": str(args.source_root),
            "plan": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "source_digests": source_digests,
            "source_checks": source_checks,
            "control_checks": control_checks,
            "training_performed": False,
            "optimizer_steps": 0,
            "epochs": 0,
        },
    )
    write_jsonl(args.output_root / "dataset_manifest.jsonl", dataset_manifest)
    write_json(args.output_root / "checkpoint_manifest.json", checkpoint_manifest)
    progress(
        args.output_root / "progress.jsonl", "run_start", expected_rows=EXPECTED_ROWS
    )
    rows = evaluate_same_checkpoint_panel(
        tasks=tasks,
        checkpoint_manifest=checkpoint_manifest,
        source_controls=source_controls,
        datasets=datasets,
        source_digests=source_digests,
        batch_size=args.batch_size,
        device=args.device,
    )
    write_jsonl(args.output_root / "results.jsonl", rows)
    write_csv(args.output_root / "comparison.csv", rows)
    gate = adjudicate(
        rows,
        source_checks=source_checks,
        control_checks=control_checks,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(rows),
        "expected_rows": EXPECTED_ROWS,
        "training_rows": 0,
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
            "seed_results": gate["seed_results"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(rows),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def require_fresh_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("K1-AJ output root must be fresh")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "require_fresh_output_root"]
