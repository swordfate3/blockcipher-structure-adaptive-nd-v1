from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    write_json,
    write_jsonl,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_edge_context_covariance_k1by11 import (
    CONFIG_PATH,
    EXPECTED_RESULT_ROWS,
    RUN_ID,
    adjudicate,
    authority_digests,
    build_readiness,
    comparison_rows,
    evaluate,
    load_and_validate_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the K1-BY11 edge-context covariance representation."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--readiness-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-BY11 run_id must remain frozen as {RUN_ID}")
    config = load_and_validate_config(args.config)
    if args.device != config["audit"]["device"]:
        raise ValueError("K1-BY11 device drifted from frozen config")
    readiness = build_readiness(config)
    if readiness.get("execution_authorized") is not True:
        raise ValueError(f"K1-BY11 readiness failed: {readiness}")
    require_fresh_output_root(args.output_root)
    args.output_root.mkdir(parents=True)
    source_before = authority_digests(config)
    write_json(
        args.output_root / "preflight.json",
        {
            **readiness,
            "config": str(args.config),
            "config_sha256": file_sha256(args.config),
            "source_digests": source_before,
            "device": args.device,
            "neural_training_performed": False,
            "optimizer_steps": 0,
            "epochs": 0,
        },
    )
    if args.readiness_only:
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 0

    progress(args.output_root / "progress.jsonl", "k1by11_audit_start")
    result_rows, final_evaluation, model_metadata = evaluate(config)
    source_after = authority_digests(config)
    gate = adjudicate(
        config,
        result_rows=result_rows,
        final_evaluation=final_evaluation,
        model_metadata=model_metadata,
        readiness=readiness,
        sources_unchanged=source_before == source_after,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(result_rows),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "source_digests_before": source_before,
        "source_digests_after": source_after,
        "neural_training_performed": False,
        "optimizer_steps": 0,
        "epochs": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "method_status": gate["method_status"],
        "decision": gate["decision"],
        "research_gate_passed": gate["research_gate_passed"],
        "seed_results": gate["seed_results"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
        "result_rows": len(result_rows),
        "optimizer_steps": 0,
    }
    write_jsonl(args.output_root / "results.jsonl", result_rows)
    write_comparison_csv(
        args.output_root / "condition_comparison.csv",
        comparison_rows(gate),
    )
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    write_json(args.output_root / "summary.json", summary)
    write_json(args.output_root / "model_metadata.json", model_metadata)
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        result_rows=len(result_rows),
        optimizer_steps=0,
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def require_fresh_output_root(path: Path) -> None:
    protected = ("preflight.json", "results.jsonl", "progress.jsonl", "gate.json")
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-BY11 output root already contains audit artifacts")


def write_comparison_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "seed",
        "tap_index",
        "tap",
        "correct_runtime_probe_auc",
        "affine_runtime_probe_auc",
        "shuffled_edges_probe_auc",
        "correct_minus_affine_probe_auc",
        "correct_minus_shuffled_probe_auc",
        "margin_pass",
        "first_margin_loss",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "write_comparison_csv"]
