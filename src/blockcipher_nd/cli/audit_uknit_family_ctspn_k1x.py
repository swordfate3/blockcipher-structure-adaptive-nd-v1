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
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1x import (
    EXPECTED_GRADIENT_ROWS,
    EXPECTED_INFERENCE_ROWS,
    RUN_ID,
    adjudicate,
    audit_gradient_panel,
    audit_inference_panel,
    comparison_rows,
    source_binding_checks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit uKNIT K1-X compact projection optimization geometry."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-X run_id must remain frozen as {RUN_ID}")
    require_fresh_output_root(args.output_root)
    args.output_root.mkdir(parents=True)
    progress(args.output_root / "progress.jsonl", "k1x_audit_start")

    source_checks = source_binding_checks()
    write_json(
        args.output_root / "source_bindings.json",
        {
            "run_id": RUN_ID,
            "status": "pass" if source_checks and all(source_checks.values()) else "fail",
            "checks": source_checks,
        },
    )
    if not source_checks or not all(source_checks.values()):
        gate = adjudicate(
            inference_rows=[],
            gradient_rows=[],
            source_checks=source_checks,
        )
        write_artifacts(args.output_root, [], [], gate)
        return 1

    inference_rows = audit_inference_panel(device=args.device)
    gradient_rows = audit_gradient_panel()
    gate = adjudicate(
        inference_rows=inference_rows,
        gradient_rows=gradient_rows,
        source_checks=source_checks,
    )
    write_artifacts(args.output_root, inference_rows, gradient_rows, gate)
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def write_artifacts(
    output_root: Path,
    inference_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> None:
    write_jsonl(output_root / "results.jsonl", inference_rows)
    write_jsonl(output_root / "gradients.jsonl", gradient_rows)
    write_json(output_root / "gate.json", gate)
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate.get("protocol_checks", {}).values()) else "fail",
        "checks": gate.get("protocol_checks", {}),
        "errors": gate.get("failed_protocol_checks", []),
        "result_rows": len(inference_rows),
        "expected_result_rows": EXPECTED_INFERENCE_ROWS,
        "gradient_rows": len(gradient_rows),
        "expected_gradient_rows": EXPECTED_GRADIENT_ROWS,
    }
    write_json(output_root / "validation.json", validation)
    write_json(
        output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "status": gate["status"],
            "decision": gate["decision"],
            "seed_results": gate.get("seed_results", {}),
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    write_comparison_csv(output_root / "comparison.csv", comparison_rows(gate))


def write_comparison_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "seed",
        "k1w_exact_auc",
        "k1w_zero_histogram_auc",
        "k1w_wrong_sbox_same_checkpoint_auc",
        "k1w_full_minus_zero_auc",
        "k1w_exact_minus_wrong_sbox_auc",
        "k1w_histogram_effective_gate",
        "folded_effective_update_ratio",
        "slot_gradient_relative_error",
        "folded_gradient_relative_error",
        "k1t_folded_exact_auc",
        "k1t_folded_zero_histogram_auc",
        "k1t_folded_wrong_sbox_same_checkpoint_auc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "source_bindings.json",
        "results.jsonl",
        "gradients.jsonl",
        "gate.json",
        "validation.json",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-X output root already contains audit artifacts")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "write_artifacts", "write_comparison_csv"]
