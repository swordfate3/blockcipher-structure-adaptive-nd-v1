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
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1z import (
    EXPECTED_CONFIRMATION_ROWS,
    EXPECTED_GRID_ROWS,
    RUN_ID,
    adjudicate,
    comparison_rows,
    run_audit,
    source_binding_checks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit uKNIT K1-Z compact histogram branch interference."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-Z run_id must remain frozen as {RUN_ID}")
    require_fresh_output_root(args.output_root)
    args.output_root.mkdir(parents=True)
    progress(args.output_root / "progress.jsonl", "k1z_audit_start")
    source_checks = source_binding_checks()
    write_json(
        args.output_root / "source_bindings.json",
        {
            "run_id": RUN_ID,
            "status": "pass" if source_checks and all(source_checks.values()) else "fail",
            "checks": source_checks,
        },
    )
    if source_checks and all(source_checks.values()):
        grid_rows, confirmation_rows = run_audit(device=args.device)
    else:
        grid_rows, confirmation_rows = [], []
    gate = adjudicate(
        grid_rows=grid_rows,
        confirmation_rows=confirmation_rows,
        source_checks=source_checks,
    )
    write_artifacts(args.output_root, grid_rows, confirmation_rows, gate)
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
    grid_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> None:
    write_jsonl(output_root / "alpha_grid.jsonl", grid_rows)
    write_jsonl(output_root / "results.jsonl", confirmation_rows)
    write_json(output_root / "gate.json", gate)
    write_json(
        output_root / "validation.json",
        {
            "run_id": RUN_ID,
            "status": "pass" if all(gate.get("protocol_checks", {}).values()) else "fail",
            "checks": gate.get("protocol_checks", {}),
            "errors": gate.get("failed_protocol_checks", []),
            "grid_rows": len(grid_rows),
            "expected_grid_rows": EXPECTED_GRID_ROWS,
            "result_rows": len(confirmation_rows),
            "expected_result_rows": EXPECTED_CONFIRMATION_ROWS,
        },
    )
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
        "selected_alpha",
        "train_selected_auc",
        "validation_selected_exact_auc",
        "validation_selected_wrong_sbox_auc",
        "validation_exact_minus_wrong_sbox",
        "validation_alpha0_auc",
        "validation_alpha1_auc",
        "anchor_auc",
        "retention_threshold",
        "selected_minus_anchor",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def require_fresh_output_root(path: Path) -> None:
    if path.exists() and any(
        (path / name).exists()
        for name in (
            "source_bindings.json",
            "alpha_grid.jsonl",
            "results.jsonl",
            "gate.json",
        )
    ):
        raise ValueError("K1-Z output root already contains audit artifacts")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "write_artifacts", "write_comparison_csv"]
