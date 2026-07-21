from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_r9_external_relation_source_readiness import (
    RUN_ID,
    ExternalRelationSourceConfig,
    audit_external_relation_sources,
    result_rows,
    serializable_config,
    sha256,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit direct external-source eligibility for the frozen PRESENT r9 ATM model."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--public-results-root", required=True, type=Path)
    parser.add_argument("--e104-root", required=True, type=Path)
    parser.add_argument("--e101-gate", required=True, type=Path)
    parser.add_argument("--paper-manifest", required=True, type=Path)
    parser.add_argument("--hwang-text", required=True, type=Path)
    parser.add_argument("--split-text", required=True, type=Path)
    parser.add_argument("--claasp-text", required=True, type=Path)
    parser.add_argument("--splitandcancel-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExternalRelationSourceConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False, "search": False})
    e101_gate = _load_json(args.e101_gate)
    split_repository = _split_repository_audit(args.splitandcancel_root)
    audit = audit_external_relation_sources(
        config,
        public_results_root=args.public_results_root,
        e104_root=args.e104_root,
        e101_gate=e101_gate,
        e101_gate_sha256=sha256(args.e101_gate),
        paper_manifest=args.paper_manifest,
        hwang_text=args.hwang_text,
        split_text=args.split_text,
        claasp_text=args.claasp_text,
        split_repository=split_repository,
    )
    gate = audit["gate"]
    rows = result_rows(config, source_rows=audit["source_rows"], gate=gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_r9_external_relation_source_readiness",
        "experiment": "e106",
        "config": serializable_config(config),
        "training_performed": False,
        "search_performed": False,
        "remote_scale": False,
        "splitandcancel_repository": split_repository,
        "input_hashes": {
            "e101_gate": sha256(args.e101_gate),
            "paper_manifest": sha256(args.paper_manifest),
            "hwang_text": sha256(args.hwang_text),
            "split_text": sha256(args.split_text),
            "claasp_text": sha256(args.claasp_text),
            "e104_validation": sha256(args.e104_root / "validation.json"),
            "e104_relations": sha256(args.e104_root / "results/relations.json"),
        },
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "protocol_checks": audit["protocol_checks"],
        "e104_novelty": audit["e104_novelty"],
        "e104_fold_overlap": audit["e104_fold_overlap"],
        "source_matrix": audit["source_rows"],
        "gate": gate,
    }
    _write_csv(args.output_root / "source_matrix.csv", audit["source_rows"])
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True)
    )
    return 1 if gate["status"] == "fail" else 0


def _split_repository_audit(root: Path) -> dict[str, Any]:
    tracked_files = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    readme = root / "README.md"
    return {
        "commit": subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "tracked_files": tracked_files,
        "readme_sha256": sha256(readme),
        "readme_text": readme.read_text(encoding="utf-8"),
        "worktree_clean": not subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
