from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_r9_atm_source_heldout_ranking import (
    E99_GATE_SHA256,
    E99_SUMMARY_SHA256,
    SourceHeldoutRankingConfig,
    evaluate_source_heldout,
    load_relations_json,
    replay_e99_coordinate_checkpoints,
    serializable_config,
    sha256,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_split333_generation import (
    SOURCE_HASHES,
    search_config,
)
from blockcipher_nd.tasks.innovation2.present_r9_pu_ranking_readiness import (
    audit_sources,
    load_relation_groups,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze E99 coordinate checkpoints or evaluate E104 split333 without adaptation."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--results-root", required=True, type=Path)
    freeze.add_argument("--atm-root", required=True, type=Path)
    freeze.add_argument("--e99-summary", required=True, type=Path)
    freeze.add_argument("--e99-gate", required=True, type=Path)
    freeze.add_argument("--output-root", required=True, type=Path)
    freeze.add_argument("--device", default="cpu")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--results-root", required=True, type=Path)
    evaluate.add_argument("--checkpoint-root", required=True, type=Path)
    evaluate.add_argument("--e104-root", required=True, type=Path)
    evaluate.add_argument("--output-root", required=True, type=Path)
    evaluate.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "freeze":
        return _freeze(args)
    return _evaluate(args)


def _freeze(args: argparse.Namespace) -> int:
    config = SourceHeldoutRankingConfig()
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "freeze_start", {"heldout_source_read": False})
    actual_commit = subprocess.run(
        ["git", "-C", str(args.atm_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_audit = audit_sources(args.results_root, actual_commit=actual_commit)
    input_checks = {
        **source_audit["checks"],
        "e99_summary_hash_matches": sha256(args.e99_summary) == E99_SUMMARY_SHA256,
        "e99_gate_hash_matches": sha256(args.e99_gate) == E99_GATE_SHA256,
    }
    if not all(input_checks.values()):
        gate = {
            "run_id": config.freeze_run_id,
            "status": "fail",
            "decision": "innovation2_present_r9_e99_checkpoint_replay_source_invalid",
            "input_checks": input_checks,
            "next_action": {"action": "repair frozen public source inputs before training"},
        }
        _write_json(args.output_root / "gate.json", gate)
        _write_json(args.output_root / "source_audit.json", source_audit)
        _write_progress(progress, "freeze_done", {"status": "fail"})
        return 1

    groups = load_relation_groups(args.results_root)
    e99_summary = json.loads(args.e99_summary.read_text(encoding="utf-8"))
    replay = replay_e99_coordinate_checkpoints(
        config,
        groups=groups,
        e99_summary=e99_summary,
        output_root=args.output_root,
        device=args.device,
        progress_callback=lambda event, payload: _write_progress(progress, event, payload),
    )
    replay["gate"]["input_checks"] = input_checks
    replay["manifest"]["input_checks"] = input_checks
    replay["manifest"]["public_source_commit"] = actual_commit
    metadata = {
        "run_id": config.freeze_run_id,
        "task": "innovation2_present_r9_atm_e99_coordinate_checkpoint_replay",
        "experiment": "e105_freeze",
        "config": serializable_config(config),
        "device": args.device,
        "training_performed": True,
        "training_source": "ATM public eight PRESENT r9 splits only",
        "heldout_source_read": False,
        "checkpoint_selection": "E99 fixed final epoch; no heldout selection",
        "claim_scope": "deterministic E99 checkpoint replay only; not a new neural result",
    }
    _write_json(args.output_root / "checkpoint_manifest.json", replay["manifest"])
    _write_json(args.output_root / "gate.json", replay["gate"])
    _write_json(args.output_root / "source_audit.json", source_audit)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_csv(args.output_root / "replay_metrics.csv", replay["replay_rows"])
    _write_jsonl(
        args.output_root / "results.jsonl",
        [
            {
                "run_id": config.freeze_run_id,
                "task": "innovation2_present_r9_atm_e99_coordinate_checkpoint_replay",
                "status": replay["gate"]["status"],
                "decision": replay["gate"]["decision"],
                **{key: value for key, value in row.items() if key != "metric_checks"},
            }
            for row in replay["replay_rows"]
        ],
    )
    _write_progress(
        progress,
        "freeze_done",
        {"status": replay["gate"]["status"], "decision": replay["gate"]["decision"]},
    )
    return 1 if replay["gate"]["status"] == "fail" else 0


def _evaluate(args: argparse.Namespace) -> int:
    config = SourceHeldoutRankingConfig()
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "evaluation_start", {"training_on_heldout": False})
    manifest_path = args.checkpoint_root / "checkpoint_manifest.json"
    e104_gate_path = args.e104_root / "gate.json"
    relations_path = args.e104_root / "relations.json"
    marker_path = args.e104_root / "generation_passed.marker"
    source_hashes_path = args.e104_root / "source_hashes.json"
    model_contract_path = args.e104_root / "model_contract.json"
    search_metadata_path = args.e104_root / "search_state/metadata.json"
    required = (
        manifest_path,
        e104_gate_path,
        relations_path,
        marker_path,
        source_hashes_path,
        model_contract_path,
        search_metadata_path,
    )
    if not all(path.is_file() for path in required):
        gate = {
            "run_id": config.evaluation_run_id,
            "status": "fail",
            "decision": "innovation2_present_r9_split333_source_heldout_inputs_missing",
            "required_files": {str(path): path.is_file() for path in required},
            "next_action": {"action": "wait for verified E104 retrieval; do not evaluate partial data"},
        }
        _write_json(args.output_root / "gate.json", gate)
        _write_progress(progress, "evaluation_done", {"status": "fail"})
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    e104_gate = json.loads(e104_gate_path.read_text(encoding="utf-8"))
    source_hashes = json.loads(source_hashes_path.read_text(encoding="utf-8"))
    model_contract = json.loads(model_contract_path.read_text(encoding="utf-8"))
    search_metadata = json.loads(search_metadata_path.read_text(encoding="utf-8"))
    generation_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_parameter_hash = search_config().parameter_hash()
    e104_evidence_checks = {
        "source_hashes_match_frozen_atm": source_hashes == SOURCE_HASHES,
        "model_sha256_matches_full_present64": model_contract.get("sha256")
        == "ccc91bfdb16e6104eeca6fde32ec71951f130c261e17b5b7202e6197304166d2",
        "search_parameter_hash_matches": search_metadata.get("parameter_hash")
        == expected_parameter_hash,
        "generation_marker_parameter_hash_matches": generation_marker.get("parameter_hash")
        == expected_parameter_hash,
    }
    relations = load_relations_json(relations_path)
    public_groups = load_relation_groups(args.results_root)
    evaluation = evaluate_source_heldout(
        config,
        public_groups=public_groups,
        heldout_relations=relations,
        checkpoint_manifest=manifest,
        checkpoint_root=args.checkpoint_root,
        e104_gate=e104_gate,
        e104_evidence_checks=e104_evidence_checks,
        device=args.device,
    )
    metadata = {
        "run_id": config.evaluation_run_id,
        "task": "innovation2_present_r9_atm_split333_source_heldout_ranking",
        "experiment": "e105_evaluate",
        "config": serializable_config(config),
        "device": args.device,
        "training_performed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "candidate_semantics": "unlabeled, not negative",
        "input_hashes": {
            "checkpoint_manifest": sha256(manifest_path),
            "e104_gate": sha256(e104_gate_path),
            "e104_relations": sha256(relations_path),
            "e104_generation_marker": sha256(marker_path),
            "e104_source_hashes": sha256(source_hashes_path),
            "e104_model_contract": sha256(model_contract_path),
            "e104_search_metadata": sha256(search_metadata_path),
        },
        "claim_scope": evaluation["gate"].get("claim_scope", "protocol invalid"),
    }
    common = {
        "run_id": config.evaluation_run_id,
        "task": "innovation2_present_r9_atm_split333_source_heldout_ranking",
        "status": evaluation["gate"]["status"],
        "decision": evaluation["gate"]["decision"],
        "candidate_semantics": "unlabeled_not_negative",
    }
    _write_json(args.output_root / "gate.json", evaluation["gate"])
    _write_json(args.output_root / "summary.json", evaluation)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_jsonl(
        args.output_root / "results.jsonl",
        [{**common, **row} for row in evaluation["result_rows"]],
    )
    _write_csv(args.output_root / "relation_ranks.csv", evaluation["rank_rows"])
    _write_progress(
        progress,
        "evaluation_done",
        {"status": evaluation["gate"]["status"], "decision": evaluation["gate"]["decision"]},
    )
    return 1 if evaluation["gate"]["status"] == "fail" else 0


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
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flattened = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    fieldnames = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


if __name__ == "__main__":
    raise SystemExit(main())
