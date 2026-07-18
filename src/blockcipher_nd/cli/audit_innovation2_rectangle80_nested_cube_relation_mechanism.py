from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.rectangle80_nested_cube_relation_mechanism import (
    RUN_ID,
    NestedCubeRelationConfig,
    build_prefix_tensor,
    evaluate_relation_gate,
    evaluate_relation_modes,
    load_e94_sources,
    make_relation_maps,
    result_rows_for_relation_gate,
    serializable_config,
    validate_relation_maps,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit E95 RECTANGLE-80 nested-cube relation mechanism without neural training."
        )
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = NestedCubeRelationConfig(run_id=args.run_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(
        progress,
        "run_start",
        {"training": False, "cipher": "RECTANGLE-80", "rounds": 4},
    )
    sources = load_e94_sources(args.source_root)
    _write_progress(progress, "e94_source_loaded", sources["checks"])
    prefix = build_prefix_tensor(sources["chains"])
    _write_progress(progress, "r3_prefix_ready", {"shape": list(prefix.shape)})
    maps = make_relation_maps(sources["chains"])
    relation_checks = validate_relation_maps(sources["chains"], maps)
    _write_progress(progress, "relation_maps_ready", relation_checks)
    relation_results = evaluate_relation_modes(
        config, sources, prefix, maps
    )
    _write_progress(
        progress,
        "ridge_isotonic_complete",
        {"reports": relation_results["reports"]},
    )
    gate = evaluate_relation_gate(
        config, sources, prefix, maps, relation_results
    )
    result_rows = result_rows_for_relation_gate(config, gate)
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_rectangle80_nested_cube_relation_mechanism",
        "experiment": "e95",
        "source_root": str(args.source_root),
        "source_hashes": sources["hashes"],
        "config": serializable_config(config),
        "feature_contract": {
            "own_r3_prefix": 13,
            "predecessor_r3_prefix": 13,
            "successor_r3_prefix": 13,
            "neighbor_masks": 2,
            "dimension_one_hot": 3,
            "total": 44,
        },
        "relation_modes": list(relation_results["reports"]),
        "label_features_forbidden": True,
        "validation_statistics_forbidden": True,
        "training_performed": False,
        "neural_training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    serializable_maps = {
        name: {str(key): value for key, value in mapping.items()}
        for name, mapping in maps.items()
        if name != "split_indices"
    }
    serializable_maps["split_indices"] = maps["split_indices"]
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "metrics": gate["metrics"],
        "gate": gate,
        "result_rows": result_rows,
    }

    _write_json(
        args.output_root / "relation_maps.json",
        {"maps": serializable_maps, "checks": relation_checks},
    )
    _write_json(
        args.output_root / "ridge_reports.json",
        relation_results["reports"],
    )
    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate, "output_root": str(args.output_root)}, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
