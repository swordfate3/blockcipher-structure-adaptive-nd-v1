from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import read_jsonl, write_json
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ac import (
    CONTROL_MODELS,
    adjudicate as adjudicate_k1ac,
    read_tasks,
    task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ad import (
    EXPECTED_SEEDS,
    RUN_ID,
    SOURCE_DECISION,
    adjudicate,
    evaluate_same_checkpoint,
    load_validation_cache,
)


PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_dialga_retention_"
    "k1ac_16pair_2048_seed0_seed1.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit K1-AC exact checkpoints under a same-state wrong-S-box intervention."
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--batch-size", default=256, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-AD run_id must remain frozen as {RUN_ID}")
    _require_fresh_output_root(args.output_root)

    source_paths = {
        "results": args.source_root / "results.jsonl",
        "progress": args.source_root / "progress.jsonl",
        "gate": args.source_root / "gate.json",
        "preflight": args.source_root / "preflight.json",
    }
    if not all(path.is_file() for path in source_paths.values()):
        raise ValueError("K1-AD source run is incomplete")
    source_rows = read_jsonl(source_paths["results"])
    progress_rows = read_jsonl(source_paths["progress"])
    persisted_gate = _read_json(source_paths["gate"])
    preflight = _read_json(source_paths["preflight"])
    tasks = read_tasks(PLAN)
    recomputed_gate = adjudicate_k1ac(
        tasks=tasks,
        result_rows=source_rows,
        progress_rows=progress_rows,
        readiness=preflight,
    )
    if persisted_gate != recomputed_gate:
        raise ValueError("persisted K1-AC gate does not match recomputed source evidence")
    if persisted_gate.get("decision") != SOURCE_DECISION or persisted_gate.get("status") != "hold":
        raise ValueError("K1-AD requires the completed K1-AC semantic-attribution hold")

    mapped_tasks = task_map(tasks)
    exact_rows = _exact_source_rows(source_rows)
    source_digests = {name: file_sha256(path) for name, path in source_paths.items() if name != "preflight"}
    result_rows: list[dict[str, Any]] = []
    args.output_root.mkdir(parents=True)
    _progress(args.output_root / "progress.jsonl", "run_start")
    for seed in EXPECTED_SEEDS:
        source_row = exact_rows[seed]
        cache_dir = _validation_cache_dir(progress_rows, seed)
        dataset, cache_digests = load_validation_cache(cache_dir)
        checkpoint = Path(str(source_row["training"]["checkpoint_output"]))
        result_rows.extend(
            evaluate_same_checkpoint(
                seed=seed,
                task=mapped_tasks[(seed, "virtual_slot_exact")],
                source_row=source_row,
                checkpoint_path=checkpoint,
                dataset=dataset,
                cache_digests=cache_digests,
                source_results_sha256=source_digests["results"],
                source_gate_sha256=source_digests["gate"],
                source_progress_sha256=source_digests["progress"],
                batch_size=args.batch_size,
                device=args.device,
            )
        )
        _progress(args.output_root / "progress.jsonl", "seed_done", seed=seed)

    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    gate = adjudicate(result_rows)
    write_json(args.output_root / "gate.json", gate)
    write_json(
        args.output_root / "validation.json",
        {
            "run_id": RUN_ID,
            "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
            "checks": gate["protocol_checks"],
            "errors": gate["failed_protocol_checks"],
            "source_gate_recomputed": True,
            "result_rows": len(result_rows),
            "expected_rows": 4,
        },
    )
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
    _write_comparison(args.output_root / "comparison.csv", gate["seed_results"])
    _progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def _exact_source_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    mapped = {
        int(row["seed"]): row
        for row in rows
        if row.get("model") == CONTROL_MODELS["virtual_slot_exact"]
    }
    if set(mapped) != set(EXPECTED_SEEDS):
        raise ValueError("K1-AD exact source rows are incomplete")
    return mapped


def _validation_cache_dir(rows: Sequence[Mapping[str, Any]], seed: int) -> Path:
    matches = [
        Path(str(row["cache_path"]))
        for row in rows
        if row.get("event") == "cache_done"
        and row.get("model") == CONTROL_MODELS["virtual_slot_exact"]
        and row.get("split") == "validation"
        and row.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one exact K1-AC validation cache for seed {seed}")
    return matches[0]


def _write_comparison(path: Path, seed_results: Mapping[str, Mapping[str, Any]]) -> None:
    fields = (
        "seed",
        "exact_auc",
        "wrong_sbox_auc",
        "source_exact_auc",
        "exact_minus_wrong_sbox_auc",
        "max_abs_probability_delta",
        "mean_abs_probability_delta",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed, values in sorted(seed_results.items()):
            writer.writerow({"seed": int(seed), **dict(values)})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _progress(path: Path, event: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"run_id": RUN_ID, "event": event, **payload}, sort_keys=True) + "\n")


def _require_fresh_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("K1-AD output root must be fresh")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args"]
