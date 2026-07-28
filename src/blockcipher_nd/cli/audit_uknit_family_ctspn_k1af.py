from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import read_jsonl, write_json
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ac import read_tasks, task_map
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ad import load_validation_cache
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ae import (
    adjudicate as adjudicate_k1ae,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1af import (
    EXPECTED_SEEDS,
    RUN_ID,
    adjudicate,
    evaluate_single_pair_replay,
)


SOURCE_DECISION = "innovation1_uknit_family_ctspn_k1ae_gf2_base_path_dominates"
PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_dialga_retention_"
    "k1ac_16pair_2048_seed0_seed1.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay frozen K1-AC states on individual Dialga ciphertext pairs."
    )
    parser.add_argument("--k1ae-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--batch-size", default=256, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-AF run_id must remain frozen as {RUN_ID}")
    _require_fresh_output_root(args.output_root)
    source_results_path = args.k1ae_root / "results.jsonl"
    source_gate_path = args.k1ae_root / "gate.json"
    if not source_results_path.is_file() or not source_gate_path.is_file():
        raise ValueError("K1-AF source K1-AE artifacts are incomplete")
    source_rows = read_jsonl(source_results_path)
    persisted_gate = _read_json(source_gate_path)
    recomputed_gate = adjudicate_k1ae(source_rows)
    if persisted_gate != recomputed_gate or persisted_gate.get("decision") != SOURCE_DECISION:
        raise ValueError("K1-AF requires the exact completed K1-AE GF(2)-base diagnosis")

    tasks = task_map(read_tasks(PLAN))
    full_rows = _full_source_rows(source_rows)
    result_rows: list[dict[str, Any]] = []
    args.output_root.mkdir(parents=True)
    _progress(args.output_root / "progress.jsonl", "run_start")
    for seed in EXPECTED_SEEDS:
        source = full_rows[seed]
        dataset, cache_digests = load_validation_cache(Path(str(source["cache_dir"])))
        result_rows.extend(
            evaluate_single_pair_replay(
                seed=seed,
                task=tasks[(seed, "virtual_slot_exact")],
                checkpoint_path=Path(str(source["checkpoint_path"])),
                dataset=dataset,
                cache_digests=cache_digests,
                source_k1ae_gate_sha256=file_sha256(source_gate_path),
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
            "expected_rows": 72,
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
    _write_positions(args.output_root / "per_position.csv", gate["seed_results"])
    _progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def _full_source_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    mapped = {
        int(row["seed"]): row
        for row in rows
        if row.get("condition") == "full"
    }
    if set(mapped) != set(EXPECTED_SEEDS):
        raise ValueError("K1-AF K1-AE full rows are incomplete")
    return mapped


def _write_positions(path: Path, seed_results: Mapping[str, Mapping[str, Any]]) -> None:
    fields = ("seed", "pair_position", "exact_auc", "wrong_sbox_auc", "exact_minus_wrong_auc")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed, result in sorted(seed_results.items()):
            for row in result["per_position"]:
                writer.writerow({"seed": int(seed), **dict(row)})


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
        raise ValueError("K1-AF output root must be fresh")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args"]
