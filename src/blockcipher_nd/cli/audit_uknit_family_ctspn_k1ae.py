from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import read_jsonl, write_json
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ac import (
    adjudicate as adjudicate_k1ac,
    read_tasks,
    task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ad import (
    SOURCE_DECISION as K1AC_SOURCE_DECISION,
    adjudicate as adjudicate_k1ad,
    load_validation_cache,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ae import (
    EXPECTED_SEEDS,
    RUN_ID,
    adjudicate,
    evaluate_branch_ablation,
)


K1AD_SOURCE_DECISION = "innovation1_uknit_family_ctspn_k1ad_discriminative_sbox_use_failed"
PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_dialga_retention_"
    "k1ac_16pair_2048_seed0_seed1.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate K1-AA edge and histogram residuals under frozen K1-AC checkpoints."
    )
    parser.add_argument("--k1ac-root", required=True, type=Path)
    parser.add_argument("--k1ad-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--batch-size", default=256, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-AE run_id must remain frozen as {RUN_ID}")
    _require_fresh_output_root(args.output_root)
    tasks = read_tasks(PLAN)
    mapped_tasks = task_map(tasks)

    k1ac_paths = {
        "results": args.k1ac_root / "results.jsonl",
        "progress": args.k1ac_root / "progress.jsonl",
        "gate": args.k1ac_root / "gate.json",
        "preflight": args.k1ac_root / "preflight.json",
    }
    k1ad_paths = {
        "results": args.k1ad_root / "results.jsonl",
        "gate": args.k1ad_root / "gate.json",
    }
    if not all(path.is_file() for path in (*k1ac_paths.values(), *k1ad_paths.values())):
        raise ValueError("K1-AE source artifacts are incomplete")

    k1ac_rows = read_jsonl(k1ac_paths["results"])
    k1ac_progress = read_jsonl(k1ac_paths["progress"])
    k1ac_gate = _read_json(k1ac_paths["gate"])
    recomputed_k1ac = adjudicate_k1ac(
        tasks=tasks,
        result_rows=k1ac_rows,
        progress_rows=k1ac_progress,
        readiness=_read_json(k1ac_paths["preflight"]),
    )
    if k1ac_gate != recomputed_k1ac or k1ac_gate.get("decision") != K1AC_SOURCE_DECISION:
        raise ValueError("K1-AE requires the exact completed K1-AC semantic hold")

    k1ad_rows = read_jsonl(k1ad_paths["results"])
    k1ad_gate = _read_json(k1ad_paths["gate"])
    recomputed_k1ad = adjudicate_k1ad(k1ad_rows)
    if k1ad_gate != recomputed_k1ad or k1ad_gate.get("decision") != K1AD_SOURCE_DECISION:
        raise ValueError("K1-AE requires the exact completed K1-AD discriminative hold")

    exact_k1ad_rows = _exact_k1ad_rows(k1ad_rows)
    result_rows: list[dict[str, Any]] = []
    args.output_root.mkdir(parents=True)
    _progress(args.output_root / "progress.jsonl", "run_start")
    for seed in EXPECTED_SEEDS:
        source = exact_k1ad_rows[seed]
        cache_dir = Path(str(source["cache_dir"]))
        dataset, cache_digests = load_validation_cache(cache_dir)
        result_rows.extend(
            evaluate_branch_ablation(
                seed=seed,
                task=mapped_tasks[(seed, "virtual_slot_exact")],
                source_row=source,
                checkpoint_path=Path(str(source["checkpoint_path"])),
                dataset=dataset,
                cache_digests=cache_digests,
                source_k1ac_gate_sha256=file_sha256(k1ac_paths["gate"]),
                source_k1ad_results_sha256=file_sha256(k1ad_paths["results"]),
                source_k1ad_gate_sha256=file_sha256(k1ad_paths["gate"]),
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
            "source_gates_recomputed": True,
            "result_rows": len(result_rows),
            "expected_rows": 8,
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


def _exact_k1ad_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    mapped = {
        int(row["seed"]): row
        for row in rows
        if row.get("condition") == "exact"
    }
    if set(mapped) != set(EXPECTED_SEEDS):
        raise ValueError("K1-AE exact K1-AD rows are incomplete")
    return mapped


def _write_comparison(path: Path, seed_results: Mapping[str, Mapping[str, Any]]) -> None:
    fields = (
        "seed",
        "full_auc",
        "histogram_off_auc",
        "edge_off_auc",
        "base_only_auc",
        "full_minus_histogram_off_auc",
        "full_minus_edge_off_auc",
        "full_minus_base_only_auc",
        "learned_edge_gate",
        "learned_histogram_gate",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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
        raise ValueError("K1-AE output root must be fresh")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args"]
