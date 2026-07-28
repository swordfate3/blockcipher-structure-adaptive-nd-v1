from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from blockcipher_nd.cli.train import main as train_main
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.planning.matrix import build_tasks, tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    CANDIDATE_MODEL,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_TRAINING_ROWS,
    READINESS_RUN_ID,
    RUN_ID,
    adjudicate_k1h,
    evaluate_k1h_panel,
    load_bound_datasets,
    validate_k1h_source_bindings,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run K1-H exact operator-tied latent training and the frozen "
            "three-split candidate/control/Runtime-E4 panel."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--readiness-root", required=True, type=Path)
    parser.add_argument("--k1f-root", required=True, type=Path)
    parser.add_argument("--k1g-root", required=True, type=Path)
    parser.add_argument("--k1-root", required=True, type=Path)
    parser.add_argument("--k1-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--resume-evaluation", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-H run_id must remain frozen as {RUN_ID}")
    candidate_tasks = tasks(args.plan)
    anchor_tasks = tasks(args.k1_plan)
    readiness = read_json(args.readiness_root / "gate.json")
    if not readiness_valid(readiness):
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 4
    dataset_manifest = read_jsonl(args.k1g_root / "dataset_manifest.jsonl")
    anchor_results = read_jsonl(args.k1_root / "results.jsonl")
    anchor_manifest = read_json(args.k1_root / "checkpoint_manifest.json")
    source_checks = validate_k1h_source_bindings(
        candidate_tasks=candidate_tasks,
        dataset_manifest=dataset_manifest,
        anchor_tasks=anchor_tasks,
        anchor_results=anchor_results,
        anchor_checkpoint_manifest=anchor_manifest,
    )
    if not all(source_checks.values()):
        raise ValueError(f"K1-H source bindings failed: {source_checks}")

    k1f_preflight = read_json(args.k1f_root / "preflight.json")
    source_cache_root = Path(str(k1f_preflight["source_cache_root"]))
    train_argv = training_argv(args, source_cache_root)
    train_args = parse_train_args(train_argv)
    if build_tasks(train_args) != candidate_tasks:
        raise ValueError("K1-H training parser drifted from the frozen plan")

    if args.resume_evaluation:
        validate_resume_root(args, source_cache_root)
    else:
        require_fresh_output_root(args.output_root)
        args.output_root.mkdir(parents=True)
        preflight = {
            "run_id": RUN_ID,
            "status": "pass",
            "execution_authorized": True,
            "plan": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "readiness_root": str(args.readiness_root),
            "readiness_gate_sha256": file_sha256(args.readiness_root / "gate.json"),
            "k1f_root": str(args.k1f_root),
            "k1g_root": str(args.k1g_root),
            "k1_root": str(args.k1_root),
            "k1_plan": str(args.k1_plan),
            "source_cache_root": str(source_cache_root),
            "source_checks": source_checks,
        }
        write_json(args.output_root / "preflight.json", preflight)
        write_jsonl(args.output_root / "dataset_manifest.jsonl", dataset_manifest)
        train_main(train_argv)

    training_rows = read_jsonl(args.output_root / "results.jsonl")
    if len(training_rows) != EXPECTED_TRAINING_ROWS:
        raise ValueError("K1-H did not produce four training rows")
    cache_checks = cache_reuse_checks(read_jsonl(args.output_root / "progress.jsonl"))
    if not all(cache_checks.values()):
        raise ValueError(f"K1-H source cache reuse failed: {cache_checks}")
    candidate_manifest = checkpoint_manifest(training_rows)
    write_json(args.output_root / "checkpoint_manifest.json", candidate_manifest)
    datasets = load_bound_datasets(dataset_manifest)
    progress(
        args.output_root / "progress.jsonl",
        "k1h_frozen_panel_start",
        expected_rows=EXPECTED_EVALUATION_ROWS,
    )
    evaluation_rows = evaluate_k1h_panel(
        candidate_tasks=candidate_tasks,
        candidate_training_rows=training_rows,
        candidate_checkpoint_manifest=candidate_manifest,
        anchor_tasks=anchor_tasks,
        anchor_results=anchor_results,
        anchor_checkpoint_manifest=anchor_manifest,
        datasets=datasets,
        device=args.device,
    )
    write_jsonl(args.output_root / "controls.jsonl", evaluation_rows)
    write_csv(args.output_root / "split_attribution.csv", evaluation_rows)
    gate = adjudicate_k1h(
        tasks=candidate_tasks,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        readiness_gate=readiness,
    )
    gate["cache_checks"] = cache_checks
    gate["source_checks"] = source_checks
    validation = {
        "run_id": RUN_ID,
        "status": (
            "pass"
            if all(gate["protocol_checks"].values())
            and all(cache_checks.values())
            and all(source_checks.values())
            else "fail"
        ),
        "checks": {
            **gate["protocol_checks"],
            **cache_checks,
            **source_checks,
        },
        "errors": gate["failed_protocol_checks"],
        "training_rows": len(training_rows),
        "expected_training_rows": EXPECTED_TRAINING_ROWS,
        "evaluation_rows": len(evaluation_rows),
        "expected_evaluation_rows": EXPECTED_EVALUATION_ROWS,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "training_rows": len(training_rows),
        "evaluation_rows": len(evaluation_rows),
        "seed_results": gate["seed_results"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    write_json(args.output_root / "summary.json", summary)
    write_history_csv(
        args.output_root / "results.jsonl",
        args.output_root / "history.csv",
    )
    progress(
        args.output_root / "progress.jsonl",
        "k1h_gate_done",
        status=gate["status"],
        decision=gate["decision"],
        evaluation_rows=len(evaluation_rows),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def training_argv(args: argparse.Namespace, source_cache_root: Path) -> list[str]:
    return [
        "--plan",
        str(args.plan),
        "--device",
        args.device,
        "--batch-size",
        str(EXPECTED_BATCH_SIZE),
        "--hidden-bits",
        "32",
        "--dataset-cache-root",
        str(source_cache_root),
        "--dataset-cache-chunk-size",
        "1024",
        "--dataset-cache-workers",
        "1",
        "--checkpoint-output-dir",
        str(args.output_root / "checkpoints"),
        "--progress-output",
        str(args.output_root / "progress.jsonl"),
        "--output",
        str(args.output_root / "results.jsonl"),
    ]


def readiness_valid(gate: Mapping[str, Any]) -> bool:
    return (
        gate.get("run_id") == READINESS_RUN_ID
        and gate.get("status") == "pass"
        and gate.get("optimizer_step_authorized") is True
        and all(gate.get("protocol_checks", {}).values())
        and all(gate.get("evidence_checks", {}).values())
    )


def cache_reuse_checks(events: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    source_events = [
        row
        for row in events
        if row.get("event") in {"cache_reuse", "cache_start"}
        and row.get("split") in {"train", "validation"}
    ]
    return {
        "eight_training_and_validation_caches_reused": (
            sum(row.get("event") == "cache_reuse" for row in source_events) == 8
        ),
        "no_training_or_validation_cache_regenerated": not any(
            row.get("event") == "cache_start" for row in source_events
        ),
    }


def checkpoint_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries = []
    for row in rows:
        checkpoint = Path(str(row["training"]["checkpoint_output"]))
        if row.get("model") != CANDIDATE_MODEL or not checkpoint.is_file():
            raise ValueError("K1-H missing candidate checkpoint")
        entries.append(
            {
                "cipher_key": row["cipher_key"],
                "seed": row["seed"],
                "model": row["model"],
                "selected_checkpoint": row["training"]["selected_checkpoint"],
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
            }
        )
    return {"run_id": RUN_ID, "status": "pass", "entries": entries}


def validate_resume_root(args: argparse.Namespace, source_cache_root: Path) -> None:
    preflight = read_json(args.output_root / "preflight.json")
    rows = read_jsonl(args.output_root / "results.jsonl")
    if (
        preflight.get("run_id") != RUN_ID
        or preflight.get("plan_sha256") != file_sha256(args.plan)
        or preflight.get("source_cache_root") != str(source_cache_root)
        or len(rows) != EXPECTED_TRAINING_ROWS
    ):
        raise ValueError("K1-H resume root does not match frozen training")
    checkpoint_manifest(rows)


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "results.jsonl",
        "controls.jsonl",
        "progress.jsonl",
        "gate.json",
        "checkpoints",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-H output root already contains run artifacts")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "cipher_key",
        "seed",
        "split",
        "source_role",
        "condition",
        "rows",
        "auc",
        "exact_minus_condition_auc",
        "max_abs_probability_delta_from_exact",
        "mean_abs_probability_delta_from_exact",
        "dataset_sha256",
        "checkpoint_sha256",
        "state_dict_sha256",
        "operator_routing_sha256",
        "training_performed",
        "optimizer_steps",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def progress(path: Path, event: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event": event, "time": time.time(), **payload},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
