from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    read_jsonl,
    write_json,
)
from blockcipher_nd.cli.train import main as train_main
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_r5_published_architecture_baselines_k1bz import (
    ARCHITECTURES,
    K1BS_ROOT,
    RUN_ID,
    adjudicate,
    build_readiness,
    candidate_protocol_frozen,
    read_tasks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the uKNIT r5 K1-BZ published architecture baselines."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--readiness-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-BZ run_id must remain frozen as {RUN_ID}")
    tasks = read_tasks(args.plan)
    if not candidate_protocol_frozen(tasks):
        raise ValueError("K1-BZ plan does not match the frozen protocol")
    readiness = build_readiness(tasks)
    if readiness.get("optimizer_step_authorized") is not True:
        raise ValueError(f"K1-BZ readiness failed: {readiness}")
    require_fresh_output_root(args.output_root)
    args.output_root.mkdir(parents=True)
    write_json(
        args.output_root / "preflight.json",
        {
            **readiness,
            "plan": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "cache_source": str(K1BS_ROOT / "cache/uknit64"),
        },
    )
    if args.readiness_only:
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 0

    link_k1bs_cache(args.output_root)
    train_argv = training_argv(args)
    if build_tasks(parse_train_args(train_argv)) != tasks:
        raise ValueError("K1-BZ training parser drifted from the frozen plan")
    train_main(train_argv)
    result_rows = read_jsonl(args.output_root / "results.jsonl")
    progress_rows = read_jsonl(args.output_root / "progress.jsonl")
    gate = adjudicate(tasks, result_rows, progress_rows, readiness)
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "result_rows": len(result_rows),
        "expected_rows": 4,
    }
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    write_json(
        args.output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "status": gate["status"],
            "decision": gate["decision"],
            "seed_results": gate["seed_results"],
            "selected_remote_candidate": gate["selected_remote_candidate"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    write_comparison_csv(args.output_root / "architecture_comparison.csv", gate)
    write_history_csv(
        args.output_root / "results.jsonl", args.output_root / "history.csv"
    )
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def training_argv(args: argparse.Namespace) -> list[str]:
    return [
        "--plan",
        str(args.plan),
        "--device",
        args.device,
        "--batch-size",
        "64",
        "--hidden-bits",
        "32",
        "--dataset-cache-root",
        str(args.output_root / "cache"),
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


def link_k1bs_cache(output_root: Path) -> None:
    source = (K1BS_ROOT / "cache/uknit64").resolve(strict=True)
    cache_root = output_root / "cache"
    cache_root.mkdir()
    (cache_root / "uknit64").symlink_to(source, target_is_directory=True)


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "results.jsonl",
        "progress.jsonl",
        "gate.json",
        "checkpoints",
        "cache",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-BZ output root already contains run artifacts")


def write_comparison_csv(path: Path, gate: Mapping[str, Any]) -> None:
    fields = (
        "seed",
        *(f"{architecture}_auc" for architecture in ARCHITECTURES),
        "structure_expert_auc",
        "autond_dbitnet_auc",
        *(f"{architecture}_minus_autond" for architecture in ARCHITECTURES),
    )
    rows: list[dict[str, Any]] = []
    for seed, values in sorted(gate.get("seed_results", {}).items()):
        rows.append(
            {
                "seed": int(seed),
                **{
                    f"{architecture}_auc": values["auc_by_architecture"][architecture]
                    for architecture in ARCHITECTURES
                },
                "structure_expert_auc": values["k1bs_anchors"]["structure_expert"],
                "autond_dbitnet_auc": values["k1bs_anchors"]["autond_dbitnet"],
                **{
                    f"{architecture}_minus_autond": values["adapter_minus_autond"][architecture]
                    for architecture in ARCHITECTURES
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["link_k1bs_cache", "main", "parse_args", "training_argv"]
