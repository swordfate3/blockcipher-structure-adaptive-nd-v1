from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.run_uknit_family_ctspn_k1m import (
    progress,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)
from blockcipher_nd.cli.run_uknit_family_ctspn_k1r import (
    load_k1r_datasets,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import load_bound_state
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import build_k1n_control
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    EXPECTED_SEEDS,
    checkpoint_map,
    task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1s import (
    EXPECTED_SOURCE_DIGESTS,
    RUN_ID,
    adjudicate_k1s,
    evaluate_k1s,
    source_binding_checks,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the zero-training uKNIT K1-S representation-access audit."
    )
    parser.add_argument("--k1q-root", required=True, type=Path)
    parser.add_argument("--k1r-root", required=True, type=Path)
    parser.add_argument("--k1r-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-S run_id must remain frozen as {RUN_ID}")
    if args.batch_size <= 0:
        raise ValueError("K1-S batch size must be positive")
    require_fresh_output_root(args.output_root)

    source_paths = source_artifact_paths(args)
    source_digests = {
        name: file_sha256(path) for name, path in source_paths.items()
    }
    k1q_gate = read_json(args.k1q_root / "gate.json")
    k1q_validation = read_json(args.k1q_root / "validation.json")
    k1r_gate = read_json(args.k1r_root / "gate.json")
    k1r_validation = read_json(args.k1r_root / "validation.json")
    dataset_manifest = [
        row
        for row in read_jsonl(args.k1q_root / "dataset_manifest.jsonl")
        if row.get("phase") == "confirmation" and int(row.get("cell", -1)) == 11
    ]
    checkpoint_source = read_json(args.k1r_root / "checkpoint_manifest.json")
    source_checks = source_binding_checks(
        source_digests=source_digests,
        k1q_gate=k1q_gate,
        k1q_validation=k1q_validation,
        k1r_gate=k1r_gate,
        k1r_validation=k1r_validation,
        dataset_manifest=dataset_manifest,
        checkpoint_entries=checkpoint_source.get("entries", []),
    )
    datasets = load_k1r_datasets(dataset_manifest)
    source_checks["six_cache_payload_digests_verified"] = len(datasets) == 6
    if not all(source_checks.values()):
        raise ValueError(f"K1-S source binding failed: {source_checks}")

    tasks = read_tasks(args.k1r_plan)
    exact_tasks = {
        seed: task_map(tasks)[(seed, "exact_composition")] for seed in EXPECTED_SEEDS
    }
    checkpoint_sources = checkpoint_map(checkpoint_source)
    models: dict[int, Any] = {}
    checkpoint_bindings: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        binding = dict(checkpoint_sources[(seed, "exact_composition")])
        checkpoint_path = Path(str(binding["path"]))
        state, checkpoint_sha = load_bound_state(checkpoint_path, binding)
        model = build_k1n_control(
            task=exact_tasks[seed],
            condition="exact_composition",
            input_bits=int(datasets[(seed, "train_seen")].features.shape[1]),
        )
        model.load_state_dict(state, strict=True)
        state_sha = tensor_mapping_sha256(state)
        if tensor_mapping_sha256(model.state_dict()) != state_sha:
            raise ValueError("K1-S strict checkpoint load changed learned state")
        model.eval()
        models[seed] = model
        checkpoint_bindings[seed] = {
            **binding,
            "sha256": checkpoint_sha,
            "state_dict_sha256": state_sha,
            "strict_state_dict_load": True,
        }

    args.output_root.mkdir(parents=True)
    write_json(
        args.output_root / "preflight.json",
        {
            "run_id": RUN_ID,
            "status": "pass",
            "execution_authorized": True,
            "training_authorized": False,
            "optimizer_steps": 0,
            "source_digests": source_digests,
            "expected_source_digests": EXPECTED_SOURCE_DIGESTS,
            "source_checks": source_checks,
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )
    write_jsonl(args.output_root / "dataset_manifest.jsonl", dataset_manifest)
    write_json(
        args.output_root / "checkpoint_manifest.json",
        {
            "run_id": RUN_ID,
            "status": "pass",
            "entries": [checkpoint_bindings[seed] for seed in EXPECTED_SEEDS],
        },
    )
    progress(args.output_root / "progress.jsonl", "k1s_audit_start")
    k1q_feature_rows = read_jsonl(args.k1q_root / "feature_manifest.jsonl")
    k1q_scorer_rows = read_jsonl(args.k1q_root / "scorer_manifest.jsonl")
    k1q_result_rows = read_jsonl(args.k1q_root / "results.jsonl")
    feature_rows, scorer_rows, result_rows = evaluate_k1s(
        datasets=datasets,
        models=models,
        checkpoint_bindings=checkpoint_bindings,
        k1q_feature_rows=k1q_feature_rows,
        k1q_scorer_rows=k1q_scorer_rows,
        k1q_result_rows=k1q_result_rows,
        batch_size=args.batch_size,
    )
    write_jsonl(args.output_root / "feature_manifest.jsonl", feature_rows)
    write_jsonl(args.output_root / "scorer_manifest.jsonl", scorer_rows)
    write_jsonl(args.output_root / "results.jsonl", result_rows)
    write_csv(args.output_root / "tap_attribution.csv", result_rows)
    k1r_logit_rows = read_jsonl(args.k1r_root / "controls.jsonl")
    gate = adjudicate_k1s(
        feature_rows=feature_rows,
        scorer_rows=scorer_rows,
        result_rows=result_rows,
        source_checks=source_checks,
        k1r_logit_rows=k1r_logit_rows,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "feature_rows": len(feature_rows),
        "scorer_rows": len(scorer_rows),
        "result_rows": len(result_rows),
    }
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    write_json(
        args.output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "status": gate["status"],
            "decision": gate["decision"],
            "tap_accessible_on_all_fresh_splits": gate[
                "tap_accessible_on_all_fresh_splits"
            ],
            "seed_results": gate["seed_results"],
            "next_action": gate["next_action"],
            "claim_scope": gate["claim_scope"],
        },
    )
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        feature_rows=len(feature_rows),
        scorer_rows=len(scorer_rows),
        result_rows=len(result_rows),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def source_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "k1q_gate": args.k1q_root / "gate.json",
        "k1q_dataset_manifest": args.k1q_root / "dataset_manifest.jsonl",
        "k1q_results": args.k1q_root / "results.jsonl",
        "k1q_feature_manifest": args.k1q_root / "feature_manifest.jsonl",
        "k1q_scorer_manifest": args.k1q_root / "scorer_manifest.jsonl",
        "k1q_validation": args.k1q_root / "validation.json",
        "k1r_plan": args.k1r_plan,
        "k1r_gate": args.k1r_root / "gate.json",
        "k1r_checkpoint_manifest": args.k1r_root / "checkpoint_manifest.json",
        "k1r_results": args.k1r_root / "results.jsonl",
        "k1r_controls": args.k1r_root / "controls.jsonl",
        "k1r_validation": args.k1r_root / "validation.json",
    }


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "feature_manifest.jsonl",
        "scorer_manifest.jsonl",
        "results.jsonl",
        "gate.json",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-S output root already contains run artifacts")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "source_artifact_paths"]
