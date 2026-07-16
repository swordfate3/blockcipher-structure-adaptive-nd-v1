from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.high_round_integral_experiment import (
    HighRoundIntegralExperimentConfig,
    adjudicate_high_round_integral,
)


POLICY_VERSION = "innovation2_r8_bridge_candidate_only_after_anchor_layout_audit_v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-adjudicate retrieved Innovation 2 high-round integral artifacts "
            "with the current local gate policy."
        )
    )
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--remote-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invalidate-anchor-layout", action="store_true")
    parser.add_argument("--expected-source-commit", default=None)
    return parser.parse_args(argv)


def readjudicate_retrieved_artifacts(
    artifacts: Path,
    remote_config_path: Path,
    *,
    invalidate_anchor_layout: bool,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    remote_config = _load_object(remote_config_path)
    experiment = remote_config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("remote config is missing the experiment object")
    rows = _load_jsonl(artifacts / "results.jsonl")
    dataset_summary = _load_object(artifacts / "dataset_summary.json")
    fixed_baselines = _load_object(artifacts / "fixed_baselines.json")
    config = HighRoundIntegralExperimentConfig(
        run_id=str(remote_config["run_id"]),
        output_root=artifacts,
        cache_root=Path(str(remote_config["dataset_cache_root"])),
        rounds=int(experiment["rounds"]),
        train_rows=int(experiment["train_total_rows"]),
        validation_rows=int(experiment["validation_total_rows"]),
        test_rows=int(experiment["test_total_rows"]),
        multiset_count=int(experiment["multisets_per_sample"]),
        epochs=int(experiment["epochs"]),
        batch_size=int(experiment["batch_size"]),
        seed=int(experiment["seed"]),
        base_channels=int(experiment["base_channels"]),
        head_bits=int(experiment["head_bits"]),
        block_count=int(experiment["block_count"]),
        dropout=float(experiment["dropout"]),
        learning_rate=float(experiment["learning_rate"]),
        weight_decay=float(experiment["weight_decay"]),
        device="cuda",
        cache_chunk_size=int(remote_config["dataset_cache_chunk_size"]),
        gate_mode=str(experiment["gate_mode"]),
    )
    gate = adjudicate_high_round_integral(
        config,
        rows=rows,
        dataset_summary=dataset_summary,
        fixed_baselines=fixed_baselines,
    )
    source_revision_path = artifacts / "git_revision.txt"
    source_revision = (
        source_revision_path.read_text(encoding="utf-8").strip()
        if source_revision_path.is_file()
        else ""
    )
    source_revision_matches = (
        expected_source_commit is None or source_revision == expected_source_commit
    )
    if not source_revision_matches:
        gate["status"] = "fail"
        gate["decision"] = (
            "innovation2_high_round_integral_readjudication_source_mismatch"
        )
        gate["next_action"] = (
            "Reject plan alignment and retrieve artifacts from the exact frozen "
            "source commit before interpreting any metric."
        )
    exclusions = []
    if invalidate_anchor_layout:
        exclusions.append(
            {
                "role": "anchor",
                "reason": "global_two_multiset_reshape_did_not_preserve_per_multiset_paper_grid",
                "allowed_use": "historical_run_record_only",
            }
        )
    return {
        **gate,
        "readjudication": {
            "policy_version": POLICY_VERSION,
            "source_remote_gate": str(artifacts / "gate.json"),
            "remote_config": str(remote_config_path),
            "anchor_layout_invalidated": invalidate_anchor_layout,
            "evidence_exclusions": exclusions,
            "candidate_only_absolute_signal_gate": True,
            "source_revision": source_revision,
            "expected_source_commit": expected_source_commit,
            "source_revision_matches_expected": source_revision_matches,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = readjudicate_retrieved_artifacts(
        args.artifacts,
        args.remote_config,
        invalidate_anchor_layout=args.invalidate_anchor_layout,
        expected_source_commit=args.expected_source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 1 if report["status"] == "fail" else 0


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
