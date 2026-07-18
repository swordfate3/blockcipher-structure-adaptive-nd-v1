from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from blockcipher_nd.tasks.innovation2.atm_multicoordinate_support import (
    multicoordinate_support_pool,
    run_multicoordinate_support_phase_a,
)
from blockcipher_nd.tasks.innovation2.atm_native_sat_witness_provider import (
    NativeSatProviderConfig,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E61 ATM support Phase A.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--e60-gate", required=True, type=Path)
    parser.add_argument("--wall-clock-cap-seconds", type=int, default=60)
    parser.add_argument("--projected-key-cap", type=int, default=1 << 12)
    parser.add_argument("--trail-model-cap", type=int, default=1 << 16)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--supports-output", type=Path)
    parser.add_argument("--relations-output", type=Path)
    parser.add_argument("--progress-output", type=Path)
    return parser.parse_args(argv)


def evaluate_phase_a(
    config: NativeSatProviderConfig,
    *,
    e60_gate: dict[str, Any],
    model: dict[str, Any] | None,
    rows: Sequence[dict[str, Any]],
    relation_payload: dict[str, Any] | None,
    planned_coordinates: int,
    worker_status: str,
    wall_clock_cap_seconds: int,
) -> dict[str, Any]:
    exact_rows = [row for row in rows if row.get("support", {}).get("status") == "exact"]
    key_dependent_rows = [
        row
        for row in exact_rows
        if row.get("support", {}).get("key_dependent") is True
    ]
    unknown_coordinates = planned_coordinates - len(exact_rows)
    positives = (
        relation_payload.get("positive_relations", []) if relation_payload else []
    )
    pairs = (
        relation_payload.get("matched_relation_pairs", []) if relation_payload else []
    )
    all_saved_masks_replay = bool(exact_rows) and all(
        row["support"].get("replay_verified") is True for row in exact_rows
    )
    all_positive_constant_replays_exact = bool(pairs) and all(
        pair.get("positive_constant_replay", {}).get("status") == "exact"
        for pair in pairs
    )
    all_negative_witnesses_replay_odd = bool(pairs) and all(
        pair.get("negative_witness_replay", {}).get("status") == "exact"
        and pair["negative_witness_replay"].get("xor_parity") == 1
        for pair in pairs
    )
    source_checks = {
        "e60_gate_is_width_hold": e60_gate.get("status") == "hold"
        and e60_gate.get("decision")
        == "innovation2_atm_r2_cone_matched_panel_width_not_ready",
        "e60_scalar_validation_is_16": e60_gate.get("metrics", {}).get(
            "scalar_validated_constant_rows"
        )
        == 16,
        "e60_requests_multicoordinate_design": e60_gate.get("next_action", {}).get(
            "multi_coordinate_design"
        )
        is True,
        "model_rounds_are_two": bool(model and model.get("rounds") == 2),
        "model_has_three_key_additions": bool(
            model and model.get("key_additions") == 3
        ),
        "model_has_192_independent_key_variables": bool(
            model
            and model.get("key_variables") == 192
            and model.get("key_model") == "independent_round_keys"
        ),
        "frozen_coordinate_pool_has_240_rows": len(multicoordinate_support_pool())
        == planned_coordinates
        == 240,
    }
    readiness_checks = {
        "completed_coordinates_at_least_64": len(rows) >= 64,
        "unknown_fraction_at_most_0p25": unknown_coordinates
        / planned_coordinates
        <= 0.25,
        "exact_key_dependent_supports_at_least_16": len(key_dependent_rows) >= 16,
        "all_saved_odd_masks_replay_odd": all_saved_masks_replay,
        "low_weight_positive_relations_at_least_4": len(positives) >= 4,
        "matched_strict_negative_pairs_at_least_4": len(pairs) >= 4,
        "all_positive_constant_replays_exact": all_positive_constant_replays_exact,
        "all_negative_relation_witnesses_replay_odd": (
            all_negative_witnesses_replay_odd
        ),
    }
    metrics = {
        "planned_coordinates": planned_coordinates,
        "completed_coordinates": len(rows),
        "exact_coordinates": len(exact_rows),
        "unknown_coordinates": unknown_coordinates,
        "exact_constant_coordinates": len(exact_rows) - len(key_dependent_rows),
        "exact_key_dependent_coordinates": len(key_dependent_rows),
        "low_weight_positive_relations": len(positives),
        "matched_strict_negative_pairs": len(pairs),
        "worker_status": worker_status,
    }
    if not all(source_checks.values()):
        status = "fail"
        decision = "innovation2_atm_r2_multicoordinate_support_protocol_invalid"
        action = "repair E60 ownership, coordinate pool, or independent-key model"
    elif worker_status != "completed" or len(rows) < 64:
        status = "hold"
        decision = "innovation2_atm_r2_multicoordinate_support_runtime_not_ready"
        action = "stop this exact support route without raising caps or using remote GPU"
    elif not all(readiness_checks.values()):
        status = "hold"
        decision = "innovation2_atm_r2_multicoordinate_support_width_not_ready"
        action = "do not train RCCA; strict cancellation label width is insufficient"
    else:
        status = "pass"
        decision = "innovation2_atm_r2_multicoordinate_support_phase_b_ready"
        action = "build the matched 256-per-class Phase B label atlas before training"
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "readiness_checks": readiness_checks,
        "metrics": metrics,
        "wall_clock_cap_seconds": wall_clock_cap_seconds,
        "projected_key_cap": config.projected_key_cap,
        "trail_model_cap": config.trail_model_cap,
        "claim_scope": (
            "PRESENT two-round independent-round-key coordinate key-polynomial "
            "support and GF(2) cancellation audit; not actual PRESENT-80 labels, "
            "neural training, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "phase_b_label_atlas": status == "pass",
            "training": False,
            "remote_scale": False,
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_atm_r2_multicoordinate_support_phase_a",
            "metric": key,
            "value": value,
            "status": status,
            "decision": decision,
            "training_performed": False,
        }
        for key, value in metrics.items()
        if key != "worker_status"
    ]
    return {"gate": gate, "result_rows": result_rows}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return _worker_main(args)
    config = NativeSatProviderConfig(
        run_id=args.run_id,
        projected_key_cap=args.projected_key_cap,
        trail_model_cap=args.trail_model_cap,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    model_output = args.output_root / "model.json"
    supports_output = args.output_root / "coordinate_supports.jsonl"
    relations_output = args.output_root / "relation_candidates.json"
    for path in (model_output, supports_output, relations_output):
        path.unlink(missing_ok=True)
    queries = list(multicoordinate_support_pool())
    _write_json(args.output_root / "query_plan.json", {"queries": queries})
    _write_progress(
        progress,
        "worker_start",
        {
            "planned_coordinates": len(queries),
            "wall_clock_cap_seconds": args.wall_clock_cap_seconds,
        },
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-id",
        args.run_id,
        "--output-root",
        str(args.output_root),
        "--atm-root",
        str(args.atm_root),
        "--e60-gate",
        str(args.e60_gate),
        "--projected-key-cap",
        str(config.projected_key_cap),
        "--trail-model-cap",
        str(config.trail_model_cap),
        "--worker",
        "--model-output",
        str(model_output),
        "--supports-output",
        str(supports_output),
        "--relations-output",
        str(relations_output),
        "--progress-output",
        str(progress),
    ]
    worker_status = "failed"
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
            timeout=args.wall_clock_cap_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        if completed.returncode == 0:
            worker_status = "completed"
    except subprocess.TimeoutExpired as exc:
        worker_status = "timeout"
        stdout = _decode_timeout(exc.stdout)
        stderr = _decode_timeout(exc.stderr)
    (args.output_root / "worker.stdout.log").write_text(stdout, encoding="utf-8")
    (args.output_root / "worker.stderr.log").write_text(stderr, encoding="utf-8")
    model = _load_json(model_output)
    rows = _load_jsonl(supports_output)
    relations = _load_json(relations_output)
    e60_gate = _load_json(args.e60_gate) or {}
    evaluation = evaluate_phase_a(
        config,
        e60_gate=e60_gate,
        model=model,
        rows=rows,
        relation_payload=relations,
        planned_coordinates=len(queries),
        worker_status=worker_status,
        wall_clock_cap_seconds=args.wall_clock_cap_seconds,
    )
    gate = evaluation["gate"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_atm_r2_multicoordinate_support_phase_a",
        "config": serializable_config(config),
        "wall_clock_cap_seconds": args.wall_clock_cap_seconds,
        "external_source_root": str(args.atm_root),
        "source_e60_gate": str(args.e60_gate),
        "rounds": 2,
        "key_model": "independent_round_keys",
        "coordinate_pool": "input cell0 monomials 0..15 x output cell0 monomials 1..15",
        "support_is_label_oracle_only": True,
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "model": model,
        "worker_status": worker_status,
        "gate": gate,
        "relations": relations,
        "result_rows": evaluation["result_rows"],
    }
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress,
        "run_done",
        {
            "status": gate["status"],
            "decision": gate["decision"],
            "completed_coordinates": len(rows),
        },
    )
    print(json.dumps({"gate": gate}, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _worker_main(args: argparse.Namespace) -> int:
    required = (
        args.model_output,
        args.supports_output,
        args.relations_output,
        args.progress_output,
    )
    if any(path is None for path in required):
        raise ValueError("worker artifact paths are required")
    run_multicoordinate_support_phase_a(
        args.atm_root,
        projected_key_cap=args.projected_key_cap,
        trail_model_cap=args.trail_model_cap,
        model_output=args.model_output,
        supports_output=args.supports_output,
        relations_output=args.relations_output,
        progress_output=args.progress_output,
    )
    return 0


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        payload
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for payload in [json.loads(line)]
        if isinstance(payload, dict)
    ]


def _decode_timeout(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "event": event,
                    **payload,
                },
                sort_keys=True,
            )
            + "\n"
        )


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
