from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from blockcipher_nd.tasks.innovation2.present_open_3sdp_exact_oracle import (
    AUDIT_VECTOR_CHECKS,
    ExactOracleConfig,
    audit_sbox_transition_parity,
    build_multi_mask_fixtures,
    build_present_exact_anf_snapshots,
    build_strict_fixtures,
    evaluate_exact_oracle_readiness,
    serializable_config,
    validate_exact_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E53-A PRESENT exact-ANF and cancellation fixtures."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--fixtures-per-class", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExactOracleConfig(
        run_id=args.run_id,
        mode=args.mode,
        rounds=tuple(args.rounds),
        fixtures_per_class=args.fixtures_per_class,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    _write_progress(
        progress,
        "run_start",
        {
            "run_id": config.run_id,
            "mode": config.mode,
            "training": False,
            "five_round_subset": False,
        },
    )

    start = perf_counter()
    snapshots = build_present_exact_anf_snapshots(config.rounds)
    snapshot_seconds = perf_counter() - start
    snapshot_metrics = {
        str(rounds): {
            "minimum": min(len(polynomial) for polynomial in outputs),
            "maximum": max(len(polynomial) for polynomial in outputs),
            "total": sum(len(polynomial) for polynomial in outputs),
        }
        for rounds, outputs in snapshots.items()
    }
    _write_progress(
        progress,
        "exact_anf_ready",
        {"seconds": snapshot_seconds, "round_metrics": snapshot_metrics},
    )

    fixtures: list[dict[str, Any]] = []
    vector_checks: dict[int, dict[str, Any]] = {}
    for rounds in config.rounds:
        outputs = snapshots[rounds]
        fixtures.extend(
            build_strict_fixtures(
                outputs,
                rounds=rounds,
                fixtures_per_class=config.fixtures_per_class,
                seed=config.seed,
            )
        )
        fixtures.extend(
            build_multi_mask_fixtures(
                outputs,
                rounds=rounds,
                count=4 if config.mode == "audit" else 2,
                seed=config.seed + rounds * 100,
            )
        )
        vector_checks[rounds] = validate_exact_outputs(
            outputs,
            rounds=rounds,
            count=AUDIT_VECTOR_CHECKS[rounds]
            if config.mode == "audit"
            else 2,
            seed=config.seed,
        )
    _write_progress(
        progress,
        "exact_fixtures_ready",
        {
            "fixtures": len(fixtures),
            "positive": sum(row["status"] == "positive" for row in fixtures),
            "negative": sum(row["status"] == "negative" for row in fixtures),
        },
    )

    transition = audit_sbox_transition_parity()
    glpk = _probe_sage_glpk()
    _write_progress(
        progress,
        "transition_and_glpk_ready",
        {
            "existence_only_false_positives": transition["metrics"][
                "existence_only_false_positives"
            ],
            "glpk_backend": glpk["backend"],
            "binary_fixture_passed": glpk["binary_fixture_passed"],
        },
    )
    evaluation = evaluate_exact_oracle_readiness(
        config,
        snapshots=snapshots,
        fixtures=fixtures,
        vector_checks=vector_checks,
        transition=transition,
        glpk=glpk,
    )
    gate = evaluation["gate"]

    provider_manifest = {
        "run_id": config.run_id,
        "providers": [
            {
                "provider_id": "exact_full_present_anf_oracle",
                "status": "ready"
                if gate["decision"]
                == "innovation2_present_r5_open_3sdp_exact_oracle_ready"
                else "not_ready",
                "scope": "PRESENT-80 rounds one and two only",
                "variables": {
                    "plaintext": 64,
                    "key": 80,
                    "total": 144,
                },
                "semantics": (
                    "full Boolean ANF with key and inactive plaintext variables "
                    "symbolic; cube superpoly extracted exactly"
                ),
                "round_metrics": snapshot_metrics,
            },
            {
                "provider_id": "sage_glpk_3sdp_candidate",
                "status": "runtime_ready_enumerator_not_implemented"
                if glpk["binary_fixture_passed"]
                else "runtime_not_ready",
                "runtime": glpk,
                "exact_oracle_agreement": False,
                "five_round_subset_executed": False,
            },
        ],
        "control": {
            "provider_id": "existence_only_without_gf2_parity",
            "status": "rejected",
            "false_positive_transitions": transition["metrics"][
                "existence_only_false_positives"
            ],
        },
        "training_performed": False,
        "remote_scale": False,
    }
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_r5_open_3sdp_exact_oracle",
        "config": serializable_config(config),
        "cipher": "PRESENT-80",
        "target_rounds": 5,
        "executed_rounds": list(config.rounds),
        "target": (
            "masked cube XOR is zero for every key and inactive plaintext offset"
        ),
        "training_performed": False,
        "remote_scale": False,
        "five_round_subset_executed": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "provider_manifest": provider_manifest,
        "snapshot_seconds": snapshot_seconds,
        "metrics": evaluation["metrics"],
        "gate": gate,
        "result_rows": evaluation["result_rows"],
        "vector_checks": vector_checks,
        "transition": {
            "metrics": transition["metrics"],
            "checks": transition["checks"],
            "cancellation_examples": transition["cancellation_examples"],
        },
    }
    certificates = [
        row for row in fixtures if row["status"] == "positive"
    ] + [
        {
            "certificate_type": "sbox_even_trail_cancellation",
            **row,
        }
        for row in transition["cancellation_examples"]
    ]
    witnesses = [row for row in fixtures if row["status"] == "negative"]
    _write_jsonl(args.output_root / "fixtures.jsonl", fixtures)
    _write_jsonl(args.output_root / "certificates.jsonl", certificates)
    _write_jsonl(args.output_root / "witnesses.jsonl", witnesses)
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
    _write_json(args.output_root / "provider_manifest.json", provider_manifest)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps(
            {
                "gate": gate,
                "output_root": str(args.output_root),
                "snapshot_seconds": snapshot_seconds,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if gate["status"] == "fail" else 0


def _probe_sage_glpk() -> dict[str, Any]:
    sage = shutil.which("sage")
    result: dict[str, Any] = {
        "sage_executable": sage,
        "sage_version": None,
        "backend": None,
        "objective": None,
        "solution": None,
        "binary_fixture_passed": False,
        "error": None,
    }
    if sage is None:
        result["error"] = "sage executable unavailable"
        return result
    version = subprocess.run(
        [sage, "--version"], capture_output=True, text=True, check=False, timeout=30
    )
    result["sage_version"] = (version.stdout or version.stderr).strip()
    code = (
        "import json; "
        "p=MixedIntegerLinearProgram(maximization=False); "
        "x=p.new_variable(binary=True); "
        "p.add_constraint(x[0]>=1); p.set_objective(x[0]); "
        "objective=p.solve(); solution=p.get_values(x[0]); "
        "print(json.dumps({'backend':type(p.get_backend()).__name__, "
        "'objective':objective, 'solution':solution}))"
    )
    probe = subprocess.run(
        [sage, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={**os.environ, "HOME": "/tmp"},
    )
    if probe.returncode:
        lines = (probe.stderr or probe.stdout).strip().splitlines()
        result["error"] = lines[-1] if lines else "unknown Sage/GLPK error"
        return result
    try:
        payload = json.loads(probe.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        result["error"] = f"invalid Sage/GLPK probe output: {exc}"
        return result
    result.update(payload)
    result["binary_fixture_passed"] = (
        result["backend"] == "GLPKBackend"
        and float(result["objective"]) == 1.0
        and float(result["solution"]) == 1.0
    )
    return result


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
