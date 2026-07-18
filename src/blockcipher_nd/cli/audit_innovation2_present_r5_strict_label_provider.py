from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_r5_strict_label_provider_coverage import (
    StrictLabelCoverageConfig,
    evaluate_coverage,
    inspect_claasp_p1,
    serializable_config,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    build_raw_atlas,
    make_output_masks,
    make_structures,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E52 PRESENT-80 r5 strict-label provider coverage."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--claasp-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--structure-count", type=int, default=96)
    parser.add_argument("--witness-keys", type=int, default=16)
    parser.add_argument("--offsets-per-structure", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = StrictLabelCoverageConfig(
        run_id=args.run_id,
        mode=args.mode,
        structure_count=args.structure_count,
        witness_keys=args.witness_keys,
        offsets_per_structure=args.offsets_per_structure,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    _write_progress(
        progress,
        "run_start",
        {"run_id": config.run_id, "mode": config.mode, "training": False},
    )

    structures = make_structures(config)
    masks = make_output_masks()
    _write_progress(
        progress,
        "fixture_ready",
        {"structures": len(structures), "masks": len(masks)},
    )
    raw = build_raw_atlas(config, structures, masks)
    counts = {
        status: sum(row["status"] == status for row in raw["rows"])
        for status in ("positive", "negative", "unknown")
    }
    _write_progress(progress, "p0_complete", counts)

    actual_commit = _git_head(args.claasp_root)
    runtime = _probe_sage_runtime(args.claasp_root)
    p1 = inspect_claasp_p1(
        args.claasp_root,
        actual_commit=actual_commit,
        runtime=runtime,
    )
    _write_progress(
        progress,
        "p1_feasibility_complete",
        {
            "provider_status": p1["status"],
            "execution_available": p1["runtime"]["execution_available"],
        },
    )
    evaluation = evaluate_coverage(config, structures, masks, raw, p1)
    gate = evaluation["gate"]

    p0_manifest = {
        "provider_id": "P0_active_variable_support_overapprox",
        "status": "completed_insufficient"
        if not gate["provider_checks"]["p0_coverage_sufficient"]
        else "completed_sufficient",
        "semantics": (
            "sound active-variable ANF support overapproximation; absence of the "
            "full cube monomial proves zero XOR for every key and inactive offset"
        ),
        "counts": counts,
        "support_size_minimum": evaluation["metrics"]["support_size_minimum"],
        "support_size_maximum": evaluation["metrics"]["support_size_maximum"],
        "support_saturation_fraction": evaluation["metrics"][
            "support_saturation_fraction"
        ],
    }
    provider_manifest = {
        "run_id": config.run_id,
        "provider_order": [p0_manifest["provider_id"], p1["provider_id"]],
        "providers": [p0_manifest, p1],
        "selection_rule": (
            "P0 first; P1 only after P0 coverage failure; P2 is not selected until "
            "P1 has an executable semantic fixture or is explicitly closed"
        ),
        "training_performed": False,
    }
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_r5_strict_label_provider_coverage",
        "config": serializable_config(config),
        "cipher": "PRESENT-80",
        "rounds": config.rounds,
        "active_dimension": 8,
        "output_query": "64-bit nonzero linear mask",
        "positive_semantics": (
            "masked cube XOR is zero for every 80-bit key and every inactive offset"
        ),
        "negative_semantics": (
            "a concrete key and inactive offset produce masked cube XOR one"
        ),
        "unknown_semantics": (
            "the current provider neither proves zero nor finds a concrete witness"
        ),
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "provider_manifest": provider_manifest,
        "metrics": evaluation["metrics"],
        "gate": gate,
        "result_rows": evaluation["result_rows"],
    }

    _write_csv(args.output_root / "labels.csv", raw["rows"])
    _write_jsonl(
        args.output_root / "certificates.jsonl",
        [row for row in raw["rows"] if row["status"] == "positive"],
    )
    _write_jsonl(
        args.output_root / "witnesses.jsonl",
        [row for row in raw["rows"] if row["status"] == "negative"],
    )
    _write_json(
        args.output_root / "structures.json",
        {
            "structures": [
                {
                    "index": structure.index,
                    "structure_id": structure.structure_id,
                    "role": structure.role,
                    "active_bits": list(structure.active_bits),
                    "active_mask_hex": f"0x{structure.active_mask:016X}",
                    "split": "validation" if not structure.index % 4 else "train",
                }
                for structure in structures
            ]
        },
    )
    _write_json(
        args.output_root / "masks.json",
        {
            "masks": [
                {
                    "index": mask.index,
                    "mask_id": mask.mask_id,
                    "family": mask.family,
                    "mask_hex": f"0x{mask.value:016X}",
                    "bits": list(mask.bits),
                }
                for mask in masks
            ]
        },
    )
    _write_json(args.output_root / "provider_manifest.json", provider_manifest)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
    if gate["status"] == "pass":
        _write_csv(args.output_root / "matched_contrast.csv", evaluation["matched_rows"])
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
                "providers": provider_manifest["providers"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if gate["status"] == "fail" else 0


def _probe_sage_runtime(claasp_root: Path) -> dict[str, Any]:
    sage = shutil.which("sage")
    runtime: dict[str, Any] = {
        "sage_executable": sage,
        "sage_version": None,
        "sage_modules": {},
        "claasp_present_importable": False,
        "claasp_present_import_error": None,
        "gurobi_license_status": "not_checked_package_missing",
        "relevant_docker_image_found": False,
    }
    if sage is None:
        runtime["claasp_present_import_error"] = "sage executable unavailable"
        return runtime
    version = subprocess.run(
        [sage, "--version"], capture_output=True, text=True, check=False, timeout=30
    )
    runtime["sage_version"] = (version.stdout or version.stderr).strip()
    env = {**os.environ, "HOME": "/tmp", "PYTHONPATH": str(claasp_root)}
    module_probe = subprocess.run(
        [
            sage,
            "-python",
            "-c",
            (
                "import importlib.util,json; "
                "print(json.dumps({n: bool(importlib.util.find_spec(n)) "
                "for n in ['sage','bitstring','gurobipy']}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=env,
    )
    try:
        runtime["sage_modules"] = json.loads(module_probe.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        runtime["sage_modules"] = {}
    import_probe = subprocess.run(
        [
            sage,
            "-python",
            "-c",
            (
                "from claasp.ciphers.block_ciphers.present_block_cipher import "
                "PresentBlockCipher; PresentBlockCipher(number_of_rounds=5)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=env,
    )
    runtime["claasp_present_importable"] = import_probe.returncode == 0
    if import_probe.returncode:
        error_lines = (import_probe.stderr or import_probe.stdout).strip().splitlines()
        runtime["claasp_present_import_error"] = error_lines[-1] if error_lines else "unknown"
    elif runtime["sage_modules"].get("gurobipy"):
        runtime["gurobi_license_status"] = "not_verified"
    return runtime


def _git_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result.stdout.strip() if result.returncode == 0 else None


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
