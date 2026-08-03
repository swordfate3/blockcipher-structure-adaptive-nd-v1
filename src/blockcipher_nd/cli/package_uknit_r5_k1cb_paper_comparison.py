from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from blockcipher_nd.tasks.innovation1.uknit_r5_invariant_autond_closeout_k1ca import (
    RUN_ID as K1CA_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_r5_published_comparison_k1cb import (
    EXPECTED_RESULT_ROWS,
    RUN_ID,
)


RESULT_FILES = (
    "results.jsonl",
    "validation-plan.json",
    "source_cache_audit.json",
    "validation.json",
    "gate.json",
    "summary.json",
    "history.csv",
)
SOURCE_FILES = (
    "configs/experiment/innovation1/innovation1_uknit_r5_k1cb_published_comparison_262144_seed3_seed4.csv",
    "configs/remote/innovation1_uknit_k1cb_published_comparison_262144_seed3_seed4_gpu0_20260803.json",
    "docs/experiments/innovation1-uknit-r5-k1cb-published-network-paper-comparison-plan.md",
)
SOURCE_ARCHIVE_NAMES = ("plan.csv", "remote_config.json", "experiment_plan.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the remote uKNIT K1-CB paper-comparison archive."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-k1ca-root", required=True, type=Path)
    parser.add_argument("--source-commit-file", required=True, type=Path)
    parser.add_argument("--expected-source-commit-file", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_commit = _read_sha(args.source_commit_file)
    expected_commit = _read_sha(args.expected_source_commit_file)
    if not source_commit or source_commit != expected_commit:
        raise ValueError("source revision does not match the launch pin")

    results_root = args.run_root / "results"
    logs_root = args.run_root / "logs"
    checkpoints_root = args.run_root / "checkpoints"
    rows = _read_jsonl(results_root / "results.jsonl")
    if len(rows) != EXPECTED_RESULT_ROWS:
        raise ValueError(f"expected {EXPECTED_RESULT_ROWS} results, got {len(rows)}")
    checkpoints = sorted(checkpoints_root.glob("*.pt"))
    if len(checkpoints) != EXPECTED_RESULT_ROWS or any(
        path.stat().st_size == 0 for path in checkpoints
    ):
        raise ValueError("K1-CB requires six non-empty checkpoints")

    source_audit = _read_json(results_root / "source_cache_audit.json")
    if source_audit.get("status") != "pass" or not all(
        source_audit.get("checks", {}).values()
    ):
        raise ValueError("K1-CB source cache audit did not pass")

    source_results_root = args.source_k1ca_root / "results"
    source_archive_root = (
        args.source_k1ca_root / "source" / "results_archive" / K1CA_RUN_ID
    )
    source_files = {
        "gate.json": source_results_root / "gate.json",
        "results.jsonl": source_results_root / "results.jsonl",
        "cache_manifest.json": source_archive_root / "cache_manifest.json",
        "run_manifest.json": source_archive_root / "run_manifest.json",
    }
    if any(not path.is_file() for path in source_files.values()):
        missing = [name for name, path in source_files.items() if not path.is_file()]
        raise ValueError(f"missing K1-CA source archive files: {missing}")

    if args.archive_root.exists():
        shutil.rmtree(args.archive_root)
    args.archive_root.mkdir(parents=True)
    for name in RESULT_FILES:
        _copy(results_root / name, args.archive_root / name)
    _copy(logs_root / "progress.jsonl", args.archive_root / "progress.jsonl")
    _copy(args.source_commit_file, args.archive_root / "git_revision.txt")
    _copy(
        args.expected_source_commit_file,
        args.archive_root / "source_expected_commit.txt",
    )
    for source, archive_name in zip(SOURCE_FILES, SOURCE_ARCHIVE_NAMES):
        _copy(args.source_root / source, args.archive_root / archive_name)

    source_archive = args.archive_root / "source_k1ca"
    source_archive.mkdir()
    for name, path in source_files.items():
        _copy(path, source_archive / name)

    checkpoint_archive = args.archive_root / "checkpoints"
    checkpoint_archive.mkdir()
    checkpoint_entries: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints):
        archive_name = f"checkpoint_{index:02d}.pt"
        destination = checkpoint_archive / archive_name
        shutil.copy2(checkpoint, destination)
        checkpoint_entries.append(
            {
                "source_path": checkpoint.relative_to(args.run_root).as_posix(),
                "archive_path": f"checkpoints/{archive_name}",
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )

    run_manifest = {
        "run_id": RUN_ID,
        "source_run_id": K1CA_RUN_ID,
        "source_commit": source_commit,
        "result_rows": len(rows),
        "checkpoint_count": len(checkpoints),
        "new_cache_count": 0,
        "source_cache_reuses": 12,
        "result_sync": "verified_result_branch_with_local_raw_fallback",
        "claim_scope": (
            "same-data two-seed 262144/class uKNIT r5 project-protocol paper "
            "comparison; published architecture adaptations, not exact reproduction "
            "or formal million-scale evidence"
        ),
    }
    _write_json(
        args.archive_root / "checkpoint_manifest.json",
        {"count": len(checkpoint_entries), "checkpoints": checkpoint_entries},
    )
    _write_json(args.archive_root / "run_manifest.json", run_manifest)
    (args.archive_root / "plot_deferred.marker").write_text(
        "plot_deferred_to_verified_local_retrieval\n", encoding="utf-8"
    )
    (args.archive_root / "visual_qa_pending.marker").write_text(
        "visual_qa_pending\n", encoding="utf-8"
    )
    (args.archive_root / ".gitattributes").write_text("* -text\n", encoding="utf-8")

    artifacts = [
        {
            "path": path.relative_to(args.archive_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(args.archive_root.rglob("*"))
        if path.is_file() and path.name not in {"artifact_manifest.json", "SHA256SUMS"}
    ]
    _write_json(
        args.archive_root / "artifact_manifest.json",
        {"count": len(artifacts), "artifacts": artifacts},
    )
    checksum_files = [
        path
        for path in sorted(args.archive_root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.archive_root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(args.archive_root).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _read_sha(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return ""
    return (
        value
        if len(value) == 40 and all(char in "0123456789abcdef" for char in value)
        else ""
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"missing or empty archive source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
