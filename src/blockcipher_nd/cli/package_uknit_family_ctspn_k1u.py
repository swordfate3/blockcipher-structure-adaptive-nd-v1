from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1u import (
    EXPECTED_CACHE_CREATIONS,
    EXPECTED_RESULT_ROWS,
    RUN_ID,
)


RESULT_FILES = (
    "results.jsonl",
    "validation-plan.json",
    "validation.json",
    "gate.json",
    "summary.json",
    "history.csv",
)
SOURCE_FILES = (
    "configs/experiment/innovation1/innovation1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4.csv",
    "configs/remote/innovation1_uknit_k1u_position_residual_medium_65536_seed3_seed4_gpu1_20260728.json",
    "docs/experiments/innovation1-uknit-family-ctspn-position-residual-k1u-medium-plan.md",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and package the remote uKNIT K1-U result archive."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-commit-file", required=True, type=Path)
    parser.add_argument("--expected-source-commit-file", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = package_k1u_archive(
        run_root=args.run_root,
        source_root=args.source_root,
        source_commit_file=args.source_commit_file,
        expected_source_commit_file=args.expected_source_commit_file,
        archive_root=args.archive_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def package_k1u_archive(
    *,
    run_root: Path,
    source_root: Path,
    source_commit_file: Path,
    expected_source_commit_file: Path,
    archive_root: Path,
) -> dict[str, Any]:
    source_commit = _read_sha(source_commit_file)
    expected_source_commit = _read_sha(expected_source_commit_file)
    if not source_commit or source_commit != expected_source_commit:
        raise ValueError("source revision does not match the launch pin")

    results_root = run_root / "results"
    logs_root = run_root / "logs"
    checkpoints_root = run_root / "checkpoints"
    cache_root = run_root / "cache"
    result_rows = _read_jsonl(results_root / "results.jsonl")
    if len(result_rows) != EXPECTED_RESULT_ROWS:
        raise ValueError(
            f"expected {EXPECTED_RESULT_ROWS} K1-U result rows, got {len(result_rows)}"
        )

    checkpoints = sorted(checkpoints_root.glob("*.pt"))
    if len(checkpoints) != EXPECTED_RESULT_ROWS or any(
        path.stat().st_size == 0 for path in checkpoints
    ):
        raise ValueError("K1-U requires exactly six non-empty checkpoint files")

    metadata_paths = sorted(cache_root.rglob("metadata.json"))
    if len(metadata_paths) != EXPECTED_CACHE_CREATIONS:
        raise ValueError("K1-U requires exactly four cache metadata files")
    cache_entries = []
    for index, metadata_path in enumerate(metadata_paths):
        features_path = metadata_path.with_name("features.npy")
        labels_path = metadata_path.with_name("labels.npy")
        relative_cache_path = metadata_path.parent.relative_to(cache_root)
        split = (
            "validation" if "validation" in relative_cache_path.parts else "train"
        )
        if not features_path.is_file() or not labels_path.is_file():
            raise ValueError(f"incomplete cache payload: {metadata_path.parent}")
        if features_path.stat().st_size == 0 or labels_path.stat().st_size == 0:
            raise ValueError(f"empty cache payload: {metadata_path.parent}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("generation_chunk_size") != 1024:
            raise ValueError("cache chunk size drifted from 1024")
        if metadata.get("generation_workers") != 1:
            raise ValueError("cache worker count drifted from 1")
        cache_entries.append(
            {
                "index": index,
                "cache_path": metadata_path.parent.relative_to(run_root).as_posix(),
                "archive_path": (
                    (Path("validation_cache") / relative_cache_path).as_posix()
                    if split == "validation"
                    else None
                ),
                "metadata_sha256": _sha256(metadata_path),
                "features_bytes": features_path.stat().st_size,
                "features_sha256": _sha256(features_path),
                "labels_bytes": labels_path.stat().st_size,
                "labels_sha256": _sha256(labels_path),
                "total_rows": metadata.get("total_rows"),
                "input_bits": metadata.get("input_bits"),
                "split": split,
            }
        )

    if archive_root.exists():
        shutil.rmtree(archive_root)
    archive_root.mkdir(parents=True)
    for name in RESULT_FILES:
        _copy_required(results_root / name, archive_root / name)
    _copy_required(logs_root / "progress.jsonl", archive_root / "progress.jsonl")
    _copy_required(source_commit_file, archive_root / "git_revision.txt")
    _copy_required(
        expected_source_commit_file, archive_root / "source_expected_commit.txt"
    )
    for source_file in SOURCE_FILES:
        _copy_required(source_root / source_file, archive_root / Path(source_file).name)

    cache_metadata_root = archive_root / "cache_metadata"
    cache_metadata_root.mkdir()
    for index, metadata_path in enumerate(metadata_paths):
        shutil.copy2(
            metadata_path,
            cache_metadata_root / f"cache_{index:02d}_metadata.json",
        )

    archived_checkpoints = archive_root / "checkpoints"
    archived_checkpoints.mkdir()
    for path in checkpoints:
        shutil.copy2(path, archived_checkpoints / path.name)

    validation_entries = [
        entry for entry in cache_entries if entry["split"] == "validation"
    ]
    if len(validation_entries) != 2:
        raise ValueError("K1-U requires exactly two validation caches")
    for entry in validation_entries:
        source = run_root / entry["cache_path"]
        destination = archive_root / entry["archive_path"]
        destination.mkdir(parents=True)
        for name in ("features.npy", "labels.npy", "metadata.json"):
            shutil.copy2(source / name, destination / name)

    log_archive = archive_root / "logs"
    log_archive.mkdir()
    for path in sorted(logs_root.glob(f"{RUN_ID}_*")):
        if path.is_file():
            shutil.copy2(path, log_archive / path.name)

    checkpoint_manifest = {
        "count": len(checkpoints),
        "checkpoints": [
            {
                "path": path.relative_to(run_root).as_posix(),
                "archive_path": (Path("checkpoints") / path.name).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in checkpoints
        ],
    }
    cache_manifest = {"count": len(cache_entries), "caches": cache_entries}
    run_manifest = {
        "run_id": RUN_ID,
        "source_commit": source_commit,
        "result_rows": len(result_rows),
        "checkpoint_count": len(checkpoints),
        "cache_count": len(cache_entries),
        "archived_checkpoint_count": len(checkpoints),
        "archived_validation_cache_count": len(validation_entries),
        "result_sync": "verified_result_branch_with_local_raw_fallback",
        "claim_scope": (
            "remote 65536/class uKNIT r5 medium diagnostic; not formal, "
            "paper-scale, attack, SOTA, transfer, or universal-SPN evidence"
        ),
    }
    _write_json(archive_root / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(archive_root / "cache_manifest.json", cache_manifest)
    _write_json(archive_root / "run_manifest.json", run_manifest)
    (archive_root / "plot_deferred.marker").write_text(
        "plot_deferred_to_verified_local_retrieval\n", encoding="utf-8"
    )
    (archive_root / "visual_qa_pending.marker").write_text(
        "visual_qa_pending\n", encoding="utf-8"
    )
    (archive_root / ".gitattributes").write_text("* -text\n", encoding="utf-8")

    artifact_entries = [
        {
            "path": path.relative_to(archive_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(archive_root.rglob("*"))
        if path.is_file() and path.name not in {"artifact_manifest.json", "SHA256SUMS"}
    ]
    _write_json(
        archive_root / "artifact_manifest.json",
        {"count": len(artifact_entries), "artifacts": artifact_entries},
    )
    checksum_files = [
        path
        for path in sorted(archive_root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (archive_root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(archive_root).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )
    return run_manifest


def _copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return values


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "package_k1u_archive", "parse_args"]
