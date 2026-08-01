from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from blockcipher_nd.tasks.innovation1.uknit_r5_architecture_medium_k1bt import (
    EXPECTED_CACHE_CREATIONS,
    EXPECTED_RESULT_ROWS,
    RUN_ID,
)


RESULT_FILES = ("results.jsonl", "validation-plan.json", "validation.json", "gate.json", "summary.json", "history.csv")
SOURCE_FILES = (
    "configs/experiment/innovation1/innovation1_uknit_r5_neural_architecture_medium_k1bt_16pair_65536_seed3_seed4.csv",
    "configs/remote/innovation1_uknit_k1bt_architecture_medium_65536_seed3_seed4_gpu1_20260731.json",
    "docs/experiments/innovation1-uknit-r5-neural-architecture-medium-k1bt-plan.md",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the remote uKNIT K1-BT archive.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-commit-file", required=True, type=Path)
    parser.add_argument("--expected-source-commit-file", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = package_archive(
        run_root=args.run_root,
        source_root=args.source_root,
        source_commit_file=args.source_commit_file,
        expected_source_commit_file=args.expected_source_commit_file,
        archive_root=args.archive_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def package_archive(
    *, run_root: Path, source_root: Path, source_commit_file: Path,
    expected_source_commit_file: Path, archive_root: Path,
) -> dict[str, Any]:
    source_commit = _read_sha(source_commit_file)
    expected_commit = _read_sha(expected_source_commit_file)
    if not source_commit or source_commit != expected_commit:
        raise ValueError("source revision does not match the launch pin")
    results_root = run_root / "results"
    logs_root = run_root / "logs"
    checkpoints_root = run_root / "checkpoints"
    cache_root = run_root / "cache"
    rows = _read_jsonl(results_root / "results.jsonl")
    if len(rows) != EXPECTED_RESULT_ROWS:
        raise ValueError(f"expected {EXPECTED_RESULT_ROWS} K1-BT results, got {len(rows)}")
    checkpoints = sorted(checkpoints_root.glob("*.pt"))
    if len(checkpoints) != EXPECTED_RESULT_ROWS or any(path.stat().st_size == 0 for path in checkpoints):
        raise ValueError("K1-BT requires four non-empty checkpoints")
    metadata_paths = sorted(cache_root.rglob("metadata.json"))
    if len(metadata_paths) != EXPECTED_CACHE_CREATIONS:
        raise ValueError("K1-BT requires four cache metadata files")
    cache_entries: list[dict[str, Any]] = []
    for index, metadata_path in enumerate(metadata_paths):
        features = metadata_path.with_name("features.npy")
        labels = metadata_path.with_name("labels.npy")
        if not features.is_file() or not labels.is_file() or features.stat().st_size == 0 or labels.stat().st_size == 0:
            raise ValueError(f"incomplete cache payload: {metadata_path.parent}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("generation_chunk_size") != 1024 or metadata.get("generation_workers") != 1:
            raise ValueError("cache generation settings drifted")
        relative = metadata_path.parent.relative_to(cache_root)
        split = "validation" if "validation" in relative.parts else "train"
        cache_entries.append({
            "index": index, "cache_path": metadata_path.parent.relative_to(run_root).as_posix(),
            "metadata_sha256": _sha256(metadata_path), "features_bytes": features.stat().st_size,
            "features_sha256": _sha256(features), "labels_bytes": labels.stat().st_size,
            "labels_sha256": _sha256(labels), "total_rows": metadata.get("total_rows"),
            "input_bits": metadata.get("input_bits"), "split": split,
        })
    if archive_root.exists():
        shutil.rmtree(archive_root)
    archive_root.mkdir(parents=True)
    for name in RESULT_FILES:
        _copy(results_root / name, archive_root / name)
    _copy(logs_root / "progress.jsonl", archive_root / "progress.jsonl")
    _copy(source_commit_file, archive_root / "git_revision.txt")
    _copy(expected_source_commit_file, archive_root / "source_expected_commit.txt")
    for source in SOURCE_FILES:
        _copy(source_root / source, archive_root / Path(source).name)
    checkpoint_archive = archive_root / "checkpoints"
    checkpoint_archive.mkdir()
    for checkpoint in checkpoints:
        shutil.copy2(checkpoint, checkpoint_archive / checkpoint.name)
    metadata_archive = archive_root / "cache_metadata"
    metadata_archive.mkdir()
    for index, metadata_path in enumerate(metadata_paths):
        shutil.copy2(metadata_path, metadata_archive / f"cache_{index:02d}_metadata.json")
    logs_archive = archive_root / "logs"
    logs_archive.mkdir()
    for path in sorted(logs_root.glob(f"{RUN_ID}_*")):
        if path.is_file():
            shutil.copy2(path, logs_archive / _archive_log_name(path))
    checkpoint_manifest = {
        "count": len(checkpoints),
        "checkpoints": [{"path": path.relative_to(run_root).as_posix(), "archive_path": f"checkpoints/{path.name}", "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in checkpoints],
    }
    run_manifest = {
        "run_id": RUN_ID, "source_commit": source_commit, "result_rows": len(rows),
        "checkpoint_count": len(checkpoints), "cache_count": len(cache_entries),
        "result_sync": "verified_result_branch_with_local_raw_fallback",
        "claim_scope": "remote 65536/class uKNIT r5 medium architecture confirmation; not formal or paper-scale evidence",
    }
    _write_json(archive_root / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(archive_root / "cache_manifest.json", {"count": len(cache_entries), "caches": cache_entries})
    _write_json(archive_root / "run_manifest.json", run_manifest)
    (archive_root / "plot_deferred.marker").write_text("plot_deferred_to_verified_local_retrieval\n", encoding="utf-8")
    (archive_root / "visual_qa_pending.marker").write_text("visual_qa_pending\n", encoding="utf-8")
    (archive_root / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    artifacts = [{"path": path.relative_to(archive_root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in sorted(archive_root.rglob("*")) if path.is_file() and path.name not in {"artifact_manifest.json", "SHA256SUMS"}]
    _write_json(archive_root / "artifact_manifest.json", {"count": len(artifacts), "artifacts": artifacts})
    checksum_files = [path for path in sorted(archive_root.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    (archive_root / "SHA256SUMS").write_text("".join(f"{_sha256(path)}  {path.relative_to(archive_root).as_posix()}\n" for path in checksum_files), encoding="utf-8")
    return run_manifest


def _archive_log_name(path: Path) -> str:
    return path.name.removeprefix(f"{RUN_ID}_")


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _read_sha(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return ""
    return value if len(value) == 40 and all(c in "0123456789abcdef" for c in value) else ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RESULT_FILES", "SOURCE_FILES", "package_archive"]
