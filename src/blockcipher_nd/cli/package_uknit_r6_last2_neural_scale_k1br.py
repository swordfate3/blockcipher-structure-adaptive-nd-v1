from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from blockcipher_nd.tasks.innovation1.uknit_r6_last2_neural_scale_k1br import (
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
    "configs/experiment/innovation1/innovation1_uknit_r6_last2_neural_scale_k1br_262144_seed3.csv",
    "configs/remote/innovation1_uknit_r6_last2_neural_scale_k1br_262144_seed3_gpu1_20260730.json",
    "docs/experiments/innovation1-uknit-r6-last2-neural-scale-k1br-plan.md",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package uKNIT r6 K1-BR evidence.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-commit-file", required=True, type=Path)
    parser.add_argument("--expected-source-commit-file", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = package_k1br_archive(
        run_root=args.run_root,
        source_root=args.source_root,
        source_commit_file=args.source_commit_file,
        expected_source_commit_file=args.expected_source_commit_file,
        archive_root=args.archive_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def package_k1br_archive(
    *,
    run_root: Path,
    source_root: Path,
    source_commit_file: Path,
    expected_source_commit_file: Path,
    archive_root: Path,
) -> dict[str, Any]:
    source_commit = _read_sha(source_commit_file)
    if not source_commit or source_commit != _read_sha(expected_source_commit_file):
        raise ValueError("source revision does not match launch pin")
    results_root = run_root / "results"
    logs_root = run_root / "logs"
    result_rows = _read_jsonl(results_root / "results.jsonl")
    if len(result_rows) != EXPECTED_RESULT_ROWS:
        raise ValueError(f"expected {EXPECTED_RESULT_ROWS} result rows")
    checkpoints = sorted((run_root / "checkpoints").glob("*.pt"))
    if len(checkpoints) != EXPECTED_RESULT_ROWS or any(
        path.stat().st_size == 0 for path in checkpoints
    ):
        raise ValueError("K1-BR requires three non-empty checkpoints")
    metadata_paths = sorted((run_root / "cache").rglob("metadata.json"))
    if len(metadata_paths) != EXPECTED_CACHE_CREATIONS:
        raise ValueError("K1-BR requires two completed disk caches")
    cache_entries = []
    for path in metadata_paths:
        features = path.with_name("features.npy")
        labels = path.with_name("labels.npy")
        if (
            not features.is_file()
            or not labels.is_file()
            or not features.stat().st_size
            or not labels.stat().st_size
        ):
            raise ValueError(f"incomplete cache: {path.parent}")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if (
            metadata.get("generation_chunk_size") != 1024
            or metadata.get("generation_workers") != 1
        ):
            raise ValueError("cache generation protocol drifted")
        cache_entries.append(
            {
                "cache_path": path.parent.relative_to(run_root).as_posix(),
                "metadata_sha256": _sha256(path),
                "features_bytes": features.stat().st_size,
                "labels_bytes": labels.stat().st_size,
                "metadata": metadata,
            }
        )
    archive_root.mkdir(parents=True, exist_ok=False)
    for name in RESULT_FILES:
        _copy(results_root / name, archive_root / name)
    _copy(logs_root / "progress.jsonl", archive_root / "progress.jsonl")
    _copy(source_commit_file, archive_root / "git_revision.txt")
    _copy(expected_source_commit_file, archive_root / "source_expected_commit.txt")
    gpu_logs = sorted(logs_root.glob("*_gpu_info.txt"))
    if len(gpu_logs) != 1:
        raise ValueError("expected one GPU inventory log")
    _copy(gpu_logs[0], archive_root / "gpu_info.txt")
    for source in SOURCE_FILES:
        _copy(source_root / source, archive_root / Path(source).name)
    checkpoint_manifest = {
        "count": len(checkpoints),
        "checkpoints": [
            {
                "path": path.relative_to(run_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in checkpoints
        ],
        "note": "Large checkpoint binaries remain in the G:\\lxy run root and are represented by hashes only.",
    }
    _write_json(archive_root / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(
        archive_root / "cache_manifest.json",
        {"count": len(cache_entries), "caches": cache_entries},
    )
    report = {
        "run_id": RUN_ID,
        "source_commit": source_commit,
        "result_rows": len(result_rows),
        "checkpoint_count": len(checkpoints),
        "cache_count": len(cache_entries),
        "result_sync": "verified_result_branch_with_raw_fallback",
        "claim_scope": "single-seed remote 262144/class uKNIT r6 larger diagnostic; not formal or paper-scale",
    }
    _write_json(archive_root / "run_manifest.json", report)
    (archive_root / "plot_deferred.marker").write_text(
        "plot_deferred_to_local_retrieval\n", encoding="utf-8"
    )
    (archive_root / "visual_qa_pending.marker").write_text(
        "visual_qa_pending\n", encoding="utf-8"
    )
    (archive_root / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    artifacts = [
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
        {"count": len(artifacts), "artifacts": artifacts},
    )
    checksum_paths = [
        path
        for path in sorted(archive_root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (archive_root / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(archive_root).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )
    return report


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows


def _read_sha(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return ""
    return (
        value
        if len(value) == 40 and all(c in "0123456789abcdef" for c in value)
        else ""
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
