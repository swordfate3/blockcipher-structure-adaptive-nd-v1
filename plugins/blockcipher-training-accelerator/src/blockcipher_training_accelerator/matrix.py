from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MatrixShard:
    index: int
    path: str
    rows: int


@dataclass(frozen=True)
class MatrixSplitResult:
    plan_path: str
    output_dir: str
    strategy: str
    input_rows: int
    shards: list[MatrixShard]
    manifest_path: str

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shards"] = [asdict(shard) for shard in self.shards]
        return payload


def split_matrix(
    *,
    plan_path: Path,
    shards: int,
    output_dir: Path,
    strategy: str = "round-robin",
    prefix: str | None = None,
) -> MatrixSplitResult:
    if shards < 1:
        raise ValueError("shards must be at least 1")
    if strategy not in {"round-robin", "contiguous"}:
        raise ValueError(f"unsupported strategy: {strategy}")
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)

    with plan_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"plan has no CSV header: {plan_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    output_dir.mkdir(parents=True, exist_ok=True)
    assigned = assign_rows(rows, shards=shards, strategy=strategy)
    shard_prefix = prefix or plan_path.stem
    shard_reports: list[MatrixShard] = []
    for index, shard_rows in enumerate(assigned):
        shard_path = output_dir / f"{shard_prefix}.shard{index:02d}.csv"
        with shard_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shard_rows)
        shard_reports.append(
            MatrixShard(index=index, path=str(shard_path), rows=len(shard_rows))
        )

    manifest_path = output_dir / f"{shard_prefix}.manifest.json"
    result = MatrixSplitResult(
        plan_path=str(plan_path),
        output_dir=str(output_dir),
        strategy=strategy,
        input_rows=len(rows),
        shards=shard_reports,
        manifest_path=str(manifest_path),
    )
    manifest_path.write_text(
        json.dumps(result.to_json_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def assign_rows(
    rows: list[dict[str, str]],
    *,
    shards: int,
    strategy: str,
) -> list[list[dict[str, str]]]:
    assigned: list[list[dict[str, str]]] = [[] for _ in range(shards)]
    if strategy == "round-robin":
        for index, row in enumerate(rows):
            assigned[index % shards].append(row)
        return assigned
    if strategy == "contiguous":
        for shard_index, row_range in enumerate(contiguous_ranges(len(rows), shards)):
            start, end = row_range
            assigned[shard_index].extend(rows[start:end])
        return assigned
    raise ValueError(f"unsupported strategy: {strategy}")


def contiguous_ranges(total: int, shards: int) -> list[tuple[int, int]]:
    base = total // shards
    remainder = total % shards
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(shards):
        extra = 1 if index < remainder else 0
        end = start + base + extra
        ranges.append((start, end))
        start = end
    return ranges
