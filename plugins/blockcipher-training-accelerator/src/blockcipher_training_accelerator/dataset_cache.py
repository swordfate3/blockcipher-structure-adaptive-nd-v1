from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import shutil
import time
from pathlib import Path

import numpy as np

from blockcipher_nd.data.cache import make_chunked_differential_dataset
from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.difference_profiles import difference_for_profile
from blockcipher_nd.tasks.innovation1.spn_trail_family import make_trail_family_dataset


@dataclass(frozen=True)
class DatasetCacheBenchConfig:
    cipher: str
    rounds: int
    samples_per_class: int
    pairs_per_sample: int
    sample_structure: str
    negative_mode: str
    feature_encoding: str
    seed: int
    chunk_size: int
    workers: tuple[int, ...]
    output_root: str
    difference_profile: str | None = None
    difference_member: int = 0
    input_difference: int | None = None
    key: int | None = None
    reuse: bool = False


@dataclass(frozen=True)
class DatasetCacheBenchRow:
    workers: int
    duration_seconds: float
    rows_per_second: float
    total_rows: int
    features_shape: tuple[int, ...]
    labels_shape: tuple[int, ...]
    label_values: tuple[int, ...]
    cache_status: str
    generation_workers: int
    generation_chunk_size: int
    input_bits: int
    progress_events: int
    cache_dir: str

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["features_shape"] = list(self.features_shape)
        payload["labels_shape"] = list(self.labels_shape)
        payload["label_values"] = list(self.label_values)
        return payload


@dataclass(frozen=True)
class DatasetCacheBenchReport:
    run_id: str
    protocol: dict[str, object]
    rows: tuple[DatasetCacheBenchRow, ...]
    summary_path: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "protocol": self.protocol,
            "rows": [row.to_json_dict() for row in self.rows],
            "summary_path": self.summary_path,
        }


@dataclass(frozen=True)
class TrailFamilyCacheBenchConfig:
    samples_per_class: int
    pairs_per_sample: int
    seed: int
    chunk_size: int
    workers: tuple[int, ...]
    output_root: str
    rounds: int = 7
    difference_profile: str = "present_zhang_wang2022_mcnd"
    difference_member: int = 0
    input_difference: int | None = None
    key: int = 0
    negative_mode: str = "encrypted_random_plaintexts"
    sample_structure: str = "zhang_wang_case2_official_mcnd"
    key_rotation_interval: int = 0
    beam_width: int = 4
    depth: int = 3
    false_family: bool = False
    reuse: bool = False


@dataclass(frozen=True)
class TrailFamilyCacheBenchRow:
    workers: int
    duration_seconds: float
    rows_per_second: float
    total_rows: int
    features_shape: tuple[int, ...]
    labels_shape: tuple[int, ...]
    label_values: tuple[int, ...]
    cache_status: str
    feature_route: str
    feature_dim: int
    progress_events: int
    cache_dir: str

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["features_shape"] = list(self.features_shape)
        payload["labels_shape"] = list(self.labels_shape)
        payload["label_values"] = list(self.label_values)
        return payload


@dataclass(frozen=True)
class TrailFamilyCacheBenchReport:
    run_id: str
    protocol: dict[str, object]
    rows: tuple[TrailFamilyCacheBenchRow, ...]
    summary_path: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "protocol": self.protocol,
            "rows": [row.to_json_dict() for row in self.rows],
            "summary_path": self.summary_path,
        }


def run_dataset_cache_benchmark(config: DatasetCacheBenchConfig) -> DatasetCacheBenchReport:
    if config.samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")
    if config.pairs_per_sample < 1:
        raise ValueError("pairs_per_sample must be at least 1")
    if config.chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not config.workers:
        raise ValueError("at least one worker count is required")
    if any(worker < 1 for worker in config.workers):
        raise ValueError("worker counts must be at least 1")

    input_difference = _resolve_input_difference(config)
    cipher = build_cipher(config.cipher, config.rounds, key=config.key)
    dataset_config = DifferentialDatasetConfig(
        cipher=cipher,
        input_difference=input_difference,
        samples_per_class=config.samples_per_class,
        seed=config.seed,
        shuffle=False,
        feature_encoding=config.feature_encoding,
        pairs_per_sample=config.pairs_per_sample,
        negative_mode=config.negative_mode,
        sample_structure=config.sample_structure,
    )

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[DatasetCacheBenchRow] = []
    for workers in config.workers:
        cache_dir = output_root / f"workers_{workers}"
        if cache_dir.exists() and not config.reuse:
            shutil.rmtree(cache_dir)
        events: list[str] = []

        def progress(event: str, _payload: dict[str, object]) -> None:
            events.append(event)

        started = time.perf_counter()
        dataset = make_chunked_differential_dataset(
            dataset_config,
            cache_dir=cache_dir,
            chunk_size=config.chunk_size,
            workers=workers,
            reuse=config.reuse,
            progress_callback=progress,
        )
        duration = time.perf_counter() - started
        total_rows = int(dataset.labels.shape[0])
        labels = np.asarray(dataset.labels)
        metadata = dataset.metadata
        row = DatasetCacheBenchRow(
            workers=workers,
            duration_seconds=round(duration, 6),
            rows_per_second=round(total_rows / duration, 3),
            total_rows=total_rows,
            features_shape=tuple(int(value) for value in dataset.features.shape),
            labels_shape=tuple(int(value) for value in dataset.labels.shape),
            label_values=tuple(sorted(int(value) for value in np.unique(labels).tolist())),
            cache_status=str(metadata.get("cache_status")),
            generation_workers=int(metadata.get("generation_workers", workers)),
            generation_chunk_size=int(metadata.get("generation_chunk_size", config.chunk_size)),
            input_bits=int(metadata.get("input_bits", dataset.features.shape[1])),
            progress_events=len(events),
            cache_dir=str(cache_dir),
        )
        rows.append(row)

    summary_path = output_root / "summary.json"
    report = DatasetCacheBenchReport(
        run_id=output_root.name,
        protocol={
            "cipher": config.cipher,
            "rounds": config.rounds,
            "input_difference": input_difference,
            "difference_profile": config.difference_profile,
            "difference_member": config.difference_member,
            "samples_per_class": config.samples_per_class,
            "pairs_per_sample": config.pairs_per_sample,
            "sample_structure": config.sample_structure,
            "negative_mode": config.negative_mode,
            "feature_encoding": config.feature_encoding,
            "seed": config.seed,
            "chunk_size": config.chunk_size,
            "workers": list(config.workers),
            "reuse": config.reuse,
        },
        rows=tuple(rows),
        summary_path=str(summary_path),
    )
    summary_path.write_text(
        json.dumps(report.to_json_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def run_trail_family_cache_benchmark(config: TrailFamilyCacheBenchConfig) -> TrailFamilyCacheBenchReport:
    if config.samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")
    if config.pairs_per_sample < 1:
        raise ValueError("pairs_per_sample must be at least 1")
    if config.chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not config.workers:
        raise ValueError("at least one worker count is required")
    if any(worker < 1 for worker in config.workers):
        raise ValueError("worker counts must be at least 1")

    input_difference = (
        config.input_difference
        if config.input_difference is not None
        else difference_for_profile(config.difference_profile, config.difference_member)
    )
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[TrailFamilyCacheBenchRow] = []
    for workers in config.workers:
        cache_root = output_root / f"workers_{workers}"
        if cache_root.exists() and not config.reuse:
            shutil.rmtree(cache_root)
        progress_output = output_root / f"workers_{workers}_progress.jsonl"
        if progress_output.exists() and not config.reuse:
            progress_output.unlink()

        started = time.perf_counter()
        features, labels = make_trail_family_dataset(
            rounds=config.rounds,
            key=config.key,
            input_difference=input_difference,
            seed=config.seed,
            samples_per_class=config.samples_per_class,
            pairs_per_sample=config.pairs_per_sample,
            negative_mode=config.negative_mode,
            sample_structure=config.sample_structure,
            key_rotation_interval=config.key_rotation_interval,
            beam_width=config.beam_width,
            depth=config.depth,
            false_family=config.false_family,
            feature_cache_root=cache_root,
            feature_cache_chunk_size=config.chunk_size,
            feature_cache_workers=workers,
            progress_output=progress_output,
            split="bench",
        )
        duration = time.perf_counter() - started
        labels_array = np.asarray(labels)
        progress_events = _line_count(progress_output)
        cache_status = "reused" if _progress_has_event(progress_output, "trail_family_cache_reuse") else "created"
        total_rows = int(labels_array.shape[0])
        rows.append(
            TrailFamilyCacheBenchRow(
                workers=workers,
                duration_seconds=round(duration, 6),
                rows_per_second=round(total_rows / duration, 3),
                total_rows=total_rows,
                features_shape=tuple(int(value) for value in features.shape),
                labels_shape=tuple(int(value) for value in labels_array.shape),
                label_values=tuple(sorted(int(value) for value in np.unique(labels_array).tolist())),
                cache_status=cache_status,
                feature_route="trail_family_consistency",
                feature_dim=int(features.shape[1]),
                progress_events=progress_events,
                cache_dir=str(cache_root),
            )
        )

    summary_path = output_root / "summary.json"
    report = TrailFamilyCacheBenchReport(
        run_id=output_root.name,
        protocol={
            "feature_route": "trail_family_consistency",
            "rounds": config.rounds,
            "input_difference": input_difference,
            "difference_profile": config.difference_profile,
            "difference_member": config.difference_member,
            "samples_per_class": config.samples_per_class,
            "pairs_per_sample": config.pairs_per_sample,
            "sample_structure": config.sample_structure,
            "negative_mode": config.negative_mode,
            "seed": config.seed,
            "chunk_size": config.chunk_size,
            "workers": list(config.workers),
            "beam_width": config.beam_width,
            "depth": config.depth,
            "false_family": config.false_family,
            "reuse": config.reuse,
        },
        rows=tuple(rows),
        summary_path=str(summary_path),
    )
    summary_path.write_text(
        json.dumps(report.to_json_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _progress_has_event(path: Path, event: str) -> bool:
    if not path.exists():
        return False
    return f'"event": "{event}"' in path.read_text(encoding="utf-8")


def _resolve_input_difference(config: DatasetCacheBenchConfig) -> int:
    if config.input_difference is not None:
        return config.input_difference
    if config.difference_profile:
        return difference_for_profile(config.difference_profile, config.difference_member)
    raise ValueError("either input_difference or difference_profile is required")


def parse_int(value: str) -> int:
    return int(value, 0)
