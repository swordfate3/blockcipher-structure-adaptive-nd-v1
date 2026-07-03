from __future__ import annotations

import json

from blockcipher_training_accelerator.cli import main
from blockcipher_training_accelerator.dataset_cache import (
    DatasetCacheBenchConfig,
    TrailFamilyCacheBenchConfig,
    run_dataset_cache_benchmark,
    run_trail_family_cache_benchmark,
)


def test_run_dataset_cache_benchmark_writes_summary(tmp_path):
    report = run_dataset_cache_benchmark(
        DatasetCacheBenchConfig(
            cipher="speck32",
            rounds=3,
            samples_per_class=8,
            pairs_per_sample=2,
            sample_structure="independent_pairs",
            negative_mode="encrypted_random_plaintexts",
            feature_encoding="ciphertext_pair_xor_bits",
            seed=7,
            chunk_size=4,
            workers=(1, 2),
            output_root=str(tmp_path / "cache_bench"),
            input_difference=0x0040,
        )
    )

    assert len(report.rows) == 2
    assert report.rows[0].workers == 1
    assert report.rows[1].workers == 2
    assert report.rows[0].total_rows == 16
    assert report.rows[0].label_values == (0, 1)
    assert report.rows[1].generation_workers == 2

    summary = json.loads((tmp_path / "cache_bench" / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"]["cipher"] == "speck32"
    assert summary["protocol"]["input_difference"] == 0x0040
    assert summary["rows"][1]["workers"] == 2
    assert summary["rows"][1]["generation_workers"] == 2


def test_run_dataset_cache_benchmark_can_use_difference_profile(tmp_path):
    report = run_dataset_cache_benchmark(
        DatasetCacheBenchConfig(
            cipher="present80",
            rounds=7,
            samples_per_class=2,
            pairs_per_sample=1,
            sample_structure="zhang_wang_case2_official_mcnd",
            negative_mode="encrypted_random_plaintexts",
            feature_encoding="ciphertext_pair_bits",
            seed=9,
            chunk_size=1,
            workers=(1,),
            output_root=str(tmp_path / "present_cache_bench"),
            difference_profile="present_zhang_wang2022_mcnd",
        )
    )

    assert report.protocol["input_difference"] == 0x0000000000000009
    assert report.rows[0].total_rows == 4
    assert report.rows[0].features_shape == (4, 128)


def test_run_trail_family_cache_benchmark_writes_summary(tmp_path):
    report = run_trail_family_cache_benchmark(
        TrailFamilyCacheBenchConfig(
            samples_per_class=2,
            pairs_per_sample=1,
            seed=11,
            chunk_size=1,
            workers=(1, 2),
            output_root=str(tmp_path / "trail_family_cache_bench"),
            rounds=7,
            difference_profile="present_zhang_wang2022_mcnd",
            negative_mode="encrypted_random_plaintexts",
            sample_structure="zhang_wang_case2_official_mcnd",
            key_rotation_interval=0,
            beam_width=2,
            depth=1,
        )
    )

    assert len(report.rows) == 2
    assert report.rows[0].workers == 1
    assert report.rows[1].workers == 2
    assert report.rows[0].total_rows == 4
    assert report.rows[0].label_values == (0, 1)
    assert report.rows[0].feature_route == "trail_family_consistency"
    assert report.rows[0].cache_status == "created"

    summary = json.loads((tmp_path / "trail_family_cache_bench" / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"]["feature_route"] == "trail_family_consistency"
    assert summary["protocol"]["input_difference"] == 0x0000000000000009
    assert summary["rows"][1]["workers"] == 2


def test_cli_runs_trail_family_cache_benchmark(tmp_path):
    output_root = tmp_path / "trail_family_cli_bench"

    main(
        [
            "bench-trail-family-cache",
            "--samples-per-class",
            "2",
            "--pairs-per-sample",
            "1",
            "--seed",
            "13",
            "--chunk-size",
            "1",
            "--workers",
            "1",
            "2",
            "--output-root",
            str(output_root),
            "--beam-width",
            "2",
            "--depth",
            "1",
        ]
    )

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"]["feature_route"] == "trail_family_consistency"
    assert [row["workers"] for row in summary["rows"]] == [1, 2]
