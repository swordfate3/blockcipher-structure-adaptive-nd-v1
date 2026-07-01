from __future__ import annotations

import csv
import json

from blockcipher_training_accelerator.matrix import split_matrix


def test_split_matrix_round_robin_preserves_header_and_rows(tmp_path):
    plan = tmp_path / "plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cipher", "model_key", "seed"])
        writer.writeheader()
        writer.writerows(
            [
                {"cipher": "PRESENT-80", "model_key": "a", "seed": "0"},
                {"cipher": "PRESENT-80", "model_key": "b", "seed": "0"},
                {"cipher": "PRESENT-80", "model_key": "c", "seed": "0"},
                {"cipher": "PRESENT-80", "model_key": "d", "seed": "0"},
            ]
        )

    result = split_matrix(
        plan_path=plan,
        shards=2,
        output_dir=tmp_path / "shards",
        strategy="round-robin",
    )

    assert result.input_rows == 4
    assert [shard.rows for shard in result.shards] == [2, 2]
    first_shard = list(csv.DictReader(open(result.shards[0].path, newline="", encoding="utf-8")))
    second_shard = list(csv.DictReader(open(result.shards[1].path, newline="", encoding="utf-8")))
    assert [row["model_key"] for row in first_shard] == ["a", "c"]
    assert [row["model_key"] for row in second_shard] == ["b", "d"]

    manifest = json.loads(open(result.manifest_path, encoding="utf-8").read())
    assert manifest["strategy"] == "round-robin"
    assert manifest["input_rows"] == 4


def test_split_matrix_contiguous_handles_empty_tail_shards(tmp_path):
    plan = tmp_path / "plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cipher", "model_key"])
        writer.writeheader()
        writer.writerow({"cipher": "PRESENT-80", "model_key": "a"})

    result = split_matrix(
        plan_path=plan,
        shards=3,
        output_dir=tmp_path / "shards",
        strategy="contiguous",
    )

    assert [shard.rows for shard in result.shards] == [1, 0, 0]
