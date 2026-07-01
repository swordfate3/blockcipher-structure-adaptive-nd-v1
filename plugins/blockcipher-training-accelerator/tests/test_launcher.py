from __future__ import annotations

import csv

from blockcipher_training_accelerator.launcher import build_shard_commands
from blockcipher_training_accelerator.matrix import split_matrix


def test_build_shard_commands_uses_existing_train_cli_and_distinct_outputs(tmp_path):
    plan = tmp_path / "plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cipher", "model_key"])
        writer.writeheader()
        writer.writerows(
            [
                {"cipher": "PRESENT-80", "model_key": "a"},
                {"cipher": "PRESENT-80", "model_key": "b"},
            ]
        )
    split_result = split_matrix(plan_path=plan, shards=2, output_dir=tmp_path / "shards")

    launch_plan = build_shard_commands(
        split_result,
        output_dir=tmp_path / "runs",
        python_executable="python",
        train_script="scripts/train",
        devices=["cuda:0", "cuda:1"],
        extra_args=["--epochs", "2"],
    )

    assert len(launch_plan.commands) == 2
    assert launch_plan.commands[0].device == "cuda:0"
    assert launch_plan.commands[1].device == "cuda:1"
    assert "--plan" in launch_plan.commands[0].argv
    assert str(tmp_path / "runs" / "shard00.jsonl") in launch_plan.commands[0].argv
    assert str(tmp_path / "runs" / "shard01.progress.jsonl") in launch_plan.commands[1].argv
