from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from blockcipher_training_accelerator.matrix import MatrixSplitResult


@dataclass(frozen=True)
class ShardCommand:
    shard_index: int
    device: str
    argv: list[str]

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LaunchPlan:
    commands: list[ShardCommand]

    def to_json_dict(self) -> dict[str, object]:
        return {"commands": [command.to_json_dict() for command in self.commands]}


def build_shard_commands(
    split_result: MatrixSplitResult,
    *,
    output_dir: Path,
    python_executable: str,
    train_script: str,
    devices: list[str],
    extra_args: list[str] | None = None,
) -> LaunchPlan:
    if not devices:
        raise ValueError("at least one device is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[ShardCommand] = []
    for shard in split_result.shards:
        device = devices[shard.index % len(devices)]
        output_path = output_dir / f"shard{shard.index:02d}.jsonl"
        progress_path = output_dir / f"shard{shard.index:02d}.progress.jsonl"
        argv = [
            python_executable,
            train_script,
            "--plan",
            shard.path,
            "--device",
            device,
            "--output",
            str(output_path),
            "--progress-output",
            str(progress_path),
        ]
        if extra_args:
            argv.extend(extra_args)
        commands.append(ShardCommand(shard_index=shard.index, device=device, argv=argv))
    return LaunchPlan(commands=commands)
