from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.engine.matrix_runner import parse_args
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.engine.pretraining import run_optional_pretraining
from blockcipher_nd.engine.task_runner import run_task
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.planning.next_action_readiness import launch_artifacts
from blockcipher_nd.registry.model_factory import build_model


PLAN_HEADER = (
    "cipher,structure,network,model_key,architecture_rank,score,rounds,seed,"
    "samples_per_class,pretrain_rounds,pretrain_round_sequence\n"
)


def write_plan(
    path: Path,
    *,
    rounds: int = 9,
    samples_per_class: int = 2,
    pretrain_rounds: str = "",
    pretrain_round_sequence: str = "",
) -> None:
    path.write_text(
        PLAN_HEADER
        + (
            "PRESENT-80,SPN,AutoND-DBitNet-2023,autond_dbitnet2023,1,100,"
            f"{rounds},0,{samples_per_class},{pretrain_rounds},"
            f'"{pretrain_round_sequence}"\n'
        ),
        encoding="utf-8",
    )


def test_autond_dbitnet2023_builds_with_paper_geometry() -> None:
    model = build_model("autond_dbitnet2023", input_bits=128, hidden_bits=7)

    assert model.dilations == [63, 31, 15, 7, 3]
    assert model.output_width == 9
    assert model.output_channels == 96
    assert model.flattened_width == 864
    assert [
        (layer.in_features, layer.out_features)
        for layer in model.classifier
        if isinstance(layer, nn.Linear)
    ] == [(864, 256), (256, 256), (256, 64), (64, 1)]


def test_autond_dbitnet2023_returns_one_logit_per_sample() -> None:
    model = build_model("autond_dbitnet2023", input_bits=128, hidden_bits=7)
    model.eval()

    logits = model(torch.randint(0, 2, (3, 128), dtype=torch.float32))

    assert logits.shape == (3, 1)


def test_autond_dbitnet2023_exposes_auditable_model_geometry() -> None:
    model = build_model("autond_dbitnet2023", input_bits=128, hidden_bits=7)

    assert model_metadata(model) == {
        "dilations": [63, 31, 15, 7, 3],
        "output_width": 9,
        "output_channels": 96,
        "flattened_width": 864,
        "l2_coefficient": 1e-5,
    }


def test_autond_dbitnet2023_regularizes_only_dense_kernels() -> None:
    model = build_model("autond_dbitnet2023", input_bits=128, hidden_bits=7)
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                module.weight.fill_(7.0)

    model(torch.zeros(2, 128))
    assert model.last_auxiliary_loss.item() == pytest.approx(0.0)

    first_dense = next(layer for layer in model.classifier if isinstance(layer, nn.Linear))
    with torch.no_grad():
        first_dense.weight.fill_(1.0)
    model(torch.zeros(2, 128))

    assert model.last_auxiliary_loss.item() == pytest.approx(
        first_dense.weight.numel() * 1e-5
    )


def test_plan_parses_pretrain_round_sequence(tmp_path: Path) -> None:
    plan = tmp_path / "sequence.csv"
    write_plan(plan, pretrain_round_sequence="[5,6,7,8]")

    task = build_tasks(parse_args(["--plan", str(plan)]))[0]

    assert task["pretrain_round_sequence"] == (5, 6, 7, 8)


def test_cli_parses_pretrain_round_sequence() -> None:
    args = parse_args(["--pretrain-round-sequence", "[5,6,7,8]"])

    assert args.pretrain_round_sequence == (5, 6, 7, 8)


def test_task_seed_controls_model_initialization(tmp_path: Path) -> None:
    plan = tmp_path / "seeded.csv"
    write_plan(plan)
    args = parse_args(
        [
            "--plan",
            str(plan),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--train-eval-interval",
            "0",
        ]
    )
    task = build_tasks(args)[0]

    torch.manual_seed(123)
    first = run_task(task, args)
    torch.manual_seed(987)
    second = run_task(task, args)

    assert first["metrics"] == second["metrics"]


def test_scalar_pretrain_rounds_remains_a_one_stage_curriculum(tmp_path: Path) -> None:
    plan = tmp_path / "scalar.csv"
    write_plan(plan, pretrain_rounds="8")
    args = parse_args(
        ["--plan", str(plan), "--device", "cpu", "--pretrain-epochs", "1"]
    )
    task = build_tasks(args)[0]

    result = run_optional_pretraining(
        nn.Linear(128, 1),
        task,
        args,
        pair_bits=128,
        progress_path=None,
        index=1,
        total=1,
    )

    assert result is not None
    assert task["pretrain_rounds"] == 8
    assert task["pretrain_round_sequence"] == ()
    assert result.metadata["round_sequence"] == [8]
    assert [stage["rounds"] for stage in result.metadata["curriculum_stages"]] == [8]


def test_curriculum_trains_stages_in_order_and_records_metadata(tmp_path: Path) -> None:
    plan = tmp_path / "curriculum.csv"
    write_plan(plan, pretrain_round_sequence="[5,6]")
    args = parse_args(
        [
            "--plan",
            str(plan),
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--pretrain-epochs",
            "1",
            "--train-eval-interval",
            "0",
        ]
    )
    task = build_tasks(args)[0]
    progress = tmp_path / "progress.jsonl"

    result = run_optional_pretraining(
        nn.Linear(128, 1),
        task,
        args,
        pair_bits=128,
        progress_path=str(progress),
        index=1,
        total=1,
    )

    assert result is not None
    assert result.metadata["round_sequence"] == [5, 6]
    assert [stage["rounds"] for stage in result.metadata["curriculum_stages"]] == [5, 6]
    assert all(stage["epochs_ran"] == 1 for stage in result.metadata["curriculum_stages"])
    assert all("metrics" in stage for stage in result.metadata["curriculum_stages"])
    assert all("selected_checkpoint" in stage for stage in result.metadata["curriculum_stages"])
    progress_rows = [
        json.loads(line)
        for line in progress.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [
        row["pretrain_rounds"]
        for row in progress_rows
        if row["event"] == "pretrain_cache_ready"
    ] == [5, 6]


def test_curriculum_rejects_sequence_containing_target_round(tmp_path: Path) -> None:
    plan = tmp_path / "invalid.csv"
    write_plan(plan, rounds=8, pretrain_round_sequence="[5,8]")
    args = parse_args(["--plan", str(plan), "--pretrain-epochs", "1"])
    task = build_tasks(args)[0]

    with pytest.raises(ValueError, match="must be lower than target rounds"):
        run_optional_pretraining(
            nn.Linear(128, 1),
            task,
            args,
            pair_bits=128,
            progress_path=None,
            index=1,
            total=1,
        )


def test_curriculum_rejects_non_increasing_rounds(tmp_path: Path) -> None:
    plan = tmp_path / "invalid.csv"
    write_plan(plan, pretrain_round_sequence="[5,7,6,8]")
    args = parse_args(["--plan", str(plan), "--pretrain-epochs", "1"])
    task = build_tasks(args)[0]

    with pytest.raises(ValueError, match="strictly increasing"):
        run_optional_pretraining(
            nn.Linear(128, 1),
            task,
            args,
            pair_bits=128,
            progress_path=None,
            index=1,
            total=1,
        )


def test_autond_remote_package_locks_strict_medium_protocol(tmp_path: Path) -> None:
    config_path = Path(
        "configs/remote/"
        "innovation1_spn_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    readiness = remote_readiness_report(config_path)
    artifacts = launch_artifacts(config_path)
    launcher = Path(artifacts["launcher"]).read_text(encoding="utf-8")
    monitor = Path(artifacts["monitor"]).read_text(encoding="utf-8")

    assert readiness["status"] == "pass"
    assert "autond_dbitnet_protocol_lock" in readiness["checked_invariants"]
    assert readiness["expected_rows"] == 1
    assert readiness["max_samples_per_class"] == 65536
    assert artifacts["status"] == "pass"
    assert config["device"] == "cuda:1"
    assert config["amsgrad"] is True
    assert config["pretrain_round_sequence"] == [5, 6, 7, 8]
    assert config["negative_mode"] == "encrypted_random_plaintexts"
    assert "--pretrain-round-sequence \"[5,6,7,8]\"" in launcher
    assert "--amsgrad" in launcher
    assert "--dataset-cache-root" in launcher
    assert "echo result_lines=%RESULT_LINES% >" in launcher
    assert "echo expected_rows=1 >>" in launcher
    assert "cmd.exe /k" not in launcher
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher
    assert "C:\\Users" not in launcher
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor

    config["amsgrad"] = False
    invalid_config = tmp_path / "invalid_autond_remote.json"
    invalid_config.write_text(json.dumps(config), encoding="utf-8")
    invalid_readiness = remote_readiness_report(invalid_config)
    assert invalid_readiness["status"] == "fail"
    assert "autond_dbitnet amsgrad=False expected=True" in invalid_readiness["errors"]
