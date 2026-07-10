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
    "samples_per_class,pretrain_rounds,pretrain_round_sequence,"
    "optimizer_state_transition\n"
)


def write_plan(
    path: Path,
    *,
    rounds: int = 9,
    samples_per_class: int = 2,
    pretrain_rounds: str = "",
    pretrain_round_sequence: str = "",
    optimizer_state_transition: str = "reset_each_stage",
) -> None:
    path.write_text(
        PLAN_HEADER
        + (
            "PRESENT-80,SPN,AutoND-DBitNet-2023,autond_dbitnet2023,1,100,"
            f"{rounds},0,{samples_per_class},{pretrain_rounds},"
            f'"{pretrain_round_sequence}",{optimizer_state_transition}\n'
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


def test_cli_parses_optimizer_state_transition() -> None:
    args = parse_args(
        ["--optimizer-state-transition", "carry_across_stages"]
    )

    assert args.optimizer_state_transition == "carry_across_stages"


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
            "--checkpoint-metric",
            "val_loss",
            "--restore-best-checkpoint",
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
    assert all(
        stage["checkpoint_metric"] == "val_loss"
        for stage in result.metadata["curriculum_stages"]
    )
    assert result.metadata["optimizer_state_transition"] == "reset_each_stage"
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


def test_curriculum_carries_optimizer_state_into_target_round(tmp_path: Path) -> None:
    plan = tmp_path / "curriculum-carry.csv"
    write_plan(
        plan,
        pretrain_round_sequence="[5,6]",
        optimizer_state_transition="carry_across_stages",
    )
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
            "--pretrain-epochs",
            "1",
            "--checkpoint-metric",
            "val_loss",
            "--restore-best-checkpoint",
            "--train-eval-interval",
            "0",
        ]
    )
    task = build_tasks(args)[0]

    row = run_task(task, args)

    pretraining = row["training"]["pretraining"]
    stages = pretraining["curriculum_stages"]
    assert task["optimizer_state_transition"] == "carry_across_stages"
    assert pretraining["optimizer_state_transition"] == "carry_across_stages"
    assert [stage["optimizer_state_reused"] for stage in stages] == [False, True]
    assert stages[1]["optimizer_state_step_before"] == stages[0][
        "optimizer_state_step_after"
    ]
    assert row["training"]["optimizer_state_transition"] == "carry_across_stages"
    assert row["training"]["optimizer_state_reused"] is True
    assert row["training"]["optimizer_state_step_before"] == stages[-1][
        "optimizer_state_step_after"
    ]


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


def test_autond_readiness_allows_val_loss_checkpoint_ablation(tmp_path: Path) -> None:
    base_plan = Path(
        "configs/experiment/innovation1/"
        "innovation1_spn_present_autond_dbitnet_strict_65k_seed0.csv"
    )
    plan = tmp_path / "autond_val_loss.csv"
    plan.write_text(
        base_plan.read_text(encoding="utf-8").replace("val_accuracy", "val_loss"),
        encoding="utf-8",
    )
    base_config = Path(
        "configs/remote/"
        "innovation1_spn_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710.json"
    )
    config = json.loads(base_config.read_text(encoding="utf-8"))
    config["plan"] = str(plan)
    config["checkpoint_metric"] = "val_loss"
    config_path = tmp_path / "autond_val_loss.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    readiness = remote_readiness_report(config_path)

    assert readiness["status"] == "pass", readiness["errors"]
    assert "autond_dbitnet_protocol_lock" in readiness["checked_invariants"]


@pytest.mark.parametrize(
    ("plan_name", "samples_per_class", "pretrain_epochs"),
    [
        (
            "innovation1_spn_present_autond_dbitnet_r1a_valloss_smoke_seed0.csv",
            128,
            2,
        ),
        (
            "innovation1_spn_present_autond_dbitnet_r1a_valloss_65k_seed0.csv",
            65536,
            10,
        ),
    ],
)
def test_autond_r1a_plans_lock_val_loss_single_variable(
    plan_name: str,
    samples_per_class: int,
    pretrain_epochs: int,
) -> None:
    plan = Path("configs/experiment/innovation1") / plan_name
    task = build_tasks(parse_args(["--plan", str(plan)]))[0]

    assert task["model_key"] == "autond_dbitnet2023"
    assert task["samples_per_class"] == samples_per_class
    assert task["pretrain_round_sequence"] == (5, 6, 7, 8)
    assert task["pretrain_epochs"] == pretrain_epochs
    assert task["checkpoint_metric"] == "val_loss"
    assert task["restore_best_checkpoint"] is True
    assert task["negative_mode"] == "encrypted_random_plaintexts"
    assert task["validation_key"] == 0x11111111111111111111

    if samples_per_class == 65536:
        anchor = build_tasks(
            parse_args(
                [
                    "--plan",
                    "configs/experiment/innovation1/"
                    "innovation1_spn_present_autond_dbitnet_strict_65k_seed0.csv",
                ]
            )
        )[0]
        for field in (
            "model_key",
            "rounds",
            "seed",
            "samples_per_class",
            "pairs_per_sample",
            "feature_encoding",
            "negative_mode",
            "train_key",
            "validation_key",
            "key_rotation_interval",
            "sample_structure",
            "difference_profile",
            "loss",
            "learning_rate",
            "optimizer",
            "weight_decay",
            "lr_scheduler",
            "pretrain_round_sequence",
            "pretrain_epochs",
        ):
            assert task[field] == anchor[field]
        assert anchor["checkpoint_metric"] == "val_accuracy"


def test_autond_r1a_remote_package_locks_checkpoint_ablation() -> None:
    config_path = Path(
        "configs/remote/"
        "innovation1_spn_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    readiness = remote_readiness_report(config_path)
    artifacts = launch_artifacts(config_path)
    launcher = Path(artifacts["launcher"]).read_text(encoding="utf-8")
    monitor = Path(artifacts["monitor"]).read_text(encoding="utf-8")

    assert readiness["status"] == "pass", readiness["errors"]
    assert readiness["expected_rows"] == 1
    assert readiness["max_samples_per_class"] == 65536
    assert artifacts["status"] == "pass"
    assert config["device"] == "cuda:1"
    assert config["checkpoint_metric"] == "val_loss"
    assert config["pretrain_round_sequence"] == [5, 6, 7, 8]
    assert config["negative_mode"] == "encrypted_random_plaintexts"
    assert "--checkpoint-metric val_loss" in launcher
    assert "--pretrain-round-sequence \"[5,6,7,8]\"" in launcher
    assert "--dataset-cache-root" in launcher
    assert "cmd.exe /k" not in launcher
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher
    assert "C:\\Users" not in launcher
    assert '"checkpoint_metric"' in monitor
    assert '"optimizer_state_transition"' in monitor
    assert "i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710" in monitor
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor


def test_autond_c2_remote_package_locks_optimizer_carry() -> None:
    config_path = Path(
        "configs/remote/"
        "innovation1_spn_present_autond_dbitnet_r1a_c2_optcarry_65k_seed0_gpu1_20260710.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    readiness = remote_readiness_report(config_path)
    artifacts = launch_artifacts(config_path)
    launcher = Path(artifacts["launcher"]).read_text(encoding="utf-8")
    monitor = Path(artifacts["monitor"]).read_text(encoding="utf-8")

    assert readiness["status"] == "pass", readiness["errors"]
    assert readiness["expected_rows"] == 1
    assert readiness["max_samples_per_class"] == 65536
    assert artifacts["status"] == "pass"
    assert config["device"] == "cuda:1"
    assert config["checkpoint_metric"] == "val_loss"
    assert config["optimizer_state_transition"] == "carry_across_stages"
    assert config["pretrain_round_sequence"] == [5, 6, 7, 8]
    assert config["negative_mode"] == "encrypted_random_plaintexts"
    assert "--checkpoint-metric val_loss" in launcher
    assert "--optimizer-state-transition carry_across_stages" in launcher
    assert "--pretrain-round-sequence \"[5,6,7,8]\"" in launcher
    assert "--dataset-cache-root" in launcher
    assert "cmd.exe /k" not in launcher
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher
    assert "C:\\Users" not in launcher
    assert '"optimizer_step_continuity"' in monitor
    assert '"optimizer_state_reused"' in monitor
    assert '"accuracy_delta_vs_c1"' in monitor
    assert "i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710" in monitor
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor


@pytest.mark.parametrize(
    ("plan_name", "samples_per_class", "pretrain_epochs"),
    [
        (
            "innovation1_spn_present_autond_dbitnet_r1a_c2_optcarry_smoke_seed0.csv",
            128,
            2,
        ),
        (
            "innovation1_spn_present_autond_dbitnet_r1a_c2_optcarry_65k_seed0.csv",
            65536,
            10,
        ),
    ],
)
def test_autond_c2_plans_change_only_optimizer_transition(
    plan_name: str,
    samples_per_class: int,
    pretrain_epochs: int,
) -> None:
    plan = Path("configs/experiment/innovation1") / plan_name
    task = build_tasks(parse_args(["--plan", str(plan)]))[0]

    assert task["samples_per_class"] == samples_per_class
    assert task["pretrain_epochs"] == pretrain_epochs
    assert task["checkpoint_metric"] == "val_loss"
    assert task["optimizer_state_transition"] == "carry_across_stages"
    assert task["negative_mode"] == "encrypted_random_plaintexts"

    if samples_per_class == 65536:
        c1 = build_tasks(
            parse_args(
                [
                    "--plan",
                    "configs/experiment/innovation1/"
                    "innovation1_spn_present_autond_dbitnet_r1a_valloss_65k_seed0.csv",
                ]
            )
        )[0]
        for field in (
            "model_key",
            "rounds",
            "seed",
            "samples_per_class",
            "pairs_per_sample",
            "feature_encoding",
            "negative_mode",
            "train_key",
            "validation_key",
            "key_rotation_interval",
            "sample_structure",
            "difference_profile",
            "loss",
            "learning_rate",
            "optimizer",
            "weight_decay",
            "lr_scheduler",
            "checkpoint_metric",
            "restore_best_checkpoint",
            "pretrain_round_sequence",
            "pretrain_epochs",
        ):
            assert task[field] == c1[field]
        assert c1["optimizer_state_transition"] == "reset_each_stage"
