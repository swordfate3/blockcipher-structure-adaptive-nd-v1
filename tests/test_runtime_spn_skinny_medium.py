from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.gate_runtime_spn_skinny_medium_joint import (
    main as joint_gate_main,
)
from blockcipher_nd.cli.gate_runtime_spn_skinny_medium import (
    _verify_checkpoint_payloads,
    main as single_gate_main,
    render_skinny_medium_svg,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_attribution import MODELS
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_medium import (
    RTG2B_RUN_STEM,
    adjudicate_runtime_spn_skinny_medium,
    adjudicate_runtime_spn_skinny_medium_joint,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_rtg2b_launch import (
    JOINT_DECISION,
    JOINT_RUN_ID,
    adjudicate_runtime_spn_skinny_rtg2b_launch,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_rtg2b_seed1_launch import (
    SEED0_DECISION as RTG2B_SEED0_DECISION,
    SEED0_RUN_ID as RTG2B_SEED0_RUN_ID,
    adjudicate_runtime_spn_skinny_rtg2b_seed1_launch,
)
from blockcipher_nd.registry.model_factory import build_model


ROOT = Path(__file__).resolve().parents[1]
RUN_STEM = "i1_rtg2a_skinny64_general_gf2_medium_65536"


def _rows(
    seed: int,
    aucs: dict[str, float],
    *,
    samples_per_class: int = 65_536,
) -> list[dict[str, object]]:
    cache_root = (
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        f"{RUN_STEM}_seed{seed}_20260724\\cache"
    )
    common_training = {
        "epochs": 5,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "batch_size": 64,
        "train_eval_interval": 1,
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "selected_checkpoint": "best",
        "train_rows": samples_per_class * 2,
        "validation_rows": samples_per_class,
        "model_options": {
            "processor_steps": 2,
            "pair_embedding_dim": 128,
            "dropout": 0.0,
            "sbox_context_mode": "late_pair",
        },
        "dataset_cache_root": cache_root,
        "dataset_cache_chunk_size": 1024,
        "dataset_cache_workers": 1,
        "train_dataset_storage": "disk",
        "validation_dataset_storage": "disk",
    }
    rows = []
    for role, model in MODELS.items():
        auc = aucs[role]
        history = [
            {
                "epoch": float(epoch),
                "learning_rate": 0.0001,
                "train_loss": 0.25,
                "train_eval_loss": 0.693,
                "train_accuracy": 0.5,
                "train_auc": auc - 0.03 + 0.01 * epoch,
                "val_loss": 0.693,
                "val_accuracy": 0.5,
                "val_auc": auc - 0.05 + 0.01 * epoch,
            }
            for epoch in range(1, 6)
        ]
        training = {
            **common_training,
            "epochs_ran": 5,
            "best_epoch": 5,
            "best_checkpoint_metric": auc,
            "selected_checkpoint": "best",
            "stopped_epoch": 0,
        }
        rows.append(
            {
                "cipher": "SKINNY-64/64",
                "model": model,
                "rounds": 7,
                "seed": seed,
                "samples_per_class": samples_per_class,
                "dataset_label_mode": "balanced_per_class",
                "pairs_per_sample": 4,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "difference_profile": "skinny64_gohr2022_single_key",
                "difference_member": 0,
                "input_difference": 0x2000,
                "train_key": 0,
                "validation_key": 0x1111111111111111,
                "parameter_count": 442466,
                "trainable_parameter_count": 442466,
                "input_bit_order": "project_msb_to_runtime_lsb",
                "metrics": {"auc": auc},
                "history": history,
                "training": training,
            }
        )
    return rows


def test_medium_seed0_pass_authorizes_only_identical_seed1() -> None:
    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="seed0",
        rows=_rows(0, {"true": 0.61, "corrupted": 0.56, "independent": 0.53}),
        expected_seed=0,
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_rtg2a_skinny_medium_seed0_supported"
    assert gate["next_action"] == "run the identical conditional seed1 RTG2-A matrix"
    assert all(gate["protocol_checks"].values())
    assert gate["training_dynamics"]["true"] == {
        "model": MODELS["true"],
        "best_epoch": 5,
        "epochs_ran": 5,
        "first_val_auc": pytest.approx(0.57),
        "best_val_auc": pytest.approx(0.61),
        "final_val_auc": pytest.approx(0.61),
        "best_train_auc": pytest.approx(0.63),
        "train_minus_val_auc_at_best": pytest.approx(0.02),
        "first_to_best_val_auc_gain": pytest.approx(0.04),
        "best_to_final_val_auc_change": pytest.approx(0.0),
    }


def test_single_gate_can_read_immutable_source_and_write_local_outputs(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "remote_archive"
    output_root = tmp_path / "local_readjudication"
    source_root.mkdir()
    results_path = source_root / "results.jsonl"
    results_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in _rows(
                0,
                {"true": 0.61, "corrupted": 0.56, "independent": 0.53},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    source_before = {
        path.relative_to(source_root): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file()
    }

    exit_code = single_gate_main(
        [
            "--run-id",
            "immutable-source",
            "--run-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--seed",
            "0",
            "--no-plot",
        ]
    )

    source_after = {
        path.relative_to(source_root): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    assert exit_code == 0
    assert source_after == source_before
    assert (output_root / "gate.json").is_file()
    assert (output_root / "summary.json").is_file()
    assert (output_root / "validation.json").is_file()
    assert (output_root / "history.csv").is_file()
    assert (output_root / "progress.jsonl").is_file()


def test_retrieved_checkpoints_strictly_replay_result_rows(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    rows = _rows(
        1,
        {"true": 0.61, "corrupted": 0.56, "independent": 0.53},
        samples_per_class=262_144,
    )
    for index, row in enumerate(rows, start=1):
        filename = f"row{index:04d}_{row['model']}_seed1.pt"
        row["training"]["checkpoint_output"] = f"G:\\lxy\\runs\\{filename}"
        row["training"].update(
            {
                "device": "cuda",
                "lr_scheduler": "none",
                "seed": 1,
            }
        )
        model = build_model(
            row["model"],
            input_bits=512,
            hidden_bits=64,
            pair_bits=128,
            structure="SPN",
            model_options=row["training"]["model_options"],
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "history": row["history"],
                "final_metrics": row["metrics"],
                "metadata": row["training"],
            },
            checkpoint_dir / filename,
        )

    report = _verify_checkpoint_payloads(rows, checkpoint_dir)

    assert report["status"] == "pass"
    assert report["file_set_exact"] is True
    assert len(report["entries"]) == 3
    assert all(all(entry["checks"].values()) for entry in report["entries"])

    payload_path = checkpoint_dir / report["actual_files"][0]
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    payload["final_metrics"]["auc"] = 0.5
    torch.save(payload, payload_path)

    failed = _verify_checkpoint_payloads(rows, checkpoint_dir)

    assert failed["status"] == "fail"
    assert any("final_metrics_exact" in error for error in failed["errors"])


def test_medium_seed0_hold_blocks_seed1_and_larger_scale() -> None:
    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="held",
        rows=_rows(0, {"true": 0.552, "corrupted": 0.550, "independent": 0.51}),
        expected_seed=0,
    )

    assert gate["status"] == "hold"
    assert "launch seed1" in gate["blocked_actions"][-1]
    assert any("262144/class" in action for action in gate["blocked_actions"])


def test_medium_seed1_plot_recommends_two_seed_synthesis(tmp_path: Path) -> None:
    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="seed1",
        rows=_rows(1, {"true": 0.61, "corrupted": 0.56, "independent": 0.53}),
        expected_seed=1,
    )
    output = tmp_path / "curves.svg"

    render_skinny_medium_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "下一步汇总两颗种子后裁决扩样" in svg
    assert "推进下一颗种子" not in svg


def test_rtg2b_seed0_pass_preserves_scaled_protocol(tmp_path: Path) -> None:
    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="rtg2b-seed0",
        rows=_rows(
            0,
            {"true": 0.64, "corrupted": 0.60, "independent": 0.51},
            samples_per_class=262_144,
        ),
        expected_seed=0,
        phase="rtg2b",
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_rtg2b_skinny_scale_seed0_supported"
    assert gate["protocol_checks"]["frozen_rtg2b_scale_and_task"] is True
    assert gate["samples_per_class"] == 262_144
    assert gate["train_rows"] == 524_288
    assert gate["validation_rows"] == 262_144
    assert "without launching it automatically" in gate["next_action"]

    output = tmp_path / "rtg2b.svg"
    render_skinny_medium_svg(gate, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新1 RTG2B" in svg
    assert "训练 262144/class，验证 131072/class" in svg
    assert "可准备相同协议的 seed1 复验" in svg


def test_rtg2b_rejects_unscaled_rtg2a_rows() -> None:
    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="rtg2b-wrong-scale",
        rows=_rows(0, {"true": 0.64, "corrupted": 0.60, "independent": 0.51}),
        expected_seed=0,
        phase="rtg2b",
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["frozen_rtg2b_scale_and_task"] is False


def test_rtg2b_launch_gate_holds_until_exact_source_is_published() -> None:
    common = {
        "source_commit": "a" * 40,
        "upstream_ref": "origin/main",
        "joint_gate": {
            "run_id": JOINT_RUN_ID,
            "status": "pass",
            "decision": JOINT_DECISION,
            "protocol_checks": {"complete": True},
            "research_checks": {"supported": True},
        },
        "joint_validation": {"status": "pass"},
        "readiness_status": "pass",
        "source_commit_valid": True,
        "source_commit_exists": True,
        "source_assets_committed": True,
        "source_assets_match": True,
        "plans_match_scale_only": True,
    }

    held = adjudicate_runtime_spn_skinny_rtg2b_launch(
        **common,
        source_commit_published=False,
    )
    passed = adjudicate_runtime_spn_skinny_rtg2b_launch(
        **common,
        source_commit_published=True,
    )

    assert held["status"] == "hold"
    assert held["should_ssh"] is True
    assert held["ssh_allowed"] is False
    assert held["launch_authorized"] is False
    assert passed["status"] == "pass"
    assert passed["decision"] == "innovation1_rtg2b_seed0_remote_launch_authorized"
    assert passed["launch_authorized"] is True


def test_rtg2b_plan_models_and_remote_assets_are_ready() -> None:
    plan_path = ROOT / (
        "configs/experiment/innovation1/"
        "innovation1_spn_skinny64_runtime_e4_scale_rtg2b_262144_seed0.csv"
    )
    config_path = ROOT / (
        "configs/remote/"
        "innovation1_rtg2b_skinny64_general_gf2_scale_262144_seed0_gpu0_20260724.json"
    )
    with plan_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert {row["seed"] for row in rows} == {"0"}
    assert {row["samples_per_class"] for row in rows} == {"262144"}
    assert {row["negative_mode"] for row in rows} == {"encrypted_random_plaintexts"}
    assert {row["pairs_per_sample"] for row in rows} == {"4"}
    parameter_counts = set()
    for row in rows:
        model = build_model(
            row["model_key"],
            input_bits=512,
            hidden_bits=64,
            pair_bits=128,
            model_options=json.loads(row["model_options"]),
        )
        parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
    assert parameter_counts == {442466}

    report = remote_readiness_report(config_path)
    assert report["status"] == "pass", report["errors"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["expected_rows"] == 3
    assert config["epochs"] == 5
    assert config["batch_size"] == 64
    assert config["dataset_cache"] is True
    assert config["dataset_cache_chunk_size"] == 1024
    assert config["dataset_cache_workers"] == 1
    assert config["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
    )

    generated = ROOT / "configs/remote/generated"
    run_script = (
        generated / "run_i1_rtg2b_skinny64_general_gf2_scale_262144_seed0_20260724.cmd"
    ).read_text(encoding="utf-8")
    launch_script = (
        generated
        / "launch_i1_rtg2b_skinny64_general_gf2_scale_262144_seed0_20260724.cmd"
    ).read_text(encoding="utf-8")
    monitor_path = (
        generated
        / "monitor_i1_rtg2b_skinny64_general_gf2_scale_262144_seed0_20260724.sh"
    )
    monitor_script = monitor_path.read_text(encoding="utf-8")
    combined = run_script + launch_script
    assert "cmd.exe /c" in launch_script
    assert "cmd.exe /k" not in combined
    assert "EnableDelayedExpansion" not in combined
    assert "!" not in combined
    assert '--dataset-cache-root "%CACHE_ROOT%"' in run_script
    assert "--dataset-cache-chunk-size 1024" in run_script
    assert "--dataset-cache-workers 1" in run_script
    assert "--phase rtg2b" in run_script
    assert "visual_qa_pending.marker" in run_script + monitor_script
    assert "innovation1_rtg2b_seed0_remote_launch_authorized" in monitor_script
    assert "bounded_start_confirmation_passed" in monitor_script
    assert "exit /b 9 && set" not in monitor_script
    assert "exit /b 9 & set" not in monitor_script
    assert 'cmd.exe /d /s /c \\"if exist ${REMOTE_LAUNCH_ROOT} (' in monitor_script
    assert ") else (set GIT_SSH_COMMAND=ssh -i " in monitor_script
    assert "StrictHostKeyChecking=accept-new&& git clone" in monitor_script
    assert "confirm_started_bounded" in monitor_script
    assert "for attempt in $(seq 1 30)" in monitor_script
    assert "sleep 2" in monitor_script
    launch_returned = monitor_script.index("remote_launcher_returned")
    start_confirmed = monitor_script.index("confirm_started_bounded || exit 8")
    launched_marker = monitor_script.index('touch "${LAUNCHED_MARKER}"')
    assert launch_returned < start_confirmed < launched_marker
    assert "retrieved_from_verified_result_branch.marker" in monitor_script
    assert "scripts/index-results" in monitor_script
    subprocess.run(["bash", "-n", str(monitor_path)], check=True)


def test_rtg2b_seed1_launch_requires_verified_seed0_and_publication() -> None:
    seed0_gate = {
        "run_id": RTG2B_SEED0_RUN_ID,
        "phase": "rtg2b",
        "seed": 0,
        "status": "pass",
        "decision": RTG2B_SEED0_DECISION,
        "protocol_checks": {"complete": True},
        "research_checks": {"supported": True},
        "aucs": {"true": 0.64, "corrupted": 0.60, "independent": 0.51},
        "margins": {
            "true_minus_corrupted": 0.04,
            "true_minus_independent": 0.13,
        },
    }
    common = {
        "source_commit": "a" * 40,
        "upstream_ref": "origin/main",
        "artifact_names": {
            "gate.json",
            "results.jsonl",
            "retrieved_from_verified_result_branch.marker",
            "validation.local.json",
            "visual_qa_passed.marker",
        },
        "seed0_gate": seed0_gate,
        "seed0_validation": {
            "status": "pass",
            "expected_rows": 3,
            "result_rows": 3,
            "errors": [],
        },
        "readiness_status": "pass",
        "source_commit_valid": True,
        "source_commit_exists": True,
        "training_commit_exists": True,
        "protected_changes": [],
        "source_assets_committed": True,
        "source_assets_match": True,
        "plans_match_seed_only": True,
    }

    held = adjudicate_runtime_spn_skinny_rtg2b_seed1_launch(
        **common,
        source_commit_published=False,
    )
    passed = adjudicate_runtime_spn_skinny_rtg2b_seed1_launch(
        **common,
        source_commit_published=True,
    )

    assert held["status"] == "hold"
    assert held["should_ssh"] is True
    assert held["ssh_allowed"] is False
    assert passed["status"] == "pass"
    assert passed["decision"] == "innovation1_rtg2b_seed1_remote_launch_authorized"
    assert passed["launch_authorized"] is True


def test_rtg2b_seed1_launch_fails_closed_without_visual_qa() -> None:
    gate = adjudicate_runtime_spn_skinny_rtg2b_seed1_launch(
        source_commit="a" * 40,
        upstream_ref="origin/main",
        artifact_names={
            "gate.json",
            "results.jsonl",
            "retrieved_from_verified_result_branch.marker",
            "validation.local.json",
        },
        seed0_gate={
            "run_id": RTG2B_SEED0_RUN_ID,
            "phase": "rtg2b",
            "seed": 0,
            "status": "pass",
            "decision": RTG2B_SEED0_DECISION,
            "protocol_checks": {"complete": True},
            "research_checks": {"supported": True},
            "aucs": {"true": 0.64, "corrupted": 0.60, "independent": 0.51},
            "margins": {
                "true_minus_corrupted": 0.04,
                "true_minus_independent": 0.13,
            },
        },
        seed0_validation={
            "status": "pass",
            "expected_rows": 3,
            "result_rows": 3,
            "errors": [],
        },
        readiness_status="pass",
        source_commit_valid=True,
        source_commit_exists=True,
        source_commit_published=True,
        training_commit_exists=True,
        protected_changes=[],
        source_assets_committed=True,
        source_assets_match=True,
        plans_match_seed_only=True,
    )

    assert gate["status"] == "fail"
    assert gate["evidence_checks"]["verified_seed0_artifacts_complete"] is False
    assert gate["launch_authorized"] is False


def test_rtg2b_seed1_plan_and_remote_assets_are_ready() -> None:
    seed0_path = ROOT / (
        "configs/experiment/innovation1/"
        "innovation1_spn_skinny64_runtime_e4_scale_rtg2b_262144_seed0.csv"
    )
    seed1_path = ROOT / (
        "configs/experiment/innovation1/"
        "innovation1_spn_skinny64_runtime_e4_scale_rtg2b_262144_seed1.csv"
    )
    with seed0_path.open(newline="", encoding="utf-8") as handle:
        seed0_rows = list(csv.DictReader(handle))
    with seed1_path.open(newline="", encoding="utf-8") as handle:
        seed1_rows = list(csv.DictReader(handle))

    ignored = {"network", "family", "seed", "evidence", "literature"}
    assert len(seed1_rows) == 3
    assert {row["seed"] for row in seed1_rows} == {"1"}
    assert {row["samples_per_class"] for row in seed1_rows} == {"262144"}
    parameter_counts = set()
    for seed0, seed1 in zip(seed0_rows, seed1_rows, strict=True):
        fields = set(seed0) | set(seed1)
        assert all(seed0[field] == seed1[field] for field in fields - ignored)
        model = build_model(
            seed1["model_key"],
            input_bits=512,
            hidden_bits=64,
            pair_bits=128,
            model_options=json.loads(seed1["model_options"]),
        )
        parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
    assert parameter_counts == {442466}

    config_path = ROOT / (
        "configs/remote/"
        "innovation1_rtg2b_skinny64_general_gf2_scale_262144_seed1_gpu0_20260724.json"
    )
    report = remote_readiness_report(config_path)
    assert report["status"] == "pass", report["errors"]

    generated = ROOT / "configs/remote/generated"
    run_script = (
        generated / "run_i1_rtg2b_skinny64_general_gf2_scale_262144_seed1_20260724.cmd"
    ).read_text(encoding="utf-8")
    launch_script = (
        generated
        / "launch_i1_rtg2b_skinny64_general_gf2_scale_262144_seed1_20260724.cmd"
    ).read_text(encoding="utf-8")
    monitor_path = (
        generated
        / "monitor_i1_rtg2b_skinny64_general_gf2_scale_262144_seed1_20260724.sh"
    )
    monitor_script = monitor_path.read_text(encoding="utf-8")
    combined = run_script + launch_script
    assert "--seed 1" in run_script
    assert "--phase rtg2b" in run_script
    assert "cmd.exe /c" in launch_script
    assert "cmd.exe /k" not in combined
    assert "EnableDelayedExpansion" not in combined
    assert "!" not in combined
    assert "innovation1_rtg2b_skinny_scale_seed0_supported" in launch_script
    assert "innovation1_rtg2b_seed1_remote_launch_authorized" in monitor_script
    assert '--dataset-cache-root "%CACHE_ROOT%"' in run_script
    assert "visual_qa_pending.marker" in run_script + monitor_script
    assert "scripts/index-results" in monitor_script
    subprocess.run(["bash", "-n", str(monitor_path)], check=True)


def test_medium_protocol_mismatch_fails_closed() -> None:
    rows = _rows(0, {"true": 0.61, "corrupted": 0.56, "independent": 0.53})
    rows[1]["samples_per_class"] = 65535

    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="invalid",
        rows=rows,
        expected_seed=0,
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_rtg2a_skinny_medium_protocol_invalid"


@pytest.mark.parametrize("malformation", ["missing_history", "wrong_best_epoch"])
def test_medium_checkpoint_history_mismatch_fails_closed(malformation: str) -> None:
    rows = _rows(0, {"true": 0.61, "corrupted": 0.56, "independent": 0.53})
    if malformation == "missing_history":
        rows[0]["history"] = rows[0]["history"][:-1]
    else:
        rows[0]["training"]["best_epoch"] = 4

    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="invalid-history",
        rows=rows,
        expected_seed=0,
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_rtg2a_skinny_medium_protocol_invalid"
    assert gate["protocol_checks"]["complete_five_epoch_checkpoint_replay"] is False


def _seed_gate(seed: int, *, supported: bool = True) -> dict[str, object]:
    aucs = (
        {"true": 0.61, "corrupted": 0.56, "independent": 0.53}
        if supported
        else {"true": 0.552, "corrupted": 0.550, "independent": 0.51}
    )
    return adjudicate_runtime_spn_skinny_medium(
        run_id=f"{RUN_STEM}_seed{seed}_20260724",
        rows=_rows(seed, aucs),
        expected_seed=seed,
    )


def _rtg2b_seed_gate(seed: int, *, supported: bool = True) -> dict[str, object]:
    aucs = (
        {"true": 0.64, "corrupted": 0.60, "independent": 0.51}
        if supported
        else {"true": 0.552, "corrupted": 0.550, "independent": 0.51}
    )
    return adjudicate_runtime_spn_skinny_medium(
        run_id=f"{RTG2B_RUN_STEM}_seed{seed}_20260724",
        rows=_rows(seed, aucs, samples_per_class=262_144),
        expected_seed=seed,
        phase="rtg2b",
    )


def test_medium_joint_gate_requires_both_seeds_to_pass() -> None:
    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=f"{RUN_STEM}_joint_seed0_seed1_20260724",
        gates=[_seed_gate(0), _seed_gate(1)],
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_rtg2a_skinny_medium_two_seed_supported"
    assert all(gate["protocol_checks"].values())
    assert gate["research_checks"]["both_medium_seeds_supported"] is True
    assert "262144/class" in gate["next_action"]


def test_medium_joint_gate_holds_when_seed1_is_not_supported() -> None:
    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=f"{RUN_STEM}_joint_seed0_seed1_20260724",
        gates=[_seed_gate(0), _seed_gate(1, supported=False)],
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation1_rtg2a_skinny_medium_two_seed_not_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert gate["research_checks"]["both_medium_seeds_supported"] is False


def test_rtg2b_joint_gate_requires_two_scaled_seed_passes() -> None:
    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=f"{RTG2B_RUN_STEM}_joint_seed0_seed1_20260724",
        gates=[_rtg2b_seed_gate(0), _rtg2b_seed_gate(1)],
        phase="rtg2b",
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == ("innovation1_rtg2b_skinny_scale_two_seed_supported")
    assert gate["phase"] == "rtg2b"
    assert gate["samples_per_class"] == 262_144
    assert gate["protocol_checks"]["source_run_ids_match_frozen_rtg2b"] is True
    assert all(gate["protocol_checks"].values())
    assert "target-head cross-cipher adaptation" in gate["next_action"]
    assert any("1000000/class" in action for action in gate["blocked_actions"])


def test_rtg2b_joint_gate_rejects_rtg2a_sources() -> None:
    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=f"{RTG2B_RUN_STEM}_joint_seed0_seed1_20260724",
        gates=[_seed_gate(0), _seed_gate(1)],
        phase="rtg2b",
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == ("innovation1_rtg2b_skinny_scale_joint_protocol_invalid")
    assert gate["protocol_checks"]["source_run_ids_match_frozen_rtg2b"] is False
    assert gate["protocol_checks"]["source_phases_match_joint_phase"] is False


def test_medium_joint_gate_fails_closed_on_duplicate_seed_sources() -> None:
    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=f"{RUN_STEM}_joint_seed0_seed1_20260724",
        gates=[_seed_gate(0), _seed_gate(0)],
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation1_rtg2a_skinny_medium_joint_protocol_invalid"
    )
    assert gate["protocol_checks"]["two_distinct_seed_gates_complete"] is False


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("run_id", "wrong-run", "source_run_ids_match_frozen_rtg2a"),
        (
            "thresholds",
            {"true_auc": 0.54, "control_margin": 0.005},
            "frozen_thresholds_preserved",
        ),
        (
            "aucs",
            {"true": "not-a-number", "corrupted": 0.56, "independent": 0.53},
            "finite_source_metrics",
        ),
        ("protocol_checks", [True], "source_protocol_checks_passed"),
        ("decision", "inconsistent", "source_gate_contracts_consistent"),
    ],
)
def test_medium_joint_gate_fails_closed_on_malformed_source_evidence(
    field: str,
    value: object,
    failed_check: str,
) -> None:
    seed1 = _seed_gate(1)
    seed1[field] = value

    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=f"{RUN_STEM}_joint_seed0_seed1_20260724",
        gates=[_seed_gate(0), seed1],
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation1_rtg2a_skinny_medium_joint_protocol_invalid"
    )
    assert gate["protocol_checks"][failed_check] is False


def test_medium_joint_gate_cli_writes_hashed_source_evidence(tmp_path: Path) -> None:
    source_paths = []
    for seed in (0, 1):
        path = tmp_path / f"seed{seed}" / "gate.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(_seed_gate(seed), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_paths.append(path)
    output_root = tmp_path / "joint"

    status = joint_gate_main(
        [
            "--run-id",
            f"{RUN_STEM}_joint_seed0_seed1_20260724",
            "--seed0-gate",
            str(source_paths[0]),
            "--seed1-gate",
            str(source_paths[1]),
            "--output-root",
            str(output_root),
        ]
    )

    assert status == 0
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (output_root / "validation.json").read_text(encoding="utf-8")
    )
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert validation["status"] == "pass"
    assert len(validation["sources"]) == 2
    assert all(len(source["sha256"]) == 64 for source in validation["sources"])
    assert summary["training_performed"] is False
    assert (output_root / "progress.jsonl").is_file()


def test_rtg2b_joint_gate_cli_writes_scaled_synthesis(tmp_path: Path) -> None:
    source_paths = []
    for seed in (0, 1):
        path = tmp_path / f"seed{seed}" / "gate.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(_rtg2b_seed_gate(seed), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_paths.append(path)
    output_root = tmp_path / "joint"

    status = joint_gate_main(
        [
            "--run-id",
            f"{RTG2B_RUN_STEM}_joint_seed0_seed1_20260724",
            "--seed0-gate",
            str(source_paths[0]),
            "--seed1-gate",
            str(source_paths[1]),
            "--phase",
            "rtg2b",
            "--output-root",
            str(output_root),
        ]
    )

    assert status == 0
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert gate["phase"] == "rtg2b"
    assert summary["task"] == (
        "innovation1_rtg2b_skinny_general_gf2_scale_two_seed_synthesis"
    )


def test_medium_plans_change_only_scale_seed_and_descriptive_fields() -> None:
    import csv

    frozen_fields = {
        "cipher",
        "structure",
        "model_key",
        "architecture_rank",
        "score",
        "rounds",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "train_key",
        "validation_key",
        "key_rotation_interval",
        "sample_structure",
        "integral_active_nibble",
        "difference_profile",
        "difference_member",
        "loss",
        "learning_rate",
        "optimizer",
        "optimizer_state_transition",
        "weight_decay",
        "lr_scheduler",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "target_epochs",
        "pretrain_epochs",
        "model_options",
    }
    with (
        ROOT
        / "configs/experiment/innovation1/innovation1_spn_skinny64_runtime_e4_attribution_t2c_2048_seed0.csv"
    ).open(newline="", encoding="utf-8") as handle:
        anchor = list(csv.DictReader(handle))

    for seed in (0, 1):
        path = ROOT / (
            "configs/experiment/innovation1/"
            f"innovation1_spn_skinny64_runtime_e4_medium_rtg2a_65536_seed{seed}.csv"
        )
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert {row["seed"] for row in rows} == {str(seed)}
        assert {row["samples_per_class"] for row in rows} == {"65536"}
        assert [row["model_key"] for row in rows] == [
            row["model_key"] for row in anchor
        ]
        for expected, observed in zip(anchor, rows, strict=True):
            assert {field: observed[field] for field in frozen_fields} == {
                field: expected[field] for field in frozen_fields
            }


def test_medium_remote_assets_are_ready_and_windows_safe() -> None:
    run_script = (
        ROOT
        / "configs/remote/generated/run_i1_rtg2a_skinny64_general_gf2_medium_65536_20260724.cmd"
    ).read_text(encoding="utf-8")
    launch_script = (
        ROOT
        / "configs/remote/generated/launch_i1_rtg2a_skinny64_general_gf2_medium_65536_20260724.cmd"
    ).read_text(encoding="utf-8")
    monitor_script = (
        ROOT
        / "configs/remote/generated/monitor_i1_rtg2a_skinny64_general_gf2_medium_65536_20260724.sh"
    ).read_text(encoding="utf-8")

    for seed in (0, 1):
        config_path = ROOT / (
            "configs/remote/"
            f"innovation1_rtg2a_skinny64_general_gf2_medium_65536_seed{seed}_gpu0_20260724.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        report = remote_readiness_report(config_path)
        assert report["status"] == "pass", report["errors"]
        assert config["expected_rows"] == 3
        assert config["epochs"] == 5
        assert config["batch_size"] == 64
        assert config["dataset_cache_chunk_size"] == 1024
        assert config["dataset_cache_workers"] == 1
        assert config["dataset_cache_root"].startswith(
            "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        )

    combined = run_script + launch_script
    assert "cmd.exe /c" in launch_script
    assert "cmd.exe /k" not in combined
    assert "EnableDelayedExpansion" not in combined
    assert "!" not in combined
    assert '--dataset-cache-root "%CACHE_ROOT%"' in run_script
    assert "--dataset-cache-chunk-size 1024" in run_script
    assert "--dataset-cache-workers 1" in run_script
    assert '--progress "%LOG_DIR%\\progress.jsonl"' in run_script
    assert "--no-plot" in run_script
    assert "visual_qa_pending.marker" in run_script + monitor_script
    assert "seed0 gate did not pass" in launch_script
    assert "if errorlevel 1 (" in launch_script
    assert "scripts/index-results" in monitor_script
    assert "gate-runtime-spn-skinny-medium" in monitor_script
    assert "gate-runtime-spn-skinny-medium-joint" in monitor_script
    assert f'{RUN_STEM}_joint_seed0_seed1_20260724"' in monitor_script
    assert "adjudicate_joint || exit 5" in monitor_script
    assert "validated_destination_reused" in monitor_script
    assert monitor_script.count('if [[ ! -f "${destination}/gate.json" ]]') == 2
    assert monitor_script.count('return "${gate_exit}"') == 2
    assert "conditional_seed1_launched.marker" in monitor_script
    assert "conditional_seed1_launched_after_resume" in monitor_script
    assert "resumed_medium_pair_joint_adjudicated_and_indexed" in monitor_script
    assert 'SEED1_SOURCE_COMMIT="${1:-}"' in monitor_script
    assert "pushed-seed1-source-commit" in monitor_script
    assert 'LAUNCH_GATE_PATH="${2:-outputs/local_readiness/' in monitor_script
    assert "innovation1_rtg2a_seed1_remote_launch_authorized" in monitor_script
    assert "g.get('should_ssh') is True" in monitor_script
    assert "g.get('ssh_allowed') is True" in monitor_script
    assert "g.get('launch_authorized') is True" in monitor_script
    assert "g.get('source_commit') == '${SEED1_SOURCE_COMMIT}'" in monitor_script
    assert "launch gate does not authorize remote contact" in monitor_script
    assert "stage_fallback_seed0_authorization" in monitor_script
    assert "RAW_RETRIEVAL_NOTICE.txt" in monitor_script
    assert "validation.local.json" in monitor_script
    assert "visual_qa_passed.marker" in monitor_script
    assert 'JOINT_RESULT_ROOT="outputs/remote_results_incomplete"' in monitor_script
    assert "${SEED1_SOURCE_COMMIT} 1 0" in monitor_script
    assert 'revision="$(tr -d' not in monitor_script
    assert "launch_i1_rtg2a" in monitor_script
    assert "medium_pair_complete.marker" in monitor_script
    subprocess.run(
        [
            "bash",
            "-n",
            str(
                ROOT
                / "configs/remote/generated/monitor_i1_rtg2a_skinny64_general_gf2_medium_65536_20260724.sh"
            ),
        ],
        check=True,
    )
    assert "C:\\Users" not in combined.replace(
        "C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519",
        "",
    )


def test_medium_gate_cli_import_does_not_require_matplotlib() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ModuleNotFoundError('matplotlib intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import blockcipher_nd.cli.gate_runtime_spn_skinny_medium
print('import=pass')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import=pass"


def test_medium_gate_scripts_bootstrap_src_for_uninstalled_remote_python() -> None:
    for name in (
        "gate-runtime-spn-skinny-medium",
        "gate-runtime-spn-skinny-medium-joint",
    ):
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "from pathlib import Path" in script
        assert "import sys" in script
        assert (
            'sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))'
            in script
        )
