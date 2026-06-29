from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.engine.matrix_runner import parse_args
from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.engine.modeling import infer_pair_bits
from blockcipher_nd.features.registry import (
    encode_ciphertext_pair,
    is_supported_feature_encoding,
    pair_bits_for_encoding,
)
from blockcipher_nd.planning.invp_gate import gate_invp_only_result
from blockcipher_nd.planning.invp_postprocess import postprocess_invp_only_result
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.registry.cipher_profiles import CipherProfile
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.spn_candidate_evidence import make_candidate_dataset


def test_matrix_runner_lives_in_engine_package():
    args = parse_args(
        ["--ciphers", "speck32", "--models", "mlp", "--rounds", "1", "--dataset-cache-workers", "2"]
    )

    assert args.ciphers == ["speck32"]
    assert args.models == ["mlp"]
    assert args.rounds == [1]
    assert args.learning_rate == 1e-3
    assert args.dataset_cache_workers == 2


def test_official_epoch_cyclic_lr_matches_zhang_wang_schedule():
    from blockcipher_nd.training.optim import make_scheduler
    from blockcipher_nd.training.types import TrainingConfig

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    config = TrainingConfig(
        epochs=12,
        batch_size=4,
        learning_rate=0.0001,
        lr_scheduler="official_cyclic",
        max_learning_rate=0.002,
    )
    scheduler = make_scheduler(optimizer, config, train_size=16)

    observed = []
    for epoch in range(1, 12):
        scheduler.step_epoch(epoch)
        observed.append(round(optimizer.param_groups[0]["lr"], 10))

    assert observed[0] == 0.002
    assert observed[9] == 0.0001
    assert observed[10] == 0.002
    assert observed[:10] == sorted(observed[:10], reverse=True)


def test_zhang_wang_262k_official_cyclic_plan_is_medium_diagnostic():
    plan = "configs/experiment/innovation1/innovation1_spn_present_zhang_wang2022_keras_official_cyclic_r7_262k.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]

    assert task["rounds"] == 7
    assert task["samples_per_class"] == 262144
    assert task["pairs_per_sample"] == 16
    assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert task["negative_mode"] == "encrypted_random_plaintexts"
    assert task["model_key"] == "present_zhang_wang_keras_mcnd"
    assert task["lr_scheduler"] == "official_cyclic"
    assert task["max_learning_rate"] == 0.002
    assert task["checkpoint_metric"] == "val_auc"
    assert task["restore_best_checkpoint"] is True


def test_zhang_wang_1m_official_cyclic_plan_is_single_seed_paper_scale():
    plan = "configs/experiment/innovation1/innovation1_spn_present_zhang_wang2022_keras_official_cyclic_r7_1m.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]

    assert task["rounds"] == 7
    assert task["seed"] == 0
    assert task["samples_per_class"] == 1_000_000
    assert task["pairs_per_sample"] == 16
    assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert task["negative_mode"] == "encrypted_random_plaintexts"
    assert task["model_key"] == "present_zhang_wang_keras_mcnd"
    assert task["lr_scheduler"] == "official_cyclic"
    assert task["max_learning_rate"] == 0.002
    assert task["checkpoint_metric"] == "val_auc"
    assert task["restore_best_checkpoint"] is True
    assert "SINGLE_SEED" in task["matching_evidence"]


def test_present_nibble_paligned_mcnd_smoke_plan_preserves_official_protocol():
    plan = "configs/experiment/innovation1/innovation1_spn_present_nibble_paligned_mcnd_smoke.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]

    assert task["rounds"] == 7
    assert task["samples_per_class"] == 8
    assert task["pairs_per_sample"] == 16
    assert task["feature_encoding"] == "ciphertext_pair_bits"
    assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert task["negative_mode"] == "encrypted_random_plaintexts"
    assert task["model_key"] == "present_nibble_paligned_mcnd"
    assert task["model_options"]["blocks"] == 1
    assert task["model_options"]["spn_mixer_depth"] == 1
    assert "SMOKE only" in task["matching_evidence"]


def test_present_nibble_paligned_mcnd_64k_screen_plan_is_same_budget_diagnostic():
    plan = "configs/experiment/innovation1/innovation1_spn_present_nibble_paligned_mcnd_r7_64k_screen.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_paligned_mcnd",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["samples_per_class"] == 65536
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM 65536/class diagnostic" in task["matching_evidence"]


def test_present_nibble_paligned_mcnd_262k_scalecheck_plan_is_same_budget_diagnostic():
    plan = "configs/experiment/innovation1/innovation1_spn_present_nibble_paligned_mcnd_r7_262k_scalecheck.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_paligned_mcnd",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM 262144/class diagnostic" in task["matching_evidence"]


def test_present_n1v2_262k_structure_ablation_plan_is_same_protocol():
    plan = "configs/experiment/innovation1/innovation1_spn_present_n1v2_structure_ablation_r7_262k.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_paligned_mcnd",
        "present_nibble_paligned_spn_only",
        "present_nibble_paligned_gated_mcnd",
        "present_nibble_shuffled_paligned_gated_mcnd",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["seed"] == 0
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM 262144/class structure ablation" in task["matching_evidence"]


def test_present_n2_transition_backbone_262k_plan_is_same_protocol():
    plan = "configs/experiment/innovation1/innovation1_spn_present_n2_transition_backbone_r7_262k.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_paligned_spn_only",
        "present_nibble_paligned_transition",
        "present_nibble_paligned_transition_residual",
        "present_nibble_shuffled_transition_residual",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["seed"] == 0
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM 262144/class" in task["matching_evidence"]
        assert "N2" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]


def test_present_spn_only_attribution_262k_plan_is_same_protocol():
    plan = "configs/experiment/innovation1/innovation1_spn_present_spn_only_attribution_r7_262k.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_paligned_spn_only",
        "present_nibble_delta_only_spn_only",
        "present_nibble_invp_only_spn_only",
        "present_nibble_shuffled_paligned_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["seed"] == 0
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM 262144/class SPN-only attribution" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]


def test_present_invp_centered_262k_plan_is_same_protocol():
    plan = "configs/experiment/innovation1/innovation1_spn_present_invp_centered_r7_262k.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
        "present_nibble_paligned_spn_only",
        "present_nibble_delta_only_spn_only",
        "present_nibble_shuffled_paligned_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["seed"] == 0
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM 262144/class" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]


def test_present_invp_centered_seed1_fast_262k_plan_is_lean_confirmation():
    plan = "configs/experiment/innovation1/innovation1_spn_present_invp_centered_seed1_fast_r7_262k.csv"
    args = parse_args(["--plan", plan, "--train-eval-interval", "0"])
    tasks = build_tasks(args)

    assert args.train_eval_interval == 0
    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["seed"] == 1
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "seed1 fast confirmation" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]


def test_present_invp_centered_seed1_fast_smoke_plan_is_small():
    plan = "configs/experiment/innovation1/innovation1_spn_present_invp_centered_seed1_fast_smoke.csv"
    args = parse_args(["--plan", plan, "--train-eval-interval", "0"])
    tasks = build_tasks(args)

    assert args.train_eval_interval == 0
    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["seed"] == 1
        assert task["samples_per_class"] == 8
        assert task["pairs_per_sample"] == 16
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert "SMOKE only" in task["matching_evidence"]


def test_present_nibble_paligned_mcnd_1m_seed0_plan_is_same_budget_paper_scale_diagnostic():
    plan = "configs/experiment/innovation1/innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_paligned_mcnd",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["seed"] == 0
        assert task["samples_per_class"] == 1_000_000
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "PAPER_SCALE_SINGLE_SEED 1000000/class" in task["matching_evidence"]


def test_present_nibble_paligned_mcnd_1m_remote_config_uses_parallel_dataset_cache():
    path = Path("configs/remote/innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0_gpu1_20260626.json")
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["plan"].endswith(
        "innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0.csv"
    )
    assert config["expected_rows"] == 2
    assert config["device"] == "cuda:1"
    assert config["dataset_cache"] is True
    assert config["dataset_cache_workers"] == 4
    assert "1000000/class diagnostic" in config["claim_scope"]


def test_present_invp_only_1m_seed0_plan_is_single_row_paper_scale_diagnostic():
    plan = "configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv"
    args = parse_args(["--plan", plan, "--train-eval-interval", "0"])
    tasks = build_tasks(args)

    assert args.train_eval_interval == 0
    assert [task["model_key"] for task in tasks] == ["present_nibble_invp_only_spn_only"]
    task = tasks[0]
    assert task["rounds"] == 7
    assert task["seed"] == 0
    assert task["samples_per_class"] == 1_000_000
    assert task["pairs_per_sample"] == 16
    assert task["feature_encoding"] == "ciphertext_pair_bits"
    assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert task["negative_mode"] == "encrypted_random_plaintexts"
    assert task["lr_scheduler"] == "official_cyclic"
    assert task["max_learning_rate"] == 0.002
    assert task["checkpoint_metric"] == "val_auc"
    assert task["restore_best_checkpoint"] is True
    assert "PAPER_SCALE_SINGLE_SEED 1000000/class" in task["matching_evidence"]
    assert "not multi-seed formal or breakthrough evidence" in task["matching_evidence"]


def test_present_invp_only_1m_remote_config_uses_fast_parallel_dataset_cache():
    path = Path("configs/remote/innovation1_spn_present_invp_only_r7_1m_seed0_gpu1_20260629.json")
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["plan"].endswith("innovation1_spn_present_invp_only_r7_1m_seed0.csv")
    assert config["expected_rows"] == 1
    assert config["device"] == "cuda:1"
    assert config["train_eval_interval"] == 0
    assert config["dataset_cache"] is True
    assert config["dataset_cache_workers"] == 4
    assert "1000000/class InvP-only SPN diagnostic" in config["claim_scope"]


def test_present_invp_only_1m_seed1_plan_is_conditional_confirmation():
    plan = "configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed1.csv"
    args = parse_args(["--plan", plan, "--train-eval-interval", "0"])
    tasks = build_tasks(args)

    assert args.train_eval_interval == 0
    assert [task["model_key"] for task in tasks] == ["present_nibble_invp_only_spn_only"]
    task = tasks[0]
    assert task["rounds"] == 7
    assert task["seed"] == 1
    assert task["samples_per_class"] == 1_000_000
    assert task["pairs_per_sample"] == 16
    assert task["feature_encoding"] == "ciphertext_pair_bits"
    assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert task["negative_mode"] == "encrypted_random_plaintexts"
    assert task["lr_scheduler"] == "official_cyclic"
    assert task["max_learning_rate"] == 0.002
    assert task["checkpoint_metric"] == "val_auc"
    assert task["restore_best_checkpoint"] is True
    assert "CONDITIONAL_PAPER_SCALE_CONFIRMATION" in task["matching_evidence"]


def test_present_invp_only_1m_seed1_remote_config_uses_fast_parallel_dataset_cache():
    path = Path("configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json")
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["plan"].endswith("innovation1_spn_present_invp_only_r7_1m_seed1.csv")
    assert config["expected_rows"] == 1
    assert config["device"] == "cuda:1"
    assert config["train_eval_interval"] == 0
    assert config["dataset_cache"] is True
    assert config["dataset_cache_workers"] == 4
    assert "CONDITIONAL_PAPER_SCALE_CONFIRMATION" in config["claim_scope"]
    assert ">= +0.001 AUC gate" in config["claim_scope"]
    assert ">= +0.003 is the strong single-seed gate" in config["claim_scope"]
    assert "cmd.exe /c" in config["launch_policy"]


def test_removed_legacy_experiment_and_generated_script_roots():
    assert not Path("experiments").exists()
    assert not Path("scripts/generated").exists()
    assert not Path("src/blockcipher_nd/innovation_one.py").exists()
    assert Path("configs/experiment/innovation1").is_dir()
    assert Path("configs/remote").is_dir()


def test_cipher_profile_lives_in_registry():
    profile = CipherProfile.present80()

    assert profile.name == "PRESENT-80"
    assert profile.structure == "SPN"


def test_scripts_are_thin_package_entrypoints():
    wrappers = [
        Path("scripts/train"),
        Path("scripts/smoke"),
        Path("scripts/spn-candidate-evidence"),
        Path("scripts/spn-active-pattern"),
        Path("scripts/audit-spn-features"),
        Path("scripts/validate-results"),
        Path("scripts/gate-invp-result"),
        Path("scripts/postprocess-invp-result"),
        Path("scripts/plot-results"),
        Path("scripts/evaluate-zhang-wang-checkpoint"),
    ]

    for wrapper in wrappers:
        text = wrapper.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        assert len(lines) <= 12, wrapper
        assert "blockcipher_nd.cli" in text


def test_invp_only_gate_selects_seed1_for_strong_delta(tmp_path):
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        json.dumps(
            {
                "model": "present_nibble_invp_only_spn_only",
                "metrics": {
                    "auc": 0.7971,
                    "accuracy": 0.72,
                    "calibrated_accuracy": 0.723,
                    "loss": 0.54,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = gate_invp_only_result(result_path, reference_auc=0.793897025948)

    assert report["status"] == "pass"
    assert report["decision"] == "launch_invp_seed1_confirmation"
    assert report["action"] == "launch_prepared_seed1_1m_config"
    assert report["auc_delta"] > 0.003
    assert report["paligned_mcnd_1m_auc"] == 0.794619119358
    assert report["auc_delta_vs_paligned_mcnd_1m"] > 0.0


def test_invp_only_gate_routes_tied_result_to_ddt_graph(tmp_path):
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        json.dumps(
            {
                "model": "present_nibble_invp_only_spn_only",
                "metrics": {"auc": 0.7941, "calibrated_accuracy": 0.719},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = gate_invp_only_result(result_path, reference_auc=0.793897025948)

    assert report["status"] == "pass"
    assert report["decision"] == "enter_ddt_graph_route"
    assert report["action"] == "implement_ddt_graph_conditional_plan"


def test_invp_only_gate_threshold_boundaries_are_stable(tmp_path):
    reference_auc = 0.793897025948
    cases = [
        (reference_auc + 0.003, "launch_invp_seed1_confirmation"),
        (reference_auc + 0.001, "run_seed1_before_claiming"),
        (reference_auc - 0.001, "enter_ddt_graph_route"),
        (reference_auc - 0.001001, "discard_invp_only_as_main_1m_candidate"),
    ]

    for index, (auc, expected_decision) in enumerate(cases):
        result_path = tmp_path / f"results_{index}.jsonl"
        result_path.write_text(
            json.dumps(
                {
                    "model": "present_nibble_invp_only_spn_only",
                    "metrics": {"auc": auc, "calibrated_accuracy": 0.719},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        report = gate_invp_only_result(result_path, reference_auc=reference_auc)

        assert report["status"] == "pass"
        assert report["decision"] == expected_decision
        assert "auc_delta_vs_paligned_mcnd_1m" in report


def test_invp_only_branch_plan_matches_gate_thresholds():
    invp_plan = Path("docs/experiments/innovation1-invp-only-1m-scale-plan.md").read_text(
        encoding="utf-8"
    )
    ddt_plan = Path("docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md").read_text(
        encoding="utf-8"
    )

    assert "seed0 validated AUC delta over Zhang/Wang 1M anchor `>= +0.001`" in invp_plan
    assert "seed0 delta `< +0.001`" in invp_plan
    assert "InvP-only 1M AUC - Zhang/Wang 1M anchor AUC < +0.001" in ddt_plan
    assert "weakly positive from `+0.001` to `+0.003` AUC" in ddt_plan


def test_invp_only_gate_fails_on_wrong_model_or_missing_auc(tmp_path):
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        json.dumps({"model": "present_zhang_wang_keras_mcnd", "metrics": {"accuracy": 0.7}})
        + "\n",
        encoding="utf-8",
    )

    report = gate_invp_only_result(result_path)

    assert report["status"] == "fail"
    assert report["decision"] == "invalid"
    assert any("expected_model" in error for error in report["errors"])
    assert "missing_or_invalid_metric=auc" in report["errors"]


def test_invp_only_postprocess_writes_validation_plot_history_and_branch_gate(tmp_path):
    plan_path = tmp_path / "plan.csv"
    results_path = tmp_path / "results.jsonl"
    output_dir = tmp_path / "postprocess"
    plan_doc_path = tmp_path / "invp-plan.md"
    plan_doc_path.write_text(
        "# InvP Plan\n\n**Status:** running remotely / tmux monitor active\n",
        encoding="utf-8",
    )
    plan_row = {
        "cipher": "PRESENT-80",
        "structure": "SPN",
        "rounds": "7",
        "seed": "0",
        "samples_per_class": "1000000",
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "train_key": "0x0",
        "validation_key": "0x11111111111111111111",
        "pairs_per_sample": "16",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "integral_active_nibble": "0",
        "key_rotation_interval": "0",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": "0",
        "loss": "mse",
        "learning_rate": "0.0001",
        "optimizer": "adam",
        "weight_decay": "0.00001",
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": "true",
        "early_stopping_patience": "8",
        "early_stopping_min_delta": "0.0001",
        "model_key": "present_nibble_invp_only_spn_only",
    }
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan_row))
        writer.writeheader()
        writer.writerow(plan_row)
    result_row = {
        "cipher": "PRESENT-80",
        "structure": "SPN",
        "rounds": 7,
        "seed": 0,
        "samples_per_class": 1_000_000,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "train_key": 0,
        "validation_key": int("11111111111111111111", 16),
        "pairs_per_sample": 16,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "integral_active_nibble": 0,
        "key_rotation_interval": 0,
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "model": "present_nibble_invp_only_spn_only",
        "selected_model": "present_nibble_invp_only_spn_only",
        "metrics": {"auc": 0.7971, "accuracy": 0.72, "calibrated_accuracy": 0.723, "loss": 0.54},
        "training": {
            "loss": "mse",
            "learning_rate": 0.0001,
            "optimizer": "adam",
            "weight_decay": 0.00001,
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 0.0001,
        },
        "history": [
            {
                "epoch": 1,
                "learning_rate": 0.002,
                "train_accuracy": 0.71,
                "train_auc": 0.79,
                "train_eval_loss": 0.55,
                "val_accuracy": 0.72,
                "val_auc": 0.7971,
                "val_loss": 0.54,
            }
        ],
    }
    results_path.write_text(json.dumps(result_row) + "\n", encoding="utf-8")

    report = postprocess_invp_only_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_invp",
        expected_rows=1,
        plan_doc_path=plan_doc_path,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "launch_invp_seed1_confirmation"
    assert (output_dir / "unit_invp_local_result_gate.json").exists()
    assert (output_dir / "unit_invp_curves.svg").exists()
    assert (output_dir / "unit_invp_history.csv").exists()
    assert (output_dir / "unit_invp_branch_gate.json").exists()
    assert (output_dir / "unit_invp_postprocess_summary.json").exists()
    assert (output_dir / "unit_invp_postprocess_summary.md").exists()
    summary = json.loads((output_dir / "unit_invp_postprocess_summary.json").read_text())
    assert summary["reference_auc"] == 0.793897025948
    assert summary["paligned_mcnd_1m_auc"] == 0.794619119358
    assert summary["auc_delta"] > 0.003
    assert summary["auc_delta_vs_paligned_mcnd_1m"] > 0.0
    assert summary["plan_doc"] == str(plan_doc_path)
    assert summary["next_action"]["branch"] == "seed1_confirmation"
    assert summary["next_action"]["should_launch_remote"] is True
    assert summary["next_action"]["launch_remote_config"].endswith(
        "innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json"
    )
    assert any("seed1" in step for step in summary["next_steps"])
    assert any("tmux watcher or sub-agent" in step for step in summary["next_steps"])
    markdown = (output_dir / "unit_invp_postprocess_summary.md").read_text()
    assert "auc_delta_vs_zhang_wang_1m" in markdown
    assert "auc_delta_vs_paligned_mcnd_1m" in markdown
    assert "launch_invp_seed1_confirmation" in markdown
    assert "plan_doc" in markdown
    assert "Next Steps:" in markdown
    assert "Launch configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json" in markdown
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert "**Status:** completed / postprocessed / branch gated" in plan_doc
    assert "<!-- invp-postprocess:unit_invp:start -->" in plan_doc
    assert "| AUC | `0.797100000000` |" in plan_doc
    assert "| Decision | `launch_invp_seed1_confirmation` |" in plan_doc
    assert "| Next action branch | `seed1_confirmation` |" in plan_doc
    assert "| Next steps | `" in plan_doc
    assert "| Results JSONL | `" in plan_doc

    postprocess_invp_only_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_invp",
        expected_rows=1,
        plan_doc_path=plan_doc_path,
    )
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert plan_doc.count("<!-- invp-postprocess:unit_invp:start -->") == 1
    assert plan_doc.count("### unit_invp Postprocess Result") == 1


def test_invp_only_postprocess_next_steps_route_tied_result_to_ddt(tmp_path):
    plan_path = tmp_path / "plan.csv"
    results_path = tmp_path / "results.jsonl"
    output_dir = tmp_path / "postprocess"
    plan_row = {
        "cipher": "PRESENT-80",
        "structure": "SPN",
        "rounds": "7",
        "seed": "0",
        "samples_per_class": "1000000",
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "train_key": "0x0",
        "validation_key": "0x11111111111111111111",
        "pairs_per_sample": "16",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "integral_active_nibble": "0",
        "key_rotation_interval": "0",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": "0",
        "loss": "mse",
        "learning_rate": "0.0001",
        "optimizer": "adam",
        "weight_decay": "0.00001",
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": "true",
        "early_stopping_patience": "8",
        "early_stopping_min_delta": "0.0001",
        "model_key": "present_nibble_invp_only_spn_only",
    }
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan_row))
        writer.writeheader()
        writer.writerow(plan_row)
    result_row = {
        "cipher": "PRESENT-80",
        "structure": "SPN",
        "rounds": 7,
        "seed": 0,
        "samples_per_class": 1_000_000,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "train_key": 0,
        "validation_key": int("11111111111111111111", 16),
        "pairs_per_sample": 16,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "integral_active_nibble": 0,
        "key_rotation_interval": 0,
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "model": "present_nibble_invp_only_spn_only",
        "selected_model": "present_nibble_invp_only_spn_only",
        "metrics": {"auc": 0.7939, "accuracy": 0.718, "calibrated_accuracy": 0.719, "loss": 0.55},
        "training": {
            "loss": "mse",
            "learning_rate": 0.0001,
            "optimizer": "adam",
            "weight_decay": 0.00001,
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 0.0001,
        },
        "history": [
            {
                "epoch": 1,
                "learning_rate": 0.002,
                "train_accuracy": 0.71,
                "train_auc": 0.79,
                "train_eval_loss": 0.55,
                "val_accuracy": 0.718,
                "val_auc": 0.7939,
                "val_loss": 0.55,
            }
        ],
    }
    results_path.write_text(json.dumps(result_row) + "\n", encoding="utf-8")

    report = postprocess_invp_only_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_invp_tied",
        expected_rows=1,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "enter_ddt_graph_route"
    assert report["next_action"]["branch"] == "ddt_graph"
    assert report["next_action"]["should_launch_remote"] is False
    assert report["next_action"]["requires_implementation"] is True
    assert report["next_action"]["plan_doc"].endswith("innovation1-spn-ddt-graph-conditional-plan.md")
    assert any("DDT graph route" in step for step in report["next_steps"])
    assert not any("seed1" in step.lower() for step in report["next_steps"])


def test_differential_data_layer_has_small_modules():
    generator = Path("src/blockcipher_nd/data/differential/generator.py")
    rows = Path("src/blockcipher_nd/data/differential/rows.py")
    metadata = Path("src/blockcipher_nd/data/differential/metadata.py")
    validation = Path("src/blockcipher_nd/data/differential/validation.py")

    assert generator.exists()
    assert rows.exists()
    assert metadata.exists()
    assert validation.exists()
    assert len(generator.read_text(encoding="utf-8").splitlines()) <= 80


def test_present_inception_mcnd_is_split_by_architecture():
    facade = Path("src/blockcipher_nd/models/structure/spn/present_inception_mcnd.py")
    expected_modules = [
        Path("src/blockcipher_nd/models/structure/spn/present_inception_blocks.py"),
        Path("src/blockcipher_nd/models/structure/spn/present_inception_pair.py"),
        Path("src/blockcipher_nd/models/structure/spn/present_inception_matrix.py"),
        Path("src/blockcipher_nd/models/structure/spn/present_inception_global_matrix.py"),
        Path("src/blockcipher_nd/models/structure/spn/present_inception_pair_stack.py"),
    ]

    assert facade.exists()
    assert len(facade.read_text(encoding="utf-8").splitlines()) <= 60
    for module in expected_modules:
        assert module.exists()


def test_engine_task_runner_is_pipeline_orchestration():
    runner = Path("src/blockcipher_nd/engine/task_runner.py")
    expected_modules = [
        Path("src/blockcipher_nd/engine/task_config.py"),
        Path("src/blockcipher_nd/engine/pretraining.py"),
        Path("src/blockcipher_nd/engine/results.py"),
    ]

    assert runner.exists()
    assert len(runner.read_text(encoding="utf-8").splitlines()) <= 160
    for module in expected_modules:
        assert module.exists()


def test_candidate_evidence_cache_writes_and_reuses(tmp_path):
    progress_path = tmp_path / "progress.jsonl"
    features, labels = make_candidate_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=3,
        samples_per_class=4,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_mcnd",
        key_rotation_interval=2,
        beam_width=2,
        depth=2,
        feature_cache_root=tmp_path / "candidate_cache",
        feature_cache_chunk_size=2,
        progress_output=progress_path,
        split="train",
    )

    assert features.shape == (8, 126)
    assert labels.shape == (8,)
    assert set(np.unique(labels).tolist()) == {0, 1}

    reused_features, reused_labels = make_candidate_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=3,
        samples_per_class=4,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_mcnd",
        key_rotation_interval=2,
        beam_width=2,
        depth=2,
        feature_cache_root=tmp_path / "candidate_cache",
        feature_cache_chunk_size=2,
        progress_output=progress_path,
        split="train",
    )

    assert np.array_equal(np.asarray(features), np.asarray(reused_features))
    assert np.array_equal(np.asarray(labels), np.asarray(reused_labels))
    progress_text = progress_path.read_text(encoding="utf-8")
    assert "candidate_cache_done" in progress_text
    assert "candidate_cache_reuse" in progress_text


def test_zhang_wang_official_anchor_plan_generates_dataset():
    plan = "configs/experiment/innovation1/innovation1_spn_present_zhang_wang2022_keras_official_anchor_smoke.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]
    cipher = build_cipher(task["cipher_key"], rounds=task["rounds"], key=task["train_key"])

    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=task["input_difference"],
            samples_per_class=2,
            seed=task["seed"],
            feature_encoding=task["feature_encoding"],
            pairs_per_sample=task["pairs_per_sample"],
            negative_mode=task["negative_mode"],
            sample_structure=task["sample_structure"],
        )
    )

    assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert dataset.features.shape == (4, 2048)
    assert dataset.metadata["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert dataset.metadata["key_schedule"] == "per_pair_random"
    assert set(np.unique(dataset.labels).tolist()) == {0, 1}


def test_present_nibble_paligned_view_encodes_delta_and_inverse_p_layer():
    cipher = build_cipher("present80", rounds=7, key=0)
    left = 0x0123456789ABCDEF
    right = 0x1111111111111111

    encoded = encode_ciphertext_pair(
        left,
        right,
        width=64,
        feature_encoding="present_nibble_paligned_view",
        cipher=cipher,
    )
    words = _decode_present_cell_matrix_words(encoded, word_count=2, width=64)

    assert is_supported_feature_encoding("present_nibble_paligned_view")
    assert pair_bits_for_encoding(64, "present_nibble_paligned_view") == 128
    assert words[0] == left ^ right
    assert words[1] == cipher.inverse_permutation_layer(left ^ right)


def test_present_nibble_paligned_mcnd_derives_same_spn_view_as_feature_encoder():
    from blockcipher_nd.features.encoders.bitwise import pair_to_bits
    from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
        PresentNibblePAlignedMCNDDistinguisher,
    )

    cipher = build_cipher("present80", rounds=7, key=0)
    left = 0x0123456789ABCDEF
    right = 0x1111111111111111
    model = PresentNibblePAlignedMCNDDistinguisher(
        input_bits=128,
        pair_bits=128,
        base_channels=4,
        blocks=1,
        spn_mixer_depth=1,
    )
    raw = torch.tensor([pair_to_bits(left, right, 64)], dtype=torch.float32)
    expected = encode_ciphertext_pair(
        left,
        right,
        width=64,
        feature_encoding="present_nibble_paligned_view",
        cipher=cipher,
    )

    observed = model._present_nibble_paligned_view(raw).reshape(-1).to(torch.uint8).tolist()

    assert observed == expected


def test_present_nibble_spn_only_attribution_views_are_distinct():
    from blockcipher_nd.features.encoders.bitwise import pair_to_bits
    from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
        PresentNibbleDeltaOnlySpnOnlyDistinguisher,
        PresentNibbleInvPOnlySpnOnlyDistinguisher,
        PresentNibblePAlignedSpnOnlyDistinguisher,
        PresentNibbleShuffledPAlignedSpnOnlyDistinguisher,
    )

    left = 0x0123456789ABCDEF
    right = 0x1111111111111111
    raw = torch.tensor([pair_to_bits(left, right, 64)], dtype=torch.float32)
    common = {
        "input_bits": 128,
        "pair_bits": 128,
        "base_channels": 4,
        "spn_mixer_depth": 1,
    }

    full = PresentNibblePAlignedSpnOnlyDistinguisher(**common)
    delta = PresentNibbleDeltaOnlySpnOnlyDistinguisher(**common)
    invp = PresentNibbleInvPOnlySpnOnlyDistinguisher(**common)
    shuffled = PresentNibbleShuffledPAlignedSpnOnlyDistinguisher(**common)

    full_view = full.spn_encoder.present_nibble_paligned_view(raw)
    delta_view = delta.spn_encoder.present_nibble_paligned_view(raw)
    invp_view = invp.spn_encoder.present_nibble_paligned_view(raw)
    shuffled_view = shuffled.spn_encoder.present_nibble_paligned_view(raw)
    full_words = _decode_present_cell_matrix_words(
        full_view.reshape(-1).to(torch.uint8).tolist(),
        word_count=2,
        width=64,
    )
    delta_words = _decode_present_cell_matrix_words(
        delta_view.reshape(-1).to(torch.uint8).tolist(),
        word_count=1,
        width=64,
    )
    invp_words = _decode_present_cell_matrix_words(
        invp_view.reshape(-1).to(torch.uint8).tolist(),
        word_count=1,
        width=64,
    )
    shuffled_words = _decode_present_cell_matrix_words(
        shuffled_view.reshape(-1).to(torch.uint8).tolist(),
        word_count=2,
        width=64,
    )

    assert full_view.shape == (1, 1, 128)
    assert delta_view.shape == (1, 1, 64)
    assert invp_view.shape == (1, 1, 64)
    assert shuffled_view.shape == (1, 1, 128)
    assert full_words[0] == left ^ right
    assert full_words[0] == delta_words[0]
    assert full_words[1] == invp_words[0]
    assert shuffled_words[0] == left ^ right
    assert shuffled_words[1] != full_words[1]


def test_zhang_wang_official_anchor_uses_independent_key_per_basic_pair(monkeypatch):
    from blockcipher_nd.ciphers.spn.present import Present80

    observed_keys: list[int] = []
    original_encrypt = Present80.encrypt

    def record_key(self, plaintext):
        observed_keys.append(self.key)
        return original_encrypt(self, plaintext)

    monkeypatch.setattr(Present80, "encrypt", record_key)
    cipher = build_cipher("present80", rounds=7, key=0)

    make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x9,
            samples_per_class=1,
            seed=0,
            shuffle=False,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=16,
            negative_mode="encrypted_random_plaintexts",
            sample_structure="zhang_wang_case2_official_mcnd",
        )
    )

    positive_keys = observed_keys[:32]
    negative_keys = observed_keys[32:]
    assert len(positive_keys) == 32
    assert len(negative_keys) == 32
    assert len(set(positive_keys[::2])) == 16
    assert len(set(negative_keys[::2])) == 16
    for pair_index in range(16):
        offset = pair_index * 2
        assert positive_keys[offset] == positive_keys[offset + 1]
        assert negative_keys[offset] == negative_keys[offset + 1]


def test_zhang_wang_official_anchor_model_alias_builds():
    plan = "configs/experiment/innovation1/innovation1_spn_present_zhang_wang2022_keras_official_anchor_smoke.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]
    pair_bits = infer_pair_bits(64, task["feature_encoding"])

    model = build_model(
        task["model_key"],
        input_bits=pair_bits * task["pairs_per_sample"],
        hidden_bits=8,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=task["model_options"],
    )

    assert model.__class__.__name__ == "PresentZhangWangKerasMCNDDistinguisher"
    with torch.no_grad():
        logits = model(torch.zeros(2, pair_bits * task["pairs_per_sample"]))
    assert logits.shape == (2, 1)


def test_present_nibble_paligned_mcnd_model_alias_builds():
    pair_bits = infer_pair_bits(64, "ciphertext_pair_bits")
    model = build_model(
        "present_nibble_paligned_mcnd",
        input_bits=pair_bits * 16,
        hidden_bits=4,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={"blocks": 1, "spn_mixer_depth": 1},
    )

    assert model.__class__.__name__ == "PresentNibblePAlignedMCNDDistinguisher"
    with torch.no_grad():
        logits = model(torch.zeros(2, pair_bits * 16))
    assert logits.shape == (2, 1)


def test_present_nibble_paligned_ablation_models_build_and_forward():
    input_bits = 16 * 128
    features = torch.randint(0, 2, (4, input_bits), dtype=torch.float32)

    for model_key in [
        "present_nibble_paligned_spn_only",
        "present_nibble_delta_only_spn_only",
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
        "present_nibble_shuffled_paligned_spn_only",
        "present_nibble_paligned_gated_mcnd",
        "present_nibble_shuffled_paligned_gated_mcnd",
        "present_nibble_paligned_transition",
        "present_nibble_paligned_transition_residual",
        "present_nibble_shuffled_transition_residual",
    ]:
        model = build_model(
            model_key,
            input_bits=input_bits,
            hidden_bits=32,
            pair_bits=128,
            model_options={
                "blocks": 2,
                "spn_mixer_depth": 1,
                "transition_mixer_depth": 1,
                "activation": "relu",
                "norm": "layernorm",
            },
        )
        logits = model(features)
        assert logits.shape == (4, 1)


def test_result_plan_alignment_is_planning_api(tmp_path):
    plan_path = tmp_path / "plan.csv"
    result_path = tmp_path / "results.jsonl"
    plan_row = {
        "cipher": "speck32",
        "model": "mlp",
        "rounds": "1",
        "seed": "0",
        "samples_per_class": "8",
        "pairs_per_sample": "1",
    }
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan_row))
        writer.writeheader()
        writer.writerow(plan_row)
    result_path.write_text(
        json.dumps(
            {
                "cipher": "speck32",
                "model": "mlp",
                "rounds": 1,
                "seed": 0,
                "samples_per_class": 8,
                "pairs_per_sample": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_result_plan_alignment(plan_path, result_path)

    assert report["status"] == "pass"
    assert report["plan_rows"] == 1
    assert report["result_rows"] == 1


def test_training_history_plot_outputs_svg_and_csv(tmp_path):
    from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv

    results_path = tmp_path / "results.jsonl"
    svg_path = tmp_path / "curves.svg"
    csv_path = tmp_path / "history.csv"
    results_path.write_text(
        json.dumps(
            {
                "cipher": "SPECK32/64",
                "model": "mlp",
                "selected_model": "mlp",
                "rounds": 1,
                "seed": 0,
                "samples_per_class": 8,
                "pairs_per_sample": 1,
                "history": [
                    {
                        "epoch": 1.0,
                        "train_loss": 0.7,
                        "train_eval_loss": 0.69,
                        "train_accuracy": 0.55,
                        "train_auc": 0.58,
                        "val_loss": 0.71,
                        "val_accuracy": 0.5,
                        "val_auc": 0.52,
                        "learning_rate": 0.001,
                    },
                    {
                        "epoch": 2.0,
                        "train_loss": 0.65,
                        "train_eval_loss": 0.64,
                        "train_accuracy": 0.65,
                        "train_auc": 0.7,
                        "val_loss": 0.68,
                        "val_accuracy": 0.6,
                        "val_auc": 0.62,
                        "learning_rate": 0.001,
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    plot_report = plot_jsonl_training_curves(results_path, svg_path)
    csv_report = write_history_csv(results_path, csv_path)

    assert plot_report["series"] == 6
    assert csv_report["rows"] == 2
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "Matplotlib" in svg_text
    assert "Epoch" in svg_text
    assert "Accuracy (%)" in svg_text
    assert "AUC (%)" in svg_text
    assert "Validation summary" in svg_text
    assert "62.0% @ e2" in svg_text
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "train_accuracy" in csv_text
    assert "val_auc" in csv_text


def test_training_history_records_train_and_validation_metrics():
    from blockcipher_nd.ciphers.arx.speck import Speck32_64
    from blockcipher_nd.data.differential import DifferentialDatasetConfig
    from blockcipher_nd.data.differential.generator import make_differential_dataset
    from blockcipher_nd.models.baseline.mlp import MlpDistinguisher
    from blockcipher_nd.training import TrainingConfig, train_binary_classifier

    cipher = Speck32_64(rounds=1, key=0)
    train_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x40,
            samples_per_class=8,
            seed=0,
            shuffle=True,
        )
    )
    validation_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x40,
            samples_per_class=8,
            seed=1,
            shuffle=True,
        )
    )
    model = MlpDistinguisher(input_bits=train_dataset.features.shape[1], hidden_bits=8)

    result = train_binary_classifier(
        model,
        train_dataset,
        validation_dataset,
        TrainingConfig(epochs=1, batch_size=4, device="cpu"),
    )

    epoch = result.history[0]
    for key in [
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_auc",
        "val_loss",
        "val_accuracy",
        "val_auc",
        "learning_rate",
    ]:
        assert key in epoch


def test_training_fast_mode_can_skip_epoch_train_metrics():
    from blockcipher_nd.ciphers.arx.speck import Speck32_64
    from blockcipher_nd.data.differential import DifferentialDatasetConfig
    from blockcipher_nd.data.differential.generator import make_differential_dataset
    from blockcipher_nd.models.baseline.mlp import MlpDistinguisher
    from blockcipher_nd.training import TrainingConfig, train_binary_classifier

    cipher = Speck32_64(rounds=1, key=0)
    train_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x40,
            samples_per_class=8,
            seed=0,
            shuffle=True,
        )
    )
    validation_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x40,
            samples_per_class=8,
            seed=1,
            shuffle=True,
        )
    )
    model = MlpDistinguisher(input_bits=train_dataset.features.shape[1], hidden_bits=8)

    result = train_binary_classifier(
        model,
        train_dataset,
        validation_dataset,
        TrainingConfig(epochs=1, batch_size=4, device="cpu", train_eval_interval=0),
    )

    epoch = result.history[0]
    assert epoch["train_eval_loss"] is None
    assert epoch["train_accuracy"] is None
    assert epoch["train_auc"] is None
    assert epoch["val_loss"] is not None
    assert epoch["val_accuracy"] is not None
    assert epoch["val_auc"] is not None
    assert result.metadata["train_eval_interval"] == 0


def _decode_present_cell_matrix_words(encoded: list[int], *, word_count: int, width: int) -> list[int]:
    cells = [[0, 0, 0, 0] for _ in range((word_count * width) // 4)]
    offset = 0
    for bit_index in range(4):
        for cell in cells:
            cell[bit_index] = encoded[offset]
            offset += 1
    bits = [bit for cell in cells for bit in cell]
    words = []
    for start in range(0, len(bits), width):
        value = 0
        for bit in bits[start : start + width]:
            value = (value << 1) | bit
        words.append(value)
    return words
