from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
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
from blockcipher_nd.planning.invp_attribution_gate import gate_invp_attribution_controls
from blockcipher_nd.planning.ddt_graph_gate import gate_ddt_graph_result
from blockcipher_nd.planning.candidate_trail_gate import gate_candidate_trail_result
from blockcipher_nd.planning.transition_spectrum_gate import gate_transition_spectrum_result
from blockcipher_nd.planning.candidate_trail_postprocess import postprocess_candidate_trail_result
from blockcipher_nd.planning.transition_spectrum_postprocess import postprocess_transition_spectrum_result
from blockcipher_nd.cli.monitor_health import monitor_health_report
from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.planning.invp_postprocess import postprocess_invp_only_result
from blockcipher_nd.planning.invp_attribution_postprocess import postprocess_invp_attribution_controls
from blockcipher_nd.planning.ddt_graph_postprocess import postprocess_ddt_graph_result
from blockcipher_nd.planning.topology_aware_postprocess import postprocess_topology_aware_result
from blockcipher_nd.cli.plan_next_action import plan_next_action
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.cli.monitor_health import _health_status
from blockcipher_nd.registry.cipher_profiles import CipherProfile
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.features.spn_transition_spectrum import (
    present_bit_transition_spectrum_features,
    present_pair_bit_transition_spectrum_features,
)
from blockcipher_nd.tasks.innovation1.spn_candidate_evidence import make_candidate_dataset
from blockcipher_nd.tasks.innovation1 import spn_active_pattern, spn_candidate_evidence, spn_feature_audit
from blockcipher_nd.tasks.innovation1.spn_transition_spectrum import (
    make_transition_spectrum_dataset,
)
from blockcipher_nd.cli import spn_candidate_evidence_matrix, spn_transition_spectrum_matrix
from blockcipher_nd.cli.summarize_spn_evidence import summarize_spn_evidence


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


def test_present_ddt_graph_262k_plan_is_conditional_same_protocol_matrix():
    plan = "configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_paligned_transition_residual",
        "present_nibble_no_ddt_graph",
        "present_nibble_ddt_graph",
        "present_nibble_shuffled_ddt_graph",
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
        assert "MEDIUM 262144/class N3" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]


def test_present_ddt_graph_262k_seed1_plan_is_conditional_confirmation_matrix():
    plan = "configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k_seed1.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_paligned_transition_residual",
        "present_nibble_no_ddt_graph",
        "present_nibble_ddt_graph",
        "present_nibble_shuffled_ddt_graph",
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
        assert "CONDITIONAL_MEDIUM_SEED1 262144/class N3" in task["matching_evidence"]
        assert "support_ddt_graph_route" in task["matching_evidence"]
        assert "weak_ddt_graph_signal" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]


def test_present_ddt_graph_262k_remote_config_is_method_extension_ready():
    path = Path("configs/remote/innovation1_spn_present_ddt_graph_r7_262k_gpu0_20260630.json")
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["plan"].endswith("innovation1_spn_present_ddt_graph_r7_262k.csv")
    assert config["expected_rows"] == 5
    assert config["device"] == "cuda:0"
    assert config["train_eval_interval"] == 0
    assert config["dataset_cache"] is True
    assert config["dataset_cache_root"].startswith("G:\\lxy\\blockcipher-structure-adaptive-nd-runs")
    assert config["dataset_cache_workers"] == 4
    assert "cmd.exe /c" in config["launch_policy"]
    assert "cmd.exe /k" not in config["launch_policy"]
    assert "launch as DDT/topology method-extension after InvP attribution-control support" in config["launch_policy"]
    assert "MEDIUM 262144/class N3 DDT graph method-extension diagnostic" in config["claim_scope"]
    assert "not formal reproduction or breakthrough evidence" in config["claim_scope"]


def test_present_ddt_graph_262k_seed1_remote_config_is_conditional_ready():
    path = Path("configs/remote/innovation1_spn_present_ddt_graph_r7_262k_seed1_gpu1_20260630.json")
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["plan"].endswith("innovation1_spn_present_ddt_graph_r7_262k_seed1.csv")
    assert config["expected_rows"] == 5
    assert config["device"] == "cuda:1"
    assert config["train_eval_interval"] == 0
    assert config["dataset_cache"] is True
    assert config["dataset_cache_root"].startswith("G:\\lxy\\blockcipher-structure-adaptive-nd-runs")
    assert config["dataset_cache_workers"] == 4
    assert "cmd.exe /c" in config["launch_policy"]
    assert "cmd.exe /k" not in config["launch_policy"]
    assert "support_ddt_graph_route or weak_ddt_graph_signal" in config["launch_policy"]
    assert "CONDITIONAL_MEDIUM_SEED1 262144/class N3 DDT graph seed1" in config["claim_scope"]
    assert "interpret as confirmation" in config["claim_scope"]
    assert "interpret as variance check" in config["claim_scope"]
    assert "not formal reproduction or breakthrough evidence" in config["claim_scope"]
    assert remote_readiness_report(path)["status"] == "pass"


def test_present_topology_aware_network_smoke_plan_is_protocol_locked():
    plan = "configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_smoke.csv"
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_p_layer_graph_spn_only",
        "present_nibble_invp_shuffled_p_layer_graph_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["seed"] == 0
        assert task["samples_per_class"] == 8
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "SMOKE only" in task["matching_evidence"]
        assert "not accuracy evidence" in task["matching_evidence"]


def test_present_topology_aware_network_262k_plan_is_lean_same_protocol_matrix():
    plan = "configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_r7_262k.csv"
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_p_layer_graph_spn_only",
        "present_nibble_invp_shuffled_p_layer_graph_spn_only",
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
        assert "MEDIUM 262144/class topology-aware" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]


def test_present_topology_aware_network_seed1_plan_is_conditional_same_protocol_matrix():
    plan = "configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_r7_262k_seed1.csv"
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_p_layer_graph_spn_only",
        "present_nibble_invp_shuffled_p_layer_graph_spn_only",
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
        assert "CONDITIONAL_MEDIUM_SEED1 262144/class topology-aware" in task["matching_evidence"]
        assert "support_topology_aware_network_route" in task["matching_evidence"]
        assert "weak_topology_aware_network_signal" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]


def test_present_topology_aware_network_remote_config_and_assets_are_ready():
    cases = [
        (
            Path("configs/remote/innovation1_spn_present_topology_aware_network_r7_262k_gpu0_20260701.json"),
            Path(
                "configs/remote/generated/"
                "run_i1_spn_topology_aware_network_r7_262k_seed0_gpu0_20260701.cmd"
            ),
            Path(
                "configs/remote/generated/"
                "monitor_i1_spn_topology_aware_network_r7_262k_seed0_gpu0_20260701.sh"
            ),
            "innovation1_spn_present_topology_aware_network_r7_262k.csv",
            "cuda:0",
            "MEDIUM 262144/class topology-aware network diagnostic",
            "weak_ddt_graph_signal",
        ),
        (
            Path("configs/remote/innovation1_spn_present_topology_aware_network_r7_262k_seed1_gpu1_20260701.json"),
            Path(
                "configs/remote/generated/"
                "run_i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701.cmd"
            ),
            Path(
                "configs/remote/generated/"
                "monitor_i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701.sh"
            ),
            "innovation1_spn_present_topology_aware_network_r7_262k_seed1.csv",
            "cuda:1",
            "CONDITIONAL_MEDIUM_SEED1 262144/class topology-aware network",
            "weak_topology_aware_network_signal",
        ),
        (
            Path("configs/remote/innovation1_spn_present_topology_aware_network_r7_262k_gpu0_retry1_20260701.json"),
            Path(
                "configs/remote/generated/"
                "run_i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701.cmd"
            ),
            Path(
                "configs/remote/generated/"
                "monitor_i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701.sh"
            ),
            "innovation1_spn_present_topology_aware_network_r7_262k.csv",
            "cuda:0",
            "MEDIUM 262144/class topology-aware network seed0 retry",
            "launch_stalled",
        ),
    ]

    for config_path, launcher, monitor, plan_name, device, claim_scope, policy_marker in cases:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        launcher_text = launcher.read_text(encoding="utf-8")
        monitor_text = monitor.read_text(encoding="utf-8")
        report = remote_readiness_report(config_path)

        assert config["expected_rows"] == 3
        assert config["plan"].endswith(plan_name)
        assert config["device"] == device
        assert config["dataset_cache"] is True
        assert config["dataset_cache_root"].startswith("G:\\lxy\\blockcipher-structure-adaptive-nd-runs")
        assert config["dataset_cache_workers"] in {4, 8}
        assert "cmd.exe /c" in config["launch_policy"]
        assert "cmd.exe /k" not in config["launch_policy"]
        assert claim_scope in config["claim_scope"]
        assert policy_marker in config["launch_policy"]
        assert report["status"] == "pass"
        assert report["plan_rows"] == 3
        assert "medium_scale_dataset_cache" in report["checked_invariants"]

        assert "cmd.exe /k" not in launcher_text
        assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
        assert "C:\\Users" not in launcher_text
        assert "Desktop" not in launcher_text
        assert "Downloads" not in launcher_text
        assert "AppData" not in launcher_text
        assert plan_name in launcher_text
        assert f"--device {device}" in launcher_text
        assert f"--dataset-cache-workers {config['dataset_cache_workers']}" in launcher_text
        assert "--negative-mode encrypted_random_plaintexts" in launcher_text
        assert "--sample-structure zhang_wang_case2_official_mcnd" in launcher_text
        assert "--dataset-cache-root" in launcher_text

        assert "postprocess-topology-aware-result" in monitor_text
        assert "--expected-rows \"${EXPECTED_ROWS}\"" in monitor_text
        assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text
        assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text


def test_topology_aware_plan_records_launch_stalled_retry():
    plan = Path("docs/experiments/innovation1-spn-topology-aware-network-conditional-plan.md").read_text(
        encoding="utf-8"
    )

    assert "status = launch_stalled / operational failure / no model evidence" in plan
    assert "torch_info_empty_before_git_or_training" in plan
    assert "No started marker, git artifact, progress JSONL" in plan
    assert "not training evidence" in plan
    assert "i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701" in plan
    assert "same protocol and matrix as the original seed0 plan" in plan


def test_present_pairset_aggregation_control_smoke_plans_are_protocol_locked():
    scorer_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pairset_aggregation_control_single_pair_smoke.csv"
    )
    learned_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pairset_aggregation_control_smoke.csv"
    )
    scorer_tasks = build_tasks(parse_args(["--plan", scorer_plan]))
    learned_tasks = build_tasks(parse_args(["--plan", learned_plan]))

    assert [task["model_key"] for task in scorer_tasks] == [
        "present_nibble_invp_only_spn_only",
    ]
    assert [task["model_key"] for task in learned_tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    assert scorer_tasks[0]["pairs_per_sample"] == 1
    for task in [*scorer_tasks, *learned_tasks]:
        assert task["rounds"] == 7
        assert task["seed"] == 0
        assert task["samples_per_class"] == 8
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "SMOKE only" in task["matching_evidence"]
        assert "not accuracy evidence" in task["matching_evidence"]
    for task in learned_tasks:
        assert task["pairs_per_sample"] == 16


def test_present_pairset_aggregation_control_262k_plans_are_staged_and_protocol_locked():
    scorer_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k.csv"
    )
    learned_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pairset_aggregation_control_r7_262k.csv"
    )
    scorer_tasks = build_tasks(parse_args(["--plan", scorer_plan]))
    learned_tasks = build_tasks(parse_args(["--plan", learned_plan]))

    assert [task["model_key"] for task in scorer_tasks] == [
        "present_nibble_invp_only_spn_only",
    ]
    assert [task["model_key"] for task in learned_tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    assert scorer_tasks[0]["pairs_per_sample"] == 1
    for task in [*scorer_tasks, *learned_tasks]:
        assert task["rounds"] == 7
        assert task["seed"] == 0
        assert task["samples_per_class"] == 262144
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM 262144/class" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]
    for task in learned_tasks:
        assert task["pairs_per_sample"] == 16


def test_present_pairset_aggregation_control_remote_configs_are_gated_and_ready():
    plan_doc = Path("docs/experiments/innovation1-pairset-aggregation-control-plan.md").read_text(encoding="utf-8")
    scorer_path = Path(
        "configs/remote/"
        "innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k_gpu1_20260630.json"
    )
    learned_path = Path(
        "configs/remote/innovation1_spn_present_pairset_aggregation_control_r7_262k_gpu1_20260630.json"
    )
    scorer_config = json.loads(scorer_path.read_text(encoding="utf-8"))
    learned_config = json.loads(learned_path.read_text(encoding="utf-8"))

    assert scorer_config["expected_rows"] == 1
    assert learned_config["expected_rows"] == 2
    assert scorer_config["pairset_stage"] == "single_pair_scorer_checkpoint"
    assert learned_config["pairset_stage"] == "learned_pairset_plus_frozen_aggregation_gate"
    assert scorer_config["checkpoint_output"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert learned_config["requires_checkpoint"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert learned_config["frozen_aggregation_output"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    for config in [scorer_config, learned_config]:
        assert config["device"] == "cuda:1"
        assert config["train_eval_interval"] == 0
        assert config["dataset_cache"] is True
        assert config["dataset_cache_root"].startswith("G:\\lxy\\blockcipher-structure-adaptive-nd-runs")
        assert config["dataset_cache_workers"] == 4
        assert "cmd.exe /c" in config["launch_policy"]
        assert "cmd.exe /k" not in config["launch_policy"]
        assert "G:\\lxy" in config["launch_policy"]
        assert "i1_spn_topology_aware_network_r7_262k_seed0_gpu0_20260701" in config["launch_policy"]
        assert "topology-aware run" in config["launch_policy"]
        assert "explicit user route choice selects pair-set attribution" in config["launch_policy"]
        assert "MEDIUM 262144/class pair-set aggregation-control" in config["claim_scope"]
        assert "not formal reproduction or breakthrough evidence" in config["claim_scope"]

    scorer_report = remote_readiness_report(scorer_path)
    learned_report = remote_readiness_report(learned_path)
    assert scorer_report["status"] == "pass"
    assert learned_report["status"] == "pass"
    assert "pairset_aggregation_stage_lock" in scorer_report["checked_invariants"]
    assert "pairset_aggregation_stage_lock" in learned_report["checked_invariants"]
    assert "candidate_trail_protocol_lock" not in learned_report["checked_invariants"]
    assert "pairset_aggregation_stage_lock" in plan_doc
    assert "pairset_stage = single_pair_scorer_checkpoint" in plan_doc
    assert "checkpoint_output under G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in plan_doc
    assert "pairset_stage = learned_pairset_plus_frozen_aggregation_gate" in plan_doc
    assert "requires_checkpoint under G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in plan_doc
    assert "frozen_aggregation_output under G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in plan_doc


def test_pairset_aggregation_readiness_rejects_missing_stage_artifacts(tmp_path):
    config = json.loads(
        Path("configs/remote/innovation1_spn_present_pairset_aggregation_control_r7_262k_gpu1_20260630.json").read_text(
            encoding="utf-8"
        )
    )
    config.pop("requires_checkpoint")
    config["frozen_aggregation_output"] = "C:\\Users\\bad\\frozen_aggregation_summary.json"
    path = tmp_path / "bad_pairset_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "fail"
    assert "pairset_aggregation_stage_lock" in report["checked_invariants"]
    assert any("pairset_aggregation missing requires_checkpoint" in error for error in report["errors"])
    assert any("pairset_aggregation frozen_aggregation_output must stay under" in error for error in report["errors"])


def test_archived_active_pattern_remote_config_is_not_launchable():
    path = Path("configs/remote/innovation1_spn_present_active_pattern_r7_screen_gpu1_20260622.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    report = remote_readiness_report(path)

    assert config["launch_enabled"] is False
    assert config["archive_status"] == "archived_historical_screen_only"
    assert "ARCHIVED SCREEN only" in config["claim_scope"]
    assert "do not launch" in config["launch_policy"]
    assert report["status"] == "fail"
    assert any("launch_enabled=false" in error for error in report["errors"])


def test_legacy_candidate_evidence_remote_config_fails_current_protocol_lock():
    path = Path("configs/remote/innovation1_spn_candidate_evidence_r7_65536_gpu0_20260623.json")
    report = remote_readiness_report(path)

    assert report["status"] == "fail"
    assert any(
        "candidate_trail sample_structure=zhang_wang_case2_mcnd expected=zhang_wang_case2_official_mcnd"
        in error
        for error in report["errors"]
    )
    assert any(
        "candidate_trail validation_key=0xffffffffffffffffffff expected=0x11111111111111111111"
        in error
        for error in report["errors"]
    )
    assert any("candidate_trail key_rotation_interval=1024 expected=0" in error for error in report["errors"])
    assert any("candidate_trail cache root must stay under G:\\lxy" in error for error in report["errors"])


def test_candidate_trail_conditional_remote_smoke_config_is_ready_but_gated():
    path = Path("configs/remote/innovation1_spn_present_candidate_trail_consistency_smoke_gpu1_20260701.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    report = remote_readiness_report(path)

    assert config["plan"] == (
        "configs\\experiment\\innovation1\\innovation1_spn_present_candidate_trail_consistency_smoke.json"
    )
    assert config["expected_rows"] == 1
    assert config["device"] == "cuda:1"
    assert config["negative_mode"] == "encrypted_random_plaintexts"
    assert config["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert config["validation_key"] == "0x11111111111111111111"
    assert config["key_rotation_interval"] == 0
    assert config["feature_cache_root"].startswith("G:\\lxy\\blockcipher-structure-adaptive-nd-runs")
    assert "cmd.exe /c" in config["launch_policy"]
    assert "cmd.exe /k" not in config["launch_policy"]
    assert "conditional launch only after current topology-aware run" in config["launch_policy"]
    assert "i1_spn_topology_aware_network_r7_262k_seed0_gpu0_20260701" in config["launch_policy"]
    assert "branch gate selects candidate-trail" in config["launch_policy"]
    assert "not accuracy evidence" in config["claim_scope"]
    assert "not a medium diagnostic" in config["claim_scope"]

    assert report["status"] == "pass"
    assert report["plan_rows"] == 1
    assert report["expected_rows"] == 1
    assert report["max_samples_per_class"] == 2
    assert report["errors"] == []


def test_candidate_trail_seed1_remote_config_is_ready_but_gate_triggered():
    path = Path("configs/remote/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    report = remote_readiness_report(path)

    assert config["plan"] == (
        "configs\\experiment\\innovation1\\innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1.json"
    )
    assert config["expected_rows"] == 4
    assert config["device"] == "cuda:1"
    assert config["negative_mode"] == "encrypted_random_plaintexts"
    assert config["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert config["validation_key"] == "0x11111111111111111111"
    assert config["key_rotation_interval"] == 0
    assert config["feature_cache_workers"] == 4
    assert config["dataset_cache_workers"] == 4
    assert config["feature_cache_root"].startswith("G:\\lxy\\blockcipher-structure-adaptive-nd-runs")
    assert "cmd.exe /c" in config["launch_policy"]
    assert "cmd.exe /k" not in config["launch_policy"]
    assert "support_candidate_trail_route or weak_candidate_trail_signal" in config["launch_policy"]
    assert "not paper-scale" in config["claim_scope"]
    assert "not formal" in config["claim_scope"]

    assert report["status"] == "pass"
    assert report["plan_rows"] == 4
    assert report["expected_rows"] == 4
    assert report["max_samples_per_class"] == 262144
    assert report["errors"] == []
    assert "candidate_trail_protocol_lock" in report["checked_invariants"]
    assert "medium_scale_dataset_cache" in report["checked_invariants"]
    assert not any("feature_cache_workers=1" in warning for warning in report["warnings"])

    plan = json.loads(
        Path("configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["common"]["seed"] == 1
    assert plan["common"]["feature_cache_workers"] == 4
    assert plan["rows"][0]["row_type"] == "external_anchor"
    assert plan["rows"][0]["model"] == "present_nibble_invp_only_spn_only"
    assert plan["rows"][0]["anchor_auc"] == pytest.approx(0.7931563919119071)
    assert {row["model"] for row in plan["rows"][1:]} == {"linear", "mlp", "shuffled_cells"}


def test_candidate_trail_consistency_plan_is_gated_to_current_protocol():
    plan = Path("docs/experiments/innovation1-candidate-trail-consistency-plan.md").read_text(
        encoding="utf-8"
    )

    assert "active next data-representation branch" in plan
    assert "i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630" in plan
    assert "`zhang_wang_case2_official_mcnd`" in plan
    assert "`encrypted_random_plaintexts`" in plan
    assert "`0x11111111111111111111`" in plan
    assert "sample_structure = zhang_wang_case2_mcnd" in plan
    assert "scripts/check-remote-readiness enforces candidate-trail/candidate-evidence" in plan
    assert "innovation1_spn_present_candidate_trail_consistency_smoke_gpu1_20260701.json" in plan
    assert "candidate_trail_consistency_linear" in plan
    assert "candidate_trail_consistency_mlp" in plan
    assert "missing feature_mode or feature_mode not in {cell_structured, cell_structured_shuffled}" in plan
    assert "feature_mode in {cell_structured, cell_structured_shuffled}" in plan
    assert "Local cell-structured candidate-trail features are implemented" in plan
    assert "feature_mode = cell_structured" in plan
    assert "feature_mode = cell_structured_shuffled" in plan
    assert "candidate_trail_consistency_shuffled_cells" in plan
    assert "expected_rows = 4" in plan
    assert "candidate-trail consistency medium diagnostic positive" in plan
    assert "formal route evidence" in plan
    assert "i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701" in plan
    assert "decision = stop_topology_aware_network_route" in plan
    assert "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702" in plan
    assert "i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702" in plan
    assert "innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.json" in plan
    assert "feature_cache_workers = 4" in plan
    assert (
        "configs/remote/generated/run_i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702.cmd"
        in plan
    )
    assert (
        "configs/remote/generated/run_i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.cmd"
        in plan
    )
    assert "topology-aware network   -> true P-layer message passing over InvP cells" in plan


def test_bit_transition_spectrum_plan_is_conditional_next_branch():
    plan = Path("docs/experiments/innovation1-bit-transition-spectrum-plan.md").read_text(
        encoding="utf-8"
    )

    assert "conditional next-branch plan" in plan
    assert "do_not_launch_until" in plan
    assert "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702" in plan
    assert "candidate-trail seed0 decision = stop_candidate_trail_route" in plan
    assert "candidate-trail seed0 decision = support_candidate_trail_route" in plan
    assert "launch candidate-trail 262144/class seed1 first" in plan
    assert "`zhang_wang_case2_official_mcnd`" in plan
    assert "`encrypted_random_plaintexts`" in plan
    assert "`0x11111111111111111111`" in plan
    assert "`present_nibble_invp_only_spn_only`" in plan
    assert "`bit_transition_spectrum_linear`" in plan
    assert "`bit_transition_spectrum_mlp`" in plan
    assert "`bit_transition_spectrum_shuffled_p`" in plan
    assert "true transition-spectrum route >= shuffled-P control + 0.001 AUC" in plan
    assert "do not create or launch the medium remote config" in plan
    assert "scripts/spn-transition-spectrum-matrix" in plan
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in plan
    assert "implementation_status = local route implemented" in plan
    assert "streamed disk-backed feature cache" in plan
    assert "src/blockcipher_nd/planning/transition_spectrum_postprocess.py" in plan
    assert "no code implementation yet" not in plan


def test_trail_family_consistency_plan_is_conditional_next_hypothesis():
    plan = Path("docs/experiments/innovation1-trail-family-consistency-plan.md").read_text(
        encoding="utf-8"
    )

    assert "conditional next-hypothesis plan only" in plan
    assert "do_not_launch_until" in plan
    assert "candidate-trail seed0 decision = stop_candidate_trail_route" in plan
    assert "bit-transition-spectrum seed0 decision = stop_transition_spectrum_route" in plan
    assert "candidate-trail seed0 decision = support_candidate_trail_route" in plan
    assert "support_transition_spectrum_route" in plan
    assert "`PRESENT-80`" in plan
    assert "`zhang_wang_case2_official_mcnd`" in plan
    assert "`encrypted_random_plaintexts`" in plan
    assert "`0x11111111111111111111`" in plan
    assert "`present_nibble_invp_only_spn_only`" in plan
    assert "`trail_family_consistency_linear`" in plan
    assert "`trail_family_consistency_mlp`" in plan
    assert "`trail_family_consistency_false_family`" in plan
    assert "true trail-family route >= false-family control + 0.001 AUC" in plan
    assert "do not create or launch the medium remote config" in plan
    assert "scripts/spn-trail-family-matrix" in plan
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in plan
    assert "implementation_status = not implemented" in plan
    assert "remote_config_status = do not create until trigger" in plan


def test_bit_transition_spectrum_features_are_stable_and_controlled():
    cipher = build_cipher("present80", 7, key=0)
    pairs = [
        (0x0123456789ABCDEF, 0x0123456789ABCDE6),
        (0xFEDCBA9876543210, 0xFEDCBA9876543209),
    ]

    true_features = present_bit_transition_spectrum_features(
        pairs,
        width=64,
        cipher=cipher,
        shuffled=False,
    )
    shuffled_features = present_bit_transition_spectrum_features(
        pairs,
        width=64,
        cipher=cipher,
        shuffled=True,
    )

    assert true_features.dtype == np.float32
    assert true_features.shape == shuffled_features.shape == (1020,)
    assert np.isfinite(true_features).all()
    assert np.isfinite(shuffled_features).all()
    assert not np.array_equal(true_features, shuffled_features)


def test_bit_transition_spectrum_single_pair_preserves_active_count():
    cipher = build_cipher("present80", 7, key=0)
    features = present_pair_bit_transition_spectrum_features(
        0x0000000000000000,
        0x00000000000000FF,
        width=64,
        cipher=cipher,
    )

    assert features.dtype == np.float32
    assert features.shape == (338,)
    assert features[0] == features[1]
    assert np.isfinite(features).all()


def test_bit_transition_spectrum_rejects_empty_pairset():
    cipher = build_cipher("present80", 7, key=0)

    with pytest.raises(ValueError, match="pairs must not be empty"):
        present_bit_transition_spectrum_features([], width=64, cipher=cipher)


def test_bit_transition_spectrum_smoke_config_preserves_official_protocol():
    config_path = Path(
        "configs/experiment/innovation1/innovation1_spn_present_bit_transition_spectrum_smoke.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["run_id"] == "i1_bit_transition_spectrum_smoke_20260702"
    assert config["common"]["samples_per_class"] == 2
    assert config["common"]["pairs_per_sample"] == 1
    assert config["common"]["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert config["common"]["negative_mode"] == "encrypted_random_plaintexts"
    assert config["common"]["validation_key"] == "0x11111111111111111111"
    assert config["common"]["key_rotation_interval"] == 0
    assert [row.get("model") for row in config["rows"]] == [
        "present_nibble_invp_only_spn_only",
        "linear",
        "mlp",
        "shuffled_p",
    ]
    assert "not accuracy evidence" in config["description"]


def test_bit_transition_spectrum_dataset_cache_writes_and_reuses(tmp_path):
    progress_path = tmp_path / "transition_progress.jsonl"
    features, labels = make_transition_spectrum_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=17,
        samples_per_class=3,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_official_mcnd",
        key_rotation_interval=0,
        feature_cache_root=tmp_path / "transition_cache",
        feature_cache_chunk_size=2,
        feature_cache_workers=1,
        progress_output=progress_path,
        split="train",
    )
    reused_features, reused_labels = make_transition_spectrum_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=17,
        samples_per_class=3,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_official_mcnd",
        key_rotation_interval=0,
        feature_cache_root=tmp_path / "transition_cache",
        feature_cache_chunk_size=2,
        feature_cache_workers=1,
        progress_output=progress_path,
        split="train",
    )

    assert features.shape == (6, 1020)
    assert labels.shape == (6,)
    assert set(np.unique(labels).tolist()) == {0, 1}
    assert np.array_equal(np.asarray(features), np.asarray(reused_features))
    assert np.array_equal(np.asarray(labels), np.asarray(reused_labels))
    progress_text = progress_path.read_text(encoding="utf-8")
    assert "transition_spectrum_cache_flush_start" in progress_text
    assert "transition_spectrum_cache_done" in progress_text
    assert "transition_spectrum_cache_reuse" in progress_text
    metadata_path = next((tmp_path / "transition_cache").glob("train/*/metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "feature_cache_workers" not in metadata
    assert '"workers": 1' in progress_text


def test_bit_transition_spectrum_cache_reuses_across_worker_counts(tmp_path):
    progress_path = tmp_path / "transition_worker_reuse_progress.jsonl"
    cache_root = tmp_path / "transition_worker_reuse_cache"
    common = {
        "rounds": 7,
        "key": 0,
        "input_difference": 0x9,
        "seed": 17,
        "samples_per_class": 4,
        "pairs_per_sample": 2,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "key_rotation_interval": 0,
        "feature_cache_root": cache_root,
        "feature_cache_chunk_size": 2,
        "progress_output": progress_path,
        "split": "train",
    }

    features, labels = make_transition_spectrum_dataset(**common, feature_cache_workers=1)
    reused_features, reused_labels = make_transition_spectrum_dataset(**common, feature_cache_workers=2)

    assert np.array_equal(np.asarray(features), np.asarray(reused_features))
    assert np.array_equal(np.asarray(labels), np.asarray(reused_labels))
    progress_text = progress_path.read_text(encoding="utf-8")
    assert '"workers": 1' in progress_text
    assert '"workers": 2' in progress_text
    assert "transition_spectrum_cache_reuse" in progress_text
    assert len(list(cache_root.glob("train/*/metadata.json"))) == 1


def test_bit_transition_spectrum_matrix_outputs_anchor_and_candidate_rows(tmp_path):
    config = tmp_path / "transition_matrix.json"
    output = tmp_path / "transition_matrix.jsonl"
    config.write_text(
        json.dumps(
            {
                "output": str(output),
                "common": {
                    "rounds": 7,
                    "seed": 0,
                    "samples_per_class": 2,
                    "pairs_per_sample": 1,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "zhang_wang_case2_official_mcnd",
                    "difference_profile": "present_zhang_wang2022_mcnd",
                    "difference_member": 0,
                    "train_key": "0x00000000000000000000",
                    "validation_key": "0x11111111111111111111",
                    "key_rotation_interval": 0,
                    "feature_cache_root": str(tmp_path / "cache"),
                    "feature_cache_chunk_size": 1,
                    "epochs": 1,
                    "learning_rate": 0.01,
                    "device": "cpu",
                },
                "rows": [
                    {
                        "row_type": "external_anchor",
                        "model": "present_nibble_invp_only_spn_only",
                        "anchor_auc": 0.52,
                        "anchor_calibrated_accuracy": 0.5,
                    },
                    {"model": "linear", "progress_output": str(tmp_path / "linear_progress.jsonl")},
                    {"model": "mlp", "progress_output": str(tmp_path / "mlp_progress.jsonl")},
                    {"model": "shuffled_p", "progress_output": str(tmp_path / "shuffled_progress.jsonl")},
                ],
            }
        ),
        encoding="utf-8",
    )

    spn_transition_spectrum_matrix.main(["--config", str(config)])

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["model"] for row in rows] == [
        "present_nibble_invp_only_spn_only",
        "bit_transition_spectrum_linear",
        "bit_transition_spectrum_mlp",
        "bit_transition_spectrum_shuffled_p",
    ]
    assert rows[0]["row_type"] == "external_anchor"
    assert rows[1]["feature_route"] == "bit_transition_spectrum"
    assert rows[1]["feature_dim"] == 1020
    assert rows[3]["shuffled_p"] is True
    assert rows[1]["auc"] == rows[1]["val_auc"]
    assert rows[2]["calibrated_accuracy"] == rows[2]["metrics"]["calibrated_accuracy"]


def test_present_pairset_aggregation_control_remote_launch_assets_are_stage_aware():
    launcher = Path(
        "configs/remote/generated/"
        "run_i1_pairset_aggregation_control_r7_262k_seed0_gpu1_20260630.cmd"
    )
    monitor = Path(
        "configs/remote/generated/"
        "monitor_i1_pairset_aggregation_control_r7_262k_seed0_gpu1_20260630.sh"
    )
    launcher_text = launcher.read_text(encoding="utf-8")
    monitor_text = monitor.read_text(encoding="utf-8")

    assert "cmd.exe /k" not in launcher_text
    assert "cmd.exe /k" not in monitor_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "C:\\Users" not in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "i1_pairset_single_pair_scorer_r7_262k_seed0_gpu1_20260630" in launcher_text
    assert "--checkpoint-output \"%CHECKPOINT_DIR%\\single_pair_invp.pt\"" in launcher_text
    assert "innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k.csv" in launcher_text
    assert "innovation1_spn_present_pairset_aggregation_control_r7_262k.csv" in launcher_text
    assert "scripts\\evaluate-pairset-aggregation" in launcher_text
    assert "--scorer-pairs-per-sample 1" in launcher_text
    assert "--scorer-model-options" not in launcher_text
    assert "--aggregation-mode sum_logodds" in launcher_text
    assert "--negative-mode encrypted_random_plaintexts" in launcher_text
    assert "--sample-structure zhang_wang_case2_official_mcnd" in launcher_text
    assert "--dataset-cache-root" in launcher_text
    assert "stage_a_done" in launcher_text
    assert "stage_b_done" in launcher_text
    assert "frozen_aggregation_done" in launcher_text

    assert "checkpoints" in monitor_text
    assert "single_pair_invp.pt" in monitor_text
    assert "frozen_aggregation_summary.json" in monitor_text
    assert "postprocess-pairset-aggregation" in monitor_text
    assert "--expected-rows \"${EXPECTED_ROWS}\"" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text
    assert "completed_missing_or_incomplete_results" in monitor_text


def test_present_ddt_graph_remote_launch_assets_are_g_lxy_scoped():
    assets = [
        (
            Path("configs/remote/generated/run_i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630.cmd"),
            Path("configs/remote/generated/monitor_i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630.sh"),
            "innovation1_spn_present_ddt_graph_r7_262k.csv",
            "cuda:0",
        ),
        (
            Path("configs/remote/generated/run_i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630.cmd"),
            Path("configs/remote/generated/monitor_i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630.sh"),
            "innovation1_spn_present_ddt_graph_r7_262k_seed1.csv",
            "cuda:1",
        ),
    ]

    for launcher, monitor, plan_name, device in assets:
        launcher_text = launcher.read_text(encoding="utf-8")
        monitor_text = monitor.read_text(encoding="utf-8")

        assert "cmd.exe /k" not in launcher_text
        assert "cmd.exe /k" not in monitor_text
        assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
        assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
        assert "C:\\Users" not in launcher_text
        assert "Desktop" not in launcher_text
        assert "Downloads" not in launcher_text
        assert "AppData" not in launcher_text
        assert plan_name in launcher_text
        assert f"--device {device}" in launcher_text
        assert "--negative-mode encrypted_random_plaintexts" in launcher_text
        assert "--sample-structure zhang_wang_case2_official_mcnd" in launcher_text
        assert "--dataset-cache-root" in launcher_text
        assert "postprocess-ddt-graph-result" in monitor_text


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
    assert ">= +0.001 weak-positive AUC gate" in task["matching_evidence"]
    assert ">= +0.003 is the strong single-seed gate" in task["matching_evidence"]


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


def test_present_invp_attribution_controls_1m_plan_is_lean_same_protocol():
    plan = "configs/experiment/innovation1/innovation1_spn_present_invp_attribution_controls_r7_1m_seed0.csv"
    args = parse_args(["--plan", plan, "--train-eval-interval", "0"])
    tasks = build_tasks(args)

    assert args.train_eval_interval == 0
    assert [task["model_key"] for task in tasks] == [
        "present_nibble_delta_only_spn_only",
        "present_nibble_shuffled_paligned_spn_only",
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
        assert "PAPER_SCALE_ATTRIBUTION_CONTROL 1000000/class" in task["matching_evidence"]
        assert "not formal breakthrough evidence" in task["matching_evidence"]


def test_present_invp_attribution_controls_1m_remote_config_is_ready_shape():
    path = Path(
        "configs/remote/innovation1_spn_present_invp_attribution_controls_r7_1m_seed0_gpu0_20260630.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["plan"].endswith("innovation1_spn_present_invp_attribution_controls_r7_1m_seed0.csv")
    assert config["expected_rows"] == 2
    assert config["device"] == "cuda:0"
    assert config["train_eval_interval"] == 0
    assert config["dataset_cache"] is True
    assert config["dataset_cache_workers"] == 4
    assert "PAPER_SCALE_ATTRIBUTION_CONTROL" in config["claim_scope"]
    assert "DeltaC-only and shuffled-P controls" in config["claim_scope"]
    assert "cmd.exe /c" in config["launch_policy"]
    assert "G:\\lxy" in config["launch_policy"]


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
        Path("scripts/spn-candidate-evidence-matrix"),
        Path("scripts/spn-transition-spectrum-matrix"),
        Path("scripts/spn-active-pattern"),
        Path("scripts/audit-spn-features"),
        Path("scripts/validate-results"),
        Path("scripts/gate-invp-result"),
        Path("scripts/gate-candidate-trail"),
        Path("scripts/gate-transition-spectrum"),
        Path("scripts/postprocess-invp-result"),
        Path("scripts/postprocess-candidate-trail"),
        Path("scripts/postprocess-transition-spectrum"),
        Path("scripts/monitor-health"),
        Path("scripts/check-remote-readiness"),
        Path("scripts/plan-next-action"),
        Path("scripts/summarize-spn-evidence"),
        Path("scripts/plot-results"),
        Path("scripts/evaluate-zhang-wang-checkpoint"),
    ]

    for wrapper in wrappers:
        text = wrapper.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        assert len(lines) <= 12, wrapper
        assert "blockcipher_nd.cli" in text


def test_summarize_spn_evidence_reports_route_level_state(tmp_path):
    root = tmp_path / "remote_results"
    invp0 = root / "i1_invp_only_r7_1m_seed0_gpu1_20260629"
    invp1 = root / "i1_invp_only_r7_1m_seed1_gpu1_20260629"
    attribution = root / "i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630"
    ddt = root / "i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630"
    topology = root / "i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701"
    candidate = root / "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702"
    for path in [invp0, invp1, attribution, ddt, topology, candidate / "monitor"]:
        path.mkdir(parents=True)

    _write_test_json(
        invp0 / "i1_invp_only_r7_1m_seed0_gpu1_20260629_postprocess_summary.json",
        {
            "run_id": "i1_invp_only_r7_1m_seed0_gpu1_20260629",
            "status": "pass",
            "validation_status": "pass",
            "decision": "launch_invp_seed1_confirmation",
            "auc": 0.797470988906,
            "calibrated_accuracy": 0.721351,
            "claim_scope": "1000000/class single-seed gate only",
        },
    )
    _write_test_json(
        invp1 / "i1_invp_only_r7_1m_seed1_gpu1_20260629_postprocess_summary.json",
        {
            "run_id": "i1_invp_only_r7_1m_seed1_gpu1_20260629",
            "status": "pass",
            "validation_status": "pass",
            "decision": "confirm_invp_route",
            "auc": 0.797347588554,
            "calibrated_accuracy": 0.721855,
            "claim_scope": "1000000/class seed1 confirmation",
        },
    )
    _write_test_json(
        attribution / "i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630_postprocess_summary.json",
        {
            "run_id": "i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630",
            "status": "pass",
            "validation_status": "pass",
            "decision": "support_invp_structural_attribution",
            "invp_seed0_auc": 0.797470988906,
            "invp_seed1_auc": 0.797347588554,
            "invp_min_auc": 0.797347588554,
            "max_control_auc": 0.793621524954,
            "attribution_margin": 0.0037260636,
            "claim_scope": "1000000/class attribution-control gate",
        },
    )
    _write_test_json(
        ddt / "i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630_postprocess_summary.json",
        {
            "run_id": "i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630",
            "status": "pass",
            "validation_status": "pass",
            "decision": "weak_ddt_graph_signal",
            "margin_vs_best_control_auc": 0.000626281631,
            "claim_scope": "262144/class medium diagnostic DDT graph gate",
            "next_action": {"branch": "ddt_graph_seed1_variance_check", "should_launch_remote": True},
        },
    )
    _write_test_json(
        topology / "i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701_postprocess_summary.json",
        {
            "run_id": "i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_topology_aware_network_route",
            "claim_scope": "262144/class medium diagnostic topology-aware network gate",
        },
    )
    (candidate / "monitor" / "monitor.log").write_text("2026-07-02T16:54:00+08:00 running\n")
    (candidate / "logs").mkdir()
    (candidate / "logs" / "candidate_trail_linear_progress.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "candidate_cache_start", "total_rows": 524288}),
                json.dumps(
                    {
                        "event": "candidate_cache_positive_chunk",
                        "rows_done": 114688,
                        "total_rows": 524288,
                        "class_rows_done": 114688,
                        "class_total": 262144,
                        "chunk_rows": 8192,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    assert report["status"] == "pass"
    assert report["summaries_scanned"] == 5
    assert report["strongest_route"]["decision"] == "support_invp_structural_attribution"
    assert report["strongest_route"]["evidence_level"] == (
        "two_seed_1000000_class_positive_with_attribution_control"
    )
    assert report["active_recommendation"]["branch"] == "wait_for_candidate_trail_result"
    assert report["active_recommendation"]["run_id"] == "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702"
    assert report["active_recommendation"]["status"] in {"running", "stale_monitor"}
    assert report["active_recommendation"]["postprocess_allowed"] is False
    assert "heartbeat" in report["active_recommendation"]
    assert "needs_main_thread_intervention" in report["active_recommendation"]
    assert report["active_recommendation"]["progress_summary"]["cache_class_rows_done"] == 114688
    assert report["active_recommendation"]["progress_summary"]["cache_chunk_rows"] == 8192
    assert report["active_recommendation"]["progress_summary"]["cache_chunk_index"] == 14
    assert report["active_recommendation"]["progress_summary"]["cache_class_chunk_index"] == 14
    assert report["active_recommendation"]["progress_summary"]["line_count"] == 2
    assert report["active_recommendation"]["progress_summary"]["parsed_line_count"] == 2
    assert report["active_recommendation"]["progress_summary"]["cache_class_progress_percent"] == 43.75
    assert report["active_recommendation"]["progress_summary"]["cache_total_progress_percent"] == 21.875
    assert "scripts/monitor-health" in report["active_recommendation"]["monitor_health_command"]
    assert "scripts/postprocess-candidate-trail" in report["active_recommendation"]["postprocess_when_ready_command"]
    assert "--expected-rows 4" in report["active_recommendation"]["postprocess_when_ready_command"]
    by_run_id = {route["run_id"]: route for route in report["routes"]}
    assert by_run_id["i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630"]["evidence_scale"] == "medium_diagnostic"
    assert by_run_id["i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630"]["route_state"] == "superseded"
    assert by_run_id["i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630"]["effective_next_action"] == {
        "should_launch_remote": False,
        "reason": "superseded_by_later_route_decision",
    }
    assert by_run_id["i1_invp_only_r7_1m_seed0_gpu1_20260629"]["evidence_scale"] == (
        "paper_scale_single_seed"
    )
    assert by_run_id["i1_invp_only_r7_1m_seed0_gpu1_20260629"]["route_state"] == "superseded"
    assert by_run_id["i1_invp_only_r7_1m_seed0_gpu1_20260629"]["effective_next_action"] == {
        "should_launch_remote": False,
        "reason": "superseded_by_later_route_decision",
    }
    assert by_run_id["i1_invp_only_r7_1m_seed1_gpu1_20260629"]["route_state"] == "superseded"


def test_summarize_spn_evidence_exposes_stale_candidate_monitor(tmp_path):
    root = tmp_path / "remote_results"
    candidate = root / "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702"
    (candidate / "monitor").mkdir(parents=True)
    (candidate / "logs").mkdir()
    (candidate / "monitor" / "monitor.log").write_text("2026-01-01T00:00:00+08:00 running\n")

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "wait_for_candidate_trail_result"
    assert active["status"] == "stale_monitor"
    assert active["heartbeat"]["is_stale"] is True
    assert active["needs_main_thread_intervention"] is True
    assert active["postprocess_allowed"] is False


def test_summarize_spn_evidence_routes_ready_candidate_result_to_postprocess(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702"
    candidate = root / run_id
    (candidate / "monitor").mkdir(parents=True)
    (candidate / "results").mkdir()
    (candidate / "logs").mkdir()
    (candidate / "monitor" / "monitor.log").write_text("2026-07-02T18:10:00+08:00 running\n")
    (candidate / "logs" / "run_done.marker").write_text("done\n", encoding="utf-8")
    (candidate / "results" / f"{run_id}.jsonl").write_text(
        "\n".join(json.dumps({"model": f"row_{index}", "auc": 0.79}) for index in range(4)) + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "postprocess_candidate_trail_result"
    assert active["status"] == "result_ready"
    assert active["postprocess_allowed"] is True
    assert active["results_jsonl_line_count"] == 4
    assert active["expected_rows"] == 4
    assert "scripts/postprocess-candidate-trail" in " ".join(active["postprocess_command"])


def test_summarize_spn_evidence_routes_candidate_stop_to_transition_spectrum(tmp_path):
    root = tmp_path / "remote_results"
    candidate = root / "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702"
    candidate.mkdir(parents=True)
    _write_test_json(
        candidate / "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_candidate_trail_route",
            "claim_scope": "candidate-trail consistency diagnostic gate",
            "best_candidate_auc": 0.7915,
            "anchor_auc": 0.7920,
            "shuffled_auc": 0.7921,
            "next_action": {
                "branch": "stop_candidate_trail_route",
                "should_launch_remote": False,
                "fallback_plan": "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
            },
        },
    )

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "bit_transition_spectrum_seed0"
    assert active["decision"] == "stop_candidate_trail_route"
    assert active["should_launch_remote"] is False
    assert active["next_action"]["requires_implementation"] is True
    assert active["next_action"]["next_plan_doc"] == "docs/experiments/innovation1-bit-transition-spectrum-plan.md"
    assert "bit_transition_spectrum_r7_262k_seed0" in active["next_action"]["suggested_plan_config"]
    assert active["next_action"]["suggested_feature_cache_workers"] == 4
    assert "scripts/check-remote-readiness" in active["next_action"]["readiness_command"]


def test_summarize_spn_evidence_prioritizes_transition_spectrum_after_candidate(tmp_path):
    root = tmp_path / "remote_results"
    candidate = root / "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702"
    transition = root / "i1_bit_transition_spectrum_r7_262k_seed0_gpu0_20260702"
    candidate.mkdir(parents=True)
    transition.mkdir(parents=True)
    _write_test_json(
        candidate / "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_candidate_trail_route",
            "claim_scope": "candidate-trail consistency diagnostic gate",
            "next_action": {
                "branch": "stop_candidate_trail_route",
                "should_launch_remote": False,
                "requires_implementation": True,
            },
        },
    )
    _write_test_json(
        transition / "i1_bit_transition_spectrum_r7_262k_seed0_gpu0_20260702_postprocess_summary.json",
        {
            "run_id": "i1_bit_transition_spectrum_r7_262k_seed0_gpu0_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "weak_transition_spectrum_signal",
            "claim_scope": "bit-transition-spectrum medium diagnostic gate",
            "best_candidate_auc": 0.7924,
            "anchor_auc": 0.7920,
            "next_action": {
                "branch": "transition_spectrum_variance_check",
                "should_launch_remote": False,
                "requires_implementation": True,
            },
        },
    )

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "transition_spectrum_seed1_variance_check"
    assert active["decision"] == "weak_transition_spectrum_signal"
    assert active["next_action"]["next_plan_doc"] == "docs/experiments/innovation1-bit-transition-spectrum-plan.md"
    assert "bit_transition_spectrum_r7_262k_seed1" in active["next_action"]["suggested_plan_config"]
    by_run_id = {route["run_id"]: route for route in report["routes"]}
    assert by_run_id["i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702"]["route_state"] == "superseded"
    assert by_run_id["i1_bit_transition_spectrum_r7_262k_seed0_gpu0_20260702"]["route_state"] == (
        "current_or_historical"
    )


def _write_test_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_innovation1_task_defaults_preserve_current_or_explicit_legacy_protocols(tmp_path):
    active_args = spn_active_pattern.parse_args(["--output", str(tmp_path / "active.jsonl")])
    audit_args = spn_feature_audit.parse_args(["--output", str(tmp_path / "audit.json")])
    candidate_args = spn_candidate_evidence.parse_args(["--output", str(tmp_path / "candidate.jsonl")])

    assert active_args.sample_structure == "zhang_wang_case2_official_mcnd"
    assert audit_args.sample_structure == "zhang_wang_case2_official_mcnd"
    assert candidate_args.sample_structure == "zhang_wang_case2_official_mcnd"
    assert candidate_args.validation_key == 0x11111111111111111111
    assert candidate_args.key_rotation_interval == 0


def test_candidate_trail_smoke_config_preserves_official_protocol():
    config_path = Path(
        "configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_smoke.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = spn_candidate_evidence.parse_args(["--config", str(config_path)])

    assert config["run_id"] == "i1_candidate_trail_consistency_smoke_20260701"
    assert args.output == Path(config["output"])
    assert args.rounds == 7
    assert args.samples_per_class == 2
    assert args.pairs_per_sample == 1
    assert args.sample_structure == "zhang_wang_case2_official_mcnd"
    assert args.negative_mode == "encrypted_random_plaintexts"
    assert args.validation_key == 0x11111111111111111111
    assert args.key_rotation_interval == 0
    assert args.model == "mlp"
    assert args.feature_mode == "cell_structured"
    assert args.feature_cache_root == Path(config["feature_cache_root"])
    assert "SMOKE only" in config["description"]
    assert "not accuracy evidence" in config["description"]


def test_candidate_trail_config_can_be_overridden_from_cli(tmp_path):
    config_path = Path(
        "configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_smoke.json"
    )
    args = spn_candidate_evidence.parse_args(
        [
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "override.jsonl"),
            "--model",
            "linear",
            "--samples-per-class",
            "3",
        ]
    )

    assert args.output == tmp_path / "override.jsonl"
    assert args.model == "linear"
    assert args.feature_mode == "cell_structured"
    assert args.samples_per_class == 3
    assert args.sample_structure == "zhang_wang_case2_official_mcnd"


def test_candidate_trail_cli_accepts_feature_cache_workers(tmp_path):
    args = spn_candidate_evidence.parse_args(
        [
            "--output",
            str(tmp_path / "candidate.jsonl"),
            "--feature-cache-workers",
            "2",
        ]
    )

    assert args.feature_cache_workers == 2


def test_candidate_trail_rejects_mismatched_shuffled_feature_mode(tmp_path):
    config_path = Path(
        "configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_smoke.json"
    )
    with pytest.raises(SystemExit, match="shuffled_cells requires feature_mode=cell_structured_shuffled"):
        spn_candidate_evidence.parse_args(
            [
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / "bad.jsonl"),
                "--model",
                "shuffled_cells",
            ]
        )


def test_scripts_readme_documents_monitor_health_result_states():
    text = Path("scripts/README.md").read_text(encoding="utf-8")

    assert "`result_ready`" in text
    assert "`completed_missing_results`" in text
    assert "`results_empty`" in text
    assert "at least one non-empty row" in text
    assert "Neither state should be postprocessed" in text


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


def test_ddt_conditional_plan_records_method_extension_launch_gate():
    invp_plan = Path("docs/experiments/innovation1-invp-only-1m-scale-plan.md").read_text(
        encoding="utf-8"
    )
    ddt_plan = Path("docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md").read_text(
        encoding="utf-8"
    )

    assert "seed0 validated AUC delta over Zhang/Wang 1M anchor `>= +0.001`" in invp_plan
    assert "seed0 delta `< +0.001`" in invp_plan
    assert "i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630" in ddt_plan
    assert "decision = support_invp_structural_attribution" in ddt_plan
    assert "method-extension diagnostic route" in ddt_plan
    assert "rescue branch" in ddt_plan
    assert "failed InvP explanation" in ddt_plan
    assert "check-remote-readiness status = pass" in ddt_plan
    assert "CPU smoke status = pass, 3/3 rows" in ddt_plan


def test_invp_only_plan_records_bounded_monitor_health_command():
    plan = Path("docs/experiments/innovation1-invp-only-1m-scale-plan.md").read_text(
        encoding="utf-8"
    )

    assert "scripts/monitor-health" in plan
    assert "--run-id i1_invp_only_r7_1m_seed0_gpu1_20260629" in plan
    assert "--tmux-session monitor_i1_invp_only_1m_20260629" in plan
    assert "--plan configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv" in plan
    assert "--plan-doc docs/experiments/innovation1-invp-only-1m-scale-plan.md" in plan
    assert "`postprocess_command`" in plan
    assert "`completed_missing_results`" in plan
    assert "`results_empty`" in plan
    assert "do not postprocess until the JSONL exists" in plan
    assert "at least one result row is present" in plan
    assert "must not be used as a main-thread polling loop" in plan


def test_invp_attribution_docs_record_route_specific_monitor_health_command():
    plan = Path("docs/experiments/innovation1-invp-only-formal-attribution-plan.md").read_text(
        encoding="utf-8"
    )
    summary = Path("docs/experiments/innovation1-invp-route-level-evidence-summary.md").read_text(
        encoding="utf-8"
    )

    for text in [plan, summary]:
        assert "scripts/monitor-health" in text
        assert "--run-id i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630" in text
        assert "--plan-doc docs/experiments/innovation1-invp-only-formal-attribution-plan.md" in text
        assert "--plan-doc docs/experiments/innovation1-invp-route-level-evidence-summary.md" in text
        assert "--expected-rows 2" in text
        assert "--postprocess-kind invp_attribution" in text
        assert "`postprocess_command`" in text
        assert "`result_ready`" in text
        assert "`completed_missing_results`" in text
        assert "`results_empty`" in text
        assert "`results_incomplete`" in text


def test_ddt_graph_plan_records_route_specific_monitor_health_command():
    plan = Path("docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md").read_text(
        encoding="utf-8"
    )

    assert "scripts/monitor-health" in plan
    assert "--run-id i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630" in plan
    assert "--plan configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv" in plan
    assert "--plan-doc docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md" in plan
    assert "--expected-rows 5" in plan
    assert "--postprocess-kind ddt_graph" in plan
    assert "`postprocess_command`" in plan
    assert "`result_ready`" in plan
    assert "`completed_missing_results`" in plan
    assert "`results_empty`" in plan
    assert "`results_incomplete`" in plan
    assert "`postprocessed`" in plan
    assert "`postprocess_failed`" in plan
    assert "do not rerun postprocess" in plan
    assert "monitor/postprocess_stderr.log" in plan
    assert "<run_id>_next_action_readiness.json" in plan
    assert "support_ddt_graph_route -> DDT seed1 remote config readiness" in plan
    assert "stop_ddt_graph_route    -> candidate-trail plan handoff" in plan


def test_ddt_graph_plan_marks_seed1_next_action_as_historical():
    plan = Path("docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md").read_text(
        encoding="utf-8"
    )

    assert "seed1 postprocess table above preserves the route-specific artifact fields" in plan
    assert "must not be interpreted" in plan
    assert "as an instruction to rerun seed1" in plan
    assert "route-level decision below supersedes it" in plan
    assert "topology-aware network route was activated" in plan


def test_topology_aware_plan_records_next_action_readiness_artifact():
    plan = Path("docs/experiments/innovation1-spn-topology-aware-network-conditional-plan.md").read_text(
        encoding="utf-8"
    )

    assert "script = scripts/postprocess-topology-aware-result" in plan
    assert "module = src/blockcipher_nd/planning/topology_aware_postprocess.py" in plan
    assert "<run_id>_next_action_readiness.json" in plan
    assert "`support_topology_aware_network_route`" in plan
    assert "`topology_aware_seed1_confirmation`" in plan
    assert "`weak_topology_aware_network_signal`" in plan
    assert "`topology_aware_seed1_variance_check`" in plan
    assert "`stop_topology_aware_network_route`" in plan
    assert "`candidate_trail_consistency`" in plan
    assert "it intentionally does not launch a remote job" in plan


def test_scripts_readme_documents_ddt_next_action_readiness_artifact():
    text = Path("scripts/README.md").read_text(encoding="utf-8")

    assert "DDT graph and topology-aware postprocess" in text
    assert "Candidate-trail and bit-transition-spectrum postprocess" in text
    assert "<run_id>_next_action_readiness.json" in text
    assert "implementation checklist" in text
    assert "next-branch readiness artifact" in text
    assert "`postprocessed` means the watcher already ran the route-specific postprocess" in text
    assert "`postprocess_failed` means inspect the monitor" in text
    assert "`--postprocess-kind topology_aware`" in text
    assert "Route-specific entries in `checked_invariants`" in text
    assert "`candidate_trail_protocol_lock`" in text
    assert "candidate-trail configs" in text
    assert "`pairset_aggregation_stage_lock`" in text
    assert "pair-set" in text
    assert "aggregation configs" in text
    assert "scripts/summarize-spn-evidence" in text
    assert "`active_recommendation`" in text
    assert "`wait_for_candidate_trail_result`" in text
    assert "transition-spectrum decisions ahead of older" in text
    assert "candidate-trail decisions" in text


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


def _write_invp_postprocess_plan(path: Path) -> None:
    row = {
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_invp_postprocess_result(
    path: Path,
    *,
    auc: float,
    accuracy: float,
    calibrated_accuracy: float,
    loss: float,
) -> None:
    row = {
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
        "metrics": {
            "auc": auc,
            "accuracy": accuracy,
            "calibrated_accuracy": calibrated_accuracy,
            "loss": loss,
        },
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
                "train_eval_loss": loss + 0.01,
                "val_accuracy": accuracy,
                "val_auc": auc,
                "val_loss": loss,
            }
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _write_invp_attribution_control_result(
    handle,
    *,
    model: str,
    auc: float,
    accuracy: float = 0.71,
    calibrated_accuracy: float = 0.711,
    loss: float = 0.55,
) -> None:
    row = {
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
        "model": model,
        "selected_model": model,
        "metrics": {
            "auc": auc,
            "accuracy": accuracy,
            "calibrated_accuracy": calibrated_accuracy,
            "loss": loss,
        },
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
                "val_accuracy": accuracy,
                "val_auc": auc,
                "val_loss": loss,
            }
        ],
    }
    handle.write(json.dumps(row) + "\n")


def _write_ddt_graph_result(
    handle,
    *,
    model: str,
    auc: float,
    accuracy: float = 0.71,
    calibrated_accuracy: float = 0.711,
    loss: float = 0.55,
) -> None:
    row = {
        "cipher": "PRESENT-80",
        "structure": "SPN",
        "rounds": 7,
        "seed": 0,
        "samples_per_class": 262144,
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
        "model": model,
        "selected_model": model,
        "metrics": {
            "auc": auc,
            "accuracy": accuracy,
            "calibrated_accuracy": calibrated_accuracy,
            "loss": loss,
        },
    }
    handle.write(json.dumps(row) + "\n")


def _write_ddt_graph_result_set(
    path: Path,
    *,
    invp_auc: float,
    transition_no_ddt_auc: float,
    no_ddt_graph_auc: float,
    ddt_auc: float,
    shuffled_auc: float,
    invp_calibrated_accuracy: float = 0.711,
    ddt_calibrated_accuracy: float = 0.712,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        _write_ddt_graph_result(
            handle,
            model="present_nibble_invp_only_spn_only",
            auc=invp_auc,
            calibrated_accuracy=invp_calibrated_accuracy,
        )
        _write_ddt_graph_result(
            handle,
            model="present_nibble_paligned_transition_residual",
            auc=transition_no_ddt_auc,
        )
        _write_ddt_graph_result(
            handle,
            model="present_nibble_no_ddt_graph",
            auc=no_ddt_graph_auc,
        )
        _write_ddt_graph_result(
            handle,
            model="present_nibble_ddt_graph",
            auc=ddt_auc,
            calibrated_accuracy=ddt_calibrated_accuracy,
        )
        _write_ddt_graph_result(
            handle,
            model="present_nibble_shuffled_ddt_graph",
            auc=shuffled_auc,
        )


def test_invp_attribution_controls_gate_supports_structural_attribution(tmp_path):
    results_path = tmp_path / "controls.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        _write_invp_attribution_control_result(
            handle,
            model="present_nibble_delta_only_spn_only",
            auc=0.7930,
        )
        _write_invp_attribution_control_result(
            handle,
            model="present_nibble_shuffled_paligned_spn_only",
            auc=0.7940,
        )

    report = gate_invp_attribution_controls(results_path)

    assert report["status"] == "pass"
    assert report["decision"] == "support_invp_structural_attribution"
    assert report["action"] == "write_route_level_attribution_summary"
    assert report["max_control_auc"] == 0.7940
    assert report["attribution_margin"] > 0.003
    assert report["controls"]["present_nibble_delta_only_spn_only"]["delta_vs_reference_auc"] < 0
    assert report["controls"]["present_nibble_shuffled_paligned_spn_only"]["delta_vs_reference_auc"] > 0
    assert "true InvP/P-layer alignment" in report["interpretation"]


def test_invp_attribution_controls_gate_weakens_claim_when_control_matches(tmp_path):
    results_path = tmp_path / "controls.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        _write_invp_attribution_control_result(
            handle,
            model="present_nibble_delta_only_spn_only",
            auc=0.7980,
        )
        _write_invp_attribution_control_result(
            handle,
            model="present_nibble_shuffled_paligned_spn_only",
            auc=0.7940,
        )

    report = gate_invp_attribution_controls(results_path)

    assert report["status"] == "pass"
    assert report["decision"] == "weaken_invp_structural_attribution"
    assert report["action"] == "switch_to_new_spn_structure_hypothesis_or_variance_audit"
    assert report["attribution_margin"] < 0
    assert "not supported" in report["interpretation"]


def test_ddt_graph_gate_supports_route_when_true_graph_beats_controls(tmp_path):
    results_path = tmp_path / "ddt_graph.jsonl"
    _write_ddt_graph_result_set(
        results_path,
        invp_auc=0.7940,
        transition_no_ddt_auc=0.7945,
        no_ddt_graph_auc=0.7946,
        ddt_auc=0.7960,
        shuffled_auc=0.7948,
    )

    report = gate_ddt_graph_result(results_path)

    assert report["status"] == "pass"
    assert report["decision"] == "support_ddt_graph_route"
    assert report["action"] == "run_262k_seed1_confirmation_before_1m_scale"
    assert report["max_control_auc"] == 0.7948
    assert report["margin_vs_best_control_auc"] > 0.001
    assert report["margin_vs_no_ddt_graph_auc"] > 0.001
    assert report["margin_vs_shuffled_auc"] > 0.001
    assert "medium-scale diagnostic support" in report["interpretation"]
    assert "not paper-scale" in report["claim_scope"]


def test_ddt_graph_gate_marks_weak_signal_when_margin_is_small(tmp_path):
    results_path = tmp_path / "ddt_graph_weak.jsonl"
    _write_ddt_graph_result_set(
        results_path,
        invp_auc=0.7940,
        transition_no_ddt_auc=0.7945,
        no_ddt_graph_auc=0.7946,
        ddt_auc=0.7950,
        shuffled_auc=0.7948,
    )

    report = gate_ddt_graph_result(results_path)

    assert report["status"] == "pass"
    assert report["decision"] == "weak_ddt_graph_signal"
    assert report["action"] == "run_prepared_262k_seed1_variance_check_before_scaling"
    assert 0 < report["margin_vs_best_control_auc"] < 0.001


def test_ddt_graph_postprocess_routes_weak_signal_to_seed1_variance_check(tmp_path):
    plan_path = Path("configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv")
    results_path = tmp_path / "ddt_graph_weak.jsonl"
    output_dir = tmp_path / "postprocess"
    plan_doc_path = tmp_path / "ddt-plan.md"
    plan_doc_path.write_text("# DDT Plan\n", encoding="utf-8")
    _write_ddt_graph_result_set(
        results_path,
        invp_auc=0.7940,
        transition_no_ddt_auc=0.7945,
        no_ddt_graph_auc=0.7946,
        ddt_auc=0.7950,
        shuffled_auc=0.7948,
    )

    report = postprocess_ddt_graph_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_ddt_graph_weak",
        expected_rows=5,
        plan_doc_path=plan_doc_path,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "weak_ddt_graph_signal"
    assert report["next_action"]["branch"] == "ddt_graph_seed1_variance_check"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["launch_remote_config"].endswith(
        "innovation1_spn_present_ddt_graph_r7_262k_seed1_gpu1_20260630.json"
    )
    assert report["next_action"]["readiness_command"].startswith(
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness"
    )
    assert report["next_action"]["run_id"] == "i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630"
    assert any("seed1 variance check" in step for step in report["next_steps"])
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert "| Next action branch | `ddt_graph_seed1_variance_check` |" in plan_doc
    assert "| Next action should launch remote | `True` |" in plan_doc
    assert "innovation1_spn_present_ddt_graph_r7_262k_seed1_gpu1_20260630.json" in plan_doc


def test_ddt_graph_gate_stops_when_control_matches_or_calibration_regresses(tmp_path):
    results_path = tmp_path / "ddt_graph_stop.jsonl"
    _write_ddt_graph_result_set(
        results_path,
        invp_auc=0.7940,
        transition_no_ddt_auc=0.7945,
        no_ddt_graph_auc=0.7962,
        ddt_auc=0.7960,
        shuffled_auc=0.7948,
    )

    report = gate_ddt_graph_result(results_path)

    assert report["status"] == "pass"
    assert report["decision"] == "stop_ddt_graph_route"
    assert report["action"] == "record_negative_or_tied_evidence_and_switch_hypothesis"
    assert report["margin_vs_no_ddt_graph_auc"] < 0
    assert report["margin_vs_no_ddt_auc"] == report["margin_vs_no_ddt_graph_auc"]
    assert "do not scale this route to 1M" in report["interpretation"]

    regressed_path = tmp_path / "ddt_graph_calibration_regressed.jsonl"
    _write_ddt_graph_result_set(
        regressed_path,
        invp_auc=0.7940,
        transition_no_ddt_auc=0.7945,
        no_ddt_graph_auc=0.7946,
        ddt_auc=0.7960,
        shuffled_auc=0.7948,
        invp_calibrated_accuracy=0.712,
        ddt_calibrated_accuracy=0.711,
    )

    regressed = gate_ddt_graph_result(regressed_path)

    assert regressed["status"] == "pass"
    assert regressed["decision"] == "stop_ddt_graph_route"
    assert regressed["calibrated_delta_vs_invp"] < 0


def test_ddt_graph_gate_fails_when_expected_rows_are_missing(tmp_path):
    results_path = tmp_path / "ddt_graph_missing.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        _write_ddt_graph_result(
            handle,
            model="present_nibble_ddt_graph",
            auc=0.7960,
        )

    report = gate_ddt_graph_result(results_path)

    assert report["status"] == "fail"
    assert report["decision"] == "invalid"
    assert any("expected_rows=5" in error for error in report["errors"])
    assert any("missing_models" in error for error in report["errors"])


def _write_topology_aware_result(
    path: Path,
    model: str,
    auc: float,
    *,
    calibrated_accuracy: float = 0.712,
    accuracy: float = 0.71,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "model": model,
                    "selected_model": model,
                    "cipher": "PRESENT-80",
                    "structure": "SPN",
                    "rounds": 7,
                    "seed": 0,
                    "samples_per_class": 262144,
                    "pairs_per_sample": 16,
                    "feature_encoding": "ciphertext_pair_bits",
                    "sample_structure": "zhang_wang_case2_official_mcnd",
                    "negative_mode": "encrypted_random_plaintexts",
                    "difference_profile": "present_zhang_wang2022_mcnd",
                    "difference_member": 0,
                    "train_key": 0,
                    "validation_key": 0x11111111111111111111,
                    "key_rotation_interval": 0,
                    "integral_active_nibble": 0,
                    "metrics": {
                        "auc": auc,
                        "accuracy": accuracy,
                        "calibrated_accuracy": calibrated_accuracy,
                        "loss": 0.55,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )


def _write_topology_aware_result_set(
    path: Path,
    *,
    invp_auc: float,
    true_graph_auc: float,
    shuffled_auc: float,
    invp_calibrated_accuracy: float = 0.712,
    true_graph_calibrated_accuracy: float = 0.713,
) -> None:
    _write_topology_aware_result(
        path,
        "present_nibble_invp_only_spn_only",
        invp_auc,
        calibrated_accuracy=invp_calibrated_accuracy,
    )
    _write_topology_aware_result(
        path,
        "present_nibble_invp_p_layer_graph_spn_only",
        true_graph_auc,
        calibrated_accuracy=true_graph_calibrated_accuracy,
    )
    _write_topology_aware_result(
        path,
        "present_nibble_invp_shuffled_p_layer_graph_spn_only",
        shuffled_auc,
        calibrated_accuracy=invp_calibrated_accuracy,
    )


def test_topology_aware_postprocess_routes_support_to_prepared_seed1(tmp_path):
    plan_path = Path("configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_r7_262k.csv")
    results_path = tmp_path / "topology_aware.jsonl"
    output_dir = tmp_path / "postprocess"
    plan_doc_path = tmp_path / "topology-plan.md"
    plan_doc_path.write_text("# Topology Plan\n", encoding="utf-8")
    _write_topology_aware_result_set(
        results_path,
        invp_auc=0.7940,
        true_graph_auc=0.7960,
        shuffled_auc=0.7946,
    )

    report = postprocess_topology_aware_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_topology_aware",
        expected_rows=3,
        plan_doc_paths=[plan_doc_path],
    )
    next_action_report = plan_next_action(Path(report["summary"]))

    assert report["status"] == "pass"
    assert report["decision"] == "support_topology_aware_network_route"
    assert report["next_action"]["branch"] == "topology_aware_seed1_confirmation"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["launch_remote_config"].endswith(
        "innovation1_spn_present_topology_aware_network_r7_262k_seed1_gpu1_20260701.json"
    )
    assert report["next_action"]["readiness_command"].startswith(
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness"
    )
    assert report["next_action"]["run_id"] == "i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701"
    readiness_path = Path(report["next_action_readiness"])
    assert readiness_path.exists()
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "topology_aware_seed1_confirmation"
    assert readiness["readiness_pass"] is True
    assert [item["role"] for item in readiness["readiness_reports"]] == ["primary"]
    assert next_action_report["status"] == "pass"
    assert next_action_report["branch"] == "topology_aware_seed1_confirmation"
    markdown = (output_dir / "unit_topology_aware_postprocess_summary.md").read_text(encoding="utf-8")
    assert "Next Action:" in markdown
    assert "topology_aware_seed1_confirmation" in markdown
    assert "unit_topology_aware_next_action_readiness.json" in markdown
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert "| Next action branch | `topology_aware_seed1_confirmation` |" in plan_doc
    assert "| Next action should launch remote | `True` |" in plan_doc
    assert "innovation1_spn_present_topology_aware_network_r7_262k_seed1_gpu1_20260701.json" in plan_doc
    assert "| Next action readiness | `" in plan_doc


def test_topology_aware_postprocess_routes_weak_signal_to_seed1_variance_check(tmp_path):
    plan_path = Path("configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_r7_262k.csv")
    results_path = tmp_path / "topology_aware_weak.jsonl"
    output_dir = tmp_path / "postprocess"
    plan_doc_path = tmp_path / "topology-plan.md"
    plan_doc_path.write_text("# Topology Plan\n", encoding="utf-8")
    _write_topology_aware_result_set(
        results_path,
        invp_auc=0.7940,
        true_graph_auc=0.7945,
        shuffled_auc=0.7942,
    )

    report = postprocess_topology_aware_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_topology_aware_weak",
        expected_rows=3,
        plan_doc_paths=[plan_doc_path],
    )
    next_action_report = plan_next_action(Path(report["summary"]))

    assert report["status"] == "pass"
    assert report["decision"] == "weak_topology_aware_network_signal"
    assert report["next_action"]["branch"] == "topology_aware_seed1_variance_check"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["launch_remote_config"].endswith(
        "innovation1_spn_present_topology_aware_network_r7_262k_seed1_gpu1_20260701.json"
    )
    assert report["next_action"]["run_id"] == "i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701"
    assert any("seed1 variance check" in step for step in report["next_steps"])
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "topology_aware_seed1_variance_check"
    assert readiness["readiness_pass"] is True
    assert next_action_report["status"] == "pass"
    assert next_action_report["branch"] == "topology_aware_seed1_variance_check"
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert "| Next action branch | `topology_aware_seed1_variance_check` |" in plan_doc
    assert "| Next action should launch remote | `True` |" in plan_doc
    assert "innovation1_spn_present_topology_aware_network_r7_262k_seed1_gpu1_20260701.json" in plan_doc


def test_topology_aware_postprocess_routes_stop_to_candidate_trail_plan(tmp_path):
    plan_path = Path("configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_r7_262k.csv")
    results_path = tmp_path / "topology_aware_stop.jsonl"
    output_dir = tmp_path / "postprocess"
    plan_doc_path = tmp_path / "topology-plan.md"
    plan_doc_path.write_text("# Topology Plan\n", encoding="utf-8")
    _write_topology_aware_result_set(
        results_path,
        invp_auc=0.7940,
        true_graph_auc=0.7935,
        shuffled_auc=0.7942,
    )

    report = postprocess_topology_aware_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_topology_aware_stop",
        expected_rows=3,
        plan_doc_paths=[plan_doc_path],
    )
    next_action_report = plan_next_action(Path(report["summary"]))

    assert report["status"] == "pass"
    assert report["decision"] == "stop_topology_aware_network_route"
    assert report["next_action"]["branch"] == "candidate_trail_consistency"
    assert report["next_action"]["should_launch_remote"] is False
    assert report["next_action"]["requires_implementation"] is True
    assert report["next_action"]["plan_doc"] == "docs/experiments/innovation1-candidate-trail-consistency-plan.md"
    assert report["next_action_readiness"]
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "candidate_trail_consistency"
    assert readiness["readiness_reports"] == []
    assert next_action_report["status"] == "pass"
    assert next_action_report["should_launch_remote"] is False
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert "| Next action branch | `candidate_trail_consistency` |" in plan_doc
    assert "| Next action should launch remote | `False` |" in plan_doc
    assert "Switch to the candidate-trail consistency data representation plan" in plan_doc


def test_ddt_graph_postprocess_routes_stop_to_candidate_trail_plan(tmp_path):
    plan_path = Path("configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv")
    results_path = tmp_path / "ddt_graph_stop.jsonl"
    output_dir = tmp_path / "postprocess"
    plan_doc_path = tmp_path / "ddt-plan.md"
    plan_doc_path.write_text("# DDT Plan\n", encoding="utf-8")
    _write_ddt_graph_result_set(
        results_path,
        invp_auc=0.7940,
        transition_no_ddt_auc=0.7945,
        no_ddt_graph_auc=0.7962,
        ddt_auc=0.7960,
        shuffled_auc=0.7948,
    )

    report = postprocess_ddt_graph_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_ddt_graph_stop",
        expected_rows=5,
        plan_doc_path=plan_doc_path,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "stop_ddt_graph_route"
    assert report["next_action"]["branch"] == "candidate_trail_consistency"
    assert report["next_action"]["should_launch_remote"] is False
    assert report["next_action"]["requires_implementation"] is True
    assert report["next_action"]["plan_doc"] == "docs/experiments/innovation1-candidate-trail-consistency-plan.md"
    assert any("candidate-trail consistency data representation plan" in step for step in report["next_steps"])
    assert any("pair-set aggregation as a deferred attribution control" in step for step in report["next_steps"])
    readiness_path = Path(report["next_action_readiness"])
    assert readiness_path.exists()
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "candidate_trail_consistency"
    assert readiness["readiness_pass"] is False
    assert readiness["readiness_reports"] == []
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert "| Next action branch | `candidate_trail_consistency` |" in plan_doc
    assert "| Next action should launch remote | `False` |" in plan_doc
    assert "| Next action readiness | `" in plan_doc


def test_plan_next_action_checks_ddt_seed1_readiness(tmp_path):
    plan_path = Path("configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv")
    results_path = tmp_path / "ddt_graph.jsonl"
    output_dir = tmp_path / "postprocess"
    _write_ddt_graph_result_set(
        results_path,
        invp_auc=0.7940,
        transition_no_ddt_auc=0.7945,
        no_ddt_graph_auc=0.7946,
        ddt_auc=0.7960,
        shuffled_auc=0.7948,
    )

    postprocess_report = postprocess_ddt_graph_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_ddt_graph_next_action",
        expected_rows=5,
    )
    report = plan_next_action(Path(postprocess_report["summary"]))

    assert report["status"] == "pass"
    assert report["branch"] == "ddt_graph_seed1_confirmation"
    assert report["should_launch_remote"] is True
    assert report["readiness_pass"] is True
    assert report["launch_checklist"]
    assert "pushed commit" in report["launch_checklist"][0]
    assert "do not SSH-poll" in " ".join(report["launch_checklist"])
    assert [item["role"] for item in report["readiness_reports"]] == ["primary"]
    assert report["readiness_reports"][0]["readiness"]["status"] == "pass"


def test_plan_next_action_keeps_ddt_stop_on_candidate_trail_plan(tmp_path):
    plan_path = Path("configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv")
    results_path = tmp_path / "ddt_graph_stop.jsonl"
    output_dir = tmp_path / "postprocess"
    _write_ddt_graph_result_set(
        results_path,
        invp_auc=0.7940,
        transition_no_ddt_auc=0.7945,
        no_ddt_graph_auc=0.7962,
        ddt_auc=0.7960,
        shuffled_auc=0.7948,
    )

    postprocess_report = postprocess_ddt_graph_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_ddt_graph_pairset_next_action",
        expected_rows=5,
    )
    report = plan_next_action(Path(postprocess_report["summary"]))

    assert report["status"] == "pass"
    assert report["branch"] == "candidate_trail_consistency"
    assert report["should_launch_remote"] is False
    assert report["readiness_pass"] is False
    assert report["readiness_reports"] == []
    assert report["requires_implementation"] is True
    assert report["implementation_checklist"]
    assert "candidate_trail_consistency" in report["implementation_checklist"][0]
    assert "docs/experiments/innovation1-candidate-trail-consistency-plan.md" in " ".join(
        report["implementation_checklist"]
    )
    assert "remote launch" in " ".join(report["implementation_checklist"])
    assert report["launch_checklist"] == []


def test_plan_next_action_reports_missing_remote_config(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "run_id": "unit_missing_config",
                "decision": "unit",
                "action": "unit",
                "next_action": {
                    "branch": "missing_config",
                    "should_launch_remote": True,
                    "launch_remote_config": str(tmp_path / "missing.json"),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = plan_next_action(summary)

    assert report["status"] == "fail"
    assert report["branch"] == "missing_config"
    assert report["readiness_pass"] is False
    assert report["launch_checklist"] == ["Do not launch until all readiness reports pass."]
    assert report["readiness_reports"][0]["readiness"]["status"] == "fail"
    assert "missing.json" in report["errors"][0]


def test_ddt_graph_postprocess_updates_plan_doc(tmp_path):
    plan_path = Path("configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv")
    results_path = tmp_path / "ddt_graph.jsonl"
    output_dir = tmp_path / "postprocess"
    plan_doc_path = tmp_path / "ddt-plan.md"
    plan_doc_path.write_text("# DDT Plan\n", encoding="utf-8")
    _write_ddt_graph_result_set(
        results_path,
        invp_auc=0.7940,
        transition_no_ddt_auc=0.7945,
        no_ddt_graph_auc=0.7946,
        ddt_auc=0.7960,
        shuffled_auc=0.7948,
    )

    report = postprocess_ddt_graph_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_ddt_graph",
        expected_rows=5,
        plan_doc_path=plan_doc_path,
    )

    assert report["status"] == "pass"
    assert report["validation_status"] == "pass"
    assert report["ddt_graph_status"] == "pass"
    assert report["decision"] == "support_ddt_graph_route"
    assert report["next_action"]["branch"] == "ddt_graph_seed1_confirmation"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["launch_remote_config"].endswith(
        "innovation1_spn_present_ddt_graph_r7_262k_seed1_gpu1_20260630.json"
    )
    assert report["next_action"]["readiness_command"].startswith(
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness"
    )
    assert report["next_action"]["run_id"] == "i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630"
    assert (output_dir / "unit_ddt_graph_local_result_gate.json").exists()
    assert (output_dir / "unit_ddt_graph_curves.svg").exists()
    assert (output_dir / "unit_ddt_graph_history.csv").exists()
    assert (output_dir / "unit_ddt_graph_ddt_graph_gate.json").exists()
    assert (output_dir / "unit_ddt_graph_postprocess_summary.json").exists()
    assert (output_dir / "unit_ddt_graph_postprocess_summary.md").exists()
    readiness_path = output_dir / "unit_ddt_graph_next_action_readiness.json"
    assert readiness_path.exists()
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "ddt_graph_seed1_confirmation"
    assert readiness["readiness_pass"] is True
    assert [item["role"] for item in readiness["readiness_reports"]] == ["primary"]
    assert readiness["readiness_reports"][0]["readiness"]["status"] == "pass"
    markdown = (output_dir / "unit_ddt_graph_postprocess_summary.md").read_text()
    assert "support_ddt_graph_route" in markdown
    assert "present_nibble_ddt_graph" in markdown
    assert "Launch configs/remote/innovation1_spn_present_ddt_graph_r7_262k_seed1_gpu1_20260630.json" in markdown
    assert "unit_ddt_graph_next_action_readiness.json" in markdown
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert "## Retrieved DDT Graph Result" in plan_doc
    assert "<!-- ddt-graph-postprocess:unit_ddt_graph:start -->" in plan_doc
    assert "| Decision | `support_ddt_graph_route` |" in plan_doc
    assert "| Next action branch | `ddt_graph_seed1_confirmation` |" in plan_doc
    assert "| Next action should launch remote | `True` |" in plan_doc
    assert "innovation1_spn_present_ddt_graph_r7_262k_seed1_gpu1_20260630.json" in plan_doc
    assert "| Next action readiness | `" in plan_doc

    postprocess_ddt_graph_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_ddt_graph",
        expected_rows=5,
        plan_doc_path=plan_doc_path,
    )
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert plan_doc.count("<!-- ddt-graph-postprocess:unit_ddt_graph:start -->") == 1
    assert plan_doc.count("### unit_ddt_graph DDT Graph Result") == 1


def _write_invp_attribution_controls_plan(path: Path) -> None:
    rows = [
        {
            "cipher": "PRESENT-80",
            "structure": "SPN",
            "network": "DeltaOnly",
            "model_key": "present_nibble_delta_only_spn_only",
            "family": "delta",
            "architecture_rank": "0",
            "score": "130",
            "rounds": "7",
            "seed": "0",
            "samples_per_class": "1000000",
            "pairs_per_sample": "16",
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "train_key": "0x00000000000000000000",
            "validation_key": "0x11111111111111111111",
            "key_rotation_interval": "0",
            "sample_structure": "zhang_wang_case2_official_mcnd",
            "integral_active_nibble": "0",
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
            "model_options": "{}",
            "evidence": "PAPER_SCALE_ATTRIBUTION_CONTROL 1000000/class",
            "literature": "unit",
        },
        {
            "cipher": "PRESENT-80",
            "structure": "SPN",
            "network": "Shuffled",
            "model_key": "present_nibble_shuffled_paligned_spn_only",
            "family": "shuffled",
            "architecture_rank": "1",
            "score": "125",
            "rounds": "7",
            "seed": "0",
            "samples_per_class": "1000000",
            "pairs_per_sample": "16",
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "train_key": "0x00000000000000000000",
            "validation_key": "0x11111111111111111111",
            "key_rotation_interval": "0",
            "sample_structure": "zhang_wang_case2_official_mcnd",
            "integral_active_nibble": "0",
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
            "model_options": "{}",
            "evidence": "PAPER_SCALE_ATTRIBUTION_CONTROL 1000000/class",
            "literature": "unit",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_invp_attribution_controls_postprocess_updates_plan_doc(tmp_path):
    plan_path = tmp_path / "controls_plan.csv"
    results_path = tmp_path / "controls.jsonl"
    output_dir = tmp_path / "postprocess"
    plan_doc_path = tmp_path / "formal-attribution-plan.md"
    plan_doc_path.write_text("# Formal Attribution\n", encoding="utf-8")
    _write_invp_attribution_controls_plan(plan_path)
    with results_path.open("w", encoding="utf-8") as handle:
        _write_invp_attribution_control_result(
            handle,
            model="present_nibble_delta_only_spn_only",
            auc=0.7930,
        )
        _write_invp_attribution_control_result(
            handle,
            model="present_nibble_shuffled_paligned_spn_only",
            auc=0.7940,
        )

    report = postprocess_invp_attribution_controls(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_attr_controls",
        expected_rows=2,
        plan_doc_path=plan_doc_path,
    )

    assert report["status"] == "pass"
    assert report["validation_status"] == "pass"
    assert report["attribution_status"] == "pass"
    assert report["decision"] == "support_invp_structural_attribution"
    assert report["next_action"]["branch"] == "route_level_attribution_summary"
    assert report["next_action"]["should_launch_remote"] is False
    assert (output_dir / "unit_attr_controls_local_result_gate.json").exists()
    assert (output_dir / "unit_attr_controls_curves.svg").exists()
    assert (output_dir / "unit_attr_controls_history.csv").exists()
    assert (output_dir / "unit_attr_controls_attribution_gate.json").exists()
    assert (output_dir / "unit_attr_controls_postprocess_summary.json").exists()
    assert (output_dir / "unit_attr_controls_postprocess_summary.md").exists()
    markdown = (output_dir / "unit_attr_controls_postprocess_summary.md").read_text()
    assert "support_invp_structural_attribution" in markdown
    assert "present_nibble_delta_only_spn_only" in markdown
    assert "active topology-aware network route" in markdown
    assert "candidate-trail / transition consistency" in markdown
    assert "new DDT/topology route" not in markdown
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert "## Retrieved Attribution Control Result" in plan_doc
    assert "<!-- invp-attribution-postprocess:unit_attr_controls:start -->" in plan_doc
    assert "| Decision | `support_invp_structural_attribution` |" in plan_doc
    assert "| Next action branch | `route_level_attribution_summary` |" in plan_doc
    assert "active topology-aware network route" in plan_doc
    assert "candidate-trail / transition consistency" in plan_doc
    assert "new DDT/topology route" not in plan_doc

    postprocess_invp_attribution_controls(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_attr_controls",
        expected_rows=2,
        plan_doc_path=plan_doc_path,
    )
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert plan_doc.count("<!-- invp-attribution-postprocess:unit_attr_controls:start -->") == 1
    assert plan_doc.count("### unit_attr_controls Attribution Control Result") == 1


def test_invp_attribution_controls_postprocess_updates_multiple_plan_docs(tmp_path):
    plan_path = tmp_path / "controls_plan.csv"
    results_path = tmp_path / "controls.jsonl"
    output_dir = tmp_path / "postprocess"
    formal_doc = tmp_path / "formal-attribution-plan.md"
    route_doc = tmp_path / "route-summary.md"
    formal_doc.write_text("# Formal Attribution\n", encoding="utf-8")
    route_doc.write_text("# Route Summary\n", encoding="utf-8")
    _write_invp_attribution_controls_plan(plan_path)
    with results_path.open("w", encoding="utf-8") as handle:
        _write_invp_attribution_control_result(
            handle,
            model="present_nibble_delta_only_spn_only",
            auc=0.7930,
        )
        _write_invp_attribution_control_result(
            handle,
            model="present_nibble_shuffled_paligned_spn_only",
            auc=0.7940,
        )

    report = postprocess_invp_attribution_controls(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_attr_multi",
        expected_rows=2,
        plan_doc_paths=[formal_doc, route_doc],
    )

    assert report["status"] == "pass"
    assert report["plan_docs"] == [str(formal_doc), str(route_doc)]
    assert report["plan_doc"] == str(formal_doc)
    for doc in [formal_doc, route_doc]:
        text = doc.read_text(encoding="utf-8")
        assert "## Retrieved Attribution Control Result" in text
        assert "<!-- invp-attribution-postprocess:unit_attr_multi:start -->" in text
        assert "| Decision | `support_invp_structural_attribution` |" in text
    summary = json.loads((output_dir / "unit_attr_multi_postprocess_summary.json").read_text())
    assert summary["plan_docs"] == [str(formal_doc), str(route_doc)]


def test_invp_only_postprocess_writes_validation_plot_history_and_branch_gate(tmp_path):
    plan_path = tmp_path / "plan.csv"
    results_path = tmp_path / "results.jsonl"
    output_dir = tmp_path / "postprocess"
    plan_doc_path = tmp_path / "invp-plan.md"
    plan_doc_path.write_text(
        "# InvP Plan\n\n**Status:** running remotely / tmux monitor active\n",
        encoding="utf-8",
    )
    _write_invp_postprocess_plan(plan_path)
    _write_invp_postprocess_result(
        results_path,
        auc=0.7971,
        accuracy=0.72,
        calibrated_accuracy=0.723,
        loss=0.54,
    )

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
    assert summary["next_action"]["readiness_command"].startswith(
        "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness --config "
    )
    assert summary["next_action"]["launch_remote_config"] in summary["next_action"]["readiness_command"]
    assert any("seed1" in step for step in summary["next_steps"])
    assert any(summary["next_action"]["readiness_command"] in step for step in summary["next_steps"])
    assert any("tmux watcher or sub-agent" in step for step in summary["next_steps"])
    markdown = (output_dir / "unit_invp_postprocess_summary.md").read_text()
    assert "auc_delta_vs_zhang_wang_1m" in markdown
    assert "auc_delta_vs_paligned_mcnd_1m" in markdown
    assert "launch_invp_seed1_confirmation" in markdown
    assert "plan_doc" in markdown
    assert "Next Action:" in markdown
    assert "- branch: `seed1_confirmation`" in markdown
    assert "- readiness_command: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness" in markdown
    assert "Next Steps:" in markdown
    assert "Launch configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json" in markdown
    assert "scripts/check-remote-readiness" in markdown
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert "**Status:** completed / postprocessed / branch gated" in plan_doc
    assert "<!-- invp-postprocess:unit_invp:start -->" in plan_doc
    assert "| AUC | `0.797100000000` |" in plan_doc
    assert "| Decision | `launch_invp_seed1_confirmation` |" in plan_doc
    assert "| Next action branch | `seed1_confirmation` |" in plan_doc
    assert "| Next action readiness command | `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness" in plan_doc
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
    plan_doc_path = tmp_path / "plan.md"
    _write_invp_postprocess_plan(plan_path)
    plan_doc_path.write_text("# InvP Plan\n\n**Status:** running remotely / tmux monitor active\n", encoding="utf-8")
    _write_invp_postprocess_result(
        results_path,
        auc=0.7939,
        accuracy=0.718,
        calibrated_accuracy=0.719,
        loss=0.55,
    )

    report = postprocess_invp_only_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_invp_tied",
        expected_rows=1,
        plan_doc_path=plan_doc_path,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "enter_ddt_graph_route"
    assert report["next_action"]["branch"] == "ddt_graph"
    assert report["next_action"]["should_launch_remote"] is False
    assert report["next_action"]["requires_implementation"] is True
    assert report["next_action"]["plan_doc"].endswith("innovation1-spn-ddt-graph-conditional-plan.md")
    assert report["next_action"]["implementation_aliases"] == [
        "present_nibble_no_ddt_graph",
        "present_nibble_ddt_graph",
        "present_nibble_shuffled_ddt_graph",
    ]
    assert "src/blockcipher_nd/registry/model_families/spn.py" in report["next_action"]["implementation_files"]
    assert any(
        "tensor-native DDT cell features" in step
        for step in report["next_action"]["implementation_checklist"]
    )
    assert any("CPU smoke CSV" in step for step in report["next_action"]["implementation_checklist"])
    markdown = (output_dir / "unit_invp_tied_postprocess_summary.md").read_text()
    assert "Next Action:" in markdown
    assert "- branch: `ddt_graph`" in markdown
    assert "- implementation_aliases:" in markdown
    assert "`present_nibble_no_ddt_graph`" in markdown
    assert "`present_nibble_ddt_graph`" in markdown
    assert "- implementation_checklist:" in markdown
    assert "`Implement tensor-native DDT cell features from ciphertext_pair_bits.`" in markdown
    assert any("DDT graph route" in step for step in report["next_steps"])
    assert not any("seed1" in step.lower() for step in report["next_steps"])
    plan_doc = plan_doc_path.read_text(encoding="utf-8")
    assert "| Next action implementation aliases | `present_nibble_no_ddt_graph; present_nibble_ddt_graph; present_nibble_shuffled_ddt_graph` |" in plan_doc
    assert "Implement tensor-native DDT cell features from ciphertext_pair_bits." in plan_doc


def test_invp_only_postprocess_underperforming_result_keeps_ddt_checklist(tmp_path):
    plan_path = tmp_path / "plan.csv"
    results_path = tmp_path / "results.jsonl"
    output_dir = tmp_path / "postprocess"
    _write_invp_postprocess_plan(plan_path)
    _write_invp_postprocess_result(
        results_path,
        auc=0.7900,
        accuracy=0.710,
        calibrated_accuracy=0.711,
        loss=0.56,
    )

    report = postprocess_invp_only_result(
        plan_path=plan_path,
        results_path=results_path,
        output_dir=output_dir,
        run_id="unit_invp_under",
        expected_rows=1,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "discard_invp_only_as_main_1m_candidate"
    assert report["next_action"]["branch"] == "discard_invp_main"
    assert report["next_action"]["should_launch_remote"] is False
    assert report["next_action"]["requires_implementation"] is True
    assert report["next_action"]["implementation_aliases"] == [
        "present_nibble_no_ddt_graph",
        "present_nibble_ddt_graph",
        "present_nibble_shuffled_ddt_graph",
    ]
    assert any(
        "tensor-native DDT cell features" in step
        for step in report["next_action"]["implementation_checklist"]
    )


def test_monitor_health_reports_running_result_ready_and_failed(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_run"
    run_root = root / run_id
    monitor = run_root / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text(
        "\n".join(
            [
                "monitor_start 2026-06-29T14:45:53+08:00",
                "2026-06-29T15:00:02+08:00 running",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")

    plan = tmp_path / "plan.csv"
    plan.write_text("header\n", encoding="utf-8")
    plan_doc = tmp_path / "plan.md"
    plan_doc.write_text("# plan\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        now=datetime.fromisoformat("2026-06-29T15:05:00+08:00"),
    )

    assert report["status"] == "running"
    assert report["needs_main_thread_intervention"] is False
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []
    assert report["results_jsonl_exists"] is False
    assert report["results_jsonl_line_count"] == 0
    assert report["heartbeat"]["is_stale"] is False
    assert report["artifact_files"] == [
        "monitor/monitor.log",
        "monitor/monitor_ssh_stderr.log",
    ]
    assert report["scp_stderr_exists"] is False
    assert report["scp_stderr_warnings"] == []
    assert report["scp_stderr_errors"] == []

    (monitor / "scp_stderr.log").write_text(
        "\n".join(
            [
                "scp: G:/lxy/blockcipher-structure-adaptive-nd-runs/unit_run/logs: No such file or directory",
                "scp: G:/lxy/blockcipher-structure-adaptive-nd-runs/unit_run/results: No such file or directory",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        now=datetime.fromisoformat("2026-06-29T15:05:00+08:00"),
    )

    assert report["status"] == "running"
    assert report["needs_main_thread_intervention"] is False
    assert report["scp_stderr_exists"] is True
    assert report["scp_stderr_errors"] == []
    assert report["scp_stderr_missing_artifact_line_count"] == 2
    assert report["scp_stderr_stale_missing_artifacts"] is False
    assert report["scp_stderr_persistent_missing_artifacts"] is False
    assert report["scp_stderr_warnings"] == [
        "scp reported remote artifact paths missing; this is normal before "
        "the remote run creates logs/results, but should clear once artifacts exist"
    ]

    (monitor / "monitor.log").write_text(
        "\n".join(
            [
                "monitor_start 2026-06-29T14:45:53+08:00",
                "2026-06-29T15:00:02+08:00 sync",
                "2026-06-29T15:00:03+08:00 running",
                "2026-06-29T15:02:02+08:00 sync",
                "2026-06-29T15:02:03+08:00 running",
                "2026-06-29T15:04:02+08:00 sync",
                "2026-06-29T15:04:03+08:00 running",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        now=datetime.fromisoformat("2026-06-29T15:05:00+08:00"),
    )

    assert report["status"] == "running"
    assert report["scp_stderr_missing_artifact_line_count"] == 2
    assert report["scp_stderr_stale_missing_artifacts"] is True
    assert report["scp_stderr_persistent_missing_artifacts"] is False
    assert report["scp_stderr_warnings"] == []

    (monitor / "scp_stderr.log").write_text(
        "\n".join(
            [
                "scp: G:/lxy/blockcipher-structure-adaptive-nd-runs/unit_run/logs: No such file or directory",
                "scp: G:/lxy/blockcipher-structure-adaptive-nd-runs/unit_run/results: No such file or directory",
                "scp: G:/lxy/blockcipher-structure-adaptive-nd-runs/unit_run/logs: No such file or directory",
                "scp: G:/lxy/blockcipher-structure-adaptive-nd-runs/unit_run/results: No such file or directory",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        now=datetime.fromisoformat("2026-06-29T15:05:00+08:00"),
    )

    assert report["status"] == "remote_artifacts_missing"
    assert report["needs_main_thread_intervention"] is True
    assert report["scp_stderr_errors"] == []
    assert report["scp_stderr_missing_artifact_line_count"] == 4
    assert report["scp_stderr_persistent_missing_artifacts"] is True

    synced_logs = run_root / "logs"
    synced_logs.mkdir()
    (synced_logs / "launch_env.txt").write_text("started\n", encoding="utf-8")
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        now=datetime.fromisoformat("2026-06-29T15:05:00+08:00"),
    )

    assert report["status"] == "running"
    assert report["needs_main_thread_intervention"] is False
    assert report["scp_stderr_persistent_missing_artifacts"] is True

    (monitor / "scp_stderr.log").write_text("scp: Connection reset by peer\n", encoding="utf-8")
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        now=datetime.fromisoformat("2026-06-29T15:05:00+08:00"),
    )

    assert report["status"] == "unhealthy"
    assert report["needs_main_thread_intervention"] is True
    assert report["scp_stderr_errors"] == ["scp: Connection reset by peer"]

    (monitor / "scp_stderr.log").write_text("", encoding="utf-8")

    (run_root / "done.marker").write_text("done\n", encoding="utf-8")
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
    )

    assert report["status"] == "completed_missing_results"
    assert report["needs_main_thread_intervention"] is True
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []
    assert report["done_markers"] == ["done.marker"]

    results = run_root / "results"
    results.mkdir()
    (results / f"{run_id}.jsonl").write_text("\n", encoding="utf-8")
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
    )

    assert report["status"] == "results_empty"
    assert report["needs_main_thread_intervention"] is True
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []
    assert report["results_jsonl_exists"] is True
    assert report["results_jsonl_line_count"] == 0

    (results / f"{run_id}.jsonl").write_text("{}\n", encoding="utf-8")
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
    )

    assert report["status"] == "result_ready"
    assert report["postprocess_allowed"] is True
    assert report["results_jsonl_exists"] is True
    assert report["results_jsonl_line_count"] == 1
    assert report["expected_rows"] == 0
    assert report["postprocess_command"] == [
        "env",
        "UV_CACHE_DIR=/tmp/uv-cache",
        "MPLCONFIGDIR=/tmp/mplconfig",
        "uv",
        "run",
        "python",
        "scripts/postprocess-invp-result",
        "--plan",
        str(plan),
        "--results",
        str(results / f"{run_id}.jsonl"),
        "--output-dir",
        str(run_root),
        "--run-id",
        run_id,
        "--expected-rows",
        "1",
        "--update-plan-doc",
        str(plan_doc),
    ]

    (run_root / "failed.marker").write_text("failed\n", encoding="utf-8")
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
    )

    assert report["status"] == "failed"
    assert report["needs_main_thread_intervention"] is True
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []
    assert report["failed_markers"] == ["failed.marker"]


def test_monitor_health_marks_launch_stalled_before_training_logs(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "launch_stalled_unit"
    run_root = root / run_id
    monitor = run_root / "monitor"
    logs = run_root / "logs"
    monitor.mkdir(parents=True)
    logs.mkdir()
    (monitor / "monitor.log").write_text(
        "\n".join(
            [
                "monitor_start 2026-07-01T14:45:53+08:00",
                "2026-07-01T14:49:20+08:00 sync",
                "2026-07-01T14:49:21+08:00 running",
                "2026-07-01T15:03:35+08:00 sync",
                "2026-07-01T15:03:36+08:00 running",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / f"{run_id}_gpu_info.txt").write_text("gpu\n", encoding="utf-8")
    (logs / f"{run_id}_launch_env.txt").write_text("run_id=launch_stalled_unit\n", encoding="utf-8")
    (logs / f"{run_id}_torch_info.txt").write_text("", encoding="utf-8")
    (logs / f"{run_id}_torch_info_stderr.txt").write_text("", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=3,
        now=datetime.fromisoformat("2026-07-01T15:04:00+08:00"),
    )

    assert report["status"] == "launch_stalled"
    assert report["needs_main_thread_intervention"] is True
    assert report["launch_state"]["is_stalled"] is True
    assert report["launch_state"]["reason"] == "torch_info_empty_before_git_or_training"

    (logs / f"{run_id}_started.marker").write_text("started\n", encoding="utf-8")
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=3,
        now=datetime.fromisoformat("2026-07-01T15:04:00+08:00"),
    )

    assert report["status"] == "running"
    assert report["needs_main_thread_intervention"] is False
    assert report["launch_state"]["is_stalled"] is False


def test_monitor_health_requires_expected_result_rows(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_matrix"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-06-29T15:00:00+08:00 running\n", encoding="utf-8")
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    results = root / run_id / "results"
    results.mkdir()
    result_path = results / f"{run_id}.jsonl"
    result_path.write_text("{}\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=2,
    )

    assert report["status"] == "results_incomplete"
    assert report["results_jsonl_line_count"] == 1
    assert report["expected_rows"] == 2
    assert report["needs_main_thread_intervention"] is True
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []

    result_path.write_text("{}\n{}\n", encoding="utf-8")
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=2,
    )

    assert report["status"] == "result_ready"
    assert report["postprocess_allowed"] is True


def test_monitor_health_keeps_running_when_partial_rows_arrive_before_done(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_matrix_partial_running"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-06-29T15:00:00+08:00 running\n", encoding="utf-8")
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    results = root / run_id / "results"
    results.mkdir()
    (results / f"{run_id}.jsonl").write_text("{}\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=2,
        now=datetime.fromisoformat("2026-06-29T15:01:00+08:00"),
    )

    assert report["status"] == "running"
    assert report["results_jsonl_line_count"] == 1
    assert report["expected_rows"] == 2
    assert report["needs_main_thread_intervention"] is False
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []

    (root / run_id / "done.marker").write_text("done\n", encoding="utf-8")
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=2,
        now=datetime.fromisoformat("2026-06-29T15:01:00+08:00"),
    )

    assert report["status"] == "results_incomplete"
    assert report["needs_main_thread_intervention"] is True
    assert report["postprocess_allowed"] is False


def test_monitor_health_distinguishes_postprocessed_and_failed_postprocess(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_postprocessed"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    results = root / run_id / "results"
    results.mkdir()
    (results / f"{run_id}.jsonl").write_text("{}\n", encoding="utf-8")

    (monitor / "monitor.log").write_text(
        "\n".join(
            [
                "2026-06-29T15:00:00+08:00 result_ready",
                "2026-06-29T15:00:03+08:00 postprocess_done",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = monitor_health_report(run_id=run_id, root=root, expected_rows=1)

    assert report["status"] == "postprocessed"
    assert report["needs_main_thread_intervention"] is True
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []

    (monitor / "monitor.log").write_text(
        "\n".join(
            [
                "2026-06-29T15:00:00+08:00 result_ready",
                "2026-06-29T15:00:03+08:00 postprocess_failed",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = monitor_health_report(run_id=run_id, root=root, expected_rows=1)

    assert report["status"] == "postprocess_failed"
    assert report["needs_main_thread_intervention"] is True
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []


def test_monitor_health_emits_route_specific_postprocess_commands(tmp_path):
    root = tmp_path / "remote_results"
    plan = tmp_path / "plan.csv"
    plan.write_text("model_key\nrow_a\nrow_b\n", encoding="utf-8")
    plan_doc = tmp_path / "plan.md"
    plan_doc.write_text("# plan\n", encoding="utf-8")
    summary_doc = tmp_path / "summary.md"
    summary_doc.write_text("# summary\n", encoding="utf-8")

    attribution_run = "unit_attr"
    attribution_root = root / attribution_run
    attribution_monitor = attribution_root / "monitor"
    attribution_monitor.mkdir(parents=True)
    (attribution_monitor / "monitor.log").write_text("2026-06-29T15:00:00+08:00 running\n", encoding="utf-8")
    (attribution_monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    attribution_results = attribution_root / "results"
    attribution_results.mkdir()
    attribution_jsonl = attribution_results / f"{attribution_run}.jsonl"
    attribution_jsonl.write_text("{}\n{}\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=attribution_run,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        plan_doc_paths=[summary_doc],
        postprocess_kind="invp_attribution",
    )

    assert report["status"] == "result_ready"
    assert report["expected_rows"] == 2
    assert report["postprocess_command"] == [
        "env",
        "UV_CACHE_DIR=/tmp/uv-cache",
        "MPLCONFIGDIR=/tmp/mplconfig",
        "uv",
        "run",
        "python",
        "scripts/postprocess-invp-attribution-controls",
        "--plan",
        str(plan),
        "--results",
        str(attribution_jsonl),
        "--output-dir",
        str(attribution_root),
        "--run-id",
        attribution_run,
        "--expected-rows",
        "2",
        "--update-plan-doc",
        str(plan_doc),
        "--update-plan-doc",
        str(summary_doc),
    ]

    ddt_run = "unit_ddt"
    ddt_root = root / ddt_run
    ddt_monitor = ddt_root / "monitor"
    ddt_monitor.mkdir(parents=True)
    (ddt_monitor / "monitor.log").write_text("2026-06-29T15:00:00+08:00 running\n", encoding="utf-8")
    (ddt_monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    ddt_results = ddt_root / "results"
    ddt_results.mkdir()
    ddt_jsonl = ddt_results / f"{ddt_run}.jsonl"
    ddt_jsonl.write_text("{}\n{}\n{}\n{}\n{}\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=ddt_run,
        root=root,
        plan_path=plan,
        expected_rows=5,
        postprocess_kind="ddt_graph",
    )

    assert report["status"] == "result_ready"
    assert report["expected_rows"] == 5
    assert report["postprocess_command"] == [
        "env",
        "UV_CACHE_DIR=/tmp/uv-cache",
        "MPLCONFIGDIR=/tmp/mplconfig",
        "uv",
        "run",
        "python",
        "scripts/postprocess-ddt-graph-result",
        "--plan",
        str(plan),
        "--results",
        str(ddt_jsonl),
        "--output-dir",
        str(ddt_root),
        "--run-id",
        ddt_run,
        "--expected-rows",
        "5",
    ]

    topology_run = "unit_topology"
    topology_root = root / topology_run
    topology_monitor = topology_root / "monitor"
    topology_monitor.mkdir(parents=True)
    (topology_monitor / "monitor.log").write_text("2026-06-29T17:00:00+08:00 running\n", encoding="utf-8")
    (topology_monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    topology_results = topology_root / "results"
    topology_results.mkdir()
    topology_jsonl = topology_results / f"{topology_run}.jsonl"
    topology_jsonl.write_text("{}\n{}\n{}\n", encoding="utf-8")
    topology_plan = Path(
        "configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_r7_262k.csv"
    )
    topology_doc = tmp_path / "topology-plan.md"

    report = monitor_health_report(
        run_id=topology_run,
        root=root,
        plan_path=topology_plan,
        plan_doc_paths=[topology_doc],
        postprocess_kind="topology_aware",
    )

    assert report["status"] == "result_ready"
    assert report["expected_rows"] == 3
    assert report["postprocess_allowed"] is True
    assert report["postprocess_command"] == [
        "env",
        "UV_CACHE_DIR=/tmp/uv-cache",
        "MPLCONFIGDIR=/tmp/mplconfig",
        "uv",
        "run",
        "python",
        "scripts/postprocess-topology-aware-result",
        "--plan",
        str(topology_plan),
        "--results",
        str(topology_jsonl),
        "--output-dir",
        str(topology_root),
        "--run-id",
        topology_run,
        "--expected-rows",
        "3",
        "--update-plan-doc",
        str(topology_doc),
    ]

    candidate_run = "i1_candidate_trail_result_ready"
    candidate_root = root / candidate_run
    candidate_monitor = candidate_root / "monitor"
    candidate_monitor.mkdir(parents=True)
    (candidate_monitor / "monitor.log").write_text("2026-06-29T18:00:00+08:00 running\n", encoding="utf-8")
    (candidate_monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    candidate_results = candidate_root / "results"
    candidate_results.mkdir()
    candidate_jsonl = candidate_results / f"{candidate_run}.jsonl"
    candidate_jsonl.write_text("{}\n{}\n{}\n", encoding="utf-8")
    candidate_plan = (
        Path("configs/experiment/innovation1/")
        / "innovation1_spn_present_candidate_trail_consistency_smoke.json"
    )
    candidate_doc = tmp_path / "candidate-plan.md"

    report = monitor_health_report(
        run_id=candidate_run,
        root=root,
        plan_path=candidate_plan,
        plan_doc_paths=[candidate_doc],
        expected_rows=3,
        postprocess_kind="candidate_trail",
    )

    assert report["status"] == "result_ready"
    assert report["expected_rows"] == 3
    assert report["postprocess_allowed"] is True
    assert report["postprocess_command"] == [
        "env",
        "UV_CACHE_DIR=/tmp/uv-cache",
        "MPLCONFIGDIR=/tmp/mplconfig",
        "uv",
        "run",
        "python",
        "scripts/postprocess-candidate-trail",
        "--plan",
        str(candidate_plan),
        "--results",
        str(candidate_jsonl),
        "--output-dir",
        str(candidate_root),
        "--run-id",
        candidate_run,
        "--expected-rows",
        "3",
        "--update-plan-doc",
        str(candidate_doc),
    ]

    candidate_default_run = "i1_candidate_trail_default_rows"
    candidate_default_root = root / candidate_default_run
    candidate_default_monitor = candidate_default_root / "monitor"
    candidate_default_monitor.mkdir(parents=True)
    (candidate_default_monitor / "monitor.log").write_text(
        "2026-06-29T18:30:00+08:00 running\n",
        encoding="utf-8",
    )
    (candidate_default_monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    candidate_default_results = candidate_default_root / "results"
    candidate_default_results.mkdir()
    candidate_default_jsonl = candidate_default_results / f"{candidate_default_run}.jsonl"
    candidate_default_jsonl.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=candidate_default_run,
        root=root,
        plan_path=candidate_plan,
        postprocess_kind="candidate_trail",
    )

    assert report["status"] == "result_ready"
    assert report["expected_rows"] == 4
    assert report["postprocess_command"] == [
        "env",
        "UV_CACHE_DIR=/tmp/uv-cache",
        "MPLCONFIGDIR=/tmp/mplconfig",
        "uv",
        "run",
        "python",
        "scripts/postprocess-candidate-trail",
        "--plan",
        str(candidate_plan),
        "--results",
        str(candidate_default_jsonl),
        "--output-dir",
        str(candidate_default_root),
        "--run-id",
        candidate_default_run,
        "--expected-rows",
        "4",
    ]

    candidate_incomplete_run = "i1_candidate_trail_incomplete_rows"
    candidate_incomplete_root = root / candidate_incomplete_run
    candidate_incomplete_monitor = candidate_incomplete_root / "monitor"
    candidate_incomplete_monitor.mkdir(parents=True)
    (candidate_incomplete_monitor / "monitor.log").write_text(
        "2026-06-29T18:45:00+08:00 running\n",
        encoding="utf-8",
    )
    (candidate_incomplete_monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    candidate_incomplete_results = candidate_incomplete_root / "results"
    candidate_incomplete_results.mkdir()
    candidate_incomplete_jsonl = candidate_incomplete_results / f"{candidate_incomplete_run}.jsonl"
    candidate_incomplete_jsonl.write_text("{}\n{}\n{}\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=candidate_incomplete_run,
        root=root,
        plan_path=candidate_plan,
        postprocess_kind="candidate_trail",
    )

    assert report["status"] == "results_incomplete"
    assert report["expected_rows"] == 4
    assert report["results_jsonl_line_count"] == 3
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []

    pairset_run = "i1_pairset_aggregation_control_unit"
    pairset_root = root / pairset_run
    pairset_monitor = pairset_root / "monitor"
    pairset_monitor.mkdir(parents=True)
    (pairset_monitor / "monitor.log").write_text("2026-06-29T15:00:00+08:00 running\n", encoding="utf-8")
    (pairset_monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    pairset_results = pairset_root / "results"
    pairset_results.mkdir()
    pairset_jsonl = pairset_results / f"{pairset_run}.jsonl"
    pairset_jsonl.write_text("{}\n{}\n", encoding="utf-8")
    stage_a_jsonl = pairset_results / f"{pairset_run}.jsonl".replace(
        "i1_pairset_aggregation_control", "i1_pairset_single_pair_scorer", 1
    )
    frozen_summary = pairset_results / "frozen_aggregation_summary.json"
    checkpoint = pairset_root / "checkpoints" / "single_pair_invp.pt"

    report = monitor_health_report(
        run_id=pairset_run,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        expected_rows=2,
        postprocess_kind="pairset_aggregation",
    )

    assert report["status"] == "waiting_for_auxiliary_artifacts"
    assert report["needs_main_thread_intervention"] is False
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []
    assert report["auxiliary_artifacts"] == [
        {
            "role": "single_pair_results",
            "path": str(stage_a_jsonl),
            "exists": False,
        },
        {
            "role": "single_pair_checkpoint",
            "path": str(checkpoint),
            "exists": False,
        },
        {
            "role": "frozen_summary",
            "path": str(frozen_summary),
            "exists": False,
        }
    ]

    stage_a_jsonl.write_text("{}\n", encoding="utf-8")
    report = monitor_health_report(
        run_id=pairset_run,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        expected_rows=2,
        postprocess_kind="pairset_aggregation",
    )

    assert report["status"] == "waiting_for_auxiliary_artifacts"
    assert [artifact["exists"] for artifact in report["auxiliary_artifacts"]] == [True, False, False]

    frozen_summary.write_text('{"status":"pass"}\n', encoding="utf-8")
    report = monitor_health_report(
        run_id=pairset_run,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        expected_rows=2,
        postprocess_kind="pairset_aggregation",
    )

    assert report["status"] == "waiting_for_auxiliary_artifacts"
    assert report["postprocess_allowed"] is False
    assert [artifact["exists"] for artifact in report["auxiliary_artifacts"]] == [True, False, True]

    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    report = monitor_health_report(
        run_id=pairset_run,
        root=root,
        plan_path=plan,
        plan_doc_path=plan_doc,
        expected_rows=2,
        postprocess_kind="pairset_aggregation",
    )

    assert report["status"] == "result_ready"
    assert report["expected_rows"] == 2
    assert [artifact["exists"] for artifact in report["auxiliary_artifacts"]] == [True, True, True]
    assert report["postprocess_command"] == [
        "env",
        "UV_CACHE_DIR=/tmp/uv-cache",
        "MPLCONFIGDIR=/tmp/mplconfig",
        "uv",
        "run",
        "python",
        "scripts/postprocess-pairset-aggregation",
        "--plan",
        str(plan),
        "--learned-results",
        str(pairset_jsonl),
        "--frozen-summary",
        str(frozen_summary),
        "--output-dir",
        str(pairset_root),
        "--run-id",
        pairset_run,
        "--expected-rows",
        "2",
        "--update-plan-doc",
        str(plan_doc),
    ]


def test_monitor_health_pairset_done_without_frozen_summary_needs_intervention(tmp_path):
    root = tmp_path / "remote_results"
    plan = tmp_path / "plan.csv"
    plan.write_text("model_key\nrow_a\nrow_b\n", encoding="utf-8")
    run_id = "i1_pairset_aggregation_control_missing_aux"
    run_root = root / run_id
    monitor = run_root / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-06-29T15:00:00+08:00 running\n", encoding="utf-8")
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    (run_root / "done.marker").write_text("done\n", encoding="utf-8")
    results = run_root / "results"
    results.mkdir()
    (results / f"{run_id}.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        plan_path=plan,
        expected_rows=2,
        postprocess_kind="pairset_aggregation",
    )

    assert report["status"] == "completed_missing_auxiliary_artifacts"
    assert report["needs_main_thread_intervention"] is True
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []
    assert report["auxiliary_artifacts"] == [
        {
            "role": "single_pair_results",
            "path": str(results / "i1_pairset_single_pair_scorer_missing_aux.jsonl"),
            "exists": False,
        },
        {
            "role": "single_pair_checkpoint",
            "path": str(run_root / "checkpoints" / "single_pair_invp.pt"),
            "exists": False,
        },
        {
            "role": "frozen_summary",
            "path": str(results / "frozen_aggregation_summary.json"),
            "exists": False,
        }
    ]


def test_monitor_health_keeps_running_when_jsonl_exists_but_empty_before_done(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_running_empty"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-06-29T20:25:50+08:00 running\n", encoding="utf-8")
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    (logs / f"{run_id}_progress.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "checkpoint_improved",
                        "model": "present_nibble_invp_only_spn_only",
                        "index": 1,
                        "total": 5,
                        "epoch": 8,
                        "epochs": 20,
                        "metric": "val_auc",
                        "value": 0.7908,
                    }
                ),
                json.dumps(
                    {
                        "event": "train_batch",
                        "stage": "training",
                        "model": "present_nibble_invp_only_spn_only",
                        "index": 1,
                        "total": 5,
                        "epoch": 9,
                        "epochs": 20,
                        "step": 306,
                        "steps_per_epoch": 512,
                        "train_rows_seen": 313344,
                        "train_rows": 524288,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    results = root / run_id / "results"
    results.mkdir()
    (results / f"{run_id}.jsonl").write_text("", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=1,
        now=datetime.fromisoformat("2026-06-29T20:26:30+08:00"),
    )

    assert report["status"] == "running"
    assert report["results_jsonl_exists"] is True
    assert report["results_jsonl_line_count"] == 0
    assert report["expected_rows"] == 1
    assert report["needs_main_thread_intervention"] is False
    assert report["postprocess_allowed"] is False
    assert report["progress_summary"]["exists"] is True
    assert report["progress_summary"]["latest_event"] == "train_batch"
    assert report["progress_summary"]["model"] == "present_nibble_invp_only_spn_only"
    assert report["progress_summary"]["index"] == 1
    assert report["progress_summary"]["total"] == 5
    assert report["progress_summary"]["epoch"] == 9
    assert report["progress_summary"]["epochs"] == 20
    assert report["progress_summary"]["step"] == 306
    assert report["progress_summary"]["steps_per_epoch"] == 512
    assert report["progress_summary"]["train_rows_seen"] == 313344
    assert report["progress_summary"]["model_progress_percent"] == 9.0
    assert report["progress_summary"]["epoch_progress_percent"] == pytest.approx(59.766)
    assert report["progress_summary"]["train_rows_progress_percent"] == pytest.approx(59.766)
    assert report["progress_summary"]["best_checkpoint_metric"] == 0.7908
    assert report["progress_summary"]["best_epoch"] == 8
    assert report["progress_summary"]["checkpoint_metric"] == "val_auc"


def test_monitor_health_reports_feature_cache_progress_percent(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_cache_progress"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-02T16:25:00+08:00 running\n", encoding="utf-8")
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    (logs / "candidate_trail_linear_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "candidate_cache_positive_chunk",
                "time": 1000.0,
                "rows_done": 32768,
                "total_rows": 524288,
                "class_rows_done": 32768,
                "class_total": 262144,
                "chunk_rows": 32768,
                "workers": 2,
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "event": "candidate_cache_positive_chunk",
                "time": 1064.0,
                "rows_done": 65536,
                "total_rows": 524288,
                "class_rows_done": 65536,
                "class_total": 262144,
                "chunk_rows": 8192,
                "workers": 2,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=4,
        now=datetime.fromisoformat("2026-07-02T16:26:00+08:00"),
    )

    progress = report["progress_summary"]
    assert report["status"] == "running"
    assert progress["latest_event"] == "candidate_cache_positive_chunk"
    assert progress["cache_rows_done"] == 65536
    assert progress["cache_total_rows"] == 524288
    assert progress["cache_class_rows_done"] == 65536
    assert progress["cache_class_total"] == 262144
    assert progress["cache_chunk_rows"] == 8192
    assert progress["cache_chunk_index"] == 8
    assert progress["cache_class_chunk_index"] == 8
    assert progress["cache_total_progress_percent"] == pytest.approx(12.5)
    assert progress["cache_class_progress_percent"] == pytest.approx(25.0)
    assert progress["cache_rows_per_second"] == pytest.approx(512.0)
    assert progress["cache_eta_seconds"] == 896


def test_monitor_health_keeps_cache_eta_null_without_progress_timestamps(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_cache_progress_without_timestamps"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-02T16:25:00+08:00 running\n", encoding="utf-8")
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    (logs / "candidate_trail_linear_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "candidate_cache_positive_chunk",
                "rows_done": 65536,
                "total_rows": 524288,
                "class_rows_done": 65536,
                "class_total": 262144,
                "chunk_rows": 8192,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=4,
        now=datetime.fromisoformat("2026-07-02T16:26:00+08:00"),
    )

    progress = report["progress_summary"]
    assert progress["cache_total_progress_percent"] == pytest.approx(12.5)
    assert progress["cache_chunk_rows"] == 8192
    assert progress["cache_chunk_index"] == 8
    assert progress["cache_class_chunk_index"] == 8
    assert progress["cache_rows_per_second"] is None
    assert progress["cache_eta_seconds"] is None


def test_monitor_health_marks_stale_running_heartbeat(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_stale"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text(
        "2026-06-29T15:00:00+08:00 running\n",
        encoding="utf-8",
    )
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        stale_after_seconds=1800,
        now=datetime.fromisoformat("2026-06-29T15:45:01+08:00"),
    )

    assert report["status"] == "stale_monitor"
    assert report["heartbeat"]["is_stale"] is True
    assert report["heartbeat"]["age_seconds"] == 2701
    assert report["needs_main_thread_intervention"] is True
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []


def test_monitor_health_does_not_treat_tmux_check_error_as_missing_session():
    status = _health_status(
        run_root_exists=True,
        has_synced_remote_artifacts=False,
        results_jsonl_exists=False,
        results_jsonl_line_count=0,
        expected_rows=None,
        done_markers=[],
        failed_markers=[],
        stderr_text="",
        scp_stderr_report={
            "errors": [],
            "warnings": [],
            "tail": [],
            "missing_artifact_line_count": 0,
            "persistent_missing_artifacts": False,
        },
        launch_state={"is_stalled": False},
        recent_monitor_lines=["2026-06-29T16:38:40+08:00 running"],
        heartbeat={"is_stale": False},
        tmux={
            "checked": True,
            "session": "unit",
            "exists": None,
            "returncode": 1,
            "stderr": "error connecting to /tmp/tmux-1000/default (Operation not permitted)",
            "check_error": True,
        },
    )

    assert status == "running"


def test_seed1_remote_readiness_gate_passes_for_prepared_invp_confirmation():
    report = remote_readiness_report(
        Path("configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json")
    )

    assert report["status"] == "pass"
    assert report["run_id"] == "i1_invp_only_r7_1m_seed1_gpu1_20260629"
    assert report["plan_rows"] == 1
    assert report["expected_rows"] == 1
    assert report["max_samples_per_class"] == 1_000_000
    assert report["errors"] == []
    assert "medium_scale_dataset_cache" in report["checked_invariants"]
    assert "candidate_trail_protocol_lock" not in report["checked_invariants"]
    assert "pairset_aggregation_stage_lock" not in report["checked_invariants"]


def test_remote_readiness_gate_rejects_bad_medium_scale_cache(tmp_path):
    config = json.loads(
        Path("configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json").read_text(
            encoding="utf-8"
        )
    )
    config["dataset_cache"] = False
    config["dataset_cache_root"] = "C:\\Users\\bad\\cache"
    path = tmp_path / "bad_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "fail"
    assert any("dataset_cache must be true" in error for error in report["errors"])
    assert any("dataset_cache_root must stay under" in error for error in report["errors"])


def test_remote_readiness_gate_accepts_candidate_trail_json_plan(tmp_path):
    config = {
        "run_id": "i1_candidate_trail_consistency_smoke_remote_unit",
        "task_name": "i1_candidate_trail_consistency_smoke_remote_unit",
        "archive_work_id": "i1_candidate_trail_consistency_smoke_remote_unit",
        "plan": (
            "configs\\experiment\\innovation1\\"
            "innovation1_spn_present_candidate_trail_consistency_smoke.json"
        ),
        "expected_rows": 1,
        "device": "cuda:0",
        "epochs": 1,
        "batch_size": 2048,
        "learning_rate": 0.01,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "negative_mode": "encrypted_random_plaintexts",
        "validation_key": "0x11111111111111111111",
        "key_rotation_interval": 0,
        "feature_mode": "cell_structured",
        "feature_cache_root": (
            "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\candidate_trail_smoke_cache"
        ),
        "branch": "main",
        "repo_url": "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git",
        "source_commit": "recorded_in_remote_run_script_git_revision",
        "result_sync": "local_tmux_monitor_scp_fallback",
        "monitor_script_name": "monitor_i1_candidate_trail_consistency_smoke_remote_unit.sh",
        "claim_scope": "candidate-trail JSON-plan readiness smoke only; not accuracy evidence",
        "launch_policy": (
            "candidate-trail smoke readiness; use pushed GitHub commit; keep artifacts "
            "under G:\\lxy; generated commands must use cmd.exe /c; local tmux monitor "
            "retrieves results automatically"
        ),
    }
    path = tmp_path / "candidate_trail_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "pass"
    assert report["plan_rows"] == 1
    assert report["expected_rows"] == 1
    assert report["max_samples_per_class"] == 2
    assert report["errors"] == []
    assert "medium_scale_dataset_cache" not in report["checked_invariants"]
    assert "candidate_trail_protocol_lock" in report["checked_invariants"]
    assert "pairset_aggregation_stage_lock" not in report["checked_invariants"]


def test_remote_readiness_gate_rejects_candidate_trail_without_feature_mode(tmp_path):
    config = json.loads(
        Path("configs/remote/innovation1_spn_present_candidate_trail_consistency_smoke_gpu1_20260701.json").read_text(
            encoding="utf-8"
        )
    )
    config.pop("feature_mode")
    path = tmp_path / "candidate_trail_missing_feature_mode.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "fail"
    assert any("candidate_trail feature_mode must be explicit cell-structured control" in error for error in report["errors"])


def test_remote_readiness_gate_rejects_bad_candidate_feature_cache_workers(tmp_path):
    config = json.loads(
        Path("configs/remote/innovation1_spn_present_candidate_trail_consistency_smoke_gpu1_20260701.json").read_text(
            encoding="utf-8"
        )
    )
    config["feature_cache_workers"] = 0
    path = tmp_path / "candidate_trail_bad_workers.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "fail"
    assert any("candidate_trail feature_cache_workers must be >= 1" in error for error in report["errors"])


def test_remote_readiness_warns_candidate_medium_single_feature_worker(tmp_path):
    config = json.loads(
        Path("configs/remote/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702.json").read_text(
            encoding="utf-8"
        )
    )
    config["feature_cache_workers"] = 1
    path = tmp_path / "candidate_trail_single_worker.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "pass"
    assert any("candidate_trail medium-scale feature_cache_workers=1" in warning for warning in report["warnings"])


def test_remote_readiness_gate_requires_candidate_matrix_runner_script(tmp_path):
    plan = tmp_path / "candidate_matrix.json"
    plan.write_text(
        json.dumps(
            {
                "output": str(tmp_path / "candidate_matrix.jsonl"),
                "common": {
                    "rounds": 7,
                    "seed": 0,
                    "samples_per_class": 65536,
                    "pairs_per_sample": 16,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "zhang_wang_case2_official_mcnd",
                    "difference_profile": "present_zhang_wang2022_mcnd",
                    "difference_member": 0,
                    "validation_key": "0x11111111111111111111",
                    "key_rotation_interval": 0,
                    "learning_rate": 0.01,
                },
                "rows": [
                    {
                        "row_type": "external_anchor",
                        "model": "present_nibble_invp_only_spn_only",
                        "anchor_auc": 0.79,
                        "anchor_calibrated_accuracy": 0.72,
                    },
                    {"model": "linear"},
                    {"model": "mlp"},
                    {"model": "shuffled_cells", "feature_mode": "cell_structured_shuffled"},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "run_id": "i1_candidate_trail_matrix_remote_unit",
        "task_name": "i1_candidate_trail_matrix_remote_unit",
        "archive_work_id": "i1_candidate_trail_matrix_remote_unit",
        "plan": str(plan),
        "expected_rows": 4,
        "device": "cuda:0",
        "epochs": 20,
        "batch_size": 2048,
        "learning_rate": 0.01,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "negative_mode": "encrypted_random_plaintexts",
        "validation_key": "0x11111111111111111111",
        "key_rotation_interval": 0,
        "dataset_cache": True,
        "dataset_cache_root": "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\shared_dataset_cache",
        "dataset_cache_chunk_size": 8192,
        "dataset_cache_workers": 4,
        "feature_cache_root": "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\candidate_trail_cache",
        "branch": "main",
        "repo_url": "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git",
        "source_commit": "recorded_in_remote_run_script_git_revision",
        "result_sync": "local_tmux_monitor_scp_fallback",
        "monitor_script_name": "monitor_i1_candidate_trail_matrix_remote_unit.sh",
        "claim_scope": "candidate-trail medium matrix readiness unit",
        "launch_policy": "candidate-trail matrix; keep artifacts under G:\\lxy; cmd.exe /c",
    }
    path = tmp_path / "candidate_matrix_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    missing_runner = remote_readiness_report(path)
    assert missing_runner["status"] == "fail"
    assert any("runner_script=scripts/spn-candidate-evidence-matrix" in error for error in missing_runner["errors"])

    config["runner_script"] = "scripts/spn-candidate-evidence-matrix"
    config["feature_cache_workers"] = 4
    path.write_text(json.dumps(config), encoding="utf-8")
    ready = remote_readiness_report(path)

    assert ready["status"] == "pass"
    assert ready["plan_rows"] == 4
    assert "candidate_trail_protocol_lock" in ready["checked_invariants"]
    assert "medium_scale_dataset_cache" in ready["checked_invariants"]
    assert not any("feature_cache_workers=1" in warning for warning in ready["warnings"])


def test_remote_readiness_gate_requires_transition_spectrum_matrix_runner_script(tmp_path):
    plan = tmp_path / "transition_spectrum_matrix.json"
    plan.write_text(
        json.dumps(
            {
                "output": str(tmp_path / "transition_spectrum_matrix.jsonl"),
                "common": {
                    "rounds": 7,
                    "seed": 0,
                    "samples_per_class": 65536,
                    "pairs_per_sample": 16,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "zhang_wang_case2_official_mcnd",
                    "difference_profile": "present_zhang_wang2022_mcnd",
                    "difference_member": 0,
                    "validation_key": "0x11111111111111111111",
                    "key_rotation_interval": 0,
                    "learning_rate": 0.003,
                },
                "rows": [
                    {
                        "row_type": "external_anchor",
                        "model": "present_nibble_invp_only_spn_only",
                        "anchor_auc": 0.79,
                        "anchor_calibrated_accuracy": 0.72,
                    },
                    {"model": "linear"},
                    {"model": "mlp"},
                    {"model": "shuffled_p"},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "run_id": "i1_transition_spectrum_matrix_remote_unit",
        "task_name": "i1_transition_spectrum_matrix_remote_unit",
        "archive_work_id": "i1_transition_spectrum_matrix_remote_unit",
        "plan": str(plan),
        "expected_rows": 4,
        "device": "cuda:0",
        "epochs": 20,
        "batch_size": 2048,
        "learning_rate": 0.003,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "negative_mode": "encrypted_random_plaintexts",
        "validation_key": "0x11111111111111111111",
        "key_rotation_interval": 0,
        "dataset_cache": True,
        "dataset_cache_root": "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\transition_spectrum_cache",
        "dataset_cache_chunk_size": 8192,
        "dataset_cache_workers": 4,
        "feature_cache_root": "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\transition_spectrum_cache",
        "branch": "main",
        "repo_url": "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git",
        "source_commit": "recorded_in_remote_run_script_git_revision",
        "result_sync": "local_tmux_monitor_scp_fallback",
        "monitor_script_name": "monitor_i1_transition_spectrum_matrix_remote_unit.sh",
        "claim_scope": "bit-transition-spectrum medium matrix readiness unit",
        "launch_policy": "bit-transition-spectrum matrix; keep artifacts under G:\\lxy; cmd.exe /c",
    }
    path = tmp_path / "transition_spectrum_matrix_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    missing_runner = remote_readiness_report(path)
    assert missing_runner["status"] == "fail"
    assert any("runner_script=scripts/spn-transition-spectrum-matrix" in error for error in missing_runner["errors"])

    config["runner_script"] = "scripts/spn-transition-spectrum-matrix"
    path.write_text(json.dumps(config), encoding="utf-8")
    ready = remote_readiness_report(path)

    assert ready["status"] == "pass"
    assert ready["plan_rows"] == 4
    assert "transition_spectrum_protocol_lock" in ready["checked_invariants"]
    assert "candidate_trail_protocol_lock" not in ready["checked_invariants"]
    assert "medium_scale_dataset_cache" in ready["checked_invariants"]


def test_remote_readiness_gate_rejects_transition_spectrum_missing_shuffled_control(tmp_path):
    plan = tmp_path / "transition_spectrum_missing_control.json"
    plan.write_text(
        json.dumps(
            {
                "output": str(tmp_path / "transition_spectrum_missing_control.jsonl"),
                "common": {
                    "rounds": 7,
                    "seed": 0,
                    "samples_per_class": 65536,
                    "pairs_per_sample": 16,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "zhang_wang_case2_official_mcnd",
                    "difference_profile": "present_zhang_wang2022_mcnd",
                    "difference_member": 0,
                    "validation_key": "0x11111111111111111111",
                    "key_rotation_interval": 0,
                    "learning_rate": 0.003,
                },
                "rows": [
                    {
                        "row_type": "external_anchor",
                        "model": "present_nibble_invp_only_spn_only",
                        "anchor_auc": 0.79,
                        "anchor_calibrated_accuracy": 0.72,
                    },
                    {"model": "linear"},
                    {"model": "mlp"},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "run_id": "i1_transition_spectrum_missing_control_remote_unit",
        "task_name": "i1_transition_spectrum_missing_control_remote_unit",
        "archive_work_id": "i1_transition_spectrum_missing_control_remote_unit",
        "plan": str(plan),
        "runner_script": "scripts/spn-transition-spectrum-matrix",
        "expected_rows": 3,
        "device": "cuda:0",
        "epochs": 20,
        "batch_size": 2048,
        "learning_rate": 0.003,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "negative_mode": "encrypted_random_plaintexts",
        "validation_key": "0x11111111111111111111",
        "key_rotation_interval": 0,
        "dataset_cache": True,
        "dataset_cache_root": "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\transition_spectrum_cache",
        "dataset_cache_chunk_size": 8192,
        "dataset_cache_workers": 4,
        "feature_cache_root": "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\transition_spectrum_cache",
        "branch": "main",
        "repo_url": "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git",
        "source_commit": "recorded_in_remote_run_script_git_revision",
        "result_sync": "local_tmux_monitor_scp_fallback",
        "monitor_script_name": "monitor_i1_transition_spectrum_missing_control_remote_unit.sh",
        "claim_scope": "bit-transition-spectrum missing control unit",
        "launch_policy": "bit-transition-spectrum matrix; keep artifacts under G:\\lxy; cmd.exe /c",
    }
    path = tmp_path / "transition_spectrum_missing_control_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "fail"
    assert any("linear, mlp, and shuffled_p rows" in error for error in report["errors"])
    assert "transition_spectrum_protocol_lock" in report["checked_invariants"]


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
    assert all("time" in json.loads(line) for line in progress_text.splitlines())


def test_candidate_evidence_cache_supports_official_case2_protocol(tmp_path):
    progress_path = tmp_path / "official_progress.jsonl"
    features, labels = make_candidate_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=5,
        samples_per_class=3,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_official_mcnd",
        key_rotation_interval=0,
        beam_width=2,
        depth=2,
        feature_cache_root=tmp_path / "official_candidate_cache",
        feature_cache_chunk_size=2,
        progress_output=progress_path,
        split="train",
    )

    assert features.shape == (6, 126)
    assert labels.shape == (6,)
    assert set(np.unique(labels).tolist()) == {0, 1}
    progress_text = progress_path.read_text(encoding="utf-8")
    assert "candidate_cache_done" in progress_text
    assert '"sample_structure": "zhang_wang_case2_official_mcnd"' in progress_text


def test_candidate_evidence_cache_supports_parallel_workers(tmp_path):
    progress_path = tmp_path / "parallel_progress.jsonl"
    cache_root = tmp_path / "parallel_candidate_cache"
    features, labels = make_candidate_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=13,
        samples_per_class=4,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_official_mcnd",
        key_rotation_interval=0,
        beam_width=2,
        depth=2,
        feature_cache_root=cache_root,
        feature_cache_chunk_size=2,
        feature_cache_workers=2,
        progress_output=progress_path,
        split="train",
    )
    reused_features, reused_labels = make_candidate_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=13,
        samples_per_class=4,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_official_mcnd",
        key_rotation_interval=0,
        beam_width=2,
        depth=2,
        feature_cache_root=cache_root,
        feature_cache_chunk_size=2,
        feature_cache_workers=2,
        progress_output=progress_path,
        split="train",
    )

    assert features.shape == (8, 126)
    assert set(np.unique(labels).tolist()) == {0, 1}
    assert np.array_equal(np.asarray(features), np.asarray(reused_features))
    assert np.array_equal(np.asarray(labels), np.asarray(reused_labels))
    progress_text = progress_path.read_text(encoding="utf-8")
    assert '"workers": 2' in progress_text
    assert "candidate_cache_reuse" in progress_text

    metadata_files = list(cache_root.glob("train/*/metadata.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert "feature_cache_workers" not in metadata
    assert metadata["cache_version"] == 3


def test_candidate_evidence_cache_reuses_across_worker_counts(tmp_path):
    progress_path = tmp_path / "worker_reuse_progress.jsonl"
    cache_root = tmp_path / "worker_reuse_candidate_cache"
    common = {
        "rounds": 7,
        "key": 0,
        "input_difference": 0x9,
        "seed": 13,
        "samples_per_class": 4,
        "pairs_per_sample": 2,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "key_rotation_interval": 0,
        "beam_width": 2,
        "depth": 2,
        "feature_cache_root": cache_root,
        "feature_cache_chunk_size": 2,
        "progress_output": progress_path,
        "split": "train",
    }

    features, labels = make_candidate_dataset(**common, feature_cache_workers=1)
    reused_features, reused_labels = make_candidate_dataset(**common, feature_cache_workers=2)

    assert np.array_equal(np.asarray(features), np.asarray(reused_features))
    assert np.array_equal(np.asarray(labels), np.asarray(reused_labels))
    progress_text = progress_path.read_text(encoding="utf-8")
    assert '"workers": 1' in progress_text
    assert '"workers": 2' in progress_text
    assert "candidate_cache_reuse" in progress_text
    assert len(list(cache_root.glob("train/*/metadata.json"))) == 1


def test_candidate_evidence_cell_structured_and_shuffled_controls_differ(tmp_path):
    common = {
        "rounds": 7,
        "key": 0,
        "input_difference": 0x9,
        "seed": 7,
        "samples_per_class": 2,
        "pairs_per_sample": 2,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "key_rotation_interval": 0,
        "beam_width": 2,
        "depth": 2,
        "feature_cache_chunk_size": 1,
        "split": "train",
    }
    true_features, true_labels = make_candidate_dataset(
        **common,
        feature_mode="cell_structured",
        feature_cache_root=tmp_path / "true_cache",
        progress_output=tmp_path / "true_progress.jsonl",
    )
    shuffled_features, shuffled_labels = make_candidate_dataset(
        **common,
        feature_mode="cell_structured_shuffled",
        feature_cache_root=tmp_path / "shuffled_cache",
        progress_output=tmp_path / "shuffled_progress.jsonl",
    )

    assert true_features.shape == shuffled_features.shape == (4, 870)
    assert np.array_equal(np.asarray(true_labels), np.asarray(shuffled_labels))
    assert not np.array_equal(np.asarray(true_features), np.asarray(shuffled_features))
    true_progress = (tmp_path / "true_progress.jsonl").read_text(encoding="utf-8")
    shuffled_progress = (tmp_path / "shuffled_progress.jsonl").read_text(encoding="utf-8")
    assert '"feature_mode": "cell_structured"' in true_progress
    assert '"feature_mode": "cell_structured_shuffled"' in shuffled_progress


def test_candidate_evidence_cell_structured_cache_reuse_is_feature_mode_scoped(tmp_path):
    common = {
        "rounds": 7,
        "key": 0,
        "input_difference": 0x9,
        "seed": 11,
        "samples_per_class": 2,
        "pairs_per_sample": 2,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "key_rotation_interval": 0,
        "beam_width": 2,
        "depth": 2,
        "feature_cache_chunk_size": 1,
        "split": "train",
    }
    cache_root = tmp_path / "candidate_cache"
    true_progress = tmp_path / "true_progress.jsonl"
    first_features, first_labels = make_candidate_dataset(
        **common,
        feature_mode="cell_structured",
        feature_cache_root=cache_root,
        progress_output=true_progress,
    )
    reused_features, reused_labels = make_candidate_dataset(
        **common,
        feature_mode="cell_structured",
        feature_cache_root=cache_root,
        progress_output=true_progress,
    )
    shuffled_features, shuffled_labels = make_candidate_dataset(
        **common,
        feature_mode="cell_structured_shuffled",
        feature_cache_root=cache_root,
        progress_output=tmp_path / "shuffled_progress.jsonl",
    )

    assert np.array_equal(np.asarray(first_features), np.asarray(reused_features))
    assert np.array_equal(np.asarray(first_labels), np.asarray(reused_labels))
    assert np.array_equal(np.asarray(first_labels), np.asarray(shuffled_labels))
    assert not np.array_equal(np.asarray(first_features), np.asarray(shuffled_features))
    assert "candidate_cache_reuse" in true_progress.read_text(encoding="utf-8")

    metadata_files = list(cache_root.glob("train/*/metadata.json"))
    assert len(metadata_files) == 2
    metadata_modes = sorted(json.loads(path.read_text(encoding="utf-8"))["feature_mode"] for path in metadata_files)
    assert metadata_modes == ["cell_structured", "cell_structured_shuffled"]


def test_candidate_evidence_cli_outputs_gate_aligned_model_key(tmp_path):
    output = tmp_path / "candidate.jsonl"
    spn_candidate_evidence.main(
        [
            "--output",
            str(output),
            "--samples-per-class",
            "2",
            "--pairs-per-sample",
            "1",
            "--epochs",
            "1",
            "--model",
            "mlp",
            "--feature-cache-root",
            str(tmp_path / "cache"),
            "--feature-cache-chunk-size",
            "1",
            "--progress-output",
            str(tmp_path / "progress.jsonl"),
            "--device",
            "cpu",
        ]
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["route"] == "candidate_trail_consistency_mlp"
    assert row["model"] == "candidate_trail_consistency_mlp"
    assert row["training_model"] == "mlp"
    assert row["training_model_family"] == "mlp"
    assert row["feature_mode"] == "cell_structured"
    assert row["feature_cache_workers"] == 1
    assert row["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert row["negative_mode"] == "encrypted_random_plaintexts"
    assert row["key_rotation_interval"] == 0
    assert row["selected_model"] == "candidate_trail_consistency_mlp"
    assert row["auc"] == row["val_auc"]
    assert row["calibrated_accuracy"] == row["metrics"]["calibrated_accuracy"]
    assert 0.0 <= row["calibrated_threshold"] <= 1.0


def test_candidate_evidence_cli_outputs_shuffled_cell_control_key(tmp_path):
    output = tmp_path / "candidate_shuffled.jsonl"
    spn_candidate_evidence.main(
        [
            "--output",
            str(output),
            "--samples-per-class",
            "2",
            "--pairs-per-sample",
            "1",
            "--epochs",
            "1",
            "--model",
            "shuffled_cells",
            "--feature-mode",
            "cell_structured_shuffled",
            "--feature-cache-root",
            str(tmp_path / "cache"),
            "--feature-cache-chunk-size",
            "1",
            "--progress-output",
            str(tmp_path / "progress.jsonl"),
            "--device",
            "cpu",
        ]
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["route"] == "candidate_trail_consistency_shuffled_cells"
    assert row["model"] == "candidate_trail_consistency_shuffled_cells"
    assert row["training_model"] == "shuffled_cells"
    assert row["training_model_family"] == "mlp"
    assert row["feature_mode"] == "cell_structured_shuffled"
    assert row["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert row["metrics"]["auc"] == row["auc"]
    assert row["metrics"]["calibrated_accuracy"] == row["calibrated_accuracy"]


def test_candidate_evidence_matrix_outputs_anchor_and_candidate_rows(tmp_path):
    config = tmp_path / "candidate_matrix.json"
    output = tmp_path / "candidate_matrix.jsonl"
    config.write_text(
        json.dumps(
            {
                "output": str(output),
                "common": {
                    "rounds": 7,
                    "seed": 0,
                    "samples_per_class": 2,
                    "pairs_per_sample": 1,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "zhang_wang_case2_official_mcnd",
                    "difference_profile": "present_zhang_wang2022_mcnd",
                    "difference_member": 0,
                    "train_key": "0x00000000000000000000",
                    "validation_key": "0x11111111111111111111",
                    "key_rotation_interval": 0,
                    "beam_width": 2,
                    "depth": 2,
                    "feature_cache_chunk_size": 1,
                    "epochs": 1,
                    "learning_rate": 0.01,
                    "device": "cpu",
                },
                "rows": [
                    {
                        "row_type": "external_anchor",
                        "model": "present_nibble_invp_only_spn_only",
                        "anchor_auc": 0.52,
                        "anchor_calibrated_accuracy": 0.5,
                    },
                    {
                        "model": "linear",
                        "feature_cache_root": str(tmp_path / "linear_cache"),
                        "progress_output": str(tmp_path / "linear_progress.jsonl"),
                    },
                    {
                        "model": "shuffled_cells",
                        "feature_mode": "cell_structured_shuffled",
                        "feature_cache_root": str(tmp_path / "shuffled_cache"),
                        "progress_output": str(tmp_path / "shuffled_progress.jsonl"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    spn_candidate_evidence_matrix.main(["--config", str(config)])

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["model"] for row in rows] == [
        "present_nibble_invp_only_spn_only",
        "candidate_trail_consistency_linear",
        "candidate_trail_consistency_shuffled_cells",
    ]
    assert rows[0]["row_type"] == "external_anchor"
    assert rows[0]["metrics"]["auc"] == 0.52
    assert rows[1]["feature_mode"] == "cell_structured"
    assert rows[2]["feature_mode"] == "cell_structured_shuffled"

    report = validate_result_plan_alignment(config, output, expected_rows=3)
    assert report["status"] == "pass"


def test_candidate_evidence_matrix_can_feed_candidate_postprocess(tmp_path):
    config = tmp_path / "candidate_matrix_4row.json"
    output = tmp_path / "candidate_matrix_4row.jsonl"
    common = {
        "rounds": 7,
        "seed": 0,
        "samples_per_class": 2,
        "pairs_per_sample": 1,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "train_key": "0x00000000000000000000",
        "validation_key": "0x11111111111111111111",
        "key_rotation_interval": 0,
        "beam_width": 2,
        "depth": 2,
        "feature_cache_chunk_size": 1,
        "epochs": 1,
        "learning_rate": 0.01,
        "device": "cpu",
    }
    config.write_text(
        json.dumps(
            {
                "output": str(output),
                "common": common,
                "rows": [
                    {
                        "row_type": "external_anchor",
                        "model": "present_nibble_invp_only_spn_only",
                        "anchor_auc": 0.50,
                        "anchor_calibrated_accuracy": 0.5,
                    },
                    {
                        "model": "linear",
                        "feature_cache_root": str(tmp_path / "linear_cache"),
                        "progress_output": str(tmp_path / "linear_progress.jsonl"),
                    },
                    {
                        "model": "mlp",
                        "feature_cache_root": str(tmp_path / "mlp_cache"),
                        "progress_output": str(tmp_path / "mlp_progress.jsonl"),
                    },
                    {
                        "model": "shuffled_cells",
                        "feature_mode": "cell_structured_shuffled",
                        "feature_cache_root": str(tmp_path / "shuffled_cache"),
                        "progress_output": str(tmp_path / "shuffled_progress.jsonl"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    spn_candidate_evidence_matrix.main(["--config", str(config)])
    report = postprocess_candidate_trail_result(
        results_path=output,
        output_dir=tmp_path / "postprocess",
        run_id="candidate_matrix_4row",
        plan_path=config,
        expected_rows=4,
        plan_doc_paths=[],
    )

    assert report["status"] == "pass"
    assert report["validation_status"] == "pass"
    assert report["candidate_trail_status"] == "pass"
    assert report["decision"] in {
        "support_candidate_trail_route",
        "weak_candidate_trail_signal",
        "stop_candidate_trail_route",
    }


def _write_candidate_trail_result(path: Path, model: str, auc: float, calibrated: float = 0.72) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "model": model,
                    "metrics": {
                        "auc": auc,
                        "accuracy": calibrated,
                        "calibrated_accuracy": calibrated,
                        "loss": 0.55,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )


def test_candidate_trail_gate_supports_route_when_candidate_beats_anchor(tmp_path):
    results = tmp_path / "candidate_trail.jsonl"
    _write_candidate_trail_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(results, "candidate_trail_consistency_linear", 0.7925)
    _write_candidate_trail_result(results, "candidate_trail_consistency_mlp", 0.7940)
    _write_candidate_trail_result(results, "candidate_trail_consistency_shuffled_cells", 0.7926)

    report = gate_candidate_trail_result(results, expected_rows=4)

    assert report["status"] == "pass"
    assert report["best_candidate_model"] == "candidate_trail_consistency_mlp"
    assert report["decision"] == "support_candidate_trail_route"
    assert report["margin_vs_anchor_auc"] > 0.001
    assert report["margin_vs_shuffled_auc"] > 0.001
    assert "not paper-scale" in report["claim_scope"]


def test_candidate_trail_gate_marks_weak_or_stop_signal(tmp_path):
    weak_results = tmp_path / "candidate_trail_weak.jsonl"
    _write_candidate_trail_result(weak_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(weak_results, "candidate_trail_consistency_linear", 0.7922)
    _write_candidate_trail_result(weak_results, "candidate_trail_consistency_mlp", 0.7924)

    weak = gate_candidate_trail_result(weak_results, expected_rows=3)
    assert weak["decision"] == "weak_candidate_trail_signal"

    stop_results = tmp_path / "candidate_trail_stop.jsonl"
    _write_candidate_trail_result(stop_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(stop_results, "candidate_trail_consistency_linear", 0.7910)
    _write_candidate_trail_result(stop_results, "candidate_trail_consistency_mlp", 0.7915)

    stop = gate_candidate_trail_result(stop_results, expected_rows=3)
    assert stop["decision"] == "stop_candidate_trail_route"
    assert stop["margin_vs_anchor_auc"] < 0


def test_candidate_trail_postprocess_writes_summary_and_updates_plan_doc(tmp_path):
    results = tmp_path / "candidate_trail.jsonl"
    _write_candidate_trail_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(results, "candidate_trail_consistency_linear", 0.7925)
    _write_candidate_trail_result(results, "candidate_trail_consistency_mlp", 0.7940)
    _write_candidate_trail_result(results, "candidate_trail_consistency_shuffled_cells", 0.7926)
    plan_doc = tmp_path / "candidate_plan.md"
    plan_doc.write_text("# Candidate Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_candidate_trail_result(
        results_path=results,
        output_dir=output_dir,
        run_id="candidate_trail_unit",
        expected_rows=4,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "support_candidate_trail_route"
    assert report["next_action"]["branch"] == "candidate_trail_seed1_confirmation"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["next_plan_doc"] == "docs/experiments/innovation1-candidate-trail-consistency-plan.md"
    assert "candidate_trail_consistency_r7_262k_seed1" in report["next_action"]["suggested_plan_config"]
    assert report["next_action"]["launch_remote_config"].endswith(
        "innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.json"
    )
    assert report["next_action"]["run_id"] == "i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702"
    assert report["next_action"]["suggested_feature_cache_workers"] == 4
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    assert report["next_action"]["launch_remote_config"] in report["next_action"]["readiness_command"]
    assert any("seed1 confirmation" in step for step in report["next_steps"])
    assert (output_dir / "candidate_trail_unit_candidate_trail_gate.json").exists()
    assert (output_dir / "candidate_trail_unit_postprocess_summary.json").exists()
    assert (output_dir / "candidate_trail_unit_postprocess_summary.md").exists()
    readiness_path = output_dir / "candidate_trail_unit_next_action_readiness.json"
    assert readiness_path.exists()
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "candidate_trail_seed1_confirmation"
    assert readiness["should_launch_remote"] is True
    assert readiness["requires_implementation"] is False
    assert readiness["readiness_pass"] is True
    assert readiness["readiness_reports"][0]["readiness"]["status"] == "pass"
    assert readiness["errors"] == []
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "## Retrieved Candidate-Trail Consistency Result" in plan_text
    assert "<!-- candidate-trail-postprocess:candidate_trail_unit:start -->" in plan_text
    assert "| Decision | `support_candidate_trail_route` |" in plan_text
    assert "| Next action readiness | `" in plan_text

    postprocess_candidate_trail_result(
        results_path=results,
        output_dir=output_dir,
        run_id="candidate_trail_unit",
        expected_rows=4,
        plan_doc_paths=[plan_doc],
    )
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert plan_text.count("<!-- candidate-trail-postprocess:candidate_trail_unit:start -->") == 1


def test_candidate_trail_postprocess_stop_points_to_transition_spectrum_plan(tmp_path):
    results = tmp_path / "candidate_trail_stop.jsonl"
    _write_candidate_trail_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(results, "candidate_trail_consistency_linear", 0.7910)
    _write_candidate_trail_result(results, "candidate_trail_consistency_mlp", 0.7915)
    output_dir = tmp_path / "postprocess"

    report = postprocess_candidate_trail_result(
        results_path=results,
        output_dir=output_dir,
        run_id="candidate_trail_stop_unit",
        expected_rows=3,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "stop_candidate_trail_route"
    assert report["next_action"]["branch"] == "stop_candidate_trail_route"
    assert report["next_action"]["should_launch_remote"] is False
    assert report["next_action"]["fallback_branch"] == "bit_transition_spectrum_seed0"
    assert report["next_action"]["fallback_plan"] == "docs/experiments/innovation1-bit-transition-spectrum-plan.md"
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    assert any("bit-transition-spectrum" in step for step in report["next_steps"])
    assert "launch_remote_config" not in report["next_action"]
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["branch"] == "stop_candidate_trail_route"
    assert readiness["next_action"]["fallback_branch"] == "bit_transition_spectrum_seed0"
    assert readiness["requires_implementation"] is True
    assert readiness["implementation_checklist"]
    assert "bit_transition_spectrum_seed0" in " ".join(readiness["implementation_checklist"])
    assert "docs/experiments/innovation1-bit-transition-spectrum-plan.md" in " ".join(
        readiness["implementation_checklist"]
    )


def test_candidate_trail_postprocess_weak_signal_exposes_seed1_or_fallback_paths(tmp_path):
    results = tmp_path / "candidate_trail_weak.jsonl"
    _write_candidate_trail_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(results, "candidate_trail_consistency_linear", 0.7922)
    _write_candidate_trail_result(results, "candidate_trail_consistency_mlp", 0.7924)
    output_dir = tmp_path / "postprocess"

    report = postprocess_candidate_trail_result(
        results_path=results,
        output_dir=output_dir,
        run_id="candidate_trail_weak_unit",
        expected_rows=3,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "weak_candidate_trail_signal"
    assert report["next_action"]["branch"] == "candidate_trail_variance_check"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["fallback_plan"] == "docs/experiments/innovation1-bit-transition-spectrum-plan.md"
    assert "candidate_trail_consistency_r7_262k_seed1" in report["next_action"]["suggested_plan_config"]
    assert report["next_action"]["launch_remote_config"].endswith(
        "innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.json"
    )
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    assert any("seed1 variance check" in step for step in report["next_steps"])
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["readiness_pass"] is True


def _write_transition_spectrum_result(path: Path, model: str, auc: float, calibrated: float = 0.72) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "model": model,
                    "metrics": {
                        "auc": auc,
                        "accuracy": calibrated,
                        "calibrated_accuracy": calibrated,
                        "loss": 0.55,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )


def test_transition_spectrum_gate_supports_route_when_candidate_beats_anchor(tmp_path):
    results = tmp_path / "transition_spectrum.jsonl"
    _write_transition_spectrum_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_linear", 0.7925)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_mlp", 0.7940)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_shuffled_p", 0.7926)

    report = gate_transition_spectrum_result(results, expected_rows=4)

    assert report["status"] == "pass"
    assert report["best_candidate_model"] == "bit_transition_spectrum_mlp"
    assert report["decision"] == "support_transition_spectrum_route"
    assert report["margin_vs_anchor_auc"] > 0.001
    assert report["margin_vs_shuffled_auc"] > 0.001
    assert "not paper-scale" in report["claim_scope"]


def test_transition_spectrum_gate_marks_weak_or_stop_signal(tmp_path):
    weak_results = tmp_path / "transition_spectrum_weak.jsonl"
    _write_transition_spectrum_result(weak_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(weak_results, "bit_transition_spectrum_linear", 0.7922)
    _write_transition_spectrum_result(weak_results, "bit_transition_spectrum_mlp", 0.7924)

    weak = gate_transition_spectrum_result(weak_results, expected_rows=3)
    assert weak["decision"] == "weak_transition_spectrum_signal"

    stop_results = tmp_path / "transition_spectrum_stop.jsonl"
    _write_transition_spectrum_result(stop_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(stop_results, "bit_transition_spectrum_linear", 0.7910)
    _write_transition_spectrum_result(stop_results, "bit_transition_spectrum_mlp", 0.7915)

    stop = gate_transition_spectrum_result(stop_results, expected_rows=3)
    assert stop["decision"] == "stop_transition_spectrum_route"
    assert stop["margin_vs_anchor_auc"] < 0


def test_transition_spectrum_postprocess_writes_summary_and_updates_plan_doc(tmp_path):
    results = tmp_path / "transition_spectrum.jsonl"
    _write_transition_spectrum_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_linear", 0.7925)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_mlp", 0.7940)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_shuffled_p", 0.7926)
    plan_doc = tmp_path / "transition_plan.md"
    plan_doc.write_text("# Transition Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_transition_spectrum_result(
        results_path=results,
        output_dir=output_dir,
        run_id="transition_spectrum_unit",
        expected_rows=4,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "support_transition_spectrum_route"
    assert report["next_action"]["branch"] == "transition_spectrum_seed1_confirmation"
    assert report["next_action"]["should_launch_remote"] is False
    assert report["next_action"]["requires_implementation"] is True
    assert report["next_action"]["next_plan_doc"] == "docs/experiments/innovation1-bit-transition-spectrum-plan.md"
    assert "bit_transition_spectrum_r7_262k_seed1" in report["next_action"]["suggested_plan_config"]
    assert report["next_action"]["suggested_feature_cache_workers"] == 4
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    assert any("seed1 plan/config" in step for step in report["next_steps"])
    assert any("feature_cache_workers" in step and "at least 4" in step for step in report["next_steps"])
    assert (output_dir / "transition_spectrum_unit_transition_spectrum_gate.json").exists()
    assert (output_dir / "transition_spectrum_unit_postprocess_summary.json").exists()
    assert (output_dir / "transition_spectrum_unit_postprocess_summary.md").exists()
    readiness_path = output_dir / "transition_spectrum_unit_next_action_readiness.json"
    assert readiness_path.exists()
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "transition_spectrum_seed1_confirmation"
    assert readiness["should_launch_remote"] is False
    assert readiness["requires_implementation"] is True
    assert readiness["readiness_pass"] is False
    assert readiness["implementation_checklist"]
    assert "transition_spectrum_seed1_confirmation" in readiness["implementation_checklist"][0]
    assert "docs/experiments/innovation1-bit-transition-spectrum-plan.md" in " ".join(
        readiness["implementation_checklist"]
    )
    assert readiness["errors"] == []
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "## Retrieved Bit-Transition-Spectrum Result" in plan_text
    assert "<!-- transition-spectrum-postprocess:transition_spectrum_unit:start -->" in plan_text
    assert "| Decision | `support_transition_spectrum_route` |" in plan_text
    assert "| Next action readiness | `" in plan_text

    postprocess_transition_spectrum_result(
        results_path=results,
        output_dir=output_dir,
        run_id="transition_spectrum_unit",
        expected_rows=4,
        plan_doc_paths=[plan_doc],
    )
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert plan_text.count("<!-- transition-spectrum-postprocess:transition_spectrum_unit:start -->") == 1


def test_transition_spectrum_postprocess_weak_and_stop_expose_next_paths(tmp_path):
    weak_results = tmp_path / "transition_spectrum_weak.jsonl"
    _write_transition_spectrum_result(weak_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(weak_results, "bit_transition_spectrum_linear", 0.7922)
    _write_transition_spectrum_result(weak_results, "bit_transition_spectrum_mlp", 0.7924)

    weak = postprocess_transition_spectrum_result(
        results_path=weak_results,
        output_dir=tmp_path / "weak_postprocess",
        run_id="transition_spectrum_weak_unit",
        expected_rows=3,
    )

    assert weak["decision"] == "weak_transition_spectrum_signal"
    assert weak["next_action"]["branch"] == "transition_spectrum_variance_check"
    assert "bit_transition_spectrum_r7_262k_seed1" in weak["next_action"]["suggested_plan_config"]
    assert weak["next_action"]["suggested_feature_cache_workers"] == 4
    assert "scripts/check-remote-readiness" in weak["next_action"]["readiness_command"]
    assert any("seed1 plan/config" in step for step in weak["next_steps"])
    assert Path(weak["next_action_readiness"]).exists()

    stop_results = tmp_path / "transition_spectrum_stop.jsonl"
    _write_transition_spectrum_result(stop_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(stop_results, "bit_transition_spectrum_linear", 0.7910)
    _write_transition_spectrum_result(stop_results, "bit_transition_spectrum_mlp", 0.7915)

    stop = postprocess_transition_spectrum_result(
        results_path=stop_results,
        output_dir=tmp_path / "stop_postprocess",
        run_id="transition_spectrum_stop_unit",
        expected_rows=3,
    )

    assert stop["decision"] == "stop_transition_spectrum_route"
    assert stop["next_action"]["branch"] == "stop_transition_spectrum_route"
    assert "trail_family_consistency" in stop["next_action"]["fallback_hypotheses"]
    assert "docs/experiments/innovation1-trail-family-consistency-plan.md" in stop["next_action"]["fallback_plan_options"]
    assert "docs/research/spn_structured_nn_research_plan.md" in stop["next_action"]["fallback_plan_options"]
    assert any("new docs/experiments plan" in step for step in stop["next_steps"])
    stop_readiness = json.loads(Path(stop["next_action_readiness"]).read_text(encoding="utf-8"))
    assert stop_readiness["branch"] == "stop_transition_spectrum_route"
    assert stop_readiness["requires_implementation"] is True
    assert stop_readiness["implementation_checklist"]
    assert "innovation1-trail-family-consistency-plan.md" in " ".join(
        stop_readiness["implementation_checklist"]
    )


def test_summarize_spn_evidence_transition_stop_points_to_trail_family_plan(tmp_path):
    root = tmp_path / "remote_results"
    transition = root / "i1_bit_transition_spectrum_r7_262k_seed0_gpu0_20260702"
    transition.mkdir(parents=True)
    _write_test_json(
        transition / "i1_bit_transition_spectrum_r7_262k_seed0_gpu0_20260702_postprocess_summary.json",
        {
            "run_id": "i1_bit_transition_spectrum_r7_262k_seed0_gpu0_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_transition_spectrum_route",
            "claim_scope": "bit-transition-spectrum medium diagnostic gate",
            "next_action": {
                "branch": "stop_transition_spectrum_route",
                "should_launch_remote": False,
                "requires_implementation": True,
            },
        },
    )

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "new_spn_hypothesis_plan"
    assert active["next_action"]["next_plan_doc"] == (
        "docs/experiments/innovation1-trail-family-consistency-plan.md"
    )
    assert "docs/experiments/innovation1-trail-family-consistency-plan.md" in active["next_action"][
        "fallback_plan_options"
    ]


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


def test_present_nibble_ddt_graph_features_match_present_sbox_ddt():
    from blockcipher_nd.features.encoders.bitwise import pair_to_bits
    from blockcipher_nd.features.encoders.present_sbox_ddt import PRESENT_SBOX_DDT
    from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
        PresentNibbleDDTGraphDistinguisher,
        PresentNibbleNoDDTGraphDistinguisher,
    )

    cipher = build_cipher("present80", rounds=7, key=0)
    left = 0x0123456789ABCDEF
    right = 0x1111111111111111
    raw = torch.tensor([pair_to_bits(left, right, 64)], dtype=torch.float32)
    model = PresentNibbleDDTGraphDistinguisher(
        input_bits=128,
        pair_bits=128,
        base_channels=4,
        ddt_mixer_depth=1,
    )
    cell_features = model.ddt_encoder.ddt_cell_features(raw)
    no_ddt_model = PresentNibbleNoDDTGraphDistinguisher(
        input_bits=128,
        pair_bits=128,
        base_channels=4,
        ddt_mixer_depth=1,
    )
    no_ddt_features = no_ddt_model.ddt_encoder.ddt_cell_features(raw)
    delta = left ^ right
    aligned = cipher.inverse_permutation_layer(delta)
    first_output_difference = (aligned >> 60) & 0xF
    expected_output_bits = [(first_output_difference >> shift) & 1 for shift in range(3, -1, -1)]
    expected_counts = [
        PRESENT_SBOX_DDT[input_difference][first_output_difference] / 16
        for input_difference in range(16)
    ]

    assert model.ddt_encoder.ddt_by_output.shape == (16, 16)
    assert cell_features.shape == (1, 1, 16, 20)
    assert no_ddt_features.shape == (1, 1, 16, 4)
    assert cell_features[0, 0, 0, :4].to(torch.uint8).tolist() == expected_output_bits
    assert no_ddt_features[0, 0, 0].to(torch.uint8).tolist() == expected_output_bits
    assert cell_features[0, 0, 0, 4:].tolist() == expected_counts


def test_present_nibble_invp_p_layer_graph_models_build_and_use_distinct_topologies():
    common = {
        "input_bits": 2048,
        "hidden_bits": 4,
        "pair_bits": 128,
        "structure": "SPN",
        "model_options": {
            "graph_mixer_depth": 1,
            "activation": "relu",
            "norm": "layernorm",
            "pooling": "topk_logsumexp",
            "top_k": 2,
        },
    }
    candidate = build_model("present_nibble_invp_p_layer_graph_spn_only", **common)
    control = build_model("present_nibble_invp_shuffled_p_layer_graph_spn_only", **common)
    features = torch.randint(0, 2, (2, 2048), dtype=torch.float32)

    candidate_logits = candidate(features)
    control_logits = control(features)
    candidate_sources = candidate.graph_encoder.mixers[0].p_sources
    control_sources = control.graph_encoder.mixers[0].p_sources
    candidate_cells = candidate.graph_encoder.invp_nibbles(features)
    control_cells = control.graph_encoder.invp_nibbles(features)

    assert candidate_logits.shape == (2, 1)
    assert control_logits.shape == (2, 1)
    assert candidate_sources.shape == control_sources.shape
    assert not torch.equal(candidate_sources, control_sources)
    assert torch.equal(candidate_cells, control_cells)


def test_evidence_pooling_topk_logsumexp_casts_scatter_weights_under_autocast():
    from blockcipher_nd.models.common.components import EvidencePooling

    pooling = EvidencePooling(
        embedding_bits=8,
        hidden_bits=4,
        mode="topk_logsumexp",
        top_k=2,
        activation="relu",
        norm="layernorm",
    )
    embeddings = torch.randn(2, 4, 8, dtype=torch.float32)

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        pooled, weights = pooling(embeddings)

    assert pooled.shape == (2, 8)
    assert weights.shape == (2, 4)
    assert weights.dtype in {torch.float32, torch.bfloat16}
    assert torch.allclose(weights.sum(dim=1).float(), torch.ones(2), atol=1e-5)


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
        "present_nibble_no_ddt_graph",
        "present_nibble_ddt_graph",
        "present_nibble_shuffled_ddt_graph",
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
                "ddt_mixer_depth": 1,
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
