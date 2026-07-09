from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from blockcipher_nd.engine.matrix_runner import parse_args
from blockcipher_nd.data.cache import make_chunked_differential_dataset
from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.engine.datasets import dataset_cache_dir
from blockcipher_nd.engine.modeling import infer_pair_bits
from blockcipher_nd.engine.task_config import build_dataset_config
from blockcipher_nd.features.registry import (
    encode_ciphertext_pair,
    is_supported_feature_encoding,
    pair_bits_for_encoding,
)
from blockcipher_nd.planning.invp_gate import gate_invp_only_result
from blockcipher_nd.planning.invp_attribution_gate import gate_invp_attribution_controls
from blockcipher_nd.planning.ddt_graph_gate import gate_ddt_graph_result
from blockcipher_nd.planning.topology_aware_gate import gate_topology_aware_result
from blockcipher_nd.planning.candidate_trail_gate import gate_candidate_trail_result
from blockcipher_nd.planning.transition_spectrum_gate import gate_transition_spectrum_result
from blockcipher_nd.planning.trail_family_gate import gate_trail_family_result
from blockcipher_nd.planning.active_auxiliary_gate import gate_active_auxiliary_result
from blockcipher_nd.planning.trail_position_residual_gate import gate_trail_position_residual
from blockcipher_nd.planning.sbox_prior_gate import gate_sbox_prior_result
from blockcipher_nd.planning.difference_screen_gate import gate_difference_screen_result
from blockcipher_nd.planning.candidate_trail_postprocess import postprocess_candidate_trail_result
from blockcipher_nd.planning.transition_spectrum_postprocess import postprocess_transition_spectrum_result
from blockcipher_nd.planning.trail_family_postprocess import postprocess_trail_family_result
from blockcipher_nd.planning.active_auxiliary_postprocess import postprocess_active_auxiliary_result
from blockcipher_nd.planning.sbox_prior_postprocess import postprocess_sbox_prior_result
from blockcipher_nd.planning.difference_confirmation_plan import create_difference_confirmation_plan
from blockcipher_nd.planning.difference_screen_postprocess import postprocess_difference_screen_result
from blockcipher_nd.planning.pair_mixer_postprocess import postprocess_pair_mixer_consistency_result
from blockcipher_nd.planning.pair_evidence_pooling_postprocess import postprocess_pair_evidence_pooling_result
from blockcipher_nd.planning.integral_inverse_feature_postprocess import (
    postprocess_integral_inverse_feature_result,
)
from blockcipher_nd.planning.r9_weak_probe_postprocess import postprocess_r9_weak_probe_result
from blockcipher_nd.planning.r8_pairset_1m_postprocess import postprocess_r8_pairset_1m_result
from blockcipher_nd.cli.monitor_health import monitor_health_report
from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.planning.next_action_readiness import launch_artifacts
from blockcipher_nd.planning.invp_postprocess import postprocess_invp_only_result
from blockcipher_nd.planning.invp_attribution_postprocess import postprocess_invp_attribution_controls
from blockcipher_nd.planning.ddt_graph_postprocess import postprocess_ddt_graph_result
from blockcipher_nd.planning.topology_aware_postprocess import postprocess_topology_aware_result
from blockcipher_nd.cli.plan_next_action import plan_next_action
from blockcipher_nd.cli.arbitrate_next_actions import arbitrate_next_actions
from blockcipher_nd.cli.advance_high_round import advance_high_round
from blockcipher_nd.cli.watch_high_round import watch_high_round
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
from blockcipher_nd.tasks.innovation1.spn_trail_family import make_trail_family_dataset
from blockcipher_nd.tasks.innovation1.spn_active_auxiliary import (
    _evaluate_active_aux_model,
    _train_active_aux_model,
    make_active_auxiliary_dataset,
)
from blockcipher_nd.cli import (
    spn_candidate_evidence_matrix,
    spn_transition_spectrum_matrix,
    spn_trail_family_matrix,
    spn_active_auxiliary_matrix,
)
from blockcipher_nd.cli.audit_integral_parity_signal import (
    integral_composite_residual_audit_from_task,
    integral_deterministic_baseline_from_task,
    integral_alignment_audit_from_task,
    integral_feature_bank_audit_from_task,
    integral_parity_audit_from_task,
)
from blockcipher_nd.cli.audit_spn_features import main as audit_spn_features_main
from blockcipher_nd.cli.summarize_spn_evidence import summarize_spn_evidence
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact


def test_matrix_runner_lives_in_engine_package():
    args = parse_args(
        ["--ciphers", "speck32", "--models", "mlp", "--rounds", "1", "--dataset-cache-workers", "2"]
    )

    assert args.ciphers == ["speck32"]
    assert args.models == ["mlp"]
    assert args.rounds == [1]
    assert args.learning_rate == 1e-3
    assert args.dataset_cache_workers == 2


def test_matrix_runner_accepts_difference_matched_integral_sample_structure():
    args = parse_args(
        [
            "--sample-structure",
            "plaintext_integral_nibble_difference_matched_negative",
        ]
    )

    assert args.sample_structure == "plaintext_integral_nibble_difference_matched_negative"


def test_matrix_runner_accepts_strict_random_integral_negative_sample_structure():
    args = parse_args(
        [
            "--sample-structure",
            "plaintext_integral_nibble_strict_random_negative",
        ]
    )

    assert args.sample_structure == "plaintext_integral_nibble_strict_random_negative"


def test_matrix_runner_accepts_same_difference_random_integral_negative_sample_structure():
    args = parse_args(
        [
            "--sample-structure",
            "plaintext_integral_nibble_same_difference_random_negative",
        ]
    )

    assert args.sample_structure == "plaintext_integral_nibble_same_difference_random_negative"


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


def test_bit_sensitivity_projection_plan_is_conditional_and_controlled():
    research = Path("docs/research/innovation1-spn-bit-sensitivity-nonneighbor-route-20260707.md")
    plan = Path("docs/experiments/innovation1-present-r8-bit-sensitivity-projection-expert-plan.md")

    research_text = research.read_text(encoding="utf-8")
    plan_text = plan.read_text(encoding="utf-8")
    combined = research_text + "\n" + plan_text

    assert "wait_for_trail_position_262k_results" in combined
    assert "negative_mode = encrypted_random_plaintexts" in combined
    assert "same-input global control" in combined
    assert "do_not_launch_remote_now" in combined
    assert "do_not_claim_formal_spn_present_evidence" in combined
    assert "selection_split = train" in combined
    assert "validation_key = 0x11111111111111111111" in combined
    assert "SPN compressed evidence pooling" in combined
    assert "Do not run a validation hyperparameter sweep" in combined
    assert "train-fitted aggregation beats the best single validation AUC" in combined
    assert "train_holdout_fraction = 0.25" in combined
    assert "mixed_train_holdout_stacking_diagnostic" in combined
    assert "stable_but_mixed_train_holdout_stacking_diagnostic" in combined
    assert "positive_selection_seeds = 5 / 5" in combined
    assert "scripts/summarize-stacked-selection" in combined
    assert "scripts/summarize-stacked-route" in combined
    assert "stable_but_mixed_cross_seed_stacking_diagnostic" in combined
    assert "scripts/fit-compressed-feature-expert" in combined
    assert "scripts/summarize-compressed-feature-expert" in combined
    assert "scripts/audit-compressed-feature-sparsity" in combined
    assert "scripts/decode-compressed-feature-sparsity" in combined
    assert "compressed_feature_expert_local_screen_positive_needs_controls" in combined
    assert "compressed_feature_expert_shuffle_train_labels_control" in combined
    assert "compressed_feature_local_positive_controls_pass_not_ensemble_gain" in combined
    assert "sparse_compressed_feature_local_screen_positive" in combined
    assert "depth_word_cell_span" in combined
    assert "span-type structural statistics" in combined
    assert "span_family_filter" in combined
    assert "selected_feature_count = 731 / 3708" in combined
    assert "scripts/audit-compressed-feature-families" in combined
    assert "scripts/export-compressed-span-blocks" in combined
    assert "scripts/summarize-compressed-span-route" in combined
    assert "compressed_spn_span_blocks" in combined
    assert "compressed_span_summary" in combined
    assert "compressed_span_summary_retains_flat_signal_controls_pass" in combined
    assert "compressed_span_summary_logistic_expert" in combined
    assert "feature_reduction_ratio = 0.3734610123119015" in combined
    assert "--include-feature-prefix" in combined
    assert "primary_feature_count = 133" in combined
    assert "auxiliary_feature_count = 140" in combined
    assert "seed0 primary_validation_auc = 0.9997234344482422" in combined
    assert "seed1 auxiliary_validation_auc = 0.9976606369018555" in combined
    assert "primary_backbone" in combined
    assert "depth_word_cell_span is the dominant span-family backbone" in combined
    assert "leave_out_depth_word_cell_span_auc" in combined
    assert "scripts/fit-compressed-span-grouped-expert" in combined
    assert "compressed_span_grouped_expert_local_screen_positive_needs_controls" in combined
    assert "two_branch_logistic" in combined
    assert "seed0 grouped_validation_auc = 0.9997968673706055" in combined
    assert "seed1 grouped_validation_auc = 0.9996414184570312" in combined
    assert "semantic_group_logistic" in combined
    assert "hybrid_group_logistic" in combined
    assert "seed0 semantic_group_validation_auc = 0.998713493347168" in combined
    assert "seed1 semantic_l2zero_validation_auc = 0.9988164901733398" in combined
    assert "seed0 hybrid_group_validation_auc = 0.9992799758911133" in combined
    assert "semantic_or_hybrid_branch_logit_decomposition_hold" in combined
    assert "scripts/fit-compressed-span-interaction-expert" in combined
    assert "raw_plus_primary_auxiliary_interactions_logistic" in combined
    assert "seed0 interaction_validation_auc = 0.9999170303344727" in combined
    assert "seed1 interaction_validation_auc = 0.9998636245727539" in combined
    assert "seed0 interaction_shuffle_validation_auc = 0.5185546875" in combined
    assert "raw_interaction_summary_tiny_positive_controls_pass_local" in combined
    assert "scripts/fit-compressed-span-block-interaction-expert" in combined
    assert "raw_plus_semantic_block_interactions_logistic" in combined
    assert "seed0 block_interaction_validation_auc = 0.9999065399169922" in combined
    assert "seed1 block_interaction_validation_auc = 0.999908447265625" in combined
    assert "semantic_block_interaction_mixed_local_diagnostic" in combined
    assert "scripts/fit-compressed-span-low-rank-interaction-expert" in combined
    assert "raw_plus_semantic_low_rank_block_interactions_logistic" in combined
    assert "seed0 rank1_low_rank_validation_auc = 0.9999256134033203" in combined
    assert "seed1 rank1_low_rank_validation_auc = 0.9999008178710938" in combined
    assert "semantic_low_rank_rank1_positive_vs_full_mixed_vs_blockstat_local" in combined
    assert "scripts/fit-compressed-span-learned-low-rank-interaction-expert" in combined
    assert "raw_plus_learned_semantic_low_rank_block_interactions" in combined
    assert "seed0 learned_low_rank_validation_auc = 0.9998531341552734" in combined
    assert "seed1 learned_low_rank_validation_auc = 0.9995737075805664" in combined
    assert "learned_low_rank_rank1_hold_local_diagnostic" in combined
    assert "seed0 svd_frozen_validation_auc = 0.9999265670776367" in combined
    assert "seed1 svd_frozen_validation_auc = 0.9999094009399414" in combined
    assert "seed1 svd_frozen_shuffle_validation_auc = 0.6085009574890137" in combined
    assert "svd_frozen_learned_rank1_recovers_auc_but_fails_seed1_shuffle_control" in combined
    assert "semantic_low_rank_block_interactions_only_logistic" in combined
    assert "seed0 interaction_only_validation_auc = 0.5190114974975586" in combined
    assert "seed1 interaction_only_validation_auc = 0.5553302764892578" in combined
    assert "interaction_only_low_rank_weak_not_primary_signal_source" in combined
    assert "strongest_single_family = primary_depth_trailword" in combined
    assert "seed0 compact_combo_auc = 0.9999017715454102" in combined
    assert "seed1 compact_combo_auc = 0.9998178482055664" in combined
    assert "compact_raw_primary_depth_trailword_aux_depth_cell_anchor_local" in combined
    assert "not a multi-network improvement" in combined


def test_present_r8_round_extension_plan_preserves_strict_protocol():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_round_extension_r8_262k_seed0.csv"
    )
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 8
        assert task["seed"] == 0
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["train_key"] == 0
        assert task["validation_key"] == int("11111111111111111111", 16)
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM 262144/class r8 round-extension diagnostic" in task["matching_evidence"]


def test_present_r8_round_extension_remote_readiness_assets_pass():
    config = Path(
        "configs/remote/"
        "innovation1_spn_present_round_extension_r8_262k_seed0_gpu0_20260704.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 3
    assert readiness["plan_rows"] == 3
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert "cmd.exe /k" not in config.read_text(encoding="utf-8")


def test_present_r9_weak_probe_configs_are_conditional_and_strict():
    smoke_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_round_extension_r9_weak_probe_smoke.csv"
    )
    medium_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_round_extension_r9_262k_seed0.csv"
    )

    smoke_tasks = build_tasks(parse_args(["--plan", smoke_plan]))
    medium_tasks = build_tasks(parse_args(["--plan", medium_plan]))

    assert [task["model_key"] for task in smoke_tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    assert [task["model_key"] for task in medium_tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
    ]

    for task in smoke_tasks:
        assert task["rounds"] == 9
        assert task["samples_per_class"] == 8
        assert task["pairs_per_sample"] == 16
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "SMOKE only r9" in task["matching_evidence"]

    for task in medium_tasks:
        assert task["rounds"] == 9
        assert task["seed"] == 0
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "conditional launch only after r8 gate" in task["matching_evidence"]


def test_present_r9_weak_probe_remote_readiness_assets_pass():
    config = Path(
        "configs/remote/"
        "innovation1_spn_present_r9_weak_probe_262k_seed0_gpu0_20260705.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 3
    assert readiness["plan_rows"] == 3
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert "cmd.exe /k" not in config.read_text(encoding="utf-8")


def test_present_r9_weak_probe_seed1_confirmation_assets_are_gate_locked():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_round_extension_r9_262k_seed1.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 9
        assert task["seed"] == 1
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "seed1 variance check" in task["matching_evidence"]
        assert "launch only if seed0 r9 weak-probe gate is positive" in task["matching_evidence"]

    config = Path(
        "configs/remote/"
        "innovation1_spn_present_r9_weak_probe_262k_seed1_gpu0_20260705.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_text = config.read_text(encoding="utf-8")
    launcher_text = Path(
        "configs/remote/generated/"
        "run_i1_present_r9_weak_probe_262k_seed1_gpu0_20260705.cmd"
    ).read_text(encoding="utf-8")
    monitor_text = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r9_weak_probe_262k_seed1_gpu0_20260705.sh"
    ).read_text(encoding="utf-8")
    plan_doc = Path("docs/experiments/innovation1-present-r9-weak-probe-plan.md").read_text(
        encoding="utf-8"
    )

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 3
    assert readiness["plan_rows"] == 3
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert "cmd.exe /c" in config_text
    assert "cmd.exe /k" not in config_text
    assert "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705" in config_data["launch_policy"]
    assert "r9 weak-probe gate returns" in config_data["launch_policy"]
    assert config_data["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert config_data["dataset_cache_workers"] == 4

    assert "cmd.exe /k" not in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "innovation1_spn_present_round_extension_r9_262k_seed1.csv" in launcher_text
    assert "--dataset-cache-workers 4" in launcher_text
    assert "r9_weak_probe_seed1_progress.jsonl" in launcher_text

    assert "cmd.exe /k" not in monitor_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "scripts/postprocess-r9-weak-probe" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text

    assert "i1_present_r9_weak_probe_262k_seed1_gpu0_20260705" in plan_doc
    assert "262144/class seed1 = medium diagnostic variance check only" in plan_doc
    assert "do not launch until seed0 is retrieved / validated / postprocessed" in plan_doc


def test_present_r9_1m_seed0_assets_are_strong_gate_locked():
    plan = "configs/experiment/innovation1/innovation1_spn_present_r9_1m_seed0.csv"
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 9
        assert task["seed"] == 0
        assert task["samples_per_class"] == 1_000_000
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "1000000/class" in task["matching_evidence"]
        assert "launch only if r9 262144/class weak-probe gate is strong positive" in task["matching_evidence"]

    config = Path("configs/remote/innovation1_spn_present_r9_1m_seed0_gpu0_20260705.json")
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_text = config.read_text(encoding="utf-8")
    launcher_text = Path(
        "configs/remote/generated/run_i1_present_r9_1m_seed0_gpu0_20260705.cmd"
    ).read_text(encoding="utf-8")
    monitor_text = Path(
        "configs/remote/generated/monitor_i1_present_r9_1m_seed0_gpu0_20260705.sh"
    ).read_text(encoding="utf-8")
    plan_doc = Path("docs/experiments/innovation1-present-r9-weak-probe-plan.md").read_text(
        encoding="utf-8"
    )

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 3
    assert readiness["plan_rows"] == 3
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert "cmd.exe /c" in config_text
    assert "cmd.exe /k" not in config_text
    assert "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705" in config_data["launch_policy"]
    assert "strong_r9_diagnostic_prepare_1m_seed0" in config_data["launch_policy"]
    assert config_data["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert config_data["dataset_cache_workers"] == 4

    assert "cmd.exe /k" not in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "innovation1_spn_present_r9_1m_seed0.csv" in launcher_text
    assert "--dataset-cache-workers 4" in launcher_text
    assert "r9_1m_progress.jsonl" in launcher_text

    assert "cmd.exe /k" not in monitor_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "scripts/postprocess-r9-weak-probe" in monitor_text
    assert "--claim-scope \"${CLAIM_SCOPE}\"" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text

    assert "i1_present_r9_1m_seed0_gpu0_20260705" in plan_doc
    assert "strong_r9_diagnostic_prepare_1m_seed0" in plan_doc
    assert "1000000/class seed0 = paper-scale single-seed diagnostic" in plan_doc


def test_present_r9_curriculum_from_r8_plan_and_remote_assets_pass():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r9_curriculum_from_r8_262k_seed0.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 9
        assert task["seed"] == 0
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert task["pretrain_rounds"] == 8
        assert task["pretrain_epochs"] == 8
        assert "r9 curriculum diagnostic" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]

    config = Path(
        "configs/remote/"
        "innovation1_spn_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_text = config.read_text(encoding="utf-8")
    launcher_text = Path(
        "configs/remote/generated/"
        "run_i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.cmd"
    ).read_text(encoding="utf-8")
    monitor_text = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.sh"
    ).read_text(encoding="utf-8")
    plan_doc = Path(
        "docs/experiments/innovation1-present-r9-curriculum-from-r8-plan.md"
    ).read_text(encoding="utf-8")

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 2
    assert readiness["plan_rows"] == 2
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert "cmd.exe /c" in config_text
    assert "cmd.exe /k" not in config_text
    assert config_data["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705" in config_text
    assert "pretrain_rounds" in config_text
    assert "pretrain_epochs" in config_text

    assert "cmd.exe /k" not in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "C:\\Users" not in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "innovation1_spn_present_r9_curriculum_from_r8_262k_seed0.csv" in launcher_text
    assert "--epochs 22" in launcher_text
    assert "--negative-mode encrypted_random_plaintexts" in launcher_text
    assert "--sample-structure zhang_wang_case2_official_mcnd" in launcher_text
    assert "--key-rotation-interval 0" in launcher_text
    assert "r9_curriculum_progress.jsonl" in launcher_text

    assert "cmd.exe /k" not in monitor_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "validate-results" in monitor_text
    assert "plot-results" in monitor_text
    assert "gate_note" in monitor_text

    assert "8 pretrain epochs on r8 + 22 target epochs on r9" in plan_doc
    assert "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705" in plan_doc
    assert "retrieved / validated / plotted / gate-noted / plan-aligned" in plan_doc


def test_present_r9_difference_screen_plan_and_remote_assets_are_prepared_not_launched():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r9_difference_screen_65k_seed0.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 7
    assert {task["model_key"] for task in tasks} == {
        "present_nibble_invp_pair_consistency_spn_only"
    }
    assert [task["difference_profile"] for task in tasks] == [
        "present_zhang_wang2022_mcnd",
        "present_wang_jain2021",
        "present_wang_jain2021",
        "present_wang_jain2021",
        "present_wang_jain2021",
        "present_autond_dbitnet2023_highround",
        "present_entropy2026_gohr",
    ]
    assert [task["difference_member"] for task in tasks] == [0, 0, 1, 2, 3, 0, 0]
    assert [task["input_difference"] for task in tasks] == [
        0x0000000000000009,
        0x0700000000000700,
        0x7000000000007000,
        0x0070000000000070,
        0x0007000000000007,
        0x000000000D000000,
        0x0000000000D00000,
    ]

    for task in tasks:
        assert task["rounds"] == 9
        assert task["seed"] == 0
        assert task["samples_per_class"] == 65536
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "changes data construction only" in task["matching_evidence"]
        assert "not formal evidence" in task["matching_evidence"]

    config = Path(
        "configs/remote/"
        "innovation1_spn_present_r9_difference_screen_65k_seed0_gpu0_20260705.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_text = config.read_text(encoding="utf-8")
    launcher_text = Path(
        "configs/remote/generated/"
        "run_i1_present_r9_difference_screen_65k_seed0_gpu0_20260705.cmd"
    ).read_text(encoding="utf-8")
    monitor_text = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r9_difference_screen_65k_seed0_gpu0_20260705.sh"
    ).read_text(encoding="utf-8")
    plan_doc = Path(
        "docs/experiments/innovation1-present-r9-difference-screen-plan.md"
    ).read_text(encoding="utf-8")

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 7
    assert readiness["plan_rows"] == 7
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert config_data["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert "cmd.exe /c" in config_text
    assert "cmd.exe /k" not in config_text
    assert "prepared but do not launch" in config_data["launch_policy"]
    assert "not same-protocol model-improvement evidence" in config_data["claim_scope"]

    assert "cmd.exe /k" not in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "innovation1_spn_present_r9_difference_screen_65k_seed0.csv" in launcher_text
    assert "--epochs 20" in launcher_text
    assert "--dataset-cache-workers 4" in launcher_text
    assert "r9_difference_screen_progress.jsonl" in launcher_text

    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "scripts/postprocess-difference-screen" in monitor_text
    assert "--update-plan-doc" in monitor_text
    assert "innovation1-present-r9-difference-screen-plan.md" in monitor_text
    assert "plot-results" in monitor_text
    assert "plan_doc_committed_and_pushed" in monitor_text

    assert "数据构造 / benchmark 搜索" in plan_doc
    assert "not same-protocol model-improvement evidence" in plan_doc
    assert "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705" in plan_doc
    assert "i1_present_r8_pairset_1m_seed0_gpu1_20260705" in plan_doc


def test_present_pair_mixer_consistency_smoke_plan_is_protocol_locked():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pair_mixer_consistency_smoke.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_pair_consistency_spn_only",
        "present_nibble_invp_pair_mixer_consistency_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 8
        assert task["seed"] == 0
        assert task["samples_per_class"] == 8
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "SMOKE only" in task["matching_evidence"]
        assert "not accuracy evidence" in task["matching_evidence"]
    assert tasks[1]["model_options"]["pair_mixer_depth"] == 1

    plan_doc = Path(
        "docs/experiments/innovation1-present-pair-mixer-consistency-plan.md"
    ).read_text(encoding="utf-8")
    assert "present_nibble_invp_pair_mixer_consistency_spn_only" in plan_doc
    assert "cross-pair mixer" in plan_doc
    assert "不启动远程" in plan_doc


def test_present_pair_mixer_consistency_r8_262k_assets_are_prepared_not_launched():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pair_mixer_consistency_r8_262k_seed0.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_pair_consistency_spn_only",
        "present_nibble_invp_pair_mixer_consistency_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 8
        assert task["seed"] == 0
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM_DIAGNOSTIC 262144/class" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]
    assert tasks[1]["model_options"]["pair_mixer_depth"] == 1

    config = Path(
        "configs/remote/"
        "innovation1_spn_present_pair_mixer_consistency_r8_262k_seed0_gpu0_20260705.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_text = config.read_text(encoding="utf-8")
    launcher_text = Path(
        "configs/remote/generated/"
        "run_i1_present_r8_pair_mixer_consistency_262k_seed0_gpu0_20260705.cmd"
    ).read_text(encoding="utf-8")
    monitor_text = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r8_pair_mixer_consistency_262k_seed0_gpu0_20260705.sh"
    ).read_text(encoding="utf-8")
    plan_doc = Path(
        "docs/experiments/innovation1-present-pair-mixer-consistency-plan.md"
    ).read_text(encoding="utf-8")

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 2
    assert readiness["plan_rows"] == 2
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert config_data["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert config_data["dataset_cache_workers"] == 4
    assert "cmd.exe /c" in config_text
    assert "cmd.exe /k" not in config_text
    assert "prepared only" in config_data["launch_policy"]
    assert "not paper-scale, formal, or breakthrough evidence" in config_data["claim_scope"]

    assert "cmd.exe /k" not in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "C:\\Users" not in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "innovation1_spn_present_pair_mixer_consistency_r8_262k_seed0.csv" in launcher_text
    assert "--epochs 30" in launcher_text
    assert "--negative-mode" not in launcher_text
    assert "--dataset-cache-workers 4" in launcher_text
    assert "r8_pair_mixer_consistency_progress.jsonl" in launcher_text

    assert "cmd.exe /k" not in monitor_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "validate-results" in monitor_text
    assert "plot-results" in monitor_text
    assert "scripts/postprocess-pair-mixer-consistency" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text
    assert "plan_doc_committed_and_pushed" in monitor_text

    assert "i1_present_r8_pair_mixer_consistency_262k_seed0_gpu0_20260705" in plan_doc
    assert "not formal evidence" in plan_doc
    assert "暂不启动远程" in plan_doc or "不启动远程" in plan_doc


def test_present_pair_evidence_pooling_screen_plans_are_protocol_locked():
    smoke_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pair_evidence_pooling_screen_smoke.csv"
    )
    screen_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pair_evidence_pooling_screen_r8_65k_seed0.csv"
    )
    smoke_tasks = build_tasks(parse_args(["--plan", smoke_plan]))
    screen_tasks = build_tasks(parse_args(["--plan", screen_plan]))

    assert [task["model_key"] for task in smoke_tasks] == [
        "present_nibble_invp_pair_consistency_spn_only",
        "present_nibble_invp_pair_mixer_consistency_spn_only",
        "present_nibble_invp_pair_mixer_consistency_spn_only",
        "present_nibble_invp_pair_mixer_consistency_spn_only",
    ]
    assert [task["model_options"]["pooling"] for task in smoke_tasks] == [
        "topk_logsumexp",
        "topk_logsumexp",
        "logsumexp",
        "topk_mean",
    ]
    assert [task["model_options"]["pooling"] for task in screen_tasks] == [
        "topk_logsumexp",
        "topk_logsumexp",
        "logsumexp",
        "topk_mean",
    ]
    for task in smoke_tasks:
        assert task["rounds"] == 8
        assert task["seed"] == 0
        assert task["samples_per_class"] == 8
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "SMOKE only" in task["matching_evidence"]
        assert "not accuracy evidence" in task["matching_evidence"]

    for task in screen_tasks:
        assert task["rounds"] == 8
        assert task["seed"] == 0
        assert task["samples_per_class"] == 65536
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "SCREEN 65536/class" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]

    plan_doc = Path(
        "docs/experiments/innovation1-present-pair-evidence-pooling-screen-plan.md"
    ).read_text(encoding="utf-8")
    assert "pair evidence pooling mode" in plan_doc
    assert "不启动远程" in plan_doc
    assert "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705" in plan_doc
    assert "i1_present_r8_pairset_1m_seed0_gpu1_20260705" in plan_doc

    config = Path(
        "configs/remote/"
        "innovation1_spn_present_pair_evidence_pooling_screen_r8_65k_seed0_gpu0_20260705.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_text = config.read_text(encoding="utf-8")
    launcher_text = Path(
        "configs/remote/generated/"
        "run_i1_present_r8_pair_evidence_pooling_screen_65k_seed0_gpu0_20260705.cmd"
    ).read_text(encoding="utf-8")
    monitor_text = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r8_pair_evidence_pooling_screen_65k_seed0_gpu0_20260705.sh"
    ).read_text(encoding="utf-8")

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 4
    assert readiness["plan_rows"] == 4
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert config_data["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert config_data["dataset_cache_workers"] == 4
    assert "cmd.exe /c" in config_text
    assert "cmd.exe /k" not in config_text
    assert "prepared but do not launch" in config_data["launch_policy"]
    assert "not paper-scale, formal, or breakthrough evidence" in config_data["claim_scope"]

    assert "cmd.exe /k" not in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "innovation1_spn_present_pair_evidence_pooling_screen_r8_65k_seed0.csv" in launcher_text
    assert "--epochs 20" in launcher_text
    assert "--dataset-cache-workers 4" in launcher_text
    assert "pair_evidence_pooling_screen_progress.jsonl" in launcher_text

    assert "cmd.exe /k" not in monitor_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "scripts/postprocess-pair-evidence-pooling" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text
    assert "plan_doc_committed_and_pushed" in monitor_text


def test_present_r9_pair_evidence_pooling_screen_assets_are_gate_locked():
    smoke_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pair_evidence_pooling_screen_r9_smoke.csv"
    )
    screen_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pair_evidence_pooling_screen_r9_65k_seed0.csv"
    )
    smoke_tasks = build_tasks(parse_args(["--plan", smoke_plan]))
    screen_tasks = build_tasks(parse_args(["--plan", screen_plan]))

    assert [task["model_key"] for task in smoke_tasks] == [
        "present_nibble_invp_pair_consistency_spn_only",
        "present_nibble_invp_pair_mixer_consistency_spn_only",
        "present_nibble_invp_pair_mixer_consistency_spn_only",
        "present_nibble_invp_pair_mixer_consistency_spn_only",
    ]
    assert [task["model_options"]["pooling"] for task in screen_tasks] == [
        "topk_logsumexp",
        "topk_logsumexp",
        "logsumexp",
        "topk_mean",
    ]
    for task in smoke_tasks:
        assert task["rounds"] == 9
        assert task["seed"] == 0
        assert task["samples_per_class"] == 8
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "SMOKE only" in task["matching_evidence"]
        assert "not accuracy evidence" in task["matching_evidence"]

    for task in screen_tasks:
        assert task["rounds"] == 9
        assert task["seed"] == 0
        assert task["samples_per_class"] == 65536
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "SCREEN 65536/class" in task["matching_evidence"]
        assert "not formal reproduction or breakthrough evidence" in task["matching_evidence"]

    plan_doc = Path(
        "docs/experiments/innovation1-present-r9-pair-evidence-pooling-screen-plan.md"
    ).read_text(encoding="utf-8")
    assert "r9 Pair-Evidence Pooling Screen" in plan_doc
    assert "gate-locked" in plan_doc
    assert "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705" in plan_doc
    assert "i1_present_r8_pairset_1m_seed0_gpu1_20260705" in plan_doc

    config = Path(
        "configs/remote/"
        "innovation1_spn_present_pair_evidence_pooling_screen_r9_65k_seed0_gpu0_20260705.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_text = config.read_text(encoding="utf-8")
    launcher_text = Path(
        "configs/remote/generated/"
        "run_i1_present_r9_pair_evidence_pooling_screen_65k_seed0_gpu0_20260705.cmd"
    ).read_text(encoding="utf-8")
    monitor_text = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r9_pair_evidence_pooling_screen_65k_seed0_gpu0_20260705.sh"
    ).read_text(encoding="utf-8")

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 4
    assert readiness["plan_rows"] == 4
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert config_data["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert config_data["dataset_cache_workers"] == 4
    assert "cmd.exe /c" in config_text
    assert "cmd.exe /k" not in config_text
    assert "prepared but do not launch" in config_data["launch_policy"]
    assert "not paper-scale, formal, or breakthrough evidence" in config_data["claim_scope"]

    assert "cmd.exe /k" not in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "innovation1_spn_present_pair_evidence_pooling_screen_r9_65k_seed0.csv" in launcher_text
    assert "--epochs 20" in launcher_text
    assert "--dataset-cache-workers 4" in launcher_text
    assert "pair_evidence_pooling_screen_r9_progress.jsonl" in launcher_text

    assert "cmd.exe /k" not in monitor_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "scripts/postprocess-pair-evidence-pooling" in monitor_text
    assert "innovation1-present-r9-pair-evidence-pooling-screen-plan.md" in monitor_text
    assert "plan_doc_committed_and_pushed" in monitor_text


def test_present_r8_integral_inverse_feature_screen_plans_are_protocol_locked():
    smoke_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_integral_inverse_feature_screen_smoke.csv"
    )
    screen_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0.csv"
    )
    smoke_tasks = build_tasks(parse_args(["--plan", smoke_plan]))
    screen_tasks = build_tasks(parse_args(["--plan", screen_plan]))

    assert [task["model_key"] for task in smoke_tasks] == [
        "present_nibble_invp_pair_consistency_spn_only",
        "present_matrix_trail_hybrid_pairset_invp",
        "present_matrix_trail_hybrid_pairset_invp_sinv",
    ]
    assert [task["feature_encoding"] for task in smoke_tasks] == [
        "ciphertext_pair_bits",
        "present_pair_xor_paligned_cell_matrix_bits",
        "present_pair_xor_paligned_sinv_cell_matrix_bits",
    ]
    assert [task["feature_encoding"] for task in screen_tasks] == [
        "ciphertext_pair_bits",
        "present_pair_xor_paligned_cell_matrix_bits",
        "present_pair_xor_paligned_sinv_cell_matrix_bits",
    ]
    for task in smoke_tasks:
        assert task["rounds"] == 8
        assert task["seed"] == 0
        assert task["samples_per_class"] == 4
        assert task["pairs_per_sample"] == 16
        assert task["sample_structure"] == "plaintext_integral_nibble"
        assert task["integral_active_nibble"] == 0
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "SMOKE only" in task["matching_evidence"]
        assert "not accuracy evidence" in task["matching_evidence"]

    for task in screen_tasks:
        assert task["rounds"] == 8
        assert task["seed"] == 0
        assert task["samples_per_class"] == 65536
        assert task["pairs_per_sample"] == 16
        assert task["sample_structure"] == "plaintext_integral_nibble"
        assert task["integral_active_nibble"] == 0
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "SCREEN 65536/class" in task["matching_evidence"]
        assert "not formal evidence" in task["matching_evidence"]

    plan_doc = Path(
        "docs/experiments/innovation1-present-r8-integral-inverse-feature-screen-plan.md"
    ).read_text(encoding="utf-8")
    research_doc = Path("docs/research/innovation1-present-higher-round-strategy.md").read_text(
        encoding="utf-8"
    )
    assert "high-round data-representation screen" in plan_doc
    assert "not Zhang/Wang same-protocol model evidence" in plan_doc
    assert "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705" in plan_doc
    assert "i1_present_r8_pairset_1m_seed0_gpu1_20260705" in plan_doc
    assert "Integral / inverse-round 数据结构路线" in research_doc
    assert "Stage H5" in research_doc

    config = Path(
        "configs/remote/"
        "innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0_gpu0_20260705.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_text = config.read_text(encoding="utf-8")
    launcher_text = Path(
        "configs/remote/generated/"
        "run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu0_20260705.cmd"
    ).read_text(encoding="utf-8")
    monitor_text = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu0_20260705.sh"
    ).read_text(encoding="utf-8")

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 3
    assert readiness["plan_rows"] == 3
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert config_data["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert config_data["dataset_cache_workers"] == 4
    assert "cmd.exe /c" in config_text
    assert "cmd.exe /k" not in config_text
    assert "prepared but do not launch" in config_data["launch_policy"]
    assert "not Zhang/Wang same-protocol model evidence" in config_data["claim_scope"]

    assert "cmd.exe /k" not in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0.csv" in launcher_text
    assert "--epochs 20" in launcher_text
    assert "--dataset-cache-workers 4" in launcher_text
    assert "r8_integral_inverse_feature_screen_progress.jsonl" in launcher_text

    assert "cmd.exe /k" not in monitor_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "scripts/advance-integral-inverse-feature-result" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text

    retry_config = Path(
        "configs/remote/"
        "innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705.json"
    )
    retry_readiness = remote_readiness_report(retry_config)
    retry_artifacts = launch_artifacts(retry_config)
    retry_config_data = json.loads(retry_config.read_text(encoding="utf-8"))
    retry_launcher_text = Path(
        "configs/remote/generated/"
        "run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705.cmd"
    ).read_text(encoding="utf-8")
    retry_monitor_text = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705.sh"
    ).read_text(encoding="utf-8")

    assert retry_readiness["status"] == "pass"
    assert retry_readiness["expected_rows"] == 3
    assert retry_artifacts["status"] == "pass"
    assert retry_config_data["batch_size"] == 512
    assert retry_config_data["dataset_cache_root"] == config_data["dataset_cache_root"]
    assert "CUDA OOM" in retry_config_data["launch_policy"]
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in retry_launcher_text
    assert "--batch-size 512" in retry_launcher_text
    assert "cmd.exe /k" not in retry_launcher_text
    assert "Desktop" not in retry_launcher_text
    assert "Downloads" not in retry_launcher_text
    assert "scripts/advance-integral-inverse-feature-result" in retry_monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in retry_monitor_text


def test_present_r10_conditional_plan_has_no_remote_assets_after_r9_gate():
    plan_doc = Path(
        "docs/experiments/innovation1-present-r10-conditional-weak-probe-plan.md"
    ).read_text(encoding="utf-8")
    research_doc = Path("docs/research/innovation1-present-higher-round-strategy.md").read_text(
        encoding="utf-8"
    )
    r10_configs = list(Path("configs/remote").glob("*r10*"))
    r10_experiment_configs = list(Path("configs/experiment/innovation1").glob("*r10*"))

    assert (
        "planned / no remote assets / r9 from-scratch gate complete / wait for r8 pair-set 1M"
        in plan_doc
    )
    assert "r9 from-scratch weak-probe 已经 retrieved / validated / gate-noted" in plan_doc
    assert "i1_present_r8_pairset_1m_seed0_gpu1_20260705" in plan_doc
    assert "no r10 remote assets" in plan_doc
    assert "不创建 r10 remote config" in research_doc
    assert r10_configs == []
    assert r10_experiment_configs == []


def test_present_r8_pairset_1m_confirmation_plan_and_remote_assets_pass():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pairset_r8_1m_seed0.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 8
        assert task["seed"] == 0
        assert task["samples_per_class"] == 1_000_000
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "1000000/class" in task["matching_evidence"]

    config = Path(
        "configs/remote/"
        "innovation1_spn_present_pairset_r8_1m_seed0_gpu1_20260705.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)
    launcher_text = Path(
        "configs/remote/generated/"
        "run_i1_present_r8_pairset_1m_seed0_gpu1_20260705.cmd"
    ).read_text(encoding="utf-8")
    monitor_text = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r8_pairset_1m_seed0_gpu1_20260705.sh"
    ).read_text(encoding="utf-8")

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 2
    assert readiness["plan_rows"] == 2
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert "cmd.exe /k" not in config.read_text(encoding="utf-8")
    assert "r8_pairset_1m_progress.jsonl" in launcher_text
    assert "scripts/postprocess-r8-pairset-1m" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text
    assert "gate_note" not in monitor_text


def test_present_r8_pairset_1m_seed1_confirmation_assets_are_gate_locked():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pairset_r8_1m_seed1.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_invp_pair_consistency_spn_only",
    ]
    for task in tasks:
        assert task["rounds"] == 8
        assert task["seed"] == 1
        assert task["samples_per_class"] == 1_000_000
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "1000000/class" in task["matching_evidence"]
        assert "launch only if seed0 r8 pair-set 1M gate is positive" in task["matching_evidence"]

    config = Path(
        "configs/remote/"
        "innovation1_spn_present_pairset_r8_1m_seed1_gpu1_20260705.json"
    )
    readiness = remote_readiness_report(config)
    artifacts = launch_artifacts(config)
    config_data = json.loads(config.read_text(encoding="utf-8"))
    config_text = config.read_text(encoding="utf-8")
    launcher_text = Path(
        "configs/remote/generated/"
        "run_i1_present_r8_pairset_1m_seed1_gpu1_20260705.cmd"
    ).read_text(encoding="utf-8")
    monitor_text = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r8_pairset_1m_seed1_gpu1_20260705.sh"
    ).read_text(encoding="utf-8")
    plan_doc = Path(
        "docs/experiments/innovation1-present-r8-round-extension-ladder-plan.md"
    ).read_text(encoding="utf-8")

    assert readiness["status"] == "pass"
    assert readiness["expected_rows"] == 2
    assert readiness["plan_rows"] == 2
    assert "medium_scale_dataset_cache" in readiness["checked_invariants"]
    assert artifacts["status"] == "pass"
    assert "cmd.exe /c" in config_text
    assert "cmd.exe /k" not in config_text
    assert "i1_present_r8_pairset_1m_seed0_gpu1_20260705" in config_data["launch_policy"]
    assert "r8 pair-set 1M gate returns" in config_data["launch_policy"]
    assert config_data["dataset_cache_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert config_data["dataset_cache_workers"] == 4

    assert "cmd.exe /k" not in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "innovation1_spn_present_pairset_r8_1m_seed1.csv" in launcher_text
    assert "--dataset-cache-workers 4" in launcher_text
    assert "r8_pairset_1m_seed1_progress.jsonl" in launcher_text

    assert "cmd.exe /k" not in monitor_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "scripts/postprocess-r8-pairset-1m" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text

    assert "i1_present_r8_pairset_1m_seed1_gpu1_20260705" in plan_doc
    assert "do not launch until seed0 r8 pair-set 1M is retrieved / validated / postprocessed" in plan_doc
    assert "1000000/class seed1 = paper-scale single-seed confirmation" in plan_doc


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


def test_prepared_trail_family_cache_worker_benchmark_assets_are_speed_only():
    launcher = Path("configs/remote/generated/run_i1_trail_family_cache_workers_4_8_20260703.cmd")
    monitor = Path("configs/remote/generated/monitor_i1_trail_family_cache_workers_4_8_20260703.sh")
    plan = Path("docs/experiments/innovation1-mapreduce-acceleration-plan.md").read_text(encoding="utf-8")
    launcher_text = launcher.read_text(encoding="utf-8")
    monitor_text = monitor.read_text(encoding="utf-8")

    assert "i1_trail_family_cache_workers_4_8_20260703" in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "C:\\Users" not in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "cmd.exe /k" not in launcher_text
    assert "bench-trail-family-cache" in launcher_text
    assert "--workers 4 8" in launcher_text
    assert "--sample-structure zhang_wang_case2_official_mcnd" in launcher_text
    assert "--negative-mode encrypted_random_plaintexts" in launcher_text
    assert "%SOURCE_DIR%\\plugins\\blockcipher-training-accelerator\\src;%SOURCE_DIR%\\src" in launcher_text

    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "results/trail_family_cache_bench/summary.json" in monitor_text
    assert "summary_ready" in monitor_text

    assert "status: prepared / not launched" in plan
    assert "trail-family feature-cache generation speed only" in plan
    assert "must not interrupt or replace" in plan


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


def test_present_r8_pairset_aggregation_control_262k_plans_are_staged_and_protocol_locked():
    scorer_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pairset_aggregation_control_single_pair_r8_262k.csv"
    )
    learned_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_pairset_aggregation_control_r8_262k.csv"
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
        assert task["rounds"] == 8
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
        assert "S-box transition prior gate seed0" in config["launch_policy"]
        assert "retrieved, validated, plan-aligned, postprocessed" in config["launch_policy"]
        assert "pair-set aggregation control" in config["launch_policy"]
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
    assert "while S-box transition prior gate seed0 is running or ungated" in plan_doc
    assert "remaining_requirement = wait for S-box transition prior gate seed0" in plan_doc
    assert "i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703" in plan_doc
    assert "i1_trail_family_r7_262k_seed0_gpu1_20260702" in plan_doc
    assert "i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702" in plan_doc
    assert "stop_topology_aware_network_route" in plan_doc


def test_present_pairset_aggregation_control_seed1_remote_configs_are_gated_and_ready():
    plan_doc = Path("docs/experiments/innovation1-pairset-aggregation-control-plan.md").read_text(encoding="utf-8")
    scorer_path = Path(
        "configs/remote/"
        "innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k_seed1_gpu1_20260702.json"
    )
    learned_path = Path(
        "configs/remote/innovation1_spn_present_pairset_aggregation_control_r7_262k_seed1_gpu1_20260702.json"
    )
    scorer_config = json.loads(scorer_path.read_text(encoding="utf-8"))
    learned_config = json.loads(learned_path.read_text(encoding="utf-8"))

    assert scorer_config["expected_rows"] == 1
    assert learned_config["expected_rows"] == 2
    assert scorer_config["pairset_stage"] == "single_pair_scorer_checkpoint"
    assert learned_config["pairset_stage"] == "learned_pairset_plus_frozen_aggregation_gate"
    assert scorer_config["run_id"] == "i1_pairset_single_pair_scorer_r7_262k_seed1_gpu1_20260702"
    assert learned_config["run_id"] == "i1_pairset_aggregation_control_r7_262k_seed1_gpu1_20260702"
    assert scorer_config["plan"].endswith(
        "innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k_seed1.csv"
    )
    assert learned_config["plan"].endswith(
        "innovation1_spn_present_pairset_aggregation_control_r7_262k_seed1.csv"
    )
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
        assert "pair-set seed0" in config["launch_policy"]
        assert "support_learned_pairset_consistency" in config["launch_policy"]
        assert "weak_pairset_consistency_signal" in config["launch_policy"]
        assert "MEDIUM 262144/class pair-set aggregation-control" in config["claim_scope"]
        assert "not formal reproduction or breakthrough evidence" in config["claim_scope"]

    scorer_report = remote_readiness_report(scorer_path)
    learned_report = remote_readiness_report(learned_path)
    assert scorer_report["status"] == "pass"
    assert learned_report["status"] == "pass"
    assert "pairset_aggregation_stage_lock" in scorer_report["checked_invariants"]
    assert "pairset_aggregation_stage_lock" in learned_report["checked_invariants"]
    assert "candidate_trail_protocol_lock" not in learned_report["checked_invariants"]
    assert "pairset_seed1_confirmation" in plan_doc
    assert "i1_pairset_aggregation_control_r7_262k_seed1_gpu1_20260702" in plan_doc


def test_present_r8_pairset_aggregation_control_remote_configs_are_prepared_and_ready():
    plan_doc = Path(
        "docs/experiments/innovation1-present-r8-pairset-aggregation-control-plan.md"
    ).read_text(encoding="utf-8")
    scorer_path = Path(
        "configs/remote/"
        "innovation1_spn_present_pairset_aggregation_control_single_pair_r8_262k_gpu0_20260705.json"
    )
    learned_path = Path(
        "configs/remote/innovation1_spn_present_pairset_aggregation_control_r8_262k_gpu0_20260705.json"
    )
    scorer_config = json.loads(scorer_path.read_text(encoding="utf-8"))
    learned_config = json.loads(learned_path.read_text(encoding="utf-8"))

    assert scorer_config["expected_rows"] == 1
    assert learned_config["expected_rows"] == 2
    assert scorer_config["pairset_stage"] == "single_pair_scorer_checkpoint"
    assert learned_config["pairset_stage"] == "learned_pairset_plus_frozen_aggregation_gate"
    assert scorer_config["run_id"] == "i1_pairset_single_pair_scorer_r8_262k_seed0_gpu0_20260705"
    assert learned_config["run_id"] == "i1_pairset_aggregation_control_r8_262k_seed0_gpu0_20260705"
    assert scorer_config["plan"].endswith(
        "innovation1_spn_present_pairset_aggregation_control_single_pair_r8_262k.csv"
    )
    assert learned_config["plan"].endswith(
        "innovation1_spn_present_pairset_aggregation_control_r8_262k.csv"
    )
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
        assert config["device"] == "cuda:0"
        assert config["train_eval_interval"] == 0
        assert config["dataset_cache"] is True
        assert config["dataset_cache_root"].startswith(
            "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
        )
        assert config["dataset_cache_workers"] == 4
        assert "cmd.exe /c" in config["launch_policy"]
        assert "cmd.exe /k" not in config["launch_policy"]
        assert "G:\\lxy" in config["launch_policy"]
        assert "r8 pair-set 1M" in config["launch_policy"]
        assert "r9 weak-probe" in config["launch_policy"]
        assert "MEDIUM 262144/class r8 pair-set aggregation-control" in config["claim_scope"]
        assert "not formal reproduction or breakthrough evidence" in config["claim_scope"]

    scorer_report = remote_readiness_report(scorer_path)
    learned_report = remote_readiness_report(learned_path)
    artifacts = launch_artifacts(learned_path)
    assert scorer_report["status"] == "pass"
    assert learned_report["status"] == "pass"
    assert artifacts["status"] == "pass"
    assert "pairset_aggregation_stage_lock" in scorer_report["checked_invariants"]
    assert "pairset_aggregation_stage_lock" in learned_report["checked_invariants"]
    assert "candidate_trail_protocol_lock" not in learned_report["checked_invariants"]
    assert "wait for active r8 pair-set 1M gate" in plan_doc
    assert "r9 from-scratch weak-probe has already completed" in plan_doc
    assert "single-pair scorer remote config = pass" in plan_doc
    assert "frozen single-pair InvP score aggregation" in plan_doc
    assert "i1_present_r8_pairset_1m_seed0_gpu1_20260705" in plan_doc
    assert "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705" in plan_doc


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


def test_pairset_aggregation_readiness_ignores_route_names_in_launch_policy():
    report = remote_readiness_report(
        Path("configs/remote/innovation1_spn_present_pairset_aggregation_control_r7_262k_gpu1_20260630.json")
    )

    assert report["status"] == "pass"
    assert "pairset_aggregation_stage_lock" in report["checked_invariants"]
    assert "candidate_trail_protocol_lock" not in report["checked_invariants"]
    assert "transition_spectrum_protocol_lock" not in report["checked_invariants"]
    assert not any("candidate_trail" in error for error in report["errors"])


def test_present_neural_ensemble_remote_config_is_ready_and_artifact_locked():
    config_path = Path(
        "configs/remote/"
        "innovation1_spn_present_neural_ensemble_r7_65k_seed0_gpu0_20260705.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = remote_readiness_report(config_path)

    assert config["run_id"] == "i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705"
    assert config["expected_rows"] == 3
    assert config["device"] == "cuda:0"
    assert config["dataset_cache"] is True
    assert config["dataset_cache_workers"] == 4
    assert config["checkpoint_output_dir"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert config["score_artifacts_root"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert config["ensemble_summary_output"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
    )
    assert "cmd.exe /c" in config["launch_policy"]
    assert "cmd.exe /k" not in config["launch_policy"]
    assert "application-level score aggregation" in config["claim_scope"]
    assert "not formal" in config["claim_scope"]

    assert report["status"] == "pass"
    assert report["expected_rows"] == 3
    assert report["plan_rows"] == 3
    assert "medium_scale_dataset_cache" in report["checked_invariants"]
    assert "neural_ensemble_score_artifact_lock" in report["checked_invariants"]


def test_present_neural_ensemble_remote_launch_assets_export_and_retrieve_scores():
    launcher = Path(
        "configs/remote/generated/"
        "run_i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705.cmd"
    )
    monitor = Path(
        "configs/remote/generated/"
        "monitor_i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705.sh"
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
    assert "SCORE_ARTIFACT_DIR" in launcher_text
    assert "--checkpoint-output-dir \"%CHECKPOINT_DIR%\"" in launcher_text
    assert "scripts\\export-checkpoint-scores" in launcher_text
    assert "scripts\\evaluate-neural-ensemble" in launcher_text
    assert "row0001_present_zhang_wang_keras_mcnd_seed0.pt" in launcher_text
    assert "row0002_present_nibble_invp_only_spn_only_seed0.pt" in launcher_text
    assert "row0003_present_nibble_ddt_graph_seed0.pt" in launcher_text
    assert "--eval-row-index 0" in launcher_text
    assert "--eval-row-index 1" in launcher_text
    assert "--eval-row-index 2" in launcher_text
    assert "neural_ensemble_summary.json" in launcher_text
    assert "score_artifacts" in monitor_text
    assert "checkpoints" in monitor_text
    assert "neural_ensemble_summary.json" in monitor_text
    assert "postprocess-neural-ensemble" in monitor_text
    assert "--ensemble-summary" in monitor_text
    assert "--expected-rows \"${EXPECTED_ROWS}\"" in monitor_text
    assert "completed_missing_or_incomplete_results" in monitor_text


def test_trail_position_medium_remote_launch_assets_export_scores_only():
    launcher = Path(
        "configs/remote/generated/"
        "run_i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706.cmd"
    )
    monitor = Path(
        "configs/remote/generated/"
        "monitor_i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706.sh"
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
    assert "SCORE_ARTIFACT_DIR" in launcher_text
    assert "TRAIL_POSITION_BEAMSTATS_MEDIUM_PREPARED_ONLY" in launcher_text
    assert "scripts\\check-remote-readiness" in launcher_text
    assert "--checkpoint-output-dir \"%CHECKPOINT_DIR%\"" in launcher_text
    assert "--dataset-cache-root \"%DATASET_CACHE_ROOT%\"" in launcher_text
    assert "--dataset-cache-workers 4" in launcher_text
    assert "scripts\\export-checkpoint-scores" in launcher_text
    assert launcher_text.count("--dataset-cache-root \"%DATASET_CACHE_ROOT%\"") >= 3
    assert launcher_text.count("--progress-output \"%LOG_DIR%\\trail_position_beamstats_score_export_progress.jsonl\"") == 2
    assert "scripts\\evaluate-neural-ensemble" not in launcher_text
    assert "row0001_present_pairset_global_stats_seed0.pt" in launcher_text
    assert "row0002_present_trail_position_stats_pairset_seed0.pt" in launcher_text
    assert "--eval-row-index 0" in launcher_text
    assert "--eval-row-index 1" in launcher_text
    assert "--expert-family trail_position_global_control" in launcher_text
    assert "--candidate-status near_neighbor_control" in launcher_text
    assert "--expert-family trail_position" in launcher_text
    assert "--candidate-status weak_positive" in launcher_text
    assert "score_artifacts" in monitor_text
    assert "checkpoints" in monitor_text
    assert "global_stats_control/models.json" in monitor_text
    assert "trail_position/models.json" in monitor_text
    assert "postprocess-neural-ensemble" not in monitor_text
    assert "completed_missing_or_incomplete_results" in monitor_text
    assert '${LOCAL_ROOT}/logs/${RUN_ID}_done.marker' in monitor_text
    assert '${LOCAL_ROOT}/logs/*done.marker' not in monitor_text


def test_trail_position_medium_remote_score_export_repair_asset():
    repair = Path(
        "configs/remote/generated/"
        "repair_i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706_score_export.cmd"
    )
    repair_text = repair.read_text(encoding="utf-8")

    assert "cmd.exe /k" not in repair_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in repair_text
    assert "C:\\Users" not in repair_text
    assert "Desktop" not in repair_text
    assert "Downloads" not in repair_text
    assert "AppData" not in repair_text
    assert "scripts\\train" not in repair_text
    assert repair_text.count("scripts\\export-checkpoint-scores") == 2
    assert "scripts\\verify-score-artifacts" in repair_text
    assert "--expected-rows 65536" in repair_text
    assert "%SCORE_ARTIFACT_DIR%\\verification_summary.json" in repair_text
    assert "present_pairset_global_stats:trail_position_global_control:near_neighbor_control" in repair_text
    assert "present_trail_position_stats_pairset:trail_position:weak_positive" in repair_text
    assert "--dataset-cache-root \"%DATASET_CACHE_ROOT%\"" in repair_text
    assert repair_text.count("--progress-output \"%LOG_DIR%\\trail_position_beamstats_score_export_repair_progress.jsonl\"") == 2
    assert "row0001_present_pairset_global_stats_seed0.pt" in repair_text
    assert "row0002_present_trail_position_stats_pairset_seed0.pt" in repair_text
    assert "%RUN_ID%_score_export_repair_started.marker" in repair_text
    assert "%RUN_ID%_score_export_done.marker" in repair_text
    assert "%RUN_ID%_done.marker" in repair_text
    assert "%RUN_ID%_score_export_repair_failed.marker" in repair_text


def test_trail_position_medium_remote_score_export_repair_launcher_asset():
    launcher = Path(
        "configs/remote/generated/"
        "launch_repair_i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706_score_export.sh"
    )
    launcher_text = launcher.read_text(encoding="utf-8")

    assert "cmd.exe /k" not in launcher_text
    assert "scripts\\train" not in launcher_text
    assert "ssh lxy-a6000" in launcher_text
    assert "cmd.exe /c if not exist" in launcher_text
    assert "git fetch origin main" in launcher_text
    assert "git pull --ff-only origin main" in launcher_text
    assert "git clone --branch main" in launcher_text
    assert "&& call" in launcher_text
    assert "G:\\\\lxy\\\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "repair_i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706_score_export.cmd" in launcher_text
    assert "outputs/remote_results/i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706/monitor" in launcher_text
    assert "score_export_repair_launch.log" in launcher_text
    assert "score_export_repair_launch_failed.marker" in launcher_text
    assert "score_export_repair_launch_done.marker" in launcher_text


def test_trail_position_score_analysis_cli_asset():
    script = Path("scripts/analyze-trail-position-scores")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.analyze_trail_position_scores" in script_text


def test_trail_position_report_cli_asset():
    script = Path("scripts/render-trail-position-report")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.render_trail_position_report" in script_text


def test_bit_sensitivity_projection_cli_asset():
    script = Path("scripts/select-bit-sensitivity-projection")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.select_bit_sensitivity_projection" in script_text


def test_export_bit_sensitivity_features_cli_asset():
    script = Path("scripts/export-bit-sensitivity-features")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.export_bit_sensitivity_features" in script_text


def test_apply_bit_sensitivity_projection_cli_asset():
    script = Path("scripts/apply-bit-sensitivity-projection")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.apply_bit_sensitivity_projection" in script_text


def test_postprocess_bit_sensitivity_projection_cli_asset():
    script = Path("scripts/postprocess-bit-sensitivity-projection")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.postprocess_bit_sensitivity_projection" in script_text


def test_export_compressed_span_blocks_cli_asset():
    script = Path("scripts/export-compressed-span-blocks")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.export_compressed_span_blocks" in script_text


def test_summarize_compressed_span_route_cli_asset():
    script = Path("scripts/summarize-compressed-span-route")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.summarize_compressed_span_route" in script_text


def test_fit_compressed_span_grouped_expert_cli_asset():
    script = Path("scripts/fit-compressed-span-grouped-expert")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.fit_compressed_span_grouped_expert" in script_text


def test_fit_compressed_span_interaction_expert_cli_asset():
    script = Path("scripts/fit-compressed-span-interaction-expert")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.fit_compressed_span_interaction_expert" in script_text


def test_fit_compressed_span_block_interaction_expert_cli_asset():
    script = Path("scripts/fit-compressed-span-block-interaction-expert")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.fit_compressed_span_block_interaction_expert" in script_text


def test_fit_compressed_span_low_rank_interaction_expert_cli_asset():
    script = Path("scripts/fit-compressed-span-low-rank-interaction-expert")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.fit_compressed_span_low_rank_interaction_expert" in script_text


def test_fit_compressed_span_learned_low_rank_interaction_expert_cli_asset():
    script = Path("scripts/fit-compressed-span-learned-low-rank-interaction-expert")
    script_text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert script.stat().st_mode & 0o111
    assert "blockcipher_nd.cli.fit_compressed_span_learned_low_rank_interaction_expert" in script_text


def test_trail_position_1m_conditional_plan_is_gated_and_controlled():
    plan = Path("docs/experiments/innovation1-present-r8-trail-position-1m-conditional-plan.md")
    assert plan.exists()
    text = plan.read_text(encoding="utf-8")

    assert "conditional plan only / no launch" in text
    assert "1000000/class" in text
    assert "encrypted_random_plaintexts" in text
    assert "present_pairset_global_stats" in text
    assert "present_trail_position_stats_pairset" in text
    assert "postprocess decision = support_trail_position_score_residual_all_runs" in text
    assert "This plan does not authorize an immediate remote run" in text
    assert "cmd.exe /c, not cmd.exe /k" in text


def test_trail_position_262k_followup_remote_assets_are_scale_up_and_verify_scores():
    for seed, gpu in [(0, 0), (1, 1)]:
        suffix = f"262k_seed{seed}_gpu{gpu}_20260706"
        run_id = f"i1_present_r8_trail_position_beamstats_{suffix}"
        plan = Path(
            "configs/experiment/innovation1/"
            f"innovation1_spn_present_r8_trail_position_beamstats_262k_seed{seed}.csv"
        )
        config = Path(
            "configs/remote/"
            f"innovation1_spn_present_r8_trail_position_beamstats_262k_seed{seed}_gpu{gpu}_20260706.json"
        )
        launcher = Path("configs/remote/generated") / f"run_{run_id}.cmd"
        monitor = Path("configs/remote/generated") / f"monitor_{run_id}.sh"

        plan_text = plan.read_text(encoding="utf-8")
        config_text = config.read_text(encoding="utf-8")
        launcher_text = launcher.read_text(encoding="utf-8")
        monitor_text = monitor.read_text(encoding="utf-8")

        assert ",262144,16," in plan_text
        assert ",65536,16," not in plan_text
        assert f"seed{seed}" in plan_text
        assert "MEDIUM DIAGNOSTIC 262144/class" in plan_text
        assert f'"run_id": "{run_id}"' in config_text
        assert f'"device": "cuda:{gpu}"' in config_text
        assert '"score_export_after_training": true' in config_text
        assert "trail_position_beamstats_262k_cache" in config_text
        assert "cmd.exe /k" not in launcher_text
        assert "scripts\\train" in launcher_text
        assert "scripts\\export-checkpoint-scores" in launcher_text
        assert "scripts\\verify-score-artifacts" in launcher_text
        assert "--expected-rows 262144" in launcher_text
        assert "%SCORE_ARTIFACT_DIR%\\verification_summary.json" in launcher_text
        assert "scripts\\evaluate-neural-ensemble" not in launcher_text
        assert f"--device cuda:{gpu}" in launcher_text
        assert "score_artifacts/verification_summary.json" in monitor_text
        assert '${LOCAL_ROOT}/logs/${RUN_ID}_done.marker' in monitor_text
        assert '${LOCAL_ROOT}/logs/*done.marker' not in monitor_text


def test_trail_position_262k_followup_local_launchers_handoff_to_tmux_monitors():
    for seed, gpu in [(0, 0), (1, 1)]:
        run_id = f"i1_present_r8_trail_position_beamstats_262k_seed{seed}_gpu{gpu}_20260706"
        launcher = Path("configs/remote/generated") / f"launch_{run_id}.sh"
        launcher_text = launcher.read_text(encoding="utf-8")

        assert "cmd.exe /k" not in launcher_text
        assert "ssh lxy-a6000" in launcher_text
        assert "cmd.exe /c" in launcher_text
        assert "&& call" in launcher_text
        assert "REMOTE_RUN_ROOT=\"G:\\\\lxy\\\\blockcipher-structure-adaptive-nd-runs" in launcher_text
        assert "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git" in launcher_text
        assert "if not exist \\\"${REMOTE_RUN_ROOT}\\\" mkdir \\\"${REMOTE_RUN_ROOT}\\\"" in launcher_text
        assert "git clone --branch main" in launcher_text
        assert "git pull --ff-only origin main" in launcher_text
        assert f"run_{run_id}.cmd" in launcher_text
        assert f"monitor_{run_id}.sh" in launcher_text
        assert f"monitor_i1_present_r8_trailpos_262k_seed{seed}_20260706" in launcher_text
        assert f"outputs/remote_results/{run_id}/monitor" in launcher_text
        assert "tmux new-session -d -s" in launcher_text
        assert "tmux has-session -t \"${MONITOR_SESSION}\"" in launcher_text
        assert "monitor_started monitor=${MONITOR_SESSION}" in launcher_text
        assert "monitor_already_running monitor=${MONITOR_SESSION}" in launcher_text
        assert launcher_text.index("monitor_started monitor=${MONITOR_SESSION}") < launcher_text.index("ssh lxy-a6000")
        assert "launch_done.marker" in launcher_text
        assert "launch_failed.marker" in launcher_text


def test_sbox_transition_prior_gate_plan_is_protocol_locked_and_deferred():
    plan = Path("docs/experiments/innovation1-sbox-transition-prior-gate-plan.md").read_text(encoding="utf-8")

    assert "Status:** seed0 launched / watcher-managed running / waiting for retrieved" in plan
    assert "Do not\nlaunch seed1 or any replacement SPN route until seed0 is retrieved" in plan
    assert "i1_trail_family_r7_262k_seed0_gpu1_20260702" in plan
    assert "PRESENT-80" in plan
    assert "zhang_wang_case2_official_mcnd" in plan
    assert "encrypted_random_plaintexts" in plan
    assert "present_zhang_wang2022_mcnd" in plan
    assert "262144/class" in plan
    assert "medium diagnostic only" in plan
    assert "present_nibble_invp_sbox_prior_gate" in plan
    assert "present_nibble_invp_no_ddt_gate" in plan
    assert "present_nibble_invp_shuffled_sbox_prior_gate" in plan
    assert "full-column DDT prior" in plan
    assert "active flag + 16 normalized DDT counts" in plan
    assert "true_prior_gate_auc >= InvP-only anchor AUC + 0.001" in plan
    assert "formal route evidence" in plan
    assert "1000000/class" in plan
    assert "cmd.exe /c" in plan
    assert "G:\\lxy" in plan


def test_sbox_transition_prior_gate_smoke_config_is_protocol_locked():
    plan = "configs/experiment/innovation1/innovation1_spn_present_sbox_transition_prior_gate_smoke.csv"
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_sbox_prior_gate",
        "present_nibble_invp_no_ddt_gate",
        "present_nibble_invp_shuffled_sbox_prior_gate",
    ]
    for task in tasks:
        assert task["cipher_key"] == "present80"
        assert task["rounds"] == 7
        assert task["seed"] == 0
        assert task["samples_per_class"] == 8
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["validation_key"] == 0x11111111111111111111
        assert "SMOKE only" in task["matching_evidence"]
        assert "not accuracy evidence" in task["matching_evidence"]


def test_sbox_transition_prior_gate_262k_seed0_assets_are_ready_and_deferred():
    plan = "configs/experiment/innovation1/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0.csv"
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_sbox_prior_gate",
        "present_nibble_invp_no_ddt_gate",
        "present_nibble_invp_shuffled_sbox_prior_gate",
    ]
    for task in tasks:
        assert task["cipher_key"] == "present80"
        assert task["rounds"] == 7
        assert task["seed"] == 0
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["validation_key"] == 0x11111111111111111111
        assert task["learning_rate"] == 0.0001
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert task["early_stopping_patience"] == 8
        assert task["early_stopping_min_delta"] == 0.0001
        assert "MEDIUM_DIAGNOSTIC" in task["matching_evidence"]
        if task["model_key"].startswith("present_nibble_invp_") and task["model_key"] != "present_nibble_invp_only_spn_only":
            assert task["model_options"]["prior_feature_mode"] == "full_column"

    remote = Path(
        "configs/remote/"
        "innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0_gpu1_20260703.json"
    )
    readiness = remote_readiness_report(remote)
    assert readiness["status"] == "pass"
    assert "sbox_prior_protocol_lock" in readiness["checked_invariants"]

    launcher = Path("configs/remote/generated/run_i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703.cmd")
    monitor = Path("configs/remote/generated/monitor_i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703.sh")
    launcher_text = launcher.read_text(encoding="utf-8")
    monitor_text = monitor.read_text(encoding="utf-8")
    assert launcher.exists()
    assert monitor.exists()
    assert "cmd.exe /k" not in launcher_text
    assert "scripts\\train" in launcher_text
    assert "innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0.csv" in launcher_text
    assert "scripts/postprocess-sbox-prior" in monitor_text
    assert "innovation1-sbox-transition-prior-gate-plan.md" in monitor_text
    assert "do not launch while trail-family" in remote.read_text(encoding="utf-8")


def test_sbox_transition_prior_gate_262k_seed1_assets_are_ready_and_conditional():
    plan = "configs/experiment/innovation1/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed1.csv"
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["model_key"] for task in tasks] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_sbox_prior_gate",
        "present_nibble_invp_no_ddt_gate",
        "present_nibble_invp_shuffled_sbox_prior_gate",
    ]
    for task in tasks:
        assert task["cipher_key"] == "present80"
        assert task["rounds"] == 7
        assert task["seed"] == 1
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["difference_profile"] == "present_zhang_wang2022_mcnd"
        assert task["validation_key"] == 0x11111111111111111111
        assert task["learning_rate"] == 0.0001
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert task["early_stopping_patience"] == 8
        assert task["early_stopping_min_delta"] == 0.0001
        assert "MEDIUM_DIAGNOSTIC" in task["matching_evidence"]
        if task["model_key"].startswith("present_nibble_invp_") and task["model_key"] != "present_nibble_invp_only_spn_only":
            assert task["model_options"]["prior_feature_mode"] == "full_column"

    remote = Path(
        "configs/remote/"
        "innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed1_gpu1_20260703.json"
    )
    readiness = remote_readiness_report(remote)
    assert readiness["status"] == "pass"
    assert "sbox_prior_protocol_lock" in readiness["checked_invariants"]

    launcher = Path("configs/remote/generated/run_i1_sbox_prior_gate_r7_262k_seed1_gpu1_20260703.cmd")
    monitor = Path("configs/remote/generated/monitor_i1_sbox_prior_gate_r7_262k_seed1_gpu1_20260703.sh")
    launcher_text = launcher.read_text(encoding="utf-8")
    monitor_text = monitor.read_text(encoding="utf-8")
    assert launcher.exists()
    assert monitor.exists()
    assert "cmd.exe /k" not in launcher_text
    assert "scripts\\train" in launcher_text
    assert "innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed1.csv" in launcher_text
    assert "scripts/postprocess-sbox-prior" in monitor_text
    assert "innovation1-sbox-transition-prior-gate-plan.md" in monitor_text
    assert "launch only after seed0" in remote.read_text(encoding="utf-8")


def test_present_sbox_prior_gate_uses_max_ddt_probability_for_prior_pooling():
    from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
        _PresentSboxTransitionPriorGateEncoder,
    )

    encoder = _PresentSboxTransitionPriorGateEncoder(input_bits=128, prior_token_dim=8, base_channels=4)
    priors = torch.zeros((1, 1, 16, 17), dtype=torch.float32)
    priors[0, 0, :, 0] = 1.0
    priors[0, 0, 0, 1] = 0.125
    priors[0, 0, 0, 5] = 0.375
    weights = encoder.prior_reliability_weights(priors)

    assert weights.shape == (1, 1, 16, 1)
    assert weights[0, 0, 0, 0] == pytest.approx(0.375)
    assert weights[0, 0, 0, 0] != pytest.approx(0.125)


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
    assert "The shuffled-cell row is not optional for the medium diagnostic matrix" in plan
    assert "formal postprocess = requires candidate_trail_consistency_shuffled_cells" in plan
    assert "never `support_candidate_trail_route`" in plan
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
    assert "The shuffled-P row is mandatory for the medium diagnostic matrix" in plan
    assert "formal postprocess requires bit_transition_spectrum_shuffled_p" in plan
    assert "never `support_transition_spectrum_route`" in plan
    assert "true transition-spectrum route >= shuffled-P control + 0.001 AUC" in plan
    assert "medium seed0/seed1 plans and remote configs prepared but gate-locked" in plan
    assert "seed0 and conditional seed1 readiness assets prepared" in plan
    assert "do not launch the seed0 medium remote config" in plan
    assert "do not launch the seed1 medium remote config" in plan
    assert "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702.json" in plan
    assert "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1_gpu1_20260702.json" in plan
    assert "scripts/spn-transition-spectrum-matrix" in plan
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in plan
    assert "implementation_status = local route implemented" in plan
    assert "streamed disk-backed feature cache" in plan
    assert "src/blockcipher_nd/planning/transition_spectrum_postprocess.py" in plan
    assert "no code implementation yet" not in plan


def test_trail_family_consistency_plan_tracks_active_seed0_handoff():
    plan = Path("docs/experiments/innovation1-trail-family-consistency-plan.md").read_text(
        encoding="utf-8"
    )

    assert "seed0 launched / watcher-managed medium diagnostic" in plan
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
    assert "seed0 launched after candidate-trail and bit-transition-spectrum stopped" in plan
    assert "seed1 prepared but gated" in plan
    assert "scripts/spn-trail-family-matrix" in plan
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in plan
    assert "implementation_status = local smoke runner/gate/postprocess implemented" in plan
    assert "medium seed0 and seed1 plan/remote/launcher/monitor prepared" in plan
    assert "remote_config_status = seed0 launched after candidate-trail and bit-transition-spectrum stopped" in plan
    assert "i1_trail_family_r7_262k_seed0_gpu1_20260702" in plan
    assert "i1_trail_family_r7_262k_seed1_gpu1_20260702" in plan
    assert "local_monitor_session = monitor_i1_trail_family_seed0_20260702" in plan
    assert "status = running" in plan
    assert "results_jsonl_exists = false" in plan
    assert "support_trail_family_route or weak_trail_family_signal from seed0" in plan
    assert "This implementation is not result evidence" in plan
    assert "scripts/gate-trail-family" in plan
    assert "scripts/postprocess-trail-family" in plan
    assert "trail_family_consistency_false_family control" in plan


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


def test_present_trail_family_features_are_deterministic_and_controlled():
    from blockcipher_nd.features.spn_trail_family import (
        present_pair_trail_family_features,
        present_pairset_trail_family_features,
    )

    cipher = build_cipher("present80", 7, key=0)
    pairs = [
        (0x0123456789ABCDEF, 0x0123456789ABCDE6),
        (0x1111111111111111, 0x1111111111111118),
        (0x2222222222222222, 0x222222222222222B),
        (0x3333333333333333, 0x333333333333333A),
    ]

    pair_features = present_pair_trail_family_features(pairs[0][0], pairs[0][1], width=64, cipher=cipher)
    true_features = present_pairset_trail_family_features(pairs, width=64, cipher=cipher)
    repeat_features = present_pairset_trail_family_features(pairs, width=64, cipher=cipher)
    false_features = present_pairset_trail_family_features(pairs, width=64, cipher=cipher, false_family=True)

    assert pair_features.dtype == np.float32
    assert true_features.dtype == np.float32
    assert true_features.shape == repeat_features.shape
    assert true_features.shape == false_features.shape
    assert true_features.shape[0] > pair_features.shape[0]
    np.testing.assert_allclose(true_features, repeat_features)
    assert np.isfinite(true_features).all()
    assert not np.allclose(true_features[:64], false_features[:64])


def test_present_pairset_trail_family_reuses_pair_templates(monkeypatch):
    import blockcipher_nd.features.spn_trail_family as trail_family

    cipher = build_cipher("present80", 7, key=0)
    pairs = [
        (0x0123456789ABCDEF, 0x0123456789ABCDE6),
        (0x1111111111111111, 0x1111111111111118),
        (0x2222222222222222, 0x222222222222222B),
    ]
    original = trail_family.present_pair_trail_family_template
    calls = 0

    def counted_template(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(trail_family, "present_pair_trail_family_template", counted_template)

    features = trail_family.present_pairset_trail_family_features(pairs, width=64, cipher=cipher)

    assert features.dtype == np.float32
    assert calls == len(pairs)


def test_present_false_family_controls_pair_level_active_masks():
    from blockcipher_nd.features.spn_trail_family import present_pairset_trail_family_features

    cipher = build_cipher("present80", 7, key=0)
    pairs = [
        (0x0123456789ABCDEF, 0x0123456789ABCDE6),
        (0x1111111111111111, 0x1111111111111118),
        (0x2222222222222222, 0x222222222222222B),
        (0x3333333333333333, 0x333333333333333A),
    ]

    true_features = present_pairset_trail_family_features(pairs, width=64, cipher=cipher)
    false_features = present_pairset_trail_family_features(pairs, width=64, cipher=cipher, false_family=True)

    cells = 16
    depth = 3
    family_dim = depth * (13 + cells)
    pair_dim = depth * (12 + cells)
    active_mean_indices = []
    for layer_index in range(depth):
        layer_start = family_dim + layer_index * (12 + cells)
        active_mean_indices.extend(range(layer_start + 12, layer_start + 12 + cells))

    true_active_means = true_features[active_mean_indices]
    false_active_means = false_features[active_mean_indices]

    assert true_active_means.sum() == pytest.approx(false_active_means.sum())
    assert not np.allclose(true_active_means, false_active_means)
    assert false_features.shape == true_features.shape == (family_dim + pair_dim * 3 + 8,)


def test_present_trail_family_rejects_empty_pairset():
    from blockcipher_nd.features.spn_trail_family import present_pairset_trail_family_features

    cipher = build_cipher("present80", 7, key=0)

    with pytest.raises(ValueError, match="pairs must not be empty"):
        present_pairset_trail_family_features([], width=64, cipher=cipher)


def test_trail_family_smoke_config_preserves_official_protocol():
    config_path = Path("configs/experiment/innovation1/innovation1_spn_present_trail_family_smoke.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["run_id"] == "i1_trail_family_smoke_20260702"
    assert config["common"]["samples_per_class"] == 2
    assert config["common"]["pairs_per_sample"] == 1
    assert config["common"]["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert config["common"]["negative_mode"] == "encrypted_random_plaintexts"
    assert config["common"]["feature_route"] == "trail_family_consistency"
    assert config["common"]["validation_key"] == "0x11111111111111111111"
    assert config["common"]["key_rotation_interval"] == 0
    assert config["common"]["feature_cache_root"].startswith("outputs/local_smoke/")
    assert [row["model"] for row in config["rows"]] == [
        "present_nibble_invp_only_spn_only",
        "linear",
        "mlp",
        "false_family",
    ]


def test_active_auxiliary_smoke_config_preserves_official_protocol():
    config_path = Path("configs/experiment/innovation1/innovation1_spn_present_active_auxiliary_smoke.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["run_id"] == "i1_active_auxiliary_smoke_20260703"
    assert config["common"]["samples_per_class"] == 2
    assert config["common"]["pairs_per_sample"] == 1
    assert config["common"]["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert config["common"]["negative_mode"] == "encrypted_random_plaintexts"
    assert config["common"]["feature_route"] == "active_pattern_auxiliary_head"
    assert config["common"]["validation_key"] == "0x11111111111111111111"
    assert config["common"]["key_rotation_interval"] == 0
    assert config["common"]["lambda_aux"] == 0.1
    assert config["common"]["dataset_cache_root"].startswith("outputs/local_smoke/")
    assert config["common"]["dataset_cache_chunk_size"] == 2
    assert config["common"]["dataset_cache_workers"] == 1
    assert config["common"]["progress_output"].startswith("outputs/local_smoke/")
    assert [row["model"] for row in config["rows"]] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_active_aux_spn_only",
        "present_nibble_invp_active_aux_shuffled_targets",
    ]


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
    assert config["common"]["feature_route"] == "bit_transition_spectrum"
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


def test_trail_family_dataset_cache_writes_and_reuses(tmp_path):
    progress_path = tmp_path / "trail_family_progress.jsonl"
    features, labels = make_trail_family_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=17,
        samples_per_class=3,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_official_mcnd",
        key_rotation_interval=0,
        beam_width=4,
        depth=3,
        feature_cache_root=tmp_path / "trail_family_cache",
        feature_cache_chunk_size=2,
        feature_cache_workers=1,
        progress_output=progress_path,
        split="train",
    )
    reused_features, reused_labels = make_trail_family_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=17,
        samples_per_class=3,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_official_mcnd",
        key_rotation_interval=0,
        beam_width=4,
        depth=3,
        feature_cache_root=tmp_path / "trail_family_cache",
        feature_cache_chunk_size=2,
        feature_cache_workers=1,
        progress_output=progress_path,
        split="train",
    )

    assert features.shape == (6, 347)
    assert labels.shape == (6,)
    assert set(np.unique(labels).tolist()) == {0, 1}
    assert np.array_equal(np.asarray(features), np.asarray(reused_features))
    assert np.array_equal(np.asarray(labels), np.asarray(reused_labels))
    progress_text = progress_path.read_text(encoding="utf-8")
    assert "trail_family_cache_flush_start" in progress_text
    assert "trail_family_cache_done" in progress_text
    assert "trail_family_cache_reuse" in progress_text
    metadata_path = next((tmp_path / "trail_family_cache").glob("train/*/metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "feature_cache_workers" not in metadata
    assert metadata["cache_version"] == 2
    assert '"workers": 1' in progress_text


def test_trail_family_cache_reuses_across_worker_counts(tmp_path):
    progress_path = tmp_path / "trail_family_worker_reuse_progress.jsonl"
    cache_root = tmp_path / "trail_family_worker_reuse_cache"
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
        "beam_width": 4,
        "depth": 3,
        "feature_cache_root": cache_root,
        "feature_cache_chunk_size": 2,
        "progress_output": progress_path,
        "split": "train",
    }

    features, labels = make_trail_family_dataset(**common, feature_cache_workers=1)
    reused_features, reused_labels = make_trail_family_dataset(**common, feature_cache_workers=2)

    assert np.array_equal(np.asarray(features), np.asarray(reused_features))
    assert np.array_equal(np.asarray(labels), np.asarray(reused_labels))
    progress_text = progress_path.read_text(encoding="utf-8")
    assert '"workers": 1' in progress_text
    assert "trail_family_cache_reuse" in progress_text
    assert len(list(cache_root.glob("train/*/metadata.json"))) == 1


def test_active_auxiliary_dataset_cache_writes_and_reuses(tmp_path):
    progress_path = tmp_path / "active_auxiliary_progress.jsonl"
    features, labels = make_active_auxiliary_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=17,
        samples_per_class=3,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_official_mcnd",
        key_rotation_interval=0,
        dataset_cache_root=tmp_path / "active_auxiliary_cache",
        dataset_cache_chunk_size=2,
        dataset_cache_workers=1,
        progress_output=progress_path,
        split="train",
    )
    reused_features, reused_labels = make_active_auxiliary_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=17,
        samples_per_class=3,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_official_mcnd",
        key_rotation_interval=0,
        dataset_cache_root=tmp_path / "active_auxiliary_cache",
        dataset_cache_chunk_size=2,
        dataset_cache_workers=1,
        progress_output=progress_path,
        split="train",
    )

    assert features.shape == (6, 256)
    assert labels.shape == (6,)
    assert set(np.unique(labels).tolist()) == {0, 1}
    assert np.array_equal(np.asarray(features), np.asarray(reused_features))
    assert np.array_equal(np.asarray(labels), np.asarray(reused_labels))
    progress_text = progress_path.read_text(encoding="utf-8")
    assert "active_auxiliary_cache_flush_start" in progress_text
    assert "active_auxiliary_cache_done" in progress_text
    assert "active_auxiliary_cache_reuse" in progress_text
    metadata_path = next((tmp_path / "active_auxiliary_cache").glob("train/*/metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["cache_type"] == "spn_active_auxiliary_raw"
    assert metadata["feature_route"] == "active_pattern_auxiliary_head"
    assert "dataset_cache_workers" not in metadata
    assert '"workers": 1' in progress_text


def test_active_auxiliary_evaluation_uses_bounded_batches():
    class RecordingActiveAuxModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_batch_sizes: list[int] = []
            self.aux_batch_sizes: list[int] = []

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            self.forward_batch_sizes.append(int(features.shape[0]))
            return features[:, :1]

        def active_mask_logits(self, features: torch.Tensor) -> torch.Tensor:
            self.aux_batch_sizes.append(int(features.shape[0]))
            pairs_per_sample = features.shape[1] // 128
            return torch.zeros((features.shape[0], pairs_per_sample, 16), device=features.device)

    model = RecordingActiveAuxModel()
    features = np.zeros((5, 256), dtype=np.float32)
    labels = np.array([0, 1, 0, 1, 0], dtype=np.uint8)

    logits, aux_loss = _evaluate_active_aux_model(
        model,
        features,
        labels,
        lambda_aux=0.1,
        batch_size=2,
        shuffled_targets=False,
        seed=17,
        device=torch.device("cpu"),
    )

    assert logits.shape == (5,)
    assert aux_loss >= 0.0
    assert model.forward_batch_sizes == [2, 2, 1]
    assert model.aux_batch_sizes == [2, 2, 1]


def test_active_auxiliary_training_writes_epoch_progress(tmp_path):
    progress_path = tmp_path / "active_auxiliary_train_progress.jsonl"
    features = np.zeros((6, 256), dtype=np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)

    _model, aux_loss = _train_active_aux_model(
        features,
        labels,
        model_name="present_nibble_invp_active_aux_spn_only",
        hidden_bits=4,
        spn_mixer_depth=1,
        epochs=2,
        learning_rate=0.001,
        lambda_aux=0.1,
        batch_size=2,
        shuffled_targets=False,
        seed=17,
        device=torch.device("cpu"),
        progress_output=progress_path,
    )

    records = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    assert aux_loss >= 0.0
    assert records[0]["event"] == "active_auxiliary_train_start"
    assert records[0]["train_rows"] == 6
    train_batches = [record for record in records if record["event"] == "train_batch"]
    assert train_batches
    assert train_batches[-1]["model"] == "present_nibble_invp_active_aux_spn_only"
    assert train_batches[-1]["epoch"] == 2
    assert train_batches[-1]["epochs"] == 2
    assert train_batches[-1]["step"] == 3
    assert train_batches[-1]["steps_per_epoch"] == 3
    assert train_batches[-1]["train_rows_seen"] == 6
    assert train_batches[-1]["train_rows"] == 6
    assert train_batches[-1]["train_rows_progress_percent"] == pytest.approx(100.0)
    epoch_events = [record for record in records if record["event"] == "active_auxiliary_epoch_end"]
    assert epoch_events[-1]["epoch"] == 2
    assert "train_loss" in epoch_events[-1]
    assert "auxiliary_loss" in epoch_events[-1]


def test_trail_family_matrix_outputs_anchor_and_candidate_rows(tmp_path):
    config = tmp_path / "trail_family_matrix.json"
    output = tmp_path / "trail_family_matrix.jsonl"
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
                    {"model": "false_family", "progress_output": str(tmp_path / "false_family_progress.jsonl")},
                ],
            }
        ),
        encoding="utf-8",
    )

    spn_trail_family_matrix.main(["--config", str(config)])

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["model"] for row in rows] == [
        "present_nibble_invp_only_spn_only",
        "trail_family_consistency_linear",
        "trail_family_consistency_mlp",
        "trail_family_consistency_false_family",
    ]
    assert rows[0]["row_type"] == "external_anchor"
    assert rows[1]["feature_route"] == "trail_family_consistency"
    assert rows[1]["feature_dim"] == 347
    assert rows[3]["false_family"] is True
    assert rows[1]["auc"] == rows[1]["val_auc"]
    assert rows[2]["calibrated_accuracy"] == rows[2]["metrics"]["calibrated_accuracy"]


def test_active_auxiliary_matrix_outputs_anchor_candidate_and_control_rows(tmp_path):
    config = tmp_path / "active_auxiliary_matrix.json"
    output = tmp_path / "active_auxiliary_matrix.jsonl"
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
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "lambda_aux": 0.1,
                    "batch_size": 4,
                    "hidden_bits": 4,
                    "dataset_cache_root": str(tmp_path / "active_aux_cache"),
                    "dataset_cache_chunk_size": 2,
                    "dataset_cache_workers": 1,
                    "progress_output": str(tmp_path / "active_aux_progress.jsonl"),
                    "device": "cpu",
                },
                "rows": [
                    {
                        "row_type": "external_anchor",
                        "model": "present_nibble_invp_only_spn_only",
                        "anchor_auc": 0.52,
                        "anchor_calibrated_accuracy": 0.5,
                    },
                    {"model": "present_nibble_invp_active_aux_spn_only"},
                    {"model": "present_nibble_invp_active_aux_shuffled_targets"},
                ],
            }
        ),
        encoding="utf-8",
    )

    spn_active_auxiliary_matrix.main(["--config", str(config)])

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["model"] for row in rows] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_active_aux_spn_only",
        "present_nibble_invp_active_aux_shuffled_targets",
    ]
    assert rows[0]["row_type"] == "external_anchor"
    assert rows[1]["feature_route"] == "active_pattern_auxiliary_head"
    assert rows[1]["dataset_cache_enabled"] is True
    assert rows[1]["dataset_cache_root"] == str(tmp_path / "active_aux_cache")
    assert rows[1]["auxiliary_target"] == "present_invp_active_mask"
    assert rows[2]["auxiliary_target"] == "shuffled_present_invp_active_mask"
    assert rows[2]["shuffled_auxiliary_targets"] is True
    assert rows[1]["auc"] == rows[1]["val_auc"]
    assert rows[1]["calibrated_accuracy"] == rows[1]["metrics"]["calibrated_accuracy"]
    assert "auxiliary_loss" in rows[1]["metrics"]


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


def test_present_pairset_aggregation_control_seed1_remote_launch_assets_are_stage_aware():
    launcher = Path(
        "configs/remote/generated/"
        "run_i1_pairset_aggregation_control_r7_262k_seed1_gpu1_20260702.cmd"
    )
    monitor = Path(
        "configs/remote/generated/"
        "monitor_i1_pairset_aggregation_control_r7_262k_seed1_gpu1_20260702.sh"
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
    assert "i1_pairset_single_pair_scorer_r7_262k_seed1_gpu1_20260702" in launcher_text
    assert "i1_pairset_aggregation_control_r7_262k_seed1_gpu1_20260702" in launcher_text
    assert "--checkpoint-output \"%CHECKPOINT_DIR%\\single_pair_invp.pt\"" in launcher_text
    assert "innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k_seed1.csv" in launcher_text
    assert "innovation1_spn_present_pairset_aggregation_control_r7_262k_seed1.csv" in launcher_text
    assert "scripts\\evaluate-pairset-aggregation" in launcher_text
    assert "--scorer-pairs-per-sample 1" in launcher_text
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


def test_present_r8_pairset_aggregation_control_remote_launch_assets_are_stage_aware():
    launcher = Path(
        "configs/remote/generated/"
        "run_i1_pairset_aggregation_control_r8_262k_seed0_gpu0_20260705.cmd"
    )
    monitor = Path(
        "configs/remote/generated/"
        "monitor_i1_pairset_aggregation_control_r8_262k_seed0_gpu0_20260705.sh"
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
    assert "i1_pairset_single_pair_scorer_r8_262k_seed0_gpu0_20260705" in launcher_text
    assert "i1_pairset_aggregation_control_r8_262k_seed0_gpu0_20260705" in launcher_text
    assert "--checkpoint-output \"%CHECKPOINT_DIR%\\single_pair_invp.pt\"" in launcher_text
    assert "innovation1_spn_present_pairset_aggregation_control_single_pair_r8_262k.csv" in launcher_text
    assert "innovation1_spn_present_pairset_aggregation_control_r8_262k.csv" in launcher_text
    assert "scripts\\evaluate-pairset-aggregation" in launcher_text
    assert "--scorer-pairs-per-sample 1" in launcher_text
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


def test_active_auxiliary_medium_remote_launch_assets_are_gated_and_path_safe():
    launcher = Path(
        "configs/remote/generated/"
        "run_i1_active_auxiliary_r7_262k_seed0_gpu1_20260703.cmd"
    )
    monitor = Path(
        "configs/remote/generated/"
        "monitor_i1_active_auxiliary_r7_262k_seed0_gpu1_20260703.sh"
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
    assert "scripts\\spn-active-auxiliary-matrix" in launcher_text
    assert "innovation1_spn_present_active_auxiliary_r7_262k_seed0.json" in launcher_text
    assert "ACTIVE_AUXILIARY_CACHE_ROOT=G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\active_auxiliary_cache" in launcher_text
    assert "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703" in launcher_text
    assert "stale_failed_marker_ignored" in monitor_text
    assert "failed_marker_is_stale" in monitor_text
    assert "EXPECTED_ROWS=\"3\"" in monitor_text
    assert "postprocess-active-auxiliary" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text
    assert "completed_missing_or_incomplete_results" in monitor_text
    artifacts = launch_artifacts(
        Path("configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_20260703.json")
    )
    assert artifacts["status"] == "pass"
    assert artifacts["launcher"] == str(launcher)
    assert artifacts["monitor"] == str(monitor)


def test_trail_family_seed1_confirmation_launcher_runs_readiness_gate():
    launcher = Path(
        "configs/remote/generated/"
        "run_i1_trail_family_r7_262k_seed1_gpu1_20260702.cmd"
    )
    monitor = Path(
        "configs/remote/generated/"
        "monitor_i1_trail_family_r7_262k_seed1_gpu1_20260702.sh"
    )
    launcher_text = launcher.read_text(encoding="utf-8")
    monitor_text = monitor.read_text(encoding="utf-8")

    assert "cmd.exe /k" not in launcher_text
    assert "cmd.exe /k" not in monitor_text
    assert "scripts\\check-remote-readiness" in launcher_text
    assert "innovation1_spn_present_trail_family_r7_262k_seed1_gpu1_20260702.json" in launcher_text
    assert "scripts\\spn-trail-family-matrix" in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "C:\\Users" not in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "EXPECTED_ROWS=\"4\"" in monitor_text
    assert "postprocess-trail-family" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text

    artifacts = launch_artifacts(
        Path("configs/remote/innovation1_spn_present_trail_family_r7_262k_seed1_gpu1_20260702.json")
    )
    assert artifacts["status"] == "pass"
    assert artifacts["launcher"] == str(launcher)
    assert artifacts["monitor"] == str(monitor)


def test_active_auxiliary_seed1_confirmation_assets_are_ready_and_path_safe():
    plan = Path("configs/experiment/innovation1/innovation1_spn_present_active_auxiliary_r7_262k_seed1.json")
    remote_config = Path(
        "configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed1_gpu1_20260703.json"
    )
    launcher = Path(
        "configs/remote/generated/"
        "run_i1_active_auxiliary_r7_262k_seed1_gpu1_20260703.cmd"
    )
    monitor = Path(
        "configs/remote/generated/"
        "monitor_i1_active_auxiliary_r7_262k_seed1_gpu1_20260703.sh"
    )

    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    remote_data = json.loads(remote_config.read_text(encoding="utf-8"))
    launcher_text = launcher.read_text(encoding="utf-8")
    monitor_text = monitor.read_text(encoding="utf-8")

    assert plan_data["run_id"] == "i1_active_auxiliary_r7_262k_seed1_gpu1_20260703"
    assert plan_data["common"]["seed"] == 1
    assert plan_data["common"]["samples_per_class"] == 262144
    assert plan_data["common"]["negative_mode"] == "encrypted_random_plaintexts"
    assert plan_data["common"]["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert plan_data["common"]["validation_key"] == "0x11111111111111111111"
    assert plan_data["common"]["dataset_cache_root"] == (
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\active_auxiliary_cache"
    )
    assert [row["model"] for row in plan_data["rows"]] == [
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_active_aux_spn_only",
        "present_nibble_invp_active_aux_shuffled_targets",
    ]

    assert remote_data["expected_rows"] == 3
    assert remote_data["runner_script"] == "scripts/spn-active-auxiliary-matrix"
    assert remote_data["plan"].endswith("innovation1_spn_present_active_auxiliary_r7_262k_seed1.json")
    assert "confirmation" in remote_data["claim_scope"]
    assert "cmd.exe /c" in remote_data["launch_policy"]
    assert "G:\\lxy" in remote_data["launch_policy"]

    assert "cmd.exe /k" not in launcher_text
    assert "cmd.exe /k" not in monitor_text
    assert "scripts\\check-remote-readiness" in launcher_text
    assert "scripts\\spn-active-auxiliary-matrix" in launcher_text
    assert "innovation1_spn_present_active_auxiliary_r7_262k_seed1.json" in launcher_text
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in launcher_text
    assert "C:\\Users" not in launcher_text
    assert "Desktop" not in launcher_text
    assert "Downloads" not in launcher_text
    assert "AppData" not in launcher_text
    assert "stale_failed_marker_ignored" in monitor_text
    assert "failed_marker_is_stale" in monitor_text
    assert "EXPECTED_ROWS=\"3\"" in monitor_text
    assert "postprocess-active-auxiliary" in monitor_text
    assert "--update-plan-doc \"${PLAN_DOC}\"" in monitor_text

    artifacts = launch_artifacts(remote_config)
    assert artifacts["status"] == "pass"
    assert artifacts["launcher"] == str(launcher)
    assert artifacts["monitor"] == str(monitor)


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
        Path("scripts/spn-trail-family-matrix"),
        Path("scripts/spn-active-pattern"),
        Path("scripts/audit-spn-features"),
        Path("scripts/validate-results"),
        Path("scripts/gate-invp-result"),
        Path("scripts/gate-candidate-trail"),
        Path("scripts/gate-transition-spectrum"),
        Path("scripts/gate-trail-family"),
        Path("scripts/postprocess-invp-result"),
        Path("scripts/postprocess-candidate-trail"),
        Path("scripts/postprocess-transition-spectrum"),
        Path("scripts/postprocess-trail-family"),
        Path("scripts/monitor-health"),
        Path("scripts/check-remote-readiness"),
        Path("scripts/plan-next-action"),
        Path("scripts/arbitrate-next-actions"),
        Path("scripts/summarize-spn-evidence"),
        Path("scripts/watch-high-round"),
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
                        "time": 1000.0,
                        "rows_done": 81920,
                        "total_rows": 524288,
                        "class_rows_done": 81920,
                        "class_total": 262144,
                        "chunk_rows": 8192,
                    }
                ),
                json.dumps(
                    {
                        "event": "candidate_cache_positive_chunk",
                        "time": 1064.0,
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
    assert "main_thread_policy" in report["active_recommendation"]
    assert "launch candidate-trail seed1" in report["active_recommendation"]["main_thread_policy"][
        "forbidden_until_gate"
    ]
    assert "wait for the watcher or sub-agent to retrieve result artifacts" in report["active_recommendation"][
        "main_thread_policy"
    ]["allowed_actions"]
    assert "heartbeat" in report["active_recommendation"]
    assert "needs_main_thread_intervention" in report["active_recommendation"]
    assert report["active_recommendation"]["progress_summary"]["cache_class_rows_done"] == 114688
    assert report["active_recommendation"]["progress_summary"]["cache_chunk_rows"] == 8192
    assert report["active_recommendation"]["progress_summary"]["cache_chunk_index"] == 14
    assert report["active_recommendation"]["progress_summary"]["cache_class_chunk_index"] == 14
    assert report["active_recommendation"]["progress_summary"]["line_count"] == 3
    assert report["active_recommendation"]["progress_summary"]["parsed_line_count"] == 3
    assert report["active_recommendation"]["progress_summary"]["cache_class_progress_percent"] == 43.75
    assert report["active_recommendation"]["progress_summary"]["cache_total_progress_percent"] == 21.875
    assert report["active_recommendation"]["progress_summary"]["cache_rows_per_second"] == pytest.approx(512.0)
    assert report["active_recommendation"]["progress_summary"]["cache_rate_window_seconds"] == pytest.approx(64.0)
    assert report["active_recommendation"]["progress_summary"]["cache_rate_window_rows"] == 32768
    assert report["active_recommendation"]["progress_summary"]["cache_eta_seconds"] == 800
    followup = report["active_recommendation"]["conditional_followup"]
    assert followup["branch"] == "candidate_trail_seed1_confirmation_or_variance_check"
    assert followup["run_id"] == "i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702"
    assert followup["launch_gate"] == "support_candidate_trail_route or weak_candidate_trail_signal"
    assert followup["readiness"]["status"] == "pass"
    assert followup["readiness_pass"] is True
    assert followup["should_launch_now"] is False
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


def test_summarize_spn_evidence_tracks_running_high_round_runs(tmp_path):
    root = tmp_path / "remote_results"
    run_root = root / "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705"
    (run_root / "monitor").mkdir(parents=True)
    (run_root / "logs").mkdir()
    (run_root / "monitor" / "monitor.log").write_text("2026-07-05T10:42:16+08:00 running\n", encoding="utf-8")
    (run_root / "logs" / "r9_weak_probe_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "train_batch",
                "model": "present_nibble_invp_only_spn_only",
                "epoch": 8,
                "epochs": 30,
                "best_checkpoint_metric": 0.5004787916550413,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "wait_for_high_round_results"
    assert active["should_launch_remote"] is False
    assert active["active_runs"][0]["run_id"] == "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705"
    assert active["active_runs"][0]["postprocess_allowed"] is False
    assert "scripts/monitor-health" in active["active_runs"][0]["monitor_health_command"]
    assert "scripts/postprocess-r9-weak-probe" in active["active_runs"][0]["postprocess_when_ready_command"]
    assert "launch r8/r9/r10 follow-up branches" in active["main_thread_policy"]["forbidden_until_gate"]


def test_summarize_spn_evidence_routes_ready_high_round_result_to_postprocess(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705"
    run_root = root / run_id
    (run_root / "monitor").mkdir(parents=True)
    (run_root / "results").mkdir()
    (run_root / "monitor" / "monitor.log").write_text("2026-07-05T12:00:00+08:00 running\n", encoding="utf-8")
    results = run_root / "results" / f"{run_id}.jsonl"
    _write_r9_weak_probe_result(results, "present_zhang_wang_keras_mcnd", 0.511)
    _write_r9_weak_probe_result(results, "present_nibble_invp_only_spn_only", 0.519)
    _write_r9_weak_probe_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.556)

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "postprocess_high_round_result"
    assert active["ready_runs"][0]["run_id"] == run_id
    assert active["ready_runs"][0]["postprocess_allowed"] is True
    assert "scripts/postprocess-r9-weak-probe" in active["ready_runs"][0]["postprocess_when_ready_command"]
    assert "run each listed high-round postprocess command" in active["main_thread_policy"]["allowed_actions"]


def test_advance_high_round_waits_without_touching_running_result(tmp_path):
    root = tmp_path / "remote_results"
    run_root = root / "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705"
    (run_root / "monitor").mkdir(parents=True)
    (run_root / "logs").mkdir()
    (run_root / "monitor" / "monitor.log").write_text("2026-07-05T10:42:16+08:00 running\n", encoding="utf-8")
    (run_root / "logs" / "r9_weak_probe_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "train_batch",
                "model": "present_nibble_invp_only_spn_only",
                "epoch": 8,
                "epochs": 30,
                "best_checkpoint_metric": 0.5004787916550413,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = advance_high_round(root=root, arbitration_output=tmp_path / "arbitration.json", update_plan_docs=False)

    assert report["status"] == "waiting"
    assert report["initial_branch"] == "wait_for_high_round_results"
    assert report["postprocessed"] == []
    assert report["arbitration"] is None
    assert report["remote_policy"] == "local_artifacts_only_no_ssh_no_remote_launch"


def test_watch_high_round_one_shot_writes_waiting_report(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "i1_present_r8_pairset_1m_seed0_gpu1_20260705"
    run_root = root / run_id
    (run_root / "monitor").mkdir(parents=True)
    (run_root / "logs").mkdir()
    (run_root / "results").mkdir()
    (run_root / "monitor" / "monitor.log").write_text(
        "2026-07-05T12:00:00+08:00 sync\n"
        "2026-07-05T12:00:01+08:00 running\n",
        encoding="utf-8",
    )
    (run_root / "logs" / "r8_pairset_1m_progress.jsonl").write_text(
        json.dumps({"event": "train_batch", "epoch": 1, "epochs": 30}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_root / "results" / f"{run_id}.jsonl").write_text("", encoding="utf-8")
    output = tmp_path / "watch_report.json"

    report = watch_high_round(
        root=root,
        output=output,
        arbitration_output=tmp_path / "arbitration.json",
        interval_seconds=0,
        max_iterations=1,
        update_plan_docs=False,
    )

    assert report["status"] == "waiting"
    assert report["watch_iteration"] == 1
    assert report["watch_policy"] == "local_artifacts_only_no_ssh_no_remote_launch"
    assert report["active_recommendation"]["branch"] == "wait_for_high_round_results"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "waiting"


def test_advance_high_round_postprocesses_ready_r8_without_plan_doc_update(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "i1_present_r8_pairset_1m_seed0_gpu1_20260705"
    run_root = root / run_id
    (run_root / "monitor").mkdir(parents=True)
    (run_root / "results").mkdir()
    (run_root / "monitor" / "monitor.log").write_text("2026-07-05T12:00:00+08:00 running\n", encoding="utf-8")
    results = run_root / "results" / f"{run_id}.jsonl"
    _write_r8_pairset_1m_result(results, "present_zhang_wang_keras_mcnd", 0.542)
    _write_r8_pairset_1m_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.549)

    report = advance_high_round(root=root, arbitration_output=tmp_path / "arbitration.json", update_plan_docs=False)

    assert report["status"] == "postprocessed"
    assert report["initial_branch"] == "postprocess_high_round_result"
    assert report["postprocessed"][0]["status"] == "pass"
    assert report["postprocessed"][0]["decision"] == "support_r8_pairset_1m_confirmation"
    assert (run_root / f"{run_id}_postprocess_summary.json").exists()
    assert (run_root / f"{run_id}_candidate_route_readiness.json").exists()
    assert report["arbitration"] is None


def test_advance_high_round_arbitrates_existing_r8_and_r9_summaries(tmp_path):
    root = tmp_path / "remote_results"
    r9_results = tmp_path / "r9_weak_probe.jsonl"
    _write_r9_weak_probe_result(r9_results, "present_zhang_wang_keras_mcnd", 0.511)
    _write_r9_weak_probe_result(r9_results, "present_nibble_invp_only_spn_only", 0.519)
    _write_r9_weak_probe_result(r9_results, "present_nibble_invp_pair_consistency_spn_only", 0.556)
    postprocess_r9_weak_probe_result(
        results_path=r9_results,
        output_dir=root / "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705",
        run_id="i1_present_r9_weak_probe_262k_seed0_gpu0_20260705",
        expected_rows=3,
    )
    r8_results = tmp_path / "r8_pairset_1m.jsonl"
    _write_r8_pairset_1m_result(r8_results, "present_zhang_wang_keras_mcnd", 0.542)
    _write_r8_pairset_1m_result(r8_results, "present_nibble_invp_pair_consistency_spn_only", 0.549)
    postprocess_r8_pairset_1m_result(
        results_path=r8_results,
        output_dir=root / "i1_present_r8_pairset_1m_seed0_gpu1_20260705",
        run_id="i1_present_r8_pairset_1m_seed0_gpu1_20260705",
        expected_rows=2,
    )
    arbitration_output = tmp_path / "high_round_next_action_arbitration.json"

    report = advance_high_round(root=root, arbitration_output=arbitration_output, update_plan_docs=False)

    assert report["status"] == "arbitrated"
    assert report["initial_branch"] == "arbitrate_high_round_next_actions"
    assert report["arbitration"]["status"] == "written"
    assert report["arbitration"]["summary_count"] == 2
    assert arbitration_output.exists()
    arbitration = json.loads(arbitration_output.read_text(encoding="utf-8"))
    assert arbitration["status"] == "selected"
    assert arbitration["selected"]["branch"] == "r9_1m_seed0_plan"


def test_summarize_spn_evidence_recommends_high_round_arbitration_after_summaries(tmp_path):
    root = tmp_path / "remote_results"
    r9_results = tmp_path / "r9_weak_probe.jsonl"
    _write_r9_weak_probe_result(r9_results, "present_zhang_wang_keras_mcnd", 0.511)
    _write_r9_weak_probe_result(r9_results, "present_nibble_invp_only_spn_only", 0.519)
    _write_r9_weak_probe_result(r9_results, "present_nibble_invp_pair_consistency_spn_only", 0.556)
    r9_report = postprocess_r9_weak_probe_result(
        results_path=r9_results,
        output_dir=root / "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705",
        run_id="i1_present_r9_weak_probe_262k_seed0_gpu0_20260705",
        expected_rows=3,
    )
    r8_results = tmp_path / "r8_pairset_1m.jsonl"
    _write_r8_pairset_1m_result(r8_results, "present_zhang_wang_keras_mcnd", 0.542)
    _write_r8_pairset_1m_result(r8_results, "present_nibble_invp_pair_consistency_spn_only", 0.549)
    r8_report = postprocess_r8_pairset_1m_result(
        results_path=r8_results,
        output_dir=root / "i1_present_r8_pairset_1m_seed0_gpu1_20260705",
        run_id="i1_present_r8_pairset_1m_seed0_gpu1_20260705",
        expected_rows=2,
    )

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "arbitrate_high_round_next_actions"
    assert active["summary_count"] == 2
    assert r9_report["summary"] in active["summaries"]
    assert r8_report["summary"] in active["summaries"]
    assert "scripts/arbitrate-next-actions" in active["arbitration_command"]
    assert "launch multiple high-round follow-up branches in parallel" in active["main_thread_policy"][
        "forbidden_until_gate"
    ]


def test_summarize_spn_evidence_exposes_nested_high_round_metrics(tmp_path):
    root = tmp_path / "remote_results"
    r9_results = tmp_path / "r9_weak_probe_near_random.jsonl"
    _write_r9_weak_probe_result(r9_results, "present_zhang_wang_keras_mcnd", 0.504)
    _write_r9_weak_probe_result(r9_results, "present_nibble_invp_only_spn_only", 0.503)
    _write_r9_weak_probe_result(r9_results, "present_nibble_invp_pair_consistency_spn_only", 0.502)
    postprocess_r9_weak_probe_result(
        results_path=r9_results,
        output_dir=root / "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705",
        run_id="i1_present_r9_weak_probe_262k_seed0_gpu0_20260705",
        expected_rows=3,
    )

    report = summarize_spn_evidence(root)

    route = {
        route["run_id"]: route
        for route in report["routes"]
    }["i1_present_r9_weak_probe_262k_seed0_gpu0_20260705"]
    assert route["metrics"]["baseline_auc"] == pytest.approx(0.504)
    assert route["metrics"]["best_candidate_auc"] == pytest.approx(0.503)
    assert route["metrics"]["best_overall_auc"] == pytest.approx(0.504)
    assert route["metrics"]["candidate_delta_vs_baseline_auc"] == pytest.approx(-0.001)
    assert route["metrics"]["best_candidate_model"] == "present_nibble_invp_only_spn_only"


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


def test_summarize_spn_evidence_uses_seed1_candidate_plan_for_seed1_run(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702"
    candidate = root / run_id
    (candidate / "monitor").mkdir(parents=True)
    (candidate / "logs").mkdir()
    (candidate / "monitor" / "monitor.log").write_text("2026-07-02T18:10:00+08:00 running\n")
    (candidate / "logs" / "candidate_trail_linear_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "candidate_cache_positive_chunk",
                "rows_done": 8192,
                "total_rows": 524288,
                "class_rows_done": 8192,
                "class_total": 262144,
                "chunk_rows": 8192,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "wait_for_candidate_trail_result"
    assert active["run_id"] == run_id
    assert "monitor_i1_candidate_trail_seed1_20260702" in active["monitor_health_command"]
    assert "candidate_trail_consistency_r7_262k_seed1.json" in active["monitor_health_command"]
    assert "candidate_trail_consistency_r7_262k_seed1.json" in active["postprocess_when_ready_command"]


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
    assert "main_thread_policy" in active
    assert "run the listed postprocess_when_ready_command" in active["main_thread_policy"]["allowed_actions"]
    assert "launch bit-transition-spectrum seed0" in active["main_thread_policy"]["forbidden_until_gate"]


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
    assert active["next_action"]["should_launch_remote"] is True
    assert active["next_action"]["requires_implementation"] is False
    assert active["next_action"]["next_plan_doc"] == "docs/experiments/innovation1-bit-transition-spectrum-plan.md"
    assert "bit_transition_spectrum_r7_262k_seed0" in active["next_action"]["suggested_plan_config"]
    assert "bit_transition_spectrum_r7_262k_seed0" in active["next_action"]["launch_remote_config"]
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
    assert "`cache_rows_remaining`" in text
    assert "`cache_class_rows_remaining`" in text


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

    assert "DDT graph, topology-aware, r9 weak-probe, and r8 pair-set 1M postprocess" in text
    assert "Candidate-trail, bit-transition-spectrum, and trail-family postprocess" in text
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
    assert "`conditional_followup`" in text
    assert "`readiness_pass`" in text
    assert "`should_launch_now = false`" in text
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


def test_ddt_graph_gate_stops_when_best_control_matches_true_graph(tmp_path):
    results_path = tmp_path / "ddt_graph_control_matches.jsonl"
    _write_ddt_graph_result_set(
        results_path,
        invp_auc=0.7940,
        transition_no_ddt_auc=0.7945,
        no_ddt_graph_auc=0.7960,
        ddt_auc=0.7960,
        shuffled_auc=0.7948,
    )

    report = gate_ddt_graph_result(results_path)

    assert report["status"] == "pass"
    assert report["decision"] == "stop_ddt_graph_route"
    assert report["margin_vs_best_control_auc"] == 0


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


def test_topology_aware_gate_stops_when_shuffled_control_matches_true_graph(tmp_path):
    results_path = tmp_path / "topology_aware_shuffled_matches.jsonl"
    _write_topology_aware_result_set(
        results_path,
        invp_auc=0.7940,
        true_graph_auc=0.7945,
        shuffled_auc=0.7945,
    )

    report = gate_topology_aware_result(results_path)

    assert report["status"] == "pass"
    assert report["decision"] == "stop_topology_aware_network_route"
    assert report["margin_vs_invp_auc"] > 0
    assert report["margin_vs_shuffled_auc"] == 0


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
    artifacts = report["readiness_reports"][0]["launch_artifacts"]
    assert artifacts["status"] == "pass"
    assert artifacts["launcher"].endswith("run_i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630.cmd")
    assert artifacts["monitor"].endswith("monitor_i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630.sh")
    assert "generated launcher" in " ".join(report["launch_checklist"])


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
    assert report["launch_checklist"][0] == (
        "Do not launch until all remote readiness reports and generated launch artifacts pass."
    )
    assert report["readiness_reports"][0]["readiness"]["status"] == "fail"
    assert "missing.json" in report["errors"][0]


def test_plan_next_action_reports_missing_generated_launch_artifacts(tmp_path):
    config = json.loads(
        Path("configs/remote/innovation1_spn_present_ddt_graph_r7_262k_seed1_gpu1_20260630.json").read_text(
            encoding="utf-8"
        )
    )
    config["monitor_script_name"] = "monitor_unit_missing_generated_artifacts.sh"
    remote_config = tmp_path / "remote.json"
    remote_config.write_text(json.dumps(config) + "\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "run_id": "unit_summary",
                "decision": "unit_ready",
                "action": "launch_unit",
                "next_action": {
                    "branch": "unit_branch",
                    "should_launch_remote": True,
                    "requires_implementation": False,
                    "launch_remote_config": str(remote_config),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = plan_next_action(summary)

    assert report["status"] == "fail"
    assert report["remote_readiness_pass"] is True
    assert report["launch_artifacts_pass"] is False
    assert report["readiness_pass"] is False
    artifacts = report["readiness_reports"][0]["launch_artifacts"]
    assert artifacts["status"] == "fail"
    assert "generated monitor script missing" in " ".join(artifacts["errors"])
    assert "launch artifacts failed" in " ".join(report["errors"])
    assert "Fix generated launch artifacts before launch" in " ".join(report["launch_checklist"])


def test_plan_next_action_maps_spn_followup_branches_to_plans(tmp_path):
    cases = [
        (
            "transition_spectrum_seed1_confirmation",
            "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
            "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json",
        ),
        (
            "transition_spectrum_variance_check",
            "docs/experiments/innovation1-bit-transition-spectrum-plan.md",
            "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json",
        ),
        (
            "trail_family_seed1_confirmation",
            "docs/experiments/innovation1-trail-family-consistency-plan.md",
            "innovation1_spn_present_trail_family_r7_262k_seed1.json",
        ),
        (
            "trail_family_variance_check",
            "docs/experiments/innovation1-trail-family-consistency-plan.md",
            "innovation1_spn_present_trail_family_r7_262k_seed1.json",
        ),
        (
            "stop_transition_spectrum_route",
            "docs/experiments/innovation1-trail-family-consistency-plan.md",
            "innovation1_spn_present_trail_family_r7_262k_seed0.json",
        ),
        (
            "stop_trail_family_route",
            "docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md",
            "innovation1_spn_present_active_auxiliary_r7_262k_seed0.json",
        ),
    ]
    for branch, expected_plan_doc, expected_plan_config in cases:
        summary = tmp_path / f"{branch}.json"
        summary.write_text(
            json.dumps(
                {
                    "run_id": f"unit_{branch}",
                    "decision": branch,
                    "action": "unit",
                    "next_action": {
                        "branch": branch,
                        "should_launch_remote": False,
                        "requires_implementation": True,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        report = plan_next_action(summary)

        checklist = " ".join(report["implementation_checklist"])
        assert report["status"] == "pass"
        assert report["branch"] == branch
        assert report["requires_implementation"] is True
        assert expected_plan_doc in checklist
        assert expected_plan_config in checklist


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
                "2026-06-29T14:46:00+08:00 running",
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
    assert report["heartbeat"]["seconds_until_stale"] == 1502
    assert report["heartbeat"]["estimated_sync_interval_seconds"] == 842
    assert report["heartbeat"]["next_expected_sync_timestamp"] == "2026-06-29T15:14:04+08:00"
    assert report["tmux_interpretation"]["state"] == "not_checked"
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
    assert report["scp_stderr_resolved_missing_artifacts"] is False
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
    assert report["scp_stderr_resolved_missing_artifacts"] is False
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
    assert report["scp_stderr_persistent_missing_artifacts"] is False
    assert report["scp_stderr_resolved_missing_artifacts"] is True

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


def test_monitor_health_accepts_train_matrix_result_file(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_matrix_run"
    run_root = root / run_id
    monitor = run_root / "monitor"
    results = run_root / "results"
    logs = run_root / "logs"
    monitor.mkdir(parents=True)
    results.mkdir()
    logs.mkdir()
    (monitor / "monitor.log").write_text(
        "\n".join(
            [
                "2026-07-06T19:00:00+08:00 sync",
                "2026-07-06T19:00:01+08:00 running",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    (logs / f"{run_id}_train_done.marker").write_text("train_done\n", encoding="utf-8")
    (results / "train_matrix.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=2,
        now=datetime.fromisoformat("2026-07-06T19:05:00+08:00"),
    )

    assert report["status"] == "result_ready"
    assert report["results_jsonl"].endswith("results/train_matrix.jsonl")
    assert report["results_jsonl_exists"] is True
    assert report["results_jsonl_line_count"] == 2
    assert report["expected_rows"] == 2


def test_monitor_health_does_not_treat_train_done_as_final_done(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_matrix_run"
    run_root = root / run_id
    monitor = run_root / "monitor"
    logs = run_root / "logs"
    monitor.mkdir(parents=True)
    logs.mkdir()
    (monitor / "monitor.log").write_text(
        "\n".join(
            [
                "2026-07-06T19:00:00+08:00 sync",
                "2026-07-06T19:00:01+08:00 running",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    (logs / f"{run_id}_train_done.marker").write_text("train_done\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=2,
        now=datetime.fromisoformat("2026-07-06T19:05:00+08:00"),
    )

    assert report["status"] == "running"
    assert report["done_markers"] == []
    assert report["needs_main_thread_intervention"] is False


def test_monitor_health_ignores_stale_failed_marker_when_progress_continues(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "stale_failed_marker_unit"
    run_root = root / run_id
    monitor = run_root / "monitor"
    logs = run_root / "logs"
    monitor.mkdir(parents=True)
    logs.mkdir()
    (monitor / "monitor.log").write_text(
        "\n".join(
            [
                "2026-07-04T10:05:00+08:00 failed",
                "2026-07-04T10:18:00+08:00 sync",
                "2026-07-04T10:18:01+08:00 running",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    failed = logs / f"{run_id}_failed.marker"
    failed.write_text("failed\n", encoding="utf-8")
    progress = logs / "active_auxiliary_progress.jsonl"
    progress.write_text(
        json.dumps(
            {
                "event": "active_auxiliary_positive_chunk",
                "rows_done": 49152,
                "total_rows": 524288,
                "class_rows_done": 49152,
                "class_total": 262144,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    old = datetime.fromisoformat("2026-07-04T10:05:00+08:00").timestamp()
    new = datetime.fromisoformat("2026-07-04T10:18:00+08:00").timestamp()
    failed.touch()
    progress.touch()
    import os

    os.utime(failed, (old, old))
    os.utime(progress, (new, new))

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=3,
        now=datetime.fromisoformat("2026-07-04T10:20:00+08:00"),
    )

    assert report["status"] == "running"
    assert report["needs_main_thread_intervention"] is False
    assert report["failed_markers"] == []
    assert report["stale_failed_markers"] == [f"logs/{run_id}_failed.marker"]


def test_monitor_health_ignores_stale_launch_failed_marker_after_started_artifact(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "stale_launch_unit"
    run_root = root / run_id
    monitor = run_root / "monitor"
    logs = run_root / "logs"
    monitor.mkdir(parents=True)
    logs.mkdir()
    (monitor / "monitor.log").write_text(
        "\n".join(
            [
                "2026-07-07T16:29:49+08:00 launch_failed status=255",
                "2026-07-07T20:20:43+08:00 running missing=18",
                "2026-07-07T20:34:43+08:00 sync",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    failed = monitor / "launch_failed.marker"
    failed.write_text("failed\n", encoding="utf-8")
    started = logs / f"{run_id}_started.marker"
    started.write_text("started\n", encoding="utf-8")
    git_revision = logs / f"{run_id}_git_revision.txt"
    git_revision.write_text("962f524\n", encoding="utf-8")
    import os

    failed_time = datetime.fromisoformat("2026-07-07T16:29:49+08:00").timestamp()
    started_time = datetime.fromisoformat("2026-07-07T20:34:44+08:00").timestamp()
    os.utime(failed, (failed_time, failed_time))
    os.utime(started, (started_time, started_time))
    os.utime(git_revision, (started_time, started_time))

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=3,
        now=datetime.fromisoformat("2026-07-07T20:45:00+08:00"),
    )

    assert report["status"] == "running"
    assert report["needs_main_thread_intervention"] is False
    assert report["failed_markers"] == []
    assert report["stale_failed_markers"] == ["monitor/launch_failed.marker"]


def test_monitor_health_reports_launched_revision_lag(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "revision_lag_unit"
    run_root = root / run_id
    monitor = run_root / "monitor"
    logs = run_root / "logs"
    monitor.mkdir(parents=True)
    logs.mkdir()
    monitor.joinpath("monitor.log").write_text(
        "2026-07-08T11:00:00+08:00 running missing=18\n",
        encoding="utf-8",
    )
    old_head = subprocess.run(
        ["git", "rev-list", "--max-count=1", "HEAD~1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    logs.joinpath(f"{run_id}_git_revision.txt").write_text(old_head + "\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        now=datetime.fromisoformat("2026-07-08T11:05:00+08:00"),
    )

    assert report["source_revision"]["launched_commit"] == old_head
    assert report["source_revision"]["current_head"] == current_head
    assert report["source_revision"]["revision_lag"] == {
        "status": "behind_current_head",
        "commits_behind": 1,
    }
    assert report["launch_state"]["has_git_revision"] is True


def test_monitor_health_reports_command_marker_progress(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "command_marker_unit"
    run_root = root / run_id
    monitor = run_root / "monitor"
    logs = run_root / "logs"
    monitor.mkdir(parents=True)
    logs.mkdir()
    monitor.joinpath("monitor.log").write_text(
        "2026-07-08T11:00:00+08:00 running missing=18\n",
        encoding="utf-8",
    )
    for index in [0, 1, 12]:
        logs.joinpath(f"{run_id}_command_{index}.marker").write_text(
            f"command_{index}\n",
            encoding="utf-8",
        )

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        now=datetime.fromisoformat("2026-07-08T11:05:00+08:00"),
    )

    assert report["command_markers"] == {
        "marker_count": 3,
        "command_indices": [0, 1, 12],
        "latest_command_index": 12,
        "latest_marker": f"logs/{run_id}_command_12.marker",
    }
    assert report["status"] == "running"


def test_present_r8_residual_bucket_plan_documents_source_selected_pool3_handoff():
    residual_plan = Path("docs/experiments/innovation1-present-r8-residual-bucket-axis-spectrum-plan.md")
    pool_plan = Path("docs/experiments/innovation1-present-diverse-expert-pool-plan.md")

    residual_text = residual_plan.read_text(encoding="utf-8")
    pool_text = pool_plan.read_text(encoding="utf-8")

    assert "source-selected residual expert handoff" in residual_text
    assert "train-only source selection" in residual_text
    assert "do_not_launch_new_remote_branch_while_residual_focus_262k_pending" in residual_text
    assert "residual_focus_source_selected_aux" in pool_text
    assert "trail_position + raw117 + source_selected_residual_focus" in pool_text


def test_present_r8_residual_bucket_plan_documents_retry1_wait_diagnosis():
    residual_plan = Path("docs/experiments/innovation1-present-r8-residual-bucket-axis-spectrum-plan.md")
    residual_text = residual_plan.read_text(encoding="utf-8")

    assert "2026-07-08 Bounded Status Check" in residual_text
    assert "single_heavy_dataset_cache_stage" in residual_text
    assert "not because many remote" in residual_text
    assert "training jobs are competing" in residual_text
    assert "`running missing=18` means the 18 planned" in residual_text
    assert "do_not_restart_while_monitor_health_is_running" in residual_text


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


def test_monitor_health_marks_launch_stalled_after_clone_without_readiness(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "clone_stalled_unit"
    run_root = root / run_id
    monitor = run_root / "monitor"
    logs = run_root / "logs"
    monitor.mkdir(parents=True)
    logs.mkdir()
    (monitor / "monitor.log").write_text(
        "\n".join(
            [
                "2026-07-04T20:38:28+08:00 sync",
                "2026-07-04T20:38:29+08:00 running",
                "2026-07-04T20:52:28+08:00 sync",
                "2026-07-04T20:52:29+08:00 running",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / f"{run_id}_gpu_info.txt").write_text("gpu\n", encoding="utf-8")
    (logs / f"{run_id}_launch_env.txt").write_text("run_id=clone_stalled_unit\n", encoding="utf-8")
    (logs / f"{run_id}_torch_info.txt").write_text("cuda available\n", encoding="utf-8")
    (logs / f"{run_id}_torch_info_stderr.txt").write_text("", encoding="utf-8")
    (logs / f"{run_id}_git_clone_stdout.txt").write_text("", encoding="utf-8")
    (logs / f"{run_id}_git_clone_stderr.txt").write_text("Cloning into source...\n", encoding="utf-8")

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=4,
        now=datetime.fromisoformat("2026-07-04T20:53:00+08:00"),
    )

    assert report["status"] == "launch_stalled"
    assert report["needs_main_thread_intervention"] is True
    assert report["launch_state"]["is_stalled"] is True
    assert report["launch_state"]["reason"] == "clone_or_checkout_before_readiness"

    (logs / f"{run_id}_git_revision.txt").write_text("abc123\n", encoding="utf-8")
    (logs / f"{run_id}_readiness.txt").write_text('{"status":"pass"}\n', encoding="utf-8")
    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=4,
        now=datetime.fromisoformat("2026-07-04T20:53:00+08:00"),
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

    trail_family_run = "i1_trail_family_result_ready"
    trail_family_root = root / trail_family_run
    trail_family_monitor = trail_family_root / "monitor"
    trail_family_monitor.mkdir(parents=True)
    (trail_family_monitor / "monitor.log").write_text(
        "2026-07-02T20:05:00+08:00 running\n",
        encoding="utf-8",
    )
    (trail_family_monitor / "monitor_ssh_stderr.log").write_text("", encoding="utf-8")
    trail_family_results = trail_family_root / "results"
    trail_family_results.mkdir()
    trail_family_jsonl = trail_family_results / f"{trail_family_run}.jsonl"
    trail_family_jsonl.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    trail_family_plan = (
        Path("configs/experiment/innovation1/")
        / "innovation1_spn_present_trail_family_smoke.json"
    )
    trail_family_doc = tmp_path / "trail-family-plan.md"

    report = monitor_health_report(
        run_id=trail_family_run,
        root=root,
        plan_path=trail_family_plan,
        plan_doc_paths=[trail_family_doc],
        postprocess_kind="trail_family",
    )

    assert report["status"] == "result_ready"
    assert report["expected_rows"] == 4
    assert report["postprocess_allowed"] is True
    assert report["postprocess_command"] == [
        "env",
        "UV_CACHE_DIR=/tmp/uv-cache",
        "MPLCONFIGDIR=/tmp/mplconfig",
        "uv",
        "run",
        "python",
        "scripts/postprocess-trail-family",
        "--plan",
        str(trail_family_plan),
        "--results",
        str(trail_family_jsonl),
        "--output-dir",
        str(trail_family_root),
        "--run-id",
        trail_family_run,
        "--expected-rows",
        "4",
        "--update-plan-doc",
        str(trail_family_doc),
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
    assert progress["cache_rows_remaining"] == 458752
    assert progress["cache_class_rows_done"] == 65536
    assert progress["cache_class_total"] == 262144
    assert progress["cache_class_rows_remaining"] == 196608
    assert progress["cache_chunk_rows"] == 8192
    assert progress["cache_chunk_index"] == 8
    assert progress["cache_class_chunk_index"] == 8
    assert progress["cache_total_progress_percent"] == pytest.approx(12.5)
    assert progress["cache_class_progress_percent"] == pytest.approx(25.0)
    assert progress["cache_rows_per_second"] == pytest.approx(512.0)
    assert progress["cache_rate_window_seconds"] == pytest.approx(64.0)
    assert progress["cache_rate_window_rows"] == 32768
    assert progress["cache_eta_seconds"] == 896


def test_monitor_health_reports_feature_cache_progress_from_command_stdout(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_cache_progress_stdout"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-07T21:19:31+08:00 running\n", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    (logs / f"{run_id}_started.marker").write_text("started\n", encoding="utf-8")
    (logs / f"{run_id}_command_0_stdout.txt").write_text(
        "warming up\n"
        + json.dumps(
            {
                "event": "cache_positive_chunk",
                "stage": "dataset_cache",
                "split": "train",
                "time": 1783439971.0,
                "rows_done": 147456,
                "total_rows": 524288,
                "class_rows_done": 147456,
                "class_total": 262144,
                "chunk_rows": 8192,
                "samples_per_class": 262144,
                "model": "present_trail_position_stats_pairset",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=18,
        now=datetime.fromisoformat("2026-07-07T21:20:00+08:00"),
    )

    progress = report["progress_summary"]
    assert report["status"] == "running"
    assert progress["exists"] is True
    assert progress["path"].endswith(f"{run_id}_command_0_stdout.txt")
    assert progress["source_kind"] == "stdout_json_progress"
    assert progress["latest_event"] == "cache_positive_chunk"
    assert progress["stage"] == "dataset_cache"
    assert progress["latest_split"] == "train"
    assert progress["model"] == "present_trail_position_stats_pairset"
    assert progress["cache_rows_done"] == 147456
    assert progress["cache_total_rows"] == 524288
    assert progress["cache_class_rows_done"] == 147456
    assert progress["cache_class_total"] == 262144
    assert progress["cache_total_progress_percent"] == pytest.approx(28.125)
    assert progress["cache_class_progress_percent"] == pytest.approx(56.25)


def test_monitor_health_reports_progress_from_extra_progress_root(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_external_progress_root"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-08T08:55:55+08:00 running\n", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    (logs / f"{run_id}_started.marker").write_text("started\n", encoding="utf-8")
    artifact_root = tmp_path / "local_audits" / "unit_external_progress_root"
    progress_dir = artifact_root / "seed0" / "dataset_cache"
    progress_dir.mkdir(parents=True)
    (progress_dir / "seed0_train_feature_export_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "cache_positive_chunk",
                "stage": "dataset_cache",
                "split": "train",
                "time": 1783472000.0,
                "rows_done": 262144,
                "total_rows": 524288,
                "class_rows_done": 262144,
                "class_total": 262144,
                "chunk_rows": 8192,
                "samples_per_class": 262144,
                "model": "present_trail_position_stats_pairset",
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "event": "cache_negative_chunk",
                "stage": "dataset_cache",
                "split": "train",
                "time": 1783472300.0,
                "rows_done": 401408,
                "total_rows": 524288,
                "class_rows_done": 139264,
                "class_total": 262144,
                "chunk_rows": 8192,
                "samples_per_class": 262144,
                "model": "present_trail_position_stats_pairset",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=18,
        progress_roots=[artifact_root],
        now=datetime.fromisoformat("2026-07-08T09:00:00+08:00"),
    )

    progress = report["progress_summary"]
    assert report["status"] == "running"
    assert progress["exists"] is True
    assert progress["source_kind"] == "external_progress_jsonl"
    assert progress["path"].endswith("seed0_train_feature_export_progress.jsonl")
    assert progress["latest_event"] == "cache_negative_chunk"
    assert progress["latest_split"] == "train"
    assert progress["cache_rows_done"] == 401408
    assert progress["cache_total_rows"] == 524288
    assert progress["cache_total_progress_percent"] == pytest.approx(76.562)
    assert progress["cache_class_rows_done"] == 139264
    assert progress["cache_class_total"] == 262144
    assert progress["cache_class_progress_percent"] == pytest.approx(53.125)


def test_monitor_health_carries_cache_dimension_workload_from_start_event(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_cache_dimension_workload"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-08T12:29:07+08:00 running\n", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    (logs / "trail_position_cache_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "cache_start",
                "stage": "dataset_cache",
                "split": "train",
                "time": 1783405102.0,
                "input_bits": 39936,
                "pairs_per_sample": 16,
                "samples_per_class": 262144,
                "total_rows": 524288,
                "model": "present_trail_position_stats_pairset",
                "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "event": "cache_negative_chunk",
                "stage": "dataset_cache",
                "split": "train",
                "time": 1783483938.0,
                "rows_done": 475136,
                "total_rows": 524288,
                "class_rows_done": 212992,
                "class_total": 262144,
                "chunk_rows": 8192,
                "samples_per_class": 262144,
                "model": "present_trail_position_stats_pairset",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=18,
        now=datetime.fromisoformat("2026-07-08T12:30:00+08:00"),
    )

    progress = report["progress_summary"]
    assert progress["latest_event"] == "cache_negative_chunk"
    assert progress["cache_input_bits"] == 39936
    assert progress["cache_pairs_per_sample"] == 16
    assert progress["cache_pair_bits"] == 2496
    assert progress["cache_total_feature_bits"] == 20_937_965_568
    assert progress["cache_rows_done_feature_bits"] == 18_975_031_296
    assert progress["cache_total_feature_bytes"] == pytest.approx(2_617_245_696.0)
    assert progress["cache_rows_done_feature_bytes"] == pytest.approx(2_371_878_912.0)


def test_monitor_health_reports_positive_and_negative_cache_class_progress(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_cache_polarity_progress"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-06T23:53:22+08:00 running\n", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    (logs / "trail_position_beamstats_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "cache_positive_chunk",
                "split": "train",
                "time": 1000.0,
                "rows_done": 262144,
                "total_rows": 524288,
                "class_rows_done": 262144,
                "class_total": 262144,
                "chunk_rows": 8192,
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "event": "cache_negative_chunk",
                "split": "train",
                "time": 1120.0,
                "rows_done": 294912,
                "total_rows": 524288,
                "class_rows_done": 32768,
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
        expected_rows=2,
        now=datetime.fromisoformat("2026-07-06T23:54:00+08:00"),
    )

    progress = report["progress_summary"]
    assert report["status"] == "running"
    assert progress["latest_event"] == "cache_negative_chunk"
    assert progress["cache_total_progress_percent"] == pytest.approx(56.25)
    assert progress["cache_class_progress_percent"] == pytest.approx(12.5)
    assert progress["cache_positive_class_rows_done"] == 262144
    assert progress["cache_positive_class_total"] == 262144
    assert progress["cache_positive_class_rows_remaining"] == 0
    assert progress["cache_positive_class_progress_percent"] == pytest.approx(100.0)
    assert progress["cache_negative_class_rows_done"] == 32768
    assert progress["cache_negative_class_total"] == 262144
    assert progress["cache_negative_class_rows_remaining"] == 229376
    assert progress["cache_negative_class_progress_percent"] == pytest.approx(12.5)


def test_monitor_health_reports_latest_progress_event_age_independent_of_file_mtime(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_progress_event_age"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-02T16:26:00+08:00 running\n", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    now = datetime.fromisoformat("2026-07-02T16:26:00+08:00")
    event_time = now.timestamp() - 60
    progress_path = logs / "unit_progress.jsonl"
    progress_path.write_text(
        json.dumps(
            {
                "event": "train_batch",
                "time": event_time,
                "epoch": 3,
                "epochs": 30,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=1,
        now=now,
    )

    progress = report["progress_summary"]
    assert progress["latest_event"] == "train_batch"
    assert progress["latest_event_time"] == datetime.fromtimestamp(event_time, tz=timezone.utc).isoformat()
    assert progress["latest_event_age_seconds"] == 60
    assert progress["age_seconds"] < progress["latest_event_age_seconds"]


def test_monitor_health_keeps_latest_cache_progress_when_new_cache_starts(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_trail_family_validation_cache_started"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-03T19:14:35+08:00 running\n", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    (logs / "trail_family_linear_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "trail_family_positive_chunk",
                "split": "train",
                "time": 1000.0,
                "rows_done": 8192,
                "total_rows": 524288,
                "class_rows_done": 8192,
                "class_total": 262144,
                "chunk_rows": 8192,
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "event": "trail_family_negative_chunk",
                "split": "train",
                "time": 2024.0,
                "rows_done": 524288,
                "total_rows": 524288,
                "class_rows_done": 262144,
                "class_total": 262144,
                "chunk_rows": 8192,
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "event": "trail_family_cache_start",
                "split": "validation",
                "time": 2025.0,
                "total_rows": 131072,
                "samples_per_class": 65536,
                "chunk_size": 8192,
                "workers": 4,
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
        now=datetime.fromisoformat("2026-07-03T19:15:00+08:00"),
    )

    progress = report["progress_summary"]
    assert progress["latest_event"] == "trail_family_cache_start"
    assert progress["latest_split"] == "validation"
    assert progress["latest_total_rows"] == 131072
    assert progress["latest_samples_per_class"] == 65536
    assert progress["cache_event"] == "trail_family_negative_chunk"
    assert progress["cache_split"] == "train"
    assert progress["cache_rows_done"] == 524288
    assert progress["cache_total_rows"] == 524288
    assert progress["cache_rows_remaining"] == 0
    assert progress["cache_class_rows_done"] == 262144
    assert progress["cache_class_progress_percent"] == pytest.approx(100.0)
    assert progress["cache_rows_per_second"] == pytest.approx(504.0)
    assert progress["cache_rate_window_seconds"] == pytest.approx(1024.0)
    assert progress["cache_rate_window_rows"] == 516096
    assert progress["cache_eta_seconds"] == 0


def test_monitor_health_uses_latest_progress_event_time_across_files(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_trail_family_multi_progress_files"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-03T19:14:35+08:00 running\n", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    (logs / "trail_family_false_family_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "trail_family_cache_start",
                "split": "train",
                "time": 3000.0,
                "total_rows": 524288,
                "samples_per_class": 262144,
                "false_family": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # This file is written later, so its mtime is newer, but its internal event
    # time is older than the false-family progress above.
    (logs / "trail_family_mlp_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "trail_family_cache_reuse",
                "split": "validation",
                "time": 2000.0,
                "total_rows": 131072,
                "samples_per_class": 65536,
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
        now=datetime.fromisoformat("2026-07-03T19:15:00+08:00"),
    )

    progress = report["progress_summary"]
    assert progress["path"].endswith("trail_family_false_family_progress.jsonl")
    assert progress["latest_event"] == "trail_family_cache_start"
    assert progress["latest_split"] == "train"
    assert progress["latest_total_rows"] == 524288
    assert progress["latest_samples_per_class"] == 262144


def test_monitor_health_cache_rate_stays_with_latest_split(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_trail_family_validation_rate"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-03T19:28:36+08:00 running\n", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    (logs / "trail_family_linear_progress.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "trail_family_negative_chunk",
                        "split": "train",
                        "time": 1000.0,
                        "rows_done": 524288,
                        "total_rows": 524288,
                        "class_rows_done": 262144,
                        "class_total": 262144,
                        "chunk_rows": 8192,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "event": "trail_family_positive_chunk",
                        "split": "validation",
                        "time": 2000.0,
                        "rows_done": 8192,
                        "total_rows": 131072,
                        "class_rows_done": 8192,
                        "class_total": 65536,
                        "chunk_rows": 8192,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "event": "trail_family_positive_chunk",
                        "split": "validation",
                        "time": 2020.0,
                        "rows_done": 16384,
                        "total_rows": 131072,
                        "class_rows_done": 16384,
                        "class_total": 65536,
                        "chunk_rows": 8192,
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=4,
        now=datetime.fromisoformat("2026-07-03T19:29:00+08:00"),
    )

    progress = report["progress_summary"]
    assert progress["latest_split"] == "validation"
    assert progress["cache_split"] == "validation"
    assert progress["cache_rows_done"] == 16384
    assert progress["cache_rows_per_second"] == pytest.approx(409.6)
    assert progress["cache_rate_window_seconds"] == pytest.approx(20.0)
    assert progress["cache_rate_window_rows"] == 8192
    assert progress["cache_eta_seconds"] == 280


def test_monitor_health_reports_progress_file_freshness(tmp_path):
    root = tmp_path / "remote_results"
    run_id = "unit_progress_file_freshness"
    monitor = root / run_id / "monitor"
    monitor.mkdir(parents=True)
    (monitor / "monitor.log").write_text("2026-07-02T16:25:00+00:00 running\n", encoding="utf-8")
    logs = root / run_id / "logs"
    logs.mkdir()
    progress_path = logs / "candidate_trail_linear_progress.jsonl"
    progress_path.write_text(
        json.dumps(
            {
                "event": "candidate_cache_positive_chunk",
                "rows_done": 8192,
                "total_rows": 131072,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    modified = datetime.fromisoformat("2026-07-02T16:00:00+00:00").timestamp()
    progress_path.touch()
    import os

    os.utime(progress_path, (modified, modified))

    report = monitor_health_report(
        run_id=run_id,
        root=root,
        expected_rows=4,
        stale_after_seconds=1800,
        now=datetime.fromisoformat("2026-07-02T16:45:01+00:00"),
    )

    progress = report["progress_summary"]
    assert report["status"] == "running"
    assert progress["path"] == str(progress_path)
    assert progress["mtime"] == "2026-07-02T16:00:00+00:00"
    assert progress["age_seconds"] == 2701
    assert progress["stale_after_seconds"] == 1800
    assert progress["is_stale"] is True


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
    assert progress["cache_rows_remaining"] == 458752
    assert progress["cache_class_rows_remaining"] == 196608
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
    assert report["heartbeat"]["seconds_until_stale"] == 0
    assert report["heartbeat"]["estimated_sync_interval_seconds"] is None
    assert report["heartbeat"]["next_expected_sync_timestamp"] is None
    assert report["needs_main_thread_intervention"] is True
    assert report["postprocess_allowed"] is False
    assert report["postprocess_command"] == []


def test_monitor_health_does_not_treat_tmux_check_error_as_missing_session():
    tmux = {
        "checked": True,
        "session": "unit",
        "exists": None,
        "returncode": 1,
        "stderr": "error connecting to /tmp/tmux-1000/default (Operation not permitted)",
        "check_error": True,
    }
    heartbeat = {"is_stale": False}
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
        heartbeat=heartbeat,
        tmux=tmux,
    )

    assert status == "running"
    from blockcipher_nd.cli.monitor_health import _tmux_interpretation

    interpretation = _tmux_interpretation(tmux, heartbeat, status)
    assert interpretation["state"] == "check_error_but_heartbeat_fresh"
    assert "do not treat this as a stopped watcher" in interpretation["message"]


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


def test_trail_position_medium_remote_readiness_uses_disk_cache_and_progress():
    config_path = Path(
        "configs/remote/"
        "innovation1_spn_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    report = remote_readiness_report(config_path)

    assert report["status"] == "pass"
    assert report["expected_rows"] == 2
    assert report["plan_rows"] == 2
    assert report["max_samples_per_class"] == 65_536
    assert "medium_scale_dataset_cache" in report["checked_invariants"]
    assert "trail_position_score_artifact_lock" in report["checked_invariants"]
    assert config["dataset_cache"] is True
    assert config["dataset_cache_root"] == (
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\trail_position_beamstats_cache"
    )
    assert config["dataset_cache_chunk_size"] == 8192
    assert config["dataset_cache_workers"] == 4
    assert config["progress_output"].startswith(
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        "i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706\\logs\\"
    )
    assert config["checkpoint_output_dir"] == (
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        "i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706\\checkpoints"
    )
    assert config["score_export_after_training"] is True
    assert config["score_artifacts_root"] == (
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        "i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706\\score_artifacts"
    )
    assert config["score_export_models"] == [
        {
            "artifact_name": "global_stats_control",
            "candidate_status": "near_neighbor_control",
            "checkpoint_filename": "row0001_present_pairset_global_stats_seed0.pt",
            "eval_row_index": 0,
            "expert_family": "trail_position_global_control",
            "model_key": "present_pairset_global_stats",
        },
        {
            "artifact_name": "trail_position",
            "candidate_status": "weak_positive",
            "checkpoint_filename": "row0002_present_trail_position_stats_pairset_seed0.pt",
            "eval_row_index": 1,
            "expert_family": "trail_position",
            "model_key": "present_trail_position_stats_pairset",
        },
    ]
    assert "cmd.exe /c" in config["launch_policy"]
    assert "cmd.exe /k" not in config["launch_policy"]
    assert "prepared only" in config["launch_policy"]
    assert "do not launch" in config["launch_policy"]
    assert "not formal" in config["claim_scope"]


def test_remote_readiness_gate_rejects_sbox_prior_without_full_column_identity(tmp_path):
    source_plan = Path(
        "configs/experiment/innovation1/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0.csv"
    )
    bad_plan = tmp_path / "sbox_prior_missing_full_column.csv"
    bad_plan.write_text(
        source_plan.read_text(encoding="utf-8").replace(',""prior_feature_mode"":""full_column""', "", 1),
        encoding="utf-8",
    )
    config = json.loads(
        Path(
            "configs/remote/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0_gpu1_20260703.json"
        ).read_text(encoding="utf-8")
    )
    config["plan"] = str(bad_plan)
    config_path = tmp_path / "sbox_prior_missing_full_column_remote.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(config_path)

    assert report["status"] == "fail"
    assert any("prior_feature_mode=None expected=full_column" in error for error in report["errors"])


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


def test_bit_transition_spectrum_seed0_remote_config_is_ready_but_gate_locked():
    path = Path(
        "configs/remote/"
        "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["run_id"] == "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702"
    assert config["runner_script"] == "scripts/spn-transition-spectrum-matrix"
    assert config["expected_rows"] == 4
    assert config["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert config["negative_mode"] == "encrypted_random_plaintexts"
    assert config["validation_key"] == "0x11111111111111111111"
    assert config["key_rotation_interval"] == 0
    assert config["feature_cache_workers"] == 4
    assert config["dataset_cache_workers"] == 4
    assert "stop_candidate_trail_route" in config["launch_policy"]
    assert "cmd.exe /c" in config["launch_policy"]
    assert "cmd.exe /k" not in config["launch_policy"]

    report = remote_readiness_report(path)

    assert report["status"] == "pass"
    assert report["plan_rows"] == 4
    assert report["max_samples_per_class"] == 262144
    assert "transition_spectrum_protocol_lock" in report["checked_invariants"]
    assert "candidate_trail_protocol_lock" not in report["checked_invariants"]
    assert "medium_scale_dataset_cache" in report["checked_invariants"]


def test_bit_transition_spectrum_seed1_remote_config_is_ready_but_gate_locked():
    path = Path(
        "configs/remote/"
        "innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1_gpu1_20260702.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))

    assert config["run_id"] == "i1_bit_transition_spectrum_r7_262k_seed1_gpu1_20260702"
    assert config["plan"].endswith(
        "configs\\experiment\\innovation1\\innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json"
    )
    assert config["runner_script"] == "scripts/spn-transition-spectrum-matrix"
    assert config["expected_rows"] == 4
    assert config["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert config["negative_mode"] == "encrypted_random_plaintexts"
    assert config["validation_key"] == "0x11111111111111111111"
    assert config["key_rotation_interval"] == 0
    assert config["feature_cache_workers"] == 4
    assert config["dataset_cache_workers"] == 4
    assert "support_transition_spectrum_route or weak_transition_spectrum_signal" in config["launch_policy"]
    assert "cmd.exe /c" in config["launch_policy"]
    assert "cmd.exe /k" not in config["launch_policy"]

    plan = json.loads(
        Path("configs/experiment/innovation1/innovation1_spn_present_bit_transition_spectrum_r7_262k_seed1.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["common"]["seed"] == 1
    assert plan["common"]["samples_per_class"] == 262144
    assert [row.get("model") for row in plan["rows"]] == [
        "present_nibble_invp_only_spn_only",
        "linear",
        "mlp",
        "shuffled_p",
    ]

    report = remote_readiness_report(path)

    assert report["status"] == "pass"
    assert report["plan_rows"] == 4
    assert report["max_samples_per_class"] == 262144
    assert "transition_spectrum_protocol_lock" in report["checked_invariants"]
    assert "candidate_trail_protocol_lock" not in report["checked_invariants"]
    assert "medium_scale_dataset_cache" in report["checked_invariants"]


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


def _write_trail_family_remote_plan(
    tmp_path: Path,
    *,
    include_false_family: bool = True,
    filename: str = "trail_family_matrix.json",
) -> Path:
    rows = [
        {
            "row_type": "external_anchor",
            "model": "present_nibble_invp_only_spn_only",
            "anchor_auc": 0.79,
            "anchor_calibrated_accuracy": 0.72,
        },
        {"model": "linear"},
        {"model": "mlp"},
    ]
    if include_false_family:
        rows.append({"model": "false_family"})
    plan = tmp_path / filename
    plan.write_text(
        json.dumps(
            {
                "output": str(tmp_path / "trail_family_matrix.jsonl"),
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
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return plan


def _trail_family_remote_config(plan: Path, *, expected_rows: int = 4) -> dict[str, object]:
    return {
        "run_id": "i1_trail_family_matrix_remote_unit",
        "task_name": "i1_trail_family_matrix_remote_unit",
        "archive_work_id": "i1_trail_family_matrix_remote_unit",
        "plan": str(plan),
        "runner_script": "scripts/spn-trail-family-matrix",
        "expected_rows": expected_rows,
        "device": "cuda:0",
        "epochs": 20,
        "batch_size": 2048,
        "learning_rate": 0.003,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "negative_mode": "encrypted_random_plaintexts",
        "validation_key": "0x11111111111111111111",
        "key_rotation_interval": 0,
        "dataset_cache": True,
        "dataset_cache_root": "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\trail_family_cache",
        "dataset_cache_chunk_size": 8192,
        "dataset_cache_workers": 4,
        "feature_cache_root": "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\trail_family_cache",
        "feature_cache_workers": 4,
        "branch": "main",
        "repo_url": "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git",
        "source_commit": "recorded_in_remote_run_script_git_revision",
        "result_sync": "local_tmux_monitor_scp_fallback",
        "monitor_script_name": "monitor_i1_trail_family_matrix_remote_unit.sh",
        "claim_scope": "trail-family medium matrix readiness unit",
        "launch_policy": "trail-family matrix; keep artifacts under G:\\lxy; cmd.exe /c",
    }


def _write_active_auxiliary_remote_plan(tmp_path: Path, *, include_control: bool = True) -> Path:
    rows = [
        {
            "row_type": "external_anchor",
            "model": "present_nibble_invp_only_spn_only",
            "anchor_auc": 0.79,
            "anchor_calibrated_accuracy": 0.72,
        },
        {"model": "present_nibble_invp_active_aux_spn_only"},
    ]
    if include_control:
        rows.append({"model": "present_nibble_invp_active_aux_shuffled_targets"})
    plan = tmp_path / "active_auxiliary_matrix.json"
    plan.write_text(
        json.dumps(
            {
                "output": str(tmp_path / "active_auxiliary_matrix.jsonl"),
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
                    "learning_rate": 0.001,
                    "lambda_aux": 0.1,
                },
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return plan


def _active_auxiliary_remote_config(plan: Path, *, expected_rows: int = 3) -> dict[str, object]:
    return {
        "run_id": "i1_active_auxiliary_matrix_remote_unit",
        "task_name": "i1_active_auxiliary_matrix_remote_unit",
        "archive_work_id": "i1_active_auxiliary_matrix_remote_unit",
        "plan": str(plan),
        "runner_script": "scripts/spn-active-auxiliary-matrix",
        "expected_rows": expected_rows,
        "device": "cuda:1",
        "epochs": 20,
        "batch_size": 2048,
        "learning_rate": 0.001,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "negative_mode": "encrypted_random_plaintexts",
        "validation_key": "0x11111111111111111111",
        "key_rotation_interval": 0,
        "dataset_cache": True,
        "dataset_cache_root": "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\active_auxiliary_cache",
        "dataset_cache_chunk_size": 8192,
        "dataset_cache_workers": 4,
        "branch": "main",
        "repo_url": "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git",
        "source_commit": "recorded_in_remote_run_script_git_revision",
        "result_sync": "local_tmux_monitor_scp_fallback",
        "monitor_script_name": "monitor_i1_active_auxiliary_matrix_remote_unit.sh",
        "claim_scope": "active-auxiliary medium matrix readiness unit",
        "launch_policy": "active-auxiliary matrix; keep artifacts under G:\\lxy; cmd.exe /c",
    }


def test_remote_readiness_gate_accepts_active_auxiliary_matrix_plan(tmp_path):
    plan = _write_active_auxiliary_remote_plan(tmp_path)
    config = _active_auxiliary_remote_config(plan)
    path = tmp_path / "active_auxiliary_matrix_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "pass"
    assert "medium_scale_dataset_cache" in report["checked_invariants"]
    assert "active_auxiliary_protocol_lock" in report["checked_invariants"]


def test_remote_readiness_gate_rejects_active_auxiliary_without_control_or_runner(tmp_path):
    plan = _write_active_auxiliary_remote_plan(tmp_path, include_control=False)
    config = _active_auxiliary_remote_config(plan, expected_rows=2)
    config["runner_script"] = "scripts/train"
    config["negative_mode"] = "random_ciphertexts"
    config["dataset_cache_root"] = "C:\\Users\\bad\\active_auxiliary_cache"
    path = tmp_path / "active_auxiliary_bad_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "fail"
    assert "active_auxiliary_protocol_lock" in report["checked_invariants"]
    assert any("runner_script=scripts/spn-active-auxiliary-matrix" in error for error in report["errors"])
    assert any("active_auxiliary negative_mode=random_ciphertexts" in error for error in report["errors"])
    assert any("true active-auxiliary and shuffled-target control rows" in error for error in report["errors"])
    assert any("dataset_cache_root must stay under" in error for error in report["errors"])


def test_remote_readiness_gate_accepts_trail_family_matrix_plan(tmp_path):
    plan = _write_trail_family_remote_plan(tmp_path)
    config = _trail_family_remote_config(plan)
    path = tmp_path / "trail_family_matrix_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "pass"
    assert report["plan_rows"] == 4
    assert report["expected_rows"] == 4
    assert "trail_family_protocol_lock" in report["checked_invariants"]
    assert "transition_spectrum_protocol_lock" not in report["checked_invariants"]
    assert "candidate_trail_protocol_lock" not in report["checked_invariants"]
    assert "medium_scale_dataset_cache" in report["checked_invariants"]


def test_prepared_trail_family_seed0_remote_readiness_passes():
    report = remote_readiness_report(
        Path("configs/remote/innovation1_spn_present_trail_family_r7_262k_seed0_gpu1_20260702.json")
    )

    assert report["status"] == "pass"
    assert report["run_id"] == "i1_trail_family_r7_262k_seed0_gpu1_20260702"
    assert report["expected_rows"] == 4
    assert report["plan_rows"] == 4
    assert report["max_samples_per_class"] == 262144
    assert "trail_family_protocol_lock" in report["checked_invariants"]
    assert "medium_scale_dataset_cache" in report["checked_invariants"]


def test_prepared_trail_family_seed1_remote_readiness_passes():
    report = remote_readiness_report(
        Path("configs/remote/innovation1_spn_present_trail_family_r7_262k_seed1_gpu1_20260702.json")
    )

    assert report["status"] == "pass"
    assert report["run_id"] == "i1_trail_family_r7_262k_seed1_gpu1_20260702"
    assert report["expected_rows"] == 4
    assert report["plan_rows"] == 4
    assert report["max_samples_per_class"] == 262144
    assert "trail_family_protocol_lock" in report["checked_invariants"]
    assert "medium_scale_dataset_cache" in report["checked_invariants"]
    assert any("trail_family feature_cache_workers=4" in warning for warning in report["warnings"])


def test_prepared_active_auxiliary_seed0_readiness_notes_cache_worker_benchmark_scope():
    report = remote_readiness_report(
        Path("configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_20260703.json")
    )

    assert report["status"] == "pass"
    assert report["run_id"] == "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    assert "active_auxiliary_protocol_lock" in report["checked_invariants"]
    assert any("active_auxiliary dataset_cache_workers=4" in warning for warning in report["warnings"])


def test_remote_readiness_gate_rejects_bad_trail_family_matrix_plan(tmp_path):
    plan = _write_trail_family_remote_plan(tmp_path, include_false_family=False)
    config = _trail_family_remote_config(plan, expected_rows=3)
    config["runner_script"] = "scripts/spn-transition-spectrum-matrix"
    config["negative_mode"] = "random_ciphertexts"
    config["sample_structure"] = "zhang_wang_case2_mcnd"
    config["validation_key"] = "0xffffffffffffffffffff"
    config["key_rotation_interval"] = 1024
    config["feature_cache_root"] = "C:\\Users\\bad\\trail_family_cache"
    config["feature_cache_workers"] = 0
    path = tmp_path / "trail_family_bad_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert report["status"] == "fail"
    assert "trail_family_protocol_lock" in report["checked_invariants"]
    assert any("runner_script=scripts/spn-trail-family-matrix" in error for error in report["errors"])
    assert any("linear, mlp, and false_family rows" in error for error in report["errors"])
    assert any("trail_family negative_mode=random_ciphertexts expected=encrypted_random_plaintexts" in error for error in report["errors"])
    assert any("trail_family sample_structure=zhang_wang_case2_mcnd expected=zhang_wang_case2_official_mcnd" in error for error in report["errors"])
    assert any("trail_family validation_key=0xffffffffffffffffffff expected=0x11111111111111111111" in error for error in report["errors"])
    assert any("trail_family key_rotation_interval=1024 expected=0" in error for error in report["errors"])
    assert any("trail_family cache root must stay under G:\\lxy" in error for error in report["errors"])
    assert any("trail_family feature_cache_workers must be >= 1" in error for error in report["errors"])


def test_remote_readiness_gate_does_not_detect_trail_family_from_launch_policy_only(tmp_path):
    plan = _write_trail_family_remote_plan(tmp_path, filename="plain_matrix.json")
    config = _trail_family_remote_config(plan)
    config["run_id"] = "i1_plain_matrix_remote_unit"
    config["task_name"] = "i1_plain_matrix_remote_unit"
    config["archive_work_id"] = "i1_plain_matrix_remote_unit"
    config["runner_script"] = "scripts/train"
    config["claim_scope"] = "plain matrix readiness unit"
    config["route"] = "plain_matrix"
    config["experiment_route"] = "plain_matrix"
    config["launch_policy"] = (
        "plain matrix; a note may mention trail_family as a future branch; "
        "keep artifacts under G:\\lxy; cmd.exe /c"
    )
    path = tmp_path / "plain_matrix_remote.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = remote_readiness_report(path)

    assert "trail_family_protocol_lock" not in report["checked_invariants"]
    assert not any("trail_family" in error for error in report["errors"])


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


def test_candidate_trail_gate_stops_when_shuffled_control_matches_true_route(tmp_path):
    results = tmp_path / "candidate_trail_shuffled_matches.jsonl"
    _write_candidate_trail_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(results, "candidate_trail_consistency_linear", 0.7925)
    _write_candidate_trail_result(results, "candidate_trail_consistency_mlp", 0.7934)
    _write_candidate_trail_result(results, "candidate_trail_consistency_shuffled_cells", 0.7934)

    report = gate_candidate_trail_result(results, expected_rows=4, require_shuffled_control=True)

    assert report["status"] == "pass"
    assert report["decision"] == "stop_candidate_trail_route"
    assert report["margin_vs_anchor_auc"] > 0.001
    assert report["margin_vs_shuffled_auc"] == 0
    assert "shuffled-cell control matches/exceeds" in report["interpretation"]


def test_candidate_trail_gate_can_require_shuffled_control(tmp_path):
    results = tmp_path / "candidate_trail_missing_shuffled.jsonl"
    _write_candidate_trail_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(results, "candidate_trail_consistency_linear", 0.7925)
    _write_candidate_trail_result(results, "candidate_trail_consistency_mlp", 0.7940)

    diagnostic = gate_candidate_trail_result(results, expected_rows=3)
    assert diagnostic["status"] == "pass"
    assert diagnostic["decision"] == "weak_candidate_trail_signal"
    assert diagnostic["warnings"] == ["missing_shuffled_control=candidate_trail_consistency_shuffled_cells"]

    strict = gate_candidate_trail_result(results, expected_rows=3, require_shuffled_control=True)
    assert strict["status"] == "fail"
    assert strict["decision"] == "invalid"
    assert strict["errors"] == ["missing_shuffled_control=candidate_trail_consistency_shuffled_cells"]


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
    assert readiness["remote_readiness_pass"] is True
    assert readiness["launch_artifacts_pass"] is True
    assert readiness["readiness_reports"][0]["readiness"]["status"] == "pass"
    assert readiness["readiness_reports"][0]["launch_artifacts"]["status"] == "pass"
    assert readiness["readiness_reports"][0]["launch_artifacts"]["launcher"].endswith(
        "run_i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.cmd"
    )
    assert readiness["readiness_reports"][0]["launch_artifacts"]["monitor"].endswith(
        "monitor_i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.sh"
    )
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


def test_candidate_trail_postprocess_requires_shuffled_control_by_default(tmp_path):
    results = tmp_path / "candidate_trail_missing_shuffled.jsonl"
    _write_candidate_trail_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(results, "candidate_trail_consistency_linear", 0.7925)
    _write_candidate_trail_result(results, "candidate_trail_consistency_mlp", 0.7940)
    output_dir = tmp_path / "postprocess"

    report = postprocess_candidate_trail_result(
        results_path=results,
        output_dir=output_dir,
        run_id="candidate_trail_missing_shuffled_unit",
        expected_rows=3,
    )

    assert report["status"] == "fail"
    assert report["candidate_trail_status"] == "fail"
    assert report["decision"] == "invalid"
    assert report["require_shuffled_control"] is True
    gate = json.loads((output_dir / "candidate_trail_missing_shuffled_unit_candidate_trail_gate.json").read_text())
    assert gate["errors"] == ["missing_shuffled_control=candidate_trail_consistency_shuffled_cells"]

    diagnostic = postprocess_candidate_trail_result(
        results_path=results,
        output_dir=output_dir / "diagnostic",
        run_id="candidate_trail_missing_shuffled_diagnostic",
        expected_rows=3,
        require_shuffled_control=False,
    )
    assert diagnostic["status"] == "pass"
    assert diagnostic["decision"] == "weak_candidate_trail_signal"
    assert diagnostic["require_shuffled_control"] is False


def test_candidate_trail_postprocess_stop_points_to_transition_spectrum_plan(tmp_path):
    results = tmp_path / "candidate_trail_stop.jsonl"
    _write_candidate_trail_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(results, "candidate_trail_consistency_linear", 0.7910)
    _write_candidate_trail_result(results, "candidate_trail_consistency_mlp", 0.7915)
    _write_candidate_trail_result(results, "candidate_trail_consistency_shuffled_cells", 0.7917)
    output_dir = tmp_path / "postprocess"

    report = postprocess_candidate_trail_result(
        results_path=results,
        output_dir=output_dir,
        run_id="candidate_trail_stop_unit",
        expected_rows=4,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "stop_candidate_trail_route"
    assert report["next_action"]["branch"] == "stop_candidate_trail_route"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["fallback_branch"] == "bit_transition_spectrum_seed0"
    assert report["next_action"]["fallback_plan"] == "docs/experiments/innovation1-bit-transition-spectrum-plan.md"
    assert "bit_transition_spectrum_r7_262k_seed0" in report["next_action"]["launch_remote_config"]
    assert report["next_action"]["suggested_feature_cache_workers"] == 4
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    assert any("bit-transition-spectrum" in step for step in report["next_steps"])
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "stop_candidate_trail_route"
    assert readiness["next_action"]["fallback_branch"] == "bit_transition_spectrum_seed0"
    assert readiness["should_launch_remote"] is True
    assert readiness["requires_implementation"] is False
    assert readiness["readiness_pass"] is True
    assert readiness["remote_readiness_pass"] is True
    assert readiness["launch_artifacts_pass"] is True
    assert readiness["implementation_checklist"] == []
    assert readiness["readiness_reports"][0]["launch_artifacts"]["status"] == "pass"


def test_candidate_trail_postprocess_weak_signal_exposes_seed1_or_fallback_paths(tmp_path):
    results = tmp_path / "candidate_trail_weak.jsonl"
    _write_candidate_trail_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_candidate_trail_result(results, "candidate_trail_consistency_linear", 0.7922)
    _write_candidate_trail_result(results, "candidate_trail_consistency_mlp", 0.7924)
    _write_candidate_trail_result(results, "candidate_trail_consistency_shuffled_cells", 0.7923)
    output_dir = tmp_path / "postprocess"

    report = postprocess_candidate_trail_result(
        results_path=results,
        output_dir=output_dir,
        run_id="candidate_trail_weak_unit",
        expected_rows=4,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "weak_candidate_trail_signal"
    assert report["next_action"]["branch"] == "candidate_trail_seed1_variance_check"
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
    assert readiness["branch"] == "candidate_trail_seed1_variance_check"
    assert readiness["readiness_pass"] is True
    assert readiness["remote_readiness_pass"] is True
    assert readiness["launch_artifacts_pass"] is True
    assert readiness["readiness_reports"][0]["launch_artifacts"]["status"] == "pass"


def _write_transition_spectrum_result(
    path: Path,
    model: str,
    auc: float,
    calibrated: float = 0.72,
    *,
    alignment_fields: dict[str, int] | None = None,
) -> None:
    row = {
        "model": model,
        "metrics": {
            "auc": auc,
            "accuracy": calibrated,
            "calibrated_accuracy": calibrated,
            "loss": 0.55,
        },
    }
    if alignment_fields is not None:
        row.update(alignment_fields)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_trail_family_result(
    path: Path,
    model: str,
    auc: float,
    calibrated: float = 0.72,
    *,
    alignment_fields: dict[str, int] | None = None,
) -> None:
    row = {
        "model": model,
        "metrics": {
            "auc": auc,
            "accuracy": calibrated,
            "calibrated_accuracy": calibrated,
            "loss": 0.55,
        },
    }
    if alignment_fields is not None:
        row.update(alignment_fields)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_active_auxiliary_result(
    path: Path,
    model: str,
    auc: float,
    calibrated: float = 0.72,
    *,
    auxiliary_loss: float = 0.4,
    alignment_fields: dict[str, int] | None = None,
) -> None:
    row = {
        "model": model,
        "feature_route": "active_pattern_auxiliary_head",
        "metrics": {
            "auc": auc,
            "accuracy": calibrated,
            "calibrated_accuracy": calibrated,
            "loss": 0.55,
            "auxiliary_loss": auxiliary_loss,
        },
    }
    if alignment_fields is not None:
        row.update(alignment_fields)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_trail_position_result(
    path: Path,
    model: str,
    *,
    seed: int,
    auc: float,
    calibrated: float = 0.74,
) -> None:
    row = {
        "model": model,
        "seed": seed,
        "metrics": {
            "auc": auc,
            "accuracy": calibrated,
            "calibrated_accuracy": calibrated,
            "loss": 0.5,
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_trail_position_control_audit(
    path: Path,
    *,
    seed: int,
    baseline_auc: float,
    active_nibble_auc: float,
    input_difference_auc: float,
    pair_order_auc: float,
) -> None:
    payload = {
        "audit": "present_trail_position_control_baseline",
        "baseline": {
            "report": {
                "seed": seed,
                "evaluation": {"composite": {"auc": baseline_auc}},
            }
        },
        "controls": [
            {
                "variant_kind": "active_nibble",
                "report": {"evaluation": {"composite": {"auc": active_nibble_auc}}},
            },
            {
                "variant_kind": "input_difference",
                "report": {"evaluation": {"composite": {"auc": input_difference_auc}}},
            },
            {
                "variant_kind": "pair_order",
                "report": {"evaluation": {"composite": {"auc": pair_order_auc}}},
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_trail_family_gate_supports_route_when_candidate_beats_anchor_and_false_family(tmp_path):
    results = tmp_path / "trail_family.jsonl"
    _write_trail_family_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_trail_family_result(results, "trail_family_consistency_linear", 0.7925)
    _write_trail_family_result(results, "trail_family_consistency_mlp", 0.7940)
    _write_trail_family_result(results, "trail_family_consistency_false_family", 0.7926)

    report = gate_trail_family_result(results, expected_rows=4)

    assert report["status"] == "pass"
    assert report["best_candidate_model"] == "trail_family_consistency_mlp"
    assert report["decision"] == "support_trail_family_route"
    assert report["margin_vs_anchor_auc"] > 0.001
    assert report["margin_vs_false_family_auc"] > 0.001
    assert "not paper-scale" in report["claim_scope"]


def test_trail_family_gate_marks_weak_or_stop_signal(tmp_path):
    weak_results = tmp_path / "trail_family_weak.jsonl"
    _write_trail_family_result(weak_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_trail_family_result(weak_results, "trail_family_consistency_linear", 0.7922)
    _write_trail_family_result(weak_results, "trail_family_consistency_mlp", 0.7924)

    weak = gate_trail_family_result(weak_results, expected_rows=3)
    assert weak["decision"] == "weak_trail_family_signal"

    stop_results = tmp_path / "trail_family_stop.jsonl"
    _write_trail_family_result(stop_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_trail_family_result(stop_results, "trail_family_consistency_linear", 0.7910)
    _write_trail_family_result(stop_results, "trail_family_consistency_mlp", 0.7915)

    stop = gate_trail_family_result(stop_results, expected_rows=3)
    assert stop["decision"] == "stop_trail_family_route"
    assert stop["margin_vs_anchor_auc"] < 0


def test_trail_family_gate_stops_when_false_family_control_matches_true_route(tmp_path):
    results = tmp_path / "trail_family_false_family_matches.jsonl"
    _write_trail_family_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_trail_family_result(results, "trail_family_consistency_linear", 0.7925)
    _write_trail_family_result(results, "trail_family_consistency_mlp", 0.7934)
    _write_trail_family_result(results, "trail_family_consistency_false_family", 0.7934)

    report = gate_trail_family_result(results, expected_rows=4, require_false_family_control=True)

    assert report["status"] == "pass"
    assert report["decision"] == "stop_trail_family_route"
    assert report["margin_vs_anchor_auc"] > 0.001
    assert report["margin_vs_false_family_auc"] == 0
    assert "false-family control matches/exceeds" in report["interpretation"]


def test_trail_family_gate_requires_calibrated_accuracy_to_support_route(tmp_path):
    results = tmp_path / "trail_family_missing_calibrated.jsonl"
    _write_trail_family_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_trail_family_result(results, "trail_family_consistency_linear", 0.7925)
    _write_trail_family_result(results, "trail_family_consistency_mlp", 0.7940)
    _write_trail_family_result(results, "trail_family_consistency_false_family", 0.7926)
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["metrics"].pop("calibrated_accuracy", None)
    results.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    report = gate_trail_family_result(results, expected_rows=4, require_false_family_control=True)

    assert report["status"] == "fail"
    assert report["decision"] == "invalid"
    assert "missing_anchor_calibrated_accuracy for present_nibble_invp_only_spn_only" in report["errors"]
    assert "missing_candidate_calibrated_accuracy for trail_family_consistency_mlp" in report["errors"]


def test_trail_family_gate_can_require_false_family_control(tmp_path):
    results = tmp_path / "trail_family_missing_false_family.jsonl"
    _write_trail_family_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_trail_family_result(results, "trail_family_consistency_linear", 0.7925)
    _write_trail_family_result(results, "trail_family_consistency_mlp", 0.7940)

    diagnostic = gate_trail_family_result(results, expected_rows=3)
    assert diagnostic["status"] == "pass"
    assert diagnostic["decision"] == "weak_trail_family_signal"
    assert diagnostic["warnings"] == ["missing_false_family_control=trail_family_consistency_false_family"]

    strict = gate_trail_family_result(results, expected_rows=3, require_false_family_control=True)
    assert strict["status"] == "fail"
    assert strict["decision"] == "invalid"
    assert strict["errors"] == ["missing_false_family_control=trail_family_consistency_false_family"]


def test_trail_family_postprocess_writes_summary_and_next_action_readiness(tmp_path):
    results = tmp_path / "trail_family.jsonl"
    _write_trail_family_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_trail_family_result(results, "trail_family_consistency_linear", 0.7925)
    _write_trail_family_result(results, "trail_family_consistency_mlp", 0.7940)
    _write_trail_family_result(results, "trail_family_consistency_false_family", 0.7926)
    plan_doc = tmp_path / "trail_family_plan.md"
    plan_doc.write_text("# Trail Family Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_trail_family_result(
        results_path=results,
        output_dir=output_dir,
        run_id="trail_family_unit",
        expected_rows=4,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "support_trail_family_route"
    assert report["next_action"]["branch"] == "trail_family_seed1_confirmation"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert "trail_family_r7_262k_seed1" in report["next_action"]["launch_remote_config"]
    assert Path(report["trail_family_gate"]).exists()
    assert Path(report["summary"]).exists()
    assert Path(report["summary_markdown"]).exists()
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["branch"] == "trail_family_seed1_confirmation"
    assert readiness["status"] == "pass"
    assert readiness["should_launch_remote"] is True
    assert readiness["readiness_pass"] is True
    assert readiness["remote_readiness_pass"] is True
    assert readiness["launch_artifacts_pass"] is True
    assert readiness["readiness_reports"][0]["readiness"]["status"] == "pass"
    assert readiness["readiness_reports"][0]["launch_artifacts"]["status"] == "pass"
    assert readiness["implementation_checklist"] == []
    assert "Trail-Family Result" in plan_doc.read_text(encoding="utf-8")


def test_trail_family_postprocess_stop_points_to_active_auxiliary_readiness(tmp_path):
    results = tmp_path / "trail_family_stop.jsonl"
    _write_trail_family_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_trail_family_result(results, "trail_family_consistency_linear", 0.7910)
    _write_trail_family_result(results, "trail_family_consistency_mlp", 0.7915)
    _write_trail_family_result(results, "trail_family_consistency_false_family", 0.7917)
    output_dir = tmp_path / "stop_postprocess"

    report = postprocess_trail_family_result(
        results_path=results,
        output_dir=output_dir,
        run_id="trail_family_stop_unit",
        expected_rows=4,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "stop_trail_family_route"
    assert report["next_action"]["branch"] == "stop_trail_family_route"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["fallback_branch"] == "active_auxiliary_seed0"
    assert (
        report["next_action"]["next_plan_doc"]
        == "docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md"
    )
    assert "active_auxiliary_r7_262k_seed0" in report["next_action"]["launch_remote_config"]
    assert (
        "docs/experiments/innovation1-sbox-transition-prior-gate-plan.md"
        in report["next_action"]["fallback_plan_options"]
    )
    assert "active_pattern_auxiliary_head" in report["next_action"]["fallback_hypotheses"]
    assert any("Launch" in step and "active-pattern auxiliary" in step for step in report["next_steps"])

    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "stop_trail_family_route"
    assert readiness["readiness_pass"] is True
    assert readiness["remote_readiness_pass"] is True
    assert readiness["launch_artifacts_pass"] is True
    assert [item["role"] for item in readiness["readiness_reports"]] == ["primary"]
    assert readiness["readiness_reports"][0]["readiness"]["status"] == "pass"
    assert readiness["readiness_reports"][0]["launch_artifacts"]["status"] == "pass"


def test_active_auxiliary_gate_supports_route_when_candidate_beats_anchor_and_shuffled_control(tmp_path):
    results = tmp_path / "active_auxiliary.jsonl"
    _write_active_auxiliary_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_spn_only", 0.7935)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_shuffled_targets", 0.7922)

    report = gate_active_auxiliary_result(results, expected_rows=3)

    assert report["status"] == "pass"
    assert report["candidate_model"] == "present_nibble_invp_active_aux_spn_only"
    assert report["decision"] == "support_active_auxiliary_route"
    assert report["margin_vs_anchor_auc"] > 0.001
    assert report["margin_vs_shuffled_auc"] > 0.001
    assert "not paper-scale" in report["claim_scope"]


def test_active_auxiliary_gate_marks_weak_or_stop_signal(tmp_path):
    weak_results = tmp_path / "active_auxiliary_weak.jsonl"
    _write_active_auxiliary_result(weak_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_active_auxiliary_result(weak_results, "present_nibble_invp_active_aux_spn_only", 0.7924)
    _write_active_auxiliary_result(weak_results, "present_nibble_invp_active_aux_shuffled_targets", 0.7921)

    weak = gate_active_auxiliary_result(weak_results, expected_rows=3)
    assert weak["decision"] == "weak_active_auxiliary_signal"

    stop_results = tmp_path / "active_auxiliary_stop.jsonl"
    _write_active_auxiliary_result(stop_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_active_auxiliary_result(stop_results, "present_nibble_invp_active_aux_spn_only", 0.7915)
    _write_active_auxiliary_result(stop_results, "present_nibble_invp_active_aux_shuffled_targets", 0.7910)

    stop = gate_active_auxiliary_result(stop_results, expected_rows=3)
    assert stop["decision"] == "stop_active_auxiliary_route"
    assert stop["margin_vs_anchor_auc"] < 0


def test_active_auxiliary_gate_stops_when_shuffled_control_matches_true_route(tmp_path):
    results = tmp_path / "active_auxiliary_shuffled_matches.jsonl"
    _write_active_auxiliary_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_spn_only", 0.7935)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_shuffled_targets", 0.7935)

    report = gate_active_auxiliary_result(results, expected_rows=3, require_shuffled_control=True)

    assert report["status"] == "pass"
    assert report["decision"] == "stop_active_auxiliary_route"
    assert report["margin_vs_anchor_auc"] > 0.001
    assert report["margin_vs_shuffled_auc"] == 0
    assert "shuffled-target control matches/exceeds" in report["interpretation"]


def test_active_auxiliary_gate_requires_calibrated_accuracy_to_support_route(tmp_path):
    results = tmp_path / "active_auxiliary_missing_calibrated.jsonl"
    _write_active_auxiliary_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_spn_only", 0.7935)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_shuffled_targets", 0.7922)
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["metrics"].pop("calibrated_accuracy", None)
    results.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    report = gate_active_auxiliary_result(results, expected_rows=3, require_shuffled_control=True)

    assert report["status"] == "fail"
    assert report["decision"] == "invalid"
    assert "missing_anchor_calibrated_accuracy for present_nibble_invp_only_spn_only" in report["errors"]
    assert "missing_candidate_calibrated_accuracy for present_nibble_invp_active_aux_spn_only" in report["errors"]


def test_active_auxiliary_gate_requires_shuffled_control(tmp_path):
    results = tmp_path / "active_auxiliary_missing_shuffled.jsonl"
    _write_active_auxiliary_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_spn_only", 0.7935)

    diagnostic = gate_active_auxiliary_result(results, expected_rows=2)
    assert diagnostic["status"] == "pass"
    assert diagnostic["decision"] == "weak_active_auxiliary_signal"
    assert diagnostic["warnings"] == [
        "missing_shuffled_control=present_nibble_invp_active_aux_shuffled_targets"
    ]

    strict = gate_active_auxiliary_result(results, expected_rows=2, require_shuffled_control=True)
    assert strict["status"] == "fail"
    assert strict["decision"] == "invalid"
    assert strict["errors"] == [
        "missing_shuffled_control=present_nibble_invp_active_aux_shuffled_targets"
    ]


def test_trail_position_residual_gate_supports_candidate_above_baseline_and_controls(tmp_path):
    results = tmp_path / "trail_position_results.jsonl"
    _write_trail_position_result(results, "present_pairset_global_stats", seed=0, auc=0.81)
    _write_trail_position_result(results, "present_trail_position_stats_pairset", seed=0, auc=0.99)
    audit = tmp_path / "trail_position_control.json"
    _write_trail_position_control_audit(
        audit,
        seed=0,
        baseline_auc=0.77,
        active_nibble_auc=0.50,
        input_difference_auc=0.52,
        pair_order_auc=0.77,
    )

    report = gate_trail_position_residual(
        results,
        baseline_audit_paths=[audit],
        margin=0.01,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "support_trail_position_neural_residual_local"
    assert report["per_seed"][0]["candidate_auc"] == 0.99
    assert report["per_seed"][0]["global_control_auc"] == 0.81
    assert report["per_seed"][0]["deterministic_baseline_auc"] == 0.77
    assert report["per_seed"][0]["max_mismatch_control_auc"] == 0.52
    assert report["per_seed"][0]["max_order_control_auc"] == 0.77
    assert report["min_candidate_margin_vs_deterministic_auc"] > 0.01
    assert report["min_candidate_margin_vs_global_auc"] > 0.01
    assert report["min_deterministic_margin_vs_mismatch_auc"] > 0.01
    assert report["pair_order_assessment"] == "pair_order_not_bottleneck"
    assert "local diagnostic" in report["claim_scope"]


def test_trail_position_residual_gate_holds_when_candidate_does_not_beat_deterministic_baseline(
    tmp_path,
):
    results = tmp_path / "trail_position_results.jsonl"
    _write_trail_position_result(results, "present_pairset_global_stats", seed=0, auc=0.81)
    _write_trail_position_result(results, "present_trail_position_stats_pairset", seed=0, auc=0.775)
    audit = tmp_path / "trail_position_control.json"
    _write_trail_position_control_audit(
        audit,
        seed=0,
        baseline_auc=0.77,
        active_nibble_auc=0.50,
        input_difference_auc=0.52,
        pair_order_auc=0.77,
    )

    report = gate_trail_position_residual(
        results,
        baseline_audit_paths=[audit],
        margin=0.01,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "hold_trail_position_neural_residual_local"
    assert report["per_seed"][0]["candidate_margin_vs_deterministic_auc"] == pytest.approx(0.005)
    assert "does not clear" in report["interpretation"]


def test_active_auxiliary_postprocess_writes_summary_and_updates_plan_doc(tmp_path):
    results = tmp_path / "active_auxiliary.jsonl"
    _write_active_auxiliary_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_spn_only", 0.7935)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_shuffled_targets", 0.7922)
    plan_doc = tmp_path / "active_auxiliary_plan.md"
    plan_doc.write_text("# Active Auxiliary Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_active_auxiliary_result(
        results_path=results,
        output_dir=output_dir,
        run_id="active_auxiliary_unit",
        expected_rows=3,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "support_active_auxiliary_route"
    assert report["next_action"]["branch"] == "active_auxiliary_seed1_confirmation"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["next_plan_doc"] == "docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md"
    assert report["next_action"]["suggested_remote_config"].endswith(
        "innovation1_spn_present_active_auxiliary_r7_262k_seed1_gpu1_20260703.json"
    )
    assert Path(report["next_action"]["suggested_remote_config"]).exists()
    assert Path(report["active_auxiliary_gate"]).exists()
    assert Path(report["summary"]).exists()
    assert Path(report["summary_markdown"]).exists()
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["branch"] == "active_auxiliary_seed1_confirmation"
    assert readiness["status"] == "pass"
    assert readiness["should_launch_remote"] is True
    assert readiness["requires_implementation"] is False
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "## Retrieved Active-Auxiliary Result" in plan_text
    assert "<!-- active-auxiliary-postprocess:active_auxiliary_unit:start -->" in plan_text
    assert "| Decision | `support_active_auxiliary_route` |" in plan_text

    postprocess_active_auxiliary_result(
        results_path=results,
        output_dir=output_dir,
        run_id="active_auxiliary_unit",
        expected_rows=3,
        plan_doc_paths=[plan_doc],
    )
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert plan_text.count("<!-- active-auxiliary-postprocess:active_auxiliary_unit:start -->") == 1


def test_active_auxiliary_postprocess_stop_does_not_launch_remote(tmp_path):
    results = tmp_path / "active_auxiliary_stop.jsonl"
    _write_active_auxiliary_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_spn_only", 0.7915)
    _write_active_auxiliary_result(results, "present_nibble_invp_active_aux_shuffled_targets", 0.7917)
    output_dir = tmp_path / "stop_postprocess"

    report = postprocess_active_auxiliary_result(
        results_path=results,
        output_dir=output_dir,
        run_id="active_auxiliary_stop_unit",
        expected_rows=3,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "stop_active_auxiliary_route"
    assert report["next_action"]["branch"] == "sbox_transition_prior_gate_seed0"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["launch_remote_config"].endswith(
        "innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0_gpu1_20260703.json"
    )
    assert Path(report["next_action"]["launch_remote_config"]).exists()
    assert "Do not scale active-pattern auxiliary supervision" in report["next_steps"][1]
    assert "S-box transition prior gate seed0" in report["next_steps"][2]
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "sbox_transition_prior_gate_seed0"
    assert readiness["should_launch_remote"] is True
    assert readiness["implementation_checklist"] == []


def _write_sbox_prior_result(
    path: Path,
    model: str,
    auc: float,
    *,
    calibrated_accuracy: float = 0.718,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "model": model,
                    "metrics": {
                        "auc": auc,
                        "accuracy": calibrated_accuracy - 0.001,
                        "calibrated_accuracy": calibrated_accuracy,
                        "loss": 0.55,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )


def test_sbox_prior_gate_supports_route_only_when_true_prior_beats_anchor_and_controls(tmp_path):
    results = tmp_path / "sbox_prior.jsonl"
    _write_sbox_prior_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_sbox_prior_result(results, "present_nibble_invp_sbox_prior_gate", 0.7936)
    _write_sbox_prior_result(results, "present_nibble_invp_no_ddt_gate", 0.7924)
    _write_sbox_prior_result(results, "present_nibble_invp_shuffled_sbox_prior_gate", 0.7921)

    report = gate_sbox_prior_result(results, expected_rows=4)

    assert report["status"] == "pass"
    assert report["candidate_model"] == "present_nibble_invp_sbox_prior_gate"
    assert report["decision"] == "support_sbox_prior_route"
    assert report["margin_vs_anchor_auc"] > 0.001
    assert report["margin_vs_best_control_auc"] > 0.001
    assert report["best_control_model"] == "present_nibble_invp_no_ddt_gate"
    assert "not paper-scale" in report["claim_scope"]


def test_sbox_prior_gate_stops_when_no_ddt_or_shuffled_control_matches_true_prior(tmp_path):
    results = tmp_path / "sbox_prior_control_match.jsonl"
    _write_sbox_prior_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_sbox_prior_result(results, "present_nibble_invp_sbox_prior_gate", 0.7935)
    _write_sbox_prior_result(results, "present_nibble_invp_no_ddt_gate", 0.7934)
    _write_sbox_prior_result(results, "present_nibble_invp_shuffled_sbox_prior_gate", 0.7920)

    report = gate_sbox_prior_result(results, expected_rows=4, require_controls=True)

    assert report["status"] == "pass"
    assert report["decision"] == "stop_sbox_prior_route"
    assert report["margin_vs_anchor_auc"] > 0.001
    assert report["margin_vs_best_control_auc"] < 0.001
    assert "no-DDT or shuffled prior control" in report["interpretation"]


def test_sbox_prior_gate_requires_calibrated_accuracy_to_support_route(tmp_path):
    results = tmp_path / "sbox_prior_missing_calibrated.jsonl"
    _write_sbox_prior_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_sbox_prior_result(results, "present_nibble_invp_sbox_prior_gate", 0.7936)
    _write_sbox_prior_result(results, "present_nibble_invp_no_ddt_gate", 0.7924)
    _write_sbox_prior_result(results, "present_nibble_invp_shuffled_sbox_prior_gate", 0.7921)
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["metrics"].pop("calibrated_accuracy", None)
    results.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    report = gate_sbox_prior_result(results, expected_rows=4, require_controls=True)

    assert report["status"] == "fail"
    assert report["decision"] == "invalid"
    assert "missing_anchor_calibrated_accuracy for present_nibble_invp_only_spn_only" in report["errors"]
    assert "missing_candidate_calibrated_accuracy for present_nibble_invp_sbox_prior_gate" in report["errors"]


def test_sbox_prior_gate_marks_weak_when_true_prior_is_best_but_below_margin(tmp_path):
    results = tmp_path / "sbox_prior_weak.jsonl"
    _write_sbox_prior_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_sbox_prior_result(results, "present_nibble_invp_sbox_prior_gate", 0.7928)
    _write_sbox_prior_result(results, "present_nibble_invp_no_ddt_gate", 0.7922)
    _write_sbox_prior_result(results, "present_nibble_invp_shuffled_sbox_prior_gate", 0.7921)

    report = gate_sbox_prior_result(results, expected_rows=4, require_controls=True)

    assert report["status"] == "pass"
    assert report["decision"] == "weak_sbox_prior_signal"
    assert 0.0 < report["margin_vs_anchor_auc"] < report["required_margin"]
    assert 0.0 < report["margin_vs_best_control_auc"] < report["required_margin"]


def test_sbox_prior_gate_requires_controls_when_strict(tmp_path):
    results = tmp_path / "sbox_prior_missing_controls.jsonl"
    _write_sbox_prior_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_sbox_prior_result(results, "present_nibble_invp_sbox_prior_gate", 0.7935)

    diagnostic = gate_sbox_prior_result(results, expected_rows=2)
    assert diagnostic["status"] == "pass"
    assert diagnostic["decision"] == "weak_sbox_prior_signal"
    assert diagnostic["warnings"] == [
        "missing_control_model=present_nibble_invp_no_ddt_gate",
        "missing_control_model=present_nibble_invp_shuffled_sbox_prior_gate",
    ]

    strict = gate_sbox_prior_result(results, expected_rows=2, require_controls=True)
    assert strict["status"] == "fail"
    assert strict["decision"] == "invalid"
    assert strict["errors"] == [
        "missing_control_model=present_nibble_invp_no_ddt_gate",
        "missing_control_model=present_nibble_invp_shuffled_sbox_prior_gate",
    ]


def test_sbox_prior_postprocess_writes_summary_and_updates_plan_doc(tmp_path):
    results = tmp_path / "sbox_prior.jsonl"
    _write_sbox_prior_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_sbox_prior_result(results, "present_nibble_invp_sbox_prior_gate", 0.7936)
    _write_sbox_prior_result(results, "present_nibble_invp_no_ddt_gate", 0.7924)
    _write_sbox_prior_result(results, "present_nibble_invp_shuffled_sbox_prior_gate", 0.7921)
    plan_doc = tmp_path / "sbox_prior_plan.md"
    plan_doc.write_text("# S-box Prior Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_sbox_prior_result(
        results_path=results,
        output_dir=output_dir,
        run_id="sbox_prior_unit",
        expected_rows=4,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "support_sbox_prior_route"
    assert report["next_action"]["branch"] == "sbox_prior_seed1_confirmation"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert "sbox_transition_prior_gate_r7_262k_seed1" in report["next_action"]["launch_remote_config"]
    assert Path(report["sbox_prior_gate"]).exists()
    assert Path(report["summary"]).exists()
    assert Path(report["summary_markdown"]).exists()
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["branch"] == "sbox_prior_seed1_confirmation"
    assert readiness["status"] == "pass"
    assert readiness["should_launch_remote"] is True
    assert readiness["readiness_pass"] is True
    assert readiness["implementation_checklist"] == []
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "## Retrieved S-box Prior Result" in plan_text
    assert "<!-- sbox-prior-postprocess:sbox_prior_unit:start -->" in plan_text
    assert "| Decision | `support_sbox_prior_route` |" in plan_text

    postprocess_sbox_prior_result(
        results_path=results,
        output_dir=output_dir,
        run_id="sbox_prior_unit",
        expected_rows=4,
        plan_doc_paths=[plan_doc],
    )
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert plan_text.count("<!-- sbox-prior-postprocess:sbox_prior_unit:start -->") == 1


def test_sbox_prior_postprocess_support_points_to_ready_seed1_assets(tmp_path):
    results = tmp_path / "sbox_prior_seed0_support.jsonl"
    _write_sbox_prior_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_sbox_prior_result(results, "present_nibble_invp_sbox_prior_gate", 0.7936)
    _write_sbox_prior_result(results, "present_nibble_invp_no_ddt_gate", 0.7924)
    _write_sbox_prior_result(results, "present_nibble_invp_shuffled_sbox_prior_gate", 0.7921)
    output_dir = tmp_path / "postprocess_ready"

    report = postprocess_sbox_prior_result(
        results_path=results,
        output_dir=output_dir,
        run_id="sbox_prior_ready_unit",
        expected_rows=4,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "support_sbox_prior_route"
    assert report["next_action"]["branch"] == "sbox_prior_seed1_confirmation"
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert "sbox_transition_prior_gate_r7_262k_seed1" in report["next_action"]["launch_remote_config"]
    readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert readiness["branch"] == "sbox_prior_seed1_confirmation"
    assert readiness["status"] == "pass"
    assert readiness["should_launch_remote"] is True
    assert readiness["readiness_pass"] is True
    assert readiness["remote_readiness_pass"] is True
    assert readiness["launch_artifacts_pass"] is True
    assert readiness["implementation_checklist"] == []


def test_monitor_health_emits_sbox_prior_postprocess_command_when_result_ready(tmp_path):
    run_id = "sbox_prior_monitor_unit"
    run_root = tmp_path / run_id
    monitor_dir = run_root / "monitor"
    results_dir = run_root / "results"
    plan = tmp_path / "sbox_prior_plan.csv"
    monitor_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (monitor_dir / "monitor.log").write_text("2026-07-03T12:00:00+08:00 running\n", encoding="utf-8")
    plan.write_text(
        "model_key\n"
        "present_nibble_invp_only_spn_only\n"
        "present_nibble_invp_sbox_prior_gate\n"
        "present_nibble_invp_no_ddt_gate\n"
        "present_nibble_invp_shuffled_sbox_prior_gate\n",
        encoding="utf-8",
    )
    results = results_dir / f"{run_id}.jsonl"
    for model, auc in [
        ("present_nibble_invp_only_spn_only", 0.7920),
        ("present_nibble_invp_sbox_prior_gate", 0.7936),
        ("present_nibble_invp_no_ddt_gate", 0.7924),
        ("present_nibble_invp_shuffled_sbox_prior_gate", 0.7921),
    ]:
        _write_sbox_prior_result(results, model, auc)

    report = monitor_health_report(
        run_id=run_id,
        root=tmp_path,
        plan_path=plan,
        expected_rows=4,
        postprocess_kind="sbox_prior",
        now=datetime.fromisoformat("2026-07-03T12:01:00+08:00"),
    )

    assert report["status"] == "result_ready"
    assert report["postprocess_allowed"] is True
    assert report["postprocess_command"][0:2] == ["env", "UV_CACHE_DIR=/tmp/uv-cache"]
    assert "scripts/postprocess-sbox-prior" in report["postprocess_command"]
    assert "--expected-rows" in report["postprocess_command"]
    assert "4" in report["postprocess_command"]


def _write_difference_screen_result(
    path: Path,
    profile: str,
    member: int,
    auc: float,
    *,
    calibrated_accuracy: float = 0.53,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "model": "present_nibble_invp_pair_consistency_spn_only",
                    "difference_profile": profile,
                    "difference_member": member,
                    "input_difference": "0x0",
                    "metrics": {
                        "auc": auc,
                        "accuracy": calibrated_accuracy - 0.001,
                        "calibrated_accuracy": calibrated_accuracy,
                        "loss": 0.69,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )


def test_difference_screen_gate_promotes_strong_non_reference_candidate(tmp_path):
    results = tmp_path / "difference_screen.jsonl"
    _write_difference_screen_result(results, "present_zhang_wang2022_mcnd", 0, 0.512)
    _write_difference_screen_result(results, "present_wang_jain2021", 0, 0.517)
    _write_difference_screen_result(results, "present_autond_dbitnet2023_highround", 0, 0.526)

    report = gate_difference_screen_result(results, expected_rows=3)

    assert report["status"] == "pass"
    assert report["best"]["difference_id"] == "present_autond_dbitnet2023_highround:0"
    assert report["reference"]["difference_id"] == "present_zhang_wang2022_mcnd:0"
    assert report["delta_vs_reference_auc"] == pytest.approx(0.014)
    assert report["decision"] == "promote_best_difference_to_262k_confirmation"
    assert report["action"] == "prepare_262k_confirmation_for_best_difference"
    assert "not same-protocol model-improvement evidence" in report["claim_scope"]


def test_difference_screen_postprocess_writes_summary_and_updates_plan_doc(tmp_path):
    results = tmp_path / "difference_screen.jsonl"
    _write_difference_screen_result(results, "present_zhang_wang2022_mcnd", 0, 0.512)
    _write_difference_screen_result(results, "present_wang_jain2021", 0, 0.526)
    plan_doc = tmp_path / "difference_screen_plan.md"
    plan_doc.write_text("# Difference Screen Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_difference_screen_result(
        results_path=results,
        output_dir=output_dir,
        run_id="difference_screen_unit",
        expected_rows=2,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "promote_best_difference_to_262k_confirmation"
    assert report["next_action"]["branch"] == "r9_difference_262k_confirmation"
    assert report["next_action"]["requires_implementation"] is True
    assert report["next_action"]["should_launch_remote"] is False
    assert "scripts/create-difference-confirmation-plan" in report["next_action"]["confirmation_plan_command"]
    assert "--selected-difference present_wang_jain2021:0" in report["next_action"]["confirmation_plan_command"]
    assert "difference_confirmation_262k_seed0_present_wang_jain2021_0.csv" in (
        report["next_action"]["confirmation_plan_command"]
    )
    assert report["best"]["difference_id"] == "present_wang_jain2021:0"
    assert Path(report["difference_screen_gate"]).exists()
    assert Path(report["summary"]).exists()
    assert Path(report["summary_markdown"]).exists()
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "## Retrieved Difference-Screen Result" in plan_text
    assert "<!-- difference-screen-postprocess:difference_screen_unit:start -->" in plan_text
    assert "| Decision | `promote_best_difference_to_262k_confirmation` |" in plan_text
    assert "| Confirmation plan command | `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create-difference-confirmation-plan" in plan_text

    postprocess_difference_screen_result(
        results_path=results,
        output_dir=output_dir,
        run_id="difference_screen_unit",
        expected_rows=2,
        plan_doc_paths=[plan_doc],
    )
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert plan_text.count("<!-- difference-screen-postprocess:difference_screen_unit:start -->") == 1


def test_difference_confirmation_plan_keeps_only_reference_and_selected_candidate(tmp_path):
    screen_plan = Path("configs/experiment/innovation1/innovation1_spn_present_r9_difference_screen_65k_seed0.csv")
    output = tmp_path / "r9_difference_confirmation_262k.csv"
    summary_path = tmp_path / "summary.json"

    summary = create_difference_confirmation_plan(
        screen_plan_path=screen_plan,
        output_path=output,
        selected_difference="present_wang_jain2021:2",
        summary_path=summary_path,
    )

    rows = list(csv.DictReader(output.open("r", encoding="utf-8", newline="")))
    assert summary["status"] == "pass"
    assert summary["rows"] == 2
    assert summary["samples_per_class"] == 262144
    assert Path(summary_path).exists()
    assert [f"{row['difference_profile']}:{row['difference_member']}" for row in rows] == [
        "present_zhang_wang2022_mcnd:0",
        "present_wang_jain2021:2",
    ]
    assert [row["architecture_rank"] for row in rows] == ["0", "1"]
    assert {row["samples_per_class"] for row in rows} == {"262144"}
    assert {row["rounds"] for row in rows} == {"9"}
    assert {row["model_key"] for row in rows} == {"present_nibble_invp_pair_consistency_spn_only"}
    assert {row["negative_mode"] for row in rows} == {"encrypted_random_plaintexts"}
    assert {row["sample_structure"] for row in rows} == {"zhang_wang_case2_official_mcnd"}
    assert all("data construction only" in row["evidence"] for row in rows)


def test_difference_confirmation_plan_rejects_reference_as_selected(tmp_path):
    screen_plan = Path("configs/experiment/innovation1/innovation1_spn_present_r9_difference_screen_65k_seed0.csv")

    with pytest.raises(ValueError, match="non-reference"):
        create_difference_confirmation_plan(
            screen_plan_path=screen_plan,
            output_path=tmp_path / "invalid.csv",
            selected_difference="present_zhang_wang2022_mcnd:0",
        )


def test_difference_screen_postprocess_rejects_incomplete_result_rows(tmp_path):
    results = tmp_path / "difference_screen_incomplete.jsonl"
    _write_difference_screen_result(results, "present_zhang_wang2022_mcnd", 0, 0.512)
    output_dir = tmp_path / "postprocess"

    report = postprocess_difference_screen_result(
        results_path=results,
        output_dir=output_dir,
        run_id="difference_screen_incomplete_unit",
        expected_rows=7,
    )

    assert report["status"] == "fail"
    assert report["validation_status"] == "not_run"
    assert report["difference_screen_status"] == "fail"
    assert report["next_action"]["branch"] == "invalid"
    gate = json.loads(Path(report["difference_screen_gate"]).read_text(encoding="utf-8"))
    assert gate["result_rows"] == 1
    assert gate["expected_rows"] == 7
    assert any("result_rows=1 expected_rows=7" in error for error in gate["errors"])


def test_monitor_health_emits_difference_screen_postprocess_command_when_result_ready(tmp_path):
    run_id = "difference_screen_monitor_unit"
    run_root = tmp_path / run_id
    monitor_dir = run_root / "monitor"
    results_dir = run_root / "results"
    plan = tmp_path / "difference_screen_plan.csv"
    monitor_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (monitor_dir / "monitor.log").write_text("2026-07-05T12:00:00+08:00 running\n", encoding="utf-8")
    plan.write_text("model_key\npresent_nibble_invp_pair_consistency_spn_only\n", encoding="utf-8")
    results = results_dir / f"{run_id}.jsonl"
    _write_difference_screen_result(results, "present_zhang_wang2022_mcnd", 0, 0.512)
    _write_difference_screen_result(results, "present_wang_jain2021", 0, 0.526)

    report = monitor_health_report(
        run_id=run_id,
        root=tmp_path,
        plan_path=plan,
        expected_rows=2,
        postprocess_kind="difference_screen",
        now=datetime.fromisoformat("2026-07-05T12:01:00+08:00"),
    )

    assert report["status"] == "result_ready"
    assert report["postprocess_allowed"] is True
    assert report["postprocess_command"][0:2] == ["env", "UV_CACHE_DIR=/tmp/uv-cache"]
    assert "scripts/postprocess-difference-screen" in report["postprocess_command"]
    assert "--expected-rows" in report["postprocess_command"]
    assert "2" in report["postprocess_command"]


def _write_pair_mixer_result(path: Path, model: str, auc: float) -> None:
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.append(
        {
            "cipher": "PRESENT-80",
            "model": model,
            "metrics": {
                "auc": auc,
                "accuracy": 0.5 + (auc - 0.5) / 2,
                "calibrated_accuracy": 0.5 + (auc - 0.5) / 3,
                "loss": 0.69,
            },
        }
    )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_pair_evidence_pooling_result(
    path: Path,
    *,
    architecture: str,
    architecture_rank: int,
    model: str,
    pooling: str,
    auc: float,
) -> None:
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.append(
        {
            "cipher": "PRESENT-80",
            "architecture": architecture,
            "architecture_rank": architecture_rank,
            "selected_model": model,
            "metrics": {
                "auc": auc,
                "accuracy": 0.5 + (auc - 0.5) / 2,
                "calibrated_accuracy": 0.5 + (auc - 0.5) / 3,
                "loss": 0.69,
            },
            "training": {
                "model_options": {
                    "pooling": pooling,
                },
            },
        }
    )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_integral_inverse_feature_result(
    path: Path,
    *,
    architecture: str,
    architecture_rank: int,
    model: str,
    feature_encoding: str,
    auc: float,
) -> None:
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.append(
        {
            "cipher": "PRESENT-80",
            "architecture": architecture,
            "architecture_rank": architecture_rank,
            "selected_model": model,
            "feature_encoding": feature_encoding,
            "metrics": {
                "auc": auc,
                "accuracy": 0.5 + (auc - 0.5) / 2,
                "calibrated_accuracy": 0.5 + (auc - 0.5) / 3,
                "loss": 0.69,
            },
            "training": {
                "feature_encoding": feature_encoding,
            },
        }
    )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_r9_weak_probe_result(path: Path, model: str, auc: float) -> None:
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.append(
        {
            "cipher": "PRESENT-80",
            "structure": "SPN",
            "rounds": 9,
            "seed": 0,
            "samples_per_class": 262144,
            "pairs_per_sample": 16,
            "model": model,
            "selected_model": model,
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "zhang_wang_case2_official_mcnd",
            "difference_profile": "present_zhang_wang2022_mcnd",
            "difference_member": 0,
            "key_rotation_interval": 0,
            "metrics": {
                "auc": auc,
                "accuracy": 0.5 + (auc - 0.5) / 2,
                "calibrated_accuracy": 0.5 + (auc - 0.5) / 3,
                "loss": 0.69,
            },
        }
    )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_r8_pairset_1m_result(path: Path, model: str, auc: float) -> None:
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.append(
        {
            "cipher": "PRESENT-80",
            "structure": "SPN",
            "rounds": 8,
            "seed": 0,
            "samples_per_class": 1000000,
            "pairs_per_sample": 16,
            "model": model,
            "selected_model": model,
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "zhang_wang_case2_official_mcnd",
            "difference_profile": "present_zhang_wang2022_mcnd",
            "difference_member": 0,
            "key_rotation_interval": 0,
            "metrics": {
                "auc": auc,
                "accuracy": 0.5 + (auc - 0.5) / 2,
                "calibrated_accuracy": 0.5 + (auc - 0.5) / 3,
                "loss": 0.69,
            },
        }
    )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_pair_mixer_postprocess_writes_summary_and_updates_plan_doc(tmp_path):
    results = tmp_path / "pair_mixer.jsonl"
    _write_pair_mixer_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.552)
    _write_pair_mixer_result(results, "present_nibble_invp_pair_mixer_consistency_spn_only", 0.557)
    plan_doc = tmp_path / "pair_mixer_plan.md"
    plan_doc.write_text("# Pair Mixer Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_pair_mixer_consistency_result(
        results_path=results,
        output_dir=output_dir,
        run_id="pair_mixer_unit",
        expected_rows=2,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "support_pair_mixer_consistency_route"
    assert report["next_action"]["branch"] == "pair_mixer_seed_or_r9_diagnostic"
    assert report["next_action"]["requires_implementation"] is True
    assert report["next_action"]["should_launch_remote"] is False
    assert report["delta_vs_anchor_auc"] == pytest.approx(0.005)
    assert Path(report["pair_mixer_gate"]).exists()
    assert Path(report["summary"]).exists()
    assert Path(report["summary_markdown"]).exists()
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "## Retrieved Pair-Mixer Result" in plan_text
    assert "<!-- pair-mixer-postprocess:pair_mixer_unit:start -->" in plan_text
    assert "| Decision | `support_pair_mixer_consistency_route` |" in plan_text

    postprocess_pair_mixer_consistency_result(
        results_path=results,
        output_dir=output_dir,
        run_id="pair_mixer_unit",
        expected_rows=2,
        plan_doc_paths=[plan_doc],
    )
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert plan_text.count("<!-- pair-mixer-postprocess:pair_mixer_unit:start -->") == 1


def test_monitor_health_emits_pair_mixer_postprocess_command_when_result_ready(tmp_path):
    run_id = "pair_mixer_monitor_unit"
    run_root = tmp_path / run_id
    monitor_dir = run_root / "monitor"
    results_dir = run_root / "results"
    plan = tmp_path / "pair_mixer_plan.csv"
    monitor_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (monitor_dir / "monitor.log").write_text("2026-07-05T12:00:00+08:00 running\n", encoding="utf-8")
    plan.write_text("model_key\npresent_nibble_invp_pair_mixer_consistency_spn_only\n", encoding="utf-8")
    results = results_dir / f"{run_id}.jsonl"
    _write_pair_mixer_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.552)
    _write_pair_mixer_result(results, "present_nibble_invp_pair_mixer_consistency_spn_only", 0.557)

    report = monitor_health_report(
        run_id=run_id,
        root=tmp_path,
        plan_path=plan,
        expected_rows=2,
        postprocess_kind="pair_mixer",
        now=datetime.fromisoformat("2026-07-05T12:01:00+08:00"),
    )

    assert report["status"] == "result_ready"
    assert report["postprocess_allowed"] is True
    assert report["postprocess_command"][0:2] == ["env", "UV_CACHE_DIR=/tmp/uv-cache"]
    assert "scripts/postprocess-pair-mixer-consistency" in report["postprocess_command"]
    assert "--expected-rows" in report["postprocess_command"]
    assert "2" in report["postprocess_command"]


def test_pair_evidence_pooling_postprocess_writes_summary_and_updates_plan_doc(tmp_path):
    results = tmp_path / "pair_evidence_pooling.jsonl"
    _write_pair_evidence_pooling_result(
        results,
        architecture="present_nibble_invp_pair_consistency_spn_only",
        architecture_rank=0,
        model="present_nibble_invp_pair_consistency_spn_only",
        pooling="topk_logsumexp",
        auc=0.552,
    )
    _write_pair_evidence_pooling_result(
        results,
        architecture="present_nibble_invp_pair_mixer_consistency_spn_only",
        architecture_rank=1,
        model="present_nibble_invp_pair_mixer_consistency_spn_only",
        pooling="topk_logsumexp",
        auc=0.556,
    )
    _write_pair_evidence_pooling_result(
        results,
        architecture="present_nibble_invp_pair_mixer_consistency_spn_only",
        architecture_rank=2,
        model="present_nibble_invp_pair_mixer_consistency_spn_only",
        pooling="logsumexp",
        auc=0.559,
    )
    _write_pair_evidence_pooling_result(
        results,
        architecture="present_nibble_invp_pair_mixer_consistency_spn_only",
        architecture_rank=3,
        model="present_nibble_invp_pair_mixer_consistency_spn_only",
        pooling="topk_mean",
        auc=0.551,
    )
    plan_doc = tmp_path / "pair_evidence_pooling_plan.md"
    plan_doc.write_text("# Pair-Evidence Pooling Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_pair_evidence_pooling_result(
        results_path=results,
        output_dir=output_dir,
        run_id="pair_evidence_pooling_unit",
        expected_rows=4,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "promote_best_pooling_to_262k_confirmation"
    assert report["next_action"]["branch"] == "pair_evidence_pooling_262k_confirmation"
    assert report["next_action"]["requires_implementation"] is True
    assert report["next_action"]["should_launch_remote"] is False
    assert report["next_action"]["selected_pooling"] == "logsumexp"
    assert report["delta_vs_anchor_auc"] == pytest.approx(0.007)
    assert Path(report["pair_evidence_pooling_gate"]).exists()
    assert Path(report["summary"]).exists()
    assert Path(report["summary_markdown"]).exists()
    assert [entry["pooling"] for entry in report["ranking"]] == [
        "logsumexp",
        "topk_logsumexp",
        "topk_logsumexp",
        "topk_mean",
    ]
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "## Retrieved Pair-Evidence Pooling Result" in plan_text
    assert "<!-- pair-evidence-pooling-postprocess:pair_evidence_pooling_unit:start -->" in plan_text
    assert "| Decision | `promote_best_pooling_to_262k_confirmation` |" in plan_text
    assert "| Best pooling | `logsumexp` |" in plan_text

    postprocess_pair_evidence_pooling_result(
        results_path=results,
        output_dir=output_dir,
        run_id="pair_evidence_pooling_unit",
        expected_rows=4,
        plan_doc_paths=[plan_doc],
    )
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert plan_text.count("<!-- pair-evidence-pooling-postprocess:pair_evidence_pooling_unit:start -->") == 1


def test_monitor_health_emits_pair_evidence_pooling_postprocess_command_when_result_ready(tmp_path):
    run_id = "pair_evidence_pooling_monitor_unit"
    run_root = tmp_path / run_id
    monitor_dir = run_root / "monitor"
    results_dir = run_root / "results"
    plan = tmp_path / "pair_evidence_pooling_plan.csv"
    monitor_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (monitor_dir / "monitor.log").write_text("2026-07-05T12:00:00+08:00 running\n", encoding="utf-8")
    plan.write_text(
        "model_key\n"
        "present_nibble_invp_pair_consistency_spn_only\n"
        "present_nibble_invp_pair_mixer_consistency_spn_only\n"
        "present_nibble_invp_pair_mixer_consistency_spn_only\n"
        "present_nibble_invp_pair_mixer_consistency_spn_only\n",
        encoding="utf-8",
    )
    results = results_dir / f"{run_id}.jsonl"
    _write_pair_evidence_pooling_result(
        results,
        architecture="present_nibble_invp_pair_consistency_spn_only",
        architecture_rank=0,
        model="present_nibble_invp_pair_consistency_spn_only",
        pooling="topk_logsumexp",
        auc=0.552,
    )
    for rank, pooling, auc in [(1, "topk_logsumexp", 0.556), (2, "logsumexp", 0.559), (3, "topk_mean", 0.551)]:
        _write_pair_evidence_pooling_result(
            results,
            architecture="present_nibble_invp_pair_mixer_consistency_spn_only",
            architecture_rank=rank,
            model="present_nibble_invp_pair_mixer_consistency_spn_only",
            pooling=pooling,
            auc=auc,
        )

    report = monitor_health_report(
        run_id=run_id,
        root=tmp_path,
        plan_path=plan,
        expected_rows=4,
        postprocess_kind="pair_evidence_pooling",
        now=datetime.fromisoformat("2026-07-05T12:01:00+08:00"),
    )

    assert report["status"] == "result_ready"
    assert report["postprocess_allowed"] is True
    assert report["postprocess_command"][0:2] == ["env", "UV_CACHE_DIR=/tmp/uv-cache"]
    assert "scripts/postprocess-pair-evidence-pooling" in report["postprocess_command"]
    assert "--expected-rows" in report["postprocess_command"]
    assert "4" in report["postprocess_command"]


def test_summarize_spn_evidence_tracks_pair_mixer_running_followup(tmp_path):
    run_id = "i1_present_r8_pair_mixer_consistency_262k_seed0_gpu0_20260705"
    run_root = tmp_path / run_id
    monitor_dir = run_root / "monitor"
    logs_dir = run_root / "logs"
    results_dir = run_root / "results"
    monitor_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (monitor_dir / "monitor.log").write_text(
        "2026-07-05T12:00:00+08:00 sync\n"
        "2026-07-05T12:00:01+08:00 running\n",
        encoding="utf-8",
    )
    (logs_dir / "pair_mixer_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "train_batch",
                "model": "present_nibble_invp_pair_mixer_consistency_spn_only",
                "epoch": 1,
                "epochs": 30,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (results_dir / f"{run_id}.jsonl").write_text("", encoding="utf-8")

    report = summarize_spn_evidence(tmp_path)
    active = report["active_recommendation"]

    assert active["branch"] == "wait_for_pair_mixer_result"
    assert active["run_id"] == run_id
    assert active["postprocess_allowed"] is False
    assert active["expected_rows"] == 2
    assert "--postprocess-kind pair_mixer" in active["monitor_health_command"]
    assert "scripts/postprocess-pair-mixer-consistency" in active["postprocess_when_ready_command"]
    assert "launch another follow-up branch in parallel" in active["main_thread_policy"]["forbidden_until_gate"]


def test_summarize_spn_evidence_prefers_running_followup_over_high_round_arbitration(tmp_path):
    r8_run_id = "i1_present_r8_pairset_1m_seed0_gpu1_20260705"
    r9_run_id = "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705"
    for run_id, decision in [
        (r8_run_id, "stop_or_rethink_r8_pairset_scale"),
        (r9_run_id, "stop_from_scratch_r9_r10_plan_curriculum_or_difference_search"),
    ]:
        run_root = tmp_path / run_id
        run_root.mkdir(parents=True)
        (run_root / f"{run_id}_postprocess_summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "pass",
                    "validation_status": "pass",
                    "decision": decision,
                    "next_action": {"should_launch_remote": False},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    followup_run_id = "i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705"
    followup_root = tmp_path / followup_run_id
    monitor_dir = followup_root / "monitor"
    logs_dir = followup_root / "logs"
    results_dir = followup_root / "results"
    monitor_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (monitor_dir / "monitor.log").write_text(
        "2026-07-05T12:00:00+08:00 sync\n"
        "2026-07-05T12:00:01+08:00 running\n",
        encoding="utf-8",
    )
    (logs_dir / "r9_curriculum_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "train_batch",
                "model": "present_nibble_invp_pair_consistency_spn_only",
                "epoch": 1,
                "epochs": 22,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (results_dir / f"{followup_run_id}.jsonl").write_text("", encoding="utf-8")

    report = summarize_spn_evidence(tmp_path)
    active = report["active_recommendation"]

    assert active["branch"] == "wait_for_r9_curriculum_result"
    assert active["run_id"] == followup_run_id
    assert active["should_launch_remote"] is False
    assert "launch another follow-up branch in parallel" in active["main_thread_policy"]["forbidden_until_gate"]


def test_summarize_spn_evidence_prefers_completed_followup_over_stale_high_round_arbitration(tmp_path):
    r8_run_id = "i1_present_r8_pairset_1m_seed0_gpu1_20260705"
    r9_run_id = "i1_present_r9_weak_probe_262k_seed0_gpu0_20260705"
    for run_id, decision in [
        (r8_run_id, "stop_or_rethink_r8_pairset_scale"),
        (r9_run_id, "stop_from_scratch_r9_r10_plan_curriculum_or_difference_search"),
    ]:
        run_root = tmp_path / run_id
        run_root.mkdir(parents=True)
        (run_root / f"{run_id}_postprocess_summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "pass",
                    "validation_status": "pass",
                    "decision": decision,
                    "next_action": {"should_launch_remote": False},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    followup_run_id = "i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705"
    followup_root = tmp_path / followup_run_id
    followup_root.mkdir(parents=True)
    (followup_root / f"{followup_run_id}_postprocess_summary.json").write_text(
        json.dumps(
            {
                "run_id": followup_run_id,
                "status": "pass",
                "validation_status": "pass",
                "decision": "stop_or_rethink_r9_curriculum_route",
                "claim_scope": "PRESENT r9 262144/class r8-to-r9 curriculum diagnostic only",
                "next_action": {
                    "branch": "stop_r9_curriculum_route",
                    "fallback_hypotheses": [
                        "r9_difference_screen",
                        "r8_integral_inverse_feature",
                        "pair_evidence_pooling",
                    ],
                    "requires_implementation": False,
                    "should_launch_remote": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = summarize_spn_evidence(tmp_path)
    active = report["active_recommendation"]

    assert active["branch"] == "stop_r9_curriculum_route"
    assert active["run_id"] == followup_run_id
    assert active["should_launch_remote"] is False
    assert active["fallback_hypotheses"] == [
        "r9_difference_screen",
        "r8_integral_inverse_feature",
        "pair_evidence_pooling",
    ]


def test_summarize_spn_evidence_tracks_pair_evidence_pooling_ready_followup(tmp_path):
    run_id = "i1_present_r8_pair_evidence_pooling_screen_65k_seed0_gpu0_20260705"
    run_root = tmp_path / run_id
    monitor_dir = run_root / "monitor"
    results_dir = run_root / "results"
    monitor_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (monitor_dir / "monitor.log").write_text("2026-07-05T12:00:00+08:00 running\n", encoding="utf-8")
    results = results_dir / f"{run_id}.jsonl"
    _write_pair_evidence_pooling_result(
        results,
        architecture="present_nibble_invp_pair_consistency_spn_only",
        architecture_rank=0,
        model="present_nibble_invp_pair_consistency_spn_only",
        pooling="topk_logsumexp",
        auc=0.552,
    )
    for rank, pooling, auc in [(1, "topk_logsumexp", 0.556), (2, "logsumexp", 0.559), (3, "topk_mean", 0.551)]:
        _write_pair_evidence_pooling_result(
            results,
            architecture="present_nibble_invp_pair_mixer_consistency_spn_only",
            architecture_rank=rank,
            model="present_nibble_invp_pair_mixer_consistency_spn_only",
            pooling=pooling,
            auc=auc,
        )

    report = summarize_spn_evidence(tmp_path)
    active = report["active_recommendation"]

    assert active["branch"] == "postprocess_pair_evidence_pooling_result"
    assert active["run_id"] == run_id
    assert active["postprocess_allowed"] is True
    assert active["results_jsonl_line_count"] == 4
    assert active["postprocess_command"][0:2] == ["env", "UV_CACHE_DIR=/tmp/uv-cache"]
    assert "scripts/postprocess-pair-evidence-pooling" in active["postprocess_command"]
    assert "scripts/postprocess-pair-evidence-pooling" in active["postprocess_when_ready_command"]
    assert "run the listed route-specific postprocess command" in active["main_thread_policy"]["allowed_actions"]


def test_summarize_spn_evidence_tracks_difference_and_integral_followups(tmp_path):
    diff_run_id = "i1_present_r9_difference_screen_65k_seed0_gpu0_20260705"
    diff_root = tmp_path / diff_run_id
    (diff_root / "monitor").mkdir(parents=True)
    (diff_root / "results").mkdir(parents=True)
    (diff_root / "monitor" / "monitor.log").write_text(
        "2026-07-05T12:00:00+08:00 running\n",
        encoding="utf-8",
    )
    diff_results = diff_root / "results" / f"{diff_run_id}.jsonl"
    for index, auc in enumerate([0.512, 0.513, 0.514, 0.515, 0.516, 0.517, 0.518]):
        _write_difference_screen_result(diff_results, f"profile_{index}", 0, auc)

    diff_report = summarize_spn_evidence(tmp_path)
    diff_active = diff_report["active_recommendation"]
    assert diff_active["branch"] == "postprocess_r9_difference_screen_result"
    assert diff_active["run_id"] == diff_run_id
    assert diff_active["postprocess_allowed"] is True
    assert diff_active["results_jsonl_line_count"] == 7
    assert "scripts/postprocess-difference-screen" in diff_active["postprocess_when_ready_command"]

    (diff_root / f"{diff_run_id}_postprocess_summary.json").write_text("{}", encoding="utf-8")
    integral_run_id = "i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705"
    integral_root = tmp_path / integral_run_id
    (integral_root / "monitor").mkdir(parents=True)
    (integral_root / "logs").mkdir(parents=True)
    (integral_root / "results").mkdir(parents=True)
    heartbeat = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
    (integral_root / "monitor" / "monitor.log").write_text(
        f"{heartbeat} sync\n"
        f"{heartbeat} running\n",
        encoding="utf-8",
    )
    (integral_root / "logs" / "integral_inverse_progress.jsonl").write_text(
        json.dumps({"event": "train_batch", "epoch": 1, "epochs": 20}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (integral_root / "results" / f"{integral_run_id}.jsonl").write_text("", encoding="utf-8")

    integral_report = summarize_spn_evidence(tmp_path)
    integral_active = integral_report["active_recommendation"]
    assert integral_active["branch"] == "wait_for_integral_inverse_feature_result"
    assert integral_active["run_id"] == integral_run_id
    assert integral_active["postprocess_allowed"] is False
    assert integral_active["expected_rows"] == 3
    assert "gpu1_retry1" in integral_active["monitor_health_command"]
    assert "scripts/advance-integral-inverse-feature-result" in integral_active["postprocess_when_ready_command"]


def test_summarize_spn_evidence_prioritizes_trail_position_262k_over_stale_followup(tmp_path):
    stale_run_id = "i1_present_r9_difference_screen_65k_seed0_gpu0_20260705"
    stale_root = tmp_path / stale_run_id
    stale_root.mkdir(parents=True)
    (stale_root / f"{stale_run_id}_postprocess_summary.json").write_text(
        json.dumps(
            {
                "run_id": stale_run_id,
                "status": "pass",
                "validation_status": "pass",
                "decision": "all_candidates_near_random_stop_difference_screen",
                "claim_scope": "PRESENT input-difference/data-construction screen",
                "next_action": {
                    "branch": "stop_current_difference_screen",
                    "should_launch_remote": False,
                    "requires_implementation": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for run_id in [
        "i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706",
        "i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706",
    ]:
        run_root = tmp_path / run_id
        (run_root / "monitor").mkdir(parents=True)
        (run_root / "logs").mkdir(parents=True)
        (run_root / "results").mkdir(parents=True)
        (run_root / "monitor" / "monitor.log").write_text(f"{_fresh_monitor_timestamp()} running\n")
        (run_root / "logs" / "trail_position_beamstats_progress.jsonl").write_text(
            json.dumps(
                {
                    "event": "trail_position_cache_positive_chunk",
                    "split": "train",
                    "rows_done": 8192,
                    "total_rows": 524288,
                    "class_rows_done": 8192,
                    "class_total": 262144,
                    "chunk_rows": 8192,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    report = summarize_spn_evidence(tmp_path)
    active = report["active_recommendation"]

    assert active["branch"] == "wait_for_trail_position_262k_results"
    assert active["should_launch_remote"] is False
    assert [entry["run_id"] for entry in active["active_runs"]] == [
        "i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706",
        "i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706",
    ]
    assert active["active_runs"][0]["scale"] == "262144/class"
    assert active["active_runs"][0]["progress_summary"]["cache_class_total"] == 262144
    assert "scripts/postprocess-trail-position-result" in active["postprocess_when_ready_command"]
    assert "--expected-score-rows 262144" in active["postprocess_when_ready_command"]
    assert active["decision_report"].endswith(
        "i1_present_r8_trail_position_beamstats_262k_decision_report.md"
    )
    assert "scripts/render-trail-position-report" in active["decision_report_command"]
    assert "i1_present_r8_trail_position_beamstats_262k_postprocess_status.json" in active[
        "decision_report_command"
    ]
    assert "SSH-poll or tmux-loop from the main thread" in active["main_thread_policy"]["forbidden_until_gate"]


def test_summarize_spn_evidence_emits_trail_position_262k_postprocess_when_ready(tmp_path):
    for run_id in [
        "i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706",
        "i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706",
    ]:
        run_root = tmp_path / run_id
        (run_root / "results").mkdir(parents=True)
        score_root = run_root / "score_artifacts"
        (score_root / "global_stats_control").mkdir(parents=True)
        (score_root / "trail_position").mkdir(parents=True)
        (run_root / "results" / "train_matrix.jsonl").write_text("{}\n{}\n", encoding="utf-8")
        (score_root / "global_stats_control" / "models.json").write_text("{}", encoding="utf-8")
        (score_root / "trail_position" / "models.json").write_text("{}", encoding="utf-8")
        (score_root / "verification_summary.json").write_text("{}", encoding="utf-8")

    report = summarize_spn_evidence(tmp_path)
    active = report["active_recommendation"]

    assert active["branch"] == "analyze_trail_position_262k_score_artifacts"
    assert active["status"] == "score_artifacts_ready"
    assert len(active["ready_runs"]) == 2
    assert active["postprocess_command"][0:2] == ["env", "UV_CACHE_DIR=/tmp/uv-cache"]
    assert "scripts/postprocess-trail-position-result" in active["postprocess_command"]
    assert active["postprocess_command"].count("--run-root") == 2
    assert "--expected-score-rows 262144" in active["postprocess_when_ready_command"]
    assert active["decision_report"].endswith(
        "i1_present_r8_trail_position_beamstats_262k_decision_report.md"
    )
    assert "scripts/render-trail-position-report" in active["decision_report_command"]
    assert "i1_present_r8_trail_position_beamstats_262k_postprocess_status.json" in active[
        "decision_report_command"
    ]
    assert "bit_sensitivity_projection_followup" in active
    followup = active["bit_sensitivity_projection_followup"]
    assert followup["status"] == "conditional_after_trail_position_postprocess"
    assert followup["should_launch_remote"] is False
    assert len(followup["per_seed_commands"]) == 2
    assert "scripts/export-bit-sensitivity-features" in followup["per_seed_commands"][0][
        "train_feature_export_command"
    ]
    assert "--split train" in followup["per_seed_commands"][0]["train_feature_export_command"]
    assert "scripts/export-bit-sensitivity-features" in followup["per_seed_commands"][0][
        "validation_feature_export_command"
    ]
    assert "--split validation" in followup["per_seed_commands"][0]["validation_feature_export_command"]
    assert "--reference-artifact" in followup["per_seed_commands"][0]["validation_feature_export_command"]
    assert "scripts/select-bit-sensitivity-projection" in followup["per_seed_commands"][0]["select_mask_command"]
    assert "scripts/apply-bit-sensitivity-projection" in followup["per_seed_commands"][0]["apply_projection_command"]
    assert "scripts/postprocess-bit-sensitivity-projection" in followup["per_seed_commands"][0]["postprocess_gate_command"]
    assert "i1_present_r8_bit_sensitivity_projection_gate_seed0.json" in followup["per_seed_commands"][0][
        "postprocess_gate_command"
    ]
    assert "run frozen-score residual and error-overlap analysis" in active["main_thread_policy"]["allowed_actions"]


def test_summarize_spn_evidence_routes_trail_position_65k_score_repair(tmp_path):
    run_id = "i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706"
    run_root = tmp_path / run_id
    (run_root / "monitor").mkdir(parents=True)
    (run_root / "results").mkdir(parents=True)
    (run_root / "checkpoints").mkdir(parents=True)
    (run_root / "logs").mkdir(parents=True)
    (run_root / "monitor" / "monitor.log").write_text(f"{_fresh_monitor_timestamp()} running\n")
    (run_root / "monitor" / "score_export_repair_launch_failed.marker").write_text("failed\n")
    (run_root / "checkpoints" / "row0001_present_pairset_global_stats_seed0.pt").write_text("checkpoint\n")
    (run_root / "checkpoints" / "row0002_present_trail_position_stats_pairset_seed0.pt").write_text(
        "checkpoint\n"
    )
    (run_root / "results" / "train_matrix.jsonl").write_text(
        json.dumps({"model": "present_pairset_global_stats", "val_auc": 0.9916}) + "\n"
        + json.dumps({"model": "present_trail_position_stats_pairset", "val_auc": 0.9999}) + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(tmp_path)
    active = report["active_recommendation"]

    assert active["branch"] == "repair_trail_position_65k_score_artifacts"
    assert active["status"] == "score_export_repair_failed"
    assert active["repair_runs"][0]["train_rows"] == 2
    assert active["repair_runs"][0]["checkpoint_count"] == 2
    assert active["repair_runs"][0]["missing_score_artifacts"] == [
        "global_stats_control/models.json",
        "trail_position/models.json",
    ]
    assert any(
        "65k/class as medium diagnostic only" in action
        for action in active["main_thread_policy"]["allowed_actions"]
    )
    assert any(
        "treat 65k/class as the main experiment" in action
        for action in active["main_thread_policy"]["forbidden_until_gate"]
    )


def test_summarize_spn_evidence_lists_deferred_running_followups(tmp_path):
    heartbeat = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
    for run_id, progress_name in [
        (
            "i1_present_r9_difference_screen_65k_seed0_gpu0_20260705",
            "r9_difference_screen_progress.jsonl",
        ),
        (
            "i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705",
            "r8_integral_inverse_feature_screen_retry1_progress.jsonl",
        ),
    ]:
        run_root = tmp_path / run_id
        (run_root / "monitor").mkdir(parents=True)
        (run_root / "logs").mkdir(parents=True)
        (run_root / "results").mkdir(parents=True)
        (run_root / "monitor" / "monitor.log").write_text(
            f"{heartbeat} sync\n"
            f"{heartbeat} running\n",
            encoding="utf-8",
        )
        (run_root / "logs" / progress_name).write_text(
            json.dumps({"event": "train_batch", "epoch": 1, "epochs": 20}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_root / "results" / f"{run_id}.jsonl").write_text("", encoding="utf-8")

    report = summarize_spn_evidence(tmp_path)
    active = report["active_recommendation"]

    assert active["branch"] == "wait_for_r9_difference_screen_result"
    assert active["run_id"] == "i1_present_r9_difference_screen_65k_seed0_gpu0_20260705"
    assert active["postprocess_allowed"] is False
    assert active["deferred_active_runs"][0]["run_id"] == (
        "i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705"
    )
    assert active["deferred_active_runs"][0]["branch"] == "wait_for_integral_inverse_feature_result"
    assert "gpu1_retry1" in active["deferred_active_runs"][0]["monitor_health_command"]


def test_integral_inverse_feature_postprocess_writes_summary_and_updates_plan_doc(tmp_path):
    results = tmp_path / "integral_inverse_feature.jsonl"
    _write_integral_inverse_feature_result(
        results,
        architecture="present_nibble_invp_pair_consistency_spn_only",
        architecture_rank=0,
        model="present_nibble_invp_pair_consistency_spn_only",
        feature_encoding="ciphertext_pair_bits",
        auc=0.552,
    )
    _write_integral_inverse_feature_result(
        results,
        architecture="present_matrix_trail_hybrid_pairset_invp",
        architecture_rank=1,
        model="present_matrix_trail_hybrid_pairset_invp",
        feature_encoding="present_pair_xor_paligned_cell_matrix_bits",
        auc=0.556,
    )
    _write_integral_inverse_feature_result(
        results,
        architecture="present_matrix_trail_hybrid_pairset_invp_sinv",
        architecture_rank=2,
        model="present_matrix_trail_hybrid_pairset_invp_sinv",
        feature_encoding="present_pair_xor_paligned_sinv_cell_matrix_bits",
        auc=0.565,
    )
    plan_doc = tmp_path / "integral_inverse_feature_plan.md"
    plan_doc.write_text("# Integral / Inverse-Feature Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_integral_inverse_feature_result(
        results_path=results,
        output_dir=output_dir,
        run_id="integral_inverse_feature_unit",
        expected_rows=3,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "promote_sinv_inverse_feature_to_262k_confirmation"
    assert report["next_action"]["branch"] == "integral_inverse_feature_262k_confirmation"
    assert report["next_action"]["requires_implementation"] is True
    assert report["next_action"]["should_launch_remote"] is False
    assert report["next_action"]["selected_feature_encoding"] == (
        "present_pair_xor_paligned_sinv_cell_matrix_bits"
    )
    assert report["delta_sinv_vs_raw_auc"] == pytest.approx(0.013)
    assert report["delta_sinv_vs_invp_auc"] == pytest.approx(0.009)
    assert Path(report["integral_inverse_feature_gate"]).exists()
    assert Path(report["summary"]).exists()
    assert Path(report["summary_markdown"]).exists()
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "## Retrieved Integral / Inverse-Feature Result" in plan_text
    assert "<!-- integral-inverse-feature-postprocess:integral_inverse_feature_unit:start -->" in plan_text
    assert "| Decision | `promote_sinv_inverse_feature_to_262k_confirmation` |" in plan_text
    assert "| Best feature | `present_pair_xor_paligned_sinv_cell_matrix_bits` |" in plan_text

    postprocess_integral_inverse_feature_result(
        results_path=results,
        output_dir=output_dir,
        run_id="integral_inverse_feature_unit",
        expected_rows=3,
        plan_doc_paths=[plan_doc],
    )
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert plan_text.count("<!-- integral-inverse-feature-postprocess:integral_inverse_feature_unit:start -->") == 1


def test_integral_inverse_feature_postprocess_rejects_incomplete_result_rows(tmp_path):
    results = tmp_path / "integral_inverse_feature_incomplete.jsonl"
    _write_integral_inverse_feature_result(
        results,
        architecture="present_nibble_invp_pair_consistency_spn_only",
        architecture_rank=0,
        model="present_nibble_invp_pair_consistency_spn_only",
        feature_encoding="ciphertext_pair_bits",
        auc=0.552,
    )
    _write_integral_inverse_feature_result(
        results,
        architecture="present_matrix_trail_hybrid_pairset_invp",
        architecture_rank=1,
        model="present_matrix_trail_hybrid_pairset_invp",
        feature_encoding="present_pair_xor_paligned_cell_matrix_bits",
        auc=0.556,
    )
    output_dir = tmp_path / "postprocess"

    report = postprocess_integral_inverse_feature_result(
        results_path=results,
        output_dir=output_dir,
        run_id="integral_inverse_feature_incomplete_unit",
        expected_rows=3,
    )

    assert report["status"] == "fail"
    assert report["validation_status"] == "not_run"
    assert report["integral_inverse_feature_status"] == "fail"
    assert report["decision"] == "invalid_integral_inverse_feature_result"
    assert report["next_action"]["branch"] == "invalid"
    gate = json.loads(Path(report["integral_inverse_feature_gate"]).read_text(encoding="utf-8"))
    assert gate["actual_rows"] == 2
    assert gate["expected_rows"] == 3
    assert any("expected_rows=3 actual_rows=2" in error for error in gate["errors"])


def test_integral_inverse_feature_advance_runs_postprocess_and_writes_summary(tmp_path):
    from blockcipher_nd.planning.integral_inverse_feature_advance import (
        advance_integral_inverse_feature_result,
    )

    results = tmp_path / "integral_inverse_feature.jsonl"
    _write_integral_inverse_feature_result(
        results,
        architecture="present_nibble_invp_pair_consistency_spn_only",
        architecture_rank=0,
        model="present_nibble_invp_pair_consistency_spn_only",
        feature_encoding="ciphertext_pair_bits",
        auc=0.552,
    )
    _write_integral_inverse_feature_result(
        results,
        architecture="present_matrix_trail_hybrid_pairset_invp",
        architecture_rank=1,
        model="present_matrix_trail_hybrid_pairset_invp",
        feature_encoding="present_pair_xor_paligned_cell_matrix_bits",
        auc=0.556,
    )
    _write_integral_inverse_feature_result(
        results,
        architecture="present_matrix_trail_hybrid_pairset_invp_sinv",
        architecture_rank=2,
        model="present_matrix_trail_hybrid_pairset_invp_sinv",
        feature_encoding="present_pair_xor_paligned_sinv_cell_matrix_bits",
        auc=0.565,
    )
    output_dir = tmp_path / "advance"

    report = advance_integral_inverse_feature_result(
        results_path=results,
        output_dir=output_dir,
        run_id="integral_inverse_advance_unit",
        plan_path=None,
        expected_rows=3,
        skip_plot=True,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "promote_sinv_inverse_feature_to_262k_confirmation"
    assert report["plot"] is None
    assert Path(report["summary"]).exists()
    assert Path(report["postprocess_summary"]).exists()


def test_monitor_health_emits_integral_inverse_feature_postprocess_command_when_result_ready(tmp_path):
    run_id = "integral_inverse_feature_monitor_unit"
    run_root = tmp_path / run_id
    monitor_dir = run_root / "monitor"
    results_dir = run_root / "results"
    plan = tmp_path / "integral_inverse_feature_plan.csv"
    monitor_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (monitor_dir / "monitor.log").write_text("2026-07-05T12:00:00+08:00 running\n", encoding="utf-8")
    plan.write_text(
        "model_key\n"
        "present_nibble_invp_pair_consistency_spn_only\n"
        "present_matrix_trail_hybrid_pairset_invp\n"
        "present_matrix_trail_hybrid_pairset_invp_sinv\n",
        encoding="utf-8",
    )
    results = results_dir / f"{run_id}.jsonl"
    _write_integral_inverse_feature_result(
        results,
        architecture="present_nibble_invp_pair_consistency_spn_only",
        architecture_rank=0,
        model="present_nibble_invp_pair_consistency_spn_only",
        feature_encoding="ciphertext_pair_bits",
        auc=0.552,
    )
    _write_integral_inverse_feature_result(
        results,
        architecture="present_matrix_trail_hybrid_pairset_invp",
        architecture_rank=1,
        model="present_matrix_trail_hybrid_pairset_invp",
        feature_encoding="present_pair_xor_paligned_cell_matrix_bits",
        auc=0.556,
    )
    _write_integral_inverse_feature_result(
        results,
        architecture="present_matrix_trail_hybrid_pairset_invp_sinv",
        architecture_rank=2,
        model="present_matrix_trail_hybrid_pairset_invp_sinv",
        feature_encoding="present_pair_xor_paligned_sinv_cell_matrix_bits",
        auc=0.565,
    )

    report = monitor_health_report(
        run_id=run_id,
        root=tmp_path,
        plan_path=plan,
        expected_rows=3,
        postprocess_kind="integral_inverse_feature",
        now=datetime.fromisoformat("2026-07-05T12:01:00+08:00"),
    )

    assert report["status"] == "result_ready"
    assert report["postprocess_allowed"] is True
    assert report["postprocess_command"][0:2] == ["env", "UV_CACHE_DIR=/tmp/uv-cache"]
    assert "scripts/advance-integral-inverse-feature-result" in report["postprocess_command"]
    assert "--expected-rows" in report["postprocess_command"]
    assert "3" in report["postprocess_command"]


def test_r9_weak_probe_postprocess_writes_summary_and_updates_plan_doc(tmp_path):
    results = tmp_path / "r9_weak_probe.jsonl"
    _write_r9_weak_probe_result(results, "present_zhang_wang_keras_mcnd", 0.511)
    _write_r9_weak_probe_result(results, "present_nibble_invp_only_spn_only", 0.519)
    _write_r9_weak_probe_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.556)
    plan_doc = tmp_path / "r9_weak_probe_plan.md"
    plan_doc.write_text("# r9 Weak Probe Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_r9_weak_probe_result(
        results_path=results,
        output_dir=output_dir,
        run_id="r9_weak_probe_unit",
        plan_path=Path("configs/experiment/innovation1/innovation1_spn_present_round_extension_r9_262k_seed0.csv"),
        expected_rows=3,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["validation_status"] == "pass"
    assert report["decision"] == "strong_r9_diagnostic_prepare_1m_seed0"
    assert report["next_action"]["branch"] == "r9_1m_seed0_plan"
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["should_launch_remote"] is True
    assert (
        report["next_action"]["launch_remote_config"]
        == "configs/remote/innovation1_spn_present_r9_1m_seed0_gpu0_20260705.json"
    )
    assert (
        report["next_action"]["suggested_remote_config"]
        == "configs/remote/innovation1_spn_present_r9_1m_seed0_gpu0_20260705.json"
    )
    assert report["next_action"]["run_id"] == "i1_present_r9_1m_seed0_gpu0_20260705"
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    assert report["best_candidate"]["model"] == "present_nibble_invp_pair_consistency_spn_only"
    assert report["candidate_delta_vs_baseline_auc"] == pytest.approx(0.045)
    assert Path(report["r9_weak_probe_gate"]).exists()
    assert Path(report["next_action_readiness"]).exists()
    next_action_readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert next_action_readiness["status"] == "pass"
    assert next_action_readiness["should_launch_remote"] is True
    assert next_action_readiness["readiness_pass"] is True
    assert Path(report["curves"]).exists()
    assert Path(report["history_csv"]).exists()
    assert Path(report["summary"]).exists()
    assert Path(report["summary_markdown"]).exists()
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "## Retrieved r9 Weak-Probe Result" in plan_text
    assert "<!-- r9-weak-probe-postprocess:r9_weak_probe_unit:start -->" in plan_text
    assert "| Decision | `strong_r9_diagnostic_prepare_1m_seed0` |" in plan_text
    assert "| Best candidate | `present_nibble_invp_pair_consistency_spn_only` |" in plan_text

    postprocess_r9_weak_probe_result(
        results_path=results,
        output_dir=output_dir,
        run_id="r9_weak_probe_unit",
        expected_rows=3,
        plan_doc_paths=[plan_doc],
    )
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert plan_text.count("<!-- r9-weak-probe-postprocess:r9_weak_probe_unit:start -->") == 1


def test_r9_curriculum_two_row_postprocess_does_not_require_baseline(tmp_path):
    results = tmp_path / "r9_curriculum.jsonl"
    _write_r9_weak_probe_result(results, "present_nibble_invp_only_spn_only", 0.511)
    _write_r9_weak_probe_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.523)
    plan_doc = tmp_path / "r9_curriculum_plan.md"
    plan_doc.write_text("# r9 Curriculum Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_r9_weak_probe_result(
        results_path=results,
        output_dir=output_dir,
        run_id="r9_curriculum_unit",
        expected_rows=2,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "support_curriculum_followup_or_seed1"
    assert report["baseline"] is None
    assert report["candidate_delta_vs_baseline_auc"] is None
    assert report["best_candidate"]["model"] == "present_nibble_invp_pair_consistency_spn_only"
    assert report["next_action"]["branch"] == "r9_curriculum_positive_review"
    assert report["next_action"]["should_launch_remote"] is False
    assert "r8-to-r9 curriculum diagnostic" in report["claim_scope"]
    assert Path(report["r9_weak_probe_gate"]).exists()
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "| Decision | `support_curriculum_followup_or_seed1` |" in plan_text


def test_r9_postprocess_accepts_paper_scale_claim_scope(tmp_path):
    results = tmp_path / "r9_1m.jsonl"
    _write_r9_weak_probe_result(results, "present_zhang_wang_keras_mcnd", 0.530)
    _write_r9_weak_probe_result(results, "present_nibble_invp_only_spn_only", 0.541)
    _write_r9_weak_probe_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.557)
    claim_scope = "PRESENT r9 1000000/class single-seed paper-scale diagnostic only"

    report = postprocess_r9_weak_probe_result(
        results_path=results,
        output_dir=tmp_path / "r9_1m_postprocess",
        run_id="r9_1m_unit",
        expected_rows=3,
        claim_scope=claim_scope,
    )

    assert report["status"] == "pass"
    assert report["claim_scope"] == claim_scope
    gate = json.loads(Path(report["r9_weak_probe_gate"]).read_text(encoding="utf-8"))
    assert gate["claim_scope"] == claim_scope


def test_r9_weak_probe_weak_positive_points_to_prepared_seed1_assets(tmp_path):
    results = tmp_path / "r9_weak_probe_weak_positive.jsonl"
    _write_r9_weak_probe_result(results, "present_zhang_wang_keras_mcnd", 0.521)
    _write_r9_weak_probe_result(results, "present_nibble_invp_only_spn_only", 0.523)
    _write_r9_weak_probe_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.526)

    report = postprocess_r9_weak_probe_result(
        results_path=results,
        output_dir=tmp_path / "weak_positive_postprocess",
        run_id="r9_weak_probe_weak_positive_unit",
        expected_rows=3,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "r9_weak_positive_prepare_seed1_or_curriculum_scale"
    assert report["next_action"]["branch"] == "r9_seed1_or_curriculum_scale_plan"
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["should_launch_remote"] is True
    assert (
        report["next_action"]["launch_remote_config"]
        == "configs/remote/innovation1_spn_present_r9_weak_probe_262k_seed1_gpu0_20260705.json"
    )
    assert (
        report["next_action"]["suggested_remote_config"]
        == "configs/remote/innovation1_spn_present_r9_weak_probe_262k_seed1_gpu0_20260705.json"
    )
    assert report["next_action"]["run_id"] == "i1_present_r9_weak_probe_262k_seed1_gpu0_20260705"
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    next_action_readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert next_action_readiness["status"] == "pass"
    assert next_action_readiness["should_launch_remote"] is True
    assert next_action_readiness["readiness_pass"] is True


def test_r9_weak_probe_weak_trace_points_to_prepared_curriculum_assets(tmp_path):
    results = tmp_path / "r9_weak_probe_weak_trace.jsonl"
    _write_r9_weak_probe_result(results, "present_zhang_wang_keras_mcnd", 0.508)
    _write_r9_weak_probe_result(results, "present_nibble_invp_only_spn_only", 0.512)
    _write_r9_weak_probe_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.517)

    report = postprocess_r9_weak_probe_result(
        results_path=results,
        output_dir=tmp_path / "weak_trace_postprocess",
        run_id="r9_weak_probe_weak_trace_unit",
        expected_rows=3,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "near_random_r9_weak_trace_check_variance_or_aggregation"
    assert report["next_action"]["branch"] == "r9_variance_or_aggregation_review"
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["should_launch_remote"] is True
    assert (
        report["next_action"]["launch_remote_config"]
        == "configs/remote/innovation1_spn_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.json"
    )
    assert report["next_action"]["run_id"] == "i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705"
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    next_action_readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert next_action_readiness["status"] == "pass"
    assert next_action_readiness["should_launch_remote"] is True
    assert next_action_readiness["readiness_pass"] is True


def test_r9_weak_probe_near_random_points_to_prepared_curriculum_assets(tmp_path):
    results = tmp_path / "r9_weak_probe_near_random.jsonl"
    _write_r9_weak_probe_result(results, "present_zhang_wang_keras_mcnd", 0.504)
    _write_r9_weak_probe_result(results, "present_nibble_invp_only_spn_only", 0.503)
    _write_r9_weak_probe_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.502)

    report = postprocess_r9_weak_probe_result(
        results_path=results,
        output_dir=tmp_path / "near_random_postprocess",
        run_id="r9_weak_probe_near_random_unit",
        expected_rows=3,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "stop_from_scratch_r9_r10_plan_curriculum_or_difference_search"
    assert report["next_action"]["branch"] == "stop_from_scratch_r9_r10"
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["should_launch_remote"] is True
    assert (
        report["next_action"]["launch_remote_config"]
        == "configs/remote/innovation1_spn_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.json"
    )
    assert report["next_action"]["run_id"] == "i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705"
    assert "r8_to_r9_curriculum" in report["next_action"]["fallback_hypotheses"]
    next_action_readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert next_action_readiness["status"] == "pass"
    assert next_action_readiness["should_launch_remote"] is True
    assert next_action_readiness["readiness_pass"] is True


def test_r9_weak_probe_baseline_best_points_to_prepared_difference_screen(tmp_path):
    results = tmp_path / "r9_weak_probe_baseline_best.jsonl"
    _write_r9_weak_probe_result(results, "present_zhang_wang_keras_mcnd", 0.534)
    _write_r9_weak_probe_result(results, "present_nibble_invp_only_spn_only", 0.529)
    _write_r9_weak_probe_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.531)

    report = postprocess_r9_weak_probe_result(
        results_path=results,
        output_dir=tmp_path / "baseline_best_postprocess",
        run_id="r9_weak_probe_baseline_best_unit",
        expected_rows=3,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "baseline_best_or_candidate_not_above_baseline"
    assert report["next_action"]["branch"] == "baseline_best_no_candidate_scale"
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["should_launch_remote"] is True
    assert (
        report["next_action"]["launch_remote_config"]
        == "configs/remote/innovation1_spn_present_r9_difference_screen_65k_seed0_gpu0_20260705.json"
    )
    assert report["next_action"]["run_id"] == "i1_present_r9_difference_screen_65k_seed0_gpu0_20260705"
    assert "r9_difference_screen" in report["next_action"]["fallback_hypotheses"]
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    next_action_readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert next_action_readiness["status"] == "pass"
    assert next_action_readiness["should_launch_remote"] is True
    assert next_action_readiness["readiness_pass"] is True


def test_monitor_health_emits_r9_weak_probe_postprocess_command_when_result_ready(tmp_path):
    run_id = "r9_weak_probe_monitor_unit"
    run_root = tmp_path / run_id
    monitor_dir = run_root / "monitor"
    results_dir = run_root / "results"
    plan = Path("configs/experiment/innovation1/innovation1_spn_present_round_extension_r9_262k_seed0.csv")
    monitor_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (monitor_dir / "monitor.log").write_text("2026-07-05T12:00:00+08:00 running\n", encoding="utf-8")
    results = results_dir / f"{run_id}.jsonl"
    _write_r9_weak_probe_result(results, "present_zhang_wang_keras_mcnd", 0.511)
    _write_r9_weak_probe_result(results, "present_nibble_invp_only_spn_only", 0.519)
    _write_r9_weak_probe_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.556)

    report = monitor_health_report(
        run_id=run_id,
        root=tmp_path,
        plan_path=plan,
        expected_rows=3,
        postprocess_kind="r9_weak_probe",
        now=datetime.fromisoformat("2026-07-05T12:01:00+08:00"),
    )

    assert report["status"] == "result_ready"
    assert report["postprocess_allowed"] is True
    assert report["postprocess_command"][0:2] == ["env", "UV_CACHE_DIR=/tmp/uv-cache"]
    assert "scripts/postprocess-r9-weak-probe" in report["postprocess_command"]
    assert "--expected-rows" in report["postprocess_command"]
    assert "3" in report["postprocess_command"]


def test_r8_pairset_1m_postprocess_writes_summary_and_updates_plan_doc(tmp_path):
    results = tmp_path / "r8_pairset_1m.jsonl"
    _write_r8_pairset_1m_result(results, "present_zhang_wang_keras_mcnd", 0.542)
    _write_r8_pairset_1m_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.549)
    plan_doc = tmp_path / "r8_pairset_1m_plan.md"
    plan_doc.write_text("# r8 Pairset Plan\n", encoding="utf-8")
    output_dir = tmp_path / "postprocess"

    report = postprocess_r8_pairset_1m_result(
        results_path=results,
        output_dir=output_dir,
        run_id="r8_pairset_1m_unit",
        plan_path=Path("configs/experiment/innovation1/innovation1_spn_present_pairset_r8_1m_seed0.csv"),
        expected_rows=2,
        plan_doc_paths=[plan_doc],
    )

    assert report["status"] == "pass"
    assert report["validation_status"] == "pass"
    assert report["decision"] == "support_r8_pairset_1m_confirmation"
    assert report["next_action"]["branch"] == "r8_pairset_seed1_or_frozen_control"
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["should_launch_remote"] is True
    assert (
        report["next_action"]["launch_remote_config"]
        == "configs/remote/innovation1_spn_present_pairset_r8_1m_seed1_gpu1_20260705.json"
    )
    assert (
        report["next_action"]["suggested_remote_config"]
        == "configs/remote/innovation1_spn_present_pairset_r8_1m_seed1_gpu1_20260705.json"
    )
    assert report["next_action"]["run_id"] == "i1_present_r8_pairset_1m_seed1_gpu1_20260705"
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    assert report["delta_vs_baseline_auc"] == pytest.approx(0.007)
    assert Path(report["r8_pairset_1m_gate"]).exists()
    assert Path(report["next_action_readiness"]).exists()
    assert Path(report["candidate_route_readiness"]).exists()
    next_action_readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert next_action_readiness["status"] == "pass"
    assert next_action_readiness["should_launch_remote"] is True
    assert next_action_readiness["readiness_pass"] is True
    candidate_readiness = json.loads(Path(report["candidate_route_readiness"]).read_text(encoding="utf-8"))
    assert candidate_readiness["status"] == "pass"
    assert candidate_readiness["candidate_routes"]["r8_pairset_1m_seed1"]["readiness_pass"] is True
    control_readiness = candidate_readiness["candidate_routes"]["r8_pairset_frozen_aggregation_control"]
    assert control_readiness["readiness_pass"] is True
    assert [item["role"] for item in control_readiness["readiness_reports"]] == ["stage_a", "primary"]
    assert report["next_action"]["control_stage_a_remote_config"].endswith(
        "innovation1_spn_present_pairset_aggregation_control_single_pair_r8_262k_gpu0_20260705.json"
    )
    assert report["next_action"]["control_stage_b_remote_config"].endswith(
        "innovation1_spn_present_pairset_aggregation_control_r8_262k_gpu0_20260705.json"
    )
    assert Path(report["curves"]).exists()
    assert Path(report["history_csv"]).exists()
    assert Path(report["summary"]).exists()
    assert Path(report["summary_markdown"]).exists()
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "## Retrieved r8 Pair-Set 1M Result" in plan_text
    assert "<!-- r8-pairset-1m-postprocess:r8_pairset_1m_unit:start -->" in plan_text
    assert "| Decision | `support_r8_pairset_1m_confirmation` |" in plan_text
    assert "| Delta vs baseline AUC | `0.007000000000` |" in plan_text
    assert "| Candidate route readiness |" in plan_text

    postprocess_r8_pairset_1m_result(
        results_path=results,
        output_dir=output_dir,
        run_id="r8_pairset_1m_unit",
        expected_rows=2,
        plan_doc_paths=[plan_doc],
    )
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert plan_text.count("<!-- r8-pairset-1m-postprocess:r8_pairset_1m_unit:start -->") == 1


def test_r8_pairset_1m_weak_positive_points_to_prepared_seed1_assets(tmp_path):
    results = tmp_path / "r8_pairset_1m_weak_positive.jsonl"
    _write_r8_pairset_1m_result(results, "present_zhang_wang_keras_mcnd", 0.542)
    _write_r8_pairset_1m_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.544)

    report = postprocess_r8_pairset_1m_result(
        results_path=results,
        output_dir=tmp_path / "weak_positive_postprocess",
        run_id="r8_pairset_1m_weak_positive_unit",
        expected_rows=2,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "weak_r8_pairset_1m_positive_needs_seed1_or_controls"
    assert report["next_action"]["branch"] == "r8_pairset_weak_positive_review"
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["should_launch_remote"] is True
    assert (
        report["next_action"]["launch_remote_config"]
        == "configs/remote/innovation1_spn_present_pairset_r8_1m_seed1_gpu1_20260705.json"
    )
    assert (
        report["next_action"]["suggested_remote_config"]
        == "configs/remote/innovation1_spn_present_pairset_r8_1m_seed1_gpu1_20260705.json"
    )
    assert report["next_action"]["run_id"] == "i1_present_r8_pairset_1m_seed1_gpu1_20260705"
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    next_action_readiness = json.loads(Path(report["next_action_readiness"]).read_text(encoding="utf-8"))
    assert next_action_readiness["status"] == "pass"
    assert next_action_readiness["should_launch_remote"] is True
    assert next_action_readiness["readiness_pass"] is True
    candidate_readiness = json.loads(Path(report["candidate_route_readiness"]).read_text(encoding="utf-8"))
    assert candidate_readiness["candidate_routes"]["r8_pairset_frozen_aggregation_control"]["readiness_pass"] is True


def test_monitor_health_emits_r8_pairset_1m_postprocess_command_when_result_ready(tmp_path):
    run_id = "r8_pairset_1m_monitor_unit"
    run_root = tmp_path / run_id
    monitor_dir = run_root / "monitor"
    results_dir = run_root / "results"
    plan = Path("configs/experiment/innovation1/innovation1_spn_present_pairset_r8_1m_seed0.csv")
    monitor_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    (monitor_dir / "monitor.log").write_text("2026-07-05T12:00:00+08:00 running\n", encoding="utf-8")
    results = results_dir / f"{run_id}.jsonl"
    _write_r8_pairset_1m_result(results, "present_zhang_wang_keras_mcnd", 0.542)
    _write_r8_pairset_1m_result(results, "present_nibble_invp_pair_consistency_spn_only", 0.549)

    report = monitor_health_report(
        run_id=run_id,
        root=tmp_path,
        plan_path=plan,
        expected_rows=2,
        postprocess_kind="r8_pairset_1m",
        now=datetime.fromisoformat("2026-07-05T12:01:00+08:00"),
    )

    assert report["status"] == "result_ready"
    assert report["postprocess_allowed"] is True
    assert report["postprocess_command"][0:2] == ["env", "UV_CACHE_DIR=/tmp/uv-cache"]
    assert "scripts/postprocess-r8-pairset-1m" in report["postprocess_command"]
    assert "--expected-rows" in report["postprocess_command"]
    assert "2" in report["postprocess_command"]


def test_arbitrate_next_actions_prefers_strong_r9_over_r8_confirmation(tmp_path):
    r9_results = tmp_path / "r9_weak_probe.jsonl"
    _write_r9_weak_probe_result(r9_results, "present_zhang_wang_keras_mcnd", 0.511)
    _write_r9_weak_probe_result(r9_results, "present_nibble_invp_only_spn_only", 0.519)
    _write_r9_weak_probe_result(r9_results, "present_nibble_invp_pair_consistency_spn_only", 0.556)
    r9_report = postprocess_r9_weak_probe_result(
        results_path=r9_results,
        output_dir=tmp_path / "r9_postprocess",
        run_id="r9_weak_probe_unit",
        expected_rows=3,
    )

    r8_results = tmp_path / "r8_pairset_1m.jsonl"
    _write_r8_pairset_1m_result(r8_results, "present_zhang_wang_keras_mcnd", 0.542)
    _write_r8_pairset_1m_result(r8_results, "present_nibble_invp_pair_consistency_spn_only", 0.549)
    r8_report = postprocess_r8_pairset_1m_result(
        results_path=r8_results,
        output_dir=tmp_path / "r8_postprocess",
        run_id="r8_pairset_1m_unit",
        expected_rows=2,
    )

    arbitration = arbitrate_next_actions(
        [Path(r8_report["summary"]), Path(r9_report["summary"])]
    )

    assert arbitration["status"] == "selected"
    assert arbitration["selected"]["branch"] == "r9_1m_seed0_plan"
    assert arbitration["selected"]["run_id"] == "i1_present_r9_1m_seed0_gpu0_20260705"
    assert arbitration["deferred"][0]["branch"] == "r8_pairset_seed1_or_frozen_control"
    deferred_readiness = arbitration["deferred"][0]["candidate_route_readiness"]
    assert deferred_readiness["status"] == "pass"
    assert (
        deferred_readiness["candidate_routes"]["r8_pairset_frozen_aggregation_control"]["readiness_pass"]
        is True
    )
    assert (
        deferred_readiness["candidate_routes"]["r8_pairset_frozen_aggregation_control"]["readiness_reports"][0][
            "role"
        ]
        == "stage_a"
    )
    assert arbitration["not_ready"] == []


def test_arbitrate_next_actions_marks_invalid_or_nonlaunchable_summaries_not_ready(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json\n", encoding="utf-8")
    r8_results = tmp_path / "r8_pairset_1m_stop.jsonl"
    _write_r8_pairset_1m_result(r8_results, "present_zhang_wang_keras_mcnd", 0.542)
    _write_r8_pairset_1m_result(r8_results, "present_nibble_invp_pair_consistency_spn_only", 0.541)
    r8_report = postprocess_r8_pairset_1m_result(
        results_path=r8_results,
        output_dir=tmp_path / "r8_stop_postprocess",
        run_id="r8_pairset_1m_stop_unit",
        expected_rows=2,
    )

    arbitration = arbitrate_next_actions([invalid, Path(r8_report["summary"])])

    assert arbitration["status"] == "no_launchable_action"
    assert arbitration["selected"] is None
    assert len(arbitration["not_ready"]) == 2
    reasons = {entry["reason"] for entry in arbitration["not_ready"]}
    assert any(reason.startswith("unreadable_or_invalid_summary") for reason in reasons)
    assert "summary_does_not_request_remote_launch" in reasons


def test_active_pattern_auxiliary_head_plan_is_current_not_archived():
    plan = Path("docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md").read_text(encoding="utf-8")

    assert "Status:** launched / watcher-managed medium diagnostic seed0" in plan
    assert "This plan is a current experiment plan" in plan
    assert "not the archived 2026-06-22 standalone" in plan
    assert "present_nibble_invp_active_aux_spn_only" in plan
    assert "present_nibble_invp_active_aux_shuffled_targets" in plan
    assert "zhang_wang_case2_official_mcnd" in plan
    assert "encrypted_random_plaintexts" in plan
    assert "lambda_aux first value = 0.1" in plan
    assert "shuffled_targets" in plan
    assert "active-auxiliary seed0 status = launched / watcher_handoff" in plan
    assert "i1_trail_family_r7_262k_seed0_gpu1_20260702" in plan
    assert "formal route evidence" in plan
    assert "Not allowed" in plan


def test_present_invp_active_aux_model_exposes_main_and_auxiliary_heads():
    model = build_model(
        "present_nibble_invp_active_aux_spn_only",
        input_bits=256,
        hidden_bits=8,
        pair_bits=128,
        structure="spn",
        model_options={"spn_mixer_depth": 1, "activation": "relu", "norm": "layernorm"},
    )
    features = torch.zeros((3, 256), dtype=torch.float32)
    features[0, 0] = 1.0
    features[0, 64] = 0.0
    features[1, 128 + 7] = 1.0
    features[1, 128 + 64 + 7] = 0.0

    logits = model(features)
    aux_logits = model.active_mask_logits(features)

    assert logits.shape == (3, 1)
    assert aux_logits.shape == (3, 2, 16)


def test_present_invp_active_aux_targets_are_deterministic_and_shuffled_controlled():
    from blockcipher_nd.features.spn_active_auxiliary import (
        present_invp_active_mask_targets,
        shuffled_active_mask_targets,
    )

    features = torch.zeros((4, 256), dtype=torch.float32)
    features[0, 0] = 1.0
    features[1, 16] = 1.0
    features[2, 128 + 32] = 1.0
    features[3, 128 + 64 + 48] = 1.0

    targets = present_invp_active_mask_targets(features, pair_bits=128)
    repeat = present_invp_active_mask_targets(features, pair_bits=128)
    shuffled = shuffled_active_mask_targets(targets, seed=123)
    shuffled_repeat = shuffled_active_mask_targets(targets, seed=123)

    assert targets.shape == (4, 2, 16)
    assert torch.equal(targets, repeat)
    assert torch.equal(shuffled, shuffled_repeat)
    assert int(shuffled.sum().item()) == int(targets.sum().item())
    assert not torch.equal(shuffled, targets)


def test_json_plan_alignment_maps_route_specific_short_model_names(tmp_path):
    transition_plan = tmp_path / "transition_plan.json"
    transition_results = tmp_path / "transition_results.jsonl"
    transition_plan.write_text(
        json.dumps(
            {
                "output": str(transition_results),
                "common": {
                    "rounds": 7,
                    "seed": 0,
                    "samples_per_class": 2,
                    "pairs_per_sample": 1,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "zhang_wang_case2_official_mcnd",
                    "difference_profile": "present_zhang_wang2022_mcnd",
                    "difference_member": 0,
                    "key_rotation_interval": 0,
                    "feature_route": "bit_transition_spectrum",
                },
                "rows": [{"model": "linear"}],
            }
        ),
        encoding="utf-8",
    )
    _write_transition_spectrum_result(
        transition_results,
        "bit_transition_spectrum_linear",
        0.5,
        alignment_fields={"rounds": 7, "seed": 0, "samples_per_class": 2},
    )

    transition = validate_result_plan_alignment(transition_plan, transition_results, expected_rows=1)
    assert transition["status"] == "pass"

    trail_plan = tmp_path / "trail_plan.json"
    trail_results = tmp_path / "trail_results.jsonl"
    trail_plan.write_text(
        json.dumps(
            {
                "output": str(trail_results),
                "common": {
                    "rounds": 7,
                    "seed": 0,
                    "samples_per_class": 2,
                    "pairs_per_sample": 1,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "zhang_wang_case2_official_mcnd",
                    "difference_profile": "present_zhang_wang2022_mcnd",
                    "difference_member": 0,
                    "key_rotation_interval": 0,
                    "feature_route": "trail_family_consistency",
                },
                "rows": [{"model": "false_family"}],
            }
        ),
        encoding="utf-8",
    )
    _write_trail_family_result(
        trail_results,
        "trail_family_consistency_false_family",
        0.5,
        alignment_fields={"rounds": 7, "seed": 0, "samples_per_class": 2},
    )

    trail = validate_result_plan_alignment(trail_plan, trail_results, expected_rows=1)
    assert trail["status"] == "pass"


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


def test_transition_spectrum_gate_stops_when_shuffled_control_matches_true_route(tmp_path):
    results = tmp_path / "transition_spectrum_shuffled_matches.jsonl"
    _write_transition_spectrum_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_linear", 0.7925)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_mlp", 0.7934)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_shuffled_p", 0.7934)

    report = gate_transition_spectrum_result(results, expected_rows=4, require_shuffled_control=True)

    assert report["status"] == "pass"
    assert report["decision"] == "stop_transition_spectrum_route"
    assert report["margin_vs_anchor_auc"] > 0.001
    assert report["margin_vs_shuffled_auc"] == 0
    assert "shuffled-P control matches/exceeds" in report["interpretation"]


def test_transition_spectrum_gate_can_require_shuffled_control(tmp_path):
    results = tmp_path / "transition_spectrum_missing_shuffled.jsonl"
    _write_transition_spectrum_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_linear", 0.7925)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_mlp", 0.7940)

    diagnostic = gate_transition_spectrum_result(results, expected_rows=3)
    assert diagnostic["status"] == "pass"
    assert diagnostic["decision"] == "weak_transition_spectrum_signal"
    assert diagnostic["warnings"] == ["missing_shuffled_control=bit_transition_spectrum_shuffled_p"]

    strict = gate_transition_spectrum_result(results, expected_rows=3, require_shuffled_control=True)
    assert strict["status"] == "fail"
    assert strict["decision"] == "invalid"
    assert strict["errors"] == ["missing_shuffled_control=bit_transition_spectrum_shuffled_p"]


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
    assert report["next_action"]["should_launch_remote"] is True
    assert report["next_action"]["requires_implementation"] is False
    assert report["next_action"]["next_plan_doc"] == "docs/experiments/innovation1-bit-transition-spectrum-plan.md"
    assert "bit_transition_spectrum_r7_262k_seed1" in report["next_action"]["suggested_plan_config"]
    assert "bit_transition_spectrum_r7_262k_seed1" in report["next_action"]["launch_remote_config"]
    assert report["next_action"]["suggested_feature_cache_workers"] == 4
    assert "scripts/check-remote-readiness" in report["next_action"]["readiness_command"]
    assert any("seed1 confirmation" in step for step in report["next_steps"])
    assert (output_dir / "transition_spectrum_unit_transition_spectrum_gate.json").exists()
    assert (output_dir / "transition_spectrum_unit_postprocess_summary.json").exists()
    assert (output_dir / "transition_spectrum_unit_postprocess_summary.md").exists()
    readiness_path = output_dir / "transition_spectrum_unit_next_action_readiness.json"
    assert readiness_path.exists()
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["status"] == "pass"
    assert readiness["branch"] == "transition_spectrum_seed1_confirmation"
    assert readiness["should_launch_remote"] is True
    assert readiness["requires_implementation"] is False
    assert readiness["readiness_pass"] is True
    assert readiness["remote_readiness_pass"] is True
    assert readiness["launch_artifacts_pass"] is True
    assert readiness["implementation_checklist"] == []
    assert readiness["readiness_reports"][0]["launch_artifacts"]["status"] == "pass"
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


def test_transition_spectrum_postprocess_requires_shuffled_control_by_default(tmp_path):
    results = tmp_path / "transition_spectrum_missing_shuffled.jsonl"
    _write_transition_spectrum_result(results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_linear", 0.7925)
    _write_transition_spectrum_result(results, "bit_transition_spectrum_mlp", 0.7940)
    output_dir = tmp_path / "postprocess"

    report = postprocess_transition_spectrum_result(
        results_path=results,
        output_dir=output_dir,
        run_id="transition_spectrum_missing_shuffled_unit",
        expected_rows=3,
    )

    assert report["status"] == "fail"
    assert report["transition_spectrum_status"] == "fail"
    assert report["decision"] == "invalid"
    assert report["require_shuffled_control"] is True
    gate = json.loads((output_dir / "transition_spectrum_missing_shuffled_unit_transition_spectrum_gate.json").read_text())
    assert gate["errors"] == ["missing_shuffled_control=bit_transition_spectrum_shuffled_p"]

    diagnostic = postprocess_transition_spectrum_result(
        results_path=results,
        output_dir=output_dir / "diagnostic",
        run_id="transition_spectrum_missing_shuffled_diagnostic",
        expected_rows=3,
        require_shuffled_control=False,
    )
    assert diagnostic["status"] == "pass"
    assert diagnostic["decision"] == "weak_transition_spectrum_signal"
    assert diagnostic["require_shuffled_control"] is False


def test_transition_spectrum_postprocess_weak_and_stop_expose_next_paths(tmp_path):
    weak_results = tmp_path / "transition_spectrum_weak.jsonl"
    _write_transition_spectrum_result(weak_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(weak_results, "bit_transition_spectrum_linear", 0.7922)
    _write_transition_spectrum_result(weak_results, "bit_transition_spectrum_mlp", 0.7924)
    _write_transition_spectrum_result(weak_results, "bit_transition_spectrum_shuffled_p", 0.7923)

    weak = postprocess_transition_spectrum_result(
        results_path=weak_results,
        output_dir=tmp_path / "weak_postprocess",
        run_id="transition_spectrum_weak_unit",
        expected_rows=4,
    )

    assert weak["decision"] == "weak_transition_spectrum_signal"
    assert weak["next_action"]["branch"] == "transition_spectrum_variance_check"
    assert weak["next_action"]["should_launch_remote"] is True
    assert weak["next_action"]["requires_implementation"] is False
    assert "bit_transition_spectrum_r7_262k_seed1" in weak["next_action"]["suggested_plan_config"]
    assert "bit_transition_spectrum_r7_262k_seed1" in weak["next_action"]["launch_remote_config"]
    assert weak["next_action"]["suggested_feature_cache_workers"] == 4
    assert "scripts/check-remote-readiness" in weak["next_action"]["readiness_command"]
    assert any("seed1 variance check" in step for step in weak["next_steps"])
    assert Path(weak["next_action_readiness"]).exists()
    weak_readiness = json.loads(Path(weak["next_action_readiness"]).read_text(encoding="utf-8"))
    assert weak_readiness["status"] == "pass"
    assert weak_readiness["readiness_pass"] is True
    assert weak_readiness["remote_readiness_pass"] is True
    assert weak_readiness["launch_artifacts_pass"] is True
    assert weak_readiness["readiness_reports"][0]["launch_artifacts"]["status"] == "pass"

    stop_results = tmp_path / "transition_spectrum_stop.jsonl"
    _write_transition_spectrum_result(stop_results, "present_nibble_invp_only_spn_only", 0.7920)
    _write_transition_spectrum_result(stop_results, "bit_transition_spectrum_linear", 0.7910)
    _write_transition_spectrum_result(stop_results, "bit_transition_spectrum_mlp", 0.7915)
    _write_transition_spectrum_result(stop_results, "bit_transition_spectrum_shuffled_p", 0.7917)

    stop = postprocess_transition_spectrum_result(
        results_path=stop_results,
        output_dir=tmp_path / "stop_postprocess",
        run_id="transition_spectrum_stop_unit",
        expected_rows=4,
    )

    assert stop["decision"] == "stop_transition_spectrum_route"
    assert stop["next_action"]["branch"] == "stop_transition_spectrum_route"
    assert stop["next_action"]["should_launch_remote"] is True
    assert stop["next_action"]["requires_implementation"] is False
    assert stop["next_action"]["fallback_branch"] == "trail_family_seed0"
    assert "trail_family_r7_262k_seed0" in stop["next_action"]["suggested_plan_config"]
    assert "trail_family_r7_262k_seed0" in stop["next_action"]["launch_remote_config"]
    assert "trail_family_consistency" in stop["next_action"]["fallback_hypotheses"]
    assert "docs/experiments/innovation1-trail-family-consistency-plan.md" in stop["next_action"]["fallback_plan_options"]
    assert "docs/research/spn_structured_nn_research_plan.md" in stop["next_action"]["fallback_plan_options"]
    assert any("trail-family seed0 fallback" in step for step in stop["next_steps"])
    stop_readiness = json.loads(Path(stop["next_action_readiness"]).read_text(encoding="utf-8"))
    assert stop_readiness["branch"] == "stop_transition_spectrum_route"
    assert stop_readiness["requires_implementation"] is False
    assert stop_readiness["readiness_pass"] is True
    assert stop_readiness["readiness_reports"][0]["readiness"]["status"] == "pass"
    assert stop_readiness["remote_readiness_pass"] is True
    assert stop_readiness["launch_artifacts_pass"] is True
    assert stop_readiness["readiness_reports"][0]["launch_artifacts"]["status"] == "pass"


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
    assert active["branch"] == "trail_family_seed0"
    assert active["should_launch_remote"] is False
    assert active["next_action"]["should_launch_remote"] is True
    assert active["next_action"]["requires_implementation"] is False
    assert active["next_action"]["next_plan_doc"] == (
        "docs/experiments/innovation1-trail-family-consistency-plan.md"
    )
    assert "trail_family_r7_262k_seed0" in active["next_action"]["launch_remote_config"]
    assert "docs/experiments/innovation1-trail-family-consistency-plan.md" in active["next_action"][
        "fallback_plan_options"
    ]


def test_summarize_spn_evidence_tracks_running_trail_family(tmp_path):
    root = tmp_path / "remote_results"
    transition = root / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702"
    trail = root / "i1_trail_family_r7_262k_seed0_gpu1_20260702"
    transition.mkdir(parents=True)
    (trail / "monitor").mkdir(parents=True)
    (trail / "logs").mkdir(parents=True)
    _write_test_json(
        transition / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_transition_spectrum_route",
            "claim_scope": "bit-transition-spectrum medium diagnostic gate",
        },
    )
    (trail / "monitor" / "monitor.log").write_text("2026-07-03T17:50:30+08:00 running\n")
    (trail / "logs" / "trail_family_linear_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "trail_family_negative_chunk",
                "split": "train",
                "time": 1000.0,
                "rows_done": 360448,
                "total_rows": 524288,
                "class_rows_done": 98304,
                "class_total": 262144,
                "chunk_rows": 8192,
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "trail_family_cache_start",
                "split": "validation",
                "time": 1100.0,
                "total_rows": 131072,
                "samples_per_class": 65536,
                "chunk_size": 8192,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "wait_for_trail_family_result"
    assert active["run_id"] == "i1_trail_family_r7_262k_seed0_gpu1_20260702"
    assert active["postprocess_allowed"] is False
    assert active["progress_summary"]["latest_event"] == "trail_family_cache_start"
    assert active["progress_summary"]["latest_split"] == "validation"
    assert active["progress_summary"]["latest_total_rows"] == 131072
    assert active["progress_summary"]["latest_samples_per_class"] == 65536
    assert active["progress_summary"]["cache_event"] == "trail_family_negative_chunk"
    assert active["progress_summary"]["cache_split"] == "train"
    assert active["progress_summary"]["cache_rows_remaining"] == 163840
    assert "monitor_i1_trail_family_seed0_20260702" in active["monitor_health_command"]
    assert "scripts/postprocess-trail-family" in active["postprocess_when_ready_command"]
    assert "launch trail-family seed1" in active["main_thread_policy"]["forbidden_until_gate"]
    assert "launch active-auxiliary seed0" in active["main_thread_policy"]["forbidden_until_gate"]
    assert "launch S-box transition prior seed0" in active["main_thread_policy"]["forbidden_until_gate"]


def test_summarize_spn_evidence_lists_deferred_candidates_while_trail_family_runs(tmp_path):
    root = tmp_path / "remote_results"
    trail = root / "i1_trail_family_r7_262k_seed0_gpu1_20260702"
    (trail / "monitor").mkdir(parents=True)
    (trail / "logs").mkdir(parents=True)
    (trail / "monitor" / "monitor.log").write_text("2026-07-03T17:50:30+08:00 running\n")
    (trail / "logs" / "trail_family_linear_progress.jsonl").write_text(
        json.dumps({"event": "trail_family_negative_chunk", "rows_done": 360448, "total_rows": 524288})
        + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    active = report["active_recommendation"]
    assert active["branch"] == "wait_for_trail_family_result"
    assert active["should_launch_remote"] is False
    assert active["conditional_followup"]["fallback_if_stop"] == "active_auxiliary_seed0"
    assert active["conditional_followup"]["fallback_after_active_auxiliary_stop"] == (
        "sbox_transition_prior_gate_seed0"
    )
    assert active["deferred_candidate_queue"] == [
        {
            "branch": "trail_family_seed1_confirmation_or_variance_check",
            "launch_gate": "support_trail_family_route or weak_trail_family_signal",
            "run_id": "i1_trail_family_r7_262k_seed1_gpu1_20260702",
            "launch_remote_config": (
                "configs/remote/innovation1_spn_present_trail_family_r7_262k_seed1_gpu1_20260702.json"
            ),
            "plan_doc": "docs/experiments/innovation1-trail-family-consistency-plan.md",
            "readiness_status": "pass",
            "status": "prepared_conditional",
        },
        {
            "branch": "active_auxiliary_seed0",
            "launch_gate": "stop/tied trail-family gate or explicit user selection",
            "run_id": "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703",
            "launch_remote_config": (
                "configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_20260703.json"
            ),
            "plan_doc": "docs/experiments/innovation1-active-pattern-auxiliary-head-plan.md",
            "readiness_status": "pass",
            "status": "prepared_deferred",
        },
        {
            "branch": "sbox_transition_prior_gate_seed0",
            "launch_gate": "stop/tied trail-family gate after active-auxiliary is not selected",
            "run_id": "i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703",
            "launch_remote_config": (
                "configs/remote/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0_gpu1_20260703.json"
            ),
            "plan_doc": "docs/experiments/innovation1-sbox-transition-prior-gate-plan.md",
            "readiness_status": "pass",
            "status": "prepared_deferred",
        },
    ]


def test_summarize_spn_evidence_tracks_running_active_auxiliary_after_trail_family_stop(tmp_path):
    root = tmp_path / "remote_results"
    transition = root / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702"
    trail = root / "i1_trail_family_r7_262k_seed0_gpu1_20260702"
    active = root / "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    transition.mkdir(parents=True)
    trail.mkdir(parents=True)
    (active / "monitor").mkdir(parents=True)
    (active / "logs").mkdir(parents=True)
    _write_test_json(
        transition / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_transition_spectrum_route",
            "claim_scope": "bit-transition-spectrum medium diagnostic gate",
        },
    )
    _write_test_json(
        trail / "i1_trail_family_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_trail_family_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_trail_family_route",
            "claim_scope": "trail-family medium diagnostic gate",
        },
    )
    (active / "monitor" / "monitor.log").write_text(f"{_fresh_monitor_timestamp()} running\n")
    _write_active_auxiliary_launch_artifacts(active)
    (active / "logs" / "active_auxiliary_progress.jsonl").write_text(
        json.dumps({"event": "active_auxiliary_cache_start", "split": "train", "total_rows": 524288})
        + "\n"
        + json.dumps(
            {
                "event": "active_auxiliary_positive_chunk",
                "split": "train",
                "time": 1000.0,
                "rows_done": 65536,
                "total_rows": 524288,
                "class_rows_done": 65536,
                "class_total": 262144,
                "chunk_rows": 8192,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    recommendation = report["active_recommendation"]
    assert recommendation["branch"] == "wait_for_active_auxiliary_result"
    assert recommendation["run_id"] == "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    assert recommendation["should_launch_remote"] is False
    assert recommendation["postprocess_allowed"] is False
    assert recommendation["progress_summary"]["latest_event"] == "active_auxiliary_positive_chunk"
    assert recommendation["progress_summary"]["cache_class_rows_done"] == 65536
    assert "monitor_i1_active_auxiliary_seed0_20260703" in recommendation["monitor_health_command"]
    assert "scripts/postprocess-active-auxiliary" in recommendation["postprocess_when_ready_command"]
    assert "--expected-rows 3" in recommendation["postprocess_when_ready_command"]
    assert recommendation["conditional_followup"]["fallback_if_stop"] == "sbox_transition_prior_gate_seed0"
    assert recommendation["conditional_followup"]["fallback_run_id"] == "i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703"
    assert "launch S-box transition prior seed0" in recommendation["main_thread_policy"]["forbidden_until_gate"]


def test_summarize_spn_evidence_keeps_active_auxiliary_with_progress_without_running_log(tmp_path):
    root = tmp_path / "remote_results"
    transition = root / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702"
    trail = root / "i1_trail_family_r7_262k_seed0_gpu1_20260702"
    active = root / "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    transition.mkdir(parents=True)
    trail.mkdir(parents=True)
    (active / "monitor").mkdir(parents=True)
    (active / "logs").mkdir(parents=True)
    _write_test_json(
        transition / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_transition_spectrum_route",
            "claim_scope": "bit-transition-spectrum medium diagnostic gate",
        },
    )
    _write_test_json(
        trail / "i1_trail_family_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_trail_family_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_trail_family_route",
            "claim_scope": "trail-family medium diagnostic gate",
        },
    )
    (active / "monitor" / "monitor.log").write_text(f"{_fresh_monitor_timestamp()} sync\n")
    _write_active_auxiliary_launch_artifacts(active)
    (active / "logs" / "active_auxiliary_progress.jsonl").write_text(
        json.dumps({"event": "active_auxiliary_cache_start", "split": "train", "total_rows": 524288}) + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    recommendation = report["active_recommendation"]
    assert recommendation["branch"] == "wait_for_active_auxiliary_result"
    assert recommendation["run_id"] == "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    assert recommendation["progress_summary"]["latest_event"] == "active_auxiliary_cache_start"
    assert recommendation["needs_main_thread_intervention"] is False


def test_summarize_spn_evidence_keeps_failed_active_auxiliary_as_active_branch(tmp_path):
    root = tmp_path / "remote_results"
    transition = root / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702"
    trail = root / "i1_trail_family_r7_262k_seed0_gpu1_20260702"
    active = root / "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    transition.mkdir(parents=True)
    trail.mkdir(parents=True)
    (active / "monitor").mkdir(parents=True)
    (active / "logs").mkdir(parents=True)
    _write_test_json(
        transition / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_transition_spectrum_route",
            "claim_scope": "bit-transition-spectrum medium diagnostic gate",
        },
    )
    _write_test_json(
        trail / "i1_trail_family_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_trail_family_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_trail_family_route",
            "claim_scope": "trail-family medium diagnostic gate",
        },
    )
    (active / "monitor" / "monitor.log").write_text(f"{_fresh_monitor_timestamp()} failed\n")
    _write_active_auxiliary_launch_artifacts(active)
    (active / "logs" / f"{active.name}_failed.marker").write_text("failed\n", encoding="utf-8")
    (active / "logs" / "active_auxiliary_progress.jsonl").write_text(
        json.dumps(
            {
                "event": "active_auxiliary_positive_chunk",
                "split": "train",
                "rows_done": 40960,
                "total_rows": 524288,
                "class_rows_done": 40960,
                "class_total": 262144,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    recommendation = report["active_recommendation"]
    assert recommendation["branch"] == "active_auxiliary_failed"
    assert recommendation["run_id"] == "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    assert recommendation["status"] == "failed"
    assert recommendation["should_launch_remote"] is False
    assert recommendation["postprocess_allowed"] is False
    assert recommendation["progress_summary"]["cache_class_rows_done"] == 40960
    assert "launch active-auxiliary seed1" in recommendation["main_thread_policy"]["forbidden_until_gate"]
    assert "launch S-box transition prior seed0" in recommendation["main_thread_policy"]["forbidden_until_gate"]


def test_summarize_spn_evidence_prefers_active_auxiliary_retry_over_stale_failed_original(tmp_path):
    root = tmp_path / "remote_results"
    transition = root / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702"
    trail = root / "i1_trail_family_r7_262k_seed0_gpu1_20260702"
    failed_original = root / "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    retry = root / "i1_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704"
    transition.mkdir(parents=True)
    trail.mkdir(parents=True)
    (failed_original / "monitor").mkdir(parents=True)
    (failed_original / "logs").mkdir(parents=True)
    (retry / "monitor").mkdir(parents=True)
    (retry / "logs").mkdir(parents=True)
    _write_test_json(
        transition / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_transition_spectrum_route",
            "claim_scope": "bit-transition-spectrum medium diagnostic gate",
        },
    )
    _write_test_json(
        trail / "i1_trail_family_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_trail_family_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_trail_family_route",
            "claim_scope": "trail-family medium diagnostic gate",
        },
    )
    (failed_original / "monitor" / "monitor.log").write_text(f"{_fresh_monitor_timestamp()} failed\n")
    _write_active_auxiliary_launch_artifacts(failed_original)
    (failed_original / "logs" / f"{failed_original.name}_failed.marker").write_text("failed\n", encoding="utf-8")
    (retry / "monitor" / "monitor.log").write_text(f"{_fresh_monitor_timestamp()} running\n")
    _write_active_auxiliary_launch_artifacts(retry)

    report = summarize_spn_evidence(root)

    recommendation = report["active_recommendation"]
    assert recommendation["branch"] == "wait_for_active_auxiliary_result"
    assert recommendation["run_id"] == "i1_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704"
    assert recommendation["needs_main_thread_intervention"] is False
    assert "seed0_retry1" in recommendation["monitor_health_command"]
    assert "monitor_i1_active_auxiliary_seed0_retry1_20260704" in recommendation["monitor_health_command"]


def test_summarize_spn_evidence_tracks_running_sbox_prior_after_active_auxiliary_stop(tmp_path):
    root = tmp_path / "remote_results"
    active_retry = root / "i1_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704"
    stale_active_original = root / "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    sbox = root / "i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703"
    (active_retry).mkdir(parents=True)
    (stale_active_original / "monitor").mkdir(parents=True)
    (stale_active_original / "logs").mkdir(parents=True)
    (sbox / "monitor").mkdir(parents=True)
    (sbox / "logs").mkdir(parents=True)
    _write_test_json(
        active_retry / "i1_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704_postprocess_summary.json",
        {
            "run_id": "i1_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_active_auxiliary_route",
            "claim_scope": "active-pattern auxiliary diagnostic gate",
            "next_action": {
                "branch": "sbox_transition_prior_gate_seed0",
                "should_launch_remote": True,
            },
        },
    )
    (stale_active_original / "monitor" / "monitor.log").write_text(
        f"{_fresh_monitor_timestamp()} running\n",
        encoding="utf-8",
    )
    _write_active_auxiliary_launch_artifacts(stale_active_original)
    (sbox / "monitor" / "monitor.log").write_text(f"{_fresh_monitor_timestamp()} running\n", encoding="utf-8")
    _write_sbox_prior_launch_artifacts(sbox)
    (sbox / "logs" / "sbox_prior_progress.jsonl").write_text(
        json.dumps({"event": "cache_start", "split": "train", "total_rows": 524288}) + "\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    recommendation = report["active_recommendation"]
    assert recommendation["branch"] == "wait_for_sbox_prior_result"
    assert recommendation["run_id"] == "i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703"
    assert recommendation["should_launch_remote"] is False
    assert recommendation["postprocess_allowed"] is False
    assert recommendation["progress_summary"]["latest_event"] == "cache_start"
    assert "monitor_i1_sbox_prior_gate_seed0_20260704" in recommendation["monitor_health_command"]
    assert "scripts/postprocess-sbox-prior" in recommendation["postprocess_when_ready_command"]
    assert "--expected-rows 4" in recommendation["postprocess_when_ready_command"]
    assert "launch S-box transition prior seed1" in recommendation["main_thread_policy"]["forbidden_until_gate"]


def _write_active_auxiliary_launch_artifacts(run_root: Path) -> None:
    run_id = run_root.name
    logs = run_root / "logs"
    (logs / f"{run_id}_started.marker").write_text("started\n", encoding="utf-8")
    (logs / f"{run_id}_torch_info.txt").write_text("cuda available\n", encoding="utf-8")
    (logs / f"{run_id}_git_revision.txt").write_text("abc123\n", encoding="utf-8")
    (logs / f"{run_id}_readiness.txt").write_text('{"status":"pass"}\n', encoding="utf-8")
    (logs / f"{run_id}_stdout.txt").write_text("", encoding="utf-8")


def _write_sbox_prior_launch_artifacts(run_root: Path) -> None:
    run_id = run_root.name
    logs = run_root / "logs"
    (logs / f"{run_id}_started.marker").write_text("started\n", encoding="utf-8")
    (logs / f"{run_id}_torch_info.txt").write_text("cuda available\n", encoding="utf-8")
    (logs / f"{run_id}_git_revision.txt").write_text("abc123\n", encoding="utf-8")
    (logs / f"{run_id}_readiness.txt").write_text('{"status":"pass"}\n', encoding="utf-8")
    (logs / f"{run_id}_stdout.txt").write_text("", encoding="utf-8")


def _fresh_monitor_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def test_summarize_spn_evidence_keeps_active_auxiliary_when_remote_artifacts_missing(tmp_path):
    root = tmp_path / "remote_results"
    transition = root / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702"
    active = root / "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    transition.mkdir(parents=True)
    (active / "monitor").mkdir(parents=True)
    _write_test_json(
        transition / "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702_postprocess_summary.json",
        {
            "run_id": "i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702",
            "status": "pass",
            "validation_status": "pass",
            "decision": "stop_transition_spectrum_route",
            "claim_scope": "bit-transition-spectrum medium diagnostic gate",
        },
    )
    (active / "monitor" / "monitor.log").write_text(
        "2026-07-04T09:42:37+08:00 sync\n"
        "2026-07-04T09:42:38+08:00 running\n"
        "2026-07-04T09:56:52+08:00 sync\n"
        "2026-07-04T09:56:53+08:00 running\n",
        encoding="utf-8",
    )
    (active / "monitor" / "scp_stderr.log").write_text(
        "scp: G:/lxy/blockcipher-structure-adaptive-nd-runs/"
        "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703/logs: No such file or directory\n"
        "scp: G:/lxy/blockcipher-structure-adaptive-nd-runs/"
        "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703/results: No such file or directory\n"
        "scp: G:/lxy/blockcipher-structure-adaptive-nd-runs/"
        "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703/logs: No such file or directory\n"
        "scp: G:/lxy/blockcipher-structure-adaptive-nd-runs/"
        "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703/results: No such file or directory\n",
        encoding="utf-8",
    )

    report = summarize_spn_evidence(root)

    recommendation = report["active_recommendation"]
    assert recommendation["branch"] == "diagnose_active_auxiliary_launch"
    assert recommendation["run_id"] == "i1_active_auxiliary_r7_262k_seed0_gpu1_20260703"
    assert recommendation["status"] == "remote_artifacts_missing"
    assert recommendation["needs_main_thread_intervention"] is True
    assert recommendation["should_launch_remote"] is False
    assert recommendation["conditional_followup"]["fallback_if_stop"] == "sbox_transition_prior_gate_seed0"
    assert "launch S-box transition prior seed0" in recommendation["main_thread_policy"]["forbidden_until_gate"]


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


def test_integral_parity_audit_detects_plaintext_integral_pair_xor_signal():
    plan = "configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_smoke.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]

    report = integral_parity_audit_from_task(
        task,
        samples_per_class=8,
        seed=7,
        key_split="validation",
    )

    assert report["status"] == "pass"
    assert report["audit"] == "integral_pair_xor_parity"
    assert report["sample_structure"] == "plaintext_integral_nibble"
    assert report["negative_mode"] == "encrypted_random_plaintexts"
    assert report["positive_pair_xor_parity_hw"]["zero_rate"] == 1.0
    assert report["negative_pair_xor_parity_hw"]["zero_rate"] == 0.0
    assert report["best_threshold"] == {
        "accuracy": 1.0,
        "threshold": 0,
        "operator": "<=",
    }
    assert report["interpretation"] == "parity_statistic_alone_nearly_separates_classes"


def test_integral_parity_audit_matched_negative_removes_pair_xor_separator():
    plan = "configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_smoke.csv"
    args = parse_args(["--plan", plan])
    task = dict(build_tasks(args)[0])
    task["sample_structure"] = "plaintext_integral_nibble_matched_negative"

    report = integral_parity_audit_from_task(
        task,
        samples_per_class=8,
        seed=7,
        key_split="validation",
    )

    assert report["status"] == "pass"
    assert report["sample_structure"] == "plaintext_integral_nibble_matched_negative"
    assert report["positive_pair_xor_parity_hw"]["zero_rate"] == 1.0
    assert report["negative_pair_xor_parity_hw"]["zero_rate"] == 1.0
    assert report["best_threshold"]["accuracy"] == 0.5
    assert report["interpretation"] == "parity_statistic_does_not_explain_result_by_itself"


def test_integral_alignment_audit_reports_pair_order_statistics():
    plan = "configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_smoke.csv"
    args = parse_args(["--plan", plan])
    task = dict(build_tasks(args)[0])
    task["sample_structure"] = "plaintext_integral_nibble_matched_negative"

    report = integral_alignment_audit_from_task(
        task,
        samples_per_class=32,
        seed=7,
        key_split="validation",
    )

    assert report["audit"] == "integral_pair_alignment"
    assert report["sample_structure"] == "plaintext_integral_nibble_matched_negative"
    assert set(report["statistics"]) == {
        "same_index_xor_hw_mean",
        "shifted_index_xor_hw_mean",
        "same_minus_shifted_xor_hw_mean",
    }
    for statistic in report["statistics"].values():
        assert statistic["positive"]["count"] == 32
        assert statistic["negative"]["count"] == 32
        assert 0.5 <= statistic["best_threshold"]["accuracy"] <= 1.0
    assert report["best_statistic"]["name"] in report["statistics"]
    assert "Deterministic local data-structure audit only" in report["claim_scope"]


def test_integral_feature_bank_audit_reports_named_deterministic_statistics():
    plan = "configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_smoke.csv"
    args = parse_args(["--plan", plan])
    task = dict(build_tasks(args)[0])
    task["sample_structure"] = "plaintext_integral_nibble_matched_negative"

    report = integral_feature_bank_audit_from_task(
        task,
        samples_per_class=32,
        seed=7,
        key_split="validation",
    )

    assert report["audit"] == "integral_feature_bank"
    assert report["sample_structure"] == "plaintext_integral_nibble_matched_negative"
    assert {
        "left_hw_mean",
        "right_hw_mean",
        "pair_xor_hw_mean",
        "left_column_sum_variance",
        "right_column_sum_variance",
        "pair_xor_column_sum_variance",
        "left_right_column_sum_l1_mean",
    }.issubset(report["statistics"])
    for statistic in report["statistics"].values():
        assert statistic["positive"]["count"] == 32
        assert statistic["negative"]["count"] == 32
        assert 0.5 <= statistic["best_threshold"]["accuracy"] <= 1.0
    assert report["best_statistic"]["name"] in report["statistics"]
    assert report["best_statistic"]["interpretation"].startswith("deterministic_statistic_")


def test_integral_feature_bank_audit_accepts_same_difference_random_negative():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_same_difference_audit_2048_seed0_seed1.csv"
    )
    args = parse_args(["--plan", plan])
    task = dict(build_tasks(args)[0])
    task["feature_encoding"] = "ciphertext_pair_bits"

    report = integral_feature_bank_audit_from_task(
        task,
        samples_per_class=16,
        seed=7,
        key_split="validation",
    )

    assert report["audit"] == "integral_feature_bank"
    assert report["sample_structure"] == "plaintext_integral_nibble_same_difference_random_negative"
    assert report["feature_encoding"] == "ciphertext_pair_bits"
    assert report["best_statistic"]["name"] in report["statistics"]


def test_integral_deterministic_baseline_reports_fixed_pair_xor_variance():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_integral_aligned_difference_control_smoke.csv"
    )
    args = parse_args(["--plan", plan])
    task = dict(build_tasks(args)[0])

    report = integral_deterministic_baseline_from_task(
        task,
        statistic="pair_xor_column_sum_variance",
        samples_per_class=32,
        seed=23,
        key_split="validation",
    )

    assert report["audit"] == "integral_deterministic_baseline"
    assert report["statistic_name"] == "pair_xor_column_sum_variance"
    assert "best_statistic" not in report
    assert report["baseline"]["positive"]["count"] == 32
    assert report["baseline"]["negative"]["count"] == 32
    assert 0.5 <= report["baseline"]["best_threshold"]["accuracy"] <= 1.0
    assert 0.5 <= report["baseline"]["auc"] <= 1.0
    assert report["interpretation"].startswith("deterministic_statistic_")
    assert "not a neural training result" in report["claim_scope"]


def test_integral_composite_residual_audit_compares_against_fixed_baseline():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_integral_aligned_difference_control_smoke.csv"
    )
    args = parse_args(["--plan", plan])
    task = dict(build_tasks(args)[0])

    report = integral_composite_residual_audit_from_task(
        task,
        baseline_statistic="pair_xor_column_sum_variance",
        samples_per_class=32,
        seed=23,
        key_split="validation",
    )

    assert report["audit"] == "integral_composite_residual"
    assert report["baseline_statistic"] == "pair_xor_column_sum_variance"
    assert report["baseline"]["name"] == "pair_xor_column_sum_variance"
    assert 0.5 <= report["baseline"]["auc"] <= 1.0
    assert "pair_xor_column_sum_variance" in report["composite"]["feature_names"]
    assert 0.5 <= report["composite"]["auc"] <= 1.0
    assert report["best_baseline_plus_one"]["baseline_name"] == "pair_xor_column_sum_variance"
    assert report["best_baseline_plus_one"]["added_feature"] in report["feature_statistics"]
    assert 0.5 <= report["best_baseline_plus_one"]["auc"] <= 1.0
    assert -0.5 <= report["delta_composite_vs_baseline_auc"] <= 0.5
    assert -0.5 <= report["delta_best_pair_vs_baseline_auc"] <= 0.5
    assert report["decision"] in {
        "residual_candidate_for_local_neural_probe",
        "single_statistic_explains_composite_signal",
    }
    assert "not a remote launch gate" in report["claim_scope"]


def test_integral_composite_residual_audit_cli_writes_report(tmp_path):
    output = tmp_path / "composite_residual.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit-integral-parity-signal",
            "--plan",
            "configs/experiment/innovation1/"
            "innovation1_spn_present_r8_integral_aligned_difference_control_smoke.csv",
            "--row-index",
            "0",
            "--audit",
            "composite-residual",
            "--samples-per-class",
            "8",
            "--seed",
            "23",
            "--key-split",
            "validation",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["audit"] == "integral_composite_residual"
    assert report["baseline_statistic"] == "pair_xor_column_sum_variance"
    assert "delta_composite_vs_baseline_auc" in report
    assert "delta_best_pair_vs_baseline_auc" in report
    assert "integral_composite_residual" in result.stdout


def test_integral_composite_residual_audit_cli_can_force_ciphertext_pair_bits(tmp_path):
    output = tmp_path / "same_difference_composite_residual.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit-integral-parity-signal",
            "--plan",
            "configs/experiment/innovation1/"
            "innovation1_spn_present_r8_same_difference_audit_2048_seed0_seed1.csv",
            "--row-index",
            "0",
            "--audit",
            "composite-residual",
            "--samples-per-class",
            "8",
            "--seed",
            "23",
            "--key-split",
            "validation",
            "--force-ciphertext-pair-bits",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["audit"] == "integral_composite_residual"
    assert report["sample_structure"] == "plaintext_integral_nibble_same_difference_random_negative"
    assert report["feature_encoding"] == "ciphertext_pair_bits"
    assert "delta_composite_vs_baseline_auc" in report


def test_analyze_deterministic_score_residual_cli_rebuilds_score_split(tmp_path):
    score_dir = tmp_path / "score_artifact"
    output = tmp_path / "deterministic_residual.json"
    cipher = build_cipher("present80", 8, key=0x11111111111111111111)
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x0000000000000009,
            samples_per_class=8,
            seed=10000,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=16,
            negative_mode="encrypted_random_plaintexts",
            sample_structure="plaintext_integral_nibble_difference_matched_negative",
            integral_active_nibble=0,
        )
    )
    labels = dataset.labels.astype(np.float32, copy=False)
    write_score_artifact(
        score_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=labels * 0.8 + 0.1,
            logits=labels * 4.0 - 2.0,
            sample_ids=np.array([str(index) for index in range(len(labels))], dtype=str),
            metadata={
                "cipher": "PRESENT-80",
                "cipher_key": "present80",
                "rounds": 8,
                "seed": 0,
                "samples_per_class": 16,
                "score_split": "validation",
                "score_samples_per_class": 8,
                "validation_samples_per_class": 8,
                "pairs_per_sample": 16,
                "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
                "difference_profile": "present_zhang_wang2022_mcnd",
                "difference_member": 0,
                "train_key": 0,
                "validation_key": 0x11111111111111111111,
                "model_key": "present_trail_position_stats_pairset",
            },
        ),
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze-deterministic-score-residual",
            "--score-artifact",
            str(score_dir),
            "--buckets",
            "4",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["rows"] == 16
    assert report["statistic"] == "pair_xor_column_sum_variance"
    assert report["neural"]["auc"] == 1.0
    assert report["bucketed_residual"]["bucket_count"] == 4
    assert report["metadata"]["sample_structure"] == "plaintext_integral_nibble_difference_matched_negative"
    assert report["deterministic_dataset"]["generation_mode"] == "in_memory"
    assert report["deterministic_dataset"]["label_order"] == "shuffled"
    assert "Frozen-score deterministic residual diagnostic only" in report["claim_scope"]


def test_analyze_deterministic_score_residual_cli_rebuilds_disk_cache_order(tmp_path):
    score_dir = tmp_path / "score_artifact"
    output = tmp_path / "deterministic_residual.json"
    cache_root = tmp_path / "deterministic_cache"
    cipher = build_cipher("present80", 8, key=0x11111111111111111111)
    config = DifferentialDatasetConfig(
        cipher=cipher,
        input_difference=0x0000000000000009,
        samples_per_class=8,
        seed=10000,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=16,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="plaintext_integral_nibble_difference_matched_negative",
        integral_active_nibble=0,
    )
    dataset = make_chunked_differential_dataset(
        config,
        cache_dir=tmp_path / "source_cache",
        chunk_size=4,
        workers=2,
    )
    labels = dataset.labels.astype(np.float32, copy=False)
    assert labels[:8].tolist() == [1.0] * 8
    assert labels[8:].tolist() == [0.0] * 8
    write_score_artifact(
        score_dir,
        EnsembleScoreArtifact(
            labels=labels,
            probabilities=labels * 0.8 + 0.1,
            logits=labels * 4.0 - 2.0,
            sample_ids=np.array([str(index) for index in range(len(labels))], dtype=str),
            metadata={
                "cipher": "PRESENT-80",
                "cipher_key": "present80",
                "rounds": 8,
                "seed": 0,
                "samples_per_class": 16,
                "score_split": "validation",
                "score_samples_per_class": 8,
                "validation_samples_per_class": 8,
                "pairs_per_sample": 16,
                "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
                "difference_profile": "present_zhang_wang2022_mcnd",
                "difference_member": 0,
                "train_key": 0,
                "validation_key": 0x11111111111111111111,
                "model_key": "present_trail_position_stats_pairset",
                "dataset_cache_enabled": True,
                "dataset_cache_chunk_size": 4,
                "dataset_cache_workers": 2,
            },
        ),
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze-deterministic-score-residual",
            "--score-artifact",
            str(score_dir),
            "--buckets",
            "4",
            "--dataset-cache-root",
            str(cache_root),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["rows"] == 16
    assert report["deterministic_dataset"]["generation_mode"] == "disk_cache_semantics"
    assert report["deterministic_dataset"]["label_order"] == "positive_then_negative"
    assert report["deterministic_dataset"]["chunk_size"] == 4
    assert report["deterministic_dataset"]["workers"] == 2
    assert report["neural"]["auc"] == 1.0


def test_integral_deterministic_baseline_script_defaults_to_baseline_audit(tmp_path):
    output = tmp_path / "baseline.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate-integral-deterministic-baseline",
            "--plan",
            "configs/experiment/innovation1/"
            "innovation1_spn_present_r8_integral_aligned_difference_control_smoke.csv",
            "--row-index",
            "0",
            "--samples-per-class",
            "8",
            "--seed",
            "23",
            "--key-split",
            "validation",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["audit"] == "integral_deterministic_baseline"
    assert report["statistic_name"] == "pair_xor_column_sum_variance"
    assert "auc" in report["baseline"]
    assert "integral_deterministic_baseline" in result.stdout


def test_integral_scrambled_positive_control_reduces_feature_bank_separator():
    plan = "configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_smoke.csv"
    args = parse_args(["--plan", plan])
    matched_task = dict(build_tasks(args)[0])
    matched_task["sample_structure"] = "plaintext_integral_nibble_matched_negative"
    scrambled_task = dict(matched_task)
    scrambled_task["sample_structure"] = "plaintext_integral_nibble_scrambled_positive"

    matched_report = integral_feature_bank_audit_from_task(
        matched_task,
        samples_per_class=64,
        seed=7,
        key_split="validation",
    )
    scrambled_report = integral_feature_bank_audit_from_task(
        scrambled_task,
        samples_per_class=64,
        seed=7,
        key_split="validation",
    )

    assert scrambled_report["audit"] == "integral_feature_bank"
    assert scrambled_report["sample_structure"] == "plaintext_integral_nibble_scrambled_positive"
    assert scrambled_report["best_statistic"]["name"] == "pair_xor_column_sum_variance"
    assert (
        scrambled_report["best_statistic"]["best_threshold"]["accuracy"]
        < matched_report["best_statistic"]["best_threshold"]["accuracy"]
    )


def test_integral_difference_matched_negative_removes_left_right_column_sum_separator():
    plan = "configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_smoke.csv"
    args = parse_args(["--plan", plan])
    task = dict(build_tasks(args)[0])
    task["sample_structure"] = "plaintext_integral_nibble_difference_matched_negative"
    task["integral_active_nibble"] = 1

    report = integral_feature_bank_audit_from_task(
        task,
        samples_per_class=64,
        seed=7,
        key_split="validation",
    )
    separator = report["statistics"]["left_right_column_sum_l1_mean"]["best_threshold"]

    assert report["sample_structure"] == "plaintext_integral_nibble_difference_matched_negative"
    assert separator["accuracy"] < 0.6


def test_integral_strict_random_negative_generates_independent_negative_rows():
    cipher = build_cipher("present80", 8, key=0)
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x0000000000000090,
            samples_per_class=2,
            seed=19,
            shuffle=False,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=16,
            negative_mode="encrypted_random_plaintexts",
            sample_structure="plaintext_integral_nibble_strict_random_negative",
            integral_active_nibble=0,
        )
    )

    assert dataset.features.shape == (4, 16 * 2 * cipher.block_bits)
    assert dataset.labels.tolist() == [1, 1, 0, 0]
    assert dataset.metadata["sample_structure"] == "plaintext_integral_nibble_strict_random_negative"
    assert dataset.metadata["negative_mode"] == "encrypted_random_plaintexts"


def test_integral_same_difference_random_negative_generates_rows():
    cipher = build_cipher("present80", 8, key=0)
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x0000000000000090,
            samples_per_class=2,
            seed=19,
            shuffle=False,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=16,
            negative_mode="encrypted_random_plaintexts",
            sample_structure="plaintext_integral_nibble_same_difference_random_negative",
            integral_active_nibble=0,
        )
    )

    assert dataset.features.shape == (4, 16 * 2 * cipher.block_bits)
    assert dataset.labels.tolist() == [1, 1, 0, 0]
    assert dataset.metadata["sample_structure"] == (
        "plaintext_integral_nibble_same_difference_random_negative"
    )
    assert dataset.metadata["negative_mode"] == "encrypted_random_plaintexts"


def test_integral_pair_shuffled_preserves_pair_multiset():
    cipher = build_cipher("present80", 8, key=0)
    common = {
        "cipher": cipher,
        "input_difference": 0x0000000000000090,
        "samples_per_class": 1,
        "seed": 19,
        "shuffle": False,
        "feature_encoding": "ciphertext_pair_bits",
        "pairs_per_sample": 16,
        "negative_mode": "encrypted_random_plaintexts",
        "integral_active_nibble": 0,
    }
    anchor = make_differential_dataset(
        DifferentialDatasetConfig(
            **common,
            sample_structure="plaintext_integral_nibble_difference_matched_negative",
        )
    )
    shuffled = make_differential_dataset(
        DifferentialDatasetConfig(
            **common,
            sample_structure="plaintext_integral_nibble_difference_matched_negative_pair_shuffled",
        )
    )
    pair_width = 2 * cipher.block_bits

    def pair_chunks(row):
        return [
            tuple(row[index * pair_width : (index + 1) * pair_width].tolist())
            for index in range(16)
        ]

    assert shuffled.features.shape == anchor.features.shape
    assert sorted(pair_chunks(shuffled.features[0])) == sorted(pair_chunks(anchor.features[0]))
    assert pair_chunks(shuffled.features[0]) != pair_chunks(anchor.features[0])


def test_integral_random_active_and_partial8_bridge_rows_generate():
    cipher = build_cipher("present80", 8, key=0)
    for sample_structure in [
        "plaintext_integral_nibble_difference_matched_negative_random_active",
        "plaintext_integral_nibble_difference_matched_negative_partial8",
    ]:
        dataset = make_differential_dataset(
            DifferentialDatasetConfig(
                cipher=cipher,
                input_difference=0x0000000000000090,
                samples_per_class=2,
                seed=19,
                shuffle=False,
                feature_encoding="ciphertext_pair_bits",
                pairs_per_sample=16,
                negative_mode="encrypted_random_plaintexts",
                sample_structure=sample_structure,
                integral_active_nibble=0,
            )
        )

        assert dataset.features.shape == (4, 16 * 2 * cipher.block_bits)
        assert dataset.labels.tolist() == [1, 1, 0, 0]
        assert dataset.metadata["sample_structure"] == sample_structure


def test_integral_random_active_metadata_and_relative_rows_generate():
    cipher = build_cipher("present80", 8, key=0)
    base_bits = 16 * 2 * cipher.block_bits
    metadata_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x0000000000000090,
            samples_per_class=2,
            seed=19,
            shuffle=False,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=16,
            negative_mode="encrypted_random_plaintexts",
            sample_structure="plaintext_integral_nibble_difference_matched_negative_random_active_metadata",
            integral_active_nibbles=(0, 1, 2, 3),
        )
    )
    relative_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x0000000000000090,
            samples_per_class=2,
            seed=19,
            shuffle=False,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=16,
            negative_mode="encrypted_random_plaintexts",
            sample_structure="plaintext_integral_nibble_difference_matched_negative_random_active_relative",
            integral_active_nibbles=(0, 1, 2, 3),
        )
    )

    assert metadata_dataset.features.shape == (4, base_bits + 16)
    assert relative_dataset.features.shape == (4, base_bits)
    assert metadata_dataset.metadata["integral_active_nibbles"] == [0, 1, 2, 3]
    assert relative_dataset.metadata["integral_active_nibbles"] == [0, 1, 2, 3]
    metadata_tail = metadata_dataset.features[:, -16:]
    assert metadata_tail.sum(axis=1).tolist() == [1, 1, 1, 1]
    assert set(np.argmax(metadata_tail, axis=1).tolist()).issubset({0, 1, 2, 3})


def test_aligned_active_difference_metadata_rows_generate():
    cipher = build_cipher("present80", 8, key=0)
    base_bits = 16 * 2 * cipher.block_bits
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x0000000000000009,
            samples_per_class=2,
            seed=19,
            shuffle=False,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=16,
            negative_mode="encrypted_random_plaintexts",
            sample_structure=(
                "plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata"
            ),
            integral_active_nibbles=(4, 5, 6, 7),
        )
    )

    assert dataset.features.shape == (4, base_bits + 16)
    assert dataset.metadata["row_metadata_bits"] == 16
    assert dataset.metadata["integral_active_nibbles"] == [4, 5, 6, 7]
    metadata_tail = dataset.features[:, -16:]
    assert metadata_tail.sum(axis=1).tolist() == [1, 1, 1, 1]
    assert set(np.argmax(metadata_tail, axis=1).tolist()).issubset({4, 5, 6, 7})


def test_active_aligned_input_difference_moves_single_nibble_delta():
    from dataclasses import replace

    from blockcipher_nd.data.differential.rows import _active_aligned_input_difference

    cipher = build_cipher("present80", 8, key=0)
    base_config = DifferentialDatasetConfig(
        cipher=cipher,
        input_difference=0x0000000000000009,
        samples_per_class=2,
        seed=19,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=16,
        negative_mode="encrypted_random_plaintexts",
        sample_structure=(
            "plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata"
        ),
    )

    assert _active_aligned_input_difference(
        replace(base_config, integral_active_nibble=0), cipher.block_bits
    ) == 0x0000000000000009
    assert _active_aligned_input_difference(
        replace(base_config, integral_active_nibble=5), cipher.block_bits
    ) == 0x0000000000900000
    assert _active_aligned_input_difference(
        replace(base_config, integral_active_nibble=15), cipher.block_bits
    ) == 0x9000000000000000


def test_active_aligned_input_difference_requires_single_nibble_delta():
    cipher = build_cipher("present80", 8, key=0)

    with pytest.raises(ValueError, match="single-nibble input_difference"):
        make_differential_dataset(
            DifferentialDatasetConfig(
                cipher=cipher,
                input_difference=0x0000000000000909,
                samples_per_class=1,
                seed=19,
                shuffle=False,
                feature_encoding="ciphertext_pair_bits",
                pairs_per_sample=16,
                negative_mode="encrypted_random_plaintexts",
                sample_structure=(
                    "plaintext_integral_nibble_aligned_difference_matched_negative_random_active"
                ),
                integral_active_nibbles=(0, 1),
            )
        )


def test_active_nibble_set_changes_cache_identity(tmp_path):
    cipher = build_cipher("present80", 8, key=0)
    task = {
        "cipher_key": "present80",
        "rounds": 8,
        "train_key": 0,
        "validation_key": 1,
    }
    common = {
        "cipher": cipher,
        "input_difference": 0x0000000000000090,
        "samples_per_class": 8,
        "seed": 19,
        "feature_encoding": "ciphertext_pair_bits",
        "pairs_per_sample": 16,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "plaintext_integral_nibble_difference_matched_negative_random_active",
    }

    train_cache = dataset_cache_dir(
        tmp_path,
        task,
        DifferentialDatasetConfig(**common, integral_active_nibbles=(0, 1, 2, 3)),
        "train",
    )
    heldout_cache = dataset_cache_dir(
        tmp_path,
        task,
        DifferentialDatasetConfig(**common, integral_active_nibbles=(4, 5, 6, 7)),
        "train",
    )

    assert train_cache != heldout_cache


def test_active_nibble_plan_split_overrides_are_parsed_and_applied(tmp_path):
    plan = tmp_path / "active_split.csv"
    plan.write_text(
        "\n".join(
            [
                "cipher,structure,network,model_key,family,architecture_rank,score,rounds,seed,"
                "samples_per_class,pairs_per_sample,feature_encoding,negative_mode,train_key,"
                "validation_key,key_rotation_interval,sample_structure,integral_active_nibble,"
                "integral_active_nibbles,validation_integral_active_nibbles,difference_profile,"
                "difference_member,loss,learning_rate,optimizer,weight_decay,lr_scheduler,"
                "max_learning_rate,checkpoint_metric,restore_best_checkpoint,early_stopping_patience,"
                "early_stopping_min_delta,model_options,evidence,literature",
                'PRESENT-80,SPN,Heldout,present_trail_position_stats_pairset,test,0,100,8,0,'
                '512,16,present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits,'
                'encrypted_random_plaintexts,0x0,0x1,0,'
                'plaintext_integral_nibble_difference_matched_negative_random_active,0,'
                '"[0,1,2,3]","[4,5,6,7]",present_zhang_wang2022_mcnd,0,mse,0.0001,'
                'adam,0.00001,none,0,val_auc,true,0,0.0,'
                '"{""activation"":""gelu"",""norm"":""layernorm""}",LOCAL DIAGNOSTIC,heldout active test',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task = build_tasks(parse_args(["--plan", str(plan)]))[0]
    cipher = build_cipher(task["cipher_key"], task["rounds"], key=task["train_key"])

    train_config = build_dataset_config(
        task,
        cipher=cipher,
        samples_per_class=task["samples_per_class"],
        seed=task["seed"],
        split="train",
    )
    validation_config = build_dataset_config(
        task,
        cipher=cipher,
        samples_per_class=task["samples_per_class"] // 2,
        seed=task["seed"] + 10_000,
        split="validation",
    )

    assert task["integral_active_nibbles"] == (0, 1, 2, 3)
    assert task["validation_integral_active_nibbles"] == (4, 5, 6, 7)
    assert train_config.integral_active_nibbles == (0, 1, 2, 3)
    assert validation_config.integral_active_nibbles == (4, 5, 6, 7)


def test_present_trail_position_model_accepts_active_metadata_bits():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    model = build_model(
        "present_trail_position_stats_pairset",
        input_bits=16 * pair_bits + 16,
        hidden_bits=16,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
            "metadata_bits": 16,
        },
    )

    logits = model(torch.zeros(2, 16 * pair_bits + 16))

    assert logits.shape == (2, 1)


def test_present_trail_position_relative_stats_conditions_on_active_metadata():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    model = build_model(
        "present_trail_position_stats_pairset",
        input_bits=16 * pair_bits + 16,
        hidden_bits=16,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
            "metadata_bits": 16,
            "active_conditioning": "relative_stats",
        },
    )
    features = torch.zeros(2, 16 * pair_bits + 16)
    for pair_index in range(16):
        start = pair_index * pair_bits
        features[:, start : start + 4] = 1.0
    features[0, -16] = 1.0
    features[1, -15] = 1.0

    stats = model._position_statistics(features)

    assert not torch.allclose(stats[0], stats[1])


def test_present_trail_position_p_layer_relative_stats_uses_present_coordinates():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    model = build_model(
        "present_trail_position_stats_pairset",
        input_bits=16 * pair_bits + 16,
        hidden_bits=16,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
            "metadata_bits": 16,
            "active_conditioning": "p_layer_relative_stats",
        },
    )
    activity = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 16).expand(
        1,
        16,
        pair_bits // 64,
        16,
    )
    features = torch.zeros(1, 16 * pair_bits + 16)
    features[0, -16] = 1.0

    conditioned = model._active_relative_activity(activity, features)

    assert model.active_cell_permutations[0].tolist()[:4] == [15, 11, 7, 3]
    assert conditioned[0, 0, 0, :4].tolist() == [15.0, 11.0, 7.0, 3.0]


def test_present_trail_position_prefix_only_control_masks_trail_words():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    common = {
        "input_bits": 16 * pair_bits,
        "hidden_bits": 16,
        "pair_bits": pair_bits,
        "structure": "SPN",
        "model_options": {
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
            "trail_position_control": "prefix_only",
        },
    }
    control = build_model("present_trail_position_stats_pairset", **common)
    anchor_options = dict(common["model_options"])
    anchor_options["trail_position_control"] = "none"
    anchor = build_model(
        "present_trail_position_stats_pairset",
        **{**common, "model_options": anchor_options},
    )
    features = torch.zeros(1, 16 * pair_bits)
    trail_start = control.prefix_words * 64
    for pair_index in range(16):
        pair_start = pair_index * pair_bits
        features[:, pair_start + trail_start : pair_start + trail_start + 64] = 1.0

    zero_stats = control._position_statistics(torch.zeros_like(features))
    prefix_only_stats = control._position_statistics(features)
    anchor_stats = anchor._position_statistics(features)

    assert torch.allclose(prefix_only_stats, zero_stats)
    assert not torch.allclose(anchor_stats, zero_stats)


def test_present_trail_position_trail_only_control_masks_prefix_words():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    common = {
        "input_bits": 16 * pair_bits,
        "hidden_bits": 16,
        "pair_bits": pair_bits,
        "structure": "SPN",
        "model_options": {
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
            "trail_position_control": "trail_only",
        },
    }
    control = build_model("present_trail_position_stats_pairset", **common)
    anchor_options = dict(common["model_options"])
    anchor_options["trail_position_control"] = "none"
    anchor = build_model(
        "present_trail_position_stats_pairset",
        **{**common, "model_options": anchor_options},
    )
    features = torch.zeros(1, 16 * pair_bits)
    for pair_index in range(16):
        pair_start = pair_index * pair_bits
        features[:, pair_start : pair_start + 64] = 1.0

    zero_stats = control._position_statistics(torch.zeros_like(features))
    trail_only_stats = control._position_statistics(features)
    anchor_stats = anchor._position_statistics(features)

    assert torch.allclose(trail_only_stats, zero_stats)
    assert not torch.allclose(anchor_stats, zero_stats)


def test_present_trail_position_reverse_control_changes_position_statistics():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    common = {
        "input_bits": 16 * pair_bits,
        "hidden_bits": 16,
        "pair_bits": pair_bits,
        "structure": "SPN",
        "model_options": {
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
        },
    }
    anchor = build_model("present_trail_position_stats_pairset", **common)
    reverse_options = dict(common["model_options"])
    reverse_options["trail_position_control"] = "reverse_trail_positions"
    reversed_model = build_model(
        "present_trail_position_stats_pairset",
        **{**common, "model_options": reverse_options},
    )
    features = torch.zeros(1, 16 * pair_bits)
    trail_start = anchor.prefix_words * 64
    for pair_index in range(16):
        pair_start = pair_index * pair_bits
        second_trail_word = trail_start + 64
        features[:, pair_start + second_trail_word : pair_start + second_trail_word + 64] = 1.0

    anchor_stats = anchor._position_statistics(features)
    reversed_stats = reversed_model._position_statistics(features)

    assert not torch.allclose(anchor_stats, reversed_stats)


def test_present_trail_position_permute_control_uses_fixed_trail_permutation():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    common = {
        "input_bits": 16 * pair_bits,
        "hidden_bits": 16,
        "pair_bits": pair_bits,
        "structure": "SPN",
        "model_options": {
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
        },
    }
    anchor = build_model("present_trail_position_stats_pairset", **common)
    permute_options = dict(common["model_options"])
    permute_options["trail_position_control"] = "permute_trail_positions"
    permuted_model = build_model(
        "present_trail_position_stats_pairset",
        **{**common, "model_options": permute_options},
    )
    features = torch.zeros(1, 16 * pair_bits)
    trail_start = anchor.prefix_words * 64
    for pair_index in range(16):
        pair_start = pair_index * pair_bits
        second_trail_word = trail_start + 64
        features[:, pair_start + second_trail_word : pair_start + second_trail_word + 64] = 1.0

    permutation = permuted_model.trail_word_permutation.tolist()
    anchor_stats = anchor._position_statistics(features)
    permuted_stats = permuted_model._position_statistics(features)

    assert sorted(permutation) == list(range(36))
    assert permutation[:6] == [0, 7, 14, 21, 28, 35]
    assert not torch.allclose(anchor_stats, permuted_stats)


def test_present_trail_position_rejects_unknown_position_control():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )

    with pytest.raises(ValueError, match="trail_position_control"):
        build_model(
            "present_trail_position_stats_pairset",
            input_bits=16 * pair_bits,
            hidden_bits=16,
            pair_bits=pair_bits,
            structure="SPN",
            model_options={
                "trail_depth": 4,
                "trail_words_per_depth": 9,
                "trail_position_control": "random_shuffle",
            },
        )


def test_present_trail_position_center_normalization_preserves_prefix_and_centers_trail():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    model = build_model(
        "present_trail_position_stats_pairset",
        input_bits=16 * pair_bits,
        hidden_bits=16,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
            "trail_normalization": "trail_center",
        },
    )
    activity = torch.zeros(2, 16, model.words_per_pair, model.cells_per_word)
    activity[:, :, : model.prefix_words] = 0.25
    activity[:, :, model.prefix_words :] = torch.linspace(
        0.0,
        1.0,
        steps=model.trail_depth * model.trail_words_per_depth * model.cells_per_word,
    ).reshape(1, 1, model.trail_depth * model.trail_words_per_depth, model.cells_per_word)

    normalized = model._trail_normalized_activity(activity)
    trail = normalized[:, :, model.prefix_words :]

    assert torch.allclose(normalized[:, :, : model.prefix_words], activity[:, :, : model.prefix_words])
    assert torch.allclose(trail.mean(dim=(2, 3)), torch.zeros(2, 16), atol=1e-6)


def test_present_trail_position_zscore_normalization_scales_trail():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    model = build_model(
        "present_trail_position_stats_pairset",
        input_bits=16 * pair_bits,
        hidden_bits=16,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
            "trail_normalization": "trail_zscore",
        },
    )
    activity = torch.zeros(1, 16, model.words_per_pair, model.cells_per_word)
    activity[:, :, model.prefix_words :] = torch.linspace(
        0.0,
        1.0,
        steps=model.trail_depth * model.trail_words_per_depth * model.cells_per_word,
    ).reshape(1, 1, model.trail_depth * model.trail_words_per_depth, model.cells_per_word)

    normalized = model._trail_normalized_activity(activity)
    trail = normalized[:, :, model.prefix_words :]

    assert torch.allclose(trail.mean(dim=(2, 3)), torch.zeros(1, 16), atol=1e-6)
    assert torch.allclose(trail.std(dim=(2, 3), unbiased=False), torch.ones(1, 16), atol=1e-6)


def test_present_trail_position_rejects_unknown_trail_normalization():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )

    with pytest.raises(ValueError, match="trail_normalization"):
        build_model(
            "present_trail_position_stats_pairset",
            input_bits=16 * pair_bits,
            hidden_bits=16,
            pair_bits=pair_bits,
            structure="SPN",
            model_options={
                "trail_depth": 4,
                "trail_words_per_depth": 9,
                "trail_normalization": "batch_magic",
            },
        )


def test_present_trail_position_gated_auxiliary_forward_shape():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    model = build_model(
        "present_trail_position_stats_pairset",
        input_bits=16 * pair_bits,
        hidden_bits=16,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
            "trail_fusion": "gated_auxiliary",
            "trail_auxiliary_scale": 0.25,
        },
    )

    logits = model(torch.zeros(3, 16 * pair_bits))

    assert logits.shape == (3, 1)
    assert model.prefix_stats_feature_bits > 0
    assert model.trail_stats_feature_bits > 0


def test_present_trail_position_gated_auxiliary_scalar_gate_forward_shape():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    model = build_model(
        "present_trail_position_stats_pairset",
        input_bits=16 * pair_bits,
        hidden_bits=16,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
            "trail_fusion": "gated_auxiliary",
            "trail_gate": "scalar",
            "trail_auxiliary_scale": 0.25,
        },
    )

    logits = model(torch.zeros(3, 16 * pair_bits))
    gate = model.trail_gate_network(torch.zeros(3, model.stats_hidden_bits))

    assert logits.shape == (3, 1)
    assert gate.shape == (3, 1)


def test_present_trail_position_gated_auxiliary_zero_scale_ignores_trail_values():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    model = build_model(
        "present_trail_position_stats_pairset",
        input_bits=16 * pair_bits,
        hidden_bits=16,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "trail_depth": 4,
            "trail_words_per_depth": 9,
            "stats_hidden_bits": 16,
            "trail_fusion": "gated_auxiliary",
            "trail_auxiliary_scale": 0.0,
        },
    )
    model.eval()
    left = torch.zeros(2, 16 * pair_bits)
    right = left.clone()
    trail_start = model.prefix_words * 64
    for pair_index in range(16):
        pair_start = pair_index * pair_bits
        right[:, pair_start + trail_start : pair_start + pair_bits] = 1.0

    with torch.no_grad():
        left_logits = model(left)
        right_logits = model(right)

    assert torch.allclose(left_logits, right_logits, atol=1e-6)


def test_present_trail_position_rejects_unknown_trail_fusion():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )

    with pytest.raises(ValueError, match="trail_fusion"):
        build_model(
            "present_trail_position_stats_pairset",
            input_bits=16 * pair_bits,
            hidden_bits=16,
            pair_bits=pair_bits,
            structure="SPN",
            model_options={
                "trail_depth": 4,
                "trail_words_per_depth": 9,
                "trail_fusion": "full_override",
            },
        )


def test_present_trail_position_rejects_unknown_trail_gate():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )

    with pytest.raises(ValueError, match="trail_gate"):
        build_model(
            "present_trail_position_stats_pairset",
            input_bits=16 * pair_bits,
            hidden_bits=16,
            pair_bits=pair_bits,
            structure="SPN",
            model_options={
                "trail_depth": 4,
                "trail_words_per_depth": 9,
                "trail_fusion": "gated_auxiliary",
                "trail_gate": "per_pair",
            },
        )


def test_present_trail_position_rejects_negative_trail_auxiliary_scale():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )

    with pytest.raises(ValueError, match="trail_auxiliary_scale"):
        build_model(
            "present_trail_position_stats_pairset",
            input_bits=16 * pair_bits,
            hidden_bits=16,
            pair_bits=pair_bits,
            structure="SPN",
            model_options={
                "trail_depth": 4,
                "trail_words_per_depth": 9,
                "trail_fusion": "gated_auxiliary",
                "trail_auxiliary_scale": -0.1,
            },
        )


def test_present_trail_value_mismatch_encodings_preserve_full_trail_shape():
    cipher = build_cipher("present80", 8, key=0)
    full_encoding = "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits"
    masked_encoding = "present_delta_paligned_sinv_sboxddt_beamstats8deep4_maskedsource_cell_matrix_bits"
    constant_encoding = "present_delta_paligned_sinv_sboxddt_beamstats8deep4_constantsource_cell_matrix_bits"
    left = 0x0123456789ABCDEF
    right = 0x1123456789ABCDEF

    full = encode_ciphertext_pair(left, right, width=64, feature_encoding=full_encoding, cipher=cipher)
    masked = encode_ciphertext_pair(left, right, width=64, feature_encoding=masked_encoding, cipher=cipher)
    constant = encode_ciphertext_pair(left, right, width=64, feature_encoding=constant_encoding, cipher=cipher)

    assert pair_bits_for_encoding(64, masked_encoding) == pair_bits_for_encoding(64, full_encoding)
    assert pair_bits_for_encoding(64, constant_encoding) == pair_bits_for_encoding(64, full_encoding)
    assert is_supported_feature_encoding(masked_encoding)
    assert is_supported_feature_encoding(constant_encoding)
    assert len(full) == len(masked) == len(constant) == 2496
    assert masked != full
    assert constant != full
    assert constant != masked


def test_parameterized_present_trail_value_mismatch_encodings_preserve_shape():
    cipher = build_cipher("present80", 8, key=0)
    full_encoding = "present_delta_paligned_sinv_sboxddt_beamstats4deep2_cell_matrix_bits"
    masked_encoding = "present_delta_paligned_sinv_sboxddt_beamstats4deep2_maskedsource_cell_matrix_bits"
    constant_encoding = "present_delta_paligned_sinv_sboxddt_beamstats4deep2_constantsource_cell_matrix_bits"
    left = 0x0123456789ABCDEF
    right = 0x1123456789ABCDEF

    full = encode_ciphertext_pair(left, right, width=64, feature_encoding=full_encoding, cipher=cipher)
    masked = encode_ciphertext_pair(left, right, width=64, feature_encoding=masked_encoding, cipher=cipher)
    constant = encode_ciphertext_pair(left, right, width=64, feature_encoding=constant_encoding, cipher=cipher)

    assert pair_bits_for_encoding(64, full_encoding) == 64 * (3 + 2 * 9)
    assert pair_bits_for_encoding(64, masked_encoding) == pair_bits_for_encoding(64, full_encoding)
    assert pair_bits_for_encoding(64, constant_encoding) == pair_bits_for_encoding(64, full_encoding)
    assert is_supported_feature_encoding(masked_encoding)
    assert is_supported_feature_encoding(constant_encoding)
    assert len(full) == len(masked) == len(constant) == 1344
    assert masked != full
    assert constant != full
    assert constant != masked


def test_integral_multi_nibble_difference_matched_negative_generates_256_pair_rows():
    cipher = build_cipher("present80", 8, key=0)
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x0700000000000700,
            samples_per_class=2,
            seed=19,
            shuffle=False,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=256,
            negative_mode="encrypted_random_plaintexts",
            sample_structure="plaintext_integral_multi_nibble_difference_matched_negative",
        )
    )

    assert dataset.features.shape == (4, 256 * 2 * cipher.block_bits)
    assert dataset.labels.tolist() == [1, 1, 0, 0]
    assert dataset.metadata["sample_structure"] == (
        "plaintext_integral_multi_nibble_difference_matched_negative"
    )


def test_present_r8_integral_pair_order_control_plan_is_local_audit_only():
    plan = "configs/experiment/innovation1/innovation1_spn_present_r8_integral_pair_order_control_smoke.csv"
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["sample_structure"] for task in tasks] == [
        "plaintext_integral_nibble_matched_negative",
        "plaintext_integral_nibble_scrambled_positive",
    ]
    for task in tasks:
        assert task["rounds"] == 8
        assert task["samples_per_class"] == 256
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert "LOCAL AUDIT only" in task["matching_evidence"]


def test_present_r8_bridge_protocol_attribution_plan_is_complete_local_diagnostic():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_bridge_protocol_attribution_512_seed0_seed1.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 30
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["samples_per_class"] for task in tasks} == {512}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["seed"] for task in tasks} == {0, 1}
    assert {
        "plaintext_integral_nibble_difference_matched_negative",
        "plaintext_integral_nibble_difference_matched_negative_pair_shuffled",
        "plaintext_integral_nibble_difference_matched_negative_random_active",
        "plaintext_integral_nibble_difference_matched_negative_partial8",
        "plaintext_integral_nibble_same_difference_random_negative",
        "independent_pairs",
    }.issubset({task["sample_structure"] for task in tasks})
    assert all("LOCAL DIAGNOSTIC" in task["matching_evidence"] for task in tasks)


def test_present_r8_active_nibble_generalization_plan_is_split_controlled():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_active_nibble_generalization_512_seed0_seed1.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 10
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["samples_per_class"] for task in tasks} == {512}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["seed"] for task in tasks} == {0, 1}
    assert {task["model_key"] for task in tasks} == {"present_trail_position_stats_pairset"}
    assert {
        "plaintext_integral_nibble_difference_matched_negative",
        "plaintext_integral_nibble_difference_matched_negative_random_active",
        "plaintext_integral_nibble_difference_matched_negative_random_active_metadata",
        "plaintext_integral_nibble_difference_matched_negative_random_active_relative",
    } == {task["sample_structure"] for task in tasks}

    heldout_tasks = [
        task
        for task in tasks
        if task["validation_integral_active_nibbles"] == (4, 5, 6, 7)
    ]
    metadata_tasks = [
        task
        for task in tasks
        if task["sample_structure"]
        == "plaintext_integral_nibble_difference_matched_negative_random_active_metadata"
    ]
    relative_tasks = [
        task
        for task in tasks
        if task["sample_structure"]
        == "plaintext_integral_nibble_difference_matched_negative_random_active_relative"
    ]

    assert len(heldout_tasks) == 2
    assert all(task["integral_active_nibbles"] == (0, 1, 2, 3) for task in heldout_tasks)
    assert len(metadata_tasks) == 2
    assert all(task["model_options"]["metadata_bits"] == 16 for task in metadata_tasks)
    assert len(relative_tasks) == 2
    assert all("relative-coordinate" in task["matching_evidence"] for task in relative_tasks)

    plan_doc = Path("docs/experiments/innovation1-present-r8-active-nibble-generalization-plan.md")
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "heldout-active" in plan_text
    assert "metadata_bits = 16" in plan_text


def test_present_r8_active_conditioned_curriculum_plan_is_local_gate_only():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_active_conditioned_curriculum_512_seed0_seed1.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 16
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["samples_per_class"] for task in tasks} == {512}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["seed"] for task in tasks} == {0, 1}
    assert {task["model_key"] for task in tasks} == {"present_trail_position_stats_pairset"}
    assert all("LOCAL DIAGNOSTIC" in task["matching_evidence"] for task in tasks)

    conditioned_tasks = [
        task
        for task in tasks
        if task["model_options"].get("active_conditioning") == "relative_stats"
    ]
    assert len(conditioned_tasks) == 12
    assert {task["model_options"]["metadata_bits"] for task in conditioned_tasks} == {16}
    assert {
        task["integral_active_nibbles"]
        for task in conditioned_tasks
        if not task["validation_integral_active_nibbles"]
    } == {
        (0,),
        (0, 1),
        (0, 1, 2, 3),
        (0, 1, 2, 3, 4, 5, 6, 7),
        tuple(range(16)),
    }
    heldout_tasks = [
        task
        for task in conditioned_tasks
        if task["validation_integral_active_nibbles"] == (4, 5, 6, 7)
    ]
    assert len(heldout_tasks) == 2
    assert all(task["integral_active_nibbles"] == (0, 1, 2, 3) for task in heldout_tasks)

    plan_doc = Path("docs/experiments/innovation1-present-r8-active-conditioned-curriculum-plan.md")
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "active_conditioning = relative_stats" in plan_text
    assert "Remote scale remains blocked" in plan_text


def test_present_r8_p_layer_relative_active_curriculum_plan_is_local_gate_only():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 16
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["samples_per_class"] for task in tasks} == {512}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["seed"] for task in tasks} == {0, 1}
    assert {task["model_key"] for task in tasks} == {"present_trail_position_stats_pairset"}
    assert all("LOCAL DIAGNOSTIC" in task["matching_evidence"] for task in tasks)

    p_layer_tasks = [
        task
        for task in tasks
        if task["model_options"].get("active_conditioning") == "p_layer_relative_stats"
    ]
    assert len(p_layer_tasks) == 12
    assert {task["model_options"]["metadata_bits"] for task in p_layer_tasks} == {16}
    assert {
        task["integral_active_nibbles"]
        for task in p_layer_tasks
        if not task["validation_integral_active_nibbles"]
    } == {
        (0,),
        (0, 1),
        (0, 1, 2, 3),
        (0, 1, 2, 3, 4, 5, 6, 7),
        tuple(range(16)),
    }
    heldout_tasks = [
        task
        for task in p_layer_tasks
        if task["validation_integral_active_nibbles"] == (4, 5, 6, 7)
    ]
    assert len(heldout_tasks) == 2
    assert all(task["integral_active_nibbles"] == (0, 1, 2, 3) for task in heldout_tasks)

    plan_doc = Path("docs/experiments/innovation1-present-r8-p-layer-relative-active-curriculum-plan.md")
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "active_conditioning = p_layer_relative_stats" in plan_text
    assert "Remote scale remains blocked" in plan_text


def test_present_r8_integral_feature_variation_control_plan_is_local_audit_only():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_integral_feature_variation_control_smoke.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 8
    assert {task["sample_structure"] for task in tasks} == {
        "plaintext_integral_nibble_difference_matched_negative"
    }
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["samples_per_class"] for task in tasks} == {256}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["feature_encoding"] for task in tasks} == {"ciphertext_pair_bits"}
    assert {task["integral_active_nibble"] for task in tasks[:4]} == {0, 1, 7, 15}
    assert {task["difference_profile"] for task in tasks[4:]} == {
        "present_zhang_wang2022_mcnd",
        "present_autond_dbitnet2023_highround",
        "present_entropy2026_gohr",
        "present_wang_jain2021_1",
    }
    assert all("LOCAL AUDIT only" in task["matching_evidence"] for task in tasks)


def test_present_r8_integral_aligned_difference_control_plan_is_local_audit_only():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_integral_aligned_difference_control_smoke.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 5
    assert {task["sample_structure"] for task in tasks} == {
        "plaintext_integral_nibble_difference_matched_negative"
    }
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["samples_per_class"] for task in tasks} == {256}
    assert [
        (task["difference_profile"], task["integral_active_nibble"])
        for task in tasks
    ] == [
        ("present_zhang_wang2022_mcnd", 0),
        ("present_autond_dbitnet2023_highround", 6),
        ("present_entropy2026_gohr", 5),
        ("present_wang_jain2021_1", 2),
        ("present_wang_jain2021_1", 14),
    ]
    assert all("LOCAL AUDIT only" in task["matching_evidence"] for task in tasks)


def test_present_r8_integral_aligned_neural_followup_plan_has_baseline_gate():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_integral_aligned_neural_followup_smoke.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 3
    assert {task["sample_structure"] for task in tasks} == {
        "plaintext_integral_nibble_difference_matched_negative"
    }
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["samples_per_class"] for task in tasks} == {256}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert [
        (task["difference_profile"], task["integral_active_nibble"])
        for task in tasks
    ] == [
        ("present_zhang_wang2022_mcnd", 0),
        ("present_autond_dbitnet2023_highround", 6),
        ("present_entropy2026_gohr", 5),
    ]
    assert {task["feature_encoding"] for task in tasks} == {"ciphertext_pair_bits"}
    assert {task["checkpoint_metric"] for task in tasks} == {"val_auc"}
    assert all("LOCAL SMOKE only" in task["matching_evidence"] for task in tasks)
    assert all("deterministic baseline AUC gate" in task["matching_evidence"] for task in tasks)
    assert all("pair_xor_column_sum_variance" in task["literature"] for task in tasks)


def test_present_r8_gpd_style_beamstats_smoke_plan_uses_existing_partial_inverse_features():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_gpd_style_beamstats_smoke.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert [task["feature_encoding"] for task in tasks] == [
        "present_pair_xor_paligned_cell_matrix_bits",
        "present_pair_xor_paligned_sinv_cell_matrix_bits",
        "present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits",
        "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits",
    ]
    assert [task["model_key"] for task in tasks] == [
        "present_matrix_trail_hybrid_pairset_invp",
        "present_matrix_trail_hybrid_pairset_invp_sinv",
        "present_matrix_trail_hybrid_pairset_invp_sinv",
        "present_matrix_trail_hybrid_pairset_invp_sinv",
    ]
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["samples_per_class"] for task in tasks} == {128}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["sample_structure"] for task in tasks} == {
        "plaintext_integral_nibble_difference_matched_negative"
    }
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["difference_profile"] for task in tasks} == {"present_zhang_wang2022_mcnd"}
    assert {task["integral_active_nibble"] for task in tasks} == {0}
    assert {task["checkpoint_metric"] for task in tasks} == {"val_auc"}
    assert all("LOCAL SMOKE only" in task["matching_evidence"] for task in tasks)
    assert all("GPD-style partial inverse" in task["matching_evidence"] for task in tasks)

    plan_doc = Path("docs/experiments/innovation1-present-r8-gpd-style-beamstats-plan.md")
    plan_text = plan_doc.read_text(encoding="utf-8")
    assert "Generic Partial Decryption" in plan_text
    assert "do not duplicate the existing single-step Sinv feature" in plan_text


def test_present_r8_gpd_style_beamstats_seed1_plan_repeats_seed0_protocol():
    seed0_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_gpd_style_beamstats_smoke.csv"
    )
    seed1_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_gpd_style_beamstats_smoke_seed1.csv"
    )
    seed0_tasks = build_tasks(parse_args(["--plan", seed0_plan]))
    seed1_tasks = build_tasks(parse_args(["--plan", seed1_plan]))

    assert len(seed1_tasks) == len(seed0_tasks) == 4
    assert {task["seed"] for task in seed1_tasks} == {1}
    for seed0_task, seed1_task in zip(seed0_tasks, seed1_tasks, strict=True):
        for field in (
            "model_key",
            "feature_encoding",
            "rounds",
            "samples_per_class",
            "pairs_per_sample",
            "sample_structure",
            "negative_mode",
            "difference_profile",
            "integral_active_nibble",
            "train_key",
            "validation_key",
            "checkpoint_metric",
            "restore_best_checkpoint",
        ):
            assert seed1_task[field] == seed0_task[field]
        assert seed1_task["matching_evidence"].startswith("LOCAL SMOKE only")
        assert "seed1 repeat" in seed1_task["matching_evidence"]


def test_present_r8_gpd_style_beamstats_512_plan_preserves_smoke_protocol():
    smoke_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_gpd_style_beamstats_smoke.csv"
    )
    diagnostic_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_gpd_style_beamstats_512_seed0.csv"
    )
    smoke_tasks = build_tasks(parse_args(["--plan", smoke_plan]))
    diagnostic_tasks = build_tasks(parse_args(["--plan", diagnostic_plan]))

    assert len(diagnostic_tasks) == len(smoke_tasks) == 4
    assert {task["samples_per_class"] for task in diagnostic_tasks} == {512}
    assert {task["seed"] for task in diagnostic_tasks} == {0}
    for smoke_task, diagnostic_task in zip(smoke_tasks, diagnostic_tasks, strict=True):
        for field in (
            "model_key",
            "feature_encoding",
            "rounds",
            "pairs_per_sample",
            "sample_structure",
            "negative_mode",
            "difference_profile",
            "integral_active_nibble",
            "train_key",
            "validation_key",
            "checkpoint_metric",
            "restore_best_checkpoint",
        ):
            assert diagnostic_task[field] == smoke_task[field]
        assert diagnostic_task["matching_evidence"].startswith("LOCAL DIAGNOSTIC only")
        assert "512/class" in diagnostic_task["matching_evidence"]
        assert "no remote launch" in diagnostic_task["matching_evidence"]


def test_present_r8_gpd_style_beamstats_512_seed1_plan_repeats_seed0_protocol():
    seed0_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_gpd_style_beamstats_512_seed0.csv"
    )
    seed1_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_gpd_style_beamstats_512_seed1.csv"
    )
    seed0_tasks = build_tasks(parse_args(["--plan", seed0_plan]))
    seed1_tasks = build_tasks(parse_args(["--plan", seed1_plan]))

    assert len(seed1_tasks) == len(seed0_tasks) == 4
    assert {task["samples_per_class"] for task in seed1_tasks} == {512}
    assert {task["seed"] for task in seed1_tasks} == {1}
    for seed0_task, seed1_task in zip(seed0_tasks, seed1_tasks, strict=True):
        for field in (
            "model_key",
            "feature_encoding",
            "rounds",
            "pairs_per_sample",
            "sample_structure",
            "negative_mode",
            "difference_profile",
            "integral_active_nibble",
            "train_key",
            "validation_key",
            "checkpoint_metric",
            "restore_best_checkpoint",
        ):
            assert seed1_task[field] == seed0_task[field]
        assert seed1_task["matching_evidence"].startswith("LOCAL DIAGNOSTIC only")
        assert "512/class" in seed1_task["matching_evidence"]
        assert "seed1 repeat" in seed1_task["matching_evidence"]
        assert "no remote launch" in seed1_task["matching_evidence"]


def test_present_r8_gpd_style_beamstats_attribution_reports_semantic_statistics():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_gpd_style_beamstats_512_seed1.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))
    beamstats_task = tasks[3]

    report = spn_feature_audit.beamstats_attribution_from_task(
        beamstats_task,
        samples_per_class=4,
        seed=123,
        key_split="validation",
    )

    assert report["status"] == "pass"
    assert report["audit"] == "present_beamstats_semantic_attribution"
    assert report["feature_encoding"] == (
        "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits"
    )
    assert report["samples_per_class"] == 4
    assert report["beam_width"] == 4
    assert report["depth"] == 3
    assert report["claim_scope"].startswith("Local semantic attribution")
    assert "score_mean" in report["statistics"]
    assert "active_mean" in report["statistics"]
    assert "disagreement_nonzero_rate" in report["statistics"]
    assert report["best_statistic"]["name"] in report["statistics"]
    assert 0.0 <= report["best_statistic"]["auc_advantage"] <= 0.5


def test_audit_spn_features_cli_writes_beamstats_attribution(tmp_path):
    output = tmp_path / "beamstats_attribution.json"
    status = audit_spn_features_main(
        [
            "--beamstats-attribution-plan",
            "configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed1.csv",
            "--row-index",
            "3",
            "--samples-per-class",
            "4",
            "--seed",
            "123",
            "--key-split",
            "validation",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit"] == "present_beamstats_semantic_attribution"
    assert payload["feature_encoding"] == (
        "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits"
    )
    assert payload["best_statistic"]["name"] in payload["statistics"]


def test_present_r8_trail_position_attribution_reports_position_statistics():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_trail_position_beamstats_512_local.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))
    trail_position_task = tasks[1]

    report = spn_feature_audit.trail_position_attribution_from_task(
        trail_position_task,
        samples_per_class=4,
        seed=123,
        key_split="validation",
        top_k=5,
    )

    assert report["status"] == "pass"
    assert report["audit"] == "present_trail_position_stats_attribution"
    assert report["feature_encoding"] == (
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits"
    )
    assert report["samples_per_class"] == 4
    assert report["trail_depth"] == 4
    assert report["trail_words_per_depth"] == 9
    assert report["position_stat_dim"] > 0
    assert report["best_statistic"]["name"] in report["top_statistics"]["feature_names"]
    assert 0.0 <= report["best_statistic"]["auc_advantage"] <= 0.5
    assert report["composite"]["combiner"] == "top_position_stat_oriented_zscore_mean"
    assert 0.0 <= report["composite"]["auc_advantage"] <= 0.5
    assert "not neural training" in report["claim_scope"]


def test_present_r8_trail_position_2048_plan_preserves_512_protocol():
    base_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_trail_position_beamstats_512_local.csv"
    )
    diagnostic_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_trail_position_beamstats_2048_local.csv"
    )
    base_tasks = build_tasks(parse_args(["--plan", base_plan]))
    diagnostic_tasks = build_tasks(parse_args(["--plan", diagnostic_plan]))

    assert len(diagnostic_tasks) == len(base_tasks) == 4
    assert {task["samples_per_class"] for task in diagnostic_tasks} == {2048}
    assert {task["seed"] for task in diagnostic_tasks} == {0, 1}
    for base_task, diagnostic_task in zip(base_tasks, diagnostic_tasks, strict=True):
        for field in (
            "model_key",
            "feature_encoding",
            "rounds",
            "pairs_per_sample",
            "sample_structure",
            "negative_mode",
            "difference_profile",
            "integral_active_nibble",
            "train_key",
            "validation_key",
            "checkpoint_metric",
            "restore_best_checkpoint",
            "model_options",
        ):
            assert diagnostic_task[field] == base_task[field]
        assert "LOCAL DIAGNOSTIC 2048/class" in diagnostic_task["matching_evidence"]
        assert "residual gate" in diagnostic_task["matching_evidence"]
        assert "no remote launch" in diagnostic_task["matching_evidence"]


def test_present_r8_trail_position_65k_plan_is_lean_medium_readiness_matrix():
    base_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_trail_position_beamstats_2048_local.csv"
    )
    medium_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_trail_position_beamstats_65k_seed0.csv"
    )
    base_tasks = build_tasks(parse_args(["--plan", base_plan]))[:2]
    medium_tasks = build_tasks(parse_args(["--plan", medium_plan]))

    assert len(medium_tasks) == len(base_tasks) == 2
    assert [task["model_key"] for task in medium_tasks] == [
        "present_pairset_global_stats",
        "present_trail_position_stats_pairset",
    ]
    assert {task["samples_per_class"] for task in medium_tasks} == {65_536}
    assert {task["seed"] for task in medium_tasks} == {0}
    for base_task, medium_task in zip(base_tasks, medium_tasks, strict=True):
        for field in (
            "model_key",
            "feature_encoding",
            "rounds",
            "pairs_per_sample",
            "sample_structure",
            "negative_mode",
            "difference_profile",
            "integral_active_nibble",
            "train_key",
            "validation_key",
            "checkpoint_metric",
            "restore_best_checkpoint",
            "model_options",
        ):
            assert medium_task[field] == base_task[field]
        assert "MEDIUM READINESS 65536/class" in medium_task["matching_evidence"]
        assert "residual gate" in medium_task["matching_evidence"]
        assert "not formal evidence" in medium_task["matching_evidence"]


def test_present_r8_cell_value_histogram_plan_is_local_diagnostic_only():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_cell_value_histogram_2048_local.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 4
    assert [task["model_key"] for task in tasks] == [
        "present_pairset_global_stats",
        "present_pairset_histogram_hybrid",
        "present_pairset_global_stats",
        "present_pairset_histogram_hybrid",
    ]
    assert {task["samples_per_class"] for task in tasks} == {2048}
    assert {task["seed"] for task in tasks} == {0, 1}
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["feature_encoding"] for task in tasks} == {
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits"
    }
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["sample_structure"] for task in tasks} == {
        "plaintext_integral_nibble_difference_matched_negative"
    }
    assert {task["difference_profile"] for task in tasks} == {"present_zhang_wang2022_mcnd"}
    assert {task["integral_active_nibble"] for task in tasks} == {0}
    assert all("LOCAL DIAGNOSTIC 2048/class" in task["matching_evidence"] for task in tasks)
    assert all("no remote launch" in task["matching_evidence"] for task in tasks)

    plan_doc = Path("docs/experiments/innovation1-present-r8-cell-value-histogram-screen-plan.md")
    doc_text = plan_doc.read_text(encoding="utf-8")
    assert "not immediate ensembling" in doc_text
    assert "negative_mode = encrypted_random_plaintexts" in doc_text
    assert "formal PRESENT result" in doc_text


def test_audit_spn_features_cli_writes_trail_position_attribution(tmp_path):
    output = tmp_path / "trail_position_attribution.json"
    status = audit_spn_features_main(
        [
            "--trail-position-attribution-plan",
            "configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv",
            "--row-index",
            "1",
            "--samples-per-class",
            "4",
            "--seed",
            "123",
            "--key-split",
            "validation",
            "--top-k",
            "5",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit"] == "present_trail_position_stats_attribution"
    assert payload["best_statistic"]["name"] in payload["top_statistics"]["feature_names"]
    assert "composite" in payload


def test_present_r8_trail_position_split_baseline_selects_on_train_and_evaluates_validation():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_trail_position_beamstats_512_local.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))
    trail_position_task = tasks[1]

    report = spn_feature_audit.trail_position_split_baseline_from_task(
        trail_position_task,
        samples_per_class=4,
        seed=123,
        top_k=5,
    )

    assert report["status"] == "pass"
    assert report["audit"] == "present_trail_position_split_baseline"
    assert report["feature_encoding"] == (
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits"
    )
    assert report["samples_per_class"] == 4
    assert report["reference"]["key_split"] == "train"
    assert report["evaluation"]["key_split"] == "validation"
    assert len(report["selected_statistics"]["indices"]) == 5
    assert len(report["selected_statistics"]["feature_names"]) == 5
    assert report["reference"]["composite"]["combiner"] == "train_selected_position_stat_oriented_zscore_mean"
    assert report["evaluation"]["composite"]["combiner"] == "train_selected_position_stat_oriented_zscore_mean"
    assert report["reference"]["composite"]["fit_key_split"] == "train"
    assert report["evaluation"]["composite"]["fit_key_split"] == "train"
    assert 0.0 <= report["evaluation"]["composite"]["auc_advantage"] <= 0.5
    assert "not neural training" in report["claim_scope"]


def test_audit_spn_features_cli_writes_trail_position_split_baseline(tmp_path):
    output = tmp_path / "trail_position_split_baseline.json"
    status = audit_spn_features_main(
        [
            "--trail-position-split-baseline-plan",
            "configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv",
            "--row-index",
            "1",
            "--samples-per-class",
            "4",
            "--seed",
            "123",
            "--top-k",
            "5",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit"] == "present_trail_position_split_baseline"
    assert payload["reference"]["key_split"] == "train"
    assert payload["evaluation"]["key_split"] == "validation"
    assert payload["evaluation"]["composite"]["fit_key_split"] == "train"
    assert len(payload["selected_statistics"]["indices"]) == 5


def test_present_r8_trail_position_control_baseline_reports_requested_controls():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_trail_position_beamstats_512_local.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))
    trail_position_task = tasks[1]

    report = spn_feature_audit.trail_position_control_baseline_from_task(
        trail_position_task,
        samples_per_class=4,
        seed=123,
        top_k=5,
        active_nibbles=(1,),
        input_differences=(0x90,),
        pair_orders=("reverse",),
    )

    assert report["status"] == "pass"
    assert report["audit"] == "present_trail_position_control_baseline"
    assert report["baseline"]["variant_kind"] == "baseline"
    assert report["baseline"]["report"]["evaluation"]["key_split"] == "validation"
    assert len(report["controls"]) == 3
    assert [control["variant_kind"] for control in report["controls"]] == [
        "active_nibble",
        "input_difference",
        "pair_order",
    ]
    assert [control["variant_label"] for control in report["controls"]] == [
        "active_nibble_1",
        "input_difference_0x90",
        "pair_order_reverse",
    ]
    for control in report["controls"]:
        assert control["report"]["reference"]["composite"]["fit_key_split"] == "train"
        assert control["report"]["evaluation"]["composite"]["fit_key_split"] == "train"
        assert 0.0 <= control["report"]["evaluation"]["composite"]["auc_advantage"] <= 0.5
    assert report["summary"]["control_count"] == 3
    assert "baseline_vs_max_control_auc_delta" in report["summary"]
    assert "not neural training" in report["claim_scope"]


def test_audit_spn_features_cli_writes_trail_position_control_baseline(tmp_path):
    output = tmp_path / "trail_position_control_baseline.json"
    status = audit_spn_features_main(
        [
            "--trail-position-control-baseline-plan",
            "configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv",
            "--row-index",
            "1",
            "--samples-per-class",
            "4",
            "--seed",
            "123",
            "--top-k",
            "5",
            "--control-active-nibbles",
            "1",
            "--control-input-differences",
            "0x90",
            "--control-pair-orders",
            "reverse",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit"] == "present_trail_position_control_baseline"
    assert payload["baseline"]["report"]["evaluation"]["key_split"] == "validation"
    assert len(payload["controls"]) == 3
    assert payload["controls"][0]["variant_label"] == "active_nibble_1"
    assert payload["controls"][1]["variant_label"] == "input_difference_0x90"
    assert payload["controls"][2]["variant_label"] == "pair_order_reverse"


def test_audit_spn_features_trail_position_split_baseline_uses_dataset_cache(tmp_path):
    output = tmp_path / "trail_position_split_baseline.json"
    cache_root = tmp_path / "trail_position_audit_cache"
    progress = tmp_path / "trail_position_audit_progress.jsonl"

    status = audit_spn_features_main(
        [
            "--trail-position-split-baseline-plan",
            "configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv",
            "--row-index",
            "1",
            "--samples-per-class",
            "4",
            "--seed",
            "123",
            "--top-k",
            "5",
            "--dataset-cache-root",
            str(cache_root),
            "--dataset-cache-chunk-size",
            "2",
            "--dataset-cache-workers",
            "1",
            "--progress-output",
            str(progress),
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit"] == "present_trail_position_split_baseline"
    assert list(cache_root.glob("present80/r8/train/*/metadata.json"))
    assert list(cache_root.glob("present80/r8/validation/*/metadata.json"))
    progress_events = [
        json.loads(line)["event"]
        for line in progress.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "cache_start" in progress_events
    assert "cache_done" in progress_events


def test_candidate_evidence_feature_probe_reports_lowdim_axis_and_composite():
    config = {
        "rounds": 7,
        "seed": 0,
        "samples_per_class": 4,
        "pairs_per_sample": 2,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
        "key_rotation_interval": 0,
        "beam_width": 2,
        "depth": 2,
        "feature_mode": "aggregate",
    }

    report = spn_feature_audit.candidate_evidence_feature_probe_from_config(
        config,
        samples_per_class=4,
        seed=123,
        key_split="validation",
    )

    assert report["audit"] == "candidate_evidence_feature_probe"
    assert report["feature_mode"] == "aggregate"
    assert report["samples_per_class"] == 4
    assert report["feature_dim"] > 0
    assert report["best_axis"]["index"] >= 0
    assert 0.0 <= report["best_axis"]["auc_advantage"] <= 0.5
    assert "feature_names" in report["top_axes"]
    assert 0.0 <= report["composite"]["auc_advantage"] <= 0.5
    assert report["decision"] in {
        "candidate_evidence_lowdim_probe_positive",
        "candidate_evidence_lowdim_probe_weak_or_negative",
    }
    assert "not neural training" in report["claim_scope"]


def test_audit_spn_features_cli_writes_candidate_evidence_feature_probe(tmp_path):
    config_path = tmp_path / "candidate_config.json"
    output = tmp_path / "candidate_feature_probe.json"
    config_path.write_text(
        json.dumps(
            {
                "rounds": 7,
                "seed": 0,
                "samples_per_class": 4,
                "pairs_per_sample": 2,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "zhang_wang_case2_official_mcnd",
                "difference_profile": "present_zhang_wang2022_mcnd",
                "difference_member": 0,
                "validation_key": "0x11111111111111111111",
                "key_rotation_interval": 0,
                "beam_width": 2,
                "depth": 2,
                "feature_mode": "aggregate",
            }
        ),
        encoding="utf-8",
    )

    status = audit_spn_features_main(
        [
            "--candidate-evidence-feature-probe-config",
            str(config_path),
            "--samples-per-class",
            "4",
            "--seed",
            "123",
            "--key-split",
            "validation",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit"] == "candidate_evidence_feature_probe"
    assert payload["feature_mode"] == "aggregate"
    assert "best_axis" in payload
    assert "composite" in payload


def test_sgp_stable_axis_report_selects_axes_that_survive_repeats_and_control():
    labels = np.array([0] * 8 + [1] * 8, dtype=np.uint8)
    probe0 = np.zeros((16, 6), dtype=np.float32)
    probe1 = np.zeros((16, 6), dtype=np.float32)
    false_family = np.zeros((16, 6), dtype=np.float32)
    probe0[:, 1] = labels
    probe0[:, 3] = 1 - labels
    probe1[:, 1] = labels
    probe1[:, 3] = 1 - labels
    probe0[:, 5] = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    probe1[:, 4] = np.linspace(1.0, 0.0, 16, dtype=np.float32)
    false_family[:, 0] = np.tile([0.0, 1.0], 8)

    report = spn_feature_audit.sgp_stable_axis_report_from_feature_matrices(
        [
            {"name": "seed0_validation", "features": probe0, "labels": labels},
            {"name": "seed1_validation", "features": probe1, "labels": labels},
        ],
        controls=[{"name": "false_family", "features": false_family, "labels": labels}],
        top_k=3,
        stable_top_k=2,
        min_composite_auc=0.9,
        min_topk_jaccard=0.3,
        min_control_delta=0.1,
    )

    assert report["audit"] == "sgp_stable_axis_audit"
    assert report["decision"] == "sgp_stable_axis_candidate"
    assert set(report["candidate_masks"]["sgp_top2_stable"]) == {1, 3}
    assert report["stability"]["topk_jaccard_min"] >= 0.5
    assert report["control_gap"]["best_composite_auc_delta"] >= 0.1
    assert report["claim_scope"].startswith("Local SGP stable-axis audit only")


def test_sgp_grouped_axis_report_keeps_unstable_axes_when_group_is_stable():
    labels = np.array([0] * 8 + [1] * 8, dtype=np.uint8)
    probe0 = np.zeros((16, 6), dtype=np.float32)
    probe1 = np.zeros((16, 6), dtype=np.float32)
    control = np.zeros((16, 6), dtype=np.float32)
    probe0[:, 1] = labels
    probe1[:, 2] = labels

    report = spn_feature_audit.sgp_grouped_axis_report_from_feature_matrices(
        [
            {"name": "seed0_validation", "features": probe0, "labels": labels},
            {"name": "seed1_validation", "features": probe1, "labels": labels},
        ],
        controls=[{"name": "false_family", "features": control, "labels": labels}],
        axis_groups=["cell0", "cell0", "cell0", "cell1", "cell1", "cell2"],
        top_k=1,
        stable_top_k=1,
        min_composite_auc=0.9,
        min_topk_jaccard=1.0,
        min_control_delta=0.1,
        source_name="synthetic_invp",
        group_scheme="synthetic_cell",
    )

    assert report["audit"] == "sgp_grouped_axis_audit"
    assert report["decision"] == "sgp_grouped_axis_candidate"
    assert report["source_name"] == "synthetic_invp"
    assert report["group_scheme"] == "synthetic_cell"
    assert report["stability"]["topk_jaccard_min"] == 1.0
    assert report["stability"]["stable_groups"] == ["cell0"]
    assert set(report["candidate_masks"]["sgp_grouped_top1_synthetic_cell"]) == {0, 1, 2}


def test_sgp_grouped_axis_report_rejects_degenerate_full_width_group_mask():
    labels = np.array([0] * 8 + [1] * 8, dtype=np.uint8)
    probe0 = np.zeros((16, 4), dtype=np.float32)
    probe1 = np.zeros((16, 4), dtype=np.float32)
    probe0[:, 0] = labels
    probe0[:, 1] = labels
    probe1[:, 2] = labels
    probe1[:, 3] = labels

    report = spn_feature_audit.sgp_grouped_axis_report_from_feature_matrices(
        [
            {"name": "seed0_validation", "features": probe0, "labels": labels},
            {"name": "seed1_validation", "features": probe1, "labels": labels},
        ],
        axis_groups=["role0", "role1", "role0", "role1"],
        top_k=2,
        stable_top_k=2,
        min_composite_auc=0.9,
        min_topk_jaccard=1.0,
        min_control_delta=0.1,
        max_selected_axis_fraction=0.75,
        group_scheme="synthetic_role",
    )

    assert report["decision"] == "sgp_grouped_axis_hold"
    assert report["degeneracy"]["selected_axis_fraction"] == 1.0
    assert report["degeneracy"]["max_selected_axis_fraction"] == 0.75


def test_sgp_grouped_axis_config_repeats_invp_groups_across_pair_slots():
    groups = spn_feature_audit._axis_groups_for_sgp_source(
        {
            "name": "invp_delta_bits",
            "kind": "differential_feature",
            "feature_encoding": "ciphertext_xor_spn_paligned_bits",
            "pairs_per_sample": 2,
        },
        feature_dim=256,
        group_scheme="word_cell",
    )

    assert len(groups) == 256
    assert groups[0] == groups[128]
    assert groups[64] == groups[192]
    assert groups[0] != groups[64]


def test_audit_spn_features_cli_writes_sgp_grouped_axis_audit(tmp_path, monkeypatch):
    output = tmp_path / "sgp_grouped_axis.json"
    config_path = tmp_path / "sgp_grouped_axis_config.json"
    config_path.write_text(json.dumps({"rounds": 8}), encoding="utf-8")

    def fake_grouped_audit(config_payload, *, samples_per_class=None, top_k=12):
        assert config_payload == {"rounds": 8}
        assert samples_per_class == 16
        assert top_k == 4
        return {
            "audit": "sgp_grouped_axis_audit",
            "decision": "sgp_grouped_axis_candidate",
            "candidate_masks": {"sgp_grouped_top4_word_cell": [64, 65, 66, 67]},
        }

    monkeypatch.setattr(spn_feature_audit, "sgp_grouped_axis_audit_from_config", fake_grouped_audit)

    status = audit_spn_features_main(
        [
            "--sgp-grouped-axis-config",
            str(config_path),
            "--samples-per-class",
            "16",
            "--top-k",
            "4",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit"] == "sgp_grouped_axis_audit"
    assert payload["candidate_masks"]["sgp_grouped_top4_word_cell"] == [64, 65, 66, 67]


def test_audit_spn_features_cli_writes_sgp_stable_axis_audit(tmp_path, monkeypatch):
    output = tmp_path / "sgp_axis.json"
    config_path = tmp_path / "sgp_axis_config.json"
    config_path.write_text(json.dumps({"rounds": 8}), encoding="utf-8")

    def fake_sgp_audit(config_payload, *, samples_per_class=None, top_k=12):
        assert config_payload == {"rounds": 8}
        assert samples_per_class == 16
        assert top_k == 4
        return {
            "audit": "sgp_stable_axis_audit",
            "decision": "sgp_stable_axis_candidate",
            "candidate_masks": {"sgp_top4_stable": [1, 3, 5, 7]},
        }

    monkeypatch.setattr(spn_feature_audit, "sgp_stable_axis_audit_from_config", fake_sgp_audit)

    status = audit_spn_features_main(
        [
            "--sgp-stable-axis-config",
            str(config_path),
            "--samples-per-class",
            "16",
            "--top-k",
            "4",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit"] == "sgp_stable_axis_audit"
    assert payload["candidate_masks"]["sgp_top4_stable"] == [1, 3, 5, 7]


def test_sgp_differential_source_uses_empty_selected_bits_by_default():
    report = spn_feature_audit.sgp_stable_axis_audit_from_config(
        {
            "rounds": 8,
            "seeds": [0, 1],
            "samples_per_class": 4,
            "pairs_per_sample": 1,
            "feature_sources": [
                {
                    "name": "invp_delta_bits",
                    "kind": "differential_feature",
                    "feature_encoding": "ciphertext_xor_spn_paligned_bits",
                }
            ],
        },
        samples_per_class=4,
        top_k=4,
    )

    assert report["audit"] == "sgp_stable_axis_audit"
    assert report["best_source"] == "invp_delta_bits"
    assert report["source_reports"][0]["summary"]["feature_dim"] == 128


def test_invp_global_stats_report_selects_stable_distribution_statistics():
    labels = np.array([0] * 8 + [1] * 8, dtype=np.uint8)
    probe0 = np.zeros((16, 256), dtype=np.float32)
    probe1 = np.zeros((16, 256), dtype=np.float32)
    probe0[labels == 1, :64] = 1.0
    probe1[labels == 1, :64] = 1.0

    report = spn_feature_audit.invp_global_stats_report_from_feature_matrices(
        [
            {"name": "seed0_validation", "features": probe0, "labels": labels},
            {"name": "seed1_validation", "features": probe1, "labels": labels},
        ],
        pairs_per_sample=2,
        pair_bits=128,
        top_k=4,
        min_composite_auc=0.9,
        min_topk_jaccard=1.0,
        min_best_stat_auc=0.9,
        source_name="synthetic_invp_stats",
    )

    assert report["audit"] == "invp_global_stats_audit"
    assert report["decision"] == "invp_global_stats_candidate"
    assert report["source_name"] == "synthetic_invp_stats"
    assert report["summary"]["stat_feature_dim"] == 92
    assert report["stability"]["topk_jaccard_min"] == 1.0
    assert report["summary"]["probe_composite_auc_min"] >= 0.9
    assert report["summary"]["best_stat_auc_min"] >= 0.9
    assert report["candidate_feature_names"]


def test_audit_spn_features_cli_writes_invp_global_stats_audit(tmp_path, monkeypatch):
    output = tmp_path / "invp_global_stats.json"
    config_path = tmp_path / "invp_global_stats_config.json"
    config_path.write_text(json.dumps({"rounds": 8}), encoding="utf-8")

    def fake_global_stats_audit(config_payload, *, samples_per_class=None, top_k=12):
        assert config_payload == {"rounds": 8}
        assert samples_per_class == 16
        assert top_k == 4
        return {
            "audit": "invp_global_stats_audit",
            "decision": "invp_global_stats_candidate",
            "candidate_feature_names": ["word0_mean"],
        }

    monkeypatch.setattr(spn_feature_audit, "invp_global_stats_audit_from_config", fake_global_stats_audit)

    status = audit_spn_features_main(
        [
            "--invp-global-stats-config",
            str(config_path),
            "--samples-per-class",
            "16",
            "--top-k",
            "4",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit"] == "invp_global_stats_audit"
    assert payload["candidate_feature_names"] == ["word0_mean"]


def test_invp_group_distribution_report_keeps_distribution_when_group_identity_moves():
    labels = np.array([0] * 8 + [1] * 8, dtype=np.uint8)
    probe0 = np.zeros((16, 8), dtype=np.float32)
    probe1 = np.zeros((16, 8), dtype=np.float32)
    probe0[labels == 1, 0:2] = 1.0
    probe1[labels == 1, 4:6] = 1.0

    report = spn_feature_audit.invp_group_distribution_report_from_feature_matrices(
        [
            {"name": "seed0_validation", "features": probe0, "labels": labels},
            {"name": "seed1_validation", "features": probe1, "labels": labels},
        ],
        group_schemes={
            "synthetic": ["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"],
        },
        top_k=3,
        min_composite_auc=0.9,
        min_topk_jaccard=1.0,
        min_best_stat_auc=0.9,
        source_name="synthetic_group_distribution",
    )

    assert report["audit"] == "invp_group_distribution_audit"
    assert report["decision"] == "invp_group_distribution_candidate"
    assert report["source_name"] == "synthetic_group_distribution"
    assert report["summary"]["stat_feature_dim"] >= 6
    assert report["stability"]["topk_jaccard_min"] == 1.0
    assert "synthetic:activity_span" in report["candidate_feature_names"]


def test_audit_spn_features_cli_writes_invp_group_distribution_audit(tmp_path, monkeypatch):
    output = tmp_path / "invp_group_distribution.json"
    config_path = tmp_path / "invp_group_distribution_config.json"
    config_path.write_text(json.dumps({"rounds": 8}), encoding="utf-8")

    def fake_group_distribution_audit(config_payload, *, samples_per_class=None, top_k=12):
        assert config_payload == {"rounds": 8}
        assert samples_per_class == 16
        assert top_k == 4
        return {
            "audit": "invp_group_distribution_audit",
            "decision": "invp_group_distribution_candidate",
            "candidate_feature_names": ["cell:activity_span"],
        }

    monkeypatch.setattr(
        spn_feature_audit,
        "invp_group_distribution_audit_from_config",
        fake_group_distribution_audit,
    )

    status = audit_spn_features_main(
        [
            "--invp-group-distribution-config",
            str(config_path),
            "--samples-per-class",
            "16",
            "--top-k",
            "4",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit"] == "invp_group_distribution_audit"
    assert payload["candidate_feature_names"] == ["cell:activity_span"]


def test_present_r8_integral_multi_active_difference_control_plan_is_local_audit_only():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_integral_multi_active_difference_control_smoke.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 1
    task = tasks[0]
    assert task["sample_structure"] == (
        "plaintext_integral_multi_nibble_difference_matched_negative"
    )
    assert task["negative_mode"] == "encrypted_random_plaintexts"
    assert task["rounds"] == 8
    assert task["samples_per_class"] == 128
    assert task["pairs_per_sample"] == 256
    assert task["feature_encoding"] == "ciphertext_pair_bits"
    assert task["difference_profile"] == "present_wang_jain2021_1"
    assert "LOCAL AUDIT only" in task["matching_evidence"]


def test_present_r8_integral_matched_negative_probe_plan_is_local_control():
    plan = "configs/experiment/innovation1/innovation1_spn_present_r8_integral_matched_negative_probe_smoke.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert len(tasks) == 3
    assert {task["sample_structure"] for task in tasks} == {"plaintext_integral_nibble_matched_negative"}
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["samples_per_class"] for task in tasks} == {256}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["feature_encoding"] for task in tasks} == {
        "ciphertext_pair_bits",
        "present_pair_xor_paligned_cell_matrix_bits",
        "present_pair_xor_paligned_sinv_cell_matrix_bits",
    }
    assert all("SMOKE only" in task["matching_evidence"] for task in tasks)


def test_present_r8_integral_matched_negative_probe_seed1_keeps_same_protocol():
    seed0_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_integral_matched_negative_probe_smoke.csv"
    )
    seed1_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_r8_integral_matched_negative_probe_smoke_seed1.csv"
    )

    seed0_tasks = build_tasks(parse_args(["--plan", seed0_plan]))
    seed1_tasks = build_tasks(parse_args(["--plan", seed1_plan]))

    assert len(seed1_tasks) == len(seed0_tasks) == 3
    assert {task["seed"] for task in seed1_tasks} == {1}
    for seed0_task, seed1_task in zip(seed0_tasks, seed1_tasks, strict=True):
        comparable_keys = {
            "model_key",
            "rounds",
            "samples_per_class",
            "pairs_per_sample",
            "feature_encoding",
            "negative_mode",
            "sample_structure",
            "integral_active_nibble",
            "difference_profile",
            "difference_member",
            "loss",
            "learning_rate",
            "optimizer",
            "weight_decay",
            "checkpoint_metric",
            "restore_best_checkpoint",
            "model_options",
        }
        assert {key: seed1_task[key] for key in comparable_keys} == {
            key: seed0_task[key] for key in comparable_keys
        }


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


def test_present_sbox_transition_prior_gate_features_have_controls():
    from blockcipher_nd.features.encoders.bitwise import pair_to_bits
    from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
        PresentNibbleInvPNoDDTGateDistinguisher,
        PresentNibbleInvPSboxPriorGateDistinguisher,
        PresentNibbleInvPShuffledSboxPriorGateDistinguisher,
    )

    left = 0x0123456789ABCDEF
    right = 0x1111111111111111
    raw = torch.tensor([pair_to_bits(left, right, 64)], dtype=torch.float32)
    candidate = PresentNibbleInvPSboxPriorGateDistinguisher(
        input_bits=128,
        pair_bits=128,
        base_channels=4,
        prior_mixer_depth=1,
    )
    no_ddt = PresentNibbleInvPNoDDTGateDistinguisher(
        input_bits=128,
        pair_bits=128,
        base_channels=4,
        prior_mixer_depth=1,
    )
    shuffled = PresentNibbleInvPShuffledSboxPriorGateDistinguisher(
        input_bits=128,
        pair_bits=128,
        base_channels=4,
        prior_mixer_depth=1,
    )

    true_prior = candidate.prior_encoder.sbox_prior_features(raw)
    no_ddt_prior = no_ddt.prior_encoder.sbox_prior_features(raw)
    shuffled_prior = shuffled.prior_encoder.sbox_prior_features(raw)

    assert true_prior.shape == (1, 1, 16, 17)
    assert no_ddt_prior.shape == true_prior.shape
    assert shuffled_prior.shape == true_prior.shape
    assert torch.all((true_prior >= 0.0) & (true_prior <= 1.0))
    assert torch.equal(no_ddt_prior[..., 0], true_prior[..., 0])
    assert torch.count_nonzero(no_ddt_prior[..., 1:]) == 0
    first_output_difference = int(candidate.prior_encoder.invp_nibbles(raw)[0, 0, 0].matmul(torch.tensor([8, 4, 2, 1], dtype=torch.float32)).item())
    expected_column = torch.tensor(
        [
            candidate.prior_encoder.ddt_by_output[first_output_difference, input_difference].item() / 16.0
            for input_difference in range(16)
        ],
        dtype=true_prior.dtype,
    )
    assert torch.allclose(true_prior[0, 0, 0, 1:], expected_column)
    assert not torch.equal(shuffled_prior, true_prior)


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


def test_present_p_layer_mixer_active_token_bias_conditions_tokens_from_metadata():
    cipher = build_cipher("present80", 8, key=0)
    pair_bits = pair_bits_for_encoding(
        cipher.block_bits,
        "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
    )
    model = build_model(
        "present_p_layer_mixer_pairset",
        input_bits=16 * pair_bits + 16,
        hidden_bits=4,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "mixer_depth": 1,
            "token_dim": 8,
            "metadata_bits": 16,
            "active_conditioning": "p_layer_active_token_bias",
            "pooling": "topk_logsumexp",
            "top_k": 2,
        },
    )
    features = torch.zeros(2, 16 * pair_bits + 16)
    features[0, -16] = 1.0
    features[1, -15] = 1.0
    pair_features = torch.zeros(2, pair_bits)
    active_metadata = features[:, -16:]

    pair_embeddings = model._encode_pairs(pair_features, active_metadata)
    logits = model(features)

    assert logits.shape == (2, 1)
    assert pair_embeddings.shape[0] == 2
    assert model.active_cell_roles[0, 15].item() == 2
    assert model.active_cell_roles[0, 11].item() == 1
    assert model.active_cell_roles[0, 7].item() == 1
    assert model.active_cell_roles[0, 3].item() == 1
    assert not torch.allclose(pair_embeddings[0], pair_embeddings[1])


def test_present_p_layer_mixer_shuffled_topology_control_changes_adjacency():
    pair_bits = pair_bits_for_encoding(64, "present_pair_xor_paligned_sinv_cell_matrix_bits")
    common = {
        "input_bits": 16 * pair_bits,
        "hidden_bits": 4,
        "pair_bits": pair_bits,
        "structure": "SPN",
        "model_options": {
            "mixer_depth": 1,
            "token_dim": 8,
            "pooling": "topk_logsumexp",
            "top_k": 2,
        },
    }
    true_model = build_model("present_p_layer_mixer_pairset", **common)
    shuffled_model = build_model(
        "present_p_layer_mixer_pairset",
        **{**common, "model_options": {**common["model_options"], "p_topology": "shuffled"}},
    )
    features = torch.zeros(2, 16 * pair_bits)

    true_logits = true_model(features)
    shuffled_logits = shuffled_model(features)

    assert true_logits.shape == (2, 1)
    assert shuffled_logits.shape == (2, 1)
    assert true_model.p_topology == "true"
    assert shuffled_model.p_topology == "shuffled"
    assert not torch.equal(
        true_model.mixer_blocks[0].p_sources,
        shuffled_model.mixer_blocks[0].p_sources,
    )


def test_present_p_layer_mixer_rejects_unknown_topology():
    pair_bits = pair_bits_for_encoding(64, "present_pair_xor_paligned_sinv_cell_matrix_bits")

    with pytest.raises(ValueError, match="p_topology"):
        build_model(
            "present_p_layer_mixer_pairset",
            input_bits=16 * pair_bits,
            hidden_bits=4,
            pair_bits=pair_bits,
            structure="SPN",
            model_options={
                "mixer_depth": 1,
                "token_dim": 8,
                "p_topology": "teleport",
            },
        )


def test_present_trail_position_stats_accepts_zero_trail_depth_for_raw_prefix_stats():
    pair_bits = pair_bits_for_encoding(64, "present_pair_xor_paligned_sinv_cell_matrix_bits")

    model = build_model(
        "present_trail_position_stats_pairset",
        input_bits=16 * pair_bits + 16,
        hidden_bits=16,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={
            "trail_depth": 0,
            "trail_words_per_depth": 0,
            "stats_hidden_bits": 64,
            "metadata_bits": 16,
            "active_conditioning": "p_layer_relative_stats",
        },
    )

    assert model.trail_depth == 0
    assert model.trail_words_per_depth == 0
    assert model.prefix_words == pair_bits // 64

    features = torch.zeros(2, 16 * pair_bits + 16)
    features[:, -16] = 1
    logits = model(features)

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_present_active_cell_graph_modes_forward_and_change_targets():
    pair_bits = pair_bits_for_encoding(64, "present_pair_xor_paligned_sinv_cell_matrix_bits")
    common = {
        "input_bits": 16 * pair_bits + 16,
        "hidden_bits": 8,
        "pair_bits": pair_bits,
        "structure": "SPN",
        "model_options": {
            "token_dim": 16,
            "graph_depth": 1,
            "metadata_bits": 16,
            "pooling": "topk_logsumexp",
            "top_k": 2,
        },
    }
    true_model = build_model(
        "present_active_cell_graph_pairset",
        **{**common, "model_options": {**common["model_options"], "graph_mode": "true"}},
    )
    shuffled_model = build_model(
        "present_active_cell_graph_pairset",
        **{**common, "model_options": {**common["model_options"], "graph_mode": "shuffled"}},
    )
    metadata_model = build_model(
        "present_active_cell_graph_pairset",
        **{**common, "model_options": {**common["model_options"], "graph_mode": "metadata_only"}},
    )
    persistent_model = build_model(
        "present_active_cell_graph_pairset",
        **{
            **common,
            "model_options": {
                **common["model_options"],
                "graph_mode": "true",
                "edge_mode": "persistent",
            },
        },
    )
    features = torch.zeros(2, 16 * pair_bits + 16)
    features[:, -16] = 1

    true_logits = true_model(features)
    shuffled_logits = shuffled_model(features)
    metadata_logits = metadata_model(features)
    persistent_logits = persistent_model(features)

    assert true_logits.shape == (2, 1)
    assert shuffled_logits.shape == (2, 1)
    assert metadata_logits.shape == (2, 1)
    assert persistent_logits.shape == (2, 1)
    assert torch.isfinite(true_logits).all()
    assert torch.isfinite(shuffled_logits).all()
    assert torch.isfinite(metadata_logits).all()
    assert torch.isfinite(persistent_logits).all()
    assert not torch.equal(true_model.target_masks, shuffled_model.target_masks)
    assert not metadata_model.target_masks.any()
    assert persistent_model.persistent_edge_sources.shape == persistent_model.persistent_edge_targets.shape
    assert persistent_model.persistent_edge_role_ids.shape[1] == persistent_model.persistent_edge_sources.numel()
    assert persistent_model.edge_mode == "persistent"


def test_present_active_cell_graph_rejects_missing_active_metadata():
    pair_bits = pair_bits_for_encoding(64, "present_pair_xor_paligned_sinv_cell_matrix_bits")

    with pytest.raises(ValueError, match="metadata"):
        build_model(
            "present_active_cell_graph_pairset",
            input_bits=16 * pair_bits,
            hidden_bits=8,
            pair_bits=pair_bits,
            structure="SPN",
            model_options={
                "token_dim": 16,
                "graph_depth": 1,
                "metadata_bits": 0,
            },
        )


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
        "present_nibble_invp_pair_mixer_consistency_spn_only",
        "present_nibble_shuffled_paligned_spn_only",
        "present_nibble_no_ddt_graph",
        "present_nibble_ddt_graph",
        "present_nibble_shuffled_ddt_graph",
        "present_nibble_invp_sbox_prior_gate",
        "present_nibble_invp_no_ddt_gate",
        "present_nibble_invp_shuffled_sbox_prior_gate",
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
                "pair_mixer_depth": 1,
                "transition_mixer_depth": 1,
                "ddt_mixer_depth": 1,
                "prior_mixer_depth": 1,
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


def test_result_plan_alignment_distinguishes_selected_bit_projection_rows(tmp_path):
    plan_path = tmp_path / "projection_plan.csv"
    result_path = tmp_path / "projection_results.jsonl"
    fieldnames = [
        "cipher",
        "model_key",
        "rounds",
        "seed",
        "samples_per_class",
        "pairs_per_sample",
        "feature_encoding",
        "selected_bit_indices",
    ]
    rows = [
        {
            "cipher": "PRESENT-80",
            "model_key": "mlp",
            "rounds": "8",
            "seed": "0",
            "samples_per_class": "8",
            "pairs_per_sample": "16",
            "feature_encoding": "ciphertext_pair_bits",
            "selected_bit_indices": "",
        },
        {
            "cipher": "PRESENT-80",
            "model_key": "mlp",
            "rounds": "8",
            "seed": "0",
            "samples_per_class": "8",
            "pairs_per_sample": "16",
            "feature_encoding": "ciphertext_pair_bits",
            "selected_bit_indices": "[0,1,64,65]",
        },
    ]
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in [
                {
                    "cipher": "PRESENT-80",
                    "model": "mlp",
                    "selected_model": "mlp",
                    "rounds": 8,
                    "seed": 0,
                    "samples_per_class": 8,
                    "pairs_per_sample": 16,
                    "feature_encoding": "ciphertext_pair_bits",
                    "training": {"selected_bit_indices": []},
                },
                {
                    "cipher": "PRESENT-80",
                    "model": "mlp",
                    "selected_model": "mlp",
                    "rounds": 8,
                    "seed": 0,
                    "samples_per_class": 8,
                    "pairs_per_sample": 16,
                    "feature_encoding": "ciphertext_pair_bits",
                    "training": {"selected_bit_indices": [0, 1, 64, 65]},
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_result_plan_alignment(plan_path, result_path)

    assert report["status"] == "pass"
    assert report["duplicate_plan_keys"] == []
    assert report["duplicate_result_keys"] == []


def test_result_plan_alignment_distinguishes_model_options_rows(tmp_path):
    plan_path = tmp_path / "model_options_plan.csv"
    result_path = tmp_path / "model_options_results.jsonl"
    fieldnames = [
        "cipher",
        "model_key",
        "rounds",
        "seed",
        "samples_per_class",
        "pairs_per_sample",
        "feature_encoding",
        "sample_structure",
        "model_options",
    ]
    rows = [
        {
            "cipher": "PRESENT-80",
            "model_key": "present_trail_position_stats_pairset",
            "rounds": "8",
            "seed": "0",
            "samples_per_class": "512",
            "pairs_per_sample": "16",
            "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
            "sample_structure": "plaintext_integral_nibble_difference_matched_negative_random_active_metadata",
            "model_options": '{"metadata_bits":16}',
        },
        {
            "cipher": "PRESENT-80",
            "model_key": "present_trail_position_stats_pairset",
            "rounds": "8",
            "seed": "0",
            "samples_per_class": "512",
            "pairs_per_sample": "16",
            "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
            "sample_structure": "plaintext_integral_nibble_difference_matched_negative_random_active_metadata",
            "model_options": '{"active_conditioning":"relative_stats","metadata_bits":16}',
        },
    ]
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "cipher": row["cipher"],
                    "model": row["model_key"],
                    "selected_model": row["model_key"],
                    "rounds": int(row["rounds"]),
                    "seed": int(row["seed"]),
                    "samples_per_class": int(row["samples_per_class"]),
                    "pairs_per_sample": int(row["pairs_per_sample"]),
                    "feature_encoding": row["feature_encoding"],
                    "sample_structure": row["sample_structure"],
                    "training": {"model_options": json.loads(row["model_options"])},
                },
                sort_keys=True,
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_result_plan_alignment(plan_path, result_path)

    assert report["status"] == "pass"
    assert report["duplicate_plan_keys"] == []
    assert report["duplicate_result_keys"] == []


def test_result_plan_alignment_distinguishes_protocol_audit_rows(tmp_path):
    plan_path = tmp_path / "protocol_audit_plan.csv"
    result_path = tmp_path / "protocol_audit_results.jsonl"
    fieldnames = [
        "cipher",
        "model_key",
        "rounds",
        "seed",
        "samples_per_class",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "integral_active_nibble",
        "key_rotation_interval",
    ]
    rows = [
        {
            "cipher": "PRESENT-80",
            "model_key": "present_pairset_global_stats",
            "rounds": "8",
            "seed": "0",
            "samples_per_class": "512",
            "pairs_per_sample": "16",
            "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "plaintext_integral_nibble_difference_matched_negative",
            "integral_active_nibble": "0",
            "key_rotation_interval": "0",
        },
        {
            "cipher": "PRESENT-80",
            "model_key": "present_pairset_global_stats",
            "rounds": "8",
            "seed": "0",
            "samples_per_class": "512",
            "pairs_per_sample": "16",
            "feature_encoding": "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "plaintext_integral_nibble_strict_random_negative",
            "integral_active_nibble": "0",
            "key_rotation_interval": "1",
        },
    ]
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "cipher": row["cipher"],
                    "model": row["model_key"],
                    "selected_model": row["model_key"],
                    "rounds": int(row["rounds"]),
                    "seed": int(row["seed"]),
                    "samples_per_class": int(row["samples_per_class"]),
                    "pairs_per_sample": int(row["pairs_per_sample"]),
                    "feature_encoding": row["feature_encoding"],
                    "negative_mode": row["negative_mode"],
                    "sample_structure": row["sample_structure"],
                    "integral_active_nibble": int(row["integral_active_nibble"]),
                    "key_rotation_interval": int(row["key_rotation_interval"]),
                },
                sort_keys=True,
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_result_plan_alignment(plan_path, result_path)

    assert report["status"] == "pass"
    assert report["duplicate_plan_keys"] == []
    assert report["duplicate_result_keys"] == []


def test_present_r8_projection_v2_smoke_plan_compares_invp_delta_priors():
    plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_truncated_projection_feature_v2_smoke.csv"
    )
    tasks = build_tasks(parse_args(["--plan", plan]))

    assert len(tasks) == 4
    assert {task["model_key"] for task in tasks} == {"mlp"}
    assert {task["rounds"] for task in tasks} == {8}
    assert {task["samples_per_class"] for task in tasks} == {512}
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["sample_structure"] for task in tasks} == {"zhang_wang_case2_official_mcnd"}
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["feature_encoding"] for task in tasks} == {"ciphertext_xor_spn_paligned_bits"}
    assert [len(task["selected_bit_indices"]) for task in tasks] == [32, 32, 16, 32]
    assert all("LOCAL SMOKE only" in task["matching_evidence"] for task in tasks)
    assert all("projection v2" in task["literature"] for task in tasks)


def test_present_r8_projection_v2_seed1_keeps_same_projection_priors():
    seed0_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_truncated_projection_feature_v2_smoke.csv"
    )
    seed1_plan = (
        "configs/experiment/innovation1/"
        "innovation1_spn_present_truncated_projection_feature_v2_smoke_seed1.csv"
    )
    seed0_tasks = build_tasks(parse_args(["--plan", seed0_plan]))
    seed1_tasks = build_tasks(parse_args(["--plan", seed1_plan]))

    assert len(seed1_tasks) == len(seed0_tasks) == 4
    assert {task["seed"] for task in seed1_tasks} == {1}
    for seed0_task, seed1_task in zip(seed0_tasks, seed1_tasks, strict=True):
        comparable_keys = {
            "model_key",
            "rounds",
            "samples_per_class",
            "pairs_per_sample",
            "feature_encoding",
            "selected_bit_indices",
            "negative_mode",
            "sample_structure",
            "difference_profile",
            "difference_member",
            "loss",
            "learning_rate",
            "optimizer",
            "weight_decay",
            "checkpoint_metric",
            "restore_best_checkpoint",
            "model_options",
        }
        assert {key: seed1_task[key] for key in comparable_keys} == {
            key: seed0_task[key] for key in comparable_keys
        }


def test_result_plan_alignment_distinguishes_difference_screen_rows(tmp_path):
    plan_path = tmp_path / "difference_screen_plan.csv"
    result_path = tmp_path / "difference_screen_results.jsonl"
    fieldnames = [
        "cipher",
        "model_key",
        "rounds",
        "seed",
        "samples_per_class",
        "pairs_per_sample",
        "feature_encoding",
        "difference_profile",
        "difference_member",
    ]
    rows = [
        {
            "cipher": "PRESENT-80",
            "model_key": "present_nibble_invp_pair_consistency_spn_only",
            "rounds": "9",
            "seed": "0",
            "samples_per_class": "65536",
            "pairs_per_sample": "16",
            "feature_encoding": "ciphertext_pair_bits",
            "difference_profile": "present_zhang_wang2022_mcnd",
            "difference_member": "0",
        },
        {
            "cipher": "PRESENT-80",
            "model_key": "present_nibble_invp_pair_consistency_spn_only",
            "rounds": "9",
            "seed": "0",
            "samples_per_class": "65536",
            "pairs_per_sample": "16",
            "feature_encoding": "ciphertext_pair_bits",
            "difference_profile": "present_wang_jain2021",
            "difference_member": "0",
        },
    ]
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "cipher": row["cipher"],
                    "model": row["model_key"],
                    "selected_model": row["model_key"],
                    "rounds": int(row["rounds"]),
                    "seed": int(row["seed"]),
                    "samples_per_class": int(row["samples_per_class"]),
                    "pairs_per_sample": int(row["pairs_per_sample"]),
                    "feature_encoding": row["feature_encoding"],
                    "difference_profile": row["difference_profile"],
                    "difference_member": int(row["difference_member"]),
                },
                sort_keys=True,
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_result_plan_alignment(plan_path, result_path)

    assert report["status"] == "pass"
    assert report["duplicate_plan_keys"] == []
    assert report["duplicate_result_keys"] == []


def test_projection_ensemble_metrics_reports_weighted_logit_mode():
    from blockcipher_nd.cli.evaluate_projection_ensemble import ensemble_metrics

    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    probabilities = np.array(
        [
            [0.10, 0.40],
            [0.20, 0.30],
            [0.80, 0.55],
            [0.70, 0.60],
        ],
        dtype=np.float32,
    )
    row_reports = [
        {"metrics": {"auc": 1.0}},
        {"metrics": {"auc": 0.75}},
    ]

    reports = ensemble_metrics(labels, probabilities, row_reports)

    assert [report["mode"] for report in reports] == [
        "probability_mean",
        "logit_mean",
        "auc_weighted_logit_mean",
    ]
    weighted = reports[2]
    assert weighted["weights"][0] > weighted["weights"][1]
    assert weighted["metrics"]["auc"] == 1.0


def test_projection_ensemble_diversity_reports_error_complementarity():
    from blockcipher_nd.cli.evaluate_projection_ensemble import diversity_metrics

    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    probabilities = np.array(
        [
            [0.10, 0.60],
            [0.80, 0.20],
            [0.40, 0.80],
            [0.90, 0.30],
        ],
        dtype=np.float32,
    )
    row_reports = [
        {"architecture": "raw_projection", "metrics": {"auc": 0.75}},
        {"architecture": "invp_projection", "metrics": {"auc": 0.75}},
    ]

    report = diversity_metrics(labels, probabilities, row_reports)

    assert report["oracle_accuracy_at_0_5"] == 1.0
    assert report["all_models_wrong_rate_at_0_5"] == 0.0
    assert len(report["pairwise"]) == 1
    pair = report["pairwise"][0]
    assert pair["left"] == "raw_projection"
    assert pair["right"] == "invp_projection"
    assert pair["disagreement_rate_at_0_5"] == 1.0
    assert pair["double_fault_rate_at_0_5"] == 0.0
    assert pair["error_jaccard_at_0_5"] == 0.0


def test_projection_ensemble_source_results_filter_selects_weak_positive_candidates(tmp_path):
    from blockcipher_nd.cli.evaluate_projection_ensemble import filter_tasks_from_source_results

    tasks = [
        {"architecture": "full_raw"},
        {"architecture": "raw_projection"},
        {"architecture": "invp_projection"},
        {"architecture": "thin_projection"},
    ]
    source_results = tmp_path / "screen.jsonl"
    rows = [
        {"architecture": "full_raw", "architecture_rank": 0, "metrics": {"auc": 0.510}},
        {"architecture": "raw_projection", "architecture_rank": 1, "metrics": {"auc": 0.512}},
        {"architecture": "invp_projection", "architecture_rank": 2, "metrics": {"auc": 0.513}},
        {"architecture": "thin_projection", "architecture_rank": 3, "metrics": {"auc": 0.504}},
    ]
    source_results.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    filtered = filter_tasks_from_source_results(tasks, source_results, weak_positive_auc=0.505)

    assert [task["architecture"] for task in filtered["tasks"]] == [
        "raw_projection",
        "invp_projection",
    ]
    assert filtered["report"]["mode"] == "weak_projection_candidates_from_source_results"
    assert filtered["report"]["selected_count"] == 2
    assert filtered["report"]["anchor_auc"] == 0.510


def test_projection_feature_gate_triggers_ensemble_for_multiple_weak_views(tmp_path):
    from blockcipher_nd.planning.projection_feature_postprocess import gate_projection_feature_result

    results_path = tmp_path / "projection_results.jsonl"
    rows = [
        {
            "architecture": "full_raw",
            "architecture_rank": 0,
            "selected_model": "mlp",
            "feature_encoding": "ciphertext_pair_bits",
            "metrics": {"auc": 0.51, "accuracy": 0.5, "calibrated_accuracy": 0.52, "loss": 0.69},
            "training": {"selected_bit_indices": []},
        },
        {
            "architecture": "raw_projection",
            "architecture_rank": 1,
            "selected_model": "mlp",
            "feature_encoding": "ciphertext_pair_bits",
            "metrics": {"auc": 0.512, "accuracy": 0.5, "calibrated_accuracy": 0.521, "loss": 0.69},
            "training": {"selected_bit_indices": [0, 1, 64, 65]},
        },
        {
            "architecture": "invp_projection",
            "architecture_rank": 2,
            "selected_model": "mlp",
            "feature_encoding": "ciphertext_xor_spn_paligned_bits",
            "metrics": {"auc": 0.513, "accuracy": 0.5, "calibrated_accuracy": 0.522, "loss": 0.69},
            "training": {"selected_bit_indices": [64, 65, 66, 67]},
        },
        {
            "architecture": "thin_projection",
            "architecture_rank": 3,
            "selected_model": "mlp",
            "feature_encoding": "ciphertext_xor_spn_paligned_bits",
            "metrics": {"auc": 0.49, "accuracy": 0.5, "calibrated_accuracy": 0.5, "loss": 0.7},
            "training": {"selected_bit_indices": [64, 65]},
        },
    ]
    results_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = gate_projection_feature_result(results_path, expected_rows=4)

    assert report["status"] == "pass"
    assert report["decision"] == "run_projection_ensemble_diagnostic"
    assert len(report["weak_ensemble_candidates"]) == 2


def test_projection_feature_advance_runs_postprocess_and_writes_summary(tmp_path):
    from blockcipher_nd.planning.projection_feature_advance import advance_projection_feature_result

    plan_path = tmp_path / "projection_plan.csv"
    result_path = tmp_path / "projection_results.jsonl"
    output_dir = tmp_path / "advance"
    fieldnames = [
        "cipher",
        "model_key",
        "rounds",
        "seed",
        "samples_per_class",
        "pairs_per_sample",
        "feature_encoding",
        "selected_bit_indices",
    ]
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "cipher": "PRESENT-80",
                    "model_key": "mlp",
                    "rounds": "8",
                    "seed": "0",
                    "samples_per_class": "8",
                    "pairs_per_sample": "16",
                    "feature_encoding": "ciphertext_pair_bits",
                    "selected_bit_indices": "",
                },
                {
                    "cipher": "PRESENT-80",
                    "model_key": "mlp",
                    "rounds": "8",
                    "seed": "0",
                    "samples_per_class": "8",
                    "pairs_per_sample": "16",
                    "feature_encoding": "ciphertext_pair_bits",
                    "selected_bit_indices": "[0,1,64,65]",
                },
            ]
        )
    result_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in [
                {
                    "architecture": "full_raw",
                    "architecture_rank": 0,
                    "cipher": "PRESENT-80",
                    "selected_model": "mlp",
                    "rounds": 8,
                    "seed": 0,
                    "samples_per_class": 8,
                    "pairs_per_sample": 16,
                    "feature_encoding": "ciphertext_pair_bits",
                    "metrics": {"auc": 0.51, "accuracy": 0.5, "calibrated_accuracy": 0.52, "loss": 0.69},
                    "training": {"selected_bit_indices": []},
                },
                {
                    "architecture": "raw_projection",
                    "architecture_rank": 1,
                    "cipher": "PRESENT-80",
                    "selected_model": "mlp",
                    "rounds": 8,
                    "seed": 0,
                    "samples_per_class": 8,
                    "pairs_per_sample": 16,
                    "feature_encoding": "ciphertext_pair_bits",
                    "metrics": {"auc": 0.53, "accuracy": 0.51, "calibrated_accuracy": 0.54, "loss": 0.68},
                    "training": {"selected_bit_indices": [0, 1, 64, 65]},
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = advance_projection_feature_result(
        results_path=result_path,
        output_dir=output_dir,
        run_id="projection_advance_unit",
        plan_path=plan_path,
        expected_rows=2,
        skip_plot=True,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "promote_projection_to_262k_confirmation"
    assert report["plot"] is None
    assert report["ensemble_ran"] is False
    assert Path(report["summary"]).exists()
    assert Path(report["postprocess_summary"]).exists()


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


def test_training_history_plot_ignores_nonfinite_points(tmp_path):
    from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves

    results_path = tmp_path / "results.jsonl"
    svg_path = tmp_path / "curves.svg"
    results_path.write_text(
        json.dumps(
            {
                "cipher": "PRESENT-80",
                "model": "present_pairset_global_stats",
                "selected_model": "present_pairset_global_stats",
                "rounds": 8,
                "seed": 0,
                "samples_per_class": 8,
                "pairs_per_sample": 16,
                "history": [
                    {
                        "epoch": 1.0,
                        "train_eval_loss": float("nan"),
                        "train_accuracy": 0.5,
                        "train_auc": 0.0,
                        "val_loss": float("nan"),
                        "val_accuracy": 0.5,
                        "val_auc": 0.0,
                        "learning_rate": 0.0001,
                    },
                    {
                        "epoch": 2.0,
                        "train_eval_loss": 0.69,
                        "train_accuracy": 0.5,
                        "train_auc": 0.0,
                        "val_loss": 0.70,
                        "val_accuracy": 0.5,
                        "val_auc": 0.0,
                        "learning_rate": 0.0001,
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = plot_jsonl_training_curves(results_path, svg_path)

    assert report["series"] == 6
    assert "<svg" in svg_path.read_text(encoding="utf-8")


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
