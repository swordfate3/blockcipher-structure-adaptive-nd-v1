from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.planning.cross_spn_mainstream_performance_gate import (
    joint_mainstream_performance_gate,
    paired_mainstream_performance_interval_gate,
    performance_decision,
)
from blockcipher_nd.training.metrics import binary_auc


def _metrics(
    *,
    source0: float,
    source1: float,
    scratch: float = 0.58,
    lstm: float = 0.59,
    resnet: float = 0.585,
) -> dict[str, float]:
    return {
        "typed_scratch": scratch,
        "typed_source0": source0,
        "typed_source1": source1,
        "lstm": lstm,
        "resnet": resnet,
    }


def test_performance_decision_supports_source_robust_superiority() -> None:
    report = performance_decision(
        primary_aucs=_metrics(source0=0.596, source1=0.595),
        final_mean_aucs=_metrics(source0=0.595, source1=0.594),
        epoch1_aucs=_metrics(source0=0.590, source1=0.589, scratch=0.58),
    )

    assert report["decision"] == "large_scale_mainstream_superiority_candidate"
    assert report["gates"]["mainstream_superiority"] is True
    assert report["gates"]["epoch1_source_robust_adaptation"] is True


def test_performance_decision_separates_competitiveness_from_superiority() -> None:
    report = performance_decision(
        primary_aucs=_metrics(source0=0.590, source1=0.590),
        final_mean_aucs=_metrics(source0=0.590, source1=0.5895),
        epoch1_aucs=_metrics(source0=0.581, source1=0.580, scratch=0.58),
    )

    assert report["decision"] == "large_scale_mainstream_competitive_no_superiority"
    assert report["gates"]["mainstream_superiority"] is False
    assert report["gates"]["mainstream_competitiveness"] is True


def test_joint_gate_requires_both_target_seeds() -> None:
    gates = [
        {
            "status": "pass",
            "errors": [],
            "expected_seed": 6,
            "decision": "large_scale_mainstream_superiority_candidate",
        },
        {
            "status": "pass",
            "errors": [],
            "expected_seed": 7,
            "decision": "large_scale_mainstream_competitive_no_superiority",
        },
    ]

    report = joint_mainstream_performance_gate(gates)

    assert report["status"] == "pass"
    assert report["decision"] == "two_seed_large_scale_mainstream_competitive_no_superiority"


def test_large_scale_plans_lock_two_target_seeds_and_five_roles() -> None:
    expected_models = {
        "gift_cross_spn_typed_cell_true",
        "gift_cross_spn_typed_cell_true_from_present_true_s0",
        "gift_cross_spn_typed_cell_true_from_present_true_s1",
        "gift64_sun_style_lstm_pairset",
        "gift64_gohr_style_resnet_pairset",
    }
    for seed in (6, 7):
        path = Path(
            "configs/experiment/innovation1/"
            f"innovation1_spn_gift64_mainstream_performance_1m_seed{seed}.csv"
        )
        tasks = build_tasks(parse_train_args(["--plan", str(path)]))

        assert len(tasks) == 5
        assert {task["model_key"] for task in tasks} == expected_models
        assert {task["seed"] for task in tasks} == {seed}
        assert {task["samples_per_class"] for task in tasks} == {1_000_000}
        assert {task["validation_samples_total"] for task in tasks} == {1_000_000}
        assert {task["final_test_samples_total"] for task in tasks} == {1_000_000}
        assert {task["final_test_repeats"] for task in tasks} == {5}
        assert {task["final_test_key"] for task in tasks} == {int("22" * 16, 16)}
        assert {task["negative_mode"] for task in tasks} == {
            "encrypted_random_plaintexts"
        }


def test_large_scale_remote_assets_pass_readiness_and_path_policy() -> None:
    for seed, gpu in ((6, 0), (7, 1)):
        config_path = Path(
            "configs/remote/"
            f"innovation1_gift64_mainstream_performance_1m_seed{seed}_gpu{gpu}_20260715.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        report = remote_readiness_report(config_path)

        assert report["status"] == "pass", report["errors"]
        assert report["plan_rows"] == 5
        assert config["physical_gpu"] == gpu
        assert config["dataset_cache"] is True
        assert config["dataset_cache_root"].startswith(
            "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
        )
        assert "cmd.exe /c" in config["launch_policy"]
        assert "cmd.exe /k" not in config["launch_policy"]

    run_script = Path(
        "configs/remote/generated/run_i1_gift64_mainstream_performance_1m_20260715.cmd"
    ).read_text(encoding="utf-8")
    launch_script = Path(
        "configs/remote/generated/launch_i1_gift64_mainstream_performance_1m_20260715.cmd"
    ).read_text(encoding="utf-8")
    monitor_script = Path(
        "configs/remote/generated/monitor_i1_gift64_mainstream_performance_1m_20260715.sh"
    ).read_text(encoding="utf-8")
    recovery_script = Path(
        "configs/remote/generated/"
        "recover_i1_gift64_mainstream_performance_1m_20260715.cmd"
    ).read_text(encoding="utf-8")
    recovery_launcher = Path(
        "configs/remote/generated/"
        "launch_recover_i1_gift64_mainstream_performance_1m_20260715.cmd"
    ).read_text(encoding="utf-8")

    assert run_script.count("call :export_score") == 5
    assert "--split final_test_1" in run_script
    assert "--dataset-cache-root" in run_script
    assert "--train-eval-interval 0" in run_script
    assert "cmd.exe /k" not in run_script
    assert launch_script.count("cmd.exe /c") == 2
    assert "cmd.exe /k" not in launch_script
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_script
    assert "scripts/index-results" in monitor_script
    assert "recovery_started.marker" in monitor_script
    assert "recovery_failed.marker" in monitor_script
    assert "mktemp -d /tmp/gift64-mainstream-retrieval" in monitor_script
    assert "cp -a" in monitor_script
    assert "results_archive/${run_id}/." not in monitor_script
    assert "results_archive/${JOINT_ID}/." not in monitor_script
    assert "gate-cross-spn-mainstream-performance" in recovery_script
    assert "gate-cross-spn-mainstream-performance-joint" in recovery_script
    assert "recovery_commit.txt" in recovery_script
    assert "primary_scores.npz" in recovery_script
    assert "scripts\\train" not in recovery_script
    assert "cmd.exe /k" not in recovery_script
    assert "cmd.exe /c" in recovery_launcher
    assert "cmd.exe /k" not in recovery_launcher
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in recovery_launcher


def test_mainstream_gate_entrypoints_bootstrap_the_source_tree() -> None:
    for path in (
        Path("scripts/gate-cross-spn-mainstream-performance"),
        Path("scripts/gate-cross-spn-mainstream-performance-joint"),
        Path("scripts/gate-cross-spn-mainstream-performance-paired"),
    ):
        text = path.read_text(encoding="utf-8")
        assert 'Path(__file__).resolve().parents[1] / "src"' in text
        assert "sys.path.insert(0," in text


def test_paired_interval_gate_supports_two_seed_superiority(tmp_path: Path) -> None:
    rng = np.random.default_rng(47)
    labels = np.tile(np.array([0.0, 1.0], dtype=np.float32), 1_000)
    sample_ids = np.arange(len(labels), dtype=np.int64)
    payloads = {}
    for seed in (6, 7):
        mainstream = rng.normal(0.0, 1.0, len(labels))
        scratch = labels + rng.normal(0.0, 0.8, len(labels))
        source0 = labels + rng.normal(0.0, 0.45, len(labels))
        source1 = labels + rng.normal(0.0, 0.45, len(labels))
        payloads[seed] = {
            "typed_scratch": scratch,
            "typed_source0": source0,
            "typed_source1": source1,
            "lstm": mainstream,
            "resnet": mainstream + rng.normal(0.0, 0.01, len(labels)),
        }

    gate_paths = {}
    score_paths = {}
    for seed, scores in payloads.items():
        gate_paths[seed] = tmp_path / f"seed{seed}_gate.json"
        score_paths[seed] = tmp_path / f"seed{seed}_scores.npz"
        gate_paths[seed].write_text(
            json.dumps(
                {
                    "status": "pass",
                    "errors": [],
                    "expected_seed": seed,
                    "decision": "large_scale_mainstream_superiority_candidate",
                    "strongest_mainstream_role": "resnet",
                    "score_rows": len(labels),
                    "primary_fresh_test_aucs": {
                        role: binary_auc(labels, values)
                        for role, values in scores.items()
                    },
                }
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            score_paths[seed],
            labels=labels,
            sample_ids=sample_ids,
            **{f"{role}_probabilities": values for role, values in scores.items()},
        )

    report = paired_mainstream_performance_interval_gate(
        seed_gate_paths=gate_paths,
        primary_score_paths=score_paths,
        replicates=80,
        chunk_size=8,
        expected_score_rows=len(labels),
    )

    assert report["status"] == "pass"
    assert report["decision"] == "two_seed_paired_mainstream_superiority_supported"
    assert report["gates"]["two_seed_paired_mainstream_superiority"] is True
