from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from blockcipher_nd.cli.plot_runtime_spn_trainable_post_expert_adapter_k1by13 import (
    render_k1by13_svg,
)
from blockcipher_nd.cli.run_runtime_spn_trainable_post_expert_adapter_k1by13 import (
    main as run_main,
    training_argv,
)
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_trainable_post_expert_adapter_k1by13 as k1by13,
)


@pytest.fixture(scope="module")
def tasks() -> list[dict[str, object]]:
    return k1by13.read_tasks()


@pytest.fixture(scope="module")
def readiness(tasks: list[dict[str, object]]) -> dict[str, object]:
    return k1by13.build_readiness(tasks=tasks)


def test_k1by13_plan_and_sources_are_exact(
    tasks: list[dict[str, object]],
) -> None:
    assert len(tasks) == 8
    assert k1by13.candidate_protocol_frozen(tasks)
    assert set(k1by13.task_map(tasks)) == k1by13.expected_keys()
    assert all(k1by13.source_binding_checks().values())


def test_k1by13_readiness_proves_zero_initialized_adapter(
    readiness: dict[str, object],
) -> None:
    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert readiness["local_training_authorized"] is False
    assert readiness["remote_cuda_training_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())

    metrics = readiness["evidence_metrics"]
    assert metrics["parameter_counts"] == {
        "anchor_correct": 235780,
        "adapter_correct": 237876,
        "adapter_affine": 237876,
        "adapter_shuffled": 237876,
    }
    assert metrics["adapter_parameter_delta"] == 2096
    assert metrics["adapter_output_gradient_l1"] > 0.0
    assert metrics["initial_output_max_delta"]["adapter_correct"] == 0.0
    assert metrics["initial_output_max_delta"]["adapter_shuffled"] == 0.0
    assert len(set(metrics["edge_binding_fingerprints"].values())) == 3
    assert (
        metrics["program_semantic_sha256"]["adapter_correct"]
        == metrics["program_semantic_sha256"]["adapter_shuffled"]
    )


def test_k1by13_training_parser_reconstructs_frozen_tasks(
    tasks: list[dict[str, object]],
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        plan=k1by13.PLAN_PATH,
        output_root=tmp_path / "run",
        device="cuda",
    )
    assert build_tasks(parse_train_args(training_argv(args))) == tasks


def test_k1by13_full_cpu_training_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        run_main(
            [
                "--plan",
                str(k1by13.PLAN_PATH),
                "--output-root",
                str(tmp_path / "cpu-run"),
                "--device",
                "cpu",
            ]
        )


def test_k1by13_readiness_cli_writes_only_zero_training_artifacts(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "readiness"
    assert (
        run_main(
            [
                "--plan",
                str(k1by13.PLAN_PATH),
                "--output-root",
                str(output_root),
                "--device",
                "cpu",
                "--readiness-only",
            ]
        )
        == 0
    )
    preflight = json.loads(
        (output_root / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["status"] == "pass"
    assert preflight["requested_device"] == "cpu"
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (output_root / "validation.json").read_text(encoding="utf-8")
    )
    assert gate["decision"] == (
        "innovation1_runtime_spn_k1by13_readiness_authorized"
    )
    assert gate["training_performed"] is False
    assert validation["optimizer_steps"] == 0
    assert (output_root / "progress.jsonl").is_file()
    assert not (output_root / "results.jsonl").exists()
    assert not (output_root / "checkpoints").exists()


def test_k1by13_checkpoint_adapter_norm_reads_output_projection(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    torch.save(
        {
            "state_dict": {
                "conditioner.post_expert_trainable_adapter."
                "output_projection.weight": torch.full((32, 16), 0.25),
                "conditioner.post_expert_trainable_adapter."
                "output_projection.bias": torch.full((32,), 0.5),
            }
        },
        checkpoint,
    )
    rows = {
        (2, "adapter_correct"): {
            "training": {"checkpoint_output": f"remote\\{checkpoint.name}"}
        }
    }
    norms, errors = k1by13.checkpoint_adapter_norms(
        rows,
        checkpoint_root=tmp_path,
    )
    assert not errors
    assert norms[(2, "adapter_correct")] == pytest.approx(
        ((32 * 16 * 0.25**2) + (32 * 0.5**2)) ** 0.5
    )


def test_k1by13_cache_gate_requires_four_creates_and_twelve_reuses() -> None:
    rows = [
        {"event": "cache_start", "split": split}
        for split in ("train", "validation")
        for _ in range(2)
    ]
    rows.extend(
        {"event": "cache_reuse", "split": split}
        for split in ("train", "validation")
        for _ in range(6)
    )
    assert k1by13.cache_protocol_frozen(rows)
    assert not k1by13.cache_protocol_frozen(rows[:-1])


def test_k1by13_adjudication_requires_both_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tasks: list[dict[str, object]],
    readiness: dict[str, object],
    tmp_path: Path,
) -> None:
    rows = _result_rows(
        {
            2: (0.68, 0.69, 0.66, 0.65),
            3: (0.67, 0.68, 0.65, 0.64),
        }
    )
    monkeypatch.setattr(k1by13, "training_protocol_frozen", lambda *_a, **_k: True)
    monkeypatch.setattr(k1by13, "cache_protocol_frozen", lambda _rows: True)
    monkeypatch.setattr(
        k1by13,
        "checkpoint_adapter_norms",
        lambda *_a, **_k: (
            {
                (seed, condition): 0.25
                for seed in k1by13.EXPECTED_SEEDS
                for condition in k1by13.CONDITIONS
                if condition != "anchor_correct"
            },
            [],
        ),
    )
    gate = k1by13.adjudicate(
        tasks=tasks,
        result_rows=rows,
        progress_rows=[],
        readiness=readiness,
        checkpoint_root=tmp_path,
    )
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_runtime_spn_k1by13_trainable_adapter_supported"
    )

    rows = _result_rows(
        {
            2: (0.68, 0.69, 0.66, 0.65),
            3: (0.67, 0.68, 0.65, 0.679),
        }
    )
    gate = k1by13.adjudicate(
        tasks=tasks,
        result_rows=rows,
        progress_rows=[],
        readiness=readiness,
        checkpoint_root=tmp_path,
    )
    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation1_runtime_spn_k1by13_capacity_without_edge_use"
    )
    assert "seed3_shuffled_margin" in gate["failed_research_checks"]


def test_k1by13_plot_contains_clear_chinese_labels(tmp_path: Path) -> None:
    gate = {
        "decision": "innovation1_runtime_spn_k1by13_trainable_adapter_supported",
        "seed_results": {
            str(seed): {
                "auc_by_condition": {
                    "anchor_correct": 0.68,
                    "adapter_correct": 0.69,
                    "adapter_affine": 0.66,
                    "adapter_shuffled": 0.65,
                },
                "correct_minus_control": {
                    "anchor_correct": 0.01,
                    "adapter_affine": 0.03,
                    "adapter_shuffled": 0.04,
                },
            }
            for seed in k1by13.EXPECTED_SEEDS
        },
    }
    output = tmp_path / "curves.svg"
    report = render_k1by13_svg(gate, output)
    svg = output.read_text(encoding="utf-8")
    assert report["status"] == "pass"
    assert "PRESENT-80 七轮" in svg
    assert "可训练适配器：正确边" in svg
    assert "可训练适配器：仿射错误边" in svg
    assert "结构优势门槛" in svg
    assert "主指标为跨密钥验证 AUC" in svg


def test_k1by13_remote_scripts_are_fail_closed_and_g_drive_only() -> None:
    generated = k1by13.ROOT / "configs" / "remote" / "generated"
    run_script = generated / f"run_{k1by13.RUN_ID}.cmd"
    launch_script = generated / f"launch_{k1by13.RUN_ID}.cmd"
    monitor_script = generated / f"monitor_{k1by13.RUN_ID}.sh"
    run_text = run_script.read_text(encoding="utf-8")
    launch_text = launch_script.read_text(encoding="utf-8")
    monitor_text = monitor_script.read_text(encoding="utf-8")

    for text in (run_text, launch_text):
        assert "!" not in text
        assert "cmd.exe /k" not in text.lower()
        assert "G:\\lxy" in text
        assert "C:\\Users" not in text
    assert "cmd.exe /c" in launch_text
    assert "rmdir /s /q" not in launch_text
    assert "set SOURCE_ROOT=G:\\lxy\\bcnd-k1by13-src" in launch_text
    assert "set SOURCE_ROOT=G:\\lxy\\bcnd-k1by13-src" in run_text
    assert "set SOURCE_ROOT=%RUN_ROOT%\\source" not in launch_text
    assert "set SOURCE_ROOT=%RUN_ROOT%\\source" not in run_text
    assert "source_expected_commit.txt" in run_text
    assert "--output-root \"%OUTPUT_ROOT%\"" in run_text
    assert "--device cuda" in run_text
    assert "results.jsonl" in run_text
    assert "gate.json" in run_text
    assert "visual_qa_pending.marker" in run_text
    assert "G:/lxy/blockcipher-structure-adaptive-nd-runs" in monitor_text
    assert "--expected-rows 8" in monitor_text
    assert "RAW FALLBACK RETRIEVAL" in monitor_text


def _result_rows(
    values: dict[int, tuple[float, float, float, float]],
) -> list[dict[str, object]]:
    rows = []
    conditions = tuple(k1by13.CONDITIONS)
    for seed, aucs in values.items():
        for condition, auc in zip(conditions, aucs, strict=True):
            rows.append(
                {
                    "seed": seed,
                    "model": k1by13.CONDITIONS[condition],
                    "metrics": {"auc": auc},
                }
            )
    return rows
