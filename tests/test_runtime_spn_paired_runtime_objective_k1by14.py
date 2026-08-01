from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest
import torch

from blockcipher_nd.cli.plot_runtime_spn_paired_runtime_objective_k1by14 import (
    render_k1by14_svg,
)
from blockcipher_nd.cli.run_runtime_spn_paired_runtime_objective_k1by14 import (
    main,
    training_argv,
)
from blockcipher_nd.engine.matrix_runner import parse_args as parse_train_args
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_paired_runtime_objective_k1by14 as k1by14,
)


@pytest.fixture(scope="module")
def tasks() -> list[dict[str, object]]:
    return k1by14.read_tasks()


@pytest.fixture(scope="module")
def readiness(tasks: list[dict[str, object]]) -> dict[str, object]:
    return k1by14.build_readiness(tasks=tasks)


def test_k1by14_plan_sources_and_readiness_are_exact(
    tasks: list[dict[str, object]],
    readiness: dict[str, object],
) -> None:
    assert len(tasks) == k1by14.EXPECTED_TRAINING_ROWS
    assert k1by14.candidate_protocol_frozen(tasks)
    assert all(k1by14.source_binding_checks().values())
    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())


def test_k1by14_model_reuses_one_parameter_set_for_counterfactual(
    tasks: list[dict[str, object]],
) -> None:
    task = k1by14.task_map(tasks)[(2, "correct_oriented")]
    torch.manual_seed(2)
    model = k1by14.build_model_for_task(task)
    assert sum(parameter.numel() for parameter in model.parameters()) == 235780
    assert not any("counterfactual" in name for name, _ in model.named_parameters())
    assert (
        model.runtime_contrast_primary_sha256
        != model.runtime_contrast_counterfactual_sha256
    )

    features = torch.randint(0, 2, (4, 2048), dtype=torch.int64).float()
    labels = torch.tensor((0.0, 1.0, 0.0, 1.0))
    model.train()
    logits = model(features).squeeze(1)
    auxiliary = model.compute_auxiliary_loss(logits, labels, "mse")
    assert auxiliary is not None
    assert math.isfinite(float(auxiliary.detach()))
    assert float(auxiliary.detach()) > 0.0
    auxiliary.backward()
    assert sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    ) > 0.0


def test_k1by14_training_parser_reconstructs_frozen_tasks(tmp_path: Path) -> None:
    args = argparse.Namespace(
        plan=k1by14.PLAN_PATH,
        output_root=tmp_path / "run",
        device="cuda",
    )
    assert build_tasks(parse_train_args(training_argv(args))) == k1by14.read_tasks()


def test_k1by14_full_cpu_training_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        main(
            [
                "--plan",
                str(k1by14.PLAN_PATH),
                "--output-root",
                str(tmp_path / "cpu"),
                "--device",
                "cpu",
            ]
        )


def test_k1by14_readiness_cli_writes_zero_training_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "readiness"
    assert (
        main(
            [
                "--plan",
                str(k1by14.PLAN_PATH),
                "--output-root",
                str(output),
                "--device",
                "cpu",
                "--readiness-only",
            ]
        )
        == 0
    )
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (output / "validation.json").read_text(encoding="utf-8")
    )
    assert gate["decision"] == "innovation1_runtime_spn_k1by14_readiness_authorized"
    assert gate["training_performed"] is False
    assert gate["optimizer_steps"] == 0
    assert validation["status"] == "pass"
    assert not (output / "results.jsonl").exists()
    assert not (output / "checkpoints").exists()


def test_k1by14_cache_gate_requires_four_creates_and_four_reuses() -> None:
    rows = [
        {"event": event, "split": split}
        for event in ("cache_start", "cache_reuse")
        for _seed in k1by14.EXPECTED_SEEDS
        for split in ("train", "validation")
    ]
    assert k1by14.cache_protocol_frozen(rows)
    assert not k1by14.cache_protocol_frozen(rows[:-1])


def test_k1by14_adjudication_requires_orientation_and_heldout_controls(
    monkeypatch: pytest.MonkeyPatch,
    tasks: list[dict[str, object]],
    readiness: dict[str, object],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        k1by14,
        "training_protocol_frozen",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(k1by14, "cache_protocol_frozen", lambda _rows: True)
    anchors = {2: 0.683736801, 3: 0.665543556}
    monkeypatch.setattr(k1by14, "k1by3_anchor_auc", lambda: anchors)
    training_rows = _training_rows()
    evaluation_rows = _evaluation_rows()
    gate = k1by14.adjudicate(
        tasks=tasks,
        result_rows=training_rows,
        evaluation_rows=evaluation_rows,
        progress_rows=[],
        readiness=readiness,
        checkpoint_root=tmp_path,
    )
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_runtime_spn_k1by14_paired_preference_supported"
    )

    failed_rows = [dict(row) for row in evaluation_rows]
    for row in failed_rows:
        if (
            row["seed"] == 3
            and row["orientation"] == "swapped_orientation"
            and row["runtime_condition"] == "affine_runtime"
        ):
            row["metrics"] = {"auc": 0.680}
    gate = k1by14.adjudicate(
        tasks=tasks,
        result_rows=training_rows,
        evaluation_rows=failed_rows,
        progress_rows=[],
        readiness=readiness,
        checkpoint_root=tmp_path,
    )
    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation1_runtime_spn_k1by14_orientation_placebo_failed"
    )


def test_k1by14_plot_uses_clear_chinese_labels(tmp_path: Path) -> None:
    gate = {
        "status": "hold",
        "decision": "innovation1_runtime_spn_k1by14_orientation_placebo_failed",
        "seed_results": {
            str(seed): {
                "ordinary_k1by3_anchor_auc": 0.66,
                "auc_by_orientation_and_runtime": {
                    "correct_oriented": {
                        "correct_runtime": 0.68,
                        "affine_runtime": 0.67,
                        "heldout_shuffled": 0.665,
                    },
                    "swapped_orientation": {
                        "correct_runtime": 0.66,
                        "affine_runtime": 0.675,
                        "heldout_shuffled": 0.655,
                    },
                },
                "correct_oriented_margins": {
                    "anchor": 0.02,
                    "swapped_primary": 0.005,
                    "same_checkpoint_affine": 0.01,
                    "same_checkpoint_heldout_shuffled": 0.015,
                },
            }
            for seed in k1by14.EXPECTED_SEEDS
        },
    }
    output = tmp_path / "curves.svg"
    report = render_k1by14_svg(gate, output)
    svg = output.read_text(encoding="utf-8")
    assert report["status"] == "written"
    assert "PRESENT 七轮成对运行时结构学习" in svg
    assert "正确结构" in svg
    assert "未见打乱" in svg
    assert "结构差值门槛 +0.005" in svg
    assert "innovation1_runtime_spn_k1by14" not in svg


def _training_rows() -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "model": k1by14.ORIENTATION_MODELS[orientation],
            "history": [
                {
                    "train_auxiliary_loss": 0.002,
                    "train_runtime_loss_gap": 0.01,
                }
            ],
        }
        for seed in k1by14.EXPECTED_SEEDS
        for orientation in k1by14.ORIENTATIONS
    ]


def _evaluation_rows() -> list[dict[str, object]]:
    aucs = {
        2: {
            "correct_oriented": {
                "correct_runtime": 0.690,
                "affine_runtime": 0.680,
                "heldout_shuffled": 0.681,
            },
            "swapped_orientation": {
                "correct_runtime": 0.670,
                "affine_runtime": 0.680,
                "heldout_shuffled": 0.660,
            },
        },
        3: {
            "correct_oriented": {
                "correct_runtime": 0.675,
                "affine_runtime": 0.667,
                "heldout_shuffled": 0.668,
            },
            "swapped_orientation": {
                "correct_runtime": 0.660,
                "affine_runtime": 0.665,
                "heldout_shuffled": 0.655,
            },
        },
    }
    return [
        {
            "seed": seed,
            "orientation": orientation,
            "runtime_condition": runtime,
            "metrics": {"auc": aucs[seed][orientation][runtime]},
            "learned_parameter_fingerprint": f"seed{seed}-{orientation}",
            "source_parameter_fingerprint": f"seed{seed}-{orientation}",
            "training_performed": False,
            "optimizer_steps": 0,
        }
        for seed in k1by14.EXPECTED_SEEDS
        for orientation in k1by14.ORIENTATIONS
        for runtime in k1by14.RUNTIME_CONDITIONS
    ]
