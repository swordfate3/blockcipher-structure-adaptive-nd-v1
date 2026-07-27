from __future__ import annotations

import json
from pathlib import Path

import torch

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1 import render_ctspn_k1d_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1d import _validate_resume_root
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.canonical_relative_path import (
    PATH_FEATURE_SCHEMA,
    build_relative_path_topology,
)
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1c import (
    RUN_ID as K1C_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1d import (
    ANCHOR_PARAMETER_COUNT,
    CANDIDATE_MODEL,
    CONTROL_CONDITIONS,
    EXPECTED_PARAMETER_COUNT,
    K1C_DECISION,
    adjudicate_k1d,
    build_k1d_control,
    build_k1d_readiness,
    frozen_relative_path_stages,
    sorted_path_token_sha256,
    _control_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_native_endpoint_k1b_2048_seed0_seed1.csv"
)
TRAINING_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_relative_path_k1d_2048_seed0_seed1.csv"
)


def _tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        SOURCE_PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def _training_tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        TRAINING_PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def _k1c_gate() -> dict[str, object]:
    return {
        "run_id": K1C_RUN_ID,
        "status": "pass",
        "decision": K1C_DECISION,
        "protocol_checks": {"source_binding": True},
    }


def test_k1d_real_source_plan_passes_zero_training_readiness() -> None:
    manifests, gate = build_k1d_readiness(
        source_tasks=_tasks(),
        k1c_gate=_k1c_gate(),
    )

    assert len(manifests) == 4
    assert gate["status"] == "pass"
    assert gate["optimizer_step_authorized"] is True
    assert gate["training_rows"] == 0
    assert gate["optimizer_steps"] == 0
    assert all(gate["protocol_checks"].values())
    assert all(gate["evidence_checks"].values())
    assert {row["path_input_values"] for row in manifests} == {76}
    assert {row["trainable_parameter_count"] for row in manifests} == {
        EXPECTED_PARAMETER_COUNT
    }
    assert EXPECTED_PARAMETER_COUNT <= ANCHOR_PARAMETER_COUNT


def test_k1d_path_schema_has_no_absolute_cell_or_cipher_identity() -> None:
    assert len(PATH_FEATURE_SCHEMA) == 76
    assert not any(
        "cell_id" in name or "cipher" in name for name in PATH_FEATURE_SCHEMA
    )
    assert (
        sum(name.startswith("reachable_source_role") for name in PATH_FEATURE_SCHEMA)
        == 16
    )


def test_k1d_geometry_is_identical_across_64_and_128_bit_structures() -> None:
    tasks = {(str(row["cipher_key"]), int(row["seed"])): row for row in _tasks()}
    uknit = build_k1d_control(
        task=tasks[("uknit64", 0)],
        condition="correct_ordered",
        input_bits=512,
    )
    dialga = build_k1d_control(
        task=tasks[("dialga128", 0)],
        condition="correct_ordered",
        input_bits=1024,
    )

    assert (
        model_metadata(uknit)["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
    )
    assert (
        model_metadata(dialga)["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
    )
    assert [
        (name, tuple(value.shape)) for name, value in uknit.state_dict().items()
    ] == [(name, tuple(value.shape)) for name, value in dialga.state_dict().items()]
    dialga.load_state_dict(uknit.state_dict(), strict=True)
    assert uknit(torch.zeros(2, 512)).shape == (2, 1)
    assert dialga(torch.zeros(2, 1024)).shape == (2, 1)


def test_k1d_cell_relabel_only_permutes_path_token_set_and_preserves_logits() -> None:
    task = next(
        row for row in _tasks() if row["cipher_key"] == "uknit64" and row["seed"] == 0
    )
    model = build_k1d_control(
        task=task,
        condition="correct_ordered",
        input_bits=512,
    )
    generator = torch.Generator().manual_seed(17)
    features = torch.randint(0, 2, (3, 512), generator=generator).float()
    runtime = features.reshape(3, 4, 2, 64).flip(-1)
    structure = model.runtime_structure
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    relabeled_runtime = torch.empty_like(runtime)
    relabeled_runtime[..., bit_permutation] = runtime

    with torch.inference_mode():
        original_views = model.backbone.relative_path_views(
            runtime, structure, relation_mode="true"
        )
        relabeled_views = model.backbone.relative_path_views(
            relabeled_runtime, relabeled, relation_mode="true"
        )
        original_logits = model.backbone(runtime, structure, relation_mode="true")
        relabeled_logits = model.backbone(
            relabeled_runtime, relabeled, relation_mode="true"
        )

    assert sorted_path_token_sha256(original_views) == sorted_path_token_sha256(
        relabeled_views
    )
    torch.testing.assert_close(original_logits, relabeled_logits, atol=1e-6, rtol=0)


def test_k1d_all_wrong_controls_change_paths_and_random_weight_outputs() -> None:
    task = next(
        row for row in _tasks() if row["cipher_key"] == "uknit64" and row["seed"] == 0
    )
    correct = build_k1d_control(
        task=task,
        condition="correct_ordered",
        input_bits=512,
    )
    state = correct.state_dict()
    features = torch.randint(
        0, 2, (4, 512), generator=torch.Generator().manual_seed(20260728)
    ).float()
    with torch.inference_mode():
        correct_views, correct_pooled, correct_logits = frozen_relative_path_stages(
            correct, features
        )
    correct_sha = sorted_path_token_sha256(correct_views)

    for condition in CONTROL_CONDITIONS[1:]:
        control = build_k1d_control(
            task=task,
            condition=condition,
            input_bits=512,
        )
        control.load_state_dict(state, strict=True)
        with torch.inference_mode():
            views, pooled, logits = frozen_relative_path_stages(control, features)
        assert sorted_path_token_sha256(views) != correct_sha
        assert float((pooled - correct_pooled).abs().max()) > 1e-6
        assert float((logits - correct_logits).abs().max()) > 1e-7


def test_k1d_topology_has_one_connected_two_transition_path_layer() -> None:
    task = next(
        row for row in _tasks() if row["cipher_key"] == "dialga128" and row["seed"] == 0
    )
    model = build_k1d_control(
        task=task,
        condition="correct_ordered",
        input_bits=1024,
    )
    topology = build_relative_path_topology(
        model.runtime_structure,
        relation_mode="true",
    )

    assert model.relative_path_compositions == 1
    assert topology.path_count > 0
    assert topology.reachability.shape == (topology.path_count, 4, 4)
    assert bool(topology.reachability.any(dim=(1, 2)).all())


def test_k1d_control_metadata_uses_adapter_class_without_missing_key() -> None:
    task = next(
        row
        for row in _training_tasks()
        if row["cipher_key"] == "uknit64" and row["seed"] == 0
    )
    model = build_k1d_control(
        task=task,
        condition="correct_ordered",
        input_bits=512,
    )

    metadata = _control_metadata(model, "correct_ordered")

    assert metadata["model_class"] == "FixedRelativePathSpnProtocolAdapter"
    assert metadata["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT


def test_k1d_resume_gate_reuses_four_completed_checkpoints(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        checkpoint = tmp_path / f"checkpoint-{index}.pt"
        checkpoint.write_bytes(b"completed")
        rows.append(
            {
                "model": CANDIDATE_MODEL,
                "training": {"checkpoint_output": str(checkpoint)},
            }
        )
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            {
                "run_id": (
                    "i1_uknit_family_ctspn_relative_path_k1d_2048_seed0_seed1_20260728"
                ),
                "status": "pass",
                "execution_authorized": True,
                "plan_sha256": file_sha256(TRAINING_PLAN),
                "source_plan_sha256": file_sha256(SOURCE_PLAN),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    _validate_resume_root(
        output_root=tmp_path,
        plan=TRAINING_PLAN,
        source_plan=SOURCE_PLAN,
    )


def test_k1d_readiness_fails_closed_without_k1c_evidence() -> None:
    _, gate = build_k1d_readiness(source_tasks=_tasks(), k1c_gate={})

    assert gate["status"] == "fail"
    assert gate["optimizer_step_authorized"] is False
    assert gate["protocol_checks"]["k1c_topology_overfit_confirmed"] is False


def test_k1d_training_gate_requires_each_uknit_seed_and_control() -> None:
    _, readiness = build_k1d_readiness(
        source_tasks=_tasks(),
        k1c_gate=_k1c_gate(),
    )
    training_rows, control_rows = _passing_training_evidence()
    k1b_gate = _k1b_gate()
    passed = adjudicate_k1d(
        tasks=_training_tasks(),
        training_rows=training_rows,
        control_rows=control_rows,
        readiness_gate=readiness,
        k1b_gate=k1b_gate,
        k1c_gate=_k1c_gate(),
    )

    assert passed["status"] == "pass"
    assert all(passed["protocol_checks"].values())
    assert all(passed["research_checks"].values())

    failed = [dict(row) for row in control_rows]
    target = next(
        row
        for row in failed
        if row["cipher_key"] == "uknit64"
        and row["seed"] == 1
        and row["condition"] == "rotated"
    )
    target["auc"] = 0.56
    held = adjudicate_k1d(
        tasks=_training_tasks(),
        training_rows=training_rows,
        control_rows=failed,
        readiness_gate=readiness,
        k1b_gate=k1b_gate,
        k1c_gate=_k1c_gate(),
    )

    assert held["status"] == "hold"
    assert held["research_checks"]["uknit64_seed1_beats_rotated"] is False


def test_k1d_plot_uses_plain_chinese_path_and_control_labels(tmp_path: Path) -> None:
    _, readiness = build_k1d_readiness(
        source_tasks=_tasks(),
        k1c_gate=_k1c_gate(),
    )
    training_rows, control_rows = _passing_training_evidence()
    gate = adjudicate_k1d(
        tasks=_training_tasks(),
        training_rows=training_rows,
        control_rows=control_rows,
        readiness_gate=readiness,
        k1b_gate=_k1b_gate(),
        k1c_gate=_k1c_gate(),
    )
    output = tmp_path / "curves.svg"

    render_ctspn_k1d_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "相对跨层扩散路径能否改善 uKNIT 类 SPN 区分" in svg
    assert "K1-D 相对跨层路径" in svg
    assert "同预算旧模型" in svg
    assert "错误拓扑" in svg


def _k1b_gate() -> dict[str, object]:
    return {
        "status": "hold",
        "decision": (
            "innovation1_uknit_family_ctspn_k1b_native_endpoint_not_supported"
        ),
        "protocol_checks": {"frozen": True},
        "seed_results": {
            "uknit64": {
                "0": {"prior_anchor_auc": 0.526, "candidate_auc": 0.510},
                "1": {"prior_anchor_auc": 0.528, "candidate_auc": 0.508},
            },
            "dialga128": {
                "0": {"candidate_auc": 0.960},
                "1": {"candidate_auc": 0.961},
            },
        },
    }


def _passing_training_evidence() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    training_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for task in _training_tasks():
        cipher = str(task["cipher_key"])
        seed = int(task["seed"])
        candidate_auc = 0.540 if cipher == "uknit64" else 0.960
        dataset_sha = f"dataset-{cipher}-{seed}"
        state_sha = f"state-{cipher}-{seed}"
        training_rows.append(
            {
                "cipher_key": cipher,
                "seed": seed,
                "model": CANDIDATE_MODEL,
                "samples_per_class": 2048,
                "pairs_per_sample": 4,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                "training": {
                    "batch_size": 64,
                    "epochs": 10,
                    "checkpoint_metric": "val_auc",
                    "selected_checkpoint": "best",
                    "train_rows": 4096,
                    "validation_rows": 2048,
                },
            }
        )
        for condition in CONTROL_CONDITIONS:
            auc = (
                candidate_auc
                if condition == "correct_ordered"
                else candidate_auc - 0.02
            )
            control_rows.append(
                {
                    "cipher_key": cipher,
                    "seed": seed,
                    "condition": condition,
                    "auc": auc,
                    "source_auc": candidate_auc,
                    "correct_minus_source_auc": 0.0,
                    "correct_minus_condition_auc": (
                        0.0 if condition == "correct_ordered" else 0.02
                    ),
                    "max_abs_probability_delta_from_correct": (
                        0.0 if condition == "correct_ordered" else 0.1
                    ),
                    "mean_abs_probability_delta_from_correct": (
                        0.0 if condition == "correct_ordered" else 0.01
                    ),
                    "dataset_sha256": dataset_sha,
                    "prior_k1b_dataset_sha256": dataset_sha,
                    "checkpoint_selected": "best",
                    "checkpoint_metric": "val_auc",
                    "state_dict_sha256": state_sha,
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )
    return training_rows, control_rows
