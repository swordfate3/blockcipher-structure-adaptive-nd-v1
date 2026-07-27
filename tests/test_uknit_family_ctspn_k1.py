from __future__ import annotations

from pathlib import Path

import pytest
import torch

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1 import render_ctspn_k1_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1 import main as run_k1
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    CONTROL_CONDITIONS,
    RUN_ID as TRAINING_RUN_ID,
    _build_control_model,
    _control_metadata,
    adjudicate_ctspn_k1,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1_readiness import (
    K0_DECISION,
    K0_RUN_ID,
    RUN_ID,
    build_ctspn_k1_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_linear_schedule_k1_2048_seed0_seed1.csv"
)


def _tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


@pytest.fixture(scope="module")
def readiness() -> tuple[list[dict[str, object]], dict[str, object]]:
    return build_ctspn_k1_readiness(
        run_id=RUN_ID,
        tasks=_tasks(),
        k0_gate={
            "run_id": K0_RUN_ID,
            "status": "pass",
            "decision": K0_DECISION,
        },
        k0_validation={"run_id": K0_RUN_ID, "status": "pass"},
        present_gate={
            "run_id": "i1_rtg3b_present80_one_to_one_formal_1000000_seed1_retry1_20260727",
            "cipher": "PRESENT-80",
            "seed": 1,
            "phase": "rtg3b",
            "samples_per_class": 1_000_000,
            "validation_samples_per_class": 500_000,
            "status": "pass",
            "decision": "innovation1_runtime_spn_present_formal_seed1_supported",
            "protocol_checks": {"frozen_protocol": True},
            "research_checks": {"supported": True},
        },
    )


def test_ctspn_k1_real_plan_passes_all_zero_training_readiness_checks(
    readiness: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    manifests, gate = readiness

    assert len(manifests) == 8
    assert gate["status"] == "pass"
    assert gate["implementation_ready"] is True
    assert gate["optimizer_step_authorized"] is True
    assert gate["training_rows"] == 0
    assert gate["optimizer_steps"] == 0
    assert all(gate["protocol_checks"].values())
    assert all(gate["evidence_checks"].values())
    assert {
        row["trainable_parameter_count"]
        for row in manifests
        if row["model"] == "runtime_spn_ct_k1_canonical_true"
    } == {438702}


def test_ctspn_k1_readiness_proves_both_widths_and_controls(
    readiness: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    _, gate = readiness
    evidence = gate["structural_evidence"]

    assert evidence["cross_width_strict_state_dict_load"] is True
    assert evidence["no_cipher_identity_tensor"] is True
    for cipher in ("uknit64", "dialga128"):
        row = evidence["per_cipher"][cipher]
        assert row["canonical_unit_inverse_exact"] is True
        assert row["canonical_edge_relation_exact"] is True
        assert row["cell_relabel_max_logit_error"] <= 1e-6
        assert row["controls"]["fingerprints_deterministic_and_distinct"] is True
        assert row["controls"]["strict_state_dict_load"] is True
        assert row["controls"]["factors_survive_checkpoint_load"] is True
        assert row["controls"]["control_logits_noncollapsed"] is True


def test_ctspn_model_factory_preserves_geometry_across_block_widths() -> None:
    common = {
        "runtime_rounds": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "temporal_hidden_dim": 76,
        "dropout": 0.0,
    }
    uknit = build_model(
        "runtime_spn_ct_k1_canonical_true",
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options={
            **common,
            "runtime_structure_path": "configs/runtime/spn/uknit64.json",
            "runtime_round_start": 3,
        },
    )
    dialga = build_model(
        "runtime_spn_ct_k1_canonical_true",
        input_bits=1024,
        hidden_bits=64,
        pair_bits=256,
        structure="SPN",
        model_options={
            **common,
            "runtime_structure_path": "configs/runtime/spn/dialga128.json",
            "runtime_round_start": 2,
        },
    )

    assert model_metadata(uknit)["trainable_parameter_count"] == 438702
    assert model_metadata(dialga)["trainable_parameter_count"] == 438702
    assert [
        (name, tuple(value.shape)) for name, value in uknit.state_dict().items()
    ] == [(name, tuple(value.shape)) for name, value in dialga.state_dict().items()]
    dialga_factor = dialga.canonical_factor_manifest_sha256
    dialga.load_state_dict(uknit.state_dict(), strict=True)
    assert dialga.canonical_factor_manifest_sha256 == dialga_factor
    assert uknit(torch.zeros(2, 512)).shape == (2, 1)
    assert dialga(torch.zeros(2, 1024)).shape == (2, 1)


def test_ctspn_readiness_fails_closed_without_k0_evidence() -> None:
    _, gate = build_ctspn_k1_readiness(
        run_id="invalid",
        tasks=[],
        k0_gate={},
        k0_validation={},
    )

    assert gate["status"] == "fail"
    assert gate["implementation_ready"] is False
    assert gate["optimizer_step_authorized"] is False
    assert gate["protocol_checks"]["k0_gate_and_validation_pass"] is False
    assert gate["protocol_checks"]["eight_row_frozen_panel"] is False


def test_ctspn_readiness_rejects_present_launch_authorization_as_result() -> None:
    _, gate = build_ctspn_k1_readiness(
        run_id="launch-only",
        tasks=_tasks(),
        k0_gate={
            "run_id": K0_RUN_ID,
            "status": "pass",
            "decision": K0_DECISION,
        },
        k0_validation={"run_id": K0_RUN_ID, "status": "pass"},
        present_gate={
            "run_id": (
                "i1_rtg3b_present80_one_to_one_formal_1000000_"
                "seed1_retry1_launch_gate_20260727"
            ),
            "status": "pass",
            "decision": "innovation1_rtg3b_present_seed1_remote_launch_authorized",
            "launch_authorized": True,
        },
    )

    assert gate["implementation_ready"] is True
    assert gate["present_formal_seed1_adjudicated"] is False
    assert gate["optimizer_step_authorized"] is False


def test_runtime_structure_rotation_preserves_transitions_and_changes_order() -> None:
    structure = load_runtime_spn_descriptor(
        "configs/runtime/spn/uknit64.json",
        rounds=2,
        round_start=3,
    ).structure

    rotated = structure.rotate_transitions()

    assert rotated.transition_sha256s() == tuple(
        reversed(structure.transition_sha256s())
    )
    assert rotated.window_sha256() != structure.window_sha256()
    assert rotated.rotate_transitions().window_sha256() == structure.window_sha256()


def test_ctspn_k1_adjudication_requires_every_cipher_seed_and_control() -> None:
    tasks = _tasks()
    training_rows, control_rows = _passing_k1_rows(tasks)

    passed = adjudicate_ctspn_k1(
        run_id=TRAINING_RUN_ID,
        task_rows=tasks,
        training_rows=training_rows,
        control_rows=control_rows,
    )

    assert passed["status"] == "pass"
    assert all(passed["protocol_checks"].values())
    assert all(passed["research_checks"].values())

    failed_control = [dict(row) for row in control_rows]
    target = next(
        row
        for row in failed_control
        if row["cipher_key"] == "uknit64"
        and row["seed"] == 1
        and row["source_role"] == "candidate"
        and row["condition"] == "rotated"
    )
    target["auc"] = 0.61
    target["correct_minus_condition_auc"] = -0.01
    held = adjudicate_ctspn_k1(
        run_id=TRAINING_RUN_ID,
        task_rows=tasks,
        training_rows=training_rows,
        control_rows=failed_control,
    )

    assert held["status"] == "hold"
    assert held["protocol_checks"]["forty_frozen_control_rows_complete"] is True
    assert held["research_checks"]["uknit64_seed1_candidate_beats_rotated"] is False
    assert "macro" not in held["decision"]


def test_ctspn_k1_five_controls_reuse_learned_state_and_change_intervention() -> None:
    tasks = _tasks()
    for source_role, model_key in (
        ("anchor", "runtime_spn_e4_equivariant_true"),
        ("candidate", "runtime_spn_ct_k1_canonical_true"),
    ):
        task = next(
            row
            for row in tasks
            if row["cipher_key"] == "uknit64"
            and row["seed"] == 0
            and row["model_key"] == model_key
        )
        correct = _build_control_model(
            task=task,
            source_role=source_role,
            condition="correct_ordered",
            input_bits=512,
        )
        state = correct.state_dict()
        state_sha256 = tensor_mapping_sha256(state)
        fingerprints = set()
        for condition in CONTROL_CONDITIONS:
            control = _build_control_model(
                task=task,
                source_role=source_role,
                condition=condition,
                input_bits=512,
            )
            control.load_state_dict(state, strict=True)
            assert tensor_mapping_sha256(control.state_dict()) == state_sha256
            fingerprints.add(
                _control_metadata(control, condition)[
                    "control_fingerprint_sha256"
                ]
            )
        assert len(fingerprints) == len(CONTROL_CONDITIONS)


def test_ctspn_k1_runner_stops_before_output_when_optimizer_not_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "run"

    monkeypatch.setattr(
        "blockcipher_nd.cli.run_uknit_family_ctspn_k1.build_ctspn_k1_readiness",
        lambda **_kwargs: (
            [],
            {
                "optimizer_step_authorized": False,
                "implementation_ready": True,
                "status": "pass",
                "decision": (
                    "innovation1_uknit_family_ctspn_k1_readiness_passed_waiting_present"
                ),
            },
        ),
    )

    exit_code = run_k1(
        [
            "--plan",
            str(PLAN),
            "--k0-gate",
            str(source),
            "--k0-validation",
            str(source),
            "--present-gate",
            str(source),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 4
    assert not output_root.exists()


def test_ctspn_k1_chinese_plot_has_complete_labels(tmp_path: Path) -> None:
    tasks = _tasks()
    training_rows, control_rows = _passing_k1_rows(tasks)
    gate = adjudicate_ctspn_k1(
        run_id=TRAINING_RUN_ID,
        task_rows=tasks,
        training_rows=training_rows,
        control_rows=control_rows,
    )
    gate["seed_results"]["dialga128"]["0"][
        "candidate_minus_no_topology"
    ] = 0.44
    gate["seed_results"]["dialga128"]["1"][
        "candidate_minus_no_topology"
    ] = 0.43
    output = tmp_path / "curves.svg"

    render_ctspn_k1_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "规范化线性层顺序是否真正帮助 uKNIT 类 SPN 区分" in svg
    assert "uKNIT-BC 五轮" in svg
    assert "Dialga-128 四轮" in svg
    assert "候选 - 无拓扑" in svg
    assert "无拓扑优势（单独尺度）" in svg


def _passing_k1_rows(
    tasks: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    training_rows = []
    control_rows = []
    for task in tasks:
        cipher = str(task["cipher_key"])
        seed = int(task["seed"])
        source_role = (
            "candidate"
            if task["model_key"] == "runtime_spn_ct_k1_canonical_true"
            else "anchor"
        )
        candidate_auc = 0.60 if cipher == "uknit64" else 0.70
        source_auc = candidate_auc if source_role == "candidate" else candidate_auc - 0.03
        training_rows.append(
            {
                "cipher": "uKNIT-BC" if cipher == "uknit64" else "Dialga-128",
                "cipher_key": cipher,
                "model": task["model_key"],
                "rounds": task["rounds"],
                "seed": seed,
                "samples_per_class": 2048,
                "pairs_per_sample": 4,
                "input_difference": 0x40,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "train_key": task["train_key"],
                "validation_key": task["validation_key"],
                "trainable_parameter_count": 438702
                if source_role == "candidate"
                else 442466,
                "metrics": {"auc": source_auc},
                "training": {
                    "batch_size": 64,
                    "epochs": 10,
                    "loss": "mse",
                    "optimizer": "adam",
                    "learning_rate": 0.0001,
                    "weight_decay": 0.00001,
                    "checkpoint_metric": "val_auc",
                    "restore_best_checkpoint": True,
                    "selected_checkpoint": "best",
                    "train_rows": 4096,
                    "validation_rows": 2048,
                    "train_dataset_storage": "disk",
                    "validation_dataset_storage": "disk",
                    "model_options": task["model_options"],
                },
            }
        )
        condition_aucs = {
            "correct_ordered": source_auc,
            "repeat_last": source_auc - 0.02,
            "rotated": source_auc - 0.02,
            "corrupted": source_auc - 0.02,
            "no_topology": source_auc - 0.02,
        }
        for index, condition in enumerate(CONTROL_CONDITIONS):
            window_control = "full"
            if condition == "repeat_last":
                window_control = "repeat_last"
            elif condition == "rotated" and source_role == "anchor":
                window_control = "rotated"
            control_rows.append(
                {
                    "cipher_key": cipher,
                    "seed": seed,
                    "source_role": source_role,
                    "condition": condition,
                    "auc": condition_aucs[condition],
                    "source_auc": source_auc,
                    "correct_minus_source_auc": 0.0,
                    "correct_minus_condition_auc": (
                        0.0
                        if condition == "correct_ordered"
                        else condition_aucs["correct_ordered"]
                        - condition_aucs[condition]
                    ),
                    "max_abs_probability_delta_from_correct": 0.0
                    if condition == "correct_ordered"
                    else 0.1,
                    "mean_abs_probability_delta_from_correct": 0.0
                    if condition == "correct_ordered"
                    else 0.01,
                    "mean_probability": 0.5,
                    "checkpoint_selected": "best",
                    "checkpoint_metric": "val_auc",
                    "checkpoint_sha256": f"checkpoint-{cipher}-{seed}-{source_role}",
                    "state_dict_sha256": f"state-{cipher}-{seed}-{source_role}",
                    "dataset_sha256": f"dataset-{cipher}-{seed}",
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                    "runtime_structure_window_control": window_control,
                    "runtime_structure_homogeneous": condition == "repeat_last",
                    "relation_mode": "independent"
                    if condition == "no_topology"
                    else "true",
                    "control_fingerprint_sha256": (
                        f"fingerprint-{cipher}-{seed}-{source_role}-{index}"
                    ),
                    "canonical_schedule_control": "rotated"
                    if source_role == "candidate" and condition == "rotated"
                    else "ordered",
                }
            )
    return training_rows, control_rows
