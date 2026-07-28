from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from blockcipher_nd.cli.plot_uknit_family_midori64_k1al import render_k1al_svg
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    INPUT_DIFFERENCE,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_sbox_transition_k1ak import (
    RUN_ID as K1AK_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_transition_causal_k1al import (
    AUDIT_CONDITIONS,
    EXPECTED_CORRECT_CHECKPOINTS,
    EXPECTED_SOURCE_DIGESTS,
    SOURCE_DECISION,
    TransitionBranchOffWrapper,
    adjudicate,
    source_binding_checks,
)


OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/midori64.json",
    "runtime_round_start": 0,
    "runtime_rounds": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "residual_gate_initial_effective": 0.05,
    "transition_gate_initial_effective": 0.05,
    "transition_value_dim": 20,
    "virtual_projection_slots": 16,
    "topology_corruption_seed": 20260729,
}


def build_true_model() -> torch.nn.Module:
    return build_model(
        "runtime_spn_ct_k1ak_sbox_transition_true",
        input_bits=512,
        hidden_bits=32,
        pair_bits=128,
        structure="SPN",
        model_options=OPTIONS,
    )


def test_k1al_branch_off_wrapper_changes_forward_without_state_mutation() -> None:
    torch.manual_seed(29)
    source = build_true_model()
    branch_off_base = build_true_model()
    branch_off_base.load_state_dict(source.state_dict(), strict=True)
    before = tensor_mapping_sha256(branch_off_base.state_dict())
    fixture = torch.randint(0, 2, (9, 512), dtype=torch.float32)

    source.eval()
    wrapper = TransitionBranchOffWrapper(branch_off_base).eval()
    with torch.no_grad():
        exact = source(fixture)
        ablated = wrapper(fixture)

    assert not torch.equal(exact, ablated)
    assert tensor_mapping_sha256(branch_off_base.state_dict()) == before


def test_k1al_source_binding_requires_exact_k1ak_evidence() -> None:
    checks = source_binding_checks(
        gate=source_gate(),
        validation={"run_id": K1AK_RUN_ID, "status": "pass", "errors": []},
        checkpoint_manifest=checkpoint_manifest(),
        source_controls=source_controls(),
        dataset_manifest=dataset_manifest(),
        source_digests=EXPECTED_SOURCE_DIGESTS,
    )
    assert all(checks.values())

    changed = checkpoint_manifest()
    changed["entries"][0]["sha256"] = "f" * 64
    failed = source_binding_checks(
        gate=source_gate(),
        validation={"run_id": K1AK_RUN_ID, "status": "pass", "errors": []},
        checkpoint_manifest=changed,
        source_controls=source_controls(),
        dataset_manifest=dataset_manifest(),
        source_digests=EXPECTED_SOURCE_DIGESTS,
    )
    assert failed["two_exact_correct_best_checkpoints"] is False


def test_k1al_gate_passes_only_when_sbox_and_branch_are_causal() -> None:
    gate = synthetic_gate()

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("transition_and_sbox_causal_use_supported")
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert gate["remote_scale"] == "no"


def test_k1al_gate_separates_nonidentifying_sbox_from_causal_branch() -> None:
    rows = result_rows()
    for row in rows:
        if row["condition"] == "wrong_sbox_same_checkpoint":
            row["auc"] = 0.648
            row["correct_minus_condition_auc"] = 0.002
    gate = synthetic_gate(rows)

    assert gate["status"] == "hold"
    assert gate["decision"].endswith(
        "transition_causal_sbox_identification_failed"
    )
    assert all(
        passed
        for name, passed in gate["research_checks"].items()
        if "beats_transition_off" in name
    )


def test_k1al_gate_rejects_noncausal_transition_branch() -> None:
    rows = result_rows()
    for row in rows:
        if row["condition"] == "transition_branch_off_same_checkpoint":
            row["auc"] = 0.648
            row["correct_minus_condition_auc"] = 0.002
    gate = synthetic_gate(rows)

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("transition_branch_causal_use_failed")


def test_k1al_gate_rejects_state_or_training_drift() -> None:
    rows = deepcopy(result_rows())
    rows[1]["state_dict_sha256"] = "changed"
    rows[1]["optimizer_steps"] = 1
    gate = synthetic_gate(rows)

    assert gate["status"] == "invalid"
    assert (
        gate["protocol_checks"][
            "same_checkpoint_state_and_dataset_within_seed_split"
        ]
        is False
    )
    assert gate["protocol_checks"]["inference_only"] is False


def test_k1al_plot_explains_same_checkpoint_branch_result(tmp_path: Path) -> None:
    output = tmp_path / "curves.svg"
    report = render_k1al_svg(synthetic_gate(), output)
    svg = output.read_text(encoding="utf-8")

    assert report["same_checkpoint"] is True
    assert report["training_performed"] is False
    assert "正确 S盒和新转移分支" in svg
    assert "全程零训练" in svg
    assert "替代捷径" in svg


def synthetic_gate(
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return adjudicate(
        result_rows() if rows is None else rows,
        source_checks={"source": True},
        control_checks={"controls": True},
    )


def source_gate() -> dict[str, object]:
    return {
        "run_id": K1AK_RUN_ID,
        "status": "hold",
        "decision": SOURCE_DECISION,
        "remote_scale": "no",
        "failed_protocol_checks": [],
    }


def checkpoint_manifest() -> dict[str, object]:
    return {
        "run_id": K1AK_RUN_ID,
        "status": "pass",
        "entries": [
            {
                "seed": seed,
                "condition": "correct_structure",
                "model": "runtime_spn_ct_k1ak_sbox_transition_true",
                "selected_checkpoint": "best",
                "sha256": EXPECTED_CORRECT_CHECKPOINTS[seed],
            }
            for seed in EXPECTED_SEEDS
        ],
    }


def source_controls() -> list[dict[str, object]]:
    return [
        {
            "run_id": K1AK_RUN_ID,
            "seed": seed,
            "split": split,
            "condition": "correct_structure",
            "strict_state_dict_load": True,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
    ]


def dataset_manifest() -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "split": split,
            "cell": 8,
            "input_difference": INPUT_DIFFERENCE,
            "rounds": 4,
            "cache_payloads_present": True,
        }
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
    ]


def result_rows() -> list[dict[str, object]]:
    aucs = {
        "correct_runtime": 0.650,
        "wrong_sbox_same_checkpoint": 0.630,
        "transition_branch_off_same_checkpoint": 0.610,
    }
    rows: list[dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        checkpoint = EXPECTED_CORRECT_CHECKPOINTS[seed]
        state = f"state-{seed}"
        for split in EXPECTED_SPLITS:
            dataset = f"dataset-{seed}-{split}"
            row_count = 4096 if split == "train_seen" else 2048
            for condition in AUDIT_CONDITIONS:
                auc = aucs[condition]
                exact = condition == "correct_runtime"
                branch_off = condition == "transition_branch_off_same_checkpoint"
                wrong_sbox = condition == "wrong_sbox_same_checkpoint"
                rows.append(
                    {
                        "run_id": "k1al",
                        "source_run_id": K1AK_RUN_ID,
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "cipher_key": "midori64",
                        "rounds": 4,
                        "auc": auc,
                        "source_correct_auc": 0.650,
                        "correct_minus_condition_auc": 0.650 - auc,
                        "max_abs_probability_delta_from_correct": (
                            0.0 if exact else 0.1
                        ),
                        "mean_abs_probability_delta_from_correct": (
                            0.0 if exact else 0.05
                        ),
                        "checkpoint_sha256": checkpoint,
                        "checkpoint_selected": "best",
                        "checkpoint_reported_seed": seed,
                        "state_dict_sha256": state,
                        "source_state_dict_sha256": state,
                        "branch_off_state_preserved": True,
                        "dataset_sha256": dataset,
                        "source_dataset_sha256": dataset,
                        "source_checkpoint_sha256": checkpoint,
                        "source_decision": SOURCE_DECISION,
                        **{
                            f"source_{name}_sha256": digest
                            for name, digest in EXPECTED_SOURCE_DIGESTS.items()
                        },
                        "composition_sha256": (
                            "wrong-composition" if wrong_sbox else "correct-composition"
                        ),
                        "sbox_transition_semantics_sha256": (
                            "wrong-sbox" if wrong_sbox else "correct-sbox"
                        ),
                        "transition_branch_enabled": not branch_off,
                        "rows": row_count,
                        "input_bits": 512,
                        "pairs_per_sample": 4,
                        "input_difference": INPUT_DIFFERENCE,
                        "negative_mode": "encrypted_random_plaintexts",
                        "sample_structure": "independent_pairs",
                        "parameter_count": 219_320,
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "epochs": 0,
                    }
                )
    return rows
