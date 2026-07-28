from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_midori64_k1aj import render_k1aj_svg
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    CONTROL_CONDITIONS,
    EXPECTED_SEEDS,
    EXPECTED_SPLITS,
    INPUT_DIFFERENCE,
    RUN_ID as K1AI_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_same_checkpoint_k1aj import (
    EXPECTED_SOURCE_DIGESTS,
    SOURCE_DECISION,
    adjudicate,
    source_binding_checks,
)


def test_k1aj_source_binding_requires_exact_k1ai_hold_and_two_checkpoints() -> None:
    checks = source_binding_checks(
        gate={
            "run_id": K1AI_RUN_ID,
            "status": "hold",
            "decision": SOURCE_DECISION,
            "remote_scale": "no",
            "failed_protocol_checks": [],
        },
        validation={"run_id": K1AI_RUN_ID, "status": "pass", "errors": []},
        checkpoint_manifest=checkpoint_manifest(),
        source_controls=source_controls(),
        dataset_manifest=dataset_manifest(),
        source_digests=EXPECTED_SOURCE_DIGESTS,
    )
    assert all(checks.values())

    failed_manifest = checkpoint_manifest()
    failed_manifest["entries"] = failed_manifest["entries"][:-1]
    failed = source_binding_checks(
        gate={
            "run_id": K1AI_RUN_ID,
            "status": "hold",
            "decision": SOURCE_DECISION,
            "remote_scale": "no",
            "failed_protocol_checks": [],
        },
        validation={"run_id": K1AI_RUN_ID, "status": "pass", "errors": []},
        checkpoint_manifest=failed_manifest,
        source_controls=source_controls(),
        dataset_manifest=dataset_manifest(),
        source_digests=EXPECTED_SOURCE_DIGESTS,
    )
    assert failed["two_correct_best_checkpoint_entries"] is False


def test_k1aj_gate_passes_only_when_same_checkpoint_uses_all_semantics() -> None:
    gate = adjudicate(
        result_rows(),
        source_checks={"source": True},
        control_checks={"controls": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("same_checkpoint_semantic_use_supported")
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert gate["remote_scale"] == "no"


def test_k1aj_gate_isolates_sbox_failure_when_diffusion_still_passes() -> None:
    rows = result_rows()
    for row in rows:
        if row["condition"] == "wrong_sbox":
            row["auc"] = 0.651
            row["correct_minus_condition_auc"] = -0.001
    gate = adjudicate(
        rows,
        source_checks={"source": True},
        control_checks={"controls": True},
    )

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("diffusion_causal_sbox_discrimination_failed")
    assert all(
        passed
        for name, passed in gate["research_checks"].items()
        if "beats_corrupted_linear" in name
    )
    assert not all(
        passed
        for name, passed in gate["research_checks"].items()
        if "beats_wrong_sbox" in name
    )


def test_k1aj_gate_rejects_checkpoint_or_training_drift() -> None:
    rows = result_rows()
    mutated = deepcopy(rows)
    mutated[1]["checkpoint_sha256"] = "f" * 64
    mutated[1]["optimizer_steps"] = 1
    gate = adjudicate(
        mutated,
        source_checks={"source": True},
        control_checks={"controls": True},
    )

    assert gate["status"] == "invalid"
    assert (
        gate["protocol_checks"]["same_checkpoint_state_and_dataset_within_seed_split"]
        is False
    )
    assert gate["protocol_checks"]["inference_only"] is False


def test_k1aj_plot_explains_same_checkpoint_causal_result(tmp_path: Path) -> None:
    rows = result_rows()
    for row in rows:
        if row["condition"] == "wrong_sbox" and row["seed"] == 6:
            row["auc"] = 0.648
            row["correct_minus_condition_auc"] = 0.002
    gate = adjudicate(
        rows,
        source_checks={"source": True},
        control_checks={"controls": True},
    )
    output = tmp_path / "curves.svg"
    report = render_k1aj_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["same_checkpoint"] is True
    assert report["training_performed"] is False
    assert "同一组 Midori64 网络权重" in svg
    assert "全程零训练" in svg
    assert "S盒会改变预测" in svg


def checkpoint_manifest() -> dict[str, object]:
    return {
        "run_id": K1AI_RUN_ID,
        "status": "pass",
        "entries": [
            {
                "seed": seed,
                "condition": "correct_structure",
                "model": "runtime_spn_ct_k1aa_virtual_slot_histogram_true",
                "selected_checkpoint": "best",
                "sha256": f"{seed - 5}" * 64,
            }
            for seed in EXPECTED_SEEDS
        ],
    }


def source_controls() -> list[dict[str, object]]:
    return [
        {
            "run_id": K1AI_RUN_ID,
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
        "correct_structure": 0.650,
        "wrong_sbox": 0.630,
        "corrupted_linear": 0.600,
        "no_structure": 0.500,
    }
    rows: list[dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        checkpoint = f"checkpoint-{seed}"
        state = f"state-{seed}"
        for split in EXPECTED_SPLITS:
            dataset = f"dataset-{seed}-{split}"
            row_count = 4096 if split == "train_seen" else 2048
            for condition in CONTROL_CONDITIONS:
                auc = aucs[condition]
                delta = 0.0 if condition == "correct_structure" else 0.1
                rows.append(
                    {
                        "run_id": "k1aj",
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "cipher_key": "midori64",
                        "rounds": 4,
                        "auc": auc,
                        "source_correct_auc": 0.650,
                        "correct_minus_condition_auc": 0.650 - auc,
                        "max_abs_probability_delta_from_correct": delta,
                        "mean_abs_probability_delta_from_correct": delta / 2,
                        "checkpoint_sha256": checkpoint,
                        "checkpoint_selected": "best",
                        "checkpoint_reported_seed": seed,
                        "state_dict_sha256": state,
                        "dataset_sha256": dataset,
                        "source_dataset_sha256": dataset,
                        "source_checkpoint_sha256": checkpoint,
                        "source_state_dict_sha256": state,
                        "source_decision": SOURCE_DECISION,
                        **{
                            f"source_{name}_sha256": digest
                            for name, digest in EXPECTED_SOURCE_DIGESTS.items()
                        },
                        "composition_sha256": f"composition-{condition}",
                        "rows": row_count,
                        "input_bits": 512,
                        "pairs_per_sample": 4,
                        "input_difference": INPUT_DIFFERENCE,
                        "negative_mode": "encrypted_random_plaintexts",
                        "sample_structure": "independent_pairs",
                        "parameter_count": 214_316,
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "epochs": 0,
                    }
                )
    return rows
