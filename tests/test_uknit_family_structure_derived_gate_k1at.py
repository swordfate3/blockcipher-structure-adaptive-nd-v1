from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_structure_derived_gate_k1at import (
    render_k1at_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import file_sha256
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    CONTROL_CONDITIONS,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_STEPS_PER_REPLICA,
    FRESH_SPLITS,
    MISMATCH_CONDITIONS,
    adjudicate_training,
    derive_structure_controls,
    load_and_validate_config,
    load_sources,
)


def test_k1at_config_freezes_same_budget_single_variable_protocol() -> None:
    config = load_and_validate_config()

    assert config["training"]["pairs_per_sample"] == 4
    assert config["training"]["optimizer_steps_total_per_replica"] == 1920
    assert config["controls"]["expected_rows"] == EXPECTED_EVALUATION_ROWS
    assert config["gates"]["cross_key_macro_improvement_per_replica"] == 0.005
    assert config["gates"]["per_panel_no_harm_margin"] == -0.005
    assert config["gates"]["minimum_passing_panels_per_mismatch"] == 10
    assert "16-pair expansion" in config["blocked_actions"]


def test_k1at_sources_and_runtime_summaries_rebind_exact_authority() -> None:
    config = load_and_validate_config()
    readiness, _k1as, dataset_rows, datasets, anchors, source_checks = load_sources(
        config
    )
    structures, controls, rows, structure_checks = derive_structure_controls(
        readiness_config=readiness,
        config=config,
    )

    assert len(dataset_rows) == len(datasets) == 18
    assert len(anchors) == 12
    assert len(structures) == len(controls) == len(rows) == 3
    assert all(source_checks.values()), source_checks
    assert all(structure_checks.values()), structure_checks
    assert all(
        set(cipher_controls) == set(CONTROL_CONDITIONS)
        for cipher_controls in controls.values()
    )


def test_k1at_gate_passes_only_improvement_no_harm_and_mismatch_gates(
    tmp_path: Path,
) -> None:
    gate = adjudicate_training(
        config=load_and_validate_config(),
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=synthetic_evaluation_rows(),
        checkpoints=synthetic_checkpoints(tmp_path),
    )

    assert gate["status"] == "pass"
    assert gate["cross_key_macro_improvement_both_replicas"] is True
    assert gate["per_panel_no_harm_all"] is True
    assert gate["descriptor_mismatch_gate_all"] is True
    assert gate["remote_scale"] == "no"


def test_k1at_gate_holds_when_one_mismatch_has_only_nine_passing_panels(
    tmp_path: Path,
) -> None:
    rows = synthetic_evaluation_rows()
    failed = 0
    for row in rows:
        if row["condition"] == "sbox_only_mismatch" and failed < 3:
            row["auc"] = 0.7095
            failed += 1
    gate = adjudicate_training(
        config=load_and_validate_config(),
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=rows,
        checkpoints=synthetic_checkpoints(tmp_path),
    )

    assert gate["status"] == "hold"
    assert gate["descriptor_mismatch_gate_all"] is False
    assert gate["mismatch_results"]["sbox_only_mismatch"]["passing_panels"] == 9
    assert "summary-identifiability" in gate["next_action"]


def test_k1at_gate_holds_on_macro_or_panel_harm(tmp_path: Path) -> None:
    rows = synthetic_evaluation_rows(correct_auc=0.702, anchor_auc=0.700)
    gate = adjudicate_training(
        config=load_and_validate_config(),
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=rows,
        checkpoints=synthetic_checkpoints(tmp_path),
    )

    assert gate["status"] == "hold"
    assert gate["cross_key_macro_improvement_both_replicas"] is False

    harmed = synthetic_evaluation_rows()
    for row in harmed:
        if (
            row["replica"] == 0
            and row["cipher_key"] == "uknit64"
            and row["split"] == "same_key_fresh"
        ):
            row["k1ao_anchor_auc"] = 0.72
    gate = adjudicate_training(
        config=load_and_validate_config(),
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=harmed,
        checkpoints=synthetic_checkpoints(tmp_path),
    )
    assert gate["status"] == "hold"
    assert gate["per_panel_no_harm_all"] is False


def test_k1at_gate_rejects_runtime_structure_mismatch(tmp_path: Path) -> None:
    rows = synthetic_evaluation_rows()
    rows[0]["runtime_structure_held_correct"] = False
    gate = adjudicate_training(
        config=load_and_validate_config(),
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=rows,
        checkpoints=synthetic_checkpoints(tmp_path),
    )

    assert gate["status"] == "invalid"
    assert (
        "runtime_structure_held_correct_for_all_controls"
        in gate["failed_protocol_checks"]
    )


def synthetic_training_rows() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "trainable_parameter_count": 219_752,
            "state_dict_entries": 55,
            "uses_cipher_identity": False,
            "structure_gate_uses_cipher_identity": False,
            "structure_gate_shared": True,
            "training": {
                "epochs": 10,
                "optimizer_steps": EXPECTED_STEPS_PER_REPLICA,
                "optimizer_state_step_min": EXPECTED_STEPS_PER_REPLICA,
                "optimizer_state_step_max": EXPECTED_STEPS_PER_REPLICA,
                "one_shared_optimizer": True,
                "equal_batches_per_cipher": True,
                "correct_summary_precomputed_once_per_cipher": True,
            },
        }
        for replica in (0, 1)
    ]


def synthetic_evaluation_rows(
    *,
    correct_auc: float = 0.71,
    anchor_auc: float = 0.70,
) -> list[dict[str, object]]:
    aucs = {
        "correct_descriptor": correct_auc,
        "full_mismatch": correct_auc - 0.02,
        "sbox_only_mismatch": correct_auc - 0.02,
        "linear_only_mismatch": correct_auc - 0.02,
        "descriptor_disabled": correct_auc - 0.01,
    }
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "split": split,
            "condition": condition,
            "auc": aucs[condition],
            "k1ao_anchor_auc": anchor_auc,
            "effective_transition_gate": 0.1,
            "runtime_structure_cipher_key": cipher,
            "runtime_structure_held_correct": True,
            "training_performed": False,
            "optimizer_steps": 0,
            "state_immutable_across_controls": True,
            "strict_state_dict_load": True,
        }
        for replica in (0, 1)
        for cipher in ("uknit64", "midori64", "dialga128")
        for split in FRESH_SPLITS
        for condition in CONTROL_CONDITIONS
    ]


def synthetic_checkpoints(tmp_path: Path) -> dict[int, dict[str, object]]:
    checkpoints = {}
    for replica in (0, 1):
        path = tmp_path / f"replica{replica}.pt"
        path.write_bytes(f"checkpoint-{replica}".encode("ascii"))
        checkpoints[replica] = {"path": str(path), "sha256": file_sha256(path)}
    return checkpoints


def test_k1at_mismatch_contract_has_exact_three_semantic_families() -> None:
    assert MISMATCH_CONDITIONS == (
        "full_mismatch",
        "sbox_only_mismatch",
        "linear_only_mismatch",
    )


def test_k1at_plot_writes_four_panel_chinese_svg(tmp_path: Path) -> None:
    rows = synthetic_evaluation_rows(correct_auc=0.702, anchor_auc=0.700)
    gate = adjudicate_training(
        config=load_and_validate_config(),
        source_checks={"source": True},
        structure_checks={"structure": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=rows,
        checkpoints=synthetic_checkpoints(tmp_path),
    )
    output = tmp_path / "curves.svg"

    report = render_k1at_svg(gate, rows, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["evaluation_rows"] == 60
    assert "运行时结构门控没有稳定改善三密码共享模型" in text
    assert "2048/class" in text
    assert "不是正式规模" in text
