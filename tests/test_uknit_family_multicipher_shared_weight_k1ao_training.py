from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_multicipher_shared_weight_k1ao_training import (
    render_k1ao_training_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao_training import (
    CONTROL_CONDITIONS,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_STEPS_PER_REPLICA,
    FRESH_SPLITS,
    adjudicate_training,
    load_and_validate_config,
    load_sources,
)


def test_k1ao_training_config_freezes_equal_batch_single_optimizer_protocol() -> None:
    config = load_and_validate_config()

    assert config["training"]["equal_batches_per_cipher_per_epoch"] == 64
    assert config["training"]["optimizer_steps_per_epoch"] == 192
    assert config["training"]["optimizer_steps_total_per_replica"] == 1920
    assert config["training"]["checkpoint_metric"] == (
        "minimum_cross_key_auc_across_ciphers"
    )
    assert config["evaluation"]["expected_rows"] == EXPECTED_EVALUATION_ROWS
    assert config["gates"]["allow_macro_average_to_rescue_failure"] is False


def test_k1ao_training_sources_rebind_all_datasets_and_anchors() -> None:
    config = load_and_validate_config()

    _readiness, dataset_rows, datasets, anchors, checks = load_sources(config)

    assert len(dataset_rows) == len(datasets) == 18
    assert len(anchors) == 12
    assert all(checks.values()), checks


def test_k1ao_training_gate_passes_only_all_per_panel_clauses(tmp_path: Path) -> None:
    config = load_and_validate_config()
    training_rows = synthetic_training_rows()
    evaluation_rows = synthetic_evaluation_rows()
    checkpoints = synthetic_checkpoints(tmp_path)

    gate = adjudicate_training(
        config=config,
        source_checks={"source": True},
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        checkpoints=checkpoints,
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("shared_semantic_training_supported")
    assert gate["retention_all_panels"] is True
    assert gate["wrong_sbox_margin_all_panels"] is True
    assert gate["branch_margin_all_panels"] is True
    assert gate["remote_scale"] == "no"


def test_k1ao_training_gate_routes_retained_operator_insensitive_result(
    tmp_path: Path,
) -> None:
    rows = synthetic_evaluation_rows()
    for row in rows:
        if row["condition"] == "wrong_sbox_same_checkpoint":
            row["auc"] = 0.699
    gate = adjudicate_training(
        config=load_and_validate_config(),
        source_checks={"source": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=rows,
        checkpoints=synthetic_checkpoints(tmp_path),
    )

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("shared_weights_operator_insensitive")
    assert "stop scale" in gate["next_action"]


def test_k1ao_training_gate_routes_retention_and_semantic_failure(
    tmp_path: Path,
) -> None:
    rows = synthetic_evaluation_rows()
    for row in rows:
        if row["condition"] == "correct_runtime":
            row["auc"] = 0.60
        elif row["condition"] == "wrong_sbox_same_checkpoint":
            row["auc"] = 0.599
    gate = adjudicate_training(
        config=load_and_validate_config(),
        source_checks={"source": True},
        training_rows=synthetic_training_rows(),
        evaluation_rows=rows,
        checkpoints=synthetic_checkpoints(tmp_path),
    )

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("retention_and_semantics_failed")
    assert "do not increase pairs" in gate["next_action"]


def test_k1ao_training_plot_requires_complete_panels(tmp_path: Path) -> None:
    output = tmp_path / "curves.svg"

    report = render_k1ao_training_svg(
        {"status": "hold"}, synthetic_evaluation_rows(), output
    )

    assert output.is_file()
    assert report["evaluation_rows"] == EXPECTED_EVALUATION_ROWS
    assert report["comparison_panels"] == 12
    assert report["formal_scale_claim_present"] is False


def synthetic_training_rows() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "training": {
                "epochs": 10,
                "optimizer_steps": EXPECTED_STEPS_PER_REPLICA,
                "optimizer_state_step_min": EXPECTED_STEPS_PER_REPLICA,
                "optimizer_state_step_max": EXPECTED_STEPS_PER_REPLICA,
                "one_shared_optimizer": True,
                "equal_batches_per_cipher": True,
            },
        }
        for replica in (0, 1)
    ]


def synthetic_evaluation_rows() -> list[dict[str, object]]:
    aucs = {
        "correct_runtime": 0.70,
        "wrong_sbox_same_checkpoint": 0.68,
        "transition_branch_off_same_checkpoint": 0.67,
    }
    return [
        {
            "replica": replica,
            "cipher_key": cipher_key,
            "split": split,
            "condition": condition,
            "auc": aucs[condition],
            "anchor_auc": 0.69,
            "training_performed": False,
            "optimizer_steps": 0,
            "state_immutable_across_controls": True,
            "strict_state_dict_load": True,
        }
        for replica in (0, 1)
        for cipher_key in ("uknit64", "midori64", "dialga128")
        for split in FRESH_SPLITS
        for condition in CONTROL_CONDITIONS
    ]


def synthetic_checkpoints(tmp_path: Path) -> dict[int, dict[str, object]]:
    checkpoints = {}
    for replica in (0, 1):
        path = tmp_path / f"replica{replica}.pt"
        path.write_bytes(f"checkpoint-{replica}".encode("ascii"))
        from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
            file_sha256,
        )

        checkpoints[replica] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return checkpoints
