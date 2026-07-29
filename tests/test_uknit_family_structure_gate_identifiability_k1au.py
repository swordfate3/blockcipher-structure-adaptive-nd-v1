from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_structure_gate_identifiability_k1au import (
    render_k1au_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_gate_identifiability_k1au import (
    EXPECTED_CONTROL_ROWS,
    EXPECTED_RESULT_ROWS,
    adjudicate,
    load_and_validate_config,
    load_authority,
)


def test_k1au_config_freezes_zero_update_layerwise_audit() -> None:
    config = load_and_validate_config()

    assert config["audit"]["training_performed"] is False
    assert config["audit"]["optimizer_steps"] == 0
    assert config["audit"]["rows_per_split"] == 32
    assert config["audit"]["expected_result_rows"] == EXPECTED_RESULT_ROWS
    assert config["audit"]["expected_control_rows"] == EXPECTED_CONTROL_ROWS
    assert config["gates"]["required_centered_correct_hidden_rank"] == 2
    assert config["gates"]["minimum_aligned_mismatch_panels"] == 15


def test_k1au_authority_rebinds_checkpoints_datasets_and_summaries() -> None:
    config = load_and_validate_config()

    (
        _readiness,
        _k1as,
        datasets,
        structures,
        controls,
        checkpoints,
        checkpoint_rows,
        checks,
    ) = load_authority(config)

    assert len(datasets) == 18
    assert len(structures) == len(controls) == 3
    assert len(checkpoints) == len(checkpoint_rows) == 2
    assert all(checks.values()), checks


def test_k1au_gate_localizes_bottleneck_after_preserved_hidden_layer() -> None:
    gate = adjudicate(
        config=load_and_validate_config(),
        source_checks={"source": True},
        results=synthetic_results(),
        controls=synthetic_controls(aligned_panels=12),
        checkpoints={0: {}, 1: {}},
        cross_replica={
            "gate_rank_correlation": -0.5,
            "jacobian_cosine_by_cipher": {
                "uknit64": 0.2,
                "midori64": 0.3,
                "dialga128": 0.4,
            },
        },
    )

    assert gate["status"] == "pass"
    assert gate["representation_preserved_through_hidden"] is True
    assert gate["final_scalar_mapping_stable"] is False
    assert gate["decision"].endswith("final_scalar_projection_bottleneck_supported")
    assert "multi-channel" in gate["next_action"]


def test_k1au_gate_holds_when_hidden_representation_collapses() -> None:
    controls = synthetic_controls(aligned_panels=18)
    for row in controls:
        row["hidden_l2_distance"] = 1e-7
    gate = adjudicate(
        config=load_and_validate_config(),
        source_checks={"source": True},
        results=synthetic_results(),
        controls=controls,
        checkpoints={0: {}, 1: {}},
        cross_replica={
            "gate_rank_correlation": 1.0,
            "jacobian_cosine_by_cipher": {
                "uknit64": 0.9,
                "midori64": 0.9,
                "dialga128": 0.9,
            },
        },
    )

    assert gate["status"] == "hold"
    assert gate["representation_preserved_through_hidden"] is False
    assert gate["decision"].endswith("hidden_representation_not_identifiable")


def synthetic_results() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "hidden_embedding_l2": 1.0,
            "projection_value": 0.1,
            "effective_gate": 0.2,
            "sbox_jacobian_l2": 0.01,
            "linear_jacobian_l2": 0.01,
            "cross_replica_gate_rank_correlation": -0.5,
            "cross_replica_jacobian_cosine": 0.3,
            "centered_correct_hidden_rank": 2,
            "uses_cipher_identity": False,
            "structure_gate_uses_cipher_identity": False,
            "structure_gate_shared": True,
            "state_immutable": True,
            "training_performed": False,
            "optimizer_steps": 0,
        }
        for replica in (0, 1)
        for cipher in ("uknit64", "midori64", "dialga128")
    ]


def synthetic_controls(*, aligned_panels: int) -> list[dict[str, object]]:
    rows = []
    unique_index = 0
    for replica in (0, 1):
        for cipher in ("uknit64", "midori64", "dialga128"):
            for condition in (
                "full_mismatch",
                "sbox_only_mismatch",
                "linear_only_mismatch",
            ):
                aligned = unique_index < aligned_panels
                unique_index += 1
                for split in ("same_key_fresh", "cross_key_validation"):
                    rows.append(
                        {
                            "replica": replica,
                            "cipher_key": cipher,
                            "condition": condition,
                            "split": split,
                            "raw_summary_l2_distance": 0.2,
                            "hidden_l2_distance": 0.1,
                            "projection_alignment_abs_cosine": 0.2 if aligned else 0.01,
                            "projection_value_delta": 0.01,
                            "effective_gate_delta": 0.01,
                            "mean_abs_logit_delta": 0.01,
                            "max_abs_logit_delta": 0.02,
                            "runtime_structure_held_correct": True,
                            "state_immutable": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def test_k1au_plot_writes_layerwise_chinese_svg(tmp_path: Path) -> None:
    results = synthetic_results()
    controls = synthetic_controls(aligned_panels=18)
    gate = adjudicate(
        config=load_and_validate_config(),
        source_checks={"source": True},
        results=results,
        controls=controls,
        checkpoints={0: {}, 1: {}},
        cross_replica={
            "gate_rank_correlation": -0.5,
            "jacobian_cosine_by_cipher": {
                "uknit64": 0.55,
                "midori64": 0.56,
                "dialga128": 0.57,
            },
        },
    )
    output = tmp_path / "curves.svg"

    report = render_k1au_svg(gate, results, controls, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["unique_control_panels"] == 18
    assert "结构信息保留到隐藏层" in text
    assert "单标量门控的密码排序不稳定" in text
    assert "不是新AUC" in text
