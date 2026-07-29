from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_linear_summary_collision_k1ba import (
    render_k1ba_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_linear_summary_collision_k1ba import (
    adjudicate_audit,
    derive_collision_controls,
    load_and_validate_config,
    load_authority,
)


def test_k1ba_source_and_distinct_operator_collisions_are_exact() -> None:
    config = load_and_validate_config()
    (
        _readiness,
        dataset_rows,
        datasets,
        structures,
        controls,
        _summary_rows,
        checkpoints,
        source_controls,
        checks,
    ) = load_authority(config)

    collision_summaries, rows = derive_collision_controls(
        structures=structures,
        controls=controls,
        corruption_seed=config["evaluation"]["corruption_seed"],
    )

    assert all(checks.values()), checks
    assert len(dataset_rows) == len(datasets) == 18
    assert len(checkpoints) == 2
    assert len(source_controls) == 24
    assert len(collision_summaries) == len(rows) == 3
    assert all(row["operator_changed"] is True for row in rows)
    assert all(row["summary_collision_exact"] is True for row in rows)
    assert all(row["collision_summary_max_abs_delta"] == 0.0 for row in rows)
    assert all(row["matrix_hamming_fraction"] >= 0.001 for row in rows)


def test_k1ba_gate_separates_collision_rank_inertia_and_invalid(
    tmp_path: Path,
) -> None:
    config = load_and_validate_config()
    checkpoints = _synthetic_checkpoints(tmp_path)
    collision_rows = _synthetic_collision_rows()
    rows = _synthetic_result_rows()

    collision = adjudicate_audit(
        config=config,
        source_checks={"source": True},
        collision_rows=collision_rows,
        rows=rows,
        checkpoints=checkpoints,
    )
    assert collision["status"] == "pass"
    assert collision["collision_results"]["mechanism_supported"] is True
    assert "not_topology_identifying" in collision["decision"]

    rank_only_rows = deepcopy(rows)
    for row in rank_only_rows:
        if row["condition"] == "same_summary_corrupted_linear":
            row["probabilities_sha256"] = "collision-different"
    rank_only_collisions = deepcopy(collision_rows)
    for row in rank_only_collisions:
        row["collision_summary_max_abs_delta"] = 0.1
        row["summary_collision_exact"] = False
    rank_only = adjudicate_audit(
        config=config,
        source_checks={"source": True},
        collision_rows=rank_only_collisions,
        rows=rank_only_rows,
        checkpoints=checkpoints,
    )
    assert rank_only["status"] == "pass"
    assert rank_only["scalar_rank_results"]["mechanism_supported"] is True
    assert "rank_inertia" in rank_only["decision"]

    invalid_rows = deepcopy(rows)
    invalid_rows[0]["state_immutable"] = False
    invalid = adjudicate_audit(
        config=config,
        source_checks={"source": True},
        collision_rows=collision_rows,
        rows=invalid_rows,
        checkpoints=checkpoints,
    )
    assert invalid["status"] == "invalid"
    assert "zero_update_correct_runtime_immutable" in invalid[
        "failed_protocol_checks"
    ]


def test_k1ba_plot_writes_four_panel_chinese_svg(tmp_path: Path) -> None:
    gate = adjudicate_audit(
        config=load_and_validate_config(),
        source_checks={"source": True},
        collision_rows=_synthetic_collision_rows(),
        rows=_synthetic_result_rows(),
        checkpoints=_synthetic_checkpoints(tmp_path),
    )
    output = tmp_path / "curves.svg"

    report = render_k1ba_svg(gate, _synthetic_collision_rows(), output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["result_panels"] == 12
    assert "18维统计摘要无法标识真实线性拓扑" in text
    assert "摘要Δ=0.0" in text
    assert "不是正式规模" in text


def _synthetic_collision_rows() -> list[dict[str, object]]:
    return [
        {
            "cipher_key": cipher,
            "operator_changed": True,
            "summary_collision_exact": True,
            "matrix_hamming_fraction": 0.1,
            "collision_summary_max_abs_delta": 0.0,
            "cross_cipher_active_linear_dimension_count": 3,
        }
        for cipher in ("uknit64", "midori64", "dialga128")
    ]


def _synthetic_result_rows() -> list[dict[str, object]]:
    rows = []
    for replica in (0, 1):
        for cipher in ("uknit64", "midori64", "dialga128"):
            for split in ("same_key_fresh", "cross_key_validation"):
                for condition in (
                    "correct_descriptor",
                    "cross_cipher_linear_mismatch",
                    "same_summary_corrupted_linear",
                ):
                    is_cross = condition == "cross_cipher_linear_mismatch"
                    rows.append(
                        {
                            "replica": replica,
                            "cipher_key": cipher,
                            "split": split,
                            "condition": condition,
                            "auc": 0.6995 if is_cross else 0.7,
                            "correct_minus_condition_auc": 0.0005
                            if is_cross
                            else 0.0,
                            "effective_edge_gate": 0.11 if is_cross else 0.1,
                            "effective_transition_gate": 0.2,
                            "edge_gate_delta_from_correct": 0.01
                            if is_cross
                            else 0.0,
                            "transition_gate_delta_from_correct": 0.0,
                            "mean_abs_probability_delta_from_correct": 0.001
                            if is_cross
                            else 0.0,
                            "maximum_abs_probability_delta_from_correct": 0.01
                            if is_cross
                            else 0.0,
                            "probability_spearman_from_correct": 0.9999,
                            "probabilities_sha256": "cross"
                            if is_cross
                            else "same",
                            "source_auc_replay_delta": 0.0,
                            "source_probability_hash_replayed": True,
                            "state_immutable": True,
                            "runtime_structure_cipher_key": cipher,
                            "runtime_structure_held_correct": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def _synthetic_checkpoints(tmp_path: Path) -> dict[int, dict[str, object]]:
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
            "best_epoch": 9,
        }
    return checkpoints
