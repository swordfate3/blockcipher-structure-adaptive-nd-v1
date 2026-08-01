from __future__ import annotations

import torch

from blockcipher_nd.cli.plot_runtime_spn_source_bundle_collision_k1by10 import (
    render_k1by10_svg,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_source_bundle_collision_k1by10 as k1by10,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_source_bundle_histogram_k1by9 as k1by9,
)


def test_k1by10_config_and_k1by9_source_are_frozen() -> None:
    config = k1by10.load_and_validate_config()

    assert config["run_id"] == k1by10.RUN_ID
    assert tuple(config["audit"]["cell_taps"]) == k1by10.CELL_TAPS
    assert all(k1by10.source_binding_checks(config).values())


def test_k1by10_partition_is_transversal_and_stage_stable() -> None:
    models, _source_row, _metadata = k1by9.build_models(
        k1by9.load_and_validate_config(),
        seed=2,
    )

    rows = k1by10.partition_rows(models)
    summary = k1by10.partition_summary(rows)

    assert len(rows) == 32
    assert {row["intersection_size"] for row in rows} == {1, 2}
    assert {row["changed_peer_count"] for row in rows} == {4, 6}
    assert summary["stage_partitions_identical"] is True
    assert all(
        stage["correct_peer_pairs"] == 24
        and stage["affine_peer_pairs"] == 24
        and stage["shared_peer_pairs"] == 4
        and stage["changed_peer_pairs"] == 40
        for stage in summary["stages"].values()
    )


def test_k1by10_readiness_replays_models_taps_and_partitions() -> None:
    readiness = k1by10.build_readiness(k1by10.load_and_validate_config())

    assert readiness["status"] == "pass", readiness
    assert readiness["execution_authorized"] is True
    assert readiness["training_authorized"] is False
    assert readiness["optimizer_steps_authorized"] == 0
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())


def test_k1by10_adjudication_identifies_same_cell_two_stage_locus(monkeypatch) -> None:
    config = k1by10.load_and_validate_config()
    rows = _probe_rows(supported_cell=5)
    partition = _partition_rows()
    monkeypatch.setattr(
        k1by10,
        "source_binding_checks",
        lambda _config: {"source": True},
    )
    monkeypatch.setattr(
        k1by9,
        "model_metadata_frozen",
        lambda _values: True,
    )

    gate, effects = k1by10.adjudicate(
        config,
        result_rows=rows,
        partition=partition,
        model_metadata={"2": {}, "3": {}},
        readiness=_readiness(),
        sources_unchanged=True,
    )

    assert len(effects) == 192
    assert gate["status"] == "pass"
    assert gate["research_gate_passed"] is True
    assert gate["supported_target_cells"] == [5]
    assert gate["decision"].endswith("oversmoothing_locus_identified")


def test_k1by10_adjudication_closes_partition_when_locus_is_not_stable(
    monkeypatch,
) -> None:
    config = k1by10.load_and_validate_config()
    monkeypatch.setattr(
        k1by10,
        "source_binding_checks",
        lambda _config: {"source": True},
    )
    monkeypatch.setattr(
        k1by9,
        "model_metadata_frozen",
        lambda _values: True,
    )

    gate, _effects = k1by10.adjudicate(
        config,
        result_rows=_probe_rows(supported_cell=None),
        partition=_partition_rows(),
        model_metadata={"2": {}, "3": {}},
        readiness=_readiness(),
        sources_unchanged=True,
    )

    assert gate["status"] == "pass"
    assert gate["research_gate_passed"] is False
    assert gate["supported_target_cells"] == []
    assert gate["decision"].endswith("no_stable_partition_locus_identified")


def test_k1by10_plot_uses_plain_language_gate_and_stage_labels(tmp_path) -> None:
    output = tmp_path / "curves.svg"
    effects = k1by10.effect_rows(_probe_rows(supported_cell=5))
    gate = {
        "run_id": k1by10.RUN_ID,
        "status": "pass",
        "research_gate_passed": True,
        "supported_target_cells": [5],
    }

    report = render_k1by10_svg(gate, effects, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 4
    assert "逐单元过度平滑审计" in svg
    assert "捕获阶段0" in svg
    assert "只允许下一步非平均的逐单元偏差残差" in svg


def _probe_rows(*, supported_cell: int | None) -> list[dict]:
    rows = []
    for seed in k1by10.EXPECTED_SEEDS:
        for representation in k1by9.REPRESENTATIONS:
            for runtime in k1by9.RUNTIME_PROGRAMS:
                for stage in k1by10.STAGES:
                    for cell in range(k1by10.TARGET_CELLS):
                        for tap_index, tap in enumerate(k1by10.CELL_TAPS):
                            auc = 0.60
                            if (
                                supported_cell is not None
                                and cell == supported_cell
                                and tap in k1by10.REQUIRED_TAPS
                                and representation
                                == "candidate_source_bundle_mean"
                            ):
                                if seed == 2 and runtime == "affine_runtime":
                                    auc = 0.61
                                if seed == 3 and runtime == "correct_runtime":
                                    auc = 0.61
                            rows.append(
                                {
                                    "seed": seed,
                                    "representation": representation,
                                    "runtime_program": runtime,
                                    "tap_stage": stage,
                                    "target_cell": cell,
                                    "tap": tap,
                                    "tap_index": tap_index,
                                    "probe_auc": auc,
                                    "discovery_positive_rows": 512,
                                    "discovery_negative_rows": 512,
                                    "evaluation_positive_rows": 512,
                                    "evaluation_negative_rows": 512,
                                }
                            )
    return rows


def _partition_rows() -> list[dict]:
    return [
        {
            "tap_stage": stage,
            "program_stage": 1 - stage,
            "target_cell": cell,
            "correct_peers": [
                cell,
                (cell + 1) % 16,
                (cell + 2) % 16,
                (cell + 3) % 16,
            ],
            "affine_peers": [
                cell,
                (cell + 4) % 16,
                (cell + 8) % 16,
                (cell + 12) % 16,
            ],
            "intersection_peers": [cell],
            "intersection_size": 1,
            "removed_peers": [
                (cell + 1) % 16,
                (cell + 2) % 16,
                (cell + 3) % 16,
            ],
            "added_peers": [
                (cell + 4) % 16,
                (cell + 8) % 16,
                (cell + 12) % 16,
            ],
            "changed_peer_count": 6,
        }
        for stage in k1by10.STAGES
        for cell in range(k1by10.TARGET_CELLS)
    ]


def _readiness() -> dict:
    return {
        "status": "pass",
        "execution_authorized": True,
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
    }


def test_partition_matrices_remain_runtime_buffers() -> None:
    models, _source_row, _metadata = k1by9.build_models(
        k1by9.load_and_validate_config(),
        seed=2,
    )
    matrix = models[
        k1by9.condition_key("candidate_source_bundle_mean", "correct_runtime")
    ].conditioner.linear_source_bundle_equivalence

    assert isinstance(matrix, torch.Tensor)
    assert matrix.requires_grad is False
