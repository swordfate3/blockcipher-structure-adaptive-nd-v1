from __future__ import annotations

import torch

from blockcipher_nd.cli.plot_runtime_spn_edge_context_covariance_k1by11 import (
    render_k1by11_svg,
)
from blockcipher_nd.models.structure.spn.ordered_primitive_conditioner import (
    LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE,
    edge_context_covariance_histogram,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_edge_context_covariance_k1by11 as k1by11,
)


def test_k1by11_config_and_sources_are_frozen() -> None:
    config = k1by11.load_and_validate_config()

    assert config["run_id"] == k1by11.RUN_ID
    assert tuple(config["audit"]["conditions"]) == k1by11.CONDITIONS
    assert all(k1by11.source_binding_checks(config).values())


def test_edge_context_covariance_preserves_mass_and_joint_relabeling() -> None:
    generator = torch.Generator().manual_seed(20260801)
    target_values = torch.randint(0, 16, (5, 7, 4), generator=generator)
    source_bits = torch.randint(
        0,
        2,
        (5, 7, 4, 4),
        generator=generator,
    ).to(torch.float32)
    edge_cells = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        dtype=torch.long,
    )
    edge_roles = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        dtype=torch.long,
    )
    masks = torch.ones(4, 2)

    result = edge_context_covariance_histogram(
        target_values,
        source_bits,
        edge_cells,
        edge_roles,
        masks,
    )
    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(4)
    relabeled = edge_context_covariance_histogram(
        target_values[..., order],
        source_bits[..., order, :],
        inverse[edge_cells[order]],
        edge_roles[order],
        masks[order],
    )

    assert result.shape == (5, 4, 16)
    assert torch.allclose(result.sum(dim=-1), torch.ones(5, 4))
    assert torch.equal(relabeled, result[..., order, :])


def test_candidate_adds_no_parameters_and_distinguishes_edge_bindings() -> None:
    config = k1by11.load_and_validate_config()
    models, _source_row, metadata = k1by11.build_models(config, seed=2)

    assert set(models) == set(k1by11.CONDITIONS)
    assert set(metadata["parameter_fingerprints"].values()) == {
        metadata["source_parameter_fingerprint"]
    }
    assert len(set(metadata["edge_source_cell_fingerprints"].values())) == 3
    for condition, model in models.items():
        if condition.startswith("candidate_edge_covariance"):
            assert (
                model.linear_histogram_mode
                == LINEAR_HISTOGRAM_EDGE_CONTEXT_COVARIANCE
            )


def test_k1by11_readiness_proves_zero_training_edge_gate() -> None:
    readiness = k1by11.build_readiness(k1by11.load_and_validate_config())

    assert readiness["status"] == "pass", readiness
    assert readiness["execution_authorized"] is True
    assert readiness["training_authorized"] is False
    assert readiness["optimizer_steps_authorized"] == 0
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())


def test_k1by11_adjudication_routes_any_control_margin_miss(monkeypatch) -> None:
    config = k1by11.load_and_validate_config()
    monkeypatch.setattr(
        k1by11,
        "source_binding_checks",
        lambda _config: {"source": True},
    )
    monkeypatch.setattr(k1by11, "model_metadata_frozen", lambda _values: True)

    gate = k1by11.adjudicate(
        config,
        result_rows=_result_rows(shuffled_miss=True),
        final_evaluation=_final_evaluation(),
        model_metadata={"2": {}, "3": {}},
        readiness=_readiness(),
        sources_unchanged=True,
    )

    assert gate["status"] == "pass"
    assert gate["research_gate_passed"] is False
    assert gate["decision"].endswith("input_modulation_not_supported")


def test_k1by11_adjudication_supports_complete_two_control_gate(monkeypatch) -> None:
    config = k1by11.load_and_validate_config()
    monkeypatch.setattr(
        k1by11,
        "source_binding_checks",
        lambda _config: {"source": True},
    )
    monkeypatch.setattr(k1by11, "model_metadata_frozen", lambda _values: True)

    gate = k1by11.adjudicate(
        config,
        result_rows=_result_rows(shuffled_miss=False),
        final_evaluation=_final_evaluation(),
        model_metadata={"2": {}, "3": {}},
        readiness=_readiness(),
        sources_unchanged=True,
    )

    assert gate["status"] == "pass"
    assert gate["research_gate_passed"] is True
    assert gate["decision"].endswith("edge_context_covariance_supported")


def test_k1by11_plot_preserves_chinese_gate_and_failure_explanation(tmp_path) -> None:
    gate = {
        "run_id": k1by11.RUN_ID,
        "status": "pass",
        "research_gate_passed": False,
        "seed_results": _plot_seed_results(),
    }
    output = tmp_path / "curves.svg"

    report = render_k1by11_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["research_gate_passed"] is False
    assert report["taps"] == [*k1by11.TAPS, "final_output"]
    assert "逐单元边上下文协方差未通过内部访问门槛" in svg
    assert "每个层级同时高于 +0.005" in svg
    assert "最终验证 AUC（局部放大）" in svg
    assert "关闭输入调制，干预位置后移" in svg


def _result_rows(*, shuffled_miss: bool) -> list[dict]:
    rows = []
    for seed in k1by11.EXPECTED_SEEDS:
        for condition in k1by11.CONDITIONS:
            for tap_index, tap in enumerate(k1by11.TAPS):
                auc = 0.62
                if condition.endswith("affine_runtime"):
                    auc = 0.60
                elif condition.endswith("correct_state_shuffled_edges"):
                    auc = 0.616 if shuffled_miss and seed == 3 else 0.60
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
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


def _final_evaluation() -> dict:
    values = {}
    for seed in k1by11.EXPECTED_SEEDS:
        aucs = {
            "anchor_local__correct_runtime": 0.68,
            "anchor_local__affine_runtime": 0.66,
            "candidate_edge_covariance__correct_runtime": 0.681,
            "candidate_edge_covariance__affine_runtime": 0.66,
            "candidate_edge_covariance__correct_state_shuffled_edges": 0.655,
        }
        values[str(seed)] = {
            "condition_auc": aucs,
            "anchor_replay": {
                condition: {
                    "source_auc": auc,
                    "replayed_auc": auc,
                    "absolute_error": 0.0,
                }
                for condition, auc in aucs.items()
                if condition.startswith("anchor_local")
            },
        }
    return values


def _readiness() -> dict:
    return {
        "status": "pass",
        "execution_authorized": True,
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
    }


def _plot_seed_results() -> dict[str, dict]:
    values = {}
    for seed, offset in (("2", 0.0), ("3", -0.004)):
        taps = {}
        for tap_index, tap in enumerate(k1by11.TAPS):
            correct = 0.68 + 0.006 * tap_index + offset
            taps[tap] = {
                "correct_runtime_probe_auc": correct,
                "affine_runtime_probe_auc": correct - 0.012,
                "shuffled_edges_probe_auc": correct - 0.009,
                "correct_minus_affine_probe_auc": 0.012,
                "correct_minus_shuffled_probe_auc": 0.009,
            }
        values[seed] = {
            "taps": taps,
            "correct_minus_affine_final_auc": 0.011,
            "correct_minus_shuffled_final_auc": 0.008,
            "anchor_correct_final_auc": 0.68 + offset,
            "candidate_correct_final_auc": 0.681 + offset,
            "candidate_affine_final_auc": 0.67 + offset,
            "candidate_shuffled_final_auc": 0.673 + offset,
        }
    return values
