from __future__ import annotations

import torch

from blockcipher_nd.cli.plot_runtime_spn_post_expert_edge_residual_k1by12 import (
    render_k1by12_svg,
)
from blockcipher_nd.models.structure.spn.ordered_primitive_conditioner import (
    POST_EXPERT_RESIDUAL_EDGE_GATED_LAPLACIAN,
    POST_EXPERT_RESIDUAL_NONE,
    PostExpertStructuralResidual,
    post_expert_edge_gated_laplacian,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_post_expert_edge_residual_k1by12 as k1by12,
)


def test_post_expert_residual_is_bounded_and_joint_relabel_equivariant() -> None:
    generator = torch.Generator().manual_seed(20260801)
    expert = torch.randn(5, 4, 8, generator=generator)
    gate = torch.randn(4, 8, generator=generator)
    cells = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    masks = torch.ones(4, 2)

    result = post_expert_edge_gated_laplacian(expert, gate, cells, masks)
    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(4)
    relabeled = post_expert_edge_gated_laplacian(
        expert[:, order],
        gate[order],
        inverse[cells[order]],
        masks[order],
    )

    assert result.shape == expert.shape
    assert float((result - expert).abs().max()) <= 1.0
    assert torch.equal(relabeled, result[:, order])


def test_disabled_post_expert_residual_is_exact_identity() -> None:
    module = PostExpertStructuralResidual(POST_EXPERT_RESIDUAL_NONE)
    values = torch.randn(3, 4, 8)

    result = module(values, torch.randn(4, 8), None, torch.ones(4, 1))

    assert result is values
    assert list(module.parameters()) == []


def test_k1by12_config_sources_and_models_are_frozen() -> None:
    config = k1by12.load_and_validate_config()
    models, _source_row, metadata = k1by12.build_models(config, seed=2)

    assert config["run_id"] == k1by12.RUN_ID
    assert all(k1by12.source_binding_checks(config).values())
    assert set(models) == set(k1by12.CONDITIONS)
    assert set(metadata["parameter_fingerprints"].values()) == {
        metadata["source_parameter_fingerprint"]
    }
    assert len(set(metadata["edge_source_cell_fingerprints"].values())) == 3
    for condition, model in models.items():
        expected = (
            POST_EXPERT_RESIDUAL_NONE
            if condition.startswith("anchor_local")
            else POST_EXPERT_RESIDUAL_EDGE_GATED_LAPLACIAN
        )
        assert model.post_expert_residual_mode == expected


def test_k1by12_readiness_proves_zero_training_post_expert_gate() -> None:
    readiness = k1by12.build_readiness(k1by12.load_and_validate_config())

    assert readiness["status"] == "pass", readiness
    assert readiness["execution_authorized"] is True
    assert readiness["training_authorized"] is False
    assert readiness["optimizer_steps_authorized"] == 0
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())


def test_k1by12_adjudication_routes_failed_control_to_trainable_adapter(
    monkeypatch,
) -> None:
    config = k1by12.load_and_validate_config()
    monkeypatch.setattr(
        k1by12,
        "source_binding_checks",
        lambda _config: {"source": True},
    )
    monkeypatch.setattr(k1by12, "model_metadata_frozen", lambda _values: True)

    gate = k1by12.adjudicate(
        config,
        result_rows=_result_rows(shuffled_miss=True),
        final_evaluation=_final_evaluation(),
        model_metadata={"2": {}, "3": {}},
        readiness=_readiness(),
        sources_unchanged=True,
    )

    assert gate["status"] == "pass"
    assert gate["research_gate_passed"] is False
    assert gate["decision"].endswith("deterministic_interventions_exhausted")
    assert "trainable post-expert adapter" in gate["next_action"]


def test_k1by12_adjudication_supports_complete_post_expert_locus(monkeypatch) -> None:
    config = k1by12.load_and_validate_config()
    monkeypatch.setattr(
        k1by12,
        "source_binding_checks",
        lambda _config: {"source": True},
    )
    monkeypatch.setattr(k1by12, "model_metadata_frozen", lambda _values: True)

    gate = k1by12.adjudicate(
        config,
        result_rows=_result_rows(shuffled_miss=False),
        final_evaluation=_final_evaluation(),
        model_metadata={"2": {}, "3": {}},
        readiness=_readiness(),
        sources_unchanged=True,
    )

    assert gate["status"] == "pass"
    assert gate["research_gate_passed"] is True
    assert gate["decision"].endswith("post_expert_locus_supported")


def test_k1by12_plot_preserves_chinese_gate_and_decision(tmp_path) -> None:
    gate = {
        "run_id": k1by12.RUN_ID,
        "status": "pass",
        "research_gate_passed": False,
        "seed_results": _plot_seed_results(),
    }
    output = tmp_path / "curves.svg"

    report = render_k1by12_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["taps"] == [*k1by12.TAPS, "final_output"]
    assert "冻结专家后的边结构残差审计" in svg
    assert "同时高于仿射和打乱边控制 +0.005" in svg
    assert "最终验证 AUC（局部放大）" in svg
    assert "停止确定性冻结干预" in svg


def _result_rows(*, shuffled_miss: bool) -> list[dict]:
    rows = []
    for seed in k1by12.EXPECTED_SEEDS:
        for condition in k1by12.CONDITIONS:
            for tap_index, tap in enumerate(k1by12.TAPS):
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
    for seed in k1by12.EXPECTED_SEEDS:
        aucs = {
            "anchor_local__correct_runtime": 0.68,
            "anchor_local__affine_runtime": 0.66,
            "candidate_post_expert__correct_runtime": 0.681,
            "candidate_post_expert__affine_runtime": 0.66,
            "candidate_post_expert__correct_state_shuffled_edges": 0.655,
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
        for tap_index, tap in enumerate(k1by12.TAPS):
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
