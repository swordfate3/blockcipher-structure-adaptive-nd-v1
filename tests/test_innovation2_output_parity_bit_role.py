from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.cli.plot_innovation2_output_parity_spn_local import (
    render_spn_local_readiness,
)
from blockcipher_nd.models.structure.spn.present_bit_role_parity_predictor import (
    PresentBitRoleParityPredictor,
    PresentBitRoleParityPredictorSpec,
    present_player,
    wrong_player,
)
from blockcipher_nd.tasks.innovation2.output_parity_bit_role import (
    adjudicate_bit_role_readiness,
    build_bit_role_data,
    validate_bit_role_contract,
)
from blockcipher_nd.tasks.innovation2.output_parity_prediction import (
    OutputParityPredictionConfig,
    OutputPredictionMlp,
)


def _small_config() -> OutputParityPredictionConfig:
    return OutputParityPredictionConfig(
        run_id="test-output-parity-bit-role",
        rounds=3,
        seed=0,
        train_rows=128,
        validation_rows=64,
        test_rows=64,
        hidden_dim=128,
        epochs=1,
        batch_size=32,
    )


def test_bit_role_players_are_distinct_bijections_and_preserve_output_shape() -> None:
    true_route = present_player()
    wrong_route = wrong_player()
    assert sorted(true_route) == list(range(64))
    assert sorted(wrong_route) == list(range(64))
    assert true_route != wrong_route
    assert true_route[:4] == (0, 16, 32, 48)
    model = PresentBitRoleParityPredictor(PresentBitRoleParityPredictorSpec())
    assert model(torch.zeros(2, 64)).shape == (2, 16)


def test_bit_role_true_and_wrong_models_are_parameter_matched() -> None:
    torch.manual_seed(1000)
    true_model = PresentBitRoleParityPredictor(
        PresentBitRoleParityPredictorSpec(p_topology="true")
    )
    torch.manual_seed(1000)
    wrong_model = PresentBitRoleParityPredictor(
        PresentBitRoleParityPredictorSpec(p_topology="wrong")
    )
    mlp = OutputPredictionMlp(16, 128)
    true_parameters = dict(true_model.named_parameters())
    wrong_parameters = dict(wrong_model.named_parameters())
    assert true_parameters.keys() == wrong_parameters.keys()
    assert all(
        torch.equal(true_parameters[name], wrong_parameters[name])
        for name in true_parameters
    )
    true_count = sum(parameter.numel() for parameter in true_model.parameters())
    wrong_count = sum(parameter.numel() for parameter in wrong_model.parameters())
    mlp_count = sum(parameter.numel() for parameter in mlp.parameters())
    assert true_count == wrong_count == 27003
    assert mlp_count == 26896
    assert abs(true_count - mlp_count) / mlp_count <= 0.01


def test_bit_role_contract_keeps_real_ciphertext_output_targets() -> None:
    config = _small_config()
    datasets = build_bit_role_data(config)
    checks = validate_bit_role_contract(config, datasets)

    assert all(checks.values())
    assert datasets["aligned"]["test"].parity_targets.shape == (64, 16)
    assert checks["labels_are_real_ciphertext_outputs_not_sample_classes"] is True


def test_bit_role_gate_requires_true_p_over_all_controls() -> None:
    config = _small_config()

    def row(model: str, auc: float) -> dict[str, object]:
        return {
            "model": model,
            "model_seed": 1000,
            "test_loss": 0.6,
            "test_accuracy": auc,
            "test_macro_auc": auc,
            "test_exact_match": 0.0,
        }

    rows = [
        row("aligned_parity_mlp", 0.52),
        row("bit_role_true_p", 0.62),
        row("bit_role_wrong_p", 0.51),
        row("bit_role_true_p_label_shuffle", 0.50),
    ]
    training = {
        "rows": rows,
        "history": [{} for _ in range(4)],
        "trained": {
            "bit_role_true_p_label_shuffle": {
                "test_target_identity": "true_aligned_parity_targets"
            }
        },
    }

    passed = adjudicate_bit_role_readiness(config, {"protocol": True}, training)
    rows[2]["test_macro_auc"] = 0.61
    generic = adjudicate_bit_role_readiness(config, {"protocol": True}, training)

    assert passed["status"] == "pass"
    assert passed["decision"] == (
        "innovation2_output_parity_present_r3_bit_role_attributed"
    )
    assert generic["status"] == "hold"
    assert generic["decision"] == (
        "innovation2_output_parity_present_r3_bit_role_generic_gain_only"
    )


def test_bit_role_plot_names_exact_route_and_real_output(tmp_path: Path) -> None:
    summary = {
        "gate": {
            "status": "hold",
            "decision": "innovation2_output_parity_present_r3_bit_role_not_ready",
            "metrics": {
                "mlp_accuracy": 0.51,
                "mlp_macro_auc": 0.52,
                "true_p_accuracy": 0.53,
                "true_p_macro_auc": 0.54,
                "wrong_p_accuracy": 0.50,
                "wrong_p_macro_auc": 0.50,
                "label_shuffle_accuracy": 0.50,
                "label_shuffle_macro_auc": 0.49,
                "true_minus_mlp_macro_auc": 0.02,
                "true_minus_wrong_p_macro_auc": 0.04,
                "true_minus_label_shuffle_macro_auc": 0.05,
            },
        }
    }
    output = tmp_path / "curves.svg"

    render_spn_local_readiness(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "精确bit-role路由门" in svg
    assert "真实密文输出parity" in svg
    assert "错误P层" in svg
    assert "没有真假或平衡类别" in svg
    assert "确定性依赖锥" in svg
