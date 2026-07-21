from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.cli.plot_innovation2_output_parity_spn_local import (
    render_spn_local_readiness,
)
from blockcipher_nd.models.structure.spn.present_output_parity_predictor import (
    PresentOutputParityPredictor,
    PresentOutputParityPredictorSpec,
    msb_token_logits_to_lsb_outputs,
    plaintext_bits_to_msb_nibble_tokens,
)
from blockcipher_nd.tasks.innovation2.output_parity_prediction import (
    OutputParityPredictionConfig,
    OutputPredictionMlp,
)
from blockcipher_nd.tasks.innovation2.output_parity_spn_local import (
    adjudicate_spn_local_readiness,
    build_spn_local_data,
    validate_spn_local_contract,
)


def _small_config() -> OutputParityPredictionConfig:
    return OutputParityPredictionConfig(
        run_id="test-output-parity-spn-local",
        rounds=3,
        seed=0,
        train_rows=128,
        validation_rows=64,
        test_rows=64,
        hidden_dim=128,
        epochs=1,
        batch_size=32,
    )


def test_present_output_parity_predictor_preserves_lsb_output_contract() -> None:
    features = torch.arange(64, dtype=torch.float32).reshape(1, 64)
    tokens = plaintext_bits_to_msb_nibble_tokens(features)
    logits = torch.arange(16, dtype=torch.float32).reshape(1, 16)

    assert torch.equal(tokens[0, 0], torch.tensor([60.0, 61.0, 62.0, 63.0]))
    assert torch.equal(tokens[0, -1], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(
        msb_token_logits_to_lsb_outputs(logits),
        torch.arange(15, -1, -1, dtype=torch.float32).reshape(1, 16),
    )
    model = PresentOutputParityPredictor(PresentOutputParityPredictorSpec())
    assert model(torch.zeros(2, 64)).shape == (2, 16)


def test_spn_local_true_and_shuffled_are_parameter_matched() -> None:
    torch.manual_seed(1000)
    true_model = PresentOutputParityPredictor(
        PresentOutputParityPredictorSpec(p_topology="true")
    )
    torch.manual_seed(1000)
    shuffled_model = PresentOutputParityPredictor(
        PresentOutputParityPredictorSpec(p_topology="shuffled")
    )
    mlp = OutputPredictionMlp(16, 128)

    true_parameters = dict(true_model.named_parameters())
    shuffled_parameters = dict(shuffled_model.named_parameters())
    assert true_parameters.keys() == shuffled_parameters.keys()
    assert all(
        torch.equal(true_parameters[name], shuffled_parameters[name])
        for name in true_parameters
    )
    true_count = sum(parameter.numel() for parameter in true_model.parameters())
    shuffled_count = sum(parameter.numel() for parameter in shuffled_model.parameters())
    mlp_count = sum(parameter.numel() for parameter in mlp.parameters())
    assert true_count == shuffled_count == 26881
    assert mlp_count == 26896
    assert abs(true_count - mlp_count) / mlp_count <= 0.01
    assert not torch.equal(
        true_model.mixer_blocks[0].p_sources,
        shuffled_model.mixer_blocks[0].p_sources,
    )


def test_spn_local_data_remains_real_ciphertext_output_prediction() -> None:
    config = _small_config()
    datasets = build_spn_local_data(config)
    checks = validate_spn_local_contract(config, datasets)

    assert all(checks.values())
    assert datasets["aligned"]["train"].features.shape == (128, 64)
    assert datasets["aligned"]["train"].parity_targets.shape == (128, 16)
    assert checks["labels_are_real_ciphertext_outputs_not_sample_classes"] is True


def test_spn_local_gate_requires_true_topology_attribution() -> None:
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
        row("spn_local_true_p", 0.62),
        row("spn_local_shuffled_p", 0.51),
        row("spn_local_true_p_label_shuffle", 0.50),
    ]
    training = {
        "rows": rows,
        "history": [{} for _ in range(4)],
        "trained": {
            "spn_local_true_p_label_shuffle": {
                "test_target_identity": "true_aligned_parity_targets"
            }
        },
    }

    passed = adjudicate_spn_local_readiness(config, {"protocol": True}, training)
    rows[2]["test_macro_auc"] = 0.61
    generic = adjudicate_spn_local_readiness(config, {"protocol": True}, training)

    assert passed["status"] == "pass"
    assert passed["decision"] == (
        "innovation2_output_parity_present_r3_spn_local_attributed"
    )
    assert passed["next_action"]["sample_classification"] is False
    assert generic["status"] == "hold"
    assert generic["decision"] == (
        "innovation2_output_parity_present_r3_spn_local_generic_gain_only"
    )


def test_spn_local_plot_explains_output_and_topology_controls(
    tmp_path: Path,
) -> None:
    summary = {
        "gate": {
            "status": "pass",
            "decision": "innovation2_output_parity_present_r3_spn_local_attributed",
            "metrics": {
                "mlp_accuracy": 0.51,
                "mlp_macro_auc": 0.52,
                "true_p_accuracy": 0.60,
                "true_p_macro_auc": 0.62,
                "shuffled_p_accuracy": 0.50,
                "shuffled_p_macro_auc": 0.51,
                "label_shuffle_accuracy": 0.50,
                "label_shuffle_macro_auc": 0.49,
                "true_minus_mlp_macro_auc": 0.10,
                "true_minus_shuffled_p_macro_auc": 0.11,
                "true_minus_label_shuffle_macro_auc": 0.13,
            },
        }
    }
    output = tmp_path / "curves.svg"

    render_spn_local_readiness(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "真实密文输出parity" in svg
    assert "错误P层" in svg
    assert "没有真假或平衡类别" in svg
    assert "不是高轮攻击" in svg
