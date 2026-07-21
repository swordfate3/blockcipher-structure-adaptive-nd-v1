from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from blockcipher_nd.cli.plot_innovation2_selected_output_spn_rescnn_hybrid import (
    render_spn_rescnn_hybrid,
)
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    SelectedOutputSpnResidualCnn,
    _present_topology_mapping,
)
from blockcipher_nd.tasks.innovation2.selected_output_spn_rescnn_hybrid import (
    MODEL_SPECS,
    OPB1_RELEASE_DECISION,
    OPB1_RUN_ID,
    SpnResCnnHybridConfig,
    adjudicate_hybrid,
    authorize_from_opb1_gate,
    hybrid_parameter_counts,
    prepare_hybrid_data,
    train_hybrid_matrix,
    validate_hybrid_contract,
)


def _tiny_config() -> SpnResCnnHybridConfig:
    return SpnResCnnHybridConfig(
        train_rows=8,
        test_rows=8,
        rescnn_channels=4,
        epochs=1,
        batch_size=4,
        data_chunk_rows=4,
    )


def _opb1_negative_gate() -> dict[str, object]:
    return {
        "run_id": OPB1_RUN_ID,
        "status": "hold",
        "decision": OPB1_RELEASE_DECISION,
        "protocol_checks": {"valid": True},
        "execution_checks": {"valid": True},
        "metrics": {"attribution_passed": False},
    }


def test_hybrid_has_three_stages_and_eight_outputs() -> None:
    model = SelectedOutputSpnResidualCnn(
        channels=4,
        stage_blocks=(1, 1, 1),
        source_for_destination=_present_topology_mapping("exact"),
    )

    assert len(model.stages) == 3
    assert model(torch.zeros((2, 64))).shape == (2, 8)


def test_hybrid_variants_match_anchor_parameter_count_and_initialization() -> None:
    config = SpnResCnnHybridConfig()
    counts = hybrid_parameter_counts(config)
    assert (
        counts["rescnn"] == counts["spn_rescnn_exact_p"] == counts["spn_rescnn_wrong_p"]
    )

    models = []
    for mode in ("exact", "wrong"):
        torch.manual_seed(20260722)
        models.append(
            SelectedOutputSpnResidualCnn(
                channels=4,
                stage_blocks=(1, 1, 1),
                source_for_destination=_present_topology_mapping(mode),
            )
        )
    assert all(
        torch.equal(dict(models[0].named_parameters())[name], parameter)
        for name, parameter in models[1].named_parameters()
    )


def test_formal_mode_requires_valid_opb1_non_attribution() -> None:
    authorize_from_opb1_gate(_opb1_negative_gate())
    invalid = _opb1_negative_gate()
    invalid["decision"] = (
        "innovation2_topology_bottleneck_ready_for_independent_confirmation"
    )
    with pytest.raises(
        ValueError, match="requires OPB1 topology-bottleneck non-attribution"
    ):
        authorize_from_opb1_gate(invalid)


def test_tiny_hybrid_matrix_is_complete_and_replayable(tmp_path: Path) -> None:
    config = _tiny_config()
    data = prepare_hybrid_data(config, tmp_path)
    checks = validate_hybrid_contract(config, data)
    training = train_hybrid_matrix(config, data, tmp_path)
    gate = adjudicate_hybrid(config, checks, training)

    assert all(checks.values())
    assert len(training["rows"]) == 32
    assert len(training["history"]) == 4
    assert len(training["checkpoints"]) == 4
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_spn_rescnn_hybrid_local_smoke_passed"


def test_formal_gate_requires_anchor_and_control_gains() -> None:
    config = SpnResCnnHybridConfig.formal(device="cpu")
    aucs = {
        MODEL_SPECS[0][0]: 0.60,
        MODEL_SPECS[1][0]: 0.70,
        MODEL_SPECS[2][0]: 0.60,
        MODEL_SPECS[3][0]: 0.50,
    }
    rows = [
        {
            "model": model,
            "msb_index": bit,
            "threshold_accuracy": 0.70,
            "majority_accuracy": 0.50,
            "accuracy_minus_majority": 0.20,
            "auc": aucs[model],
            "mse": 0.20,
        }
        for model, _, _ in MODEL_SPECS
        for bit in config.selected_msb_indices
    ]
    training = {
        "rows": rows,
        "summaries": [{"model": model} for model in aucs],
        "history": [
            {"model": model, "epoch": epoch}
            for model in aucs
            for epoch in range(1, config.epochs + 1)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in range(4)],
    }

    gate = adjudicate_hybrid(config, {"valid": True}, training)

    assert gate["status"] == "pass"
    assert (
        gate["decision"]
        == "innovation2_spn_rescnn_hybrid_candidate_requires_confirmation"
    )


def test_hybrid_plot_has_plain_chinese_scope(tmp_path: Path) -> None:
    config = _tiny_config()
    rows = [
        {"model": model, "msb_index": bit, "auc": 0.50 + 0.01 * index}
        for index, (model, _, _) in enumerate(MODEL_SPECS)
        for bit in config.selected_msb_indices
    ]
    summary = {
        "metadata": {
            "mode": "smoke",
            "selected_msb_indices": list(config.selected_msb_indices),
        },
        "bit_rows": rows,
        "gate": {"decision": "innovation2_spn_rescnn_hybrid_local_smoke_passed"},
    }
    output = tmp_path / "curves.svg"

    render_spn_rescnn_hybrid(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "SPN-ResCNN" in svg
    assert "真实密文输出bit" in svg
    assert "不是四轮" in svg


def test_training_cli_import_does_not_require_matplotlib() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ModuleNotFoundError('matplotlib intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import blockcipher_nd.cli.run_innovation2_selected_output_spn_rescnn_hybrid
print('import=pass')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import=pass"


def test_result_index_names_opc1_in_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_smoke_seed6_20260722"
    )
    assert name == "创新2 OPC1：PRESENT三轮SPN-ResCNN混合真实密文输出预测"
