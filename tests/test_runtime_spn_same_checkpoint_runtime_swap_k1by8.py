from __future__ import annotations

import torch

from blockcipher_nd.cli.plot_runtime_spn_same_checkpoint_runtime_swap_k1by8 import (
    render_k1by8_svg,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_same_checkpoint_runtime_swap_k1by8 as k1by8,
)


def test_k1by8_config_and_source_authority_are_frozen() -> None:
    config = k1by8.load_and_validate_config()

    assert config["run_id"] == k1by8.RUN_ID
    assert tuple(config["audit"]["weight_sources"]) == k1by8.WEIGHT_SOURCES
    assert tuple(config["audit"]["runtime_programs"]) == k1by8.RUNTIME_PROGRAMS
    assert all(k1by8.source_binding_checks(config).values())


def test_k1by8_parameter_only_swap_preserves_target_runtime() -> None:
    config = k1by8.load_and_validate_config()
    models, _rows, metadata = k1by8.build_swapped_models(config, seed=2)

    assert set(models) == k1by8.expected_conditions()
    parameters = metadata["parameter_fingerprints"]
    runtimes = metadata["runtime_fingerprints"]
    assert (
        parameters["correct_weights__correct_runtime"]
        == parameters["correct_weights__affine_runtime"]
    )
    assert (
        parameters["affine_weights__correct_runtime"]
        == parameters["affine_weights__affine_runtime"]
    )
    assert (
        runtimes["correct_weights__correct_runtime"]
        == runtimes["affine_weights__correct_runtime"]
    )
    assert (
        runtimes["correct_weights__affine_runtime"]
        == runtimes["affine_weights__affine_runtime"]
    )
    assert (
        runtimes["correct_weights__correct_runtime"]
        != runtimes["correct_weights__affine_runtime"]
    )


def test_k1by8_readiness_proves_zero_training_swap_geometry() -> None:
    readiness = k1by8.build_readiness(k1by8.load_and_validate_config())

    assert readiness["status"] == "pass"
    assert readiness["execution_authorized"] is True
    assert readiness["training_authorized"] is False
    assert readiness["optimizer_steps_authorized"] == 0
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())


def test_k1by8_diagonal_models_replay_source_outputs() -> None:
    config = k1by8.load_and_validate_config()
    models, _rows, _metadata = k1by8.build_swapped_models(config, seed=2)
    fixture = torch.zeros(3, 2048)

    with torch.inference_mode():
        correct = models["correct_weights__correct_runtime"](fixture)
        affine = models["affine_weights__affine_runtime"](fixture)

    assert torch.equal(
        correct,
        k1by8.source_forward(
            config,
            seed=2,
            condition="correct",
            features=fixture,
        ),
    )
    assert torch.equal(
        affine,
        k1by8.source_forward(
            config,
            seed=2,
            condition="affine_wrong_endpoint",
            features=fixture,
        ),
    )


def test_k1by8_adjudication_routes_histogram_failure(monkeypatch) -> None:
    config = k1by8.load_and_validate_config()
    rows = []
    for seed in (2, 3):
        for weight_source in k1by8.WEIGHT_SOURCES:
            for runtime_program, auc in (
                ("correct_runtime", 0.60),
                ("affine_runtime", 0.55),
            ):
                for tap_index, tap in enumerate(k1by8.TAPS):
                    value = auc
                    if (
                        seed == 3
                        and weight_source == "correct_weights"
                        and tap == "linear_histogram"
                        and runtime_program == "affine_runtime"
                    ):
                        value = 0.61
                    rows.append(
                        _probe_row(
                            seed,
                            weight_source,
                            runtime_program,
                            tap,
                            tap_index,
                            value,
                        )
                    )
    final = {
        str(seed): {
            "condition_auc": {
                condition: 0.60 if condition.endswith("correct_runtime") else 0.55
                for condition in k1by8.expected_conditions()
            },
            "diagonal_replay": {
                "correct_weights__correct_runtime": {
                    "source_auc": 0.60,
                    "replayed_auc": 0.60,
                    "absolute_error": 0.0,
                },
                "affine_weights__affine_runtime": {
                    "source_auc": 0.55,
                    "replayed_auc": 0.55,
                    "absolute_error": 0.0,
                },
            },
        }
        for seed in (2, 3)
    }
    monkeypatch.setattr(k1by8, "source_binding_checks", lambda _config: {"x": True})
    readiness = {
        "status": "pass",
        "execution_authorized": True,
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
    }

    gate = k1by8.adjudicate(
        config,
        result_rows=rows,
        final_evaluation=final,
        model_metadata={"2": {}, "3": {}},
        readiness=readiness,
        sources_unchanged=True,
    )

    assert gate["status"] == "pass"
    assert gate["research_gate_passed"] is False
    assert gate["decision"].endswith("same_checkpoint_histogram_access_loss")


def test_k1by8_plot_uses_plain_language_intervention_labels(tmp_path) -> None:
    gate = _plot_gate()
    output = tmp_path / "curves.svg"

    report = render_k1by8_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 3
    assert "同一组权重只替换运行时结构" in svg
    assert "正确检查点：运行时因果差值" in svg
    assert "四格交换后的最终输出" in svg
    assert "线性直方图" in svg


def _probe_row(
    seed: int,
    weight_source: str,
    runtime_program: str,
    tap: str,
    tap_index: int,
    auc: float,
) -> dict:
    return {
        "seed": seed,
        "condition": k1by8.condition_key(weight_source, runtime_program),
        "weight_source": weight_source,
        "runtime_program": runtime_program,
        "tap": tap,
        "tap_index": tap_index,
        "probe_auc": auc,
        "discovery_positive_rows": 512,
        "discovery_negative_rows": 512,
        "evaluation_positive_rows": 512,
        "evaluation_negative_rows": 512,
    }


def _plot_gate() -> dict:
    return {
        "run_id": k1by8.RUN_ID,
        "status": "pass",
        "decision": (
            "innovation1_runtime_spn_k1by8_"
            "independent_training_variance_identified"
        ),
        "seed_results": {
            str(seed): {
                "weights": {
                    weight_source: {
                        "taps": {
                            tap: {
                                "correct_runtime_probe_auc": 0.62,
                                "affine_runtime_probe_auc": 0.57,
                                "correct_minus_affine_runtime_probe_auc": 0.05,
                                "margin_pass": True,
                            }
                            for tap in k1by8.TAPS
                        },
                        "correct_runtime_final_auc": 0.68,
                        "affine_runtime_final_auc": 0.61,
                        "correct_minus_affine_runtime_final_auc": 0.07,
                        "final_margin_pass": True,
                        "first_margin_loss": None,
                    }
                    for weight_source in k1by8.WEIGHT_SOURCES
                },
                "primary_histogram_margin_pass": True,
                "primary_final_margin_pass": True,
                "primary_gate_pass": True,
            }
            for seed in (2, 3)
        },
    }
