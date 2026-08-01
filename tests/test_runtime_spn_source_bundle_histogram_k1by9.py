from __future__ import annotations

import torch

from blockcipher_nd.cli.plot_runtime_spn_source_bundle_histogram_k1by9 import (
    render_k1by9_svg,
)
from blockcipher_nd.models.structure.spn.ordered_primitive_conditioner import (
    LINEAR_HISTOGRAM_LOCAL,
    LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN,
    source_bundle_equivalence_matrices,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_affine_neural_attribution_k1by6 as k1by6,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_same_checkpoint_runtime_swap_k1by8 as k1by8,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_source_bundle_histogram_k1by9 as k1by9,
)


def test_k1by9_config_and_source_authority_are_frozen() -> None:
    config = k1by9.load_and_validate_config()

    assert config["run_id"] == k1by9.RUN_ID
    assert tuple(config["audit"]["representations"]) == k1by9.REPRESENTATIONS
    assert tuple(config["audit"]["runtime_programs"]) == k1by9.RUNTIME_PROGRAMS
    assert all(k1by9.source_binding_checks(config).values())


def test_source_bundle_matrix_is_relabel_invariant_equivalence_mean() -> None:
    task = k1by6.task_map(k1by6.read_tasks())[2]
    candidate_task = dict(task)
    candidate_task["model_options"] = {
        **dict(task["model_options"]),
        "linear_histogram_mode": LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN,
    }
    correct = k1by6.build_model_for_task(
        candidate_task,
        model_key=k1by6.CORRECT_MODEL,
    )
    affine = k1by6.build_model_for_task(
        candidate_task,
        model_key=k1by6.AFFINE_MODEL,
    )
    correct_matrix = correct.conditioner.linear_source_bundle_equivalence
    affine_matrix = affine.conditioner.linear_source_bundle_equivalence

    assert all(k1by9.equivalence_matrix_checks(correct_matrix).values())
    assert all(k1by9.equivalence_matrix_checks(affine_matrix).values())
    assert not torch.equal(correct_matrix, affine_matrix)
    assert torch.equal(
        correct_matrix,
        source_bundle_equivalence_matrices(
            k1by9.rename_program_source_cells(correct.conditioner.program)
        ),
    )


def test_candidate_adds_no_parameter_and_only_one_runtime_buffer() -> None:
    config = k1by9.load_and_validate_config()
    models, _source_row, metadata = k1by9.build_models(config, seed=2)

    assert set(metadata["parameter_fingerprints"].values()) == {
        metadata["source_parameter_fingerprint"]
    }
    for condition, model in models.items():
        representation, _runtime = k1by9.split_condition_key(condition)
        expected_mode = (
            LINEAR_HISTOGRAM_LOCAL
            if representation == "anchor_local"
            else LINEAR_HISTOGRAM_SOURCE_BUNDLE_MEAN
        )
        assert model.linear_histogram_mode == expected_mode
        expected_buffers = (
            k1by9.ANCHOR_BUFFER_NAMES
            if representation == "anchor_local"
            else k1by9.CANDIDATE_BUFFER_NAMES
        )
        assert set(dict(model.named_buffers())) == expected_buffers


def test_k1by9_readiness_proves_zero_training_representation_gate() -> None:
    readiness = k1by9.build_readiness(k1by9.load_and_validate_config())

    assert readiness["status"] == "pass", readiness
    assert readiness["execution_authorized"] is True
    assert readiness["training_authorized"] is False
    assert readiness["optimizer_steps_authorized"] == 0
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())


def test_k1by9_anchor_models_replay_k1by8_fixture_outputs() -> None:
    config = k1by9.load_and_validate_config()
    models, _source_row, _metadata = k1by9.build_models(config, seed=2)
    reference, _rows, _reference_metadata = k1by8.build_swapped_models(
        k1by8.load_and_validate_config(),
        seed=2,
    )
    fixture = torch.zeros(3, 2048)

    with torch.inference_mode():
        for runtime in k1by9.RUNTIME_PROGRAMS:
            assert torch.equal(
                models[k1by9.condition_key("anchor_local", runtime)](fixture),
                reference[k1by8.condition_key("correct_weights", runtime)](fixture),
            )


def test_k1by9_adjudication_routes_candidate_tap_failure(monkeypatch) -> None:
    config = k1by9.load_and_validate_config()
    rows = _result_rows(candidate_seed3_linear_failure=True)
    final = _final_evaluation()
    monkeypatch.setattr(k1by9, "source_binding_checks", lambda _config: {"x": True})
    monkeypatch.setattr(k1by9, "model_metadata_frozen", lambda _values: True)

    gate = k1by9.adjudicate(
        config,
        result_rows=rows,
        final_evaluation=final,
        model_metadata={"2": {}, "3": {}},
        readiness=_readiness(),
        sources_unchanged=True,
    )

    assert gate["status"] == "pass"
    assert gate["research_gate_passed"] is False
    assert gate["decision"].endswith("source_bundle_histogram_repair_not_supported")


def test_k1by9_adjudication_supports_complete_candidate_gate(monkeypatch) -> None:
    config = k1by9.load_and_validate_config()
    monkeypatch.setattr(k1by9, "source_binding_checks", lambda _config: {"x": True})
    monkeypatch.setattr(k1by9, "model_metadata_frozen", lambda _values: True)

    gate = k1by9.adjudicate(
        config,
        result_rows=_result_rows(candidate_seed3_linear_failure=False),
        final_evaluation=_final_evaluation(),
        model_metadata={"2": {}, "3": {}},
        readiness=_readiness(),
        sources_unchanged=True,
    )

    assert gate["status"] == "pass"
    assert gate["research_gate_passed"] is True
    assert gate["decision"].endswith("source_bundle_histogram_repair_supported")


def test_k1by9_plot_uses_plain_language_representation_labels(tmp_path) -> None:
    output = tmp_path / "curves.svg"

    report = render_k1by9_svg(_plot_gate(), output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 3
    assert "线性直方图加入相对源单元组上下文" in svg
    assert "旧表示：精确重放锚点" in svg
    assert "候选表示：源单元组均值" in svg
    assert "不调融合权重、不扩样" in svg


def _result_rows(*, candidate_seed3_linear_failure: bool) -> list[dict]:
    rows = []
    for seed in k1by9.EXPECTED_SEEDS:
        for representation in k1by9.REPRESENTATIONS:
            for runtime, auc in (("correct_runtime", 0.62), ("affine_runtime", 0.60)):
                for tap_index, tap in enumerate(k1by9.TAPS):
                    value = auc
                    if (
                        candidate_seed3_linear_failure
                        and seed == 3
                        and representation == "candidate_source_bundle_mean"
                        and runtime == "affine_runtime"
                        and tap == "linear_histogram"
                    ):
                        value = 0.63
                    rows.append(
                        {
                            "seed": seed,
                            "representation": representation,
                            "runtime_program": runtime,
                            "tap": tap,
                            "tap_index": tap_index,
                            "probe_auc": value,
                            "discovery_positive_rows": 512,
                            "discovery_negative_rows": 512,
                            "evaluation_positive_rows": 512,
                            "evaluation_negative_rows": 512,
                        }
                    )
    return rows


def _final_evaluation() -> dict:
    values = {}
    for seed in k1by9.EXPECTED_SEEDS:
        condition_auc = {
            k1by9.condition_key("anchor_local", "correct_runtime"): 0.680,
            k1by9.condition_key("anchor_local", "affine_runtime"): 0.660,
            k1by9.condition_key(
                "candidate_source_bundle_mean", "correct_runtime"
            ): 0.679,
            k1by9.condition_key(
                "candidate_source_bundle_mean", "affine_runtime"
            ): 0.659,
        }
        values[str(seed)] = {
            "condition_auc": condition_auc,
            "anchor_replay": {
                condition: {
                    "source_auc": auc,
                    "replayed_auc": auc,
                    "absolute_error": 0.0,
                }
                for condition, auc in condition_auc.items()
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


def _plot_gate() -> dict:
    return {
        "run_id": k1by9.RUN_ID,
        "status": "pass",
        "research_gate_passed": False,
        "decision": (
            "innovation1_runtime_spn_k1by9_source_bundle_histogram_repair_not_supported"
        ),
        "seed_results": {
            str(seed): {
                "representations": {
                    representation: {
                        "taps": {
                            tap: {
                                "correct_runtime_probe_auc": 0.62,
                                "affine_runtime_probe_auc": 0.60,
                                "correct_minus_affine_runtime_probe_auc": 0.02,
                                "margin_pass": True,
                            }
                            for tap in k1by9.TAPS
                        },
                        "correct_runtime_final_auc": 0.68,
                        "affine_runtime_final_auc": 0.66,
                        "correct_minus_affine_runtime_final_auc": 0.02,
                        "final_margin_pass": True,
                        "first_margin_loss": None,
                    }
                    for representation in k1by9.REPRESENTATIONS
                },
                "candidate_correct_final_minus_anchor_correct_final_auc": 0.0,
                "candidate_all_tap_margins_pass": True,
                "candidate_final_margin_pass": True,
                "candidate_retention_pass": True,
                "primary_gate_pass": True,
            }
            for seed in k1by9.EXPECTED_SEEDS
        },
    }
