from __future__ import annotations

import numpy as np
import torch

from blockcipher_nd.cli.plot_runtime_spn_learned_access_k1by7 import (
    render_k1by7_svg,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_learned_access_audit_k1by7 as k1by7,
)


def test_k1by7_config_and_source_authority_are_frozen() -> None:
    config = k1by7.load_and_validate_config()

    assert config["run_id"] == k1by7.RUN_ID
    assert tuple(config["audit"]["taps"]) == k1by7.TAPS
    assert all(k1by7.source_binding_checks(config).values())


def test_k1by7_readiness_captures_all_taps_without_training() -> None:
    readiness = k1by7.build_readiness(k1by7.load_and_validate_config())

    assert readiness["status"] == "pass"
    assert readiness["execution_authorized"] is True
    assert readiness["training_authorized"] is False
    assert readiness["optimizer_steps_authorized"] == 0
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())
    assert set(readiness["evidence_metrics"]["tap_shapes"]) == set(
        k1by7.CONDITIONS
    )


def test_k1by7_hooks_do_not_change_forward_output() -> None:
    config = k1by7.load_and_validate_config()
    models, _rows = k1by7.load_models_and_source_rows(config, seed=2)
    fixture = torch.zeros(3, 2048)
    model = models["correct"]

    with torch.inference_mode():
        before = model(fixture)
    captures = k1by7.capture_taps(model, fixture)
    with torch.inference_mode():
        after = model(fixture)

    assert tuple(captures) == k1by7.TAPS
    assert torch.equal(before, after)
    assert captures["linear_histogram"].shape == (3, 2, 16, 16)
    assert captures["pre_classifier_representation"].shape == (3, 384)


def test_k1by7_mean_difference_probe_uses_balanced_disjoint_halves() -> None:
    labels = np.tile(np.asarray([0, 1, 1, 0], dtype=np.uint8), 512)
    values = np.stack((labels, 1 - labels), axis=1).astype(np.float32)

    report = k1by7.mean_difference_probe(values, labels, epsilon=1e-6)

    assert report["probe_auc"] == 1.0
    assert report["discovery_positive_rows"] == 512
    assert report["discovery_negative_rows"] == 512
    assert report["evaluation_positive_rows"] == 512
    assert report["evaluation_negative_rows"] == 512


def test_k1by7_adjudication_localizes_first_seed3_loss(monkeypatch) -> None:
    config = k1by7.load_and_validate_config()
    rows = []
    for seed in (2, 3):
        for tap_index, tap in enumerate(k1by7.TAPS):
            correct = 0.60
            affine = 0.55
            if seed == 3 and tap_index >= 2:
                affine = 0.61
            for condition, auc in (
                ("correct", correct),
                ("affine_wrong_endpoint", affine),
            ):
                rows.append(_probe_row(seed, condition, tap, tap_index, auc))
    replay = {
        str(seed): {
            "correct": {
                "source_auc": 0.68,
                "replayed_auc": 0.68,
                "absolute_error": 0.0,
            },
            "affine_wrong_endpoint": {
                "source_auc": 0.64 if seed == 2 else 0.69,
                "replayed_auc": 0.64 if seed == 2 else 0.69,
                "absolute_error": 0.0,
            },
        }
        for seed in (2, 3)
    }
    monkeypatch.setattr(k1by7, "source_binding_checks", lambda _config: {"x": True})
    readiness = {
        "status": "pass",
        "execution_authorized": True,
        "training_authorized": False,
        "optimizer_steps_authorized": 0,
    }

    gate = k1by7.adjudicate(
        config,
        result_rows=rows,
        replay=replay,
        readiness=readiness,
        sources_unchanged=True,
    )

    assert gate["status"] == "pass"
    assert gate["method_status"] == "hold"
    assert gate["first_loss_by_seed"]["3"] == "cell_fusion"
    assert (
        gate["decision"]
        == "innovation1_runtime_spn_k1by7_first_loss_cell_fusion_identified"
    )


def test_k1by7_plot_uses_plain_language_tap_labels(tmp_path) -> None:
    gate = {
        "run_id": k1by7.RUN_ID,
        "status": "pass",
        "decision": "innovation1_runtime_spn_k1by7_first_loss_cell_fusion_identified",
        "first_loss_by_seed": {"2": None, "3": "cell_fusion"},
        "seed_results": {
            str(seed): {
                "taps": {
                    tap: {
                        "correct_probe_auc": 0.60,
                        "affine_probe_auc": 0.55 if seed == 2 else 0.61,
                        "correct_minus_affine_probe_auc": (
                            0.05 if seed == 2 else -0.01
                        ),
                        "margin_pass": seed == 2,
                    }
                    for tap in k1by7.TAPS
                },
                "source_logit_margin": 0.04 if seed == 2 else -0.01,
                "first_margin_loss": None if seed == 2 else "cell_fusion",
            }
            for seed in (2, 3)
        },
    }
    output = tmp_path / "curves.svg"

    report = render_k1by7_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["status"] == "pass"
    assert "正确 P 层语义在哪一步丢失" in svg
    assert "线性直方图" in svg
    assert "阶段池化摘要" in svg
    assert "单元融合" in svg


def _probe_row(
    seed: int,
    condition: str,
    tap: str,
    tap_index: int,
    auc: float,
) -> dict:
    return {
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
