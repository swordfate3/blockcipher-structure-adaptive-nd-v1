from __future__ import annotations

import argparse

from blockcipher_nd.cli.plot_runtime_spn_affine_neural_attribution_k1by6 import (
    render_k1by6_svg,
)
from blockcipher_nd.cli.run_runtime_spn_affine_neural_attribution_k1by6 import (
    training_argv,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_affine_neural_attribution_k1by6 as k1by6,
)


def test_k1by6_plan_sources_and_cache_are_frozen() -> None:
    tasks = k1by6.read_tasks()

    assert len(tasks) == 2
    assert k1by6.candidate_protocol_frozen(tasks)
    assert set(k1by6.task_map(tasks)) == {2, 3}
    assert all(k1by6.source_binding_checks().values())
    assert k1by6.historical_anchors() == k1by6.EXPECTED_ANCHORS


def test_k1by6_readiness_builds_identifiable_equal_geometry_control() -> None:
    readiness = k1by6.build_readiness(
        tasks=k1by6.read_tasks(),
        selected_device="cpu",
    )

    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())
    assert set(readiness["evidence_metrics"]["parameter_counts"].values()) == {
        235780
    }
    assert set(
        tuple(sorted(usage.items()))
        for usage in readiness["evidence_metrics"][
            "compiled_program_expert_usage"
        ].values()
    ) == {tuple(sorted(k1by6.EXPECTED_USAGE.items()))}
    assert (
        readiness["evidence_metrics"]["affine_control_mode"]
        == "source_endpoint_affine_m5_b1_mod64"
    )


def test_k1by6_adjudication_requires_each_seed_margin(monkeypatch) -> None:
    rows = [
        _row(2, k1by6.EXPECTED_ANCHORS[2]["correct_auc"] - 0.006),
        _row(3, k1by6.EXPECTED_ANCHORS[3]["correct_auc"] - 0.004),
    ]
    monkeypatch.setattr(k1by6, "training_protocol_frozen", lambda _rows: True)
    monkeypatch.setattr(k1by6, "cache_protocol_frozen", lambda _rows: True)
    readiness = {
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"source": True},
        "evidence_checks": {"model": True},
    }

    gate = k1by6.adjudicate(
        tasks=k1by6.read_tasks(),
        result_rows=rows,
        progress_rows=[],
        readiness=readiness,
        cache_unchanged=True,
    )

    assert gate["status"] == "hold"
    assert (
        gate["decision"]
        == "innovation1_runtime_spn_k1by6_permutation_attribution_not_supported"
    )
    assert gate["research_checks"]["seed2_correct_minus_affine_margin"] is True
    assert gate["research_checks"]["seed3_correct_minus_affine_margin"] is False
    assert gate["historical_anchor_rows_retrained"] == 0


def test_k1by6_adjudication_passes_only_when_both_margins_pass(monkeypatch) -> None:
    rows = [
        _row(seed, k1by6.EXPECTED_ANCHORS[seed]["correct_auc"] - 0.006)
        for seed in (2, 3)
    ]
    monkeypatch.setattr(k1by6, "training_protocol_frozen", lambda _rows: True)
    monkeypatch.setattr(k1by6, "cache_protocol_frozen", lambda _rows: True)
    readiness = {
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"source": True},
        "evidence_checks": {"model": True},
    }

    gate = k1by6.adjudicate(
        tasks=k1by6.read_tasks(),
        result_rows=rows,
        progress_rows=[],
        readiness=readiness,
        cache_unchanged=True,
    )

    assert gate["status"] == "pass"
    assert (
        gate["decision"]
        == "innovation1_runtime_spn_k1by6_permutation_attribution_supported"
    )
    assert all(gate["research_checks"].values())


def test_k1by6_training_argv_reuses_frozen_k1by3_cache(tmp_path) -> None:
    args = argparse.Namespace(
        plan=k1by6.PLAN_PATH,
        device="cpu",
        output_root=tmp_path / "run",
    )

    argv = training_argv(args)

    assert argv[argv.index("--dataset-cache-root") + 1] == str(
        k1by6.K1BY3_CACHE_ROOT
    )
    assert argv[argv.index("--device") + 1] == "cpu"


def test_k1by6_plot_uses_plain_language_labels(tmp_path) -> None:
    gate = {
        "run_id": k1by6.RUN_ID,
        "status": "pass",
        "decision": "innovation1_runtime_spn_k1by6_permutation_attribution_supported",
        "seed_results": {
            str(seed): {
                **k1by6.EXPECTED_ANCHORS[seed],
                "affine_wrong_endpoint_auc": 0.55,
                "affine_wrong_endpoint_accuracy": 0.53,
                "correct_minus_affine_auc": (
                    k1by6.EXPECTED_ANCHORS[seed]["correct_auc"] - 0.55
                ),
                "correct_minus_no_conditioner_auc": (
                    k1by6.EXPECTED_ANCHORS[seed]["correct_auc"]
                    - k1by6.EXPECTED_ANCHORS[seed]["no_conditioner_auc"]
                ),
            }
            for seed in (2, 3)
        },
    }
    output = tmp_path / "curves.svg"

    report = render_k1by6_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["status"] == "pass"
    assert "PRESENT 正确扩散结构神经归因" in svg
    assert "仿射错误端点" in svg
    assert "不使用结构条件器" in svg
    assert "同预算 GIFT 验证" in svg


def _row(seed: int, auc: float) -> dict:
    return {
        "seed": seed,
        "model": k1by6.AFFINE_MODEL,
        "metrics": {"auc": auc, "accuracy": 0.53},
    }
