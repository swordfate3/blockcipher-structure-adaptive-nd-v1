from __future__ import annotations

import argparse

from blockcipher_nd.cli.plot_runtime_spn_permutation_expert_k1by3 import (
    render_k1by3_svg,
)
from blockcipher_nd.cli.run_runtime_spn_permutation_expert_k1by3 import training_argv
from blockcipher_nd.tasks.innovation1 import runtime_spn_permutation_expert_k1by3 as k1by3


def test_k1by3_plan_and_source_evidence_are_frozen() -> None:
    tasks = k1by3.read_tasks()

    assert len(tasks) == 6
    assert k1by3.candidate_protocol_frozen(tasks)
    assert set(k1by3.task_map(tasks)) == k1by3.expected_keys()
    assert all(k1by3.source_binding_checks().values())
    assert "wrong_order_routing" not in k1by3.CONDITIONS


def test_k1by3_readiness_routes_only_through_permutation_expert() -> None:
    readiness = k1by3.build_readiness(tasks=k1by3.read_tasks())

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
    ) == {
        tuple(
            sorted(
                {
                    "sbox4_table": 32,
                    "linear_permutation": 32,
                    "linear_gf2": 0,
                }.items()
            )
        )
    }


def test_k1by3_adjudication_requires_both_seed_margins(monkeypatch) -> None:
    rows = []
    for seed in (2, 3):
        rows.extend(
            (
                _row(seed, "runtime_spn_k1by1_compiler_correct", 0.72),
                _row(seed, "runtime_spn_k1by1_compiler_wrong_binding", 0.53),
                _row(seed, "runtime_spn_k1by1_no_compiler_conditioner", 0.51),
            )
        )
    monkeypatch.setattr(k1by3, "training_protocol_frozen", lambda _rows: True)
    monkeypatch.setattr(k1by3, "cache_protocol_frozen", lambda _rows: True)
    readiness = {
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"source": True},
        "evidence_checks": {"model": True},
    }

    gate = k1by3.adjudicate(
        tasks=k1by3.read_tasks(),
        result_rows=rows,
        progress_rows=[],
        readiness=readiness,
    )

    assert gate["status"] == "pass"
    assert (
        gate["decision"]
        == "innovation1_runtime_spn_k1by3_permutation_expert_supported"
    )
    assert all(gate["research_checks"].values())
    assert gate["remote_scale"] == "no"


def test_k1by3_training_argv_keeps_disk_cache(tmp_path) -> None:
    args = argparse.Namespace(
        plan=k1by3.PLAN_PATH,
        device="cpu",
        output_root=tmp_path / "run",
    )

    argv = training_argv(args)

    assert "--dataset-cache-root" in argv
    assert "--checkpoint-output-dir" in argv
    assert "--progress-output" in argv
    assert "--device" in argv
    assert argv[argv.index("--device") + 1] == "cpu"


def test_k1by3_plot_uses_plain_language_labels(tmp_path) -> None:
    gate = {
        "run_id": k1by3.RUN_ID,
        "status": "pass",
        "decision": "innovation1_runtime_spn_k1by3_permutation_expert_supported",
        "seed_results": {
            str(seed): {
                "auc_by_condition": {
                    "correct_permutation_routing": 0.72,
                    "wrong_permutation_binding": 0.53,
                    "no_compiler_conditioner": 0.51,
                },
                "correct_minus_control": {
                    "wrong_permutation_binding": 0.19,
                    "no_compiler_conditioner": 0.21,
                },
            }
            for seed in (2, 3)
        },
    }
    output = tmp_path / "curves.svg"

    report = render_k1by3_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["status"] == "pass"
    assert "PRESENT 第7轮置换专家诊断" in svg
    assert "错误目标绑定" in svg
    assert "不使用结构条件器" in svg
    assert "通过信号和控制优势门槛" in svg


def _row(seed: int, model: str, auc: float) -> dict:
    return {
        "seed": seed,
        "model": model,
        "metrics": {"auc": auc},
    }
