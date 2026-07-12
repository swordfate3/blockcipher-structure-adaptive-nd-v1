from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import test_invp_state_matrix_conv2d_gate as conv_fixtures
from blockcipher_nd.planning.four_role_attribution_gate import FourRoleGateSpec
from blockcipher_nd.planning.invp_state_matrix_conv2d_gate import (
    gate_invp_state_matrix_conv2d,
)
from blockcipher_nd.planning.invp_topology_residual_gate import (
    TOPOLOGY_RESIDUAL_MODEL_ROLES,
    gate_invp_topology_residual,
)


H1_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_nibble_invp_topology_residual_spn_only",
    "shuffled_p": "present_nibble_shuffled_p_topology_residual_spn_only",
    "delta_only": "present_nibble_delta_topology_residual_spn_only",
}
CONV_MODELS_BY_ROLE = {
    "anchor": conv_fixtures.ANCHOR,
    "candidate": conv_fixtures.CANDIDATE,
    "shuffled_p": conv_fixtures.SHUFFLED,
    "delta_only": conv_fixtures.DELTA,
}
HYBRID_OPTIONS = {
    "spn_mixer_depth": 2,
    "token_mlp_ratio": 2,
    "local_channels": 16,
    "local_depth": 1,
    "local_kernel_size": 3,
    "local_residual_scale_init": 0.1,
    "activation": "relu",
    "norm": "layernorm",
    "local_norm": "batchnorm2d",
    "dropout": 0.0,
}


def _unused_decision(*args: Any, **kwargs: Any) -> tuple[str, str]:
    return "unused", "unused"


def _unused_stopped_actions(decision: str) -> list[dict[str, str]]:
    return []


def test_four_role_gate_spec_recursively_copies_and_freezes_inputs() -> None:
    roles = {"anchor": "a", "candidate": "b"}
    anchor_options = {"nested": {"values": [1, 2]}}
    hybrid_options = {"layers": [3, {"width": 4}]}
    semantic_checks = {
        "backbone": {"roles": ["candidate", "control"]},
        "shape": [4, 16],
    }
    spec = FourRoleGateSpec(
        model_roles=roles,
        anchor_options=anchor_options,
        hybrid_options=hybrid_options,
        capacity_label="test",
        semantic_checks=semantic_checks,
        allowed_seed_layouts=((0,), (0, 1)),
        readiness_next_action="next",
        claim_label="claim",
        decide=_unused_decision,
        stopped_actions=_unused_stopped_actions,
    )

    roles["anchor"] = "changed"
    anchor_options["nested"]["values"].append(9)
    hybrid_options["layers"][1]["width"] = 99
    semantic_checks["backbone"]["roles"].append("changed")
    semantic_checks["shape"][0] = 99

    assert spec.model_roles["anchor"] == "a"
    assert spec.anchor_options["nested"]["values"] == (1, 2)
    assert spec.hybrid_options["layers"] == (3, {"width": 4})
    assert spec.semantic_checks["backbone"]["roles"] == ("candidate", "control")
    assert spec.semantic_checks["shape"] == (4, 16)
    with pytest.raises(TypeError):
        spec.model_roles["anchor"] = "mutated"
    with pytest.raises(TypeError):
        spec.anchor_options["nested"]["other"] = 1
    with pytest.raises(TypeError):
        spec.semantic_checks["backbone"]["roles"][0] = "mutated"


def test_shared_gate_core_is_domain_neutral() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "src/blockcipher_nd/planning/four_role_attribution_gate.py"
    ).read_text(encoding="utf-8")

    assert "state_matrix_conv2d" not in source
    assert "topology_residual" not in source
    assert "present_nibble_" not in source
    assert "stop_conv2d_route" not in source
    assert "stop_topology_residual" not in source


INVALID_THRESHOLD_CASES = [
    ("nan", float("nan")),
    ("positive_inf", float("inf")),
    ("negative_inf", float("-inf")),
    ("negative", -1),
    ("bool", True),
    ("string", "bad"),
]
PUBLIC_THRESHOLD_FIELDS = [
    (
        "conv2d",
        gate_invp_state_matrix_conv2d,
        (
            "seed0_topology_margin",
            "seed0_representation_margin",
            "joint_architecture_margin",
            "joint_control_margin",
        ),
    ),
    (
        "topology_residual",
        gate_invp_topology_residual,
        (
            "seed0_architecture_margin",
            "seed0_topology_margin",
            "seed0_representation_margin",
            "joint_architecture_margin",
            "joint_control_margin",
        ),
    ),
]
INVALID_PUBLIC_THRESHOLDS = [
    (gate_name, gate, field, value_name, value)
    for gate_name, gate, fields in PUBLIC_THRESHOLD_FIELDS
    for field in fields
    for value_name, value in INVALID_THRESHOLD_CASES
]


@pytest.mark.parametrize(
    ("gate_name", "gate", "field", "value_name", "value"),
    INVALID_PUBLIC_THRESHOLDS,
)
def test_public_gates_reject_invalid_thresholds_before_artifact_reads(
    tmp_path: Path,
    gate_name: str,
    gate: Any,
    field: str,
    value_name: str,
    value: Any,
) -> None:
    report = gate(
        [tmp_path / "missing-results.jsonl"],
        progress_paths=[tmp_path / "missing-progress.jsonl"],
        **{field: value},
    )

    assert report["decision"] == "invalid_protocol"
    assert report["research_decision_applied"] is False
    assert any(field in error for error in report["errors"])
    assert not any("read_error" in error for error in report["errors"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "expected_seeds",
    [(1,), (1, 0), (-1,), (0, 2)],
)
@pytest.mark.parametrize(
    "gate",
    [gate_invp_state_matrix_conv2d, gate_invp_topology_residual],
)
def test_public_gates_reject_nonfrozen_seed_layout_before_artifact_reads(
    tmp_path: Path,
    gate: Any,
    expected_seeds: tuple[int, ...],
) -> None:
    report = gate(
        [tmp_path / "missing-results.jsonl"],
        progress_paths=[tmp_path / "missing-progress.jsonl"],
        expected_seeds=expected_seeds,
    )

    assert report["decision"] == "invalid_protocol"
    assert report["research_decision_applied"] is False
    assert any("allowed_seed_layouts" in error for error in report["errors"])
    assert not any("read_error" in error for error in report["errors"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _progress_path(results_path: Path) -> Path:
    return results_path.with_name(f"{results_path.stem}.progress.jsonl")


def _write_h1_run(
    results_path: Path,
    aucs: dict[str, float],
    *,
    seed: int = 0,
    samples_per_class: int = 8192,
    epochs: int = 10,
) -> None:
    rows = []
    for role, model in H1_ROLES.items():
        row = conv_fixtures._row(
            CONV_MODELS_BY_ROLE[role],
            aucs[role],
            seed=seed,
            parameter_count=1100,
            samples_per_class=samples_per_class,
            epochs=epochs,
        )
        row["model"] = model
        row["selected_model"] = model
        if role != "anchor":
            row["training"]["model_options"] = dict(HYBRID_OPTIONS)
        rows.append(row)
    conv_fixtures._write_rows(results_path, rows)

    progress_path = _progress_path(results_path)
    progress_rows = _read_jsonl(progress_path)
    role_by_conv_model = {model: role for role, model in CONV_MODELS_BY_ROLE.items()}
    for row in progress_rows:
        if row.get("model") in role_by_conv_model:
            row["model"] = H1_ROLES[role_by_conv_model[row["model"]]]
    _write_jsonl(progress_path, progress_rows)


def _gate(
    results_paths: list[Path],
    **kwargs: Any,
) -> dict[str, Any]:
    return gate_invp_topology_residual(
        results_paths,
        progress_paths=[_progress_path(path) for path in results_paths],
        **kwargs,
    )


def test_public_gates_accept_zero_thresholds(tmp_path: Path) -> None:
    h1_results = tmp_path / "h1-results.jsonl"
    _write_h1_run(
        h1_results,
        {"anchor": 0.60, "candidate": 0.61, "shuffled_p": 0.59, "delta_only": 0.58},
    )
    h1_report = gate_invp_topology_residual(
        [h1_results],
        progress_paths=[_progress_path(h1_results)],
        seed0_architecture_margin=0,
        seed0_topology_margin=0,
        seed0_representation_margin=0,
        joint_architecture_margin=0,
        joint_control_margin=0,
    )

    conv_results = tmp_path / "conv-results.jsonl"
    conv_fixtures._write(
        conv_results,
        {
            conv_fixtures.ANCHOR: 0.60,
            conv_fixtures.CANDIDATE: 0.61,
            conv_fixtures.SHUFFLED: 0.59,
            conv_fixtures.DELTA: 0.58,
        },
    )
    conv_report = gate_invp_state_matrix_conv2d(
        [conv_results],
        progress_paths=[conv_fixtures._progress_path(conv_results)],
        seed0_topology_margin=0,
        seed0_representation_margin=0,
        joint_architecture_margin=0,
        joint_control_margin=0,
    )

    assert h1_report["status"] == "pass", h1_report["errors"]
    assert conv_report["status"] == "pass", conv_report["errors"]
    json.dumps(h1_report, allow_nan=False)
    json.dumps(conv_report, allow_nan=False)


@pytest.mark.parametrize(
    ("aucs", "decision", "next_action"),
    [
        (
            {"anchor": 0.60, "candidate": 0.60, "shuffled_p": 0.59, "delta_only": 0.58},
            "stop_topology_residual",
            "keep_token_mixer_anchor_and_stop_adapter",
        ),
        (
            {"anchor": 0.60, "candidate": 0.61, "shuffled_p": 0.61, "delta_only": 0.59},
            "stop_true_topology_attribution",
            "stop_adapter_without_true_topology_attribution",
        ),
        (
            {"anchor": 0.60, "candidate": 0.61, "shuffled_p": 0.59, "delta_only": 0.61},
            "stop_invp_adapter_attribution",
            "stop_adapter_without_invp_attribution",
        ),
        (
            {
                "anchor": 0.60,
                "candidate": 0.602,
                "shuffled_p": 0.59,
                "delta_only": 0.58,
            },
            "weak_or_fragile_no_scale",
            "inspect_histories_once_and_do_not_scale",
        ),
        (
            {
                "anchor": 0.60,
                "candidate": 0.61,
                "shuffled_p": 0.608,
                "delta_only": 0.58,
            },
            "weak_or_fragile_no_scale",
            "inspect_histories_once_and_do_not_scale",
        ),
        (
            {
                "anchor": 0.60,
                "candidate": 0.61,
                "shuffled_p": 0.58,
                "delta_only": 0.608,
            },
            "weak_or_fragile_no_scale",
            "inspect_histories_once_and_do_not_scale",
        ),
        (
            {
                "anchor": 0.60,
                "candidate": 0.603,
                "shuffled_p": 0.60,
                "delta_only": 0.60,
            },
            "promote_seed1",
            "run_identical_seed1_local_gate",
        ),
    ],
)
def test_h1_single_seed_decision_order_and_boundaries(
    tmp_path: Path,
    aucs: dict[str, float],
    decision: str,
    next_action: str,
) -> None:
    results = tmp_path / "results.jsonl"
    _write_h1_run(results, aucs)

    report = _gate([results], expected_seeds=(0,))

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == decision
    assert report["next_action"] == next_action
    assert report["research_decision_applied"] is True
    assert report["models"] == H1_ROLES
    assert "topology-residual" in report["claim_scope"]
    assert "not formal, paper-scale, or breakthrough evidence" in report["claim_scope"]
    stopped = {item["action"] for item in report["stopped_actions"]}
    if decision == "promote_seed1":
        assert stopped == {"65536_per_class", "262144_per_class", "remote_scale"}
    else:
        assert stopped == {
            "seed1",
            "65536_per_class",
            "262144_per_class",
            "remote_scale",
        }


def test_h1_exact_seed0_thresholds_are_inclusive(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_h1_run(
        results,
        {"anchor": 0.6, "candidate": 0.603, "shuffled_p": 0.6, "delta_only": 0.6},
    )

    report = _gate([results], expected_seeds=(0,))

    assert report["decision"] == "promote_seed1"
    assert report["promotion_conditions"]["seed0_architecture_margin"] == 0.003
    assert report["promotion_conditions"]["seed0_topology_margin"] == 0.003
    assert report["promotion_conditions"]["seed0_representation_margin"] == 0.003


@pytest.mark.parametrize(
    ("seed1_aucs", "decision", "next_action"),
    [
        (
            {
                "anchor": 0.6,
                "candidate": 0.601,
                "shuffled_p": 0.599,
                "delta_only": 0.599,
            },
            "promote_medium_65536",
            "run_65536_per_class_two_seed_medium_confirmation",
        ),
        (
            {
                "anchor": 0.6,
                "candidate": 0.6005,
                "shuffled_p": 0.598,
                "delta_only": 0.598,
            },
            "unstable_no_remote_scale",
            "do_not_launch_remote_scale_inspect_two_seed_variance",
        ),
    ],
)
def test_h1_two_seed_decision_uses_minimum_architecture_margin(
    tmp_path: Path,
    seed1_aucs: dict[str, float],
    decision: str,
    next_action: str,
) -> None:
    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    _write_h1_run(
        seed0,
        {"anchor": 0.6, "candidate": 0.604, "shuffled_p": 0.601, "delta_only": 0.601},
        seed=0,
    )
    _write_h1_run(seed1, seed1_aucs, seed=1)

    report = _gate([seed0, seed1], expected_seeds=(0, 1))

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == decision
    assert report["next_action"] == next_action


def test_h1_readiness_is_neutral_and_frozen(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_h1_run(
        results,
        {"anchor": 0.9, "candidate": 0.1, "shuffled_p": 0.2, "delta_only": 0.3},
        samples_per_class=64,
        epochs=1,
    )

    report = _gate(
        [results],
        expected_seeds=(0,),
        samples_per_class=64,
        epochs=1,
        readiness_only=True,
    )

    assert report["status"] == "pass", report["errors"]
    assert report["decision"] == "implementation_ready"
    assert report["research_decision_applied"] is False
    assert (
        report["claim_scope"]
        == "implementation readiness only; metrics not interpreted"
    )
    assert report["next_action"] == "run_frozen_h1_seed0_local_diagnostic"
    assert {item["action"] for item in report["stopped_actions"]} == {
        "interpret_smoke_metrics",
        "remote_scale",
    }


def test_h1_report_records_frozen_semantics_and_equal_hybrid_capacity(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.jsonl"
    _write_h1_run(
        results,
        {"anchor": 0.60, "candidate": 0.61, "shuffled_p": 0.59, "delta_only": 0.58},
    )

    report = _gate([results], expected_seeds=(0,))

    assert report["parameter_counts"] == {
        "anchor": 900,
        "candidate": 1100,
        "shuffled_p": 1100,
        "delta_only": 1100,
        "candidate_to_anchor_ratio": 1100 / 900,
    }
    evidence_rows = report["protocol_evidence"]["rows"]
    assert [row["model"] for row in evidence_rows] == list(H1_ROLES.values())
    assert evidence_rows[0]["model_options"] == {
        "spn_mixer_depth": 2,
        "activation": "relu",
        "norm": "layernorm",
    }
    assert all(row["model_options"] == HYBRID_OPTIONS for row in evidence_rows[1:])
    semantic = report["semantic_checks"]
    assert isinstance(report["models"], dict)
    assert isinstance(semantic, dict)
    assert semantic["common_token_backbone"] == {
        "view_mode": "inv_p",
        "p_alignment": "true",
        "roles": ["candidate", "shuffled_p", "delta_only"],
    }
    assert semantic["adapter_mapping_identities"] == {
        "candidate": "true_inv_p",
        "shuffled_p": "deterministic_shuffled_p",
        "delta_only": "raw_delta",
    }
    assert semantic["pair_fusion"] == "token + alpha * local"
    assert semantic["hybrid_capacity_constraint"] == "equal total and trainable counts"
    assert semantic["negative_mode"] == "encrypted_random_plaintexts"
    assert semantic["effective_key_schedule"] == "per_pair_random"
    assert "not runtime tensor equality" in semantic["evidence_kind"]
    assert isinstance(semantic["common_token_backbone"]["roles"], list)
    assert json.loads(json.dumps(report))["semantic_checks"] == semantic


@pytest.mark.parametrize(
    ("anchor_reuse_count", "create_count", "reuse_count"),
    [(0, 2, 6), (1, 1, 7), (2, 0, 8)],
)
def test_h1_anchor_terminal_cache_reuse_is_valid(
    tmp_path: Path,
    anchor_reuse_count: int,
    create_count: int,
    reuse_count: int,
) -> None:
    results = tmp_path / "results.jsonl"
    _write_h1_run(
        results,
        {"anchor": 0.60, "candidate": 0.61, "shuffled_p": 0.59, "delta_only": 0.58},
    )
    progress_path = _progress_path(results)
    progress = _read_jsonl(progress_path)
    for index in range(anchor_reuse_count):
        progress[index]["event"] = "cache_reuse"
    _write_jsonl(progress_path, progress)

    report = _gate([results], expected_seeds=(0,))

    assert report["status"] == "pass", report["errors"]
    cache = report["cache_evidence"]["0"]
    assert cache["create_count"] == create_count
    assert cache["reuse_count"] == reuse_count
    assert cache["control_reuse_count"] == 6


@pytest.mark.parametrize(
    ("mutation", "error_token"),
    [
        ("wrong_role", "unexpected_selected_model"),
        ("wrong_options", "model_options"),
        ("parameter_count", "topology_residual_parameter_count_mismatch"),
        ("protocol", "negative_mode"),
        ("history", "history"),
        ("training_schedule", "key_schedule"),
        ("validation_schedule", "key_schedule"),
        ("cache", "cache_root"),
        ("control_not_reuse", "control_reuse_count"),
        ("missing_terminal", "terminal_count"),
        ("duplicate_terminal", "terminal_count"),
        ("progress_role", "model unexpected"),
        ("run_done", "run_done"),
    ],
)
def test_h1_strict_protocol_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
    error_token: str,
) -> None:
    results = tmp_path / "results.jsonl"
    _write_h1_run(
        results,
        {"anchor": 0.60, "candidate": 0.61, "shuffled_p": 0.59, "delta_only": 0.58},
    )
    rows = _read_jsonl(results)
    progress_path = _progress_path(results)
    progress = _read_jsonl(progress_path)
    if mutation == "wrong_role":
        rows[1]["model"] = "unexpected"
        rows[1]["selected_model"] = "unexpected"
    elif mutation == "wrong_options":
        rows[1]["training"]["model_options"]["local_channels"] = 15
    elif mutation == "parameter_count":
        rows[2]["parameter_count"] = 1101
    elif mutation == "protocol":
        rows[1]["negative_mode"] = "random_ciphertext"
    elif mutation == "history":
        rows[1]["history"].pop()
    elif mutation == "training_schedule":
        rows[1]["training"]["key_schedule"] = "fixed"
    elif mutation == "validation_schedule":
        rows[1]["validation"]["key_schedule"] = "fixed"
    elif mutation == "cache":
        progress[0]["cache_path"] = str(tmp_path / "unrelated" / "train")
    elif mutation == "control_not_reuse":
        progress[2]["event"] = "cache_done"
    elif mutation == "missing_terminal":
        progress.pop(2)
    elif mutation == "duplicate_terminal":
        progress.insert(3, dict(progress[2]))
    elif mutation == "progress_role":
        progress[2]["model"] = conv_fixtures.CANDIDATE
    elif mutation == "run_done":
        progress.pop()
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _write_jsonl(results, rows)
    _write_jsonl(progress_path, progress)

    report = _gate([results], expected_seeds=(0,))

    assert report["status"] == "fail"
    assert report["decision"] == "invalid_protocol"
    assert report["research_decision_applied"] is False
    assert any(error_token in error for error in report["errors"]), report["errors"]


def test_h1_swapped_result_progress_order_fails_closed(tmp_path: Path) -> None:
    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    aucs = {"anchor": 0.60, "candidate": 0.61, "shuffled_p": 0.59, "delta_only": 0.58}
    _write_h1_run(seed0, aucs, seed=0)
    _write_h1_run(seed1, aucs, seed=1)

    report = gate_invp_topology_residual(
        [seed0, seed1],
        progress_paths=[_progress_path(seed1), _progress_path(seed0)],
        expected_seeds=(0, 1),
    )

    assert report["decision"] == "invalid_protocol"
    assert report["research_decision_applied"] is False
    assert any("paired_terminal_seed" in error for error in report["errors"])


def test_conv2d_saved_r0_r1_replay_keeps_key_schema_and_decisions() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = [
        (
            "i1_present_invp_state_matrix_conv2d_smoke_seed0",
            "readiness_gate.json",
            64,
            1,
            True,
            "implementation_ready",
            "run_frozen_r1_seed0_local_diagnostic",
        ),
        (
            "i1_present_invp_state_matrix_conv2d_8192_seed0",
            "state_matrix_conv2d_gate.json",
            8192,
            10,
            False,
            "stop_conv2d_route",
            "keep_token_mixer_anchor_and_do_not_scale_conv2d",
        ),
    ]
    for (
        run_name,
        artifact_name,
        samples,
        epochs,
        readiness,
        decision,
        next_action,
    ) in cases:
        run_root = root / "outputs" / "local_smoke" / run_name
        report = gate_invp_state_matrix_conv2d(
            [run_root / "results.jsonl"],
            progress_paths=[run_root / "progress.jsonl"],
            expected_seeds=(0,),
            samples_per_class=samples,
            epochs=epochs,
            readiness_only=readiness,
        )
        saved = json.loads((run_root / artifact_name).read_text(encoding="utf-8"))

        assert report["status"] == saved["status"] == "pass"
        assert report["decision"] == saved["decision"] == decision
        assert report["models"] == saved["models"]
        assert report["seeds"] == saved["seeds"]
        assert report["parameter_counts"] == saved["parameter_counts"]
        assert report["protocol_evidence"] == saved["protocol_evidence"]
        assert report["semantic_checks"] == saved["semantic_checks"]
        assert report["promotion_conditions"] == saved["promotion_conditions"]
        assert report["claim_scope"] == saved["claim_scope"]
        assert report["research_decision_applied"] is saved["research_decision_applied"]
        cache = report["cache_evidence"]["0"]
        saved_cache = saved["cache_evidence"]["0"]
        for field in (
            "verified",
            "create_count",
            "reuse_count",
            "control_reuse_count",
            "run_done_count",
        ):
            assert cache[field] == saved_cache[field]
        assert cache["progress_path"].endswith(saved_cache["progress_path"])
        event_fields = ("role", "model", "split", "event", "original_cache_path")
        assert [
            {field: event[field] for field in event_fields} for event in cache["events"]
        ] == [
            {field: event[field] for field in event_fields}
            for event in saved_cache["events"]
        ]
        assert report["stopped_actions"] == saved["stopped_actions"]
        assert report["next_action"] == saved["next_action"] == next_action


def test_h1_public_roles_constant_is_exact() -> None:
    assert TOPOLOGY_RESIDUAL_MODEL_ROLES == H1_ROLES
