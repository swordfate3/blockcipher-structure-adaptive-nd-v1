from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import test_invp_state_matrix_conv2d_gate as conv_fixtures
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
        ("anchor_not_create", "create_count"),
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
    elif mutation == "anchor_not_create":
        progress[0]["event"] = "cache_reuse"
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
