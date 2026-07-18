from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_present_open_3sdp_exact_oracle import (
    main as audit_main,
)
from blockcipher_nd.cli.plot_innovation2_present_open_3sdp_exact_oracle import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_open_3sdp_exact_oracle import (
    ExactOracleConfig,
    audit_sbox_transition_parity,
    build_multi_mask_fixtures,
    build_present_exact_anf_snapshots,
    build_strict_fixtures,
    cube_superpoly,
    evaluate_output_polynomials,
    polynomial_product,
    polynomial_xor,
    validate_exact_outputs,
)


def test_boolean_polynomial_operations_apply_gf2_cancellation_and_idempotence() -> None:
    x0 = frozenset({1})
    x1 = frozenset({2})

    assert polynomial_xor(x0, x0) == frozenset()
    assert polynomial_product(x0, x0) == x0
    assert polynomial_product(polynomial_xor(x0, x1), x0) == frozenset({1, 3})


def test_one_round_exact_present_anf_matches_scalar_cipher() -> None:
    outputs = build_present_exact_anf_snapshots((1,))[1]
    validation = validate_exact_outputs(outputs, rounds=1, count=8, seed=20260718)

    assert validation["all_pass"]
    assert sum(len(polynomial) for polynomial in outputs) == 1907
    assert evaluate_output_polynomials(outputs, 0, 0) != 0


def test_exact_oracle_builds_strict_and_multi_mask_fixtures() -> None:
    outputs = build_present_exact_anf_snapshots((1,))[1]
    strict = build_strict_fixtures(
        outputs,
        rounds=1,
        fixtures_per_class=4,
        seed=20260718,
    )
    multi = build_multi_mask_fixtures(
        outputs,
        rounds=1,
        count=4,
        seed=20260718,
    )

    assert sum(row["status"] == "positive" for row in strict) == 4
    assert sum(row["status"] == "negative" for row in strict) == 4
    assert all(
        row["scalar_rechecks"]["all_pass"]
        for row in strict
        if row["status"] == "positive"
    )
    assert all(
        row["witness_parity"] == 1
        for row in strict
        if row["status"] == "negative"
    )
    assert all(row["component_xor_matches"] for row in multi)
    assert cube_superpoly(outputs, (0,), 1 << 1) == frozenset()


def test_sbox_transition_parity_rejects_existence_only_control() -> None:
    audit = audit_sbox_transition_parity()

    assert audit["checks"] == {
        "trail_parity_matches_exact_sbox_anf": True,
        "trail_order_invariant": True,
        "existence_only_control_is_rejected": True,
    }
    assert audit["metrics"]["candidate_transitions"] == 256
    assert audit["metrics"]["existence_transitions"] == 166
    assert audit["metrics"]["odd_parity_transitions"] == 90
    assert audit["metrics"]["existence_only_false_positives"] == 76


def test_audit_protocol_freezes_one_and_two_round_exact_oracles() -> None:
    config = ExactOracleConfig(run_id="e53")

    assert config.rounds == (1, 2)
    assert config.fixtures_per_class == 8


def test_smoke_cli_writes_auditable_phase_a_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "e53"
    exit_code = audit_main(
        [
            "--run-id",
            "e53_smoke_test",
            "--output-root",
            str(output),
            "--mode",
            "smoke",
            "--rounds",
            "1",
            "--fixtures-per-class",
            "4",
        ]
    )

    assert exit_code == 0
    expected = {
        "provider_manifest.json",
        "fixtures.jsonl",
        "certificates.jsonl",
        "witnesses.jsonl",
        "results.jsonl",
        "gate.json",
        "summary.json",
        "metadata.json",
        "progress.jsonl",
    }
    assert expected.issubset(path.name for path in output.iterdir())
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_present_r5_open_3sdp_exact_oracle_ready"
    )
    assert gate["glpk_checks"]["glpk_trail_enumerator_implemented"] is False
    manifest = json.loads(
        (output / "provider_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["control"]["status"] == "rejected"
    assert manifest["control"]["false_positive_transitions"] == 76

    assert (
        plot_main(
            [
                "--summary",
                str(output / "summary.json"),
                "--output",
                str(output / "curves.svg"),
            ]
        )
        == 0
    )
    svg = (output / "curves.svg").read_text(encoding="utf-8")
    assert "创新2 E53-A" in svg
    assert "不是五轮标签结果" in svg
