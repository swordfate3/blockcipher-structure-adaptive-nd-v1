import numpy as np

from blockcipher_nd.cli import audit_innovation2_skinny_hwang_r8_readiness as cli
from blockcipher_nd.tasks.innovation2 import skinny_hwang_r8_readiness as r8
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
)
from blockcipher_nd.tasks.innovation2.skinny_hwang_r8_readiness import (
    SkinnyHwangR8ReadinessConfig,
    evaluate_skinny_hwang_r8_readiness,
    paper_basis_masks,
)


def test_r8_paper_basis_uses_msb_first_bits() -> None:
    assert paper_basis_masks() == (
        (1 << (63 - 28)) | (1 << (63 - 44)) | (1 << (63 - 60)),
    )


def test_e21_gate_requires_exact_one_dimensional_target_and_control() -> None:
    config = SkinnyHwangR8ReadinessConfig(run_id="synthetic")
    paper = paper_basis_masks()
    orthogonal = gf2_kernel_basis(np.asarray(paper, dtype=np.uint64))
    target = np.resize(np.asarray(orthogonal, dtype=np.uint64), config.total_keys)
    control = np.resize(
        np.asarray([1 << bit for bit in range(64)], dtype=np.uint64),
        config.total_keys,
    )
    result = evaluate_skinny_hwang_r8_readiness(
        config,
        keys=tuple(range(config.total_keys)),
        e20_keys=tuple(range(10_000, 10_000 + config.total_keys)),
        base_plaintext=0,
        parity_rows=np.stack((target, control)),
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r8_hwang_kernel_reproduced"
    )
    assert result["rows"][0]["joint_rank"] == 63
    assert result["rows"][0]["joint_nullity"] == 1
    assert result["rows"][1]["joint_rank"] == 64


def test_e21_gate_holds_for_underconstrained_target() -> None:
    config = SkinnyHwangR8ReadinessConfig(run_id="hold")
    target = np.zeros(config.total_keys, dtype=np.uint64)
    control = np.resize(
        np.asarray([1 << bit for bit in range(64)], dtype=np.uint64),
        config.total_keys,
    )
    result = evaluate_skinny_hwang_r8_readiness(
        config,
        keys=tuple(range(config.total_keys)),
        e20_keys=tuple(range(10_000, 10_000 + config.total_keys)),
        base_plaintext=0,
        parity_rows=np.stack((target, control)),
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r8_hwang_kernel_not_reproduced"
    )


def test_e21_runner_returns_complete_result_with_stubbed_parity(monkeypatch) -> None:
    config = SkinnyHwangR8ReadinessConfig(run_id="runner")
    paper = paper_basis_masks()
    orthogonal = gf2_kernel_basis(np.asarray(paper, dtype=np.uint64))
    target = np.resize(np.asarray(orthogonal, dtype=np.uint64), config.total_keys)
    control = np.resize(
        np.asarray([1 << bit for bit in range(64)], dtype=np.uint64),
        config.total_keys,
    )

    def fake_collect(*args, role, **kwargs):
        return target.copy() if role == "target" else control.copy()

    monkeypatch.setattr(r8, "_collect_role_rows", fake_collect)

    result = r8.run_skinny_hwang_r8_readiness_audit(config)

    assert result["gate"]["status"] == "pass"
    assert result["keys"].shape == (768,)
    assert result["parity_rows"].shape == (2, 768)
    assert result["metadata"]["training_performed"] is False


def test_e21_cli_writes_complete_artifact_set(monkeypatch, tmp_path) -> None:
    config = SkinnyHwangR8ReadinessConfig(run_id="cli")
    paper = paper_basis_masks()
    orthogonal = gf2_kernel_basis(np.asarray(paper, dtype=np.uint64))
    target = np.resize(np.asarray(orthogonal, dtype=np.uint64), config.total_keys)
    control = np.resize(
        np.asarray([1 << bit for bit in range(64)], dtype=np.uint64),
        config.total_keys,
    )
    result = evaluate_skinny_hwang_r8_readiness(
        config,
        keys=tuple(range(config.total_keys)),
        e20_keys=tuple(range(10_000, 10_000 + config.total_keys)),
        base_plaintext=0,
        parity_rows=np.stack((target, control)),
    )
    monkeypatch.setattr(
        cli,
        "run_skinny_hwang_r8_readiness_audit",
        lambda *args, **kwargs: result,
    )
    output_root = tmp_path / "result"

    exit_code = cli.main(
        [
            "--run-id",
            "cli",
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert {
        "curves.svg",
        "gate.json",
        "kernel_basis.csv",
        "keys.npy",
        "metadata.json",
        "parity_rows.npy",
        "progress.jsonl",
        "results.jsonl",
    } <= {path.name for path in output_root.iterdir()}
