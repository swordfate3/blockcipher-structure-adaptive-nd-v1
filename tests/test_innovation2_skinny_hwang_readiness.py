import numpy as np

from blockcipher_nd.cli import audit_innovation2_skinny_hwang_readiness as cli
from blockcipher_nd.tasks.innovation2 import skinny_hwang_readiness as skinny
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
)
from blockcipher_nd.tasks.innovation2.skinny_hwang_readiness import (
    SkinnyHwangReadinessConfig,
    evaluate_skinny_hwang_readiness,
    mask_to_paper_bits,
    paper_basis_masks,
)


def test_paper_basis_uses_msb_first_output_bits() -> None:
    first = paper_basis_masks()[0]

    assert first == (1 << (63 - 4)) | (1 << (63 - 52))
    assert mask_to_paper_bits(first) == (4, 52)
    assert paper_basis_masks(mapping="lsb_first")[0] == (1 << 4) | (1 << 52)


def test_e20_gate_requires_exact_paper_span_and_active_cell_control() -> None:
    config = SkinnyHwangReadinessConfig(run_id="synthetic")
    paper = paper_basis_masks()
    orthogonal = gf2_kernel_basis(np.asarray(paper, dtype=np.uint64))
    target = np.resize(np.asarray(orthogonal, dtype=np.uint64), config.total_keys)
    control = np.resize(
        np.asarray([1 << bit for bit in range(64)], dtype=np.uint64),
        config.total_keys,
    )
    result = evaluate_skinny_hwang_readiness(
        config,
        keys=tuple(range(config.total_keys)),
        base_plaintext=0,
        parity_rows=np.stack((target, control)),
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r7_hwang_kernel_reproduced"
    )
    assert result["rows"][0]["joint_rank"] == 46
    assert result["rows"][0]["joint_nullity"] == 18
    assert result["rows"][1]["joint_rank"] == 64
    assert result["rows"][1]["joint_nullity"] == 0


def test_e20_gate_holds_when_target_kernel_is_underconstrained() -> None:
    config = SkinnyHwangReadinessConfig(run_id="underconstrained")
    target = np.zeros(config.total_keys, dtype=np.uint64)
    control = np.resize(
        np.asarray([1 << bit for bit in range(64)], dtype=np.uint64),
        config.total_keys,
    )
    result = evaluate_skinny_hwang_readiness(
        config,
        keys=tuple(range(config.total_keys)),
        base_plaintext=0,
        parity_rows=np.stack((target, control)),
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r7_hwang_kernel_not_reproduced"
    )


def test_e20_real_runner_returns_complete_artifacts_with_stubbed_rows(
    monkeypatch,
) -> None:
    config = SkinnyHwangReadinessConfig(run_id="runner")
    paper = paper_basis_masks()
    orthogonal = gf2_kernel_basis(np.asarray(paper, dtype=np.uint64))
    target = np.resize(np.asarray(orthogonal, dtype=np.uint64), config.total_keys)
    control = np.resize(
        np.asarray([1 << bit for bit in range(64)], dtype=np.uint64),
        config.total_keys,
    )

    def fake_collect(*args, role, **kwargs):
        return target.copy() if role == "target" else control.copy()

    monkeypatch.setattr(skinny, "_collect_role_rows", fake_collect)

    result = skinny.run_skinny_hwang_readiness_audit(config)

    assert len(result["rows"]) == 2
    assert result["keys"].shape == (768,)
    assert result["parity_rows"].shape == (2, 768)
    assert result["gate"]["status"] == "pass"
    assert result["metadata"]["training_performed"] is False


def test_e20_cli_writes_complete_artifact_set(monkeypatch, tmp_path) -> None:
    config = SkinnyHwangReadinessConfig(run_id="cli")
    paper = paper_basis_masks()
    orthogonal = gf2_kernel_basis(np.asarray(paper, dtype=np.uint64))
    target = np.resize(np.asarray(orthogonal, dtype=np.uint64), config.total_keys)
    control = np.resize(
        np.asarray([1 << bit for bit in range(64)], dtype=np.uint64),
        config.total_keys,
    )
    result = evaluate_skinny_hwang_readiness(
        config,
        keys=tuple(range(config.total_keys)),
        base_plaintext=0,
        parity_rows=np.stack((target, control)),
    )
    monkeypatch.setattr(cli, "run_skinny_hwang_readiness_audit", lambda *a, **k: result)
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
    assert np.load(output_root / "keys.npy", allow_pickle=False).shape == (768,)
    assert np.load(output_root / "parity_rows.npy", allow_pickle=False).shape == (
        2,
        768,
    )
