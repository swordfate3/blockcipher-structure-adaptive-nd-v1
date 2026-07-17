import numpy as np

from blockcipher_nd.cli import audit_innovation2_skinny_r8_geometry_diversity as cli
from blockcipher_nd.tasks.innovation2 import skinny_r8_geometry_diversity as geometry
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
)
from blockcipher_nd.tasks.innovation2.skinny_hwang_r8_readiness import (
    paper_basis_masks,
)
from blockcipher_nd.tasks.innovation2.skinny_r8_geometry_diversity import (
    ADJACENT_PAIRS,
    SkinnyR8GeometryDiversityConfig,
    evaluate_skinny_r8_geometry_diversity,
)


def test_e22_uses_exact_sixteen_cyclic_adjacent_pairs() -> None:
    assert len(ADJACENT_PAIRS) == 16
    assert ADJACENT_PAIRS[0] == (0, 1)
    assert ADJACENT_PAIRS[14] == (14, 15)
    assert ADJACENT_PAIRS[15] == (15, 0)


def test_e22_gate_passes_four_stable_distinct_joint_kernels() -> None:
    config = SkinnyR8GeometryDiversityConfig(run_id="pass")
    masks = (1 << 63, 1 << 62, 1 << 61, paper_basis_masks()[0])
    parity_rows = _synthetic_rows(config, masks_by_pair={0: masks[0], 1: masks[1], 2: masks[2], 14: masks[3]})

    result = evaluate_skinny_r8_geometry_diversity(
        config,
        keys=tuple(range(config.total_keys)),
        e20_keys=tuple(range(10_000, 10_768)),
        e21_keys=tuple(range(20_000, 20_768)),
        base_plaintext=0,
        parity_rows=parity_rows,
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r8_geometry_kernel_diversity_ready"
    )
    assert result["gate"]["metrics"]["nontrivial_joint_kernel_structures"] == 4
    assert result["gate"]["metrics"]["distinct_nontrivial_joint_kernel_signatures"] == 4


def test_e22_gate_holds_when_only_paper_anchor_is_nontrivial() -> None:
    config = SkinnyR8GeometryDiversityConfig(run_id="hold")
    parity_rows = _synthetic_rows(
        config,
        masks_by_pair={14: paper_basis_masks()[0]},
    )

    result = evaluate_skinny_r8_geometry_diversity(
        config,
        keys=tuple(range(config.total_keys)),
        e20_keys=tuple(range(10_000, 10_768)),
        e21_keys=tuple(range(20_000, 20_768)),
        base_plaintext=0,
        parity_rows=parity_rows,
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r8_geometry_kernel_not_diverse"
    )


def test_e22_anchor_allows_half_rank_noise_when_joint_span_is_exact() -> None:
    config = SkinnyR8GeometryDiversityConfig(run_id="half-noise")
    paper = paper_basis_masks()[0]
    parity_rows = _synthetic_rows(config, masks_by_pair={14: paper})
    extra = 1 << 63
    validation_rows = gf2_kernel_basis(
        np.asarray([paper, extra], dtype=np.uint64)
    )
    parity_rows[14, config.discovery_keys :] = np.resize(
        np.asarray(validation_rows, dtype=np.uint64), config.validation_keys
    )

    result = evaluate_skinny_r8_geometry_diversity(
        config,
        keys=tuple(range(config.total_keys)),
        e20_keys=tuple(range(10_000, 10_768)),
        e21_keys=tuple(range(20_000, 20_768)),
        base_plaintext=0,
        parity_rows=parity_rows,
    )

    anchor = result["rows"][14]
    assert anchor["discovery_nullity"] == 1
    assert anchor["validation_nullity"] == 2
    assert anchor["joint_nullity"] == 1
    assert result["gate"]["status"] == "hold"
    assert all(result["gate"]["anchor_checks"].values())


def test_e22_runner_returns_complete_result_with_stubbed_rows(monkeypatch) -> None:
    config = SkinnyR8GeometryDiversityConfig(run_id="runner")
    parity_rows = _synthetic_rows(
        config,
        masks_by_pair={14: paper_basis_masks()[0]},
    )

    def fake_collect(*args, pair_index, **kwargs):
        return parity_rows[pair_index].copy()

    monkeypatch.setattr(geometry, "_collect_pair_rows", fake_collect)

    result = geometry.run_skinny_r8_geometry_diversity_audit(config)

    assert len(result["rows"]) == 16
    assert result["keys"].shape == (128,)
    assert result["parity_rows"].shape == (16, 128)
    assert result["gate"]["status"] == "hold"
    assert result["metadata"]["training_performed"] is False


def test_e22_cli_writes_complete_artifact_set(monkeypatch, tmp_path) -> None:
    config = SkinnyR8GeometryDiversityConfig(run_id="cli")
    parity_rows = _synthetic_rows(
        config,
        masks_by_pair={14: paper_basis_masks()[0]},
    )
    result = evaluate_skinny_r8_geometry_diversity(
        config,
        keys=tuple(range(config.total_keys)),
        e20_keys=tuple(range(10_000, 10_768)),
        e21_keys=tuple(range(20_000, 20_768)),
        base_plaintext=0,
        parity_rows=parity_rows,
    )
    monkeypatch.setattr(
        cli,
        "run_skinny_r8_geometry_diversity_audit",
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


def _synthetic_rows(
    config: SkinnyR8GeometryDiversityConfig,
    *,
    masks_by_pair: dict[int, int],
) -> np.ndarray:
    full_rank = np.asarray([1 << bit for bit in range(64)], dtype=np.uint64)
    rows = np.empty((16, config.total_keys), dtype=np.uint64)
    for pair_index in range(16):
        mask = masks_by_pair.get(pair_index)
        basis_rows = (
            gf2_kernel_basis(np.asarray([mask], dtype=np.uint64))
            if mask is not None
            else tuple(int(value) for value in full_rank)
        )
        half = np.resize(
            np.asarray(basis_rows, dtype=np.uint64), config.discovery_keys
        )
        rows[pair_index] = np.concatenate((half, half))
    return rows
