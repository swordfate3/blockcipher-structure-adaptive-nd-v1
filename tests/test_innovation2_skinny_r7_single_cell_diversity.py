import numpy as np

from blockcipher_nd.cli import (
    audit_innovation2_skinny_r7_single_cell_diversity as cli,
)
from blockcipher_nd.tasks.innovation2 import skinny_r7_single_cell_diversity as diversity
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import gf2_kernel_basis
from blockcipher_nd.tasks.innovation2.skinny_hwang_readiness import paper_basis_masks
from blockcipher_nd.tasks.innovation2.skinny_r7_single_cell_diversity import (
    ACTIVE_CELLS,
    ANCHOR_CELL,
    CONTROL_CELL,
    SkinnyR7SingleCellDiversityConfig,
    evaluate_skinny_r7_single_cell_diversity,
)


def test_e24_uses_exact_sixteen_single_active_cells() -> None:
    assert ACTIVE_CELLS == tuple(range(16))
    assert CONTROL_CELL == 0
    assert ANCHOR_CELL == 15


def test_e24_gate_passes_six_stable_and_four_distinct_kernels() -> None:
    config = SkinnyR7SingleCellDiversityConfig(run_id="pass")
    masks = {
        1: (1 << 63,),
        2: (1 << 62,),
        3: (1 << 61,),
        4: (1 << 60,),
        5: (1 << 59,),
        15: paper_basis_masks(),
    }
    result = _evaluate(config, _synthetic_rows(config, masks_by_cell=masks))

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r7_single_cell_kernel_diversity_ready"
    )
    assert result["gate"]["metrics"]["nontrivial_joint_kernel_structures"] == 6
    assert result["gate"]["metrics"][
        "distinct_nontrivial_joint_kernel_signatures"
    ] == 6


def test_e24_holds_when_hwang_anchor_does_not_reproduce() -> None:
    config = SkinnyR7SingleCellDiversityConfig(run_id="anchor-hold")
    result = _evaluate(config, _synthetic_rows(config, masks_by_cell={}))

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r7_single_cell_anchor_not_reproduced"
    )


def test_e24_holds_when_only_anchor_is_nontrivial() -> None:
    config = SkinnyR7SingleCellDiversityConfig(run_id="diversity-hold")
    result = _evaluate(
        config,
        _synthetic_rows(config, masks_by_cell={15: paper_basis_masks()}),
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r7_single_cell_kernel_not_diverse"
    )


def test_e24_anchor_allows_half_rank_noise_when_joint_span_is_exact() -> None:
    config = SkinnyR7SingleCellDiversityConfig(run_id="half-noise")
    parity_rows = _synthetic_rows(
        config,
        masks_by_cell={15: paper_basis_masks()},
    )
    extra = 1 << 63
    validation_rows = gf2_kernel_basis(
        np.asarray(paper_basis_masks() + (extra,), dtype=np.uint64)
    )
    parity_rows[15, config.discovery_keys :] = np.resize(
        np.asarray(validation_rows, dtype=np.uint64), config.validation_keys
    )

    result = _evaluate(config, parity_rows)

    anchor = result["rows"][15]
    assert anchor["validation_nullity"] == 19
    assert anchor["joint_nullity"] == 18
    assert all(result["gate"]["anchor_checks"].values())
    assert result["gate"]["status"] == "hold"


def test_e24_runner_returns_complete_result_with_stubbed_rows(monkeypatch) -> None:
    config = SkinnyR7SingleCellDiversityConfig(run_id="runner")
    parity_rows = _synthetic_rows(
        config,
        masks_by_cell={15: paper_basis_masks()},
    )

    def fake_collect(*args, active_cell, **kwargs):
        return parity_rows[active_cell].copy()

    monkeypatch.setattr(diversity, "_collect_cell_rows", fake_collect)
    result = diversity.run_skinny_r7_single_cell_diversity_audit(config)

    assert len(result["rows"]) == 16
    assert result["keys"].shape == (128,)
    assert result["parity_rows"].shape == (16, 128)
    assert result["metadata"]["training_performed"] is False


def test_e24_cli_writes_complete_artifact_set(monkeypatch, tmp_path) -> None:
    config = SkinnyR7SingleCellDiversityConfig(run_id="cli")
    result = _evaluate(
        config,
        _synthetic_rows(config, masks_by_cell={15: paper_basis_masks()}),
    )
    monkeypatch.setattr(
        cli,
        "run_skinny_r7_single_cell_diversity_audit",
        lambda *args, **kwargs: result,
    )
    output_root = tmp_path / "result"

    exit_code = cli.main(
        ["--run-id", "cli", "--output-root", str(output_root)]
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


def _evaluate(
    config: SkinnyR7SingleCellDiversityConfig,
    parity_rows: np.ndarray,
) -> dict:
    return evaluate_skinny_r7_single_cell_diversity(
        config,
        keys=tuple(range(config.total_keys)),
        prior_key_sets={
            "e20": tuple(range(10_000, 10_768)),
            "e21": tuple(range(20_000, 20_768)),
            "e22": tuple(range(30_000, 30_128)),
            "e23": tuple(range(40_000, 40_128)),
        },
        base_plaintext=0,
        parity_rows=parity_rows,
    )


def _synthetic_rows(
    config: SkinnyR7SingleCellDiversityConfig,
    *,
    masks_by_cell: dict[int, tuple[int, ...]],
) -> np.ndarray:
    full_rank = np.asarray([1 << bit for bit in range(64)], dtype=np.uint64)
    rows = np.empty((len(ACTIVE_CELLS), config.total_keys), dtype=np.uint64)
    for active_cell in ACTIVE_CELLS:
        masks = masks_by_cell.get(active_cell)
        basis_rows = (
            gf2_kernel_basis(np.asarray(masks, dtype=np.uint64))
            if masks is not None
            else tuple(int(value) for value in full_rank)
        )
        half = np.resize(
            np.asarray(basis_rows, dtype=np.uint64), config.discovery_keys
        )
        rows[active_cell] = np.concatenate((half, half))
    return rows
