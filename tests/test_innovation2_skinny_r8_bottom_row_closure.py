import numpy as np

from blockcipher_nd.cli import (
    audit_innovation2_skinny_r8_bottom_row_closure as cli,
)
from blockcipher_nd.tasks.innovation2 import skinny_r8_bottom_row_closure as closure
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
)
from blockcipher_nd.tasks.innovation2.skinny_r8_bottom_row_closure import (
    AUDIT_PAIRS,
    BOTTOM_ROW_PAIRS,
    CONTROL_PAIR,
    EXPECTED_ANCHOR_MASKS,
    SkinnyR8BottomRowClosureConfig,
    evaluate_skinny_r8_bottom_row_closure,
)


def test_e23_uses_exact_bottom_row_pairs_and_control() -> None:
    assert BOTTOM_ROW_PAIRS == (
        (12, 13),
        (12, 14),
        (12, 15),
        (13, 14),
        (13, 15),
        (14, 15),
    )
    assert AUDIT_PAIRS == BOTTOM_ROW_PAIRS + (CONTROL_PAIR,)


def test_e23_gate_passes_six_stable_distinct_bottom_row_kernels() -> None:
    config = SkinnyR8BottomRowClosureConfig(run_id="pass")
    masks = {
        0: EXPECTED_ANCHOR_MASKS[(12, 13)],
        1: 1 << 63,
        2: 1 << 62,
        3: EXPECTED_ANCHOR_MASKS[(13, 14)],
        4: 1 << 61,
        5: EXPECTED_ANCHOR_MASKS[(14, 15)],
    }
    result = _evaluate(config, _synthetic_rows(config, masks_by_pair=masks))

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r8_bottom_row_pair_family_ready"
    )
    assert result["gate"]["metrics"][
        "bottom_row_nontrivial_joint_kernel_structures"
    ] == 6
    assert result["gate"]["metrics"][
        "bottom_row_distinct_joint_kernel_signatures"
    ] == 6


def test_e23_holds_when_a_known_anchor_does_not_reproduce() -> None:
    config = SkinnyR8BottomRowClosureConfig(run_id="anchor-hold")
    masks = {
        0: EXPECTED_ANCHOR_MASKS[(12, 13)],
        3: EXPECTED_ANCHOR_MASKS[(13, 14)],
    }
    result = _evaluate(config, _synthetic_rows(config, masks_by_pair=masks))

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r8_bottom_row_anchor_not_reproduced"
    )


def test_e23_holds_when_control_contains_anchor_family() -> None:
    config = SkinnyR8BottomRowClosureConfig(run_id="control-hold")
    masks = {
        0: EXPECTED_ANCHOR_MASKS[(12, 13)],
        1: 1 << 63,
        2: 1 << 62,
        3: EXPECTED_ANCHOR_MASKS[(13, 14)],
        4: 1 << 61,
        5: EXPECTED_ANCHOR_MASKS[(14, 15)],
        6: EXPECTED_ANCHOR_MASKS[(12, 13)],
    }
    result = _evaluate(config, _synthetic_rows(config, masks_by_pair=masks))

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_skinny_r8_bottom_row_pair_family_not_closed"
    )
    assert result["gate"]["metrics"]["control_contains_anchor_family"] is True


def test_e23_anchor_allows_half_rank_noise_when_joint_span_is_exact() -> None:
    config = SkinnyR8BottomRowClosureConfig(run_id="half-noise")
    masks = {
        0: EXPECTED_ANCHOR_MASKS[(12, 13)],
        3: EXPECTED_ANCHOR_MASKS[(13, 14)],
        5: EXPECTED_ANCHOR_MASKS[(14, 15)],
    }
    parity_rows = _synthetic_rows(config, masks_by_pair=masks)
    paper = EXPECTED_ANCHOR_MASKS[(14, 15)]
    extra = 1 << 63
    validation_rows = gf2_kernel_basis(np.asarray([paper, extra], dtype=np.uint64))
    parity_rows[5, config.discovery_keys :] = np.resize(
        np.asarray(validation_rows, dtype=np.uint64), config.validation_keys
    )

    result = _evaluate(config, parity_rows)

    anchor = result["rows"][5]
    assert anchor["validation_nullity"] == 2
    assert anchor["joint_nullity"] == 1
    assert all(result["gate"]["anchor_reproduction_checks"].values())
    assert result["gate"]["status"] == "hold"


def test_e23_runner_returns_complete_result_with_stubbed_rows(monkeypatch) -> None:
    config = SkinnyR8BottomRowClosureConfig(run_id="runner")
    masks = {
        0: EXPECTED_ANCHOR_MASKS[(12, 13)],
        3: EXPECTED_ANCHOR_MASKS[(13, 14)],
        5: EXPECTED_ANCHOR_MASKS[(14, 15)],
    }
    parity_rows = _synthetic_rows(config, masks_by_pair=masks)

    def fake_collect(*args, pair_index, **kwargs):
        return parity_rows[pair_index].copy()

    monkeypatch.setattr(closure, "_collect_pair_rows", fake_collect)
    result = closure.run_skinny_r8_bottom_row_closure_audit(config)

    assert len(result["rows"]) == 7
    assert result["keys"].shape == (128,)
    assert result["parity_rows"].shape == (7, 128)
    assert result["metadata"]["training_performed"] is False


def test_e23_cli_writes_complete_artifact_set(monkeypatch, tmp_path) -> None:
    config = SkinnyR8BottomRowClosureConfig(run_id="cli")
    masks = {
        0: EXPECTED_ANCHOR_MASKS[(12, 13)],
        3: EXPECTED_ANCHOR_MASKS[(13, 14)],
        5: EXPECTED_ANCHOR_MASKS[(14, 15)],
    }
    result = _evaluate(config, _synthetic_rows(config, masks_by_pair=masks))
    monkeypatch.setattr(
        cli,
        "run_skinny_r8_bottom_row_closure_audit",
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
    config: SkinnyR8BottomRowClosureConfig,
    parity_rows: np.ndarray,
) -> dict:
    return evaluate_skinny_r8_bottom_row_closure(
        config,
        keys=tuple(range(config.total_keys)),
        prior_key_sets={
            "e20": tuple(range(10_000, 10_768)),
            "e21": tuple(range(20_000, 20_768)),
            "e22": tuple(range(30_000, 30_128)),
        },
        base_plaintext=0,
        parity_rows=parity_rows,
    )


def _synthetic_rows(
    config: SkinnyR8BottomRowClosureConfig,
    *,
    masks_by_pair: dict[int, int],
) -> np.ndarray:
    full_rank = np.asarray([1 << bit for bit in range(64)], dtype=np.uint64)
    rows = np.empty((len(AUDIT_PAIRS), config.total_keys), dtype=np.uint64)
    for pair_index in range(len(AUDIT_PAIRS)):
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
