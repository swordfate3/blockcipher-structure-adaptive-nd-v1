import json

import numpy as np

from blockcipher_nd.cli import audit_innovation2_uknit_linear_integral_census as cli
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import gf2_kernel_basis
from blockcipher_nd.tasks.innovation2.uknit_linear_integral_census import (
    ACTIVE_CELLS,
    ROUNDS,
    TARGET_ROUNDS,
    UknitLinearIntegralCensusConfig,
    evaluate_uknit_linear_integral_census,
)


def _full_rank_words(count: int) -> np.ndarray:
    return np.resize(
        np.asarray([1 << bit for bit in range(64)], dtype=np.uint64),
        count,
    )


def _synthetic_result(*, signal_round: int | None = 6, signal_cell: int = 7):
    config = UknitLinearIntegralCensusConfig(
        run_id="synthetic",
        discovery_trials=64,
        validation_trials=64,
    )
    shape = (len(ROUNDS), len(ACTIVE_CELLS), config.total_trials)
    parity = np.empty(shape, dtype=np.uint64)
    controls = np.empty(shape, dtype=np.uint64)
    full_rank = _full_rank_words(config.total_trials)
    for round_index, rounds in enumerate(ROUNDS):
        for cell in ACTIVE_CELLS:
            parity[round_index, cell] = 0 if rounds == 1 else full_rank
            controls[round_index, cell] = full_rank
    if signal_round is not None:
        mask = (1 << 3) | (1 << 19) | (1 << 51)
        orthogonal = gf2_kernel_basis(np.asarray([mask], dtype=np.uint64))
        signal_words = np.resize(
            np.asarray(orthogonal, dtype=np.uint64),
            config.total_trials,
        )
        parity[ROUNDS.index(signal_round), signal_cell] = signal_words
    keys = tuple(range(config.total_trials))
    bases = np.arange(config.total_trials, dtype=np.uint64)
    return evaluate_uknit_linear_integral_census(
        config,
        keys=keys,
        base_plaintexts=bases,
        parity_rows=parity,
        random_control_rows=controls,
    )


def test_census_accepts_independently_validated_kernel_and_reports_highest_round() -> None:
    result = _synthetic_result(signal_round=6, signal_cell=7)

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["highest_supported_round"] == 6
    assert result["gate"]["highest_supported_cells"] == "7"
    signal = next(
        row
        for row in result["rows"]
        if row["rounds"] == 6 and row["active_cell"] == 7
    )
    assert signal["joint_rank"] == 63
    assert signal["joint_nullity"] == 1
    assert signal["discovery_basis_validation_survivors"] == 1
    assert signal["post_selection_false_accept_log2_bound"] == -63
    assert signal["stable_nontrivial_kernel"] is True


def test_census_holds_when_all_target_rounds_are_full_rank() -> None:
    result = _synthetic_result(signal_round=None)

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_uknit_single_cell_linear_kernel_exhausted"
    )
    assert result["gate"]["highest_supported_round"] is None
    assert all(result["gate"]["readiness_checks"].values())


def test_census_protocol_requires_r1_zero_parity_calibration() -> None:
    result = _synthetic_result(signal_round=6)
    broken = result["parity_rows"].copy()
    broken[ROUNDS.index(1), 0, 0] = 1
    config = UknitLinearIntegralCensusConfig(
        run_id="broken",
        discovery_trials=64,
        validation_trials=64,
    )
    reevaluated = evaluate_uknit_linear_integral_census(
        config,
        keys=result["keys"],
        base_plaintexts=result["base_plaintexts"],
        parity_rows=broken,
        random_control_rows=result["random_control_rows"],
    )

    assert reevaluated["gate"]["status"] == "fail"
    assert not reevaluated["gate"]["readiness_checks"][
        "r1_all_cells_rank0_nullity64"
    ]


def test_census_cli_writes_complete_artifact_set(monkeypatch, tmp_path) -> None:
    result = _synthetic_result(signal_round=6)
    monkeypatch.setattr(
        cli,
        "run_uknit_linear_integral_census",
        lambda *args, **kwargs: result,
    )
    output_root = tmp_path / "result"

    exit_code = cli.main(
        [
            "--run-id",
            "synthetic",
            "--output-root",
            str(output_root),
            "--discovery-trials",
            "64",
            "--validation-trials",
            "64",
        ]
    )

    assert exit_code == 0
    assert {
        "base_plaintexts.npy",
        "curves.svg",
        "gate.json",
        "kernel_basis.csv",
        "keys.npy",
        "metadata.json",
        "parity_rows.npy",
        "progress.jsonl",
        "random_control_rows.npy",
        "results.jsonl",
        "round_summary.csv",
    } <= {path.name for path in output_root.iterdir()}
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    assert gate["highest_supported_round"] == 6
    assert np.load(output_root / "keys.npy", allow_pickle=False).shape == (128, 2)
    assert "uKNIT-BC 单活动 cell" in (output_root / "curves.svg").read_text(
        encoding="utf-8"
    )
    assert list(TARGET_ROUNDS) == list(range(3, 12))
