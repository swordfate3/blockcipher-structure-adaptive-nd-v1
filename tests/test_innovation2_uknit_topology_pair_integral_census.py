import json

import numpy as np

from blockcipher_nd.cli import (
    audit_innovation2_uknit_topology_pair_integral_census as cli,
)
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import gf2_kernel_basis
from blockcipher_nd.tasks.innovation2.uknit_linear_integral_census import (
    collect_uknit_integral_parity_rows,
)
from blockcipher_nd.tasks.innovation2.uknit_topology_pair_integral_census import (
    CONTROL_PAIRS,
    PAIR_STRUCTURES,
    ROUNDS,
    TOPOLOGY_PAIRS,
    UknitTopologyPairCensusConfig,
    evaluate_uknit_topology_pair_integral_census,
)


def _full_rank_words(count: int) -> np.ndarray:
    return np.resize(
        np.asarray([1 << bit for bit in range(64)], dtype=np.uint64),
        count,
    )


def _synthetic_result(*, signal_round: int = 5, signal_index: int = 0):
    config = UknitTopologyPairCensusConfig(
        run_id="synthetic-pairs",
        discovery_trials=64,
        validation_trials=64,
    )
    shape = (len(ROUNDS), len(PAIR_STRUCTURES), config.total_trials)
    full_rank = _full_rank_words(config.total_trials)
    parity = np.empty(shape, dtype=np.uint64)
    controls = np.empty(shape, dtype=np.uint64)
    for round_index, rounds in enumerate(ROUNDS):
        for pair_index in range(len(PAIR_STRUCTURES)):
            parity[round_index, pair_index] = 0 if rounds == 1 else full_rank
            controls[round_index, pair_index] = full_rank
    mask = (1 << 7) | (1 << 23) | (1 << 47)
    orthogonal = gf2_kernel_basis(np.asarray([mask], dtype=np.uint64))
    parity[ROUNDS.index(signal_round), signal_index] = np.resize(
        np.asarray(orthogonal, dtype=np.uint64),
        config.total_trials,
    )
    return evaluate_uknit_topology_pair_integral_census(
        config,
        keys=tuple(range(config.total_trials)),
        base_plaintexts=np.arange(config.total_trials, dtype=np.uint64),
        parity_rows=parity,
        random_control_rows=controls,
    )


def test_topology_pair_ownership_matches_four_quartets_and_four_controls() -> None:
    assert len(TOPOLOGY_PAIRS) == 24
    assert len(CONTROL_PAIRS) == 4
    assert len(PAIR_STRUCTURES) == 28
    assert len(set(PAIR_STRUCTURES)) == 28
    assert all(pair not in TOPOLOGY_PAIRS for pair in CONTROL_PAIRS)


def test_pair_census_reports_r5_round_extension_after_fresh_validation() -> None:
    result = _synthetic_result(signal_round=5, signal_index=0)

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["highest_supported_round"] == 5
    assert result["gate"]["highest_supported_pairs"] == ["0+1"]
    assert result["gate"]["topology_coherent_pair_extended_beyond_r4"] is True
    signal = next(
        row
        for row in result["rows"]
        if row["rounds"] == 5 and row["active_pair"] == "0+1"
    )
    assert signal["joint_rank"] == 63
    assert signal["joint_nullity"] == 1
    assert signal["discovery_basis_validation_survivors"] == 1
    assert signal["post_selection_false_accept_log2_bound"] == -63


def test_pair_census_holds_when_pair_width_does_not_extend_beyond_r4() -> None:
    result = _synthetic_result(signal_round=4, signal_index=0)

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_uknit_pair_linear_kernel_no_round_extension"
    )
    assert result["gate"]["highest_supported_round"] == 4


def test_generic_collector_r1_two_cell_integral_is_zero() -> None:
    rows = collect_uknit_integral_parity_rows(
        rounds=(1,),
        structures=((0, 1),),
        keys=(0,),
        base_plaintexts=np.asarray([0x123456789ABCDEF0], dtype=np.uint64),
        trial_chunk_size=1,
    )

    assert rows.shape == (1, 1, 1)
    assert int(rows[0, 0, 0]) == 0


def test_pair_census_cli_writes_complete_artifact_set(monkeypatch, tmp_path) -> None:
    result = _synthetic_result(signal_round=5, signal_index=0)
    monkeypatch.setattr(
        cli,
        "run_uknit_topology_pair_integral_census",
        lambda *args, **kwargs: result,
    )
    output_root = tmp_path / "result"

    exit_code = cli.main(
        [
            "--run-id",
            "synthetic-pairs",
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
    assert gate["highest_supported_round"] == 5
    assert "拓扑双 cell" in (output_root / "curves.svg").read_text(
        encoding="utf-8"
    )
