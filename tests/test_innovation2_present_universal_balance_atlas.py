from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.cli.audit_innovation2_present_universal_balance_atlas import (
    main as audit_main,
)
from blockcipher_nd.cli.plot_innovation2_present_universal_balance_atlas import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    UniversalBalanceAtlasConfig,
    _scalar_parity_word,
    _select_checkerboards,
    build_checkerboard_benchmark,
    build_raw_atlas,
    checkerboard_balance,
    make_output_masks,
    make_structures,
    possible_active_monomials,
    reconstruct_present_sbox_from_anf,
)


def smoke_config(**overrides: object) -> UniversalBalanceAtlasConfig:
    values = {
        "run_id": "i2_present_universal_balance_atlas_smoke_test",
        "mode": "smoke",
        "structure_count": 8,
        "witness_keys": 4,
        "offsets_per_structure": 2,
        "match_attempts": 4,
    }
    values.update(overrides)
    return UniversalBalanceAtlasConfig(**values)


def test_present_sbox_anf_reconstructs_truth_table() -> None:
    assert tuple(reconstruct_present_sbox_from_anf(value) for value in range(16)) == (
        PRESENT_SBOX
    )


def test_output_mask_fixture_is_unique_and_covers_multi_bit_families() -> None:
    masks = make_output_masks()

    assert len(masks) == 300
    assert len({mask.value for mask in masks}) == 300
    assert {mask.family for mask in masks} == {
        "unit",
        "nibble",
        "player_pair",
        "same_nibble_pair",
        "adjacent_nibble_pair",
    }
    assert min(mask.value for mask in masks) > 0
    assert max(mask.value.bit_count() for mask in masks) == 4


def test_absent_full_cube_support_certificates_match_scalar_fixtures() -> None:
    active_bits = tuple(range(8))
    supports = possible_active_monomials(active_bits, rounds=4)
    full_cube = (1 << len(active_bits)) - 1
    certified_bits = [bit for bit, support in enumerate(supports) if full_cube not in support]

    assert certified_bits
    for key in (0, 1, 0x0123456789ABCDEF0123):
        for offset in (0, 0xFF00FF00FF00FF00):
            parity_word = _scalar_parity_word(active_bits, 4, key, offset & ~0xFF)
            assert all(not parity_word & (1 << bit) for bit in certified_bits)


def test_negative_atlas_witness_revalidates_with_scalar_present() -> None:
    config = smoke_config(structure_count=25)
    structures = make_structures(config)
    masks = make_output_masks()
    raw = build_raw_atlas(config, structures, masks)
    negative = next(row for row in raw["rows"] if row["status"] == "negative")

    structure = structures[negative["structure_index"]]
    mask = masks[negative["mask_index"]]
    parity_word = _scalar_parity_word(
        structure.active_bits,
        config.rounds,
        int(negative["witness_key_hex"], 16),
        int(negative["witness_offset_hex"], 16),
    )

    assert (parity_word & mask.value).bit_count() & 1 == 1


def test_checkerboard_selector_balances_each_structure_and_mask() -> None:
    labels = np.asarray(
        [
            [1, 0, -1, -1],
            [0, 1, -1, -1],
            [-1, -1, 1, 0],
            [-1, -1, 0, 1],
        ],
        dtype=np.int8,
    )

    edges, rectangles = _select_checkerboards(
        labels=labels,
        structure_indices=(0, 1, 2, 3),
        attempts=4,
        seed=7,
    )
    rows = [
        {
            "split": "train",
            "structure_index": structure,
            "mask_index": mask,
            "label": int(labels[structure, mask]),
        }
        for structure, mask in edges
    ]

    assert len(rectangles) == 2
    assert len(edges) == 8
    assert checkerboard_balance(rows) == {
        "duplicate_edges": 0,
        "maximum_structure_class_delta": 0,
        "maximum_mask_class_delta": 0,
    }


def test_smoke_benchmark_has_disjoint_structure_splits() -> None:
    config = smoke_config(structure_count=16, match_attempts=8)
    structures = make_structures(config)
    masks = make_output_masks()
    raw = build_raw_atlas(config, structures, masks)
    matched = build_checkerboard_benchmark(
        labels=raw["labels"],
        structures=structures,
        masks=masks,
        attempts=config.match_attempts,
    )

    assert not set(matched["split_indices"]["train"]).intersection(
        matched["split_indices"]["validation"]
    )
    assert matched["balance"]["duplicate_edges"] == 0
    assert matched["balance"]["maximum_structure_class_delta"] == 0
    assert matched["balance"]["maximum_mask_class_delta"] == 0


def test_cli_writes_auditable_smoke_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "atlas"
    exit_code = audit_main(
        [
            "--run-id",
            "i2_present_universal_balance_atlas_smoke_test",
            "--output-root",
            str(output),
            "--mode",
            "smoke",
            "--structure-count",
            "8",
            "--witness-keys",
            "4",
            "--offsets-per-structure",
            "2",
            "--match-attempts",
            "4",
        ]
    )

    assert exit_code == 0
    expected = {
        "atlas.jsonl",
        "matched_contrast.csv",
        "structures.json",
        "masks.json",
        "metadata.json",
        "summary.json",
        "results.jsonl",
        "gate.json",
        "progress.jsonl",
    }
    assert expected.issubset(path.name for path in output.iterdir())
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_universal_balance_atlas_too_narrow"
    atlas_rows = [
        json.loads(line)
        for line in (output / "atlas.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(atlas_rows) == 8 * 300
    with (output / "matched_contrast.csv").open(encoding="utf-8") as handle:
        list(csv.DictReader(handle))

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
    assert "创新2 E43" in svg


def test_official_present_fixture_remains_valid() -> None:
    assert Present80(rounds=31, key=0).encrypt(0) == 0x5579C1387B228445
