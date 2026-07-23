from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from blockcipher_nd.ciphers.spn.skinny import (
    Skinny64,
    cells_to_int,
    int_to_cells,
    mix_columns,
    shift_rows,
)
from blockcipher_nd.data.cache import make_chunked_differential_dataset
from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.data.differential.rows import (
    generate_negative_row,
    generate_positive_row,
)
from blockcipher_nd.models.structure.spn.runtime_structure import apply_gf2
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    skinny64_runtime_structure,
)
from blockcipher_nd.registry.cipher_factory import build_cipher, default_difference
from blockcipher_nd.registry.model_factory import build_model


RUN_ID = "i1_rtg1_skinny64_general_gf2_data_readiness_t2a_20260724"
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SkinnyRuntimeReadinessConfig:
    run_id: str = RUN_ID
    rounds: int = 7
    input_difference: int = 0x0000000000000040
    train_samples_per_class: int = 64
    validation_samples_per_class: int = 32
    train_seed: int = 0
    validation_seed: int = 1
    train_key: int = 0x0000000000000000
    validation_key: int = 0x1111111111111111
    pairs_per_sample: int = 4
    chunk_size: int = 16

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("T2-A freezes the SKINNY data fixture at seven rounds")
        if self.input_difference != 0x40:
            raise ValueError("T2-A freezes the adapter input-difference fixture at 0x40")
        if (self.train_samples_per_class, self.validation_samples_per_class) != (
            64,
            32,
        ):
            raise ValueError("T2-A freezes 64/class train and 32/class validation")
        if self.pairs_per_sample != 4:
            raise ValueError("T2-A freezes four ciphertext pairs per sample")


class _RecordingSkinny64:
    name = "SKINNY-64/64"
    structure = "SPN"
    block_bits = 64
    key_bits = 64

    def __init__(self, *, rounds: int, key: int) -> None:
        self.rounds = rounds
        self.key = key
        self.plaintexts: list[int] = []
        self._cipher = Skinny64(rounds=rounds, key=key)

    def encrypt(self, plaintext: int) -> int:
        self.plaintexts.append(plaintext)
        return self._cipher.encrypt(plaintext)


def run_skinny_runtime_readiness(
    config: SkinnyRuntimeReadinessConfig,
    *,
    cache_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, bool]] = {
        "cipher_factory": {},
        "strict_dataset": {},
        "cache_replay": {},
        "runtime_model": {},
        "general_gf2": {},
    }
    observations: dict[str, Any] = {}

    direct_public = Skinny64(rounds=32, key=0xF5269826FC681238)
    factory_public = build_cipher(
        "skinny64", rounds=32, key=0xF5269826FC681238
    )
    public_plaintext = 0x06034F957724D19D
    public_ciphertext = 0xBB39DFB2429B8AC7
    checks["cipher_factory"]["public_appendix_b_vector_exact"] = (
        direct_public.encrypt(public_plaintext)
        == factory_public.encrypt(public_plaintext)
        == public_ciphertext
    )
    replay_cases = (
        (0, 0x0000000000000000, 0x0000000000000000),
        (1, 0x0123456789ABCDEF, 0xFEDCBA9876543210),
        (7, config.train_key, 0x1122334455667788),
        (7, config.validation_key, 0x8877665544332211),
        (32, 0xF5269826FC681238, public_plaintext),
    )
    checks["cipher_factory"]["factory_direct_encrypt_replay_exact"] = all(
        build_cipher("skinny64", rounds, key=key).encrypt(plaintext)
        == Skinny64(rounds=rounds, key=key).encrypt(plaintext)
        for rounds, key, plaintext in replay_cases
    )
    checks["cipher_factory"]["default_difference_fixture_exact"] = (
        default_difference("skinny64") == config.input_difference
    )

    semantic_checks, semantic_observations = _dataset_semantics(config)
    checks["strict_dataset"].update(semantic_checks)
    observations.update(semantic_observations)

    train_config = _dataset_config(
        config,
        samples_per_class=config.train_samples_per_class,
        seed=config.train_seed,
        key=config.train_key,
    )
    validation_config = _dataset_config(
        config,
        samples_per_class=config.validation_samples_per_class,
        seed=config.validation_seed,
        key=config.validation_key,
    )
    datasets: dict[str, Any] = {}
    for split, dataset_config in (
        ("train", train_config),
        ("validation", validation_config),
    ):
        memory = make_differential_dataset(dataset_config)
        disk = make_chunked_differential_dataset(
            dataset_config,
            cache_dir=cache_root / split,
            chunk_size=config.chunk_size,
            workers=1,
            reuse=True,
            progress_callback=progress_callback,
            progress_context={"run_id": config.run_id, "split": split},
        )
        reused = make_chunked_differential_dataset(
            dataset_config,
            cache_dir=cache_root / split,
            chunk_size=config.chunk_size,
            workers=1,
            reuse=True,
            progress_callback=progress_callback,
            progress_context={"run_id": config.run_id, "split": split},
        )
        expected_rows = dataset_config.samples_per_class * 2
        expected_bits = config.pairs_per_sample * 128
        checks["strict_dataset"][f"{split}_shape_exact"] = (
            tuple(disk.features.shape) == (expected_rows, expected_bits)
            and tuple(disk.labels.shape) == (expected_rows,)
        )
        checks["strict_dataset"][f"{split}_balanced_labels_exact"] = (
            int(np.asarray(disk.labels).sum()) == dataset_config.samples_per_class
            and set(np.unique(np.asarray(disk.labels)).tolist()) == {0, 1}
        )
        checks["strict_dataset"][f"{split}_metadata_exact"] = (
            disk.metadata["cipher"] == "SKINNY-64/64"
            and disk.metadata["rounds"] == config.rounds
            and disk.metadata["negative_mode"] == "encrypted_random_plaintexts"
            and disk.metadata["sample_structure"] == "independent_pairs"
            and disk.metadata["pairs_per_sample"] == config.pairs_per_sample
            and disk.metadata["key_schedule"] == "fixed"
        )
        checks["cache_replay"][f"{split}_disk_matches_fresh_memory"] = (
            np.array_equal(np.asarray(disk.features), memory.features)
            and np.array_equal(np.asarray(disk.labels), memory.labels)
        )
        checks["cache_replay"][f"{split}_parameter_matched_reuse"] = (
            reused.metadata["cache_status"] == "reused"
            and np.array_equal(np.asarray(reused.features), np.asarray(disk.features))
            and np.array_equal(np.asarray(reused.labels), np.asarray(disk.labels))
        )
        datasets[split] = disk

    checks["strict_dataset"]["train_validation_keys_and_seeds_distinct"] = (
        config.train_key != config.validation_key
        and config.train_seed != config.validation_seed
    )
    checks["cache_replay"]["cache_files_complete"] = all(
        (cache_root / split / filename).is_file()
        for split in ("train", "validation")
        for filename in ("features.npy", "labels.npy", "metadata.json")
    )

    structure = skinny64_runtime_structure(rounds=2)
    general_checks, general_observations = _general_gf2_checks(structure)
    checks["general_gf2"].update(general_checks)
    observations.update(general_observations)

    model_names = (
        "skinny64_runtime_e4_equivariant_true",
        "skinny64_runtime_e4_equivariant_corrupted",
        "skinny64_runtime_e4_equivariant_independent",
    )
    models = [
        build_model(
            name,
            input_bits=config.pairs_per_sample * 128,
            hidden_bits=64,
            pair_bits=128,
            structure="SPN",
            model_options={
                "processor_steps": 2,
                "pair_embedding_dim": 128,
                "dropout": 0.0,
                "sbox_context_mode": "late_pair",
            },
        ).eval()
        for name in model_names
    ]
    geometries = [
        {name: tuple(value.shape) for name, value in model.state_dict().items()}
        for model in models
    ]
    parameter_counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    checks["runtime_model"]["three_controls_equal_parameter_geometry"] = (
        all(geometry == geometries[0] for geometry in geometries)
        and len(set(parameter_counts)) == 1
    )
    checks["runtime_model"]["runtime_structure_not_in_state_dict"] = all(
        not any(
            token in name
            for name in geometry
            for token in ("linear_matrices", "sbox_truth", "cell_membership", "bit_role")
        )
        for geometry in geometries
    )
    checks["runtime_model"]["controls_change_only_runtime_relation"] = (
        not torch.equal(
            models[0].runtime_structure.linear_matrices,
            models[1].runtime_structure.linear_matrices,
        )
        and models[0].relation_mode == models[1].relation_mode == "true"
        and models[2].relation_mode == "independent"
    )
    feature_batch = torch.from_numpy(
        np.asarray(datasets["train"].features[:4]).copy()
    ).float()
    with torch.no_grad():
        logits = [model(feature_batch) for model in models]
    checks["runtime_model"]["three_control_forward_shapes_and_values_finite"] = all(
        tuple(logit.shape) == (4, 1) and bool(torch.isfinite(logit).all())
        for logit in logits
    )
    observations["parameter_count"] = parameter_counts[0]
    observations["model_names"] = list(model_names)

    rows = [
        {
            "run_id": config.run_id,
            "category": category,
            "check": check,
            "passed": passed,
            "training_performed": False,
        }
        for category, category_checks in checks.items()
        for check, passed in category_checks.items()
    ]
    category_counts = {
        category: {
            "passed": sum(category_checks.values()),
            "total": len(category_checks),
        }
        for category, category_checks in checks.items()
    }
    all_passed = all(row["passed"] for row in rows)
    decision = (
        "innovation1_runtime_spn_skinny_general_gf2_data_ready"
        if all_passed
        else "innovation1_runtime_spn_skinny_general_gf2_data_not_ready"
    )
    gate = {
        "run_id": config.run_id,
        "status": "pass" if all_passed else "hold",
        "decision": decision,
        "checks_passed": sum(row["passed"] for row in rows),
        "checks_total": len(rows),
        "category_counts": category_counts,
        "training_performed": False,
        "empirical_topology_superiority_tested": False,
        "claim_scope": (
            "local SKINNY-64/64 cipher/data/cache/runtime-model readiness for a "
            "general GF(2) linear layer; no AUC, topology-superiority, formal-scale, "
            "paper-reproduction, attack, or SOTA claim"
        ),
        "next_action": (
            "select a literature-backed or preregistered signal-bearing SKINNY "
            "round/difference protocol before any neural training"
            if all_passed
            else "repair only the failed SKINNY adapter/readiness checks; do not train"
        ),
        "blocked_actions": [
            "neural training before protocol selection",
            "remote GPU launch",
            "reuse Innovation 2 integral labels",
            "claim general-GF(2) topology superiority",
        ],
    }
    metadata = {
        "run_id": config.run_id,
        "task": "innovation1_runtime_spn_skinny_general_gf2_data_readiness_t2a",
        "cipher": "SKINNY-64/64",
        "config": asdict(config),
        "training_performed": False,
        "dataset_contract": {
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
            "cache_root": str(cache_root),
        },
        "observations": observations,
        "claim_scope": gate["claim_scope"],
    }
    return {
        "rows": rows,
        "metadata": metadata,
        "gate": gate,
        "summary": {
            "run_id": config.run_id,
            "metadata": metadata,
            "gate": gate,
            "linear_row_degrees": general_observations["linear_row_degrees"],
        },
    }


def _dataset_config(
    config: SkinnyRuntimeReadinessConfig,
    *,
    samples_per_class: int,
    seed: int,
    key: int,
) -> DifferentialDatasetConfig:
    return DifferentialDatasetConfig(
        cipher=build_cipher("skinny64", config.rounds, key=key),
        input_difference=config.input_difference,
        samples_per_class=samples_per_class,
        seed=seed,
        shuffle=False,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=config.pairs_per_sample,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="independent_pairs",
    )


def _dataset_semantics(
    config: SkinnyRuntimeReadinessConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    positive_cipher = _RecordingSkinny64(rounds=config.rounds, key=config.train_key)
    positive_config = DifferentialDatasetConfig(
        cipher=positive_cipher,
        input_difference=config.input_difference,
        samples_per_class=1,
        seed=config.train_seed,
        shuffle=False,
        pairs_per_sample=config.pairs_per_sample,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="independent_pairs",
    )
    positive_row = generate_positive_row(
        positive_config,
        np.random.default_rng(config.train_seed),
        64,
        (1 << 64) - 1,
    )
    positive_pairs = tuple(
        zip(
            positive_cipher.plaintexts[::2],
            positive_cipher.plaintexts[1::2],
            strict=True,
        )
    )

    negative_cipher = _RecordingSkinny64(rounds=config.rounds, key=config.train_key)
    negative_config = DifferentialDatasetConfig(
        cipher=negative_cipher,
        input_difference=config.input_difference,
        samples_per_class=1,
        seed=config.train_seed,
        shuffle=False,
        pairs_per_sample=config.pairs_per_sample,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="independent_pairs",
    )
    negative_row = generate_negative_row(
        negative_config,
        np.random.default_rng(config.train_seed + 17),
        64,
    )
    negative_pairs = tuple(
        zip(
            negative_cipher.plaintexts[::2],
            negative_cipher.plaintexts[1::2],
            strict=True,
        )
    )
    return (
        {
            "positive_pairs_use_exact_input_xor_difference": (
                len(positive_cipher.plaintexts) == 2 * config.pairs_per_sample
                and all(
                    plaintext_a ^ plaintext_b == config.input_difference
                    for plaintext_a, plaintext_b in positive_pairs
                )
            ),
            "negative_pairs_encrypt_two_plaintexts_each": (
                len(negative_cipher.plaintexts) == 2 * config.pairs_per_sample
                and len(negative_pairs) == config.pairs_per_sample
                and all(
                    negative_cipher._cipher.encrypt(plaintext) >= 0
                    for pair in negative_pairs
                    for plaintext in pair
                )
            ),
            "semantic_fixture_rows_have_expected_width": (
                len(positive_row)
                == len(negative_row)
                == config.pairs_per_sample * 128
            ),
            "negative_plaintexts_are_not_fixed_difference_pairs": any(
                plaintext_a ^ plaintext_b != config.input_difference
                for plaintext_a, plaintext_b in negative_pairs
            ),
        },
        {
            "positive_encrypt_calls": len(positive_cipher.plaintexts),
            "negative_encrypt_calls": len(negative_cipher.plaintexts),
        },
    )


def _general_gf2_checks(
    structure: Any,
) -> tuple[dict[str, bool], dict[str, Any]]:
    linear = structure.linear_matrices[0]
    inverse = structure.inverse_linear_matrices[0]
    identity = torch.eye(64, dtype=torch.int64)
    states = (
        0x0000000000000000,
        0x0000000000000001,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
        0xFFFFFFFFFFFFFFFF,
    )
    bits = torch.tensor(
        [[(state >> bit) & 1 for bit in range(64)] for state in states],
        dtype=torch.float32,
    )
    transformed = apply_gf2(linear, bits)
    transformed_ints = tuple(
        sum(int(value) << bit for bit, value in enumerate(row.tolist()))
        for row in transformed.to(torch.uint8)
    )
    direct_ints = tuple(
        cells_to_int(mix_columns(shift_rows(int_to_cells(state)))) for state in states
    )
    round_trip = apply_gf2(inverse, transformed)
    corrupted = structure.corrupted()
    row_degrees = linear.sum(dim=1).tolist()
    column_degrees = linear.sum(dim=0).tolist()
    return (
        {
            "runtime_matrix_replays_shiftrows_mixcolumns_exactly": (
                transformed_ints == direct_ints
            ),
            "forward_inverse_matrix_products_are_identity": (
                torch.equal(
                    torch.remainder(linear.to(torch.int64) @ inverse.to(torch.int64), 2),
                    identity,
                )
                and torch.equal(
                    torch.remainder(inverse.to(torch.int64) @ linear.to(torch.int64), 2),
                    identity,
                )
            ),
            "forward_inverse_round_trip_exact": torch.equal(round_trip, bits),
            "linear_layer_is_general_not_permutation": (
                max(row_degrees) > 1 or max(column_degrees) > 1
            ),
            "corruption_is_deterministic_and_changes_edges": (
                not torch.equal(linear, corrupted.linear_matrices[0])
                and torch.equal(
                    corrupted.linear_matrices,
                    structure.corrupted().linear_matrices,
                )
            ),
        },
        {
            "linear_row_degrees": [int(value) for value in row_degrees],
            "linear_column_degrees": [int(value) for value in column_degrees],
            "maximum_row_degree": int(max(row_degrees)),
            "maximum_column_degree": int(max(column_degrees)),
        },
    )


__all__ = [
    "RUN_ID",
    "SkinnyRuntimeReadinessConfig",
    "run_skinny_runtime_readiness",
]
