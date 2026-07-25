from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from blockcipher_nd.cli import run_runtime_spn_cross_cipher_linear_probe as cli
from blockcipher_nd.data.differential import (
    DifferentialDataset,
    DiskDifferentialDataset,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_cross_cipher_linear_probe as probe,
)
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_skinny_rectangle_transfer as transfer,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_head_adaptation import (
    tensor_mapping_sha256,
)


def _gate_row(role: str, auc: float, *, seed: int) -> dict[str, object]:
    source_role, target_mode = probe.ROLE_SPECS[role]
    source_checkpoint_sha256 = {
        "true": transfer.SOURCE_CHECKPOINT_SHA256S["true"],
        "corrupted": transfer.SOURCE_CHECKPOINT_SHA256S["corrupted"],
        "random": None,
    }[source_role]
    raw_character = "a" if seed == 0 else "b"
    validation_character = "c" if seed == 0 else "d"
    probe_initial = "3" * 64 if seed == 0 else "4" * 64
    return {
        "seed": seed,
        "role": role,
        "source_role": source_role,
        "target_mode": target_mode,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "target_structure_sha256": "1" * 64 if target_mode == "true" else "2" * 64,
        "feature_extractor_initial_sha256": "5" * 64,
        "feature_extractor_final_sha256": "5" * 64,
        "source_classifier_initial_sha256": "6" * 64,
        "source_classifier_final_sha256": "6" * 64,
        "probe_initial_sha256": probe_initial,
        "probe_final_sha256": "7" * 64,
        "checkpoint_sha256": "8" * 64,
        "checkpoint_replay_verified": True,
        "parameter_count": probe.PROBE_PARAMETER_COUNT,
        "trainable_parameter_count": probe.PROBE_PARAMETER_COUNT,
        "trainable_parameter_names": ["weight", "bias"],
        "adapter_mode": "frozen_runtime_e4_linear_probe",
        "feature_extractor_frozen": True,
        "source_classifier_preserved": True,
        "representation_width": probe.REPRESENTATION_WIDTH,
        "train_representation_sha256": "9" * 64,
        "validation_representation_sha256": "0" * 64,
        "train_representation_metadata_sha256": "e" * 64,
        "validation_representation_metadata_sha256": "f" * 64,
        "train_representation_cache_reuse_verified": True,
        "validation_representation_cache_reuse_verified": True,
        "raw_train_feature_sha256": raw_character * 64,
        "raw_train_label_sha256": "1" * 64,
        "raw_train_metadata_sha256": raw_character * 64,
        "raw_validation_feature_sha256": validation_character * 64,
        "raw_validation_label_sha256": "2" * 64,
        "raw_validation_metadata_sha256": validation_character * 64,
        "auc": auc,
        "source_seed": 0,
        "target_train_seed": seed,
        "target_validation_seed": seed + 10_000,
        "target_cipher": "RECTANGLE-80",
        "target_rounds": 6,
        "train_rows": 4096,
        "validation_rows": 2048,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "training": {
            "epochs": probe.EPOCHS,
            "batch_size": probe.BATCH_SIZE,
            "learning_rate": probe.LEARNING_RATE,
            "weight_decay": probe.WEIGHT_DECAY,
            "selected_checkpoint": "best",
        },
        "history": [
            {
                "epoch": float(epoch),
                "train_loss": 0.25,
                "train_auc": auc - 0.01,
                "val_loss": 0.69,
                "val_auc": auc - 0.02 + 0.0002 * epoch,
                "val_accuracy": 0.5,
                "learning_rate": probe.LEARNING_RATE,
            }
            for epoch in range(1, probe.EPOCHS + 1)
        ],
    }


def _passing_rows() -> list[dict[str, object]]:
    aucs = {
        0: {
            "true_source_true_target": 0.66,
            "corrupted_source_true_target": 0.61,
            "true_source_corrupted_target": 0.59,
            "random_source_true_target": 0.57,
        },
        1: {
            "true_source_true_target": 0.64,
            "corrupted_source_true_target": 0.60,
            "true_source_corrupted_target": 0.58,
            "random_source_true_target": 0.56,
        },
    }
    return [
        _gate_row(role, aucs[seed][role], seed=seed)
        for seed in probe.TARGET_SEEDS
        for role in probe.EXPECTED_ROLES
    ]


def test_linear_probe_has_exactly_385_parameters_and_seeded_initialization() -> None:
    seed0_first = probe.deterministic_linear_probe(0)
    seed0_second = probe.deterministic_linear_probe(0)
    seed1 = probe.deterministic_linear_probe(1)

    assert sum(parameter.numel() for parameter in seed0_first.parameters()) == 385
    assert tensor_mapping_sha256(seed0_first.state_dict()) == tensor_mapping_sha256(
        seed0_second.state_dict()
    )
    assert tensor_mapping_sha256(seed0_first.state_dict()) != tensor_mapping_sha256(
        seed1.state_dict()
    )


def test_x4_gate_passes_complete_dual_seed_control_panel() -> None:
    gate = probe.adjudicate_linear_probe_panel(run_id="x4-pass", rows=_passing_rows())

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_skinny_rectangle_linear_probe_accessibility_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert all(all(checks.values()) for checks in gate["research_checks"].values())
    assert all(gate["stability_checks"].values())


@pytest.mark.parametrize(
    "role",
    [
        "corrupted_source_true_target",
        "true_source_corrupted_target",
        "random_source_true_target",
    ],
)
def test_x4_gate_holds_when_one_control_margin_fails(role: str) -> None:
    rows = _passing_rows()
    next(row for row in rows if row["seed"] == 1 and row["role"] == role)["auc"] = 0.639

    gate = probe.adjudicate_linear_probe_panel(run_id="x4-hold", rows=rows)

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation1_skinny_rectangle_linear_probe_accessibility_not_supported"
    )


def test_x4_gate_holds_when_candidate_seed_drift_exceeds_limit() -> None:
    rows = _passing_rows()
    next(
        row
        for row in rows
        if row["seed"] == 1 and row["role"] == "true_source_true_target"
    )["auc"] = 0.58
    for role, auc in (
        ("corrupted_source_true_target", 0.56),
        ("true_source_corrupted_target", 0.55),
        ("random_source_true_target", 0.54),
    ):
        next(row for row in rows if row["seed"] == 1 and row["role"] == role)["auc"] = (
            auc
        )

    gate = probe.adjudicate_linear_probe_panel(run_id="x4-drift", rows=rows)

    assert gate["status"] == "hold"
    assert gate["candidate_auc_drift"] == pytest.approx(0.08)
    assert gate["stability_checks"]["candidate_auc_drift_at_most_0p05"] is False


def test_x4_gate_fails_when_frozen_extractor_hash_changes() -> None:
    rows = deepcopy(_passing_rows())
    rows[0]["feature_extractor_final_sha256"] = "0" * 64

    gate = probe.adjudicate_linear_probe_panel(run_id="x4-invalid", rows=rows)

    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation1_skinny_rectangle_linear_probe_protocol_invalid"
    )
    assert (
        gate["protocol_checks"]["feature_extractors_and_source_classifiers_frozen"]
        is False
    )


def _raw_dataset_and_paths(
    tmp_path: Path,
) -> tuple[DifferentialDataset, dict[str, Path]]:
    features = np.zeros((4096, 512), dtype=np.uint8)
    labels = np.tile(np.array([0, 1], dtype=np.uint8), 2048)
    metadata = {
        "cipher": "RECTANGLE-80",
        "seed": 0,
        "samples_total": 4096,
    }
    paths = {
        "features": tmp_path / "features.npy",
        "labels": tmp_path / "labels.npy",
        "metadata": tmp_path / "metadata.json",
    }
    np.save(paths["features"], features)
    np.save(paths["labels"], labels)
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    return DifferentialDataset(features, labels, metadata), paths


def test_representation_cache_is_reused_and_parameter_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dataset, raw_paths = _raw_dataset_and_paths(tmp_path)
    extractor = transfer._build_target_extractor("true")

    def fake_extract(_extractor, features):
        base = features[:, :1]
        representation = base.repeat(1, probe.REPRESENTATION_WIDTH)
        return SimpleNamespace(representation=representation, logits=base)

    monkeypatch.setattr(probe, "extract_runtime_e4_representation", fake_extract)
    cache_dir = tmp_path / "representation-cache"
    first, first_metadata, first_reused = probe.extract_representation_cache(
        extractor=extractor,
        raw_dataset=raw_dataset,
        raw_paths=raw_paths,
        cache_dir=cache_dir,
        seed=0,
        split="train",
        role="true_source_true_target",
        source_role="true",
        target_mode="true",
        source_checkpoint_sha256=transfer.SOURCE_CHECKPOINT_SHA256S["true"],
    )
    second, second_metadata, second_reused = probe.extract_representation_cache(
        extractor=extractor,
        raw_dataset=raw_dataset,
        raw_paths=raw_paths,
        cache_dir=cache_dir,
        seed=0,
        split="train",
        role="true_source_true_target",
        source_role="true",
        target_mode="true",
        source_checkpoint_sha256=transfer.SOURCE_CHECKPOINT_SHA256S["true"],
    )

    assert first_reused is False
    assert second_reused is True
    assert first.features.shape == (4096, probe.REPRESENTATION_WIDTH)
    assert first.features.dtype == np.float32
    assert first_metadata == second_metadata
    assert first.cache_dir == second.cache_dir == cache_dir

    raw_paths["metadata"].write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="cache parameters changed"):
        probe.extract_representation_cache(
            extractor=extractor,
            raw_dataset=raw_dataset,
            raw_paths=raw_paths,
            cache_dir=cache_dir,
            seed=0,
            split="train",
            role="true_source_true_target",
            source_role="true",
            target_mode="true",
            source_checkpoint_sha256=transfer.SOURCE_CHECKPOINT_SHA256S["true"],
        )


def _target_metadata(*, seed: int, total: int, per_class: int) -> dict[str, object]:
    return {
        "cipher": "RECTANGLE-80",
        "rounds": 6,
        "input_difference": transfer.TARGET_INPUT_DIFFERENCE,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "independent_pairs",
        "input_bits": 512,
        "structure": "SPN",
        "seed": seed,
        "samples_total": total,
        "samples_per_class": per_class,
        "positive_rows": per_class,
        "negative_rows": per_class,
    }


def test_train_panel_completes_eight_roles_with_frozen_hashes_and_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    for role, mode in (("true", "true"), ("corrupted", "corrupted")):
        path = tmp_path / f"source-{role}.pt"
        torch.save(
            {
                "state_dict": transfer._build_target_extractor(mode).state_dict(),
                "metadata": {"selected_checkpoint": "best"},
            },
            path,
        )
        source_paths[role] = path
        source_hashes[role] = transfer.file_sha256(path)
    monkeypatch.setattr(transfer, "SOURCE_CHECKPOINT_SHA256S", source_hashes)

    source_rows = [
        {
            "model": model,
            "seed": 0,
            "samples_per_class": 1_000_000,
            "training": {"selected_checkpoint": "best"},
            "metrics": {"auc": transfer.SOURCE_AUCS[role]},
        }
        for role, model in transfer.SOURCE_MODELS.items()
    ]
    target_rows = [
        {
            "model": transfer.TARGET_MODELS["true"],
            "seed": seed,
            "samples_per_class": 2048,
            "rounds": 6,
            "metrics": {"auc": transfer.TARGET_ANCHOR_AUCS[seed]},
        }
        for seed in probe.TARGET_SEEDS
    ]
    target_datasets: dict[int, dict[str, DifferentialDataset]] = {}
    target_paths: dict[int, dict[str, dict[str, Path]]] = {}
    for seed in probe.TARGET_SEEDS:
        train = DifferentialDataset(
            np.zeros((4096, 512), dtype=np.uint8),
            np.zeros(4096, dtype=np.uint8),
            _target_metadata(seed=seed, total=4096, per_class=2048),
        )
        validation = DifferentialDataset(
            np.zeros((2048, 512), dtype=np.uint8),
            np.zeros(2048, dtype=np.uint8),
            _target_metadata(seed=seed + 10_000, total=2048, per_class=1024),
        )
        target_datasets[seed] = {"train": train, "validation": validation}
        target_paths[seed] = {}
        for split in ("train", "validation"):
            paths = {
                name: tmp_path / f"seed{seed}-{split}-{name}"
                for name in ("features", "labels", "metadata")
            }
            for path in paths.values():
                path.write_bytes(f"seed{seed}-{split}".encode())
            target_paths[seed][split] = paths

    cache_calls: set[Path] = set()

    def fake_cache(*, cache_dir, raw_paths, **kwargs):
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = cache_dir / "metadata.json"
        metadata_path.write_text("{}", encoding="utf-8")
        first = cache_dir not in cache_calls
        cache_calls.add(cache_dir)
        rows = 4096 if kwargs["split"] == "train" else 2048
        dataset = DiskDifferentialDataset(
            features=np.zeros((4, probe.REPRESENTATION_WIDTH), dtype=np.float32),
            labels=np.array([0, 1, 0, 1], dtype=np.uint8),
            metadata={},
            cache_dir=cache_dir,
        )
        character = "a" if kwargs["seed"] == 0 else "b"
        validation_character = "c" if kwargs["seed"] == 0 else "d"
        raw_character = (
            character if kwargs["split"] == "train" else validation_character
        )
        metadata = {
            "representation_sha256": "9" * 64,
            "raw_feature_sha256": raw_character * 64,
            "raw_label_sha256": "1" * 64,
            "raw_metadata_sha256": raw_character * 64,
            "rows": rows,
        }
        return dataset, metadata, not first

    aucs = {
        (seed, role): auc
        for seed, seed_aucs in (
            (
                0,
                {
                    "true_source_true_target": 0.66,
                    "corrupted_source_true_target": 0.61,
                    "true_source_corrupted_target": 0.59,
                    "random_source_true_target": 0.57,
                },
            ),
            (
                1,
                {
                    "true_source_true_target": 0.64,
                    "corrupted_source_true_target": 0.60,
                    "true_source_corrupted_target": 0.58,
                    "random_source_true_target": 0.56,
                },
            ),
        )
        for role, auc in seed_aucs.items()
    }

    def fake_train(model, _train, _validation, config, progress_callback=None):
        filename = Path(config.checkpoint_output).stem
        seed = int(filename.split("_", 1)[0].removeprefix("seed"))
        role = filename.split("_", 1)[1]
        auc = aucs[(seed, role)]
        with torch.no_grad():
            model.weight.add_(0.001)
        metrics = {"auc": auc, "accuracy": 0.55, "loss": 0.69}
        history = [
            {
                "epoch": float(epoch),
                "train_loss": 0.25,
                "train_auc": auc - 0.01,
                "val_loss": 0.69,
                "val_auc": auc,
                "val_accuracy": 0.55,
                "learning_rate": config.learning_rate,
            }
            for epoch in range(1, config.epochs + 1)
        ]
        metadata = {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "selected_checkpoint": "best",
        }
        torch.save(
            {
                "state_dict": model.state_dict(),
                "final_metrics": metrics,
                "metadata": metadata,
            },
            config.checkpoint_output,
        )
        return SimpleNamespace(
            final_metrics=metrics,
            history=history,
            metadata=metadata,
        )

    monkeypatch.setattr(probe, "extract_representation_cache", fake_cache)
    monkeypatch.setattr(probe, "train_binary_classifier", fake_train)
    rows = probe.train_linear_probe_panel(
        source_rows=source_rows,
        source_checkpoint_paths=source_paths,
        target_rows=target_rows,
        target_datasets=target_datasets,
        target_paths=target_paths,
        representation_cache_root=tmp_path / "representation-cache",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert len(rows) == 8
    assert all(row["checkpoint_replay_verified"] is True for row in rows)
    assert all(
        row["feature_extractor_initial_sha256"] == row["feature_extractor_final_sha256"]
        for row in rows
    )
    assert all(
        row["source_classifier_initial_sha256"] == row["source_classifier_final_sha256"]
        for row in rows
    )
    assert all(row["trainable_parameter_count"] == 385 for row in rows)


def test_plot_has_plain_chinese_title_and_explanations(tmp_path: Path) -> None:
    rows = _passing_rows()
    gate = probe.adjudicate_linear_probe_panel(run_id="x4-plot", rows=rows)
    output = tmp_path / "curves.svg"

    cli.render_linear_probe_svg(gate, rows, output)

    svg = output.read_text(encoding="utf-8")
    assert "冻结 SPN 结构表示能否被线性层直接读出" in svg
    assert "错误源=使用错误拓扑训练的SKINNY检查点" in svg
    assert "只训练 Linear(384,1) 共385个参数" in svg


def test_cli_refuses_to_overwrite_existing_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(cli, "validate_authorities", lambda *_: {"status": "pass"})

    with pytest.raises(ValueError, match="output root already exists"):
        cli.main(
            [
                "--source-root",
                str(tmp_path / "source"),
                "--target-root",
                str(tmp_path / "target"),
                "--output-root",
                str(output),
            ]
        )
