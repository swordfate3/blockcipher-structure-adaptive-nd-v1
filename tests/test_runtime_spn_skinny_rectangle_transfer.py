from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from blockcipher_nd.cli import run_runtime_spn_skinny_rectangle_transfer as cli
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_skinny_rectangle_transfer as transfer,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_head_adaptation import (
    tensor_mapping_sha256,
)


def _gate_row(role: str, auc: float) -> dict[str, object]:
    source_role, target_mode = transfer.ROLE_SPECS[role]
    source_hash = {
        "true": transfer.SOURCE_CHECKPOINT_SHA256S["true"],
        "corrupted": transfer.SOURCE_CHECKPOINT_SHA256S["corrupted"],
        "random": None,
    }[source_role]
    source_auc = {
        "true": transfer.SOURCE_AUCS["true"],
        "corrupted": transfer.SOURCE_AUCS["corrupted"],
        "random": None,
    }[source_role]
    return {
        "seed": 0,
        "role": role,
        "source_role": source_role,
        "target_mode": target_mode,
        "source_checkpoint_sha256": source_hash,
        "source_auc": source_auc,
        "source_samples_per_class": 1_000_000,
        "target_structure_sha256": ("1" * 64 if target_mode == "true" else "2" * 64),
        "target_head_initial_sha256": "3" * 64,
        "target_head_final_sha256": "4" * 64,
        "feature_extractor_initial_sha256": "5" * 64,
        "feature_extractor_final_sha256": "5" * 64,
        "source_classifier_initial_sha256": "6" * 64,
        "source_classifier_final_sha256": "6" * 64,
        "checkpoint_sha256": "7" * 64,
        "checkpoint_replay_verified": True,
        "parameter_count": transfer.TOTAL_PARAMETER_COUNT,
        "trainable_parameter_count": transfer.TARGET_HEAD_PARAMETER_COUNT,
        "trainable_parameter_names": [
            "target_head.0.weight",
            "target_head.0.bias",
        ],
        "adapter_mode": "frozen_runtime_e4_target_head",
        "feature_extractor_frozen": True,
        "source_classifier_preserved": True,
        "auc": auc,
        "full_target_anchor_auc": transfer.TARGET_ANCHOR_AUC,
        "train_feature_sha256": "8" * 64,
        "train_label_sha256": "9" * 64,
        "train_metadata_sha256": "a" * 64,
        "validation_feature_sha256": "b" * 64,
        "validation_label_sha256": "c" * 64,
        "validation_metadata_sha256": "d" * 64,
        "target_cipher": "RECTANGLE-80",
        "target_rounds": 6,
        "train_rows": 4096,
        "validation_rows": 2048,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "training": {"epochs": 5, "selected_checkpoint": "best"},
        "history": [
            {
                "epoch": float(epoch),
                "train_loss": 0.25,
                "train_auc": auc - 0.02,
                "val_loss": 0.69,
                "val_auc": auc - 0.001 * (5 - epoch),
                "val_accuracy": 0.5,
                "learning_rate": 1e-4,
            }
            for epoch in range(1, 6)
        ],
    }


def _passing_rows() -> list[dict[str, object]]:
    aucs = {
        "true_source_true_target": 0.62,
        "corrupted_source_true_target": 0.59,
        "true_source_corrupted_target": 0.58,
        "random_source_true_target": 0.54,
    }
    return [_gate_row(role, aucs[role]) for role in transfer.EXPECTED_ROLES]


def _source_states() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "true": transfer._build_target_extractor("true").state_dict(),
        "corrupted": transfer._build_target_extractor("corrupted").state_dict(),
    }


def test_source_states_load_strictly_and_only_target_head_is_trainable() -> None:
    source_states = _source_states()

    for source_role, target_mode in (
        ("true", "true"),
        ("corrupted", "true"),
        ("true", "corrupted"),
        ("random", "true"),
    ):
        model = transfer.prepare_transfer_model(
            source_role=source_role,
            target_mode=target_mode,
            source_state_dicts=source_states,
        )
        trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        assert trainable
        assert all(name.startswith("target_head.") for name in trainable)
        assert model.feature_extractor.training is False
        assert model.feature_extractor_frozen is True
        assert model.source_classifier_preserved is True
        assert (
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            == transfer.TARGET_HEAD_PARAMETER_COUNT
        )


def test_target_head_initialization_is_identical_across_all_roles() -> None:
    source_states = _source_states()
    hashes = {
        tensor_mapping_sha256(
            transfer.prepare_transfer_model(
                source_role=source_role,
                target_mode=target_mode,
                source_state_dicts=source_states,
            ).target_head.state_dict()
        )
        for source_role, target_mode in transfer.ROLE_SPECS.values()
    }

    assert hashes == {
        tensor_mapping_sha256(transfer.deterministic_target_head().state_dict())
    }


def test_x3a_gate_passes_complete_control_panel() -> None:
    gate = transfer.adjudicate_transfer_panel(run_id="x3a-pass", rows=_passing_rows())

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_skinny_rectangle_frozen_representation_readiness_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


@pytest.mark.parametrize(
    ("role", "check"),
    [
        (
            "corrupted_source_true_target",
            "candidate_beats_corrupted_source_by_0p005",
        ),
        (
            "true_source_corrupted_target",
            "candidate_beats_corrupted_target_by_0p005",
        ),
        ("random_source_true_target", "candidate_beats_random_source_by_0p005"),
    ],
)
def test_x3a_gate_holds_when_any_control_margin_is_missing(
    role: str,
    check: str,
) -> None:
    rows = _passing_rows()
    next(row for row in rows if row["role"] == role)["auc"] = 0.618

    gate = transfer.adjudicate_transfer_panel(run_id="x3a-hold", rows=rows)

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation1_skinny_rectangle_frozen_representation_readiness_not_supported"
    )
    assert gate["research_checks"][check] is False


def test_x3a_gate_fails_if_frozen_extractor_hash_changes() -> None:
    rows = deepcopy(_passing_rows())
    rows[0]["feature_extractor_final_sha256"] = "0" * 64

    gate = transfer.adjudicate_transfer_panel(run_id="x3a-fail", rows=rows)

    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation1_skinny_rectangle_transfer_protocol_invalid"
    )
    assert gate["protocol_checks"]["feature_extractors_frozen"] is False


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _authority_roots(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_json(
        source_root / "gate.local.json",
        {
            "status": "pass",
            "decision": "innovation1_rtg3a_skinny_formal_seed0_supported",
            "protocol_checks": {"complete": True},
            "research_checks": {"supported": True},
        },
    )
    _write_json(
        source_root / "checkpoint-verification.local.json",
        {
            "status": "pass",
            "file_set_exact": True,
            "entries": [
                {
                    "model": "skinny64_runtime_e4_equivariant_true",
                    "sha256": transfer.SOURCE_CHECKPOINT_SHA256S["true"],
                },
                {
                    "model": "skinny64_runtime_e4_equivariant_corrupted",
                    "sha256": transfer.SOURCE_CHECKPOINT_SHA256S["corrupted"],
                },
            ],
        },
    )
    (source_root / "visual_qa_passed.marker").touch()
    _write_json(
        target_root / "gate.json",
        {
            "status": "pass",
            "decision": (
                "innovation1_runtime_spn_rectangle_noncontiguous_attribution_supported"
            ),
            "protocol_checks": {"complete": True},
            "research_checks": {"seed0": {"supported": True}},
        },
    )
    _write_json(target_root / "validation.json", {"status": "pass"})
    (target_root / "visual_qa_passed.marker").touch()
    return source_root, target_root


def test_authorities_pass_only_for_complete_source_and_target_evidence(
    tmp_path: Path,
) -> None:
    source_root, target_root = _authority_roots(tmp_path)

    authority = cli.validate_authorities(source_root, target_root)

    assert authority["source_authorized"] is True
    assert authority["target_authorized"] is True


def test_source_authority_fails_closed(tmp_path: Path) -> None:
    source_root, target_root = _authority_roots(tmp_path)
    source_gate = json.loads((source_root / "gate.local.json").read_text())
    source_gate["decision"] = "wrong"
    _write_json(source_root / "gate.local.json", source_gate)

    with pytest.raises(ValueError, match="formal SKINNY seed0 authority"):
        cli.validate_authorities(source_root, target_root)


def test_target_authority_fails_closed(tmp_path: Path) -> None:
    source_root, target_root = _authority_roots(tmp_path)
    _write_json(target_root / "validation.json", {"status": "fail"})

    with pytest.raises(ValueError, match="RECTANGLE RCT1 authority"):
        cli.validate_authorities(source_root, target_root)


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


def _write_cache(
    root: Path,
    *,
    split: str,
    seed: int,
    total: int,
    per_class: int,
) -> None:
    cache = root / "cache/rectangle80/r6" / split / f"seed-{seed}_fixture"
    cache.mkdir(parents=True)
    np.save(cache / "features.npy", np.zeros((total, 512), dtype=np.uint8))
    np.save(cache / "labels.npy", np.zeros(total, dtype=np.uint8))
    _write_json(
        cache / "metadata.json",
        _target_metadata(seed=seed, total=total, per_class=per_class),
    )


def test_cache_paths_and_metadata_are_validated_exactly(tmp_path: Path) -> None:
    assert transfer.TARGET_INPUT_DIFFERENCE == 0x2100010020
    _write_cache(
        tmp_path,
        split="train",
        seed=0,
        total=4096,
        per_class=2048,
    )
    _write_cache(
        tmp_path,
        split="validation",
        seed=10000,
        total=2048,
        per_class=1024,
    )
    train, train_paths = cli._load_target_split(
        tmp_path,
        split="train",
        expected_seed=0,
    )
    validation, validation_paths = cli._load_target_split(
        tmp_path,
        split="validation",
        expected_seed=10000,
    )

    transfer._validate_target_datasets(train, validation)
    transfer._validate_dataset_paths(train_paths, validation_paths)

    invalid = DifferentialDataset(
        features=validation.features,
        labels=validation.labels,
        metadata={**validation.metadata, "seed": 999},
    )
    with pytest.raises(ValueError, match="split geometry changed"):
        transfer._validate_target_datasets(train, invalid)


def test_train_panel_preserves_frozen_hashes_and_head_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths: dict[str, Path] = {}
    source_sha256s: dict[str, str] = {}
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
        source_sha256s[role] = transfer.file_sha256(path)
    monkeypatch.setattr(transfer, "SOURCE_CHECKPOINT_SHA256S", source_sha256s)

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
            "seed": 0,
            "samples_per_class": 2048,
            "rounds": 6,
            "metrics": {"auc": transfer.TARGET_ANCHOR_AUC},
        }
    ]
    train = DifferentialDataset(
        features=np.zeros((4096, 512), dtype=np.uint8),
        labels=np.zeros(4096, dtype=np.uint8),
        metadata=_target_metadata(seed=0, total=4096, per_class=2048),
    )
    validation = DifferentialDataset(
        features=np.zeros((2048, 512), dtype=np.uint8),
        labels=np.zeros(2048, dtype=np.uint8),
        metadata=_target_metadata(seed=10000, total=2048, per_class=1024),
    )
    train_paths = {
        name: tmp_path / f"train-{name}" for name in ("features", "labels", "metadata")
    }
    validation_paths = {
        name: tmp_path / f"validation-{name}"
        for name in ("features", "labels", "metadata")
    }
    for path in (*train_paths.values(), *validation_paths.values()):
        path.write_bytes(b"frozen X3-A fixture")

    aucs = iter((0.62, 0.59, 0.58, 0.54))

    def fake_train(model, _train, _validation, config, progress_callback=None):
        auc = next(aucs)
        with torch.no_grad():
            next(model.target_head.parameters()).add_(0.001)
        metrics = {"auc": auc, "accuracy": 0.55, "loss": 0.69}
        history = [
            {
                "epoch": float(epoch),
                "train_loss": 0.25,
                "train_auc": auc - 0.02,
                "val_loss": 0.69,
                "val_auc": auc,
                "val_accuracy": 0.55,
                "learning_rate": config.learning_rate,
            }
            for epoch in range(1, config.epochs + 1)
        ]
        metadata = {
            "epochs": config.epochs,
            "selected_checkpoint": "best",
            "checkpoint_output": str(config.checkpoint_output),
        }
        torch.save(
            {
                "state_dict": model.state_dict(),
                "final_metrics": metrics,
                "metadata": metadata,
            },
            config.checkpoint_output,
        )
        if progress_callback is not None:
            progress_callback("epoch_end", {"epoch": config.epochs})
        return SimpleNamespace(
            final_metrics=metrics,
            history=history,
            metadata=metadata,
        )

    monkeypatch.setattr(transfer, "train_binary_classifier", fake_train)
    rows = transfer.train_transfer_panel(
        source_rows=source_rows,
        source_checkpoint_paths=source_paths,
        target_rows=target_rows,
        train_dataset=train,
        validation_dataset=validation,
        train_paths=train_paths,
        validation_paths=validation_paths,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert len(rows) == 4
    assert len({row["target_head_initial_sha256"] for row in rows}) == 1
    assert all(
        row["feature_extractor_initial_sha256"] == row["feature_extractor_final_sha256"]
        for row in rows
    )
    assert all(
        row["source_classifier_initial_sha256"] == row["source_classifier_final_sha256"]
        for row in rows
    )
    assert all(row["checkpoint_replay_verified"] is True for row in rows)


def test_plot_has_plain_chinese_title_and_role_explanations(tmp_path: Path) -> None:
    rows = _passing_rows()
    gate = transfer.adjudicate_transfer_panel(run_id="x3a-plot", rows=rows)
    output = tmp_path / "curves.svg"

    cli.render_transfer_svg(gate, rows, output)

    svg = output.read_text(encoding="utf-8")
    assert "正式 SKINNY 结构表示迁移到 RECTANGLE" in svg
    assert "正确源=SKINNY正式规模正确拓扑检查点" in svg
    assert "错误目标=RECTANGLE错误P层" in svg


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
