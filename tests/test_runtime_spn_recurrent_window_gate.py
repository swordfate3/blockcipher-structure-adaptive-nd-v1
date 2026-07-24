from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.gate_runtime_spn_recurrent_window import main
from blockcipher_nd.tasks.innovation1.runtime_spn_recurrent_window import (
    adjudicate_runtime_spn_recurrent_window,
)


ROLES = (
    "anchor",
    "candidate",
    "repeat_last",
    "corrupted",
    "no_topology",
)


def _row(seed: int, role: str, auc: float) -> dict[str, object]:
    model = {
        "anchor": "runtime_spn_e4_equivariant_true",
        "candidate": "runtime_spn_e4_equivariant_true",
        "repeat_last": "runtime_spn_e4_equivariant_true",
        "corrupted": "runtime_spn_e4_equivariant_corrupted",
        "no_topology": "runtime_spn_e4_equivariant_independent",
    }[role]
    options: dict[str, object] = {
        "runtime_structure_path": "configs/runtime/spn/uknit64.json",
        "runtime_round_start": 3,
        "runtime_rounds": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "edge_gate",
        "cell_input_mode": "state_triplet",
        "round_window_mode": (
            "last_transition" if role == "anchor" else "recurrent_window"
        ),
        "runtime_structure_window_control": (
            "repeat_last" if role == "repeat_last" else "full"
        ),
    }
    if role == "corrupted":
        options["topology_corruption_seed"] = 20260724
    transition_hashes = {
        "anchor": ("a" * 64, "b" * 64),
        "candidate": ("a" * 64, "b" * 64),
        "repeat_last": ("b" * 64, "b" * 64),
        "corrupted": ("c" * 64, "d" * 64),
        "no_topology": ("a" * 64, "b" * 64),
    }[role]
    window_hash = {
        "anchor": "1" * 64,
        "candidate": "1" * 64,
        "repeat_last": "2" * 64,
        "corrupted": "3" * 64,
        "no_topology": "1" * 64,
    }[role]
    history = [
        {"epoch": float(epoch), "val_auc": auc - (10 - epoch) * 0.001}
        for epoch in range(1, 11)
    ]
    return {
        "cipher": "uKNIT-BC",
        "cipher_key": "uknit64",
        "structure": "SPN",
        "model": model,
        "rounds": 5,
        "seed": seed,
        "samples_per_class": 2048,
        "dataset_label_mode": "balanced_per_class",
        "pairs_per_sample": 4,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "key_rotation_interval": 0,
        "sample_structure": "independent_pairs",
        "input_difference": 0x40,
        "train_key": 0,
        "validation_key": int("11" * 16, 16),
        "target_epochs": 10,
        "parameter_count": 442466,
        "trainable_parameter_count": 442466,
        "input_bit_order": "project_msb_to_runtime_lsb",
        "runtime_structure_round_start": 3,
        "runtime_structure_loaded_rounds": 2,
        "runtime_structure_available_rounds": 11,
        "runtime_structure_descriptor_name": "uKNIT-BC 64-bit transition structure",
        "runtime_structure_descriptor_sha256": "e" * 64,
        "runtime_structure_transition_sha256s": transition_hashes,
        "runtime_structure_window_sha256": window_hash,
        "runtime_structure_unique_transition_count": len(set(transition_hashes)),
        "runtime_structure_homogeneous": len(set(transition_hashes)) == 1,
        "metrics": {"auc": auc},
        "history": history,
        "validation": {"samples_per_class": 1024},
        "training": {
            "epochs": 10,
            "epochs_ran": 10,
            "best_epoch": 10,
            "stopped_epoch": 0,
            "best_checkpoint_metric": auc,
            "batch_size": 256,
            "learning_rate": 0.0001,
            "optimizer": "adam",
            "optimizer_state_transition": "reset_each_stage",
            "weight_decay": 0.00001,
            "loss": "mse",
            "lr_scheduler": "none",
            "checkpoint_metric": "val_auc",
            "restore_best_checkpoint": True,
            "selected_checkpoint": "best",
            "train_rows": 4096,
            "validation_rows": 2048,
            "train_dataset_storage": "disk",
            "validation_dataset_storage": "disk",
            "dataset_cache_root": f"outputs/local_cache/recurrent_seed{seed}",
            "dataset_cache_chunk_size": 1024,
            "dataset_cache_workers": 1,
            "model_options": options,
        },
    }


def _passing_rows() -> list[dict[str, object]]:
    aucs = {
        "anchor": 0.535,
        "candidate": 0.555,
        "repeat_last": 0.545,
        "corrupted": 0.540,
        "no_topology": 0.510,
    }
    return [_row(seed, role, aucs[role]) for seed in (0, 1) for role in ROLES]


def test_recurrent_window_gate_passes_only_both_seed_four_control_margins() -> None:
    gate = adjudicate_runtime_spn_recurrent_window(
        run_id="u3",
        rows=_passing_rows(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_runtime_spn_recurrent_window_two_seed_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert gate["thresholds"] == {
        "candidate_auc": 0.52,
        "candidate_minus_each_control": 0.005,
    }


def test_recurrent_window_gate_holds_when_repeat_last_matches_candidate() -> None:
    rows = _passing_rows()
    repeat_last = next(
        row
        for row in rows
        if row["seed"] == 1
        and row["training"]["model_options"][  # type: ignore[index]
            "runtime_structure_window_control"
        ]
        == "repeat_last"
    )
    repeat_last["metrics"]["auc"] = 0.553  # type: ignore[index]
    repeat_last["history"][-1]["val_auc"] = 0.553  # type: ignore[index]
    repeat_last["training"]["best_checkpoint_metric"] = 0.553  # type: ignore[index]

    gate = adjudicate_runtime_spn_recurrent_window(run_id="hold", rows=rows)

    assert gate["status"] == "hold"
    assert gate["protocol_checks"]["ten_epoch_best_checkpoint_integrity"] is True
    assert (
        gate["research_checks"]["seed1_candidate_beats_repeat_last_by_0p005"] is False
    )


def test_recurrent_window_gate_fails_closed_on_role_protocol_drift() -> None:
    rows = deepcopy(_passing_rows())
    rows[3]["training"]["model_options"][  # type: ignore[index]
        "topology_corruption_seed"
    ] = 99

    gate = adjudicate_runtime_spn_recurrent_window(run_id="invalid", rows=rows)

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["role_contracts_match_frozen_plan"] is False


def test_recurrent_window_gate_fails_closed_on_runtime_fingerprint_drift() -> None:
    rows = deepcopy(_passing_rows())
    candidate = next(
        row
        for row in rows
        if row["seed"] == 1
        and row["training"]["model_options"]["round_window_mode"]  # type: ignore[index]
        == "recurrent_window"
        and row["model"] == "runtime_spn_e4_equivariant_true"
        and row["training"]["model_options"][  # type: ignore[index]
            "runtime_structure_window_control"
        ]
        == "full"
    )
    candidate["runtime_structure_window_sha256"] = "f" * 64

    gate = adjudicate_runtime_spn_recurrent_window(
        run_id="fingerprint-drift",
        rows=rows,
    )

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["anchor_candidate_structure_equal"] is False
    assert gate["protocol_checks"]["structure_evidence_seed_invariant"] is False


def test_recurrent_window_cli_writes_gate_validation_and_summary(
    tmp_path: Path,
) -> None:
    rows = _passing_rows()
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    status = main(["--run-id", "cli-u3", "--run-root", str(tmp_path)])

    assert status == 0
    gate = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    validation = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert validation["status"] == "pass"
    assert summary["training_performed"] is True
