from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from blockcipher_nd.cli.gate_runtime_spn_dialga_d6 import audit_source_cache_reuse
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d1 import (
    MODELS as D1_MODELS,
    adjudicate_runtime_spn_dialga_d1,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d6 import (
    D6_MODELS,
    EXPECTED_PARAMETER_COUNT,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d7 import (
    D1_RUN_ID,
    adjudicate_runtime_spn_dialga_d7,
)


PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_dialga128_runtime_e5_d7_r4_2048_seed0_seed1.csv"
)
COMMON_OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/dialga128.json",
    "runtime_round_start": 2,
    "runtime_rounds": 2,
    "processor_steps": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "sbox_context_mode": "edge_gate",
    "cell_input_mode": "state_triplet",
    "round_window_mode": "recurrent_window",
    "runtime_structure_window_control": "full",
}


def _rows(
    *,
    cache_root: Path,
    aucs: dict[int, dict[str, float]],
    e5: bool,
) -> list[dict[str, object]]:
    models = D6_MODELS if e5 else D1_MODELS
    parameter_count = EXPECTED_PARAMETER_COUNT if e5 else 442_466
    window_hashes = {
        "correct": "a" * 64,
        "corrupted": "b" * 64,
        "no_topology": "a" * 64,
    }
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for role, model in models.items():
            options = COMMON_OPTIONS.copy()
            if role == "corrupted":
                options["topology_corruption_seed"] = 20260725
            auc = aucs[seed][role]
            row: dict[str, object] = {
                "cipher": "Dialga-128",
                "cipher_key": "dialga128",
                "model": model,
                "rounds": 4,
                "seed": seed,
                "samples_per_class": 2048,
                "dataset_label_mode": "balanced_per_class",
                "pairs_per_sample": 4,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "difference_profile": "",
                "difference_member": "",
                "input_difference": 0x40,
                "train_key": 0,
                "validation_key": int("11" * 32, 16),
                "parameter_count": parameter_count,
                "trainable_parameter_count": parameter_count,
                "runtime_structure_descriptor_name": (
                    "Dialga-128 20-round heterogeneous runtime SPN structure"
                ),
                "runtime_structure_descriptor_path": (
                    "/repo/configs/runtime/spn/dialga128.json"
                ),
                "runtime_structure_descriptor_sha256": "c" * 64,
                "runtime_structure_round_start": 2,
                "runtime_structure_available_rounds": 20,
                "runtime_structure_loaded_rounds": 2,
                "runtime_structure_unique_transition_count": 2,
                "runtime_structure_homogeneous": False,
                "runtime_structure_mode": {
                    "correct": "true",
                    "corrupted": "corrupted",
                    "no_topology": "independent",
                }[role],
                "runtime_structure_window_control": "full",
                "runtime_structure_transition_sha256s": ["d" * 64, "e" * 64],
                "runtime_structure_window_sha256": window_hashes[role],
                "metrics": {"auc": auc},
                "history": [
                    {"epoch": epoch + 1, "val_auc": auc - 0.009 + epoch * 0.001}
                    for epoch in range(10)
                ],
                "training": {
                    "epochs": 10,
                    "loss": "mse",
                    "optimizer": "adam",
                    "learning_rate": 0.0001,
                    "weight_decay": 0.00001,
                    "checkpoint_metric": "val_auc",
                    "restore_best_checkpoint": True,
                    "selected_checkpoint": "best",
                    "train_rows": 4096,
                    "validation_rows": 2048,
                    "input_bits": 1024,
                    "pair_bits": 256,
                    "model_options": options,
                    "train_dataset_storage": "disk",
                    "validation_dataset_storage": "disk",
                    "dataset_cache_root": str(cache_root),
                },
            }
            if e5:
                raw_gate = 0.0 if role == "no_topology" else 0.08 + 0.01 * seed
                row.update(
                    {
                        "topology_residual_mode": (
                            "independent_base_plus_bounded_topology_logit_residual"
                        ),
                        "topology_gate_initial": 0.0,
                        "topology_gate_final_raw": raw_gate,
                        "topology_gate_final_bounded": math.tanh(raw_gate),
                    }
                )
            rows.append(row)
    return rows


def _source_d1(cache_root: Path):
    rows = _rows(
        cache_root=cache_root,
        aucs={
            0: {"correct": 0.960, "corrupted": 0.850, "no_topology": 0.800},
            1: {"correct": 0.950, "corrupted": 0.840, "no_topology": 0.790},
        },
        e5=False,
    )
    gate = adjudicate_runtime_spn_dialga_d1(run_id=D1_RUN_ID, rows=rows)
    validation = {
        "run_id": D1_RUN_ID,
        "status": "pass",
        "checks": gate["protocol_checks"],
    }
    return rows, gate, validation


def _candidate_rows(cache_root: Path) -> list[dict[str, object]]:
    return _rows(
        cache_root=cache_root,
        aucs={
            0: {"correct": 0.955, "corrupted": 0.900, "no_topology": 0.880},
            1: {"correct": 0.945, "corrupted": 0.890, "no_topology": 0.870},
        },
        e5=True,
    )


def _adjudicate(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    cache_status: str = "pass",
    replay_drift: bool = False,
):
    cache_root = tmp_path / "cache"
    d1_rows, d1_gate, d1_validation = _source_d1(cache_root)
    replayed = deepcopy(d1_gate)
    if replay_drift:
        replayed["decision"] = "drifted"
    return adjudicate_runtime_spn_dialga_d7(
        run_id="d7-test",
        rows=rows if rows is not None else _candidate_rows(cache_root),
        d1_rows=d1_rows,
        persisted_d1_gate=d1_gate,
        replayed_d1_gate=replayed,
        d1_validation=d1_validation,
        expected_cache_root=cache_root,
        cache_audit={"status": cache_status, "checks": {"reuse": True}},
    )


def test_real_d7_plan_parses_as_six_frozen_rows() -> None:
    tasks = tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )

    assert len(tasks) == 6
    assert {(task["seed"], task["model_key"]) for task in tasks} == {
        (seed, model) for seed in (0, 1) for model in D6_MODELS.values()
    }
    assert all(
        task["cipher_key"] == "dialga128"
        and task["rounds"] == 4
        and task["samples_per_class"] == 2048
        and task["pairs_per_sample"] == 4
        and task["target_epochs"] == 10
        and task["input_difference"] == 0x40
        and task["negative_mode"] == "encrypted_random_plaintexts"
        and task["model_options"]["runtime_round_start"] == 2
        for task in tasks
    )


def test_d7_cache_audit_names_the_d1_source(tmp_path: Path) -> None:
    audit = audit_source_cache_reuse(
        progress_rows=[],
        expected_cache_root=tmp_path / "cache",
        source_label="d1",
    )

    assert "four_d1_cache_directories_present" in audit["checks"]
    assert "reuse_paths_are_exact_d1_cache_leaves" in audit["checks"]
    assert "reuse_paths_stay_under_d1_cache_root" in audit["checks"]
    assert all("d3" not in key for key in audit["checks"])


def test_d7_passes_only_when_e5_retains_both_d1_seeds_and_controls(
    tmp_path: Path,
) -> None:
    gate = _adjudicate(tmp_path)

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_dialga_runtime_e5_d7_r4_regression_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_d7_holds_when_one_seed_misses_retention_or_topology_control(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    retention_rows = _candidate_rows(cache_root)
    retention_rows[3]["metrics"] = {"auc": 0.930}
    retention_rows[3]["history"] = [
        {"epoch": epoch + 1, "val_auc": 0.921 + epoch * 0.001}
        for epoch in range(10)
    ]
    retention_gate = _adjudicate(tmp_path, rows=retention_rows)
    assert retention_gate["status"] == "hold"
    assert retention_gate["research_checks"]["seed1_retains_d1_within_0p010"] is False

    control_rows = _candidate_rows(cache_root)
    control_rows[1]["metrics"] = {"auc": 0.952}
    control_rows[1]["history"] = [
        {"epoch": epoch + 1, "val_auc": 0.943 + epoch * 0.001}
        for epoch in range(10)
    ]
    control_gate = _adjudicate(tmp_path, rows=control_rows)
    assert control_gate["status"] == "hold"
    assert (
        control_gate["research_checks"][
            "seed0_correct_exceeds_corrupted_by_0p005"
        ]
        is False
    )


def test_d7_fails_closed_on_source_replay_cache_or_zero_gate_drift(
    tmp_path: Path,
) -> None:
    replay_invalid = _adjudicate(tmp_path, replay_drift=True)
    assert replay_invalid["status"] == "fail"
    assert (
        replay_invalid["protocol_checks"]["d1_source_gate_recomputed_exactly"]
        is False
    )

    cache_invalid = _adjudicate(tmp_path, cache_status="fail")
    assert cache_invalid["status"] == "fail"
    assert (
        cache_invalid["protocol_checks"]["d1_cache_reused_without_generation"]
        is False
    )

    cache_root = tmp_path / "cache"
    gate_invalid_rows = _candidate_rows(cache_root)
    gate_invalid_rows[2]["topology_gate_final_raw"] = 0.01
    gate_invalid_rows[2]["topology_gate_final_bounded"] = math.tanh(0.01)
    gate_invalid = _adjudicate(tmp_path, rows=gate_invalid_rows)
    assert gate_invalid["status"] == "fail"
    assert (
        gate_invalid["protocol_checks"]["gated_residual_metadata_complete"]
        is False
    )
