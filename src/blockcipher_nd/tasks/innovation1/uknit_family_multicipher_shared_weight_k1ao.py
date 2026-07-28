from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_multicipher_shared_weight_k1ao_readiness_20260729"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_multicipher_shared_weight_k1ao_"
    "readiness_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "7dbe1d1aea5d35d7c0d1ff593d5b949a917591dbb4c9f415ac10194ed6dc49cf"
)
MODEL_KEY = "runtime_spn_ct_k1ak_sbox_transition_true"
WRONG_MODEL_KEY = "runtime_spn_ct_k1ak_sbox_transition_wrong_sbox"
EXPECTED_CIPHERS = ("uknit64", "midori64", "dialga128")
EXPECTED_SPLITS = ("train_seen", "same_key_fresh", "cross_key_validation")
EXPECTED_PARAMETER_COUNT = 219_320
EXPECTED_STATE_ENTRIES = 52
EXPECTED_DATASET_ROWS = 18
EXPECTED_RUNTIME_ROWS = 9


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("K1-AO config must be a JSON object")
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-AO config digest drifted")
    if config.get("schema_version") != 1:
        raise ValueError("K1-AO schema_version must remain 1")
    if config.get("run_id") != RUN_ID:
        raise ValueError("K1-AO run_id drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_multicipher_shared_weight_k1ao_readiness"
    ):
        raise ValueError("K1-AO experiment name drifted")
    if config.get("audit") != {
        "training_rows": 0,
        "validation_rows": 0,
        "optimizer_steps": 0,
        "data_generation": False,
        "remote": False,
        "pairs_per_sample": 4,
    }:
        raise ValueError("K1-AO zero-training contract drifted")
    model = config.get("model", {})
    if (
        model.get("model_key") != MODEL_KEY
        or int(model.get("expected_trainable_parameters", -1))
        != EXPECTED_PARAMETER_COUNT
        or int(model.get("expected_state_dict_entries", -1))
        != EXPECTED_STATE_ENTRIES
        or int(model.get("initialization_seed", -1)) != 29
    ):
        raise ValueError("K1-AO model contract drifted")
    ciphers = config.get("ciphers", [])
    if [row.get("cipher_key") for row in ciphers] != list(EXPECTED_CIPHERS):
        raise ValueError("K1-AO cipher order or membership drifted")
    expected_geometry = {
        "uknit64": (5, 64, 16, 512, 128, (3, 4), 3),
        "midori64": (4, 64, 16, 512, 128, (6, 7), 0),
        "dialga128": (4, 128, 32, 1024, 256, (0, 1), 2),
    }
    for row in ciphers:
        cipher_key = str(row["cipher_key"])
        actual = (
            int(row["rounds"]),
            int(row["block_bits"]),
            int(row["cells"]),
            int(row["input_bits"]),
            int(row["pair_bits"]),
            tuple(int(seed) for seed in row["seeds"]),
            int(row["runtime_round_start"]),
        )
        if actual != expected_geometry[cipher_key]:
            raise ValueError(f"K1-AO {cipher_key} geometry drifted")
        if int(row.get("runtime_rounds", -1)) != 2:
            raise ValueError("K1-AO requires exactly two runtime transitions")
        if set(row.get("source_digests", {})) != {
            "gate.json",
            "validation.json",
            "dataset_manifest.jsonl",
        }:
            raise ValueError("K1-AO source digest contract drifted")
    controls = config.get("controls", {})
    if controls != {
        "wrong_sbox_seed": 20260728,
        "conditions": [
            "correct_runtime",
            "wrong_sbox_same_state",
            "transition_branch_off_same_state",
        ],
    }:
        raise ValueError("K1-AO control contract drifted")
    return config


def build_runtime_model(
    cipher: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    wrong_sbox: bool = False,
) -> torch.nn.Module:
    options = {
        "runtime_structure_path": str(cipher["runtime_structure_path"]),
        "runtime_round_start": int(cipher["runtime_round_start"]),
        "runtime_rounds": int(cipher["runtime_rounds"]),
        "pair_embedding_dim": int(model["pair_embedding_dim"]),
        "transition_value_dim": int(model["transition_value_dim"]),
        "virtual_projection_slots": int(model["virtual_projection_slots"]),
        "dropout": float(model["dropout"]),
        "residual_gate_initial_effective": float(
            model["residual_gate_initial_effective"]
        ),
        "transition_gate_initial_effective": float(
            model["transition_gate_initial_effective"]
        ),
    }
    return build_model(
        WRONG_MODEL_KEY if wrong_sbox else MODEL_KEY,
        input_bits=int(cipher["input_bits"]),
        hidden_bits=int(model["hidden_bits"]),
        pair_bits=int(cipher["pair_bits"]),
        structure="SPN",
        model_options=options,
    )


def audit_source_datasets(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], DiskDifferentialDataset], dict[str, bool]]:
    selected_rows: list[dict[str, Any]] = []
    datasets: dict[tuple[str, int, str], DiskDifferentialDataset] = {}
    source_digests_exact = True
    cache_metadata_exact = True
    cache_payload_hashes_exact = True
    all_payloads_present = True

    for cipher in config["ciphers"]:
        cipher_key = str(cipher["cipher_key"])
        source_root = project_root / str(cipher["source_root"])
        observed_digests = {
            name: file_sha256(source_root / name)
            for name in cipher["source_digests"]
        }
        source_digests_exact &= observed_digests == cipher["source_digests"]
        manifest = _read_jsonl(source_root / "dataset_manifest.jsonl")
        expected_keys = {
            (cipher_key, int(seed), split)
            for seed in cipher["seeds"]
            for split in EXPECTED_SPLITS
        }
        source_rows = {
            (str(row.get("cipher_key")), int(row.get("seed", -1)), str(row.get("split"))): row
            for row in manifest
            if str(row.get("cipher_key")) == cipher_key
            and int(row.get("seed", -1)) in set(cipher["seeds"])
            and str(row.get("split")) in EXPECTED_SPLITS
        }
        if set(source_rows) != expected_keys:
            cache_metadata_exact = False
            continue
        for key in sorted(expected_keys):
            source = source_rows[key]
            cache_dir = project_root / str(source["cache_dir"])
            features_path = cache_dir / "features.npy"
            labels_path = cache_dir / "labels.npy"
            metadata_path = cache_dir / "metadata.json"
            present = all(
                path.is_file()
                for path in (features_path, labels_path, metadata_path)
            )
            all_payloads_present &= present
            if not present:
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            dataset = DiskDifferentialDataset(
                features=np.load(features_path, mmap_mode="r"),
                labels=np.load(labels_path, mmap_mode="r"),
                metadata=metadata,
                cache_dir=cache_dir,
            )
            dataset_sha256 = differential_dataset_sha256(dataset)
            expected_rows = 4096 if key[2] == "train_seen" else 2048
            expected_samples_per_class = 2048 if key[2] == "train_seen" else 1024
            metadata_ok = (
                int(metadata.get("rounds", -1)) == int(cipher["rounds"])
                and int(metadata.get("input_difference", -1))
                == int(str(cipher["input_difference_hex"]), 0)
                and int(metadata.get("input_bits", -1)) == int(cipher["input_bits"])
                and int(metadata.get("pair_bits", -1)) == int(cipher["pair_bits"])
                and int(metadata.get("pairs_per_sample", -1)) == 4
                and int(metadata.get("samples_per_class", -1))
                == expected_samples_per_class
                and int(metadata.get("total_rows", -1)) == expected_rows
                and metadata.get("negative_mode") == "encrypted_random_plaintexts"
                and metadata.get("feature_encoding") == "ciphertext_pair_bits"
                and metadata.get("sample_structure") == "independent_pairs"
                and int(source.get("rows", -1)) == expected_rows
            )
            cache_metadata_exact &= metadata_ok
            hash_ok = dataset_sha256 == str(source.get("dataset_sha256"))
            cache_payload_hashes_exact &= hash_ok
            datasets[key] = dataset
            selected_rows.append(
                {
                    "run_id": RUN_ID,
                    "cipher_key": cipher_key,
                    "rounds": int(cipher["rounds"]),
                    "seed": key[1],
                    "split": key[2],
                    "rows": expected_rows,
                    "pairs_per_sample": 4,
                    "input_bits": int(cipher["input_bits"]),
                    "input_difference_hex": str(cipher["input_difference_hex"]),
                    "cache_dir": str(source["cache_dir"]),
                    "features_sha256": file_sha256(features_path),
                    "labels_sha256": file_sha256(labels_path),
                    "metadata_sha256": file_sha256(metadata_path),
                    "dataset_sha256": dataset_sha256,
                    "expected_dataset_sha256": str(source.get("dataset_sha256")),
                    "payload_hash_exact": hash_ok,
                    "metadata_exact": metadata_ok,
                    "data_generation_performed": False,
                }
            )

    checks = {
        "nine_source_artifact_digests_exact": source_digests_exact,
        "eighteen_cache_payloads_selected": (
            len(selected_rows) == EXPECTED_DATASET_ROWS
            and len(datasets) == EXPECTED_DATASET_ROWS
        ),
        "all_cache_payload_files_present": all_payloads_present,
        "all_cache_metadata_exact": cache_metadata_exact,
        "all_cache_payload_hashes_exact": cache_payload_hashes_exact,
        "no_dataset_generation": all(
            row["data_generation_performed"] is False for row in selected_rows
        ),
    }
    return selected_rows, datasets, checks


def audit_shared_runtime(
    config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], DiskDifferentialDataset],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    model_config = config["model"]
    correct_models: dict[str, torch.nn.Module] = {}
    wrong_models: dict[str, torch.nn.Module] = {}
    with torch.random.fork_rng():
        for cipher in config["ciphers"]:
            cipher_key = str(cipher["cipher_key"])
            torch.manual_seed(int(model_config["initialization_seed"]))
            correct_models[cipher_key] = build_runtime_model(cipher, model_config)
            torch.manual_seed(int(model_config["initialization_seed"]))
            wrong_models[cipher_key] = build_runtime_model(
                cipher,
                model_config,
                wrong_sbox=True,
            )

    geometries = {
        cipher_key: tuple(
            (name, tuple(value.shape))
            for name, value in model.state_dict().items()
        )
        for cipher_key, model in correct_models.items()
    }
    initial_hashes = {
        cipher_key: tensor_mapping_sha256(model.state_dict())
        for cipher_key, model in correct_models.items()
    }
    parameter_counts = {
        cipher_key: int(model_metadata(model)["trainable_parameter_count"])
        for cipher_key, model in correct_models.items()
    }
    state_entries = {
        cipher_key: len(model.state_dict())
        for cipher_key, model in correct_models.items()
    }
    shared_state = correct_models[EXPECTED_CIPHERS[0]].state_dict()
    shared_state_hash = tensor_mapping_sha256(shared_state)
    strict_loads: dict[str, bool] = {}
    for cipher_key, model in correct_models.items():
        model.load_state_dict(shared_state, strict=True)
        strict_loads[cipher_key] = (
            tensor_mapping_sha256(model.state_dict()) == shared_state_hash
        )

    runtime_manifest: list[dict[str, Any]] = []
    for cipher in config["ciphers"]:
        cipher_key = str(cipher["cipher_key"])
        model = correct_models[cipher_key]
        runtime_manifest.append(
            {
                "run_id": RUN_ID,
                "cipher_key": cipher_key,
                "block_bits": model.runtime_structure.block_bits,
                "cells": model.runtime_structure.cells,
                "runtime_rounds": model.runtime_structure.rounds,
                "input_bits": int(cipher["input_bits"]),
                "pair_bits": int(cipher["pair_bits"]),
                "trainable_parameter_count": parameter_counts[cipher_key],
                "state_dict_entries": state_entries[cipher_key],
                "state_sha256": tensor_mapping_sha256(model.state_dict()),
                "strict_shared_state_load": strict_loads[cipher_key],
                "runtime_window_sha256": model.runtime_structure.window_sha256(),
                "uses_cipher_identity": model.uses_cipher_identity,
                "uses_absolute_cell_or_bit_identity": (
                    model.uses_absolute_cell_or_bit_identity
                ),
                "uses_runtime_native_cell_slots": model.uses_runtime_native_cell_slots,
            }
        )

    shared_model = correct_models[EXPECTED_CIPHERS[0]]
    shared_model.eval()
    persistent_runtime_hash = shared_model.runtime_structure.window_sha256()
    persistent_branch_flag = shared_model.transition_branch_enabled
    state_before = tensor_mapping_sha256(shared_model.state_dict())
    result_rows: list[dict[str, Any]] = []
    all_forward_shapes = True
    all_finite = True
    all_wrong_sbox_controls_exact = True
    all_runtime_interventions_observable = True
    with torch.no_grad():
        for cipher in config["ciphers"]:
            cipher_key = str(cipher["cipher_key"])
            seed = int(cipher["seeds"][0])
            dataset = datasets[(cipher_key, seed, "train_seen")]
            features = torch.as_tensor(
                np.array(dataset.features[:3], copy=True),
                dtype=torch.float32,
            )
            correct_structure = correct_models[cipher_key].runtime_structure
            wrong_structure = wrong_models[cipher_key].runtime_structure
            wrong_only_sbox = (
                torch.equal(
                    correct_structure.cell_membership,
                    wrong_structure.cell_membership,
                )
                and torch.equal(correct_structure.bit_role, wrong_structure.bit_role)
                and torch.equal(
                    correct_structure.linear_matrices,
                    wrong_structure.linear_matrices,
                )
                and torch.equal(
                    correct_structure.inverse_linear_matrices,
                    wrong_structure.inverse_linear_matrices,
                )
                and not torch.equal(
                    correct_structure.sbox_truth_bits,
                    wrong_structure.sbox_truth_bits,
                )
            )
            all_wrong_sbox_controls_exact &= wrong_only_sbox
            logits = {
                "correct_runtime": shared_model.logits_with_runtime(
                    features,
                    correct_structure,
                    apply_sboxes=True,
                    transition_branch_enabled=True,
                ),
                "wrong_sbox_same_state": shared_model.logits_with_runtime(
                    features,
                    wrong_structure,
                    apply_sboxes=True,
                    transition_branch_enabled=True,
                ),
                "transition_branch_off_same_state": shared_model.logits_with_runtime(
                    features,
                    correct_structure,
                    apply_sboxes=True,
                    transition_branch_enabled=False,
                ),
            }
            reference = logits["correct_runtime"]
            wrong_delta = float(
                torch.max(torch.abs(reference - logits["wrong_sbox_same_state"]))
            )
            off_delta = float(
                torch.max(
                    torch.abs(
                        reference - logits["transition_branch_off_same_state"]
                    )
                )
            )
            all_runtime_interventions_observable &= wrong_delta > 0.0 and off_delta > 0.0
            for condition in config["controls"]["conditions"]:
                current = logits[condition]
                all_forward_shapes &= tuple(current.shape) == (3, 1)
                all_finite &= bool(torch.isfinite(current).all())
                result_rows.append(
                    {
                        "run_id": RUN_ID,
                        "cipher_key": cipher_key,
                        "condition": condition,
                        "input_shape": list(features.shape),
                        "output_shape": list(current.shape),
                        "logits_sha256": tensor_mapping_sha256({"logits": current}),
                        "max_abs_delta_from_correct": (
                            0.0
                            if condition == "correct_runtime"
                            else float(torch.max(torch.abs(reference - current)))
                        ),
                        "finite": bool(torch.isfinite(current).all()),
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "shared_state_sha256": state_before,
                    }
                )

    state_after = tensor_mapping_sha256(shared_model.state_dict())
    signature = inspect.signature(shared_model.logits_with_runtime)
    allowed_arguments = {
        "features",
        "structure",
        "apply_sboxes",
        "transition_branch_enabled",
    }
    checks = {
        "three_parameter_geometries_identical": len(set(geometries.values())) == 1,
        "parameter_count_exact_all_runtimes": set(parameter_counts.values())
        == {EXPECTED_PARAMETER_COUNT},
        "state_entry_count_exact_all_runtimes": set(state_entries.values())
        == {EXPECTED_STATE_ENTRIES},
        "same_seed_initial_tensor_hash_exact": len(set(initial_hashes.values())) == 1,
        "one_state_dict_strict_loads_all_runtimes": all(strict_loads.values()),
        "all_nine_forward_shapes_exact": (
            len(result_rows) == EXPECTED_RUNTIME_ROWS and all_forward_shapes
        ),
        "all_runtime_logits_finite": all_finite,
        "wrong_sbox_changes_only_sbox": all_wrong_sbox_controls_exact,
        "runtime_interventions_observable_all_ciphers": (
            all_runtime_interventions_observable
        ),
        "shared_state_immutable_under_all_interventions": state_before == state_after,
        "persistent_runtime_binding_not_mutated": (
            shared_model.runtime_structure.window_sha256() == persistent_runtime_hash
            and shared_model.transition_branch_enabled is persistent_branch_flag
        ),
        "no_cipher_or_absolute_identity": all(
            model.uses_cipher_identity is False
            and model.uses_absolute_cell_or_bit_identity is False
            and model.uses_runtime_native_cell_slots is False
            for model in correct_models.values()
        ),
        "runtime_call_has_no_key_label_difference_or_cipher_argument": (
            set(signature.parameters) == allowed_arguments
        ),
    }
    return runtime_manifest, result_rows, checks


def adjudicate_readiness(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    runtime_checks: Mapping[str, bool],
) -> dict[str, Any]:
    protocol_checks = {
        "configuration_digest_exact": file_sha256(CONFIG_PATH)
        == EXPECTED_CONFIG_SHA256,
        "zero_training_contract_exact": config.get("audit", {}).get("training_rows")
        == 0
        and config.get("audit", {}).get("validation_rows") == 0
        and config.get("audit", {}).get("optimizer_steps") == 0
        and config.get("audit", {}).get("data_generation") is False
        and config.get("audit", {}).get("remote") is False,
        **dict(source_checks),
    }
    evidence_checks = dict(runtime_checks)
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_evidence = [name for name, passed in evidence_checks.items() if not passed]
    passed = not failed_protocol and not failed_evidence
    return {
        "run_id": RUN_ID,
        "status": "pass" if passed else "hold",
        "decision": (
            "innovation1_uknit_family_k1ao_shared_weight_runtime_ready"
            if passed
            else "innovation1_uknit_family_k1ao_shared_weight_readiness_failed"
        ),
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "failed_protocol_checks": failed_protocol,
        "failed_evidence_checks": failed_evidence,
        "training_rows": 0,
        "validation_rows": 0,
        "optimizer_steps": 0,
        "data_generation": False,
        "remote_scale": "no",
        "claim_scope": (
            "Zero-training implementation readiness only. This does not establish "
            "AUC, semantic preference after training, cross-cipher transfer, an "
            "attack, formal scale, or SOTA."
        ),
        "next_action": (
            "Preregister and run two local shared-training replicas at "
            "2048/class/cipher, 1024/class fresh evaluation, four pairs and ten "
            "epochs with equal cipher batches; evaluate correct, wrong-S-box and "
            "branch-off runtimes at each shared checkpoint."
            if passed
            else "Repair only the failed runtime, geometry, or cache-binding check; do not train or scale."
        ),
        "blocked_actions": [
            "remote execution before the shared local training gate",
            "16-pair expansion or larger samples/epochs",
            "cipher IDs, cipher-specific experts, MoE, or separate adapters",
            "using Dialga averages to hide a failed uKNIT or Midori panel",
        ],
    }


def run_readiness(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    _require_fresh_output_root(output_root)
    started = time.time()
    dataset_rows, datasets, source_checks = audit_source_datasets(
        config,
        project_root=project_root,
    )
    runtime_manifest, result_rows, runtime_checks = audit_shared_runtime(
        config,
        datasets,
    )
    gate = adjudicate_readiness(
        config=config,
        source_checks=source_checks,
        runtime_checks=runtime_checks,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if gate["status"] == "pass" else "fail",
        "checks": {**gate["protocol_checks"], **gate["evidence_checks"]},
        "errors": [
            *gate["failed_protocol_checks"],
            *gate["failed_evidence_checks"],
        ],
        "dataset_manifest_rows": len(dataset_rows),
        "expected_dataset_manifest_rows": EXPECTED_DATASET_ROWS,
        "runtime_manifest_rows": len(runtime_manifest),
        "expected_runtime_manifest_rows": len(EXPECTED_CIPHERS),
        "result_rows": len(result_rows),
        "expected_result_rows": EXPECTED_RUNTIME_ROWS,
        "training_rows": 0,
        "validation_rows": 0,
        "optimizer_steps": 0,
    }
    preflight = {
        "run_id": RUN_ID,
        "status": "pass" if all(source_checks.values()) else "fail",
        "execution_authorized": gate["status"] == "pass",
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_checks": source_checks,
        "audit": dict(config["audit"]),
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "cipher_count": len(runtime_manifest),
        "dataset_manifest_rows": len(dataset_rows),
        "result_rows": len(result_rows),
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }

    output_root.mkdir(parents=True)
    _write_json(output_root / "preflight.json", preflight)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)
    _write_jsonl(output_root / "runtime_manifest.jsonl", runtime_manifest)
    _write_jsonl(output_root / "results.jsonl", result_rows)
    _write_comparison_csv(output_root / "comparison.csv", result_rows)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _write_jsonl(
        output_root / "progress.jsonl",
        [
            {
                "event": "run_start",
                "time": started,
                "training_rows": 0,
                "optimizer_steps": 0,
            },
            {
                "event": "run_done",
                "time": time.time(),
                "status": gate["status"],
                "decision": gate["decision"],
                "dataset_manifest_rows": len(dataset_rows),
                "result_rows": len(result_rows),
                "training_rows": 0,
                "optimizer_steps": 0,
            },
        ],
    )
    return {
        "preflight": preflight,
        "dataset_manifest": dataset_rows,
        "runtime_manifest": runtime_manifest,
        "results": result_rows,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_comparison_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = (
        "cipher_key",
        "condition",
        "max_abs_delta_from_correct",
        "finite",
        "output_shape",
        "shared_state_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})


def _require_fresh_output_root(path: Path) -> None:
    if path.exists() and any(
        (path / name).exists()
        for name in ("preflight.json", "results.jsonl", "gate.json")
    ):
        raise ValueError("K1-AO readiness output already exists")


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_CIPHERS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_STATE_ENTRIES",
    "RUN_ID",
    "adjudicate_readiness",
    "audit_shared_runtime",
    "audit_source_datasets",
    "build_runtime_model",
    "load_and_validate_config",
    "run_readiness",
]
