from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from blockcipher_nd.models.structure.spn.position_preserving_operator import (
    PositionPreservingOperatorK1AzProbe,
    PositionPreservingOperatorSpec,
    trainable_parameter_geometry,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    runtime_spn_structure_from_truth_bits,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1ay import (
    build_candidate,
)
from blockcipher_nd.tasks.innovation1.uknit_family_component_separated_structure_gate_k1az import (
    load_and_validate_config as load_k1az_config,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_linear_summary_collision_k1ba import (
    load_and_validate_config as load_k1ba_config,
    load_authority as load_k1ba_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_position_preserving_operator_k1bb import (
    load_and_validate_config as load_k1bb_config,
    load_authority as load_k1bb_authority,
)
from blockcipher_nd.tasks.innovation1.uknit_family_structure_derived_gate_k1at import (
    FRESH_SPLITS,
)
from blockcipher_nd.training.metrics import binary_auc


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = (
    "i1_uknit_family_position_preserving_operator_k1bc_"
    "2048_replica0_replica1_20260729"
)
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_position_preserving_operator_"
    "k1bc_2048_replica0_replica1_20260729.json"
)
EXPECTED_CONFIG_SHA256 = (
    "e6c1418536bc0d7a9b8ca40c19bb3c9d4eee9178bf42301c129d7cda6cefb0c6"
)
EXPECTED_REPLICAS = (0, 1)
EXPECTED_EPOCHS = 10
EXPECTED_BATCH_SIZE = 64
EXPECTED_BATCHES_PER_CIPHER = 64
EXPECTED_STEPS_PER_EPOCH = 192
EXPECTED_STEPS_PER_REPLICA = 1920
EXPECTED_TRAINABLE_PARAMETERS = 41088
CONTROL_CONDITIONS = (
    "correct_operator",
    "same_summary_corrupted_operator",
    "cross_cipher_operator",
    "disabled_k1az",
)
TOPOLOGY_CONTROLS = (
    "same_summary_corrupted_operator",
    "cross_cipher_operator",
)
EXPECTED_EVALUATION_ROWS = 48
MACRO_IMPROVEMENT = 0.0
NO_HARM_MARGIN = -0.005
CONTROL_MARGIN = 0.001
MINIMUM_PASSING_PANELS = 10
MINIMUM_PASSING_PER_CIPHER = 3


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("K1-BC config digest drifted")
    if config.get("schema_version") != 1 or config.get("run_id") != RUN_ID:
        raise ValueError("K1-BC identity drifted")
    if config.get("experiment") != (
        "innovation1_uknit_family_position_preserving_operator_k1bc"
    ):
        raise ValueError("K1-BC experiment name drifted")
    if config.get("model") != {
        "hidden_dim": 32,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "modulation_scale": 0.05,
        "encoder_initialization_seeds": [40, 41],
        "frozen_anchor": "exact_k1az_epoch9_checkpoint",
        "trainable_path": "position_preserving_operator_encoder_only",
        "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "cipher_identity": False,
        "per_cipher_modules": False,
    }:
        raise ValueError("K1-BC model contract drifted")
    if config.get("replicas") != [
        {
            "replica": 0,
            "encoder_initialization_seed": 40,
            "dataset_seeds": {"uknit64": 3, "midori64": 6, "dialga128": 0},
        },
        {
            "replica": 1,
            "encoder_initialization_seed": 41,
            "dataset_seeds": {"uknit64": 4, "midori64": 7, "dialga128": 1},
        },
    ]:
        raise ValueError("K1-BC replica contract drifted")
    if config.get("training") != {
        "samples_per_class_per_cipher": 2048,
        "fresh_samples_per_class_per_cipher": 1024,
        "pairs_per_sample": 4,
        "epochs": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "equal_batches_per_cipher_per_epoch": EXPECTED_BATCHES_PER_CIPHER,
        "optimizer_steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "optimizer_steps_total_per_replica": EXPECTED_STEPS_PER_REPLICA,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "checkpoint_metric": "minimum_cross_key_auc_across_ciphers",
        "negative_mode": "encrypted_random_plaintexts",
        "execution": "local_diagnostic",
    }:
        raise ValueError("K1-BC training contract drifted")
    if config.get("controls") != {
        "splits": list(FRESH_SPLITS),
        "conditions": list(CONTROL_CONDITIONS),
        "cross_operator_construction": {
            "uknit64": "midori64_native_operator",
            "midori64": "uknit64_native_operator",
            "dialga128": "uknit64_block_diagonal_x2",
        },
        "corruption_seed": 20260729,
        "expected_rows": EXPECTED_EVALUATION_ROWS,
        "optimizer_steps": 0,
    }:
        raise ValueError("K1-BC control contract drifted")
    if config.get("gates") != {
        "cross_key_macro_improvement_per_replica": MACRO_IMPROVEMENT,
        "per_panel_no_harm_margin": NO_HARM_MARGIN,
        "correct_minus_control_margin": CONTROL_MARGIN,
        "minimum_passing_panels_per_topology_control": MINIMUM_PASSING_PANELS,
        "minimum_passing_panels_per_cipher_per_topology_control": (
            MINIMUM_PASSING_PER_CIPHER
        ),
        "require_each_replica_and_split_covered": True,
        "disabled_k1az_is_same_budget_anchor": True,
        "remote_scale": "no",
    }:
        raise ValueError("K1-BC gate contract drifted")
    return config


def load_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Mapping[tuple[str, int, str], Any],
    Mapping[str, Any],
    Mapping[str, Mapping[str, torch.Tensor | None]],
    Mapping[int, Mapping[str, Any]],
    Mapping[tuple[int, str, str], Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, bool],
]:
    readiness_source = config["readiness"]
    readiness_root = project_root / str(readiness_source["root"])
    readiness_paths = {
        name: readiness_root / name for name in readiness_source["digests"]
    }
    readiness_gate = _read_json(readiness_paths["gate.json"])
    readiness_validation = _read_json(readiness_paths["validation.json"])
    readiness_config = load_k1bb_config(
        project_root / str(readiness_source["config"])
    )
    (
        runtime_config,
        dataset_rows,
        datasets,
        structures,
        summaries,
        checkpoints,
        readiness_controls,
        inherited_readiness_checks,
    ) = load_k1bb_authority(
        readiness_config,
        project_root=project_root,
        device=device,
    )

    anchor_source = config["same_budget_anchor"]
    anchor_root = project_root / str(anchor_source["root"])
    anchor_paths = {name: anchor_root / name for name in anchor_source["digests"]}
    anchor_gate = _read_json(anchor_paths["gate.json"])
    anchor_validation = _read_json(anchor_paths["validation.json"])
    anchor_controls_rows = _read_jsonl(anchor_paths["controls.jsonl"])
    anchor_config = load_k1az_config(project_root / str(anchor_source["config"]))
    k1ba_source_config = load_k1ba_config(
        project_root / str(readiness_config["source"]["config"])
    )
    (
        _runtime_again,
        _dataset_rows_again,
        _datasets_again,
        _structures_again,
        _summaries_again,
        _summary_rows_again,
        checkpoints_again,
        source_controls,
        inherited_anchor_checks,
    ) = load_k1ba_authority(
        k1ba_source_config,
        project_root=project_root,
        device=device,
    )
    anchors = {
        (replica, cipher, split): row
        for (replica, cipher, split, condition), row in source_controls.items()
        if condition == "correct_descriptor"
    }
    expected_anchors = {
        (replica, cipher, split)
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    cross_operators = derive_cross_operator_controls(structures)
    cross_checks = validate_cross_operator_controls(structures, cross_operators)
    checks = {
        "k1bb_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == readiness_source["digests"][name]
            for name, path in readiness_paths.items()
        ),
        "k1bb_gate_authorizes_training": (
            readiness_gate.get("run_id") == readiness_source["run_id"]
            and readiness_gate.get("status") == "pass"
            and readiness_gate.get("decision")
            == readiness_source["required_decision"]
            and not readiness_gate.get("failed_protocol_checks")
            and not readiness_gate.get("failed_panel_checks")
        ),
        "k1bb_validation_passes": readiness_validation.get("status") == "pass"
        and not readiness_validation.get("errors"),
        "k1az_artifact_digests_exact": all(
            path.is_file() and file_sha256(path) == anchor_source["digests"][name]
            for name, path in anchor_paths.items()
        ),
        "k1az_gate_is_valid_hold": (
            anchor_gate.get("run_id") == anchor_source["run_id"]
            and anchor_gate.get("status") == "hold"
            and anchor_gate.get("decision") == anchor_source["required_decision"]
            and not anchor_gate.get("failed_protocol_checks")
        ),
        "k1az_validation_passes": anchor_validation.get("status") == "pass"
        and not anchor_validation.get("errors"),
        "k1az_anchor_rows_complete": len(anchor_controls_rows) == 60
        and set(anchors) == expected_anchors,
        "source_checkpoint_replay_exact": all(
            checkpoints[replica]["sha256"] == checkpoints_again[replica]["sha256"]
            and checkpoints[replica]["state_dict_sha256"]
            == checkpoints_again[replica]["state_dict_sha256"]
            for replica in EXPECTED_REPLICAS
        ),
        "replica_and_dataset_seeds_match_anchor": [
            {
                "replica": int(row["replica"]),
                "dataset_seeds": dict(row["dataset_seeds"]),
            }
            for row in config["replicas"]
        ]
        == [
            {
                "replica": int(row["replica"]),
                "dataset_seeds": dict(row["dataset_seeds"]),
            }
            for row in anchor_config["replicas"]
        ],
        **{
            f"readiness_{name}": bool(value)
            for name, value in inherited_readiness_checks.items()
        },
        **{
            f"anchor_{name}": bool(value)
            for name, value in inherited_anchor_checks.items()
        },
        **{f"cross_{name}": bool(value) for name, value in cross_checks.items()},
    }
    return (
        runtime_config,
        dataset_rows,
        datasets,
        structures,
        summaries,
        checkpoints,
        anchors,
        readiness_controls["corrupted_structures"],
        cross_operators,
        checks,
    )


def derive_cross_operator_controls(
    structures: Mapping[str, Any],
) -> dict[str, Any]:
    uknit = structures["uknit64"]
    midori = structures["midori64"]
    dialga = structures["dialga128"]
    uknit_for_midori = runtime_spn_structure_from_truth_bits(
        midori.cell_membership,
        midori.bit_role,
        midori.sbox_truth_bits,
        uknit.linear_matrices,
    )
    midori_for_uknit = runtime_spn_structure_from_truth_bits(
        uknit.cell_membership,
        uknit.bit_role,
        uknit.sbox_truth_bits,
        midori.linear_matrices,
    )
    lifted = torch.zeros_like(dialga.linear_matrices)
    lifted[:, : uknit.block_bits, : uknit.block_bits] = uknit.linear_matrices
    lifted[:, uknit.block_bits :, uknit.block_bits :] = uknit.linear_matrices
    uknit_for_dialga = runtime_spn_structure_from_truth_bits(
        dialga.cell_membership,
        dialga.bit_role,
        dialga.sbox_truth_bits,
        lifted,
    )
    return {
        "uknit64": midori_for_uknit,
        "midori64": uknit_for_midori,
        "dialga128": uknit_for_dialga,
    }


def validate_cross_operator_controls(
    structures: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "three_width_compatible_operators": set(controls) == set(EXPECTED_CIPHERS)
        and all(
            controls[cipher].block_bits == structures[cipher].block_bits
            and controls[cipher].rounds == structures[cipher].rounds
            for cipher in EXPECTED_CIPHERS
        ),
        "operators_are_distinct": all(
            not torch.equal(
                controls[cipher].linear_matrices,
                structures[cipher].linear_matrices,
            )
            for cipher in EXPECTED_CIPHERS
        ),
        "target_cell_and_sbox_semantics_held_correct": all(
            torch.equal(
                controls[cipher].cell_membership,
                structures[cipher].cell_membership,
            )
            and torch.equal(controls[cipher].bit_role, structures[cipher].bit_role)
            and torch.equal(
                controls[cipher].sbox_truth_bits,
                structures[cipher].sbox_truth_bits,
            )
            for cipher in EXPECTED_CIPHERS
        ),
        "dialga_control_is_two_uknit_blocks": torch.equal(
            controls["dialga128"].linear_matrices[:, :64, :64],
            structures["uknit64"].linear_matrices,
        )
        and torch.equal(
            controls["dialga128"].linear_matrices[:, 64:, 64:],
            structures["uknit64"].linear_matrices,
        )
        and int(controls["dialga128"].linear_matrices[:, :64, 64:].sum()) == 0
        and int(controls["dialga128"].linear_matrices[:, 64:, :64].sum()) == 0,
    }


def build_probe(
    *,
    runtime_config: Mapping[str, Any],
    structures: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    initialization_seed: int,
    model_config: Mapping[str, Any],
    device: str,
) -> PositionPreservingOperatorK1AzProbe:
    cipher_configs = {
        str(row["cipher_key"]): row for row in runtime_config["ciphers"]
    }
    with torch.random.fork_rng():
        torch.manual_seed(initialization_seed)
        anchor = build_candidate(
            cipher_configs[EXPECTED_CIPHERS[0]],
            runtime_config["model"],
            {"gate_hidden_dim": 12},
        ).to(device)
        incompatible = anchor.load_state_dict(checkpoint["state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("K1-BC strict K1-AZ checkpoint load drifted")
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        probe = PositionPreservingOperatorK1AzProbe(
            anchor,
            PositionPreservingOperatorSpec(
                hidden_dim=int(model_config["hidden_dim"]),
                pair_embedding_dim=int(model_config["pair_embedding_dim"]),
                dropout=float(model_config["dropout"]),
                modulation_scale=float(model_config["modulation_scale"]),
            ),
        ).to(device)
    if sum(
        parameter.numel()
        for parameter in probe.parameters()
        if parameter.requires_grad
    ) != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("K1-BC trainable parameter count drifted")
    if any(parameter.requires_grad for parameter in probe.anchor.parameters()):
        raise ValueError("K1-BC anchor must remain frozen")
    if structures["uknit64"].block_bits != 64 or structures["dialga128"].block_bits != 128:
        raise ValueError("K1-BC frozen structure widths drifted")
    return probe


def train_replicas(
    *,
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], Any],
    structures: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, torch.Tensor | None]],
    source_checkpoints: Mapping[int, Mapping[str, Any]],
    output_root: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    checkpoints: dict[int, dict[str, Any]] = {}
    history_rows: list[dict[str, Any]] = []
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for replica_config in config["replicas"]:
        replica = int(replica_config["replica"])
        initialization_seed = int(replica_config["encoder_initialization_seed"])
        probe = build_probe(
            runtime_config=runtime_config,
            structures=structures,
            checkpoint=source_checkpoints[replica],
            initialization_seed=initialization_seed,
            model_config=config["model"],
            device=device,
        )
        anchor_state_before = tensor_mapping_sha256(probe.anchor.state_dict())
        initial_encoder_state_sha256 = tensor_mapping_sha256(
            probe.operator_encoder.state_dict()
        )
        optimizer = torch.optim.Adam(
            probe.operator_encoder.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_min_auc = -math.inf
        best_mean_auc = -math.inf
        best_aucs: dict[str, float] = {}
        step_count = 0
        for epoch in range(1, EXPECTED_EPOCHS + 1):
            probe.train()
            permutations = {
                cipher: np.random.default_rng(
                    initialization_seed * 1000
                    + epoch * 10
                    + list(EXPECTED_CIPHERS).index(cipher)
                ).permutation(4096)
                for cipher in EXPECTED_CIPHERS
            }
            loss_sums = {cipher: 0.0 for cipher in EXPECTED_CIPHERS}
            batch_counts = {cipher: 0 for cipher in EXPECTED_CIPHERS}
            for batch_index in range(EXPECTED_BATCHES_PER_CIPHER):
                for cipher in EXPECTED_CIPHERS:
                    seed = int(replica_config["dataset_seeds"][cipher])
                    dataset = datasets[(cipher, seed, "train_seen")]
                    indices = permutations[cipher][
                        batch_index * EXPECTED_BATCH_SIZE : (batch_index + 1)
                        * EXPECTED_BATCH_SIZE
                    ]
                    features = torch.as_tensor(
                        np.array(dataset.features[indices], copy=True),
                        dtype=torch.float32,
                        device=device,
                    )
                    labels = torch.as_tensor(
                        np.array(dataset.labels[indices], copy=True),
                        dtype=torch.float32,
                        device=device,
                    ).reshape(-1, 1)
                    summary = summaries[cipher]["correct_descriptor"]
                    if summary is None:
                        raise ValueError("K1-BC correct summary is missing")
                    optimizer.zero_grad(set_to_none=True)
                    logits = probe.logits_with_operator(
                        features,
                        structures[cipher],
                        structures[cipher],
                        gate_summary=summary,
                    )
                    loss = nn.functional.mse_loss(torch.sigmoid(logits), labels)
                    loss.backward()
                    optimizer.step()
                    step_count += 1
                    loss_sums[cipher] += float(loss.detach().cpu())
                    batch_counts[cipher] += 1
            validation_aucs = {
                cipher: evaluate_probe_auc(
                    probe=probe,
                    dataset=datasets[
                        (
                            cipher,
                            int(replica_config["dataset_seeds"][cipher]),
                            "cross_key_validation",
                        )
                    ],
                    runtime_structure=structures[cipher],
                    operator_structure=structures[cipher],
                    summary=summaries[cipher]["correct_descriptor"],
                    enabled=True,
                    batch_size=EXPECTED_BATCH_SIZE,
                    device=device,
                )[0]
                for cipher in EXPECTED_CIPHERS
            }
            minimum_auc = min(validation_aucs.values())
            mean_auc = float(np.mean(list(validation_aucs.values())))
            if minimum_auc > best_min_auc or (
                minimum_auc == best_min_auc and mean_auc > best_mean_auc
            ):
                best_min_auc = minimum_auc
                best_mean_auc = mean_auc
                best_epoch = epoch
                best_aucs = dict(validation_aucs)
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in probe.operator_encoder.state_dict().items()
                }
            history_rows.append(
                {
                    "run_id": RUN_ID,
                    "replica": replica,
                    "epoch": epoch,
                    **{
                        f"train_loss_{cipher}": loss_sums[cipher]
                        / batch_counts[cipher]
                        for cipher in EXPECTED_CIPHERS
                    },
                    **{
                        f"cross_key_auc_{cipher}": validation_aucs[cipher]
                        for cipher in EXPECTED_CIPHERS
                    },
                    "cross_key_minimum_auc": minimum_auc,
                    "cross_key_macro_auc": mean_auc,
                    "optimizer_steps_cumulative": step_count,
                }
            )
            _append_progress(
                output_root / "progress.jsonl",
                "epoch_done",
                replica=replica,
                epoch=epoch,
                cross_key_minimum_auc=minimum_auc,
                cross_key_macro_auc=mean_auc,
                optimizer_steps=step_count,
            )
        if best_state is None:
            raise RuntimeError("K1-BC did not select an encoder checkpoint")
        incompatible = probe.operator_encoder.load_state_dict(best_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("K1-BC selected encoder strict load drifted")
        anchor_state_after = tensor_mapping_sha256(probe.anchor.state_dict())
        optimizer_steps = _optimizer_step_range(optimizer)
        checkpoint_path = checkpoint_root / f"replica{replica}_best.pt"
        payload = {
            "run_id": RUN_ID,
            "replica": replica,
            "best_epoch": best_epoch,
            "best_cross_key_aucs": best_aucs,
            "best_minimum_cross_key_auc": best_min_auc,
            "best_mean_cross_key_auc": best_mean_auc,
            "source_checkpoint_sha256": source_checkpoints[replica]["sha256"],
            "source_state_dict_sha256": source_checkpoints[replica][
                "state_dict_sha256"
            ],
            "initial_encoder_state_sha256": initial_encoder_state_sha256,
            "encoder_state_dict": best_state,
            "encoder_state_dict_sha256": tensor_mapping_sha256(best_state),
        }
        torch.save(payload, checkpoint_path)
        checkpoints[replica] = {
            "replica": replica,
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "best_epoch": best_epoch,
            "best_cross_key_aucs": best_aucs,
            "best_minimum_cross_key_auc": best_min_auc,
            "best_mean_cross_key_auc": best_mean_auc,
            "source_checkpoint_sha256": source_checkpoints[replica]["sha256"],
            "source_state_dict_sha256": source_checkpoints[replica][
                "state_dict_sha256"
            ],
            "initial_encoder_state_sha256": initial_encoder_state_sha256,
            "encoder_state_dict_sha256": tensor_mapping_sha256(best_state),
            "encoder_state_dict": best_state,
        }
        result_rows.append(
            {
                "run_id": RUN_ID,
                "replica": replica,
                "encoder_initialization_seed": initialization_seed,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in probe.parameters()
                    if parameter.requires_grad
                ),
                "trainable_parameter_geometry": {
                    name: list(shape)
                    for name, shape in trainable_parameter_geometry(
                        probe.operator_encoder
                    ).items()
                },
                "anchor_all_parameters_frozen": not any(
                    parameter.requires_grad for parameter in probe.anchor.parameters()
                ),
                "anchor_state_immutable": anchor_state_before == anchor_state_after,
                "uses_cipher_identity": probe.uses_cipher_identity,
                "uses_per_cipher_parameters": probe.uses_per_cipher_parameters,
                "training": {
                    "epochs": EXPECTED_EPOCHS,
                    "optimizer_steps": step_count,
                    "optimizer_state_step_min": optimizer_steps[0],
                    "optimizer_state_step_max": optimizer_steps[1],
                    "one_shared_optimizer": True,
                    "equal_batches_per_cipher": True,
                    "anchor_frozen": True,
                    "operator_encoder_only": True,
                },
                "checkpoint": {
                    "best_epoch": best_epoch,
                    "best_cross_key_aucs": best_aucs,
                    "best_minimum_cross_key_auc": best_min_auc,
                    "best_mean_cross_key_auc": best_mean_auc,
                    "encoder_state_dict_sha256": tensor_mapping_sha256(best_state),
                },
            }
        )
    return result_rows, checkpoints, history_rows


def evaluate_controls(
    *,
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    datasets: Mapping[tuple[str, int, str], Any],
    structures: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, torch.Tensor | None]],
    source_checkpoints: Mapping[int, Mapping[str, Any]],
    trained_checkpoints: Mapping[int, Mapping[str, Any]],
    anchors: Mapping[tuple[int, str, str], Mapping[str, Any]],
    corrupted_structures: Mapping[str, Any],
    cross_operators: Mapping[str, Any],
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replica_config in config["replicas"]:
        replica = int(replica_config["replica"])
        probe = build_probe(
            runtime_config=runtime_config,
            structures=structures,
            checkpoint=source_checkpoints[replica],
            initialization_seed=int(replica_config["encoder_initialization_seed"]),
            model_config=config["model"],
            device=device,
        )
        incompatible = probe.operator_encoder.load_state_dict(
            trained_checkpoints[replica]["encoder_state_dict"],
            strict=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("K1-BC evaluation encoder strict load drifted")
        probe.eval()
        checkpoint_state_sha256 = tensor_mapping_sha256(probe.state_dict())
        for cipher in EXPECTED_CIPHERS:
            seed = int(replica_config["dataset_seeds"][cipher])
            correct_summary = summaries[cipher]["correct_descriptor"]
            if correct_summary is None:
                raise ValueError("K1-BC correct summary is missing")
            operators = {
                "correct_operator": structures[cipher],
                "same_summary_corrupted_operator": corrupted_structures[cipher],
                "cross_cipher_operator": cross_operators[cipher],
                "disabled_k1az": structures[cipher],
            }
            for split in FRESH_SPLITS:
                dataset = datasets[(cipher, seed, split)]
                state_before = tensor_mapping_sha256(probe.state_dict())
                condition_aucs: dict[str, float] = {}
                condition_hashes: dict[str, str] = {}
                for condition in CONTROL_CONDITIONS:
                    auc, probabilities = evaluate_probe_auc(
                        probe=probe,
                        dataset=dataset,
                        runtime_structure=structures[cipher],
                        operator_structure=operators[condition],
                        summary=correct_summary,
                        enabled=condition != "disabled_k1az",
                        batch_size=EXPECTED_BATCH_SIZE,
                        device=device,
                    )
                    condition_aucs[condition] = auc
                    condition_hashes[condition] = _array_sha256(probabilities)
                state_after = tensor_mapping_sha256(probe.state_dict())
                correct_auc = condition_aucs["correct_operator"]
                anchor = anchors[(replica, cipher, split)]
                for condition in CONTROL_CONDITIONS:
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "replica": replica,
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "condition": condition,
                            "auc": condition_aucs[condition],
                            "k1az_anchor_auc": float(anchor["auc"]),
                            "correct_minus_k1az_auc": correct_auc
                            - float(anchor["auc"]),
                            "correct_minus_condition_auc": (
                                0.0
                                if condition == "correct_operator"
                                else correct_auc - condition_aucs[condition]
                            ),
                            "rows": int(dataset.features.shape[0]),
                            "dataset_sha256": differential_dataset_sha256(dataset),
                            "probabilities_sha256": condition_hashes[condition],
                            "source_anchor_probability_sha256": anchor[
                                "probabilities_sha256"
                            ],
                            "disabled_probability_replay_exact": (
                                condition != "disabled_k1az"
                                or condition_hashes[condition]
                                == anchor["probabilities_sha256"]
                            ),
                            "disabled_auc_replay_delta": (
                                0.0
                                if condition != "disabled_k1az"
                                else condition_aucs[condition] - float(anchor["auc"])
                            ),
                            "runtime_structure_cipher_key": cipher,
                            "runtime_structure_held_correct": True,
                            "operator_control_only": condition
                            != "correct_operator",
                            "source_checkpoint_sha256": source_checkpoints[replica][
                                "sha256"
                            ],
                            "trained_checkpoint_sha256": trained_checkpoints[replica][
                                "sha256"
                            ],
                            "state_dict_sha256": checkpoint_state_sha256,
                            "state_immutable_across_controls": state_before
                            == state_after
                            == checkpoint_state_sha256,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    return rows


def evaluate_probe_auc(
    *,
    probe: PositionPreservingOperatorK1AzProbe,
    dataset: Any,
    runtime_structure: Any,
    operator_structure: Any,
    summary: torch.Tensor | None,
    enabled: bool,
    batch_size: int,
    device: str,
) -> tuple[float, np.ndarray]:
    if summary is None:
        raise ValueError("K1-BC evaluation summary is missing")
    outputs: list[np.ndarray] = []
    probe.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset.labels), batch_size):
            features = torch.as_tensor(
                np.array(dataset.features[start : start + batch_size], copy=True),
                dtype=torch.float32,
                device=device,
            )
            logits = probe.logits_with_operator(
                features,
                runtime_structure,
                operator_structure,
                gate_summary=summary,
                enabled=enabled,
            )
            outputs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
    probabilities = np.concatenate(outputs).astype(np.float64, copy=False)
    labels = np.asarray(dataset.labels, dtype=np.float32)
    return float(binary_auc(labels, probabilities)), probabilities


def adjudicate_training(
    *,
    config: Mapping[str, Any],
    source_checks: Mapping[str, bool],
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_panels = {
        (replica, cipher, split)
        for replica in EXPECTED_REPLICAS
        for cipher in EXPECTED_CIPHERS
        for split in FRESH_SPLITS
    }
    grouped: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in evaluation_rows:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    geometries = {
        json.dumps(row.get("trainable_parameter_geometry"), sort_keys=True)
        for row in training_rows
    }
    protocol_checks = {
        "training_config_digest_exact": file_sha256(CONFIG_PATH)
        == EXPECTED_CONFIG_SHA256,
        "all_source_and_operator_bindings_exact": bool(source_checks)
        and all(source_checks.values()),
        "two_training_rows_complete": len(training_rows) == 2
        and {int(row["replica"]) for row in training_rows}
        == set(EXPECTED_REPLICAS),
        "only_fixed_geometry_operator_encoder_trainable": len(geometries) == 1
        and all(
            int(row.get("trainable_parameter_count", -1))
            == EXPECTED_TRAINABLE_PARAMETERS
            and row.get("anchor_all_parameters_frozen") is True
            and row.get("anchor_state_immutable") is True
            and row.get("uses_cipher_identity") is False
            and row.get("uses_per_cipher_parameters") is False
            for row in training_rows
        ),
        "ten_epochs_and_1920_steps_each": all(
            int(row.get("training", {}).get("epochs", -1)) == EXPECTED_EPOCHS
            and int(row.get("training", {}).get("optimizer_steps", -1))
            == EXPECTED_STEPS_PER_REPLICA
            and int(row.get("training", {}).get("optimizer_state_step_min", -1))
            == EXPECTED_STEPS_PER_REPLICA
            and int(row.get("training", {}).get("optimizer_state_step_max", -1))
            == EXPECTED_STEPS_PER_REPLICA
            and row.get("training", {}).get("one_shared_optimizer") is True
            and row.get("training", {}).get("equal_batches_per_cipher") is True
            and row.get("training", {}).get("anchor_frozen") is True
            and row.get("training", {}).get("operator_encoder_only") is True
            for row in training_rows
        ),
        "two_selected_checkpoints_complete": set(checkpoints)
        == set(EXPECTED_REPLICAS)
        and all(
            Path(str(checkpoints[replica]["path"])).is_file()
            and file_sha256(Path(str(checkpoints[replica]["path"])))
            == checkpoints[replica]["sha256"]
            and 1 <= int(checkpoints[replica]["best_epoch"]) <= EXPECTED_EPOCHS
            for replica in EXPECTED_REPLICAS
        ),
        "forty_eight_evaluation_rows_complete": len(evaluation_rows)
        == EXPECTED_EVALUATION_ROWS
        and set(grouped) == expected_panels
        and all(
            set(conditions) == set(CONTROL_CONDITIONS)
            for conditions in grouped.values()
        ),
        "evaluation_zero_step_same_checkpoint_immutable": all(
            row.get("training_performed") is False
            and int(row.get("optimizer_steps", -1)) == 0
            and row.get("state_immutable_across_controls") is True
            for row in evaluation_rows
        ),
        "runtime_structure_held_correct_for_all_controls": all(
            row.get("runtime_structure_held_correct") is True
            and row.get("runtime_structure_cipher_key") == row.get("cipher_key")
            for row in evaluation_rows
        ),
        "disabled_path_exactly_replays_k1az": all(
            row.get("disabled_probability_replay_exact") is True
            and abs(float(row.get("disabled_auc_replay_delta", math.inf))) <= 1e-12
            for row in evaluation_rows
            if row.get("condition") == "disabled_k1az"
        ),
        "all_metrics_finite": all(
            math.isfinite(float(row.get("auc", math.nan)))
            and math.isfinite(float(row.get("correct_minus_k1az_auc", math.nan)))
            and math.isfinite(
                float(row.get("correct_minus_condition_auc", math.nan))
            )
            for row in evaluation_rows
        ),
    }
    panel_results: dict[str, dict[str, Any]] = {}
    no_harm_checks: dict[str, bool] = {}
    passing: dict[str, list[tuple[int, str, str]]] = {
        condition: [] for condition in TOPOLOGY_CONTROLS
    }
    for replica, cipher, split in sorted(expected_panels):
        conditions = grouped[(replica, cipher, split)]
        correct = conditions["correct_operator"]
        correct_auc = float(correct["auc"])
        anchor_auc = float(correct["k1az_anchor_auc"])
        prefix = f"replica{replica}_{cipher}_{split}"
        no_harm_checks[f"{prefix}_no_harm"] = (
            correct_auc - anchor_auc >= NO_HARM_MARGIN
        )
        condition_aucs = {
            condition: float(conditions[condition]["auc"])
            for condition in CONTROL_CONDITIONS
        }
        margins = {
            condition: correct_auc - condition_aucs[condition]
            for condition in CONTROL_CONDITIONS
        }
        for condition in TOPOLOGY_CONTROLS:
            if margins[condition] >= CONTROL_MARGIN:
                passing[condition].append((replica, cipher, split))
        panel_results[prefix] = {
            "replica": replica,
            "cipher_key": cipher,
            "split": split,
            "correct_auc": correct_auc,
            "k1az_anchor_auc": anchor_auc,
            "correct_minus_k1az": correct_auc - anchor_auc,
            "condition_aucs": condition_aucs,
            "correct_minus_condition": margins,
            "no_harm_pass": no_harm_checks[f"{prefix}_no_harm"],
        }
    macro_results: dict[str, Any] = {}
    macro_checks: dict[str, bool] = {}
    for replica in EXPECTED_REPLICAS:
        candidate_values = [
            float(
                grouped[(replica, cipher, "cross_key_validation")][
                    "correct_operator"
                ]["auc"]
            )
            for cipher in EXPECTED_CIPHERS
        ]
        anchor_values = [
            float(
                grouped[(replica, cipher, "cross_key_validation")][
                    "correct_operator"
                ]["k1az_anchor_auc"]
            )
            for cipher in EXPECTED_CIPHERS
        ]
        candidate_macro = float(np.mean(candidate_values))
        anchor_macro = float(np.mean(anchor_values))
        improvement = candidate_macro - anchor_macro
        macro_checks[f"replica{replica}_cross_key_macro_nonnegative"] = (
            improvement >= MACRO_IMPROVEMENT
        )
        macro_results[f"replica{replica}"] = {
            "candidate_cross_key_macro_auc": candidate_macro,
            "k1az_cross_key_macro_auc": anchor_macro,
            "improvement": improvement,
            "pass": improvement >= MACRO_IMPROVEMENT,
        }
    topology_results: dict[str, Any] = {}
    topology_checks: dict[str, bool] = {}
    for condition, passed_panels in passing.items():
        counts = {
            cipher: sum(panel[1] == cipher for panel in passed_panels)
            for cipher in EXPECTED_CIPHERS
        }
        replicas = {panel[0] for panel in passed_panels}
        splits = {panel[2] for panel in passed_panels}
        count_pass = len(passed_panels) >= MINIMUM_PASSING_PANELS
        cipher_pass = all(
            count >= MINIMUM_PASSING_PER_CIPHER for count in counts.values()
        )
        coverage_pass = replicas == set(EXPECTED_REPLICAS) and splits == set(
            FRESH_SPLITS
        )
        topology_checks[f"{condition}_at_least_10_of_12"] = count_pass
        topology_checks[f"{condition}_at_least_3_of_4_per_cipher"] = cipher_pass
        topology_checks[f"{condition}_covers_both_replicas_and_splits"] = (
            coverage_pass
        )
        topology_results[condition] = {
            "passing_panels": len(passed_panels),
            "expected_panels": 12,
            "passing_per_cipher": counts,
            "covered_replicas": sorted(replicas),
            "covered_splits": sorted(splits),
            "count_pass": count_pass,
            "cipher_pass": cipher_pass,
            "coverage_pass": coverage_pass,
        }
    research_checks = {**macro_checks, **no_harm_checks, **topology_checks}
    failed_protocol = [name for name, passed in protocol_checks.items() if not passed]
    failed_research = [name for name, passed in research_checks.items() if not passed]
    macro_pass = all(macro_checks.values())
    no_harm_pass = all(no_harm_checks.values())
    topology_pass = all(topology_checks.values())
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_uknit_family_k1bc_protocol_invalid"
        next_action = (
            "Repair only the failed source, operator construction, frozen anchor, "
            "step, checkpoint or evaluation binding and resume/replay K1-BC unchanged."
        )
    elif macro_pass and no_harm_pass and topology_pass:
        status = "pass"
        decision = (
            "innovation1_uknit_family_k1bc_position_preserving_operator_"
            "training_supported"
        )
        next_action = (
            "Retain K1-BB as the local family candidate and preregister a remote "
            "65536/class/cipher disk-cache and resume readiness audit. Do not launch "
            "until the pushed-source and durable-cache gates pass."
        )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_family_k1bc_position_preserving_operator_"
            "training_not_supported"
        )
        if (macro_pass and no_harm_pass) and not topology_pass:
            next_action = (
                "Freeze both K1-BC checkpoints and audit encoder-gradient and "
                "channelwise modulation attribution; performance without correct-"
                "topology preference is not scale-worthy."
            )
        elif topology_pass and not (macro_pass and no_harm_pass):
            next_action = (
                "Inspect the harmed cipher's shared operator-encoder gradients against "
                "the frozen K1-AZ anchor; do not add capacity, balance losses or scale."
            )
        else:
            next_action = (
                "Hold K1-BC and separate optimization failure from shared-representation "
                "interference using the frozen checkpoints; do not increase pairs, data, "
                "epochs, width or use remote GPU."
            )
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": failed_protocol,
        "failed_research_checks": failed_research,
        "panel_results": panel_results,
        "macro_results": macro_results,
        "topology_results": topology_results,
        "cross_key_macro_retention_both_replicas": macro_pass,
        "per_panel_no_harm_all": no_harm_pass,
        "correct_topology_attribution_all": topology_pass,
        "remote_scale": "no",
        "claim_scope": (
            "Local 2048/class/cipher, four-pair, two-replica diagnostic only; "
            "not formal scale, an attack, arbitrary-SPN generalization, unseen-"
            "cipher transfer or SOTA evidence."
        ),
        "next_action": next_action,
        "blocked_actions": list(config["blocked_actions"]),
    }


def run_training(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
    device: str = "cpu",
) -> dict[str, Any]:
    _require_fresh_output_root(output_root)
    output_root.mkdir(parents=True)
    _append_progress(output_root / "progress.jsonl", "run_start", run_id=RUN_ID)
    (
        runtime_config,
        dataset_rows,
        datasets,
        structures,
        summaries,
        source_checkpoints,
        anchors,
        corrupted_structures,
        cross_operators,
        source_checks,
    ) = load_authority(config, project_root=project_root, device=device)
    if not all(source_checks.values()):
        raise ValueError(f"K1-BC source preflight failed: {source_checks}")
    preflight = {
        "run_id": RUN_ID,
        "status": "pass",
        "execution_authorized": True,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "device": device,
        "source_checks": source_checks,
        "training": dict(config["training"]),
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_jsonl(output_root / "dataset_manifest.jsonl", dataset_rows)
    operator_controls = {
        "run_id": RUN_ID,
        "rows": [
            {
                "cipher_key": cipher,
                "correct_window_sha256": structures[cipher].window_sha256(),
                "corrupted_window_sha256": corrupted_structures[
                    cipher
                ].window_sha256(),
                "cross_window_sha256": cross_operators[cipher].window_sha256(),
                "correct_vs_corrupted_matrix_hamming_fraction": float(
                    (
                        structures[cipher].linear_matrices
                        != corrupted_structures[cipher].linear_matrices
                    )
                    .to(torch.float64)
                    .mean()
                ),
                "correct_vs_cross_matrix_hamming_fraction": float(
                    (
                        structures[cipher].linear_matrices
                        != cross_operators[cipher].linear_matrices
                    )
                    .to(torch.float64)
                    .mean()
                ),
            }
            for cipher in EXPECTED_CIPHERS
        ],
    }
    _write_json(output_root / "operator_controls.json", operator_controls)
    training_rows, trained_checkpoints, history_rows = train_replicas(
        config=config,
        runtime_config=runtime_config,
        datasets=datasets,
        structures=structures,
        summaries=summaries,
        source_checkpoints=source_checkpoints,
        output_root=output_root,
        device=device,
    )
    evaluation_rows = evaluate_controls(
        config=config,
        runtime_config=runtime_config,
        datasets=datasets,
        structures=structures,
        summaries=summaries,
        source_checkpoints=source_checkpoints,
        trained_checkpoints=trained_checkpoints,
        anchors=anchors,
        corrupted_structures=corrupted_structures,
        cross_operators=cross_operators,
        device=device,
    )
    gate = adjudicate_training(
        config=config,
        source_checks=source_checks,
        training_rows=training_rows,
        evaluation_rows=evaluation_rows,
        checkpoints=trained_checkpoints,
    )
    checkpoint_manifest = {
        "run_id": RUN_ID,
        "status": "pass",
        "entries": [
            {
                key: value
                for key, value in trained_checkpoints[replica].items()
                if key != "encoder_state_dict"
            }
            for replica in EXPECTED_REPLICAS
        ],
    }
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "training_rows": len(training_rows),
        "evaluation_rows": len(evaluation_rows),
        "expected_training_rows": 2,
        "expected_evaluation_rows": EXPECTED_EVALUATION_ROWS,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "macro_results": gate["macro_results"],
        "topology_results": gate["topology_results"],
        "per_panel_no_harm_all": gate["per_panel_no_harm_all"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    _write_jsonl(output_root / "results.jsonl", training_rows)
    _write_jsonl(output_root / "controls.jsonl", evaluation_rows)
    _write_history_csv(output_root / "history.csv", history_rows)
    _write_json(output_root / "checkpoint_manifest.json", checkpoint_manifest)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_progress(
        output_root / "progress.jsonl",
        "run_done",
        status=gate["status"],
        decision=gate["decision"],
        training_rows=len(training_rows),
        evaluation_rows=len(evaluation_rows),
    )
    return {
        "preflight": preflight,
        "results": training_rows,
        "controls": evaluation_rows,
        "checkpoint_manifest": checkpoint_manifest,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _optimizer_step_range(
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    steps = []
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            continue
        steps.append(int(value.item()) if isinstance(value, torch.Tensor) else int(value))
    if not steps:
        return (0, 0)
    return min(steps), max(steps)


def _write_history_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("K1-BC history is empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def _append_progress(path: Path, event: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event": event, "time": time.time(), **payload},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _require_fresh_output_root(path: Path) -> None:
    if path.exists() and any(
        (path / name).exists()
        for name in ("preflight.json", "results.jsonl", "gate.json")
    ):
        raise ValueError("K1-BC output already exists")


__all__ = [
    "CONFIG_PATH",
    "CONTROL_CONDITIONS",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_EVALUATION_ROWS",
    "ROOT",
    "RUN_ID",
    "TOPOLOGY_CONTROLS",
    "adjudicate_training",
    "build_probe",
    "derive_cross_operator_controls",
    "evaluate_controls",
    "evaluate_probe_auc",
    "load_and_validate_config",
    "load_authority",
    "run_training",
    "train_replicas",
    "validate_cross_operator_controls",
]
