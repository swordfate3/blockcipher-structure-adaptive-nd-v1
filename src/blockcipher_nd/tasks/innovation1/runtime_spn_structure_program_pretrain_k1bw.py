from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    load_runtime_spn_descriptor,
    runtime_spn_structure_from_truth_bits,
)
from blockcipher_nd.models.structure.spn.structure_program_encoder import (
    RuntimeSpnProgramEncoder,
    StructureProgramEncoderSpec,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_runtime_spn_structure_program_pretrain_k1bw_20260801"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_structure_program_pretrain_k1bw_20260801.json"
)
CONTROL_NAMES = ("wrong_linear", "wrong_sbox", "wrong_order")


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("run_id") != RUN_ID
        or config.get("experiment")
        != "innovation1_runtime_spn_structure_program_pretrain_k1bw"
    ):
        raise ValueError("K1-BW experiment identity drifted")
    structures = config.get("structures")
    if not isinstance(structures, dict) or set(structures) != {
        "gift64",
        "present64",
        "rectangle64",
        "skinny64",
        "midori64",
        "uknit64",
        "dialga128",
    }:
        raise ValueError("K1-BW structure panel drifted")
    training = config.get("training", {})
    train_ciphers = training.get("train_ciphers")
    holdout_ciphers = training.get("holdout_ciphers")
    if (
        not isinstance(train_ciphers, list)
        or not isinstance(holdout_ciphers, list)
        or set(train_ciphers) & set(holdout_ciphers)
        or set(train_ciphers) | set(holdout_ciphers) != set(structures)
        or training.get("seeds") != [0, 1]
        or training.get("corruption_seeds") != [11, 23, 37, 53]
        or training.get("device") != "cpu"
        or training.get("execution") != "local_structure_only_diagnostic"
    ):
        raise ValueError("K1-BW training protocol drifted")
    model = config.get("model", {})
    if (
        model.get("cipher_name_input") is not False
        or model.get("cipher_id_input") is not False
        or model.get("actual_gf2_edges") is not True
        or model.get("sbox_truth_tables") is not True
        or model.get("transition_order") is not True
    ):
        raise ValueError("K1-BW model contract drifted")
    return config


def load_structures(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> tuple[dict[str, RuntimeSpnStructure], list[dict[str, Any]]]:
    structures: dict[str, RuntimeSpnStructure] = {}
    manifest: list[dict[str, Any]] = []
    for cipher, entry in config["structures"].items():
        path = project_root / str(entry["path"])
        loaded = load_runtime_spn_descriptor(
            path,
            rounds=int(entry["rounds"]),
            round_start=int(entry["round_start"]),
        )
        structures[cipher] = loaded.structure
        manifest.append(
            {
                "cipher_key": cipher,
                "path": str(path.relative_to(project_root)),
                "sha256": loaded.sha256,
                "round_start": loaded.round_start,
                "rounds": loaded.structure.rounds,
                "block_bits": loaded.structure.block_bits,
                "cells": loaded.structure.cells,
                "unique_transitions": loaded.structure.unique_transition_count,
            }
        )
    return structures, manifest


def structure_variants(
    structure: RuntimeSpnStructure,
    *,
    seed: int,
) -> tuple[RuntimeSpnStructure, torch.Tensor, dict[str, RuntimeSpnStructure]]:
    generator = torch.Generator().manual_seed(seed + 1000)
    permutation = torch.randperm(structure.cells, generator=generator)
    if torch.equal(permutation, torch.arange(structure.cells)):
        permutation = torch.roll(permutation, shifts=1)
    relabeled, _ = structure.relabel_cells(permutation.tolist())
    transported = torch.empty(structure.cells, dtype=torch.long)
    for old_cell, new_cell in enumerate(permutation.tolist()):
        transported[new_cell] = old_cell
    variants = {
        "wrong_linear": structure.corrupted(seed),
        "wrong_sbox": _wrong_sbox_structure(structure),
    }
    rotated = structure.rotate_transitions()
    if rotated.window_sha256() != structure.window_sha256():
        variants["wrong_order"] = rotated
    return relabeled, transported, variants


def evaluate_encoder(
    encoder: RuntimeSpnProgramEncoder,
    structures: Mapping[str, RuntimeSpnStructure],
    corruption_seeds: Sequence[int],
    *,
    model_seed: int,
    phase: str,
    holdout_ciphers: set[str],
) -> list[dict[str, Any]]:
    encoder.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for cipher, structure in structures.items():
            anchor = encoder.encode_structure(structure)
            for corruption_seed in corruption_seeds:
                relabeled, positions, variants = structure_variants(
                    structure,
                    seed=corruption_seed,
                )
                positive = encoder.encode_structure(
                    relabeled,
                    cell_position_ids=positions,
                )
                positive_cosine = float(F.cosine_similarity(anchor, positive, dim=0))
                positive_distance = 1.0 - positive_cosine
                for control, wrong_structure in variants.items():
                    wrong = encoder.encode_structure(wrong_structure)
                    wrong_cosine = float(F.cosine_similarity(anchor, wrong, dim=0))
                    wrong_distance = 1.0 - wrong_cosine
                    rows.append(
                        {
                            "model_seed": model_seed,
                            "phase": phase,
                            "cipher_key": cipher,
                            "scope": "holdout" if cipher in holdout_ciphers else "train",
                            "corruption_seed": corruption_seed,
                            "control": control,
                            "positive_cosine": positive_cosine,
                            "positive_distance": positive_distance,
                            "wrong_cosine": wrong_cosine,
                            "wrong_distance": wrong_distance,
                            "semantic_margin": wrong_distance - positive_distance,
                        }
                    )
    return rows


def train_encoder(
    config: Mapping[str, Any],
    structures: Mapping[str, RuntimeSpnStructure],
    *,
    model_seed: int,
) -> tuple[RuntimeSpnProgramEncoder, list[dict[str, Any]]]:
    training = config["training"]
    model_config = config["model"]
    torch.manual_seed(model_seed)
    encoder = RuntimeSpnProgramEncoder(
        StructureProgramEncoderSpec(
            hidden_dim=int(model_config["hidden_dim"]),
            embedding_dim=int(model_config["embedding_dim"]),
            dropout=float(model_config["dropout"]),
        )
    )
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    corruption_seeds = [int(value) for value in training["corruption_seeds"]]
    margin = float(training["triplet_margin"])
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        encoder.train()
        corruption_seed = corruption_seeds[(epoch - 1) % len(corruption_seeds)]
        losses = []
        positive_distances = []
        negative_distances = []
        for cipher in training["train_ciphers"]:
            structure = structures[cipher]
            relabeled, positions, variants = structure_variants(
                structure,
                seed=corruption_seed,
            )
            anchor = encoder.encode_structure(structure)
            positive = encoder.encode_structure(
                relabeled,
                cell_position_ids=positions,
            )
            positive_distance = 1.0 - F.cosine_similarity(anchor, positive, dim=0)
            for wrong_structure in variants.values():
                wrong = encoder.encode_structure(wrong_structure)
                wrong_distance = 1.0 - F.cosine_similarity(anchor, wrong, dim=0)
                losses.append(F.relu(margin + positive_distance - wrong_distance))
                positive_distances.append(positive_distance.detach())
                negative_distances.append(wrong_distance.detach())
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(
            {
                "model_seed": model_seed,
                "epoch": epoch,
                "loss": float(loss.detach()),
                "positive_distance": float(torch.stack(positive_distances).mean()),
                "negative_distance": float(torch.stack(negative_distances).mean()),
                "corruption_seed": corruption_seed,
            }
        )
    return encoder, history


def adjudicate(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol_checks: Mapping[str, bool],
) -> dict[str, Any]:
    gates = config["gates"]
    failed_protocol = sorted(name for name, passed in protocol_checks.items() if not passed)
    research_checks: dict[str, bool] = {}
    summaries: dict[str, Any] = {}
    for model_seed in config["training"]["seeds"]:
        seed_rows = [row for row in rows if int(row["model_seed"]) == model_seed]
        initial = {
            (row["cipher_key"], row["corruption_seed"], row["control"]): row
            for row in seed_rows
            if row["phase"] == "initial"
        }
        trained = [row for row in seed_rows if row["phase"] == "trained"]
        min_relabel = min(float(row["positive_cosine"]) for row in trained)
        research_checks[f"seed{model_seed}_relabel_invariant"] = (
            min_relabel >= float(gates["relabel_cosine_min"])
        )
        holdout = [row for row in trained if row["scope"] == "holdout"]
        min_margin = min(float(row["semantic_margin"]) for row in holdout)
        min_gain = min(
            float(row["semantic_margin"])
            - float(
                initial[
                    (row["cipher_key"], row["corruption_seed"], row["control"])
                ]["semantic_margin"]
            )
            for row in holdout
        )
        research_checks[f"seed{model_seed}_holdout_margin"] = (
            min_margin
            >= float(gates["holdout_correct_minus_wrong_cosine_distance_min"])
        )
        research_checks[f"seed{model_seed}_holdout_gain"] = (
            min_gain >= float(gates["holdout_margin_gain_over_initial_min"])
        )
        summaries[str(model_seed)] = {
            "minimum_relabel_cosine": min_relabel,
            "minimum_holdout_semantic_margin": min_margin,
            "minimum_holdout_margin_gain": min_gain,
        }
    failed_research = sorted(name for name, passed in research_checks.items() if not passed)
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1bw_protocol_invalid"
    elif failed_research:
        status = "hold"
        decision = "innovation1_runtime_spn_k1bw_structure_program_not_ready"
    else:
        status = "pass"
        decision = "innovation1_runtime_spn_k1bw_structure_program_ready"
    return {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "failed_protocol_checks": failed_protocol,
        "failed_research_checks": failed_research,
        "protocol_checks": dict(protocol_checks),
        "research_checks": research_checks,
        "seed_summaries": summaries,
        "claim_scope": (
            "Local structure-only whole-cipher-holdout representation diagnostic; "
            "not neural-distinguisher accuracy, formal scale, arbitrary-SPN transfer, "
            "an attack or SOTA evidence."
        ),
        "remote_scale": "no",
        "next_action": (
            "If pass, preregister K1-BX to condition small Runtime-E4 primitive "
            "adapters on the frozen structure embedding with correct/wrong/uniform/"
            "no-structure controls. If hold, inspect the failed holdout primitive "
            "before any distinguisher training or scale-up."
        ),
    }


def run_experiment(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("K1-BW output root already contains artifacts")
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.jsonl"
    _append_jsonl(progress_path, {"event": "run_start", "run_id": RUN_ID, "time": time.time()})
    structures, manifest = load_structures(config, project_root=project_root)
    holdout = set(config["training"]["holdout_ciphers"])
    all_rows: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    geometries = []
    state_hashes = []
    for model_seed in config["training"]["seeds"]:
        torch.manual_seed(int(model_seed))
        model_config = config["model"]
        initial_encoder = RuntimeSpnProgramEncoder(
            StructureProgramEncoderSpec(
                hidden_dim=int(model_config["hidden_dim"]),
                embedding_dim=int(model_config["embedding_dim"]),
                dropout=float(model_config["dropout"]),
            )
        )
        initial_rows = evaluate_encoder(
            initial_encoder,
            structures,
            config["training"]["corruption_seeds"],
            model_seed=int(model_seed),
            phase="initial",
            holdout_ciphers=holdout,
        )
        encoder, history = train_encoder(
            config,
            structures,
            model_seed=int(model_seed),
        )
        trained_rows = evaluate_encoder(
            encoder,
            structures,
            config["training"]["corruption_seeds"],
            model_seed=int(model_seed),
            phase="trained",
            holdout_ciphers=holdout,
        )
        all_rows.extend(initial_rows)
        all_rows.extend(trained_rows)
        all_history.extend(history)
        geometry = {name: list(value.shape) for name, value in encoder.state_dict().items()}
        geometries.append(geometry)
        state_hashes.append(_state_sha256(encoder))
        _append_jsonl(
            progress_path,
            {
                "event": "model_seed_done",
                "model_seed": model_seed,
                "epochs": len(history),
                "time": time.time(),
            },
        )
    protocol_checks = {
        "seven_descriptors_loaded": len(structures) == 7,
        "dialga_whole_cipher_holdout": holdout == {"dialga128"},
        "all_two_transition_windows": all(value.rounds == 2 for value in structures.values()),
        "parameter_geometry_seed_stable": all(value == geometries[0] for value in geometries),
        "cipher_identity_absent": initial_encoder.uses_cipher_identity is False
        and initial_encoder.uses_cipher_name is False,
        "actual_edges_sboxes_and_order_used": (
            initial_encoder.uses_actual_source_target_connectivity
            and initial_encoder.uses_sbox_truth_tables
            and initial_encoder.preserves_transition_order
        ),
        "all_metrics_finite": all(
            torch.isfinite(torch.tensor(float(row[key])))
            for row in all_rows
            for key in (
                "positive_cosine",
                "positive_distance",
                "wrong_cosine",
                "wrong_distance",
                "semantic_margin",
            )
        ),
    }
    gate = adjudicate(config, all_rows, protocol_checks=protocol_checks)
    preflight = {
        "run_id": RUN_ID,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": _file_sha256(CONFIG_PATH),
        "device": config["training"]["device"],
        "cuda_available": torch.cuda.is_available(),
        "structure_manifest": manifest,
    }
    geometry = {
        "parameter_count": sum(value.numel() for value in initial_encoder.parameters()),
        "parameter_geometry": geometries[0],
        "state_sha256s": state_hashes,
        "cipher_name_input": False,
        "cipher_id_input": False,
    }
    validation = {
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "result_rows": len(all_rows),
        "history_rows": len(all_history),
        "expected_result_rows": len(all_rows),
        "errors": gate["failed_protocol_checks"],
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "seed_summaries": gate["seed_summaries"],
        "claim_scope": gate["claim_scope"],
        "next_action": gate["next_action"],
    }
    _write_json(output_root / "preflight.json", preflight)
    _write_json(output_root / "geometry.json", geometry)
    _write_jsonl(output_root / "results.jsonl", all_rows)
    _write_history_csv(output_root / "history.csv", all_history)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "summary.json", summary)
    _append_jsonl(
        progress_path,
        {
            "event": "run_done",
            "status": gate["status"],
            "decision": gate["decision"],
            "time": time.time(),
        },
    )
    return {
        "preflight": preflight,
        "geometry": geometry,
        "results": all_rows,
        "history": all_history,
        "gate": gate,
        "validation": validation,
        "summary": summary,
    }


def _wrong_sbox_structure(structure: RuntimeSpnStructure) -> RuntimeSpnStructure:
    tables = torch.stack(
        [structure.sbox_tables(stage) for stage in range(structure.rounds)]
    )
    shifted = tables[:, :, torch.roll(torch.arange(16), shifts=1)]
    shifts = torch.arange(4, dtype=torch.long)
    truth = (((shifted[..., None] >> shifts) & 1).reshape(structure.rounds, structure.cells, 64)).to(torch.uint8)
    return runtime_spn_structure_from_truth_bits(
        structure.cell_membership,
        structure.bit_role,
        truth,
        structure.linear_matrices,
    )


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_history_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = ("model_seed", "epoch", "loss", "positive_distance", "negative_distance", "corruption_seed")
    lines = [",".join(columns)]
    lines.extend(",".join(str(row[column]) for column in columns) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "CONFIG_PATH",
    "CONTROL_NAMES",
    "ROOT",
    "RUN_ID",
    "adjudicate",
    "evaluate_encoder",
    "load_and_validate_config",
    "load_structures",
    "run_experiment",
    "structure_variants",
    "train_encoder",
]
