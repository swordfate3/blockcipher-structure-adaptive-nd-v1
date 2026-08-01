from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from blockcipher_nd.models.structure.spn.runtime_structure import RuntimeSpnStructure
from blockcipher_nd.models.structure.spn.structure_program_encoder import (
    RuntimeSpnTargetCellProgramEncoder,
    StructureProgramEncoderSpec,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_structure_program_pretrain_k1bw import (
    load_structures,
    structure_variants,
)


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_runtime_spn_target_cell_program_k1bx0_20260801"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_runtime_spn_target_cell_program_k1bx0_20260801.json"
)
CONTROL_NAMES = (
    "wrong_linear",
    "wrong_sbox",
    "wrong_order",
    "wrong_edge_binding",
)


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if (
        config.get("schema_version") != 1
        or config.get("run_id") != RUN_ID
        or config.get("experiment")
        != "innovation1_runtime_spn_target_cell_program_k1bx0"
    ):
        raise ValueError("K1-BX0 experiment identity drifted")
    source = config.get("source", {})
    source_config_path = ROOT / str(source.get("config", ""))
    source_config = _read_json(source_config_path)
    if _file_sha256(source_config_path) != source.get("digests", {}).get("config"):
        raise ValueError("K1-BX0 K1-BW config digest drifted")
    if config.get("structures") != source_config.get("structures"):
        raise ValueError("K1-BX0 changed the K1-BW structure panel")
    if config.get("training") != source_config.get("training"):
        raise ValueError("K1-BX0 changed the K1-BW training protocol")
    model = config.get("model", {})
    source_model = source_config.get("model", {})
    if (
        model.get("variant") != "target_cell_message_aggregation"
        or any(
            model.get(field) != source_model.get(field)
            for field in (
                "hidden_dim",
                "embedding_dim",
                "dropout",
                "cipher_name_input",
                "cipher_id_input",
                "actual_gf2_edges",
                "sbox_truth_tables",
                "transition_order",
            )
        )
        or model.get("target_cell_aggregation_before_pooling") is not True
        or model.get("ordered_cell_transition") is not True
    ):
        raise ValueError("K1-BX0 model contract drifted")
    if config.get("controls") != list(CONTROL_NAMES):
        raise ValueError("K1-BX0 control panel drifted")
    return config


def load_source_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    source = config["source"]
    source_root = project_root / str(source["root"])
    paths = {
        name: source_root / name
        for name in ("gate.json", "results.jsonl", "validation.json")
    }
    checks = {
        f"k1bw_{name}_digest_exact": path.is_file()
        and _file_sha256(path) == source["digests"][name]
        for name, path in paths.items()
    }
    gate = _read_json(paths["gate.json"])
    validation = _read_json(paths["validation.json"])
    checks["k1bw_expected_hold_exact"] = (
        gate.get("status") == source["required_status"]
        and gate.get("decision") == source["required_decision"]
        and not gate.get("failed_protocol_checks")
        and validation.get("status") == "pass"
    )
    rows = _read_jsonl(paths["results.jsonl"])
    checks["k1bw_two_seed_holdout_anchor_complete"] = all(
        any(
            int(row.get("model_seed", -1)) == seed
            and row.get("phase") == "trained"
            and row.get("scope") == "holdout"
            and row.get("control") == control
            for row in rows
        )
        for seed in config["training"]["seeds"]
        for control in ("wrong_linear", "wrong_sbox", "wrong_order")
    )
    return rows, checks


def build_encoder(config: Mapping[str, Any]) -> RuntimeSpnTargetCellProgramEncoder:
    model = config["model"]
    return RuntimeSpnTargetCellProgramEncoder(
        StructureProgramEncoderSpec(
            hidden_dim=int(model["hidden_dim"]),
            embedding_dim=int(model["embedding_dim"]),
            dropout=float(model["dropout"]),
        )
    )


def evaluate_encoder(
    encoder: RuntimeSpnTargetCellProgramEncoder,
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
                    seed=int(corruption_seed),
                )
                positive = encoder.encode_structure(
                    relabeled,
                    cell_position_ids=positions,
                )
                positive_cosine = float(F.cosine_similarity(anchor, positive, dim=0))
                controls: dict[str, torch.Tensor] = {
                    name: encoder.encode_structure(wrong)
                    for name, wrong in variants.items()
                }
                controls["wrong_edge_binding"] = encoder.encode_structure(
                    structure,
                    edge_binding_seed=int(corruption_seed),
                )
                for control, wrong_embedding in controls.items():
                    wrong_cosine = float(
                        F.cosine_similarity(anchor, wrong_embedding, dim=0)
                    )
                    rows.append(
                        {
                            "model_seed": model_seed,
                            "phase": phase,
                            "cipher_key": cipher,
                            "scope": "holdout" if cipher in holdout_ciphers else "train",
                            "corruption_seed": int(corruption_seed),
                            "control": control,
                            "positive_cosine": positive_cosine,
                            "positive_distance": 1.0 - positive_cosine,
                            "wrong_cosine": wrong_cosine,
                            "wrong_distance": 1.0 - wrong_cosine,
                            "semantic_margin": positive_cosine - wrong_cosine,
                        }
                    )
    return rows


def train_encoder(
    config: Mapping[str, Any],
    structures: Mapping[str, RuntimeSpnStructure],
    *,
    model_seed: int,
) -> tuple[RuntimeSpnTargetCellProgramEncoder, list[dict[str, Any]]]:
    training = config["training"]
    torch.manual_seed(model_seed)
    encoder = build_encoder(config)
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
            wrong_embeddings = [
                encoder.encode_structure(wrong) for wrong in variants.values()
            ]
            wrong_embeddings.append(
                encoder.encode_structure(
                    structure,
                    edge_binding_seed=corruption_seed,
                )
            )
            for wrong in wrong_embeddings:
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
    source_rows: Sequence[Mapping[str, Any]],
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
        control_summaries: dict[str, Any] = {}
        for control in CONTROL_NAMES:
            selected = [row for row in holdout if row["control"] == control]
            if not selected:
                continue
            minimum_margin = min(float(row["semantic_margin"]) for row in selected)
            minimum_gain = min(
                float(row["semantic_margin"])
                - float(
                    initial[
                        (
                            row["cipher_key"],
                            row["corruption_seed"],
                            row["control"],
                        )
                    ]["semantic_margin"]
                )
                for row in selected
            )
            research_checks[f"seed{model_seed}_{control}_margin"] = (
                minimum_margin
                >= float(gates["holdout_correct_minus_wrong_cosine_distance_min"])
            )
            research_checks[f"seed{model_seed}_{control}_gain"] = (
                minimum_gain >= float(gates["holdout_margin_gain_over_initial_min"])
            )
            control_summaries[control] = {
                "minimum_margin": minimum_margin,
                "minimum_gain": minimum_gain,
            }
        source_sbox = min(
            float(row["semantic_margin"])
            for row in source_rows
            if int(row["model_seed"]) == model_seed
            and row["phase"] == "trained"
            and row["scope"] == "holdout"
            and row["control"] == "wrong_sbox"
        )
        candidate_sbox = control_summaries["wrong_sbox"]["minimum_margin"]
        research_checks[f"seed{model_seed}_retains_k1bw_sbox"] = (
            candidate_sbox
            >= source_sbox - float(gates["wrong_sbox_k1bw_anchor_max_drop"])
        )
        summaries[str(model_seed)] = {
            "minimum_relabel_cosine": min_relabel,
            "controls": control_summaries,
            "k1bw_wrong_sbox_anchor": source_sbox,
            "wrong_sbox_minus_k1bw": candidate_sbox - source_sbox,
        }
    failed_research = sorted(name for name, passed in research_checks.items() if not passed)
    if failed_protocol:
        status = "invalid"
        decision = "innovation1_runtime_spn_k1bx0_protocol_invalid"
    elif failed_research:
        status = "hold"
        decision = "innovation1_runtime_spn_k1bx0_target_cell_program_not_ready"
    else:
        status = "pass"
        decision = "innovation1_runtime_spn_k1bx0_target_cell_program_ready"
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
            "Local structure-only target-cell aggregation diagnostic with a "
            "whole-cipher Dialga holdout; not differential-neural AUC, formal "
            "scale, arbitrary-SPN transfer, an attack or SOTA evidence."
        ),
        "remote_scale": "no",
        "next_action": (
            "If pass, preregister K1-BX frozen-structure differential readiness. "
            "If hold, stop learned global structure-vector pretraining and return "
            "to deterministic primitive routing without more width, epochs or scale."
        ),
    }


def run_experiment(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("K1-BX0 output root already contains artifacts")
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.jsonl"
    _append_jsonl(progress_path, {"event": "run_start", "run_id": RUN_ID, "time": time.time()})
    source_rows, source_checks = load_source_authority(
        config,
        project_root=project_root,
    )
    structures, manifest = load_structures(config, project_root=project_root)
    holdout = set(config["training"]["holdout_ciphers"])
    all_rows: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    geometries = []
    state_hashes = []
    for model_seed in config["training"]["seeds"]:
        torch.manual_seed(int(model_seed))
        initial_encoder = build_encoder(config)
        all_rows.extend(
            evaluate_encoder(
                initial_encoder,
                structures,
                config["training"]["corruption_seeds"],
                model_seed=int(model_seed),
                phase="initial",
                holdout_ciphers=holdout,
            )
        )
        encoder, history = train_encoder(
            config,
            structures,
            model_seed=int(model_seed),
        )
        all_rows.extend(
            evaluate_encoder(
                encoder,
                structures,
                config["training"]["corruption_seeds"],
                model_seed=int(model_seed),
                phase="trained",
                holdout_ciphers=holdout,
            )
        )
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
        **source_checks,
        "seven_descriptors_loaded": len(structures) == 7,
        "dialga_whole_cipher_holdout": holdout == {"dialga128"},
        "all_two_transition_windows": all(value.rounds == 2 for value in structures.values()),
        "parameter_geometry_seed_stable": all(value == geometries[0] for value in geometries),
        "cipher_identity_absent": initial_encoder.uses_cipher_identity is False
        and initial_encoder.uses_cipher_name is False,
        "target_cell_binding_before_pooling": (
            initial_encoder.aggregates_edges_at_actual_target_cell
            and initial_encoder.pools_only_after_cell_transition
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
    gate = adjudicate(
        config,
        all_rows,
        source_rows,
        protocol_checks=protocol_checks,
    )
    preflight = {
        "run_id": RUN_ID,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": _file_sha256(CONFIG_PATH),
        "device": config["training"]["device"],
        "cuda_available": torch.cuda.is_available(),
        "source_checks": source_checks,
        "structure_manifest": manifest,
    }
    geometry = {
        "parameter_count": sum(value.numel() for value in initial_encoder.parameters()),
        "parameter_geometry": geometries[0],
        "state_sha256s": state_hashes,
        "cipher_name_input": False,
        "cipher_id_input": False,
        "target_cell_aggregation_before_pooling": True,
    }
    validation = {
        "status": "pass" if not gate["failed_protocol_checks"] else "fail",
        "result_rows": len(all_rows),
        "history_rows": len(all_history),
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


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_history_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "model_seed",
        "epoch",
        "loss",
        "positive_distance",
        "negative_distance",
        "corruption_seed",
    )
    lines = [",".join(columns)]
    lines.extend(",".join(str(row[column]) for column in columns) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "CONFIG_PATH",
    "CONTROL_NAMES",
    "ROOT",
    "RUN_ID",
    "adjudicate",
    "build_encoder",
    "evaluate_encoder",
    "load_and_validate_config",
    "load_source_authority",
    "run_experiment",
    "train_encoder",
]
