from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    differential_dataset_sha256,
    file_sha256,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1_readiness import (
    ANCHOR_MODEL,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    ANCHOR_CONDITION,
    build_anchor_model,
    checkpoint_map,
    evaluation_map,
    load_bound_datasets,
    load_bound_state,
    result_map,
    task_map_for_model,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1i import (
    CANDIDATE_MODEL,
    build_k1i_control,
    candidate_task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1j import (
    EXPECTED_CHECKPOINT_DIGESTS,
    EXPECTED_INPUT_ROWS,
    EXPECTED_POOL_ROWS,
    EXPECTED_SOURCE_DECISION,
    EXPECTED_SOURCE_DIGESTS,
    INPUT_CONDITIONS,
    REPLAY_TOLERANCE,
    RUN_ID,
    adjudicate_k1j,
    coordinate_permutation,
    permutation_checks,
    permutation_sha256,
    score_input_position_controls,
    score_pool_interventions,
)
from blockcipher_nd.training.metrics import binary_auc


EXPECTED_BATCH_SIZE = 64


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the zero-training K1-J Dialga position/cell interaction "
            "attribution audit on frozen K1-I and Runtime-E4 checkpoints."
        )
    )
    parser.add_argument("--k1i-root", required=True, type=Path)
    parser.add_argument("--k1-root", required=True, type=Path)
    parser.add_argument("--k1i-plan", required=True, type=Path)
    parser.add_argument("--k1-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--batch-size", default=EXPECTED_BATCH_SIZE, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-J run_id must remain frozen as {RUN_ID}")
    if args.batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError(f"K1-J batch size must remain {EXPECTED_BATCH_SIZE}")
    require_fresh_output_root(args.output_root)

    k1i_gate_path = args.k1i_root / "gate.json"
    k1i_checkpoint_manifest_path = args.k1i_root / "checkpoint_manifest.json"
    dataset_manifest_path = args.k1i_root / "dataset_manifest.jsonl"
    runtime_checkpoint_manifest_path = args.k1_root / "checkpoint_manifest.json"
    source_digests = {
        "k1i_gate": file_sha256(k1i_gate_path),
        "k1i_checkpoint_manifest": file_sha256(k1i_checkpoint_manifest_path),
        "k1i_dataset_manifest": file_sha256(dataset_manifest_path),
        "runtime_e4_checkpoint_manifest": file_sha256(runtime_checkpoint_manifest_path),
    }
    k1i_gate = read_json(k1i_gate_path)
    k1i_manifest = read_json(k1i_checkpoint_manifest_path)
    runtime_manifest = read_json(runtime_checkpoint_manifest_path)
    dataset_manifest = read_jsonl(dataset_manifest_path)
    source_controls = evaluation_map(read_jsonl(args.k1i_root / "controls.jsonl"))
    k1i_training_rows = read_jsonl(args.k1i_root / "results.jsonl")
    runtime_training_rows = read_jsonl(args.k1_root / "results.jsonl")
    k1i_tasks = candidate_task_map(read_tasks(args.k1i_plan))
    runtime_tasks = task_map_for_model(read_tasks(args.k1_plan), ANCHOR_MODEL)
    k1i_results = result_map(k1i_training_rows, CANDIDATE_MODEL)
    runtime_results = result_map(runtime_training_rows, ANCHOR_MODEL)
    k1i_checkpoints = checkpoint_map(k1i_manifest, model=CANDIDATE_MODEL)
    runtime_checkpoints = checkpoint_map(runtime_manifest, model=ANCHOR_MODEL)
    datasets = {
        key: dataset
        for key, dataset in load_bound_datasets(dataset_manifest).items()
        if key[0] == "dialga128"
    }
    expected_dataset_keys = {
        ("dialga128", seed, split)
        for seed in (0, 1)
        for split in ("train_seen", "same_key_fresh", "cross_key_validation")
    }
    if set(datasets) != expected_dataset_keys:
        raise ValueError("K1-J requires exactly six Dialga source caches")

    checkpoint_digests: dict[tuple[str, int], str] = {}
    loaded_states: dict[tuple[str, int], Mapping[str, Any]] = {}
    checkpoint_paths: dict[tuple[str, int], Path] = {}
    for seed in (0, 1):
        key = ("dialga128", seed)
        for role, source_rows, manifests in (
            ("k1i_exact", k1i_results, k1i_checkpoints),
            ("runtime_e4", runtime_results, runtime_checkpoints),
        ):
            path = Path(str(source_rows[key]["training"]["checkpoint_output"]))
            state, digest = load_bound_state(path, manifests[key])
            checkpoint_paths[(role, seed)] = path
            loaded_states[(role, seed)] = state
            checkpoint_digests[(role, seed)] = digest

    source_checks: dict[str, bool] = {
        "k1i_source_gate_exact": (
            k1i_gate.get("run_id")
            == "i1_uknit_family_ctspn_gf2_boolean_view_k1i_2048_seed0_seed1_20260728"
            and k1i_gate.get("status") == "hold"
            and k1i_gate.get("decision") == EXPECTED_SOURCE_DECISION
            and bool(k1i_gate.get("protocol_checks"))
            and all(k1i_gate.get("protocol_checks", {}).values())
        ),
        "source_artifact_digests_exact": source_digests == EXPECTED_SOURCE_DIGESTS,
        "four_checkpoint_digests_exact": (
            checkpoint_digests == EXPECTED_CHECKPOINT_DIGESTS
        ),
        "six_dialga_caches_digest_bound": (
            set(datasets) == expected_dataset_keys
            and all(
                differential_dataset_sha256(dataset)
                == next(
                    row["dataset_sha256"]
                    for row in dataset_manifest
                    if row["cipher_key"] == "dialga128"
                    and int(row["seed"]) == seed
                    and row["split"] == split
                )
                for _, seed, split in expected_dataset_keys
                for dataset in [datasets[("dialga128", seed, split)]]
            )
        ),
    }
    structure = build_k1i_control(
        task=k1i_tasks[("dialga128", 0)],
        condition="exact_ordered",
        input_bits=1024,
    ).runtime_structure
    source_checks.update(permutation_checks(structure))
    if not all(source_checks.values()):
        raise ValueError(f"K1-J frozen source binding failed: {source_checks}")

    args.output_root.mkdir(parents=True)
    write_json(
        args.output_root / "preflight.json",
        {
            "run_id": RUN_ID,
            "status": "pass",
            "execution_authorized": True,
            "training_rows": 0,
            "optimizer_steps": 0,
            "source_digests": source_digests,
            "checkpoint_digests": {
                f"{role}_seed{seed}": digest
                for (role, seed), digest in checkpoint_digests.items()
            },
            "source_checks": source_checks,
            "k1i_root": str(args.k1i_root),
            "k1_root": str(args.k1_root),
            "k1i_plan": str(args.k1i_plan),
            "k1_plan": str(args.k1_plan),
        },
    )
    progress(args.output_root / "progress.jsonl", "k1j_audit_start")

    pool_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    replay_deltas: list[float] = []
    manual_forward_deltas: list[float] = []
    input_permutation_pair_checks: list[bool] = []
    audited_model_state_digests: dict[tuple[str, int], str] = {}
    initial_state_digests = {
        key: tensor_mapping_sha256(state) for key, state in loaded_states.items()
    }

    for seed in (0, 1):
        key = ("dialga128", seed)
        k1i_model = build_k1i_control(
            task=k1i_tasks[key],
            condition="exact_ordered",
            input_bits=1024,
        )
        k1i_model.load_state_dict(loaded_states[("k1i_exact", seed)], strict=True)
        no_topology_model = build_k1i_control(
            task=k1i_tasks[key],
            condition="no_topology",
            input_bits=1024,
        )
        no_topology_model.load_state_dict(
            loaded_states[("k1i_exact", seed)], strict=True
        )
        runtime_model = build_anchor_model(runtime_tasks[key], input_bits=1024)
        runtime_model.load_state_dict(loaded_states[("runtime_e4", seed)], strict=True)
        input_permutation_pair_checks.extend(
            permutation_sha256(
                coordinate_permutation(k1i_model.runtime_structure, condition)
            )
            == permutation_sha256(
                coordinate_permutation(runtime_model.runtime_structure, condition)
            )
            for condition in INPUT_CONDITIONS
        )

        for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
            dataset = datasets[("dialga128", seed, split)]
            features = dataset.features
            labels = np.asarray(dataset.labels, dtype=np.float32)
            progress(
                args.output_root / "progress.jsonl",
                "k1j_split_start",
                seed=seed,
                split=split,
                rows=int(features.shape[0]),
            )
            current_pool_rows, pool_probabilities = score_pool_interventions(
                model=k1i_model,
                no_topology_model=no_topology_model,
                features=features,
                labels=labels,
                seed=seed,
                split=split,
                batch_size=args.batch_size,
            )
            current_input_rows, input_probabilities = score_input_position_controls(
                models={"k1i_exact": k1i_model, "runtime_e4": runtime_model},
                features=features,
                labels=labels,
                seed=seed,
                split=split,
                batch_size=args.batch_size,
            )
            dataset_sha = differential_dataset_sha256(dataset)
            for row in current_pool_rows:
                row.update(
                    {
                        "dataset_sha256": dataset_sha,
                        "checkpoint_path": str(checkpoint_paths[("k1i_exact", seed)]),
                        "checkpoint_sha256": checkpoint_digests[("k1i_exact", seed)],
                        "state_dict_sha256": initial_state_digests[("k1i_exact", seed)],
                    }
                )
            for row in current_input_rows:
                role = str(row["model_role"])
                row.update(
                    {
                        "dataset_sha256": dataset_sha,
                        "checkpoint_path": str(checkpoint_paths[(role, seed)]),
                        "checkpoint_sha256": checkpoint_digests[(role, seed)],
                        "state_dict_sha256": initial_state_digests[(role, seed)],
                    }
                )
            pool_rows.extend(current_pool_rows)
            input_rows.extend(current_input_rows)

            source_native = float(
                source_controls[("dialga128", seed, split, "exact_ordered")]["auc"]
            )
            source_none = float(
                source_controls[("dialga128", seed, split, "no_topology")]["auc"]
            )
            source_anchor = float(
                source_controls[("dialga128", seed, split, ANCHOR_CONDITION)]["auc"]
            )
            replay_deltas.extend(
                (
                    abs(
                        binary_auc(labels, pool_probabilities["native"]) - source_native
                    ),
                    abs(
                        binary_auc(labels, pool_probabilities["no_topology"])
                        - source_none
                    ),
                    abs(
                        binary_auc(
                            labels,
                            input_probabilities[("k1i_exact", "native_input")],
                        )
                        - source_native
                    ),
                    abs(
                        binary_auc(
                            labels,
                            input_probabilities[("runtime_e4", "native_input")],
                        )
                        - source_anchor
                    ),
                )
            )
            manual_forward_deltas.append(
                float(
                    np.max(
                        np.abs(
                            pool_probabilities["native"]
                            - input_probabilities[("k1i_exact", "native_input")]
                        )
                    )
                )
            )
            progress(
                args.output_root / "progress.jsonl",
                "k1j_split_done",
                seed=seed,
                split=split,
                pool_rows=len(current_pool_rows),
                input_rows=len(current_input_rows),
            )
        audited_model_state_digests[("k1i_exact", seed)] = tensor_mapping_sha256(
            k1i_model.state_dict()
        )
        audited_model_state_digests[("runtime_e4", seed)] = tensor_mapping_sha256(
            runtime_model.state_dict()
        )
        if (
            tensor_mapping_sha256(no_topology_model.state_dict())
            != initial_state_digests[("k1i_exact", seed)]
        ):
            raise ValueError("K1-J no-topology strict load changed candidate state")
    source_checks.update(
        {
            "source_auc_replay_exact": max(replay_deltas) <= REPLAY_TOLERANCE,
            "manual_pool_forward_replay_exact": (
                max(manual_forward_deltas) <= REPLAY_TOLERANCE
            ),
            "same_input_permutations_for_both_models": all(
                input_permutation_pair_checks
            ),
            "checkpoint_states_unchanged": (
                audited_model_state_digests == initial_state_digests
            ),
            "no_new_dataset_rows_or_optimizer_steps": True,
        }
    )
    write_jsonl(args.output_root / "pool_attribution.jsonl", pool_rows)
    write_jsonl(args.output_root / "input_position_controls.jsonl", input_rows)
    write_csv(args.output_root / "attribution.csv", pool_rows, input_rows)
    gate = adjudicate_k1j(
        pool_rows=pool_rows,
        input_rows=input_rows,
        source_checks=source_checks,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "errors": gate["failed_protocol_checks"],
        "pool_rows": len(pool_rows),
        "expected_pool_rows": EXPECTED_POOL_ROWS,
        "input_rows": len(input_rows),
        "expected_input_rows": EXPECTED_INPUT_ROWS,
        "max_source_auc_replay_delta": max(replay_deltas),
        "max_manual_forward_probability_delta": max(manual_forward_deltas),
        "training_rows": 0,
        "optimizer_steps": 0,
    }
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "research_checks": gate["research_checks"],
        "fresh_results": gate["fresh_results"],
        "input_sensitivity": gate["input_sensitivity"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    write_json(args.output_root / "gate.json", gate)
    write_json(args.output_root / "validation.json", validation)
    write_json(args.output_root / "summary.json", summary)
    progress(
        args.output_root / "progress.jsonl",
        "k1j_gate_done",
        status=gate["status"],
        decision=gate["decision"],
        pool_rows=len(pool_rows),
        input_rows=len(input_rows),
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "invalid" else 0


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def require_fresh_output_root(path: Path) -> None:
    if path.exists():
        raise ValueError(f"K1-J output root already exists: {path}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSONL objects: {path}")
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    pool_rows: Sequence[Mapping[str, Any]],
    input_rows: Sequence[Mapping[str, Any]],
) -> None:
    rows = [{"audit_panel": "pool_attribution", **dict(row)} for row in pool_rows] + [
        {"audit_panel": "input_position", **dict(row)} for row in input_rows
    ]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def progress(path: Path, event: str, **payload: Any) -> None:
    row = {"time": time.time(), "run_id": RUN_ID, "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
