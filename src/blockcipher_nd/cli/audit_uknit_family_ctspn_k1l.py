from __future__ import annotations

import argparse
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
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1h import (
    checkpoint_map,
    evaluation_map,
    expected_task_keys,
    load_bound_datasets,
    load_bound_state,
    result_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import (
    CANDIDATE_MODEL,
    CONTROL_CONDITIONS,
    RUN_ID as K1K_RUN_ID,
    build_k1k_control,
    candidate_task_map,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1l import (
    AUDIT_CONDITIONS,
    EXPECTED_CHECKPOINT_DIGESTS,
    EXPECTED_GRADIENT_ROWS,
    EXPECTED_RESULT_ROWS,
    EXPECTED_SOURCE_DECISION,
    EXPECTED_SOURCE_DIGESTS,
    FULL_TOPOLOGY_CONDITIONS,
    RUN_ID,
    adjudicate_k1l,
    audit_gradient_path,
    collect_residual_path_outputs,
    label_blind_row_permutation,
    residual_metrics,
    shuffled_residual_logits,
)


EXPECTED_BATCH_SIZE = 64


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen K1-K residual gates, edge contributions, slot paths, "
            "and exact-zero gradient starvation without optimizer steps."
        )
    )
    parser.add_argument("--k1k-root", required=True, type=Path)
    parser.add_argument("--k1k-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--batch-size", default=EXPECTED_BATCH_SIZE, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"K1-L run_id must remain frozen as {RUN_ID}")
    if args.batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError(f"K1-L batch size must remain {EXPECTED_BATCH_SIZE}")
    require_fresh_output_root(args.output_root)

    source_paths = {
        "gate": args.k1k_root / "gate.json",
        "checkpoint_manifest": args.k1k_root / "checkpoint_manifest.json",
        "dataset_manifest": args.k1k_root / "dataset_manifest.jsonl",
        "controls": args.k1k_root / "controls.jsonl",
        "validation": args.k1k_root / "validation.json",
    }
    source_digests = {
        name: file_sha256(path) for name, path in source_paths.items()
    }
    gate = read_json(source_paths["gate"])
    validation = read_json(source_paths["validation"])
    checkpoint_manifest = read_json(source_paths["checkpoint_manifest"])
    dataset_manifest = read_jsonl(source_paths["dataset_manifest"])
    source_control_rows = read_jsonl(source_paths["controls"])
    source_controls = evaluation_map(source_control_rows)
    training_rows = read_jsonl(args.k1k_root / "results.jsonl")
    tasks = candidate_task_map(read_tasks(args.k1k_plan))
    training = result_map(training_rows, CANDIDATE_MODEL)
    checkpoints = checkpoint_map(checkpoint_manifest, model=CANDIDATE_MODEL)
    datasets = load_bound_datasets(dataset_manifest)

    expected_datasets = {
        (cipher, seed, split)
        for cipher, seed in expected_task_keys()
        for split in ("train_seen", "same_key_fresh", "cross_key_validation")
    }
    checkpoint_states: dict[tuple[str, int], Mapping[str, Any]] = {}
    checkpoint_digests: dict[tuple[str, int], str] = {}
    checkpoint_paths: dict[tuple[str, int], Path] = {}
    for key in sorted(expected_task_keys()):
        path = Path(str(training[key]["training"]["checkpoint_output"]))
        state, digest = load_bound_state(path, checkpoints[key])
        checkpoint_states[key] = state
        checkpoint_digests[key] = digest
        checkpoint_paths[key] = path

    source_checks = {
        "source_gate_exact": (
            gate.get("run_id") == K1K_RUN_ID
            and gate.get("status") == "hold"
            and gate.get("decision") == EXPECTED_SOURCE_DECISION
            and bool(gate.get("protocol_checks"))
            and all(gate.get("protocol_checks", {}).values())
        ),
        "source_validation_exact": (
            validation.get("run_id") == K1K_RUN_ID
            and validation.get("status") == "pass"
            and not validation.get("errors")
            and validation.get("training_rows") == 4
            and validation.get("evaluation_rows") == 60
        ),
        "source_artifact_digests_exact": source_digests == EXPECTED_SOURCE_DIGESTS,
        "four_checkpoint_digests_exact": (
            checkpoint_digests == EXPECTED_CHECKPOINT_DIGESTS
        ),
        "four_checkpoint_states_bound": (
            set(checkpoint_states) == expected_task_keys()
            and all(
                str(checkpoint_paths[key]) == checkpoints[key].get("path")
                and bool(tensor_mapping_sha256(checkpoint_states[key]))
                for key in expected_task_keys()
            )
        ),
        "twelve_caches_digest_bound": (
            len(dataset_manifest) == 12
            and set(datasets) == expected_datasets
            and all(
                differential_dataset_sha256(datasets[key])
                == next(
                    row["dataset_sha256"]
                    for row in dataset_manifest
                    if (
                        row["cipher_key"],
                        int(row["seed"]),
                        row["split"],
                    )
                    == key
                )
                for key in expected_datasets
            )
        ),
        "source_candidate_controls_complete": (
            len(
                [
                    row
                    for row in source_control_rows
                    if row.get("source_role") == "candidate"
                ]
            )
            == 48
            and all(
                (cipher, seed, split, condition) in source_controls
                for cipher, seed in expected_task_keys()
                for split in (
                    "train_seen",
                    "same_key_fresh",
                    "cross_key_validation",
                )
                for condition in CONTROL_CONDITIONS
            )
        ),
    }
    if not all(source_checks.values()):
        raise ValueError(f"K1-L frozen source binding failed: {source_checks}")

    args.output_root.mkdir(parents=True)
    write_json(
        args.output_root / "preflight.json",
        {
            "run_id": RUN_ID,
            "status": "pass",
            "execution_authorized": True,
            "training_rows": 0,
            "optimizer_steps": 0,
            "source_root": str(args.k1k_root),
            "source_plan": str(args.k1k_plan),
            "source_digests": source_digests,
            "checkpoint_digests": {
                f"{cipher}_seed{seed}": digest
                for (cipher, seed), digest in checkpoint_digests.items()
            },
            "source_checks": source_checks,
        },
    )
    progress(args.output_root / "progress.jsonl", "k1l_audit_start")

    result_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    for cipher, seed in sorted(expected_task_keys()):
        key = (cipher, seed)
        task = tasks[key]
        state = checkpoint_states[key]
        checkpoint_path = checkpoint_paths[key]
        checkpoint_sha = checkpoint_digests[key]
        state_sha = tensor_mapping_sha256(state)
        progress(
            args.output_root / "progress.jsonl",
            "k1l_checkpoint_start",
            cipher_key=cipher,
            seed=seed,
        )
        train_dataset = datasets[(cipher, seed, "train_seen")]
        gradient_model = build_k1k_control(
            task=task,
            condition="exact_ordered",
            input_bits=int(train_dataset.features.shape[1]),
        )
        gradient_model.load_state_dict(state, strict=True)
        for effective_gate in (0.0, 0.05):
            row = audit_gradient_path(
                gradient_model,
                train_dataset,
                effective_gate=effective_gate,
                batch_size=args.batch_size,
            )
            row.update(
                {
                    "run_id": RUN_ID,
                    "cipher_key": cipher,
                    "seed": seed,
                    "rows": args.batch_size,
                    "dataset_sha256": differential_dataset_sha256(train_dataset),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                    "state_dict_sha256": state_sha,
                }
            )
            gradient_rows.append(row)

        for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
            dataset = datasets[(cipher, seed, split)]
            labels = np.asarray(dataset.labels, dtype=np.float32)
            dataset_sha = differential_dataset_sha256(dataset)
            topology_outputs: dict[str, Any] = {}
            topology_models: dict[str, Any] = {}
            for audit_condition, source_condition in FULL_TOPOLOGY_CONDITIONS.items():
                model = build_k1k_control(
                    task=task,
                    condition=source_condition,
                    input_bits=int(dataset.features.shape[1]),
                )
                model.load_state_dict(state, strict=True)
                if tensor_mapping_sha256(model.state_dict()) != state_sha:
                    raise ValueError("K1-L strict load changed checkpoint state")
                outputs = collect_residual_path_outputs(
                    model,
                    dataset,
                    batch_size=args.batch_size,
                )
                metrics = residual_metrics(labels, outputs)
                source_auc = float(
                    source_controls[(cipher, seed, split, source_condition)]["auc"]
                )
                result_rows.append(
                    result_row(
                        cipher=cipher,
                        seed=seed,
                        split=split,
                        condition=audit_condition,
                        metrics=metrics,
                        model=model,
                        outputs=outputs,
                        dataset_sha=dataset_sha,
                        checkpoint_path=checkpoint_path,
                        checkpoint_sha=checkpoint_sha,
                        state_sha=state_sha,
                        source_condition=source_condition,
                        source_auc=source_auc,
                    )
                )
                topology_outputs[audit_condition] = outputs
                topology_models[audit_condition] = model

            native = topology_outputs["native_full"]
            native_model = topology_models["native_full"]
            zero_metrics = residual_metrics(labels, native, logits=native.zero_logits)
            result_rows.append(
                result_row(
                    cipher=cipher,
                    seed=seed,
                    split=split,
                    condition="gate_zero",
                    metrics=zero_metrics,
                    model=native_model,
                    outputs=native,
                    dataset_sha=dataset_sha,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha=checkpoint_sha,
                    state_sha=state_sha,
                )
            )
            for condition, slot_mask in (
                ("slot0_only", (True, False)),
                ("slot1_only", (False, True)),
            ):
                slot_outputs = collect_residual_path_outputs(
                    native_model,
                    dataset,
                    batch_size=args.batch_size,
                    slot_mask=slot_mask,
                )
                result_rows.append(
                    result_row(
                        cipher=cipher,
                        seed=seed,
                        split=split,
                        condition=condition,
                        metrics=residual_metrics(labels, slot_outputs),
                        model=native_model,
                        outputs=slot_outputs,
                        dataset_sha=dataset_sha,
                        checkpoint_path=checkpoint_path,
                        checkpoint_sha=checkpoint_sha,
                        state_sha=state_sha,
                    )
                )
            permutation = label_blind_row_permutation(
                len(labels),
                cipher=cipher,
                seed=seed,
                split=split,
            )
            shuffled_logits = shuffled_residual_logits(
                native_model,
                native,
                permutation,
                batch_size=args.batch_size,
            )
            shuffled_metrics = residual_metrics(
                labels,
                native,
                logits=shuffled_logits,
            )
            source_delta = abs(
                float(
                    next(
                        row["auc"]
                        for row in result_rows
                        if row["cipher_key"] == cipher
                        and row["seed"] == seed
                        and row["split"] == split
                        and row["condition"] == "native_full"
                    )
                )
                - float(zero_metrics["auc"])
            )
            explained = (
                min(
                    1.0,
                    abs(
                        float(
                            next(
                                row["auc"]
                                for row in result_rows
                                if row["cipher_key"] == cipher
                                and row["seed"] == seed
                                and row["split"] == split
                                and row["condition"] == "native_full"
                            )
                        )
                        - float(shuffled_metrics["auc"])
                    )
                    / source_delta,
                )
                if source_delta > 1e-12
                else 0.0
            )
            shuffled_row = result_row(
                cipher=cipher,
                seed=seed,
                split=split,
                condition="residual_row_shuffle",
                metrics=shuffled_metrics,
                model=native_model,
                outputs=native,
                dataset_sha=dataset_sha,
                checkpoint_path=checkpoint_path,
                checkpoint_sha=checkpoint_sha,
                state_sha=state_sha,
            )
            shuffled_row.update(
                {
                    "row_permutation_sha256": tensor_mapping_sha256(
                        {"permutation": permutation}
                    ),
                    "row_permutation_bijective": (
                        sorted(permutation.tolist()) == list(range(len(labels)))
                    ),
                    "row_permutation_nonidentity": not np.array_equal(
                        permutation.numpy(), np.arange(len(labels))
                    ),
                    "explained_fraction": explained,
                }
            )
            result_rows.append(shuffled_row)
        progress(
            args.output_root / "progress.jsonl",
            "k1l_checkpoint_done",
            cipher_key=cipher,
            seed=seed,
        )

    gate_result = adjudicate_k1l(
        result_rows=result_rows,
        gradient_rows=gradient_rows,
        source_checks=source_checks,
    )
    validation_result = {
        "run_id": RUN_ID,
        "status": (
            "pass" if all(gate_result["protocol_checks"].values()) else "fail"
        ),
        "checks": gate_result["protocol_checks"],
        "errors": gate_result["failed_protocol_checks"],
        "training_rows": 0,
        "optimizer_steps": 0,
        "result_rows": len(result_rows),
        "expected_result_rows": EXPECTED_RESULT_ROWS,
        "gradient_rows": len(gradient_rows),
        "expected_gradient_rows": EXPECTED_GRADIENT_ROWS,
    }
    write_jsonl(args.output_root / "results.jsonl", result_rows)
    write_jsonl(args.output_root / "gradient_attribution.jsonl", gradient_rows)
    write_json(args.output_root / "gate.json", gate_result)
    write_json(args.output_root / "validation.json", validation_result)
    progress(
        args.output_root / "progress.jsonl",
        "run_done",
        status=gate_result["status"],
        decision=gate_result["decision"],
        result_rows=len(result_rows),
        gradient_rows=len(gradient_rows),
    )
    print(json.dumps(gate_result, ensure_ascii=False, sort_keys=True))
    return 1 if gate_result["status"] == "invalid" else 0


def result_row(
    *,
    cipher: str,
    seed: int,
    split: str,
    condition: str,
    metrics: Mapping[str, float],
    model: Any,
    outputs: Any,
    dataset_sha: str,
    checkpoint_path: Path,
    checkpoint_sha: str,
    state_sha: str,
    source_condition: str | None = None,
    source_auc: float | None = None,
) -> dict[str, Any]:
    if condition not in AUDIT_CONDITIONS:
        raise ValueError(f"unknown K1-L audit condition: {condition}")
    return {
        "run_id": RUN_ID,
        "cipher_key": cipher,
        "seed": seed,
        "split": split,
        "condition": condition,
        **dict(metrics),
        "raw_gate": float(model.backbone.residual_gate.detach()),
        "effective_gate": outputs.effective_gate,
        "dataset_sha256": dataset_sha,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "state_dict_sha256": state_sha,
        "source_condition": source_condition,
        "source_auc": source_auc,
        "training_performed": False,
        "optimizer_steps": 0,
    }


def read_tasks(path: Path) -> list[dict[str, Any]]:
    return tasks_from_plan(
        path,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def require_fresh_output_root(path: Path) -> None:
    protected = (
        "preflight.json",
        "results.jsonl",
        "gradient_attribution.jsonl",
        "gate.json",
        "progress.jsonl",
    )
    if path.exists() and any((path / name).exists() for name in protected):
        raise ValueError("K1-L output root already contains audit artifacts")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def progress(path: Path, event: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event": event, "time": time.time(), **payload},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
