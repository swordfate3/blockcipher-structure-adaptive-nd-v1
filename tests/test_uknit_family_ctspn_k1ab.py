from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1ab import render_k1ab_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1ab import prepare_bound_cache_link
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1aa import (
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
    VIRTUAL_PARAMETER,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ab import (
    EXPECTED_KEYS,
    adjudicate,
    build_readiness,
    candidate_protocol_frozen,
    read_tasks,
    source_cache_manifest,
    task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_virtual_slot_pair_count_"
    "k1ab_16pair_2048_seed3_seed4.csv"
)


def test_k1ab_plan_freezes_only_sixteen_pair_change() -> None:
    tasks = read_tasks(PLAN)
    assert len(tasks) == 4
    assert set(task_map(tasks)) == EXPECTED_KEYS
    assert candidate_protocol_frozen(tasks)
    assert {task["pairs_per_sample"] for task in tasks} == {16}


def test_k1ab_source_caches_are_complete_and_exact() -> None:
    rows = source_cache_manifest()
    assert len(rows) == 4
    assert all(row["digest_matches"] for row in rows)
    assert {row["rows"] for row in rows} == {2048, 4096}


def test_k1ab_readiness_proves_sources_geometry_and_controls() -> None:
    readiness = build_readiness(read_tasks(PLAN))
    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())


def test_k1ab_gate_requires_each_seed_to_pass_all_pair_gates(tmp_path: Path) -> None:
    rows = synthetic_results(tmp_path)
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_rows=synthetic_cache_reuses(),
        readiness=synthetic_readiness(),
    )
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("16pair_supported")
    assert all(gate["research_checks"].values())

    failed = deepcopy(rows)
    failed[0]["metrics"]["auc"] = 0.57
    held = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=failed,
        progress_rows=synthetic_cache_reuses(),
        readiness=synthetic_readiness(),
    )
    assert held["status"] == "hold"


def test_k1ab_cache_link_reuses_k1v_sixteen_pair_payload(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    prepare_bound_cache_link(root)
    assert (root / "uknit64").is_symlink()
    assert (root / "uknit64").resolve().is_dir()


def test_k1ab_plot_explains_pair_change_and_controls(tmp_path: Path) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_results(tmp_path),
        progress_rows=synthetic_cache_reuses(),
        readiness=synthetic_readiness(),
    )
    output = tmp_path / "curves.svg"
    report = render_k1ab_svg(gate, output)
    svg = output.read_text(encoding="utf-8")
    assert report["panels"] == 2
    assert "从4对提升到16对密文" in svg
    assert "16对正确 S盒" in svg
    assert "增加密文对的净价值" in svg


def synthetic_readiness() -> dict[str, object]:
    return {
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"ready": True},
    }


def synthetic_results(tmp_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (3, 4):
        for condition, model in CONTROL_MODELS.items():
            checkpoint = tmp_path / f"{seed}_{condition}.pt"
            checkpoint.write_bytes(b"checkpoint")
            exact = condition == "virtual_slot_exact"
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "pairs_per_sample": 16,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                    "virtual_projection_slots": 16,
                    "virtual_projection_parameter": VIRTUAL_PARAMETER,
                    "metrics": {"auc": 0.75 if exact else 0.50},
                    "training": {
                        "input_bits": 2048,
                        "train_rows": 4096,
                        "validation_rows": 2048,
                        "epochs": 10,
                        "epochs_ran": 10,
                        "learning_rate": 1e-4,
                        "selected_checkpoint": "best",
                        "checkpoint_output": str(checkpoint),
                    },
                }
            )
    return rows


def synthetic_cache_reuses() -> list[dict[str, object]]:
    return [{"event": "cache_reuse", "index": index} for index in range(8)]
