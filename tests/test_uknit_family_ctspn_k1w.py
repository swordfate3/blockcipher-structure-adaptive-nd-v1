from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.run_uknit_family_ctspn_k1w import (
    prepare_bound_cache_links,
)
from blockcipher_nd.cli.plot_uknit_family_ctspn_k1w import render_k1w_svg
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    build_k1t_control,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1w import (
    ANCHOR_AUCS,
    CONTROL_MODELS,
    EXPECTED_KEYS,
    EXPECTED_PARAMETER_COUNT,
    adjudicate,
    build_k1w_control,
    build_readiness,
    candidate_protocol_frozen,
    fold_position_histogram_state,
    read_tasks,
    structural_readiness,
    task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_compact_invariant_k1w_2048_seed_panel.csv"
)


def test_k1w_plan_freezes_eight_compact_rows() -> None:
    tasks = read_tasks(PLAN)
    mapped = task_map(tasks)

    assert len(tasks) == 8
    assert set(mapped) == EXPECTED_KEYS
    assert candidate_protocol_frozen(tasks)
    assert {task["pairs_per_sample"] for task in tasks} == {4}
    assert {task["samples_per_class"] for task in tasks} == {2048}


def test_k1w_models_share_compact_geometry_across_cell_counts() -> None:
    mapped = task_map(read_tasks(PLAN))
    models = {}
    for cipher, seed, input_bits in (("uknit64", 3, 512), ("dialga128", 0, 1024)):
        for condition in CONTROL_MODELS:
            models[(cipher, condition)] = build_k1w_control(
                task=mapped[(cipher, seed, condition)],
                condition=condition,
                input_bits=input_bits,
            )
    geometries = {
        tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
        for model in models.values()
    }

    assert len(geometries) == 1
    assert {
        model_metadata(model)["trainable_parameter_count"] for model in models.values()
    } == {EXPECTED_PARAMETER_COUNT}
    assert models[("uknit64", "compact_exact")](torch.zeros(3, 512)).shape == (
        3,
        1,
    )
    assert models[("dialga128", "compact_exact")](torch.zeros(3, 1024)).shape == (
        3,
        1,
    )


def test_k1w_fold_is_algebraically_equivalent_in_float64() -> None:
    task = task_map(read_tasks(PLAN))[("uknit64", 3, "compact_exact")]
    old = build_k1t_control(
        task=task,
        condition="invariant_histogram_residual",
        input_bits=512,
    ).double()
    compact = build_k1w_control(
        task=task,
        condition="compact_exact",
        input_bits=512,
    ).double()
    fold_position_histogram_state(old.state_dict(), compact)
    fixture = torch.as_tensor(
        np.random.default_rng(20260728).integers(
            0,
            2,
            size=(11, 512),
            dtype=np.uint8,
        ),
        dtype=torch.float64,
    )

    old.eval()
    compact.eval()
    with torch.no_grad():
        error = float((old(fixture) - compact(fixture)).abs().max())
    assert error <= 1e-10


def test_k1w_structural_readiness_covers_both_cell_counts() -> None:
    readiness = structural_readiness(read_tasks(PLAN))

    assert readiness["status"] == "pass"
    assert all(readiness["checks"].values())
    assert set(readiness["parameter_counts"].values()) == {EXPECTED_PARAMETER_COUNT}
    assert set(readiness["relabel_max_logit_errors"]) == {"uknit64", "dialga128"}


def test_k1w_optimizer_gate_requires_fold_replay() -> None:
    checks = {"source": True}
    structure = {"status": "pass", "checks": {"structure": True}}
    fold = {
        "status": "pass",
        "seed_results": {
            str(seed): {
                "anchor_auc_replayed": True,
                "logits_equivalent": True,
                "metrics_equivalent": True,
            }
            for seed in (3, 4)
        },
    }
    readiness = build_readiness(
        tasks=read_tasks(PLAN),
        cache_rows=[],
        bindings=checks,
        fold_replay=fold,
        structure=structure,
    )

    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True

    failed_fold = deepcopy(fold)
    failed_fold["seed_results"]["4"]["logits_equivalent"] = False
    held = build_readiness(
        tasks=read_tasks(PLAN),
        cache_rows=[],
        bindings=checks,
        fold_replay=failed_fold,
        structure=structure,
    )
    assert held["status"] == "fail"
    assert held["optimizer_step_authorized"] is False


def test_k1w_gate_requires_each_cipher_seed_to_retain_anchor(tmp_path: Path) -> None:
    rows = synthetic_results(tmp_path)
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_rows=synthetic_cache_reuses(),
        readiness=synthetic_readiness(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("compact_invariant_supported")
    assert all(gate["research_checks"].values())

    failed = deepcopy(rows)
    for row in failed:
        if (
            row["cipher_key"] == "dialga128"
            and row["seed"] == 1
            and row["model"] == CONTROL_MODELS["compact_exact"]
        ):
            row["metrics"]["auc"] = ANCHOR_AUCS[("dialga128", 1)] - 0.006
    held = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=failed,
        progress_rows=synthetic_cache_reuses(),
        readiness=synthetic_readiness(),
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("dialga_retention_failed")


def test_k1w_cache_links_bind_existing_sources(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"

    prepare_bound_cache_links(cache_root)

    assert (cache_root / "uknit64").is_symlink()
    assert (cache_root / "dialga128").is_symlink()
    assert (cache_root / "uknit64").resolve().is_dir()
    assert (cache_root / "dialga128").resolve().is_dir()


def test_k1w_plot_explains_hold_and_per_cipher_gates(tmp_path: Path) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_results(tmp_path),
        progress_rows=synthetic_cache_reuses(),
        readiness=synthetic_readiness(),
    )
    output = tmp_path / "curves.svg"

    report = render_k1w_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "紧凑的不变直方图网络" in svg
    assert "uKNIT r5" in svg
    assert "Dialga r4" in svg
    assert "正确结构 - 历史锚点" in svg
    assert "正确结构 - 错误 S盒" in svg


def synthetic_readiness() -> dict[str, object]:
    return {
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"ready": True},
    }


def synthetic_results(tmp_path: Path) -> list[dict[str, object]]:
    rows = []
    for cipher, seed in (
        ("uknit64", 3),
        ("uknit64", 4),
        ("dialga128", 0),
        ("dialga128", 1),
    ):
        for condition, model in CONTROL_MODELS.items():
            checkpoint = tmp_path / f"{cipher}_{seed}_{condition}.pt"
            checkpoint.write_bytes(b"checkpoint")
            anchor = ANCHOR_AUCS[(cipher, seed)]
            auc = anchor + 0.001
            if condition == "compact_wrong_sbox" and cipher == "uknit64":
                auc = 0.50
            rows.append(
                {
                    "cipher_key": cipher,
                    "model": model,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "pairs_per_sample": 4,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                    "metrics": {"auc": auc},
                    "training": {
                        "input_bits": 512 if cipher == "uknit64" else 1024,
                        "train_rows": 4096,
                        "validation_rows": 2048,
                        "epochs": 10,
                        "epochs_ran": 10,
                        "selected_checkpoint": "best",
                        "checkpoint_output": str(checkpoint),
                    },
                }
            )
    return rows


def synthetic_cache_reuses() -> list[dict[str, object]]:
    return [{"event": "cache_reuse", "index": index} for index in range(16)]
