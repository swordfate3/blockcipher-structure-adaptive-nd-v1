from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.run_uknit_r5_published_architecture_baselines_k1bz import (
    link_k1bs_cache,
)
from blockcipher_nd.cli.plot_uknit_r5_published_architecture_baselines_k1bz import (
    render_k1bz_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_r5_published_architecture_baselines_k1bz import (
    ARCHITECTURES,
    K1BS_ANCHORS,
    adjudicate,
    build_readiness,
    candidate_protocol_frozen,
    read_tasks,
    task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_uknit_r5_published_architecture_baselines_k1bz_"
    "16pair_2048_seed3_seed4.csv"
)


def test_k1bz_plan_changes_only_published_architecture() -> None:
    tasks = read_tasks(PLAN)
    mapped = task_map(tasks)

    assert len(tasks) == 4
    assert candidate_protocol_frozen(tasks)
    assert set(mapped) == {
        (seed, architecture)
        for seed in (3, 4)
        for architecture in ARCHITECTURES
    }
    assert {task["samples_per_class"] for task in tasks} == {2048}
    assert {task["validation_samples_total"] for task in tasks} == {2048}
    assert {task["pairs_per_sample"] for task in tasks} == {16}


def test_k1bz_readiness_checks_models_views_and_cpu_exception(tmp_path: Path) -> None:
    (tmp_path / "cache/uknit64").mkdir(parents=True)

    readiness = build_readiness(read_tasks(PLAN), k1bs_root=tmp_path)

    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())
    assert readiness["evidence_metrics"]["fixture_shape"] == [4, 2048]
    assert readiness["evidence_metrics"]["liu_case3_view_shape"] == [4, 16, 3, 4, 16]


def test_k1bz_gate_promotes_only_same_adapter_on_both_seeds(tmp_path: Path) -> None:
    promoted = adjudicate(
        read_tasks(PLAN),
        synthetic_results(tmp_path, zhang=(0.57, 0.58), liu=(0.53, 0.54)),
        synthetic_progress(),
        synthetic_readiness(),
    )
    assert promoted["status"] == "pass"
    assert promoted["selected_remote_candidate"] == "zhang_wang_mcnd"
    assert promoted["remote_scale"] == "candidate"

    held = adjudicate(
        read_tasks(PLAN),
        synthetic_results(tmp_path, zhang=(0.57, 0.53), liu=(0.53, 0.57)),
        synthetic_progress(),
        synthetic_readiness(),
    )
    assert held["status"] == "hold"
    assert held["selected_remote_candidate"] is None
    assert held["remote_scale"] == "no"


def test_k1bz_links_existing_cache_without_copying(tmp_path: Path, monkeypatch) -> None:
    import blockcipher_nd.cli.run_uknit_r5_published_architecture_baselines_k1bz as runner

    source_root = tmp_path / "source"
    (source_root / "cache/uknit64").mkdir(parents=True)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(runner, "K1BS_ROOT", source_root)

    link_k1bs_cache(output_root)

    link = output_root / "cache/uknit64"
    assert link.is_symlink()
    assert link.resolve() == (source_root / "cache/uknit64").resolve()


def synthetic_readiness() -> dict[str, object]:
    return {"status": "pass", "optimizer_step_authorized": True}


def synthetic_results(
    tmp_path: Path,
    *,
    zhang: tuple[float, float],
    liu: tuple[float, float],
) -> list[dict[str, object]]:
    aucs = {"zhang_wang_mcnd": zhang, "liu_case3_conv2d": liu}
    rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate((3, 4)):
        for architecture, model in ARCHITECTURES.items():
            checkpoint = tmp_path / f"{seed}_{architecture}.pt"
            checkpoint.write_bytes(b"checkpoint")
            rows.append(
                {
                    "model": model,
                    "rounds": 5,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "pairs_per_sample": 16,
                    "input_difference": 0x0000400000000000,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "metrics": {"auc": aucs[architecture][seed_index]},
                    "training": {
                        "input_bits": 2048,
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


def synthetic_progress() -> list[dict[str, object]]:
    return [
        {"event": "cache_reuse", "seed": seed, "split": split}
        for seed in (3, 4)
        for _ in ARCHITECTURES
        for split in ("train", "validation")
    ]


def test_k1bs_anchor_constants_remain_the_completed_metrics() -> None:
    assert K1BS_ANCHORS[3]["structure_expert"] == 0.902801514
    assert K1BS_ANCHORS[4]["autond_dbitnet"] == 0.526423454


def test_k1bz_plot_states_architecture_and_protocol_boundaries(tmp_path: Path) -> None:
    gate = adjudicate(
        read_tasks(PLAN),
        synthetic_results(tmp_path, zhang=(0.49, 0.50), liu=(0.53, 0.51)),
        synthetic_progress(),
        synthetic_readiness(),
    )
    output = tmp_path / "curves.svg"

    report = render_k1bz_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "公开论文架构补充对比" in svg
    assert "Zhang/Wang MCND 适配" in svg
    assert "Liu Case-3 Conv2D 适配" in svg
    assert "相对 AutoND" in svg
    assert "不是 Zhang/Wang、Liu 或 AutoND 原论文协议复现" in svg
