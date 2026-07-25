from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d1 import (
    adjudicate_runtime_spn_dialga_d1,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d2 import (
    CORRUPTION_SEED,
    adjudicate_same_checkpoint_dialga,
    evaluate_same_checkpoint_dialga,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_counterfactual import (
    file_sha256,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Dialga Runtime-E4 topology use with frozen D1 checkpoints."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_results_path = args.d1_root / "results.jsonl"
    source_gate_path = args.d1_root / "gate.json"
    source_rows = _read_jsonl(source_results_path)
    persisted_source_gate = _read_json(source_gate_path)
    recomputed_source_gate = adjudicate_runtime_spn_dialga_d1(
        run_id=str(persisted_source_gate.get("run_id", "")),
        rows=source_rows,
    )
    if persisted_source_gate != recomputed_source_gate:
        raise ValueError("persisted D1 gate does not match recomputed source evidence")
    if persisted_source_gate.get("status") != "pass":
        raise ValueError("D2 requires a completed passing D1 source")

    descriptor = load_runtime_spn_descriptor(
        "configs/runtime/spn/dialga128.json",
        rounds=2,
        round_start=2,
    )
    correct_structure = descriptor.structure
    corrupted_structure = correct_structure.corrupted(CORRUPTION_SEED)
    source_results_sha256 = file_sha256(source_results_path)
    source_gate_sha256 = file_sha256(source_gate_path)
    result_rows: list[dict[str, Any]] = []
    _write_progress(args.output_root / "progress.jsonl", "run_start", args.run_id)

    for seed in (0, 1):
        source = _source_candidate(source_rows, seed)
        dataset, feature_path, label_path, metadata_path = _load_validation_dataset(
            source,
            seed,
        )
        checkpoint_path = Path(source["training"]["checkpoint_output"])
        result_rows.extend(
            evaluate_same_checkpoint_dialga(
                seed=seed,
                model_options=dict(source["training"]["model_options"]),
                checkpoint_path=checkpoint_path,
                dataset=dataset,
                correct_structure=correct_structure,
                corrupted_structure=corrupted_structure,
                source_auc=float(source["metrics"]["auc"]),
                checkpoint_sha256=file_sha256(checkpoint_path),
                feature_sha256=file_sha256(feature_path),
                label_sha256=file_sha256(label_path),
                metadata_sha256=file_sha256(metadata_path),
                source_results_sha256=source_results_sha256,
                source_gate_sha256=source_gate_sha256,
                descriptor_name=descriptor.name,
                descriptor_path=str(descriptor.path),
                descriptor_sha256=descriptor.sha256,
                source_descriptor_sha256=str(
                    source["runtime_structure_descriptor_sha256"]
                ),
                batch_size=args.batch_size,
                device=args.device,
            )
        )
        _write_progress(
            args.output_root / "progress.jsonl",
            "seed_done",
            args.run_id,
            seed=seed,
        )

    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    gate = adjudicate_same_checkpoint_dialga(run_id=args.run_id, rows=result_rows)
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {
            "d1_results": str(source_results_path),
            "d1_gate": str(source_gate_path),
        },
        "source_gate_recomputed": True,
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "cipher": "Dialga-128",
        "training_performed": False,
        "source_run": str(args.d1_root),
        "gate": gate,
    }
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    render_dialga_d2_svg(gate, args.output_root / "curves.svg")
    _write_progress(
        args.output_root / "progress.jsonl",
        "run_done",
        args.run_id,
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_dialga_d2_svg(gate: dict[str, Any], output_path: Path) -> None:
    conditions = ("correct", "corrupted", "no_topology")
    condition_labels = ("正确拓扑", "损坏拓扑", "无拓扑")
    colors = ("#2563EB", "#D97706", "#0F9D76")
    seed_results = gate["seed_results"]
    aucs = {
        seed: tuple(seed_results[str(seed)][f"{condition}_auc"] for condition in conditions)
        for seed in (0, 1)
    }
    margins = tuple(
        seed_results[str(seed)][field]
        for seed in (0, 1)
        for field in (
            "correct_minus_corrupted_auc",
            "correct_minus_no_topology_auc",
        )
    )
    probability_deltas = tuple(
        seed_results[str(seed)][field]
        for seed in (0, 1)
        for field in (
            "corrupted_probability_delta",
            "no_topology_probability_delta",
        )
    )
    control_labels = (
        "seed0 损坏",
        "seed0 无拓扑",
        "seed1 损坏",
        "seed1 无拓扑",
    )

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 6.7))
        figure.subplots_adjust(
            left=0.055,
            right=0.98,
            top=0.71,
            bottom=0.16,
            wspace=0.32,
        )
        figure.suptitle(
            "创新1 D2：Dialga-128 同一权重运行时拓扑替换审计",
            x=0.055,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.055,
            0.89,
            "冻结两颗 D1 正确拓扑最佳模型及其验证数据；推理时只替换正确、损坏或无拓扑结构，不重新训练。",
            ha="left",
            color="#475569",
        )
        figure.text(
            0.055,
            0.82,
            f"裁决：{_decision_text(gate)}",
            ha="left",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
        )

        x = np.arange(2, dtype=np.float64)
        width = 0.23
        all_auc_values = [value for values in aucs.values() for value in values]
        condition_handles = []
        for index, (label, color) in enumerate(zip(condition_labels, colors, strict=True)):
            values = [aucs[seed][index] for seed in (0, 1)]
            bars = axes[0].bar(
                x + (index - 1) * width,
                values,
                width,
                label=label,
                color=color,
            )
            condition_handles.append(bars)
            axes[0].bar_label(
                bars,
                labels=[f"{value:.4f}" for value in values],
                padding=4,
                fontsize=9,
            )
        axes[0].set_title("同一检查点验证 AUC", loc="left", fontweight="bold")
        axes[0].set_ylabel("AUC")
        axes[0].set_xticks(x, ("seed0", "seed1"))
        axes[0].axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1)
        lower = max(0.0, min(0.5, *all_auc_values) - 0.04)
        upper = min(1.0, max(*all_auc_values) + 0.05)
        axes[0].set_ylim(lower, upper)
        axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        figure.legend(
            condition_handles,
            condition_labels,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.055, 0.785),
            ncol=3,
            columnspacing=1.8,
            handlelength=2.4,
        )

        margin_colors = [
            "#059669" if value >= 0.005 else "#DC2626" for value in margins
        ]
        margin_bars = axes[1].bar(
            np.arange(4),
            margins,
            width=0.62,
            color=margin_colors,
        )
        axes[1].bar_label(
            margin_bars,
            labels=[f"{value:+.6f}" for value in margins],
            padding=5,
            fontsize=9,
        )
        axes[1].axhline(0.0, color="#64748B", linewidth=1)
        axes[1].axhline(
            0.005,
            color="#2563EB",
            linestyle="--",
            linewidth=1.2,
            label="通过门槛 +0.005",
        )
        axes[1].set_title("正确拓扑相对控制的 AUC 差值", loc="left", fontweight="bold")
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_xticks(np.arange(4), control_labels, rotation=18, ha="right")
        axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper left")
        axes[1].margins(y=0.25)

        delta_bars = axes[2].bar(
            np.arange(4),
            probability_deltas,
            width=0.62,
            color=("#D97706", "#0F9D76", "#D97706", "#0F9D76"),
        )
        axes[2].bar_label(
            delta_bars,
            labels=[f"{value:.3e}" for value in probability_deltas],
            padding=5,
            fontsize=9,
        )
        axes[2].axhline(
            1e-6,
            color="#2563EB",
            linestyle="--",
            linewidth=1.2,
            label="变化门槛 1e-6",
        )
        axes[2].set_yscale("log")
        axes[2].set_title("相对正确拓扑的最大预测变化", loc="left", fontweight="bold")
        axes[2].set_ylabel("最大概率差值（对数轴）")
        axes[2].set_xticks(np.arange(4), control_labels, rotation=18, ha="right")
        axes[2].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[2].legend(frameon=False, loc="upper left")
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _decision_text(gate: dict[str, Any]) -> str:
    if gate["status"] == "pass":
        return "两颗 seed 均保持正确拓扑优势，支持模型功能性使用外部 Dialga 拓扑。"
    if gate["status"] == "fail":
        return "来源或审计协议检查失败，当前 AUC 不得用于研究判断。"
    return "结构替换未在两颗 seed 同时保留判别优势，停止扩样并在本地重设计。"


def _source_candidate(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("seed") == seed
        and row.get("model") == "runtime_spn_e4_equivariant_true"
        and row.get("runtime_structure_mode") == "true"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one correct D1 source row for seed {seed}")
    return matches[0]


def _load_validation_dataset(
    source: dict[str, Any],
    seed: int,
) -> tuple[DiskDifferentialDataset, Path, Path, Path]:
    cache_root = Path(source["training"]["dataset_cache_root"])
    validation_root = cache_root / "dialga128" / "r4" / "validation"
    matches = list(validation_root.glob(f"seed-{10000 + seed}_*"))
    if len(matches) != 1:
        raise ValueError(f"expected one Dialga D1 validation cache for seed {seed}")
    cache_dir = matches[0]
    feature_path = cache_dir / "features.npy"
    label_path = cache_dir / "labels.npy"
    metadata_path = cache_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    dataset = DiskDifferentialDataset(
        features=np.load(feature_path, mmap_mode="r"),
        labels=np.load(label_path, mmap_mode="r"),
        metadata=metadata,
        cache_dir=cache_dir,
    )
    return dataset, feature_path, label_path, metadata_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_progress(path: Path, event: str, run_id: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    "run_id": run_id,
                    **payload,
                },
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
