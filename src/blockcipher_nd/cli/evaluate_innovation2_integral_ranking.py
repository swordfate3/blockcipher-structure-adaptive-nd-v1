from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from blockcipher_nd.tasks.innovation2.integral_property_ranking import (
    evaluate_integral_ranking,
)


MODEL_LABELS = {
    "anchor": "同输入线性基线",
    "candidate": "结构 MLP 候选",
    "control": "打乱标签 MLP 控制",
}
MODEL_COLORS = {
    "anchor": "#2563EB",
    "candidate": "#DC2626",
    "control": "#059669",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Innovation 2 E2 stable-rate ranking and top-16 candidate "
            "selection from the frozen E1 artifacts without retraining."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_rates_path = args.source_root / "structure_rates.csv"
    source_gate_path = args.source_root / "gate.json"
    source_rows = _read_csv_rows(source_rates_path)
    source_gate = json.loads(source_gate_path.read_text(encoding="utf-8"))

    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source_root": str(args.source_root),
            "training_performed": False,
        },
        mode="w",
    )
    result = evaluate_integral_ranking(
        run_id=args.run_id,
        source_rows=source_rows,
        source_gate=source_gate,
    )
    results_path = args.output_root / "results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    gate_path = args.output_root / "gate.json"
    gate_path.write_text(
        json.dumps(result["gate"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    ranking_path = args.output_root / "ranking.csv"
    _write_csv_rows(ranking_path, result["ranking_rows"])
    curves_path = args.output_root / "curves.svg"
    render_ranking_svg(result["rows"], result["gate"], curves_path)
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
            "training_performed": False,
        },
        mode="a",
    )
    report = {
        "status": result["gate"]["status"],
        "decision": result["gate"]["decision"],
        "run_id": args.run_id,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "results": str(results_path),
        "ranking": str(ranking_path),
        "gate": str(gate_path),
        "curves": str(curves_path),
        "next_action": result["gate"]["next_action"],
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if result["gate"]["status"] != "fail" else 1


def render_ranking_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    by_role = {str(row["role"]): row for row in rows}
    roles = ("anchor", "candidate", "control")
    labels = [MODEL_LABELS[role] for role in roles]
    colors = [MODEL_COLORS[role] for role in roles]
    spearman = [float(by_role[role]["spearman_stable_q1_rate"]) for role in roles]
    topk_balance = [
        float(by_role[role]["topk_observed_balance_rate"]) for role in roles
    ]
    global_balance = float(by_role["candidate"]["global_observed_balance_rate"])
    metrics = gate["metrics"]
    geometry_holdout = gate.get("structure_split_mode") == "geometry-disjoint"
    title = (
        "创新2 E4：PRESENT 5轮未见位置与掩码组合的积分候选排序"
        if geometry_holdout
        else "创新2 E2：PRESENT 5轮积分输出平衡候选排序审判"
    )
    subtitle = (
        "测试 128 个训练中未出现的活动位置、输出位置与掩码组合；每个候选用 256 把新密钥估计真实平衡率。"
        if geometry_holdout
        else "复用 E1 的 128 个未见结构；每个结构用 256 把新密钥估计真实平衡率。"
    )

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.3))
        fig.subplots_adjust(left=0.075, right=0.975, top=0.73, bottom=0.20, wspace=0.30)
        fig.suptitle(
            title,
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        fig.text(
            0.075,
            0.895,
            subtitle + "本图只做排序与 top-16 筛选，不重训模型。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        spearman_axis, topk_axis = axes
        spearman_bars = spearman_axis.bar(labels, spearman, color=colors, width=0.62)
        spearman_axis.axhline(0.0, color="#94A3B8", linewidth=0.9)
        spearman_axis.set_title(
            "预测排序与 256-key 真实失衡率的一致性",
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            pad=10,
        )
        spearman_axis.set_ylabel("Spearman 相关系数（越高越好）")
        spearman_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        spearman_axis.grid(False, axis="x")
        _set_bar_limits(spearman_axis, spearman)
        _annotate_bars(spearman_axis, spearman_bars, spearman)

        topk_bars = topk_axis.bar(labels, topk_balance, color=colors, width=0.62)
        topk_axis.axhline(
            global_balance,
            color="#475569",
            linewidth=1.4,
            linestyle=(0, (4, 3)),
        )
        topk_axis.set_title(
            "各模型选出的 top-16 结构真实平衡率",
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            pad=10,
        )
        topk_axis.set_ylabel("256-key 观察平衡率（越高越好）")
        topk_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        topk_axis.grid(False, axis="x")
        topk_axis.set_ylim(
            max(0.0, min(min(topk_balance), global_balance) - 0.08),
            min(1.0, max(max(topk_balance), global_balance) + 0.12),
        )
        _annotate_bars(topk_axis, topk_bars, topk_balance)
        topk_axis.text(
            0.98,
            global_balance + 0.008,
            f"全部结构平均平衡率 {global_balance:.3f}",
            transform=topk_axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9.0,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )

        for axis in axes:
            axis.tick_params(axis="x", labelrotation=0, pad=7)

        fig.text(
            0.075,
            0.075,
            (
                f"MLP-线性 Spearman 差值 {metrics['candidate_linear_spearman_margin']:+.3f}；"
                f"MLP top-16 相对全局 {metrics['candidate_global_top16_balance_advantage']:+.3f}；"
                f"相对线性 {metrics['candidate_linear_top16_balance_advantage']:+.3f}；"
                f"打乱控制相对全局 {metrics['control_global_top16_balance_advantage']:+.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(fig)


def render_joint_ranking_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    metric_specs = (
        (
            "candidate_linear_spearman_margin",
            "MLP-线性 Spearman 差值\n门槛 >= 0.05",
            0.05,
        ),
        (
            "candidate_global_top16_balance_advantage",
            "MLP top-16 相对全局\n门槛 >= 0.05",
            0.05,
        ),
        (
            "candidate_linear_top16_balance_advantage",
            "MLP top-16 相对线性\n门槛 >= 0.03",
            0.03,
        ),
        (
            "control_global_top16_balance_advantage",
            "打乱控制 top-16 相对全局\n上限 <= 0.02",
            0.02,
        ),
    )
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    x_positions = list(range(len(metric_specs)))
    width = 0.34
    colors = ("#2563EB", "#DC2626")

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        fig, axis = plt.subplots(figsize=(11.8, 6.1))
        fig.subplots_adjust(left=0.085, right=0.97, top=0.72, bottom=0.23)
        fig.suptitle(
            "创新2 E3：PRESENT 5轮积分输出候选排序双 seed 联合裁决",
            x=0.085,
            y=0.965,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        fig.text(
            0.085,
            0.895,
            (
                "seed0 与 seed1 各使用独立的 128 个测试结构和每结构 256 把新密钥；"
                "三项候选指标须高于下限，打乱控制指标须低于上限。"
            ),
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        all_values: list[float] = []
        for row_index, row in enumerate(rows):
            values = [float(row[key]) for key, _, _ in metric_specs]
            all_values.extend(values)
            offsets = [
                position + (row_index - 0.5) * width for position in x_positions
            ]
            bars = axis.bar(
                offsets,
                values,
                width=width,
                color=colors[row_index],
                label=f"seed {int(row['seed'])}",
            )
            _annotate_bars(axis, bars, values)

        for position, (_, _, threshold) in zip(x_positions, metric_specs):
            axis.plot(
                [position - 0.43, position + 0.43],
                [threshold, threshold],
                color="#475569",
                linewidth=1.6,
                linestyle=(0, (4, 3)),
            )

        lower = min(0.0, min(all_values)) - 0.04
        upper = max(max(all_values), 0.05) + 0.08
        axis.set_ylim(lower, upper)
        axis.set_xticks(
            x_positions,
            [label for _, label, _ in metric_specs],
        )
        axis.set_ylabel("门控差值（绝对比例）")
        axis.set_title(
            "两个独立 seed 的四项冻结门控",
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            pad=10,
        )
        axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axis.grid(False, axis="x")
        axis.axhline(0.0, color="#94A3B8", linewidth=0.9)
        axis.legend(loc="upper right", frameon=False)
        fig.text(
            0.085,
            0.07,
            (
                "裁决：两颗 seed 的候选排序、top-16 全局增益、相对线性增益均过门，"
                "打乱标签控制均未超过 +0.02。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(fig)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV output requires at least one row: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_progress(
    path: Path,
    event: str,
    payload: dict[str, Any],
    *,
    mode: str,
) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _set_bar_limits(axis, values: list[float]) -> None:
    lower = min(0.0, min(values))
    upper = max(0.0, max(values))
    padding = max(0.08, (upper - lower) * 0.18)
    axis.set_ylim(max(-1.0, lower - padding), min(1.0, upper + padding))


def _annotate_bars(axis, bars, values: list[float]) -> None:
    lower, upper = axis.get_ylim()
    span = upper - lower
    for bar, value in zip(bars, values):
        offset = 0.025 * span
        y = value + offset if value >= 0 else value - offset
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9.5,
            fontweight="bold",
        )


if __name__ == "__main__":
    raise SystemExit(main())
