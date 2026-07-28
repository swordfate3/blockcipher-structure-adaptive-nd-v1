from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt


FRESH_SPLITS = ("same_key_fresh", "cross_key_validation")
ROW_STYLES = {
    (0, "same_key_fresh"): ("#0F766E", "o", "seed0 同 key 新样本"),
    (0, "cross_key_validation"): ("#2563EB", "s", "seed0 跨 key"),
    (1, "same_key_fresh"): ("#D97706", "^", "seed1 同 key 新样本"),
    (1, "cross_key_validation"): ("#BE123C", "D", "seed1 跨 key"),
}
POOL_DISPLAY = (
    ("native", "精确 GF(2)\n原始"),
    ("cross_cell_role_mix", "跨 cell\n重组"),
    ("bit_pool_row_shuffle", "打乱全局\nbit 分支"),
    ("cell_pool_row_shuffle", "打乱 cell\n汇总分支"),
    ("both_pool_row_shuffle", "同时打乱\n两个分支"),
)
EXPLAIN_DISPLAY = POOL_DISPLAY[1:]
INPUT_DISPLAY = (
    ("within_cell_input_role_roll", "cell 内\nbit 角色轮转"),
    ("whole_cell_input_roll", "完整 cell\n位置轮转"),
    ("cross_cell_input_role_mix", "跨 cell\n角色混合"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-J audit chart.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--pool-results", required=True, type=Path)
    parser.add_argument("--input-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = read_json(args.gate)
    pool_rows = read_jsonl(args.pool_results)
    input_rows = read_jsonl(args.input_results)
    render_k1j_svg(gate, pool_rows, input_rows, args.output)
    report = {
        "status": "rendered_pending_visual_qa",
        "run_id": gate.get("run_id"),
        "gate_status": gate.get("status"),
        "decision": gate.get("decision"),
        "output": str(args.output),
        "visual_qa_required": True,
    }
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def render_k1j_svg(
    gate: Mapping[str, Any],
    pool_rows: Sequence[Mapping[str, Any]],
    input_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    pool = {
        (int(row["seed"]), str(row["split"]), str(row["condition"])): row
        for row in pool_rows
    }
    inputs = {
        (
            str(row["model_role"]),
            int(row["seed"]),
            str(row["split"]),
            str(row["condition"]),
        ): row
        for row in input_rows
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16, 10.2))
        figure.subplots_adjust(
            left=0.085,
            right=0.96,
            top=0.75,
            bottom=0.11,
            hspace=0.55,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1 K1-J：Dialga 强信号究竟来自位置、cell 还是联合统计",
            x=0.055,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.055,
            0.905,
            "冻结 K1-I 与 Runtime-E4 检查点；复用原数据，零训练、零优化步骤；每条线都保留两颗 seed 和两个 fresh split。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.055,
            0.848,
            "裁决：跨 cell 重组几乎不伤信号；两个不变池化分支必须联合存在，当前模型仍未识别正确矩阵的位置语义。",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#047857" if gate.get("status") == "pass" else "#B45309",
        )

        _plot_pool_auc(axes[0, 0], pool)
        _plot_explained_fraction(axes[0, 1], pool)
        _plot_input_sensitivity(axes[1, 0], inputs)
        _plot_source_gap(axes[1, 1], pool)

        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linewidth=1.3,
                markersize=6,
                label=label,
            )
            for color, marker, label in ROW_STYLES.values()
        ]
        figure.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.955, 0.965),
            frameon=False,
            ncol=2,
            fontsize=9,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)


def _plot_pool_auc(
    axis: plt.Axes,
    pool: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> None:
    x = list(range(len(POOL_DISPLAY)))
    for key, (color, marker, _label) in ROW_STYLES.items():
        values = [
            float(pool[(*key, condition)]["auc"]) for condition, _ in POOL_DISPLAY
        ]
        axis.plot(x, values, color=color, marker=marker, linewidth=1.25, markersize=5)
    axis.set_xticks(x, [label for _, label in POOL_DISPLAY])
    axis.set_ylim(0.45, 0.99)
    axis.set_ylabel("AUC")
    axis.set_title("冻结 K1-I：逐分支干预后的 fresh AUC", loc="left", fontweight="bold")
    axis.axhline(0.5, color="#9CA3AF", linestyle=(0, (3, 3)), linewidth=1)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _plot_explained_fraction(
    axis: plt.Axes,
    pool: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> None:
    x = list(range(len(EXPLAIN_DISPLAY)))
    for key, (color, marker, _label) in ROW_STYLES.items():
        values = [
            float(pool[(*key, condition)]["explained_fraction"])
            for condition, _ in EXPLAIN_DISPLAY
        ]
        axis.plot(x, values, color=color, marker=marker, linewidth=1.25, markersize=5)
    axis.set_xticks(x, [label for _, label in EXPLAIN_DISPLAY])
    axis.set_ylim(-0.04, 1.08)
    axis.set_ylabel("解释的精确 GF(2)-无拓扑 AUC 差比例")
    axis.set_title("哪种干预能解释原始信号", loc="left", fontweight="bold")
    axis.axhline(0.8, color="#047857", linestyle=(0, (4, 3)), linewidth=1.3)
    axis.text(0.02, 0.82, "通过门槛 80%", color="#047857", fontsize=8.5)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _plot_input_sensitivity(
    axis: plt.Axes,
    inputs: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> None:
    x = list(range(len(INPUT_DISPLAY)))
    role_styles = {
        "k1i_exact": ("#7C3AED", "o", -0.07, "K1-I 精确 GF(2)"),
        "runtime_e4": ("#0F766E", "s", 0.07, "旧 Runtime-E4"),
    }
    for role, (color, marker, shift, label) in role_styles.items():
        for row_index, (seed, split) in enumerate(ROW_STYLES):
            jitter = shift + (row_index - 1.5) * 0.012
            values = [
                float(
                    inputs[(role, seed, split, condition)]["native_minus_condition_auc"]
                )
                for condition, _ in INPUT_DISPLAY
            ]
            axis.plot(
                [value + jitter for value in x],
                values,
                color=color,
                marker=marker,
                linewidth=0.8,
                markersize=4,
                alpha=0.62,
                label=label if row_index == 0 else None,
            )
    axis.set_xticks(x, [label for _, label in INPUT_DISPLAY])
    axis.set_ylim(0.32, 0.50)
    axis.set_ylabel("原始输入 AUC - 位置扰动 AUC")
    axis.set_title(
        "输入坐标被破坏时，两种模型都明显退化", loc="left", fontweight="bold"
    )
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.legend(frameon=False, ncol=2, fontsize=8.5, loc="upper left")


def _plot_source_gap(
    axis: plt.Axes,
    pool: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> None:
    x = list(range(len(ROW_STYLES)))
    labels: list[str] = []
    native: list[float] = []
    no_topology: list[float] = []
    for seed, split in ROW_STYLES:
        row = pool[(seed, split, "native")]
        labels.append(f"s{seed}\n{'同 key' if split == 'same_key_fresh' else '跨 key'}")
        native.append(float(row["native_auc"]))
        no_topology.append(float(row["no_topology_auc"]))
    width = 0.34
    axis.bar(
        [value - width / 2 for value in x],
        native,
        width=width,
        color="#0F766E",
        label="精确 GF(2)",
    )
    axis.bar(
        [value + width / 2 for value in x],
        no_topology,
        width=width,
        color="#64748B",
        label="无拓扑",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0.48, 0.99)
    axis.set_ylabel("AUC（从 0.48 起）")
    axis.set_title("审计所解释的原始强信号差距", loc="left", fontweight="bold")
    axis.axhline(0.5, color="#9CA3AF", linestyle=(0, (3, 3)), linewidth=1)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.legend(frameon=False, ncol=2, fontsize=8.5, loc="upper left")


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


if __name__ == "__main__":
    raise SystemExit(main())
