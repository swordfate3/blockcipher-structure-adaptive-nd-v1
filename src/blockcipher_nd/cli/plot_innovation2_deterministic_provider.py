from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E31 deterministic label-provider contract audit."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_provider_contract_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_provider_contract_svg(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = summary["atm"]["metrics"]
    claasp = summary["claasp"]
    atm = summary["atm"]
    labels = ["结果文件", "序列化basis", "GF(2)秩", "support坐标", "singleton", "线性输出候选"]
    values = [
        metrics["result_files"],
        metrics["unique_serialized_basis_elements"],
        metrics["union_gf2_rank"],
        metrics["support_coordinates"],
        metrics["standard_basis_members"],
        metrics["linear_output_standard_basis_members"],
    ]
    contract_labels = ["当前可执行", "含线性输出候选", "0/1常数明确", "负类语义完备", "论文维数一致"]
    contract = np.asarray(
        [
            [0, 0],
            [1, 1],
            [-1, 0],
            [0, 0],
            [-1, 0],
        ],
        dtype=np.int8,
    )
    contract[0, 0] = 1 if claasp["runtime"]["current_runtime_available"] else 0
    contract[1, 0] = 1 if claasp["semantic_checks"]["selected_output_bit_api_present"] else 0
    contract[1, 1] = 1 if atm["semantic_checks"]["linear_output_candidates_exist"] else 0
    contract[2, 1] = 1 if atm["semantic_checks"]["constant_value_zero_or_one_is_known"] else 0
    contract[3, 1] = 1 if atm["semantic_checks"]["absence_from_found_subspace_is_complete_negative"] else 0
    contract[4, 1] = 1 if atm["semantic_checks"]["published_dimension_matches_recomputed_union_rank"] else 0

    decisions = {
        "innovation2_deterministic_provider_ready": "至少一个提供者契约完整；进入E32标签atlas捷径审计。",
        "innovation2_deterministic_provider_semantics_mismatch": "现有结果的常数值或负类语义不匹配；禁止直接训练。",
        "innovation2_deterministic_provider_runtime_unavailable": "同目标提供者当前不可执行；评估开放替代。",
        "innovation2_deterministic_provider_protocol_invalid": "来源版本、安全解析或结果文件协议无效。",
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.4,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.8, 8.2), gridspec_kw={"width_ratios": [1.15, 1]})
        figure.subplots_adjust(left=0.08, right=0.965, top=0.71, bottom=0.24, wspace=0.28)
        figure.suptitle(
            "创新2 E31：确定性积分标签提供者契约审计",
            x=0.08,
            y=0.96,
            ha="left",
            fontsize=15.2,
            fontweight="bold",
        )
        figure.text(
            0.08,
            0.885,
            "审计CLAASP-MP与代数转移矩阵公开实现：能否提供与当前‘仿射输入集合 + 线性输出mask + XOR=0/1’完全一致的标签。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.08,
            0.835,
            "本图不表示神经训练结果，也不把未知常数、广义多项关系或搜索未命中当作二分类标签。",
            ha="left",
            va="top",
            fontsize=9.5,
            color="#526070",
        )

        y = np.arange(len(labels))
        axes[0].barh(y, values, color=["#64748B", "#2563EB", "#0F766E", "#D97706", "#7C3AED", "#DC2626"])
        axes[0].set_yticks(y, labels)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("数量 / 维数")
        axes[0].set_title("ATM公开PRESENT-r9预计算结果", loc="left", fontweight="bold", pad=12)
        axes[0].grid(axis="x", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(values):
            axes[0].text(value + max(values) * 0.015, index, str(value), va="center", fontsize=9.2)
        axes[0].set_xlim(0, max(values) * 1.16)

        cmap = ListedColormap(["#94A3B8", "#DC2626", "#16A34A"])
        axes[1].imshow(contract + 1, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        axes[1].set_xticks([0, 1], ["CLAASP-MP", "ATM预计算"])
        axes[1].set_yticks(np.arange(len(contract_labels)), contract_labels)
        axes[1].set_title("当前目标的逐字段契约", loc="left", fontweight="bold", pad=12)
        for row in range(contract.shape[0]):
            for column in range(contract.shape[1]):
                state = contract[row, column]
                axes[1].text(
                    column,
                    row,
                    {1: "满足", 0: "不满足", -1: "未核验"}[int(state)],
                    ha="center",
                    va="center",
                    color="#FFFFFF" if state != -1 else "#0F172A",
                    fontweight="bold",
                    fontsize=9.0,
                )
        axes[1].set_xticks(np.arange(-0.5, 2, 1), minor=True)
        axes[1].set_yticks(np.arange(-0.5, len(contract_labels), 1), minor=True)
        axes[1].grid(which="minor", color="#FFFFFF", linewidth=2)
        axes[1].tick_params(which="minor", bottom=False, left=False)
        axes[1].legend(
            handles=[
                Patch(facecolor="#16A34A", label="满足"),
                Patch(facecolor="#DC2626", label="不满足"),
                Patch(facecolor="#94A3B8", label="未核验"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            frameon=False,
        )
        figure.text(
            0.08,
            0.105,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.08,
            0.055,
            "关键差异：ATM union rank=468，而论文报告dimension=470；其constant-search本身不返回常数究竟为0还是1。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.08,
            0.018,
            "证据范围：冻结公开commit与8份预计算文件的只读契约审计；不是CLAASP-MP复现、完整ATM空间复现或积分性质不存在性证明。",
            ha="left",
            va="bottom",
            fontsize=8.7,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
