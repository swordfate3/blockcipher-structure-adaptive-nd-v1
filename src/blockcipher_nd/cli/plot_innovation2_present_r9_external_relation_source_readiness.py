from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the E106 PRESENT r9 external-source eligibility matrix."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_external_source_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_external_source_readiness(summary: dict[str, Any], output: Path) -> None:
    rows = summary["source_matrix"]
    criteria = (
        ("machine_readable_relations", "机器可读\n关系"),
        ("same_rounds", "同为\nPRESENT R9"),
        ("same_key_model", "同密钥\n模型"),
        ("same_relation_semantics", "同ATM\n关系语义"),
        ("training_identity_disjoint", "与训练身份\n完全互斥"),
        ("minimum_novelty_met", "新增维度\n至少32"),
    )
    values = np.asarray(
        [
            [2 if row[key] is None else int(bool(row[key])) for key, _ in criteria]
            for row in rows
        ],
        dtype=np.int8,
    )
    source_labels = [row["source_name"] for row in rows]
    cmap = ListedColormap(("#B91C1C", "#0F766E", "#94A3B8"))
    norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), cmap.N)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.6,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axis = plt.subplots(figsize=(15.6, 8.8))
        figure.subplots_adjust(left=0.30, right=0.965, top=0.71, bottom=0.25)
        figure.text(
            0.055,
            0.955,
            "创新2 E106：PRESENT九轮外部关系来源资格审计",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.055,
            0.895,
            "目标：为冻结E99 ATM关系排序模型寻找真正独立的外部测试源；本次不训练、不搜索、不使用远程GPU。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#475569",
        )
        figure.text(
            0.055,
            0.845,
            "只有六项全部通过的非训练来源才能开放零适配评价；论文输出mask和弱密钥组合不能冒充ATM关系。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#475569",
        )
        axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        axis.set_xticks(np.arange(len(criteria)), [label for _, label in criteria])
        axis.set_yticks(np.arange(len(rows)), source_labels)
        axis.tick_params(axis="x", pad=10)
        axis.tick_params(axis="y", pad=8)
        for row_index, row in enumerate(rows):
            for column_index, (key, _) in enumerate(criteria):
                value = row[key]
                text = "无证据" if value is None else ("通过" if value else "不通过")
                axis.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9.1,
                    fontweight="bold",
                )
        axis.set_xticks(np.arange(-0.5, len(criteria), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        axis.grid(which="minor", color="#FFFFFF", linewidth=2.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_title(
            "候选来源资格矩阵：当前没有来源满足全部条件",
            loc="left",
            pad=18,
            fontsize=11.5,
            fontweight="bold",
            color="#0F172A",
        )
        axis.legend(
            handles=(
                Patch(facecolor="#0F766E", label="通过"),
                Patch(facecolor="#B91C1C", label="不通过"),
                Patch(facecolor="#94A3B8", label="无可计算证据"),
            ),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=3,
            frameon=False,
        )
        novelty = summary["e104_novelty"]
        figure.text(
            0.055,
            0.145,
            (
                "E104复核：321条关系中318条完全重复、3条文件层面新；公开秩468，联合秩468，"
                f"新增维度={novelty['new_relation_space_dimensions']}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.055,
            0.085,
            "裁决：暂无可直接评估E99的外部来源；停止坐标身份迁移，遵守E97停止门并转论文收束。",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.055,
            0.035,
            "重开条件：出现新的可靠provider或满足零重合且新增至少32维的同语义独立来源；本结果不是神经指标或SOTA。",
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#64748B",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
