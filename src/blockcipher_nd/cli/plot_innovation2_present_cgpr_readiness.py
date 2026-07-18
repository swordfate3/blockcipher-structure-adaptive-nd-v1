from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROW_IDS = (
    "cgpr_prefix_only_seed0",
    "cgpr_pair_true_seed0",
    "cgpr_pair_corrupted_seed0",
)
ROW_LABELS = ("prefix-only\n残差", "正确P\npair残差", "错误P\npair残差")
ROW_COLORS = ("#64748B", "#0F766E", "#D97706")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E50 PRESENT CGPR readiness.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_cgpr_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_cgpr_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    row_by_id = {row["row_id"]: row for row in summary["rows"]}
    trained = [row_by_id[row_id] for row_id in ROW_IDS]
    decisions = {
        "innovation2_present_cgpr_readiness_passed": (
            "零残差、冻结ridge、参数和拓扑契约通过；允许另建E51正式计划。"
        ),
        "innovation2_present_cgpr_readiness_failed": (
            "source、ridge、零等价、拓扑、参数或训练契约失败；先修实现。"
        ),
    }

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.6,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.4, 8.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.27, wspace=0.30
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E50：确定性ANF证书前端能否安全引导pair-state神经残差",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "冻结E45 1–3轮ANF-prefix ridge；神经分支只输出±0.25有界残差，不重学密码传播。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "本图是2轮本地实现readiness；AUC只用于排查异常，不作为候选性能或创新结论。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(len(trained))
        validation = [float(row["validation_auc"]) for row in trained]
        axes[0].bar(x, validation, color=ROW_COLORS, width=0.60)
        for index, value in enumerate(validation):
            axes[0].text(index, value + 0.008, f"{value:.3f}", ha="center")
        ridge_auc = float(gate["metrics"]["ridge_validation_auc"])
        axes[0].axhline(
            ridge_auc,
            color="#2563EB",
            linestyle="--",
            linewidth=1.2,
            label=f"冻结ridge {ridge_auc:.3f}",
        )
        axes[0].axhline(
            0.5,
            color="#64748B",
            linestyle=":",
            linewidth=1.1,
            label="随机水平 0.50",
        )
        axes[0].set_xticks(x, ROW_LABELS)
        axes[0].set_ylim(0.35, max(0.78, max(validation) + 0.08))
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("2轮readiness最终checkpoint", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="upper right", fontsize=8.5)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        width = 0.34
        train_auc = [float(row["train_auc"]) for row in trained]
        axes[1].bar(
            x - width / 2,
            train_auc,
            width=width,
            color="#94A3B8",
            label="训练 AUC",
        )
        axes[1].bar(
            x + width / 2,
            validation,
            width=width,
            color=ROW_COLORS,
            label="验证 AUC",
        )
        for index, (train_value, validation_value) in enumerate(
            zip(train_auc, validation, strict=True)
        ):
            axes[1].text(
                index - width / 2,
                train_value + 0.008,
                f"{train_value:.3f}",
                ha="center",
                fontsize=8.4,
            )
            axes[1].text(
                index + width / 2,
                validation_value + 0.008,
                f"{validation_value:.3f}",
                ha="center",
                fontsize=8.4,
            )
        axes[1].set_xticks(x, ROW_LABELS)
        axes[1].set_ylim(0.35, max(0.82, max([*train_auc, *validation]) + 0.08))
        axes[1].set_ylabel("AUC")
        axes[1].set_title("训练/验证差距（只作流程诊断）", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        metrics = gate["metrics"]
        counts = metrics["parameter_counts"]
        figure.text(
            0.075,
            0.178,
            f"残差参数：prefix={counts['prefix']:,}，pair={counts['pair_true']:,}，相对差={metrics['parameter_relative_spread']*100:.2f}%；零残差误差={metrics['zero_residual_max_abs_error']:.1e}。",
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.075,
            0.111,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.053,
            "证据范围：PRESENT-80四轮CGPR的2轮本地实现readiness；不是有效预测、高轮结论、新攻击或远程规模结果。",
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
