from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E52 PRESENT-80 r5 strict-label provider coverage."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_r5_strict_label_provider(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_r5_strict_label_provider(
    summary: dict[str, Any], output: Path
) -> None:
    metrics = summary["metrics"]
    gate = summary["gate"]
    p1 = summary["provider_manifest"]["providers"][1]
    counts = [metrics["positive"], metrics["negative"], metrics["unknown"]]
    coverage_actual = [
        metrics["mixed_positive_negative_structures"],
        metrics["matched_train_positive"],
        metrics["matched_validation_positive"],
    ]
    coverage_required = [48, 400, 118]
    p1_checks = [
        int(p1["readiness"]["semantically_selected"]),
        int(bool(p1["runtime"]["sage_executable"])),
        int(p1["runtime"]["claasp_present_importable"]),
        int(p1["runtime"]["gurobipy_available"]),
        int(p1["runtime"]["gurobi_license_status"] == "verified"),
    ]
    decisions = {
        "innovation2_present_r5_strict_label_bank_not_ready": (
            "五轮严格标签库未就绪；禁止训练新网络。"
        ),
        "innovation2_present_r5_strict_label_p1_subset_required": (
            "P1可执行；下一步运行固定16×64子集。"
        ),
        "innovation2_present_r5_strict_label_bank_ready": (
            "五轮严格标签门通过；先做确定性捷径归因。"
        ),
        "innovation2_present_r5_strict_label_provider_protocol_invalid": (
            "标签或反例协议无效；只修协议。"
        ),
    }

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(15.8, 8.4))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.25, wspace=0.40
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E52：PRESENT-80 五轮能否生成严格、可训练的积分平衡标签",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "目标不是预测密文值，而是预测指定输出掩码对所有密钥、所有 inactive offset 是否保持 XOR 平衡。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "P0 是可靠但偏松的 ANF 支撑上界；P1 是保留非 cube 明文变量的 CLAASP-MP 完整 superpoly 证书。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )

        x = np.arange(3)
        axes[0].bar(x, counts, color=["#0F766E", "#2563EB", "#94A3B8"])
        padding = max(counts) * 0.035
        for index, value in enumerate(counts):
            axes[0].text(
                index,
                value + padding,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontweight="bold" if index == 0 else "normal",
            )
        axes[0].set_xticks(x, ["可证明平衡", "反例非平衡", "未知"])
        axes[0].set_ylim(0, max(counts) * 1.16)
        axes[0].set_ylabel("structure × output mask")
        axes[0].set_title(
            "P0 五轮三态标签覆盖\n输出位支撑全部饱和：6,144/6,144 均为 256/256",
            loc="left",
            fontweight="bold",
            fontsize=9.8,
            pad=8,
        )
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        y = np.arange(3)
        axes[1].barh(
            y + 0.17,
            coverage_required,
            height=0.31,
            color="#CBD5E1",
            label="门槛",
        )
        axes[1].barh(
            y - 0.17,
            coverage_actual,
            height=0.31,
            color="#0F766E",
            label="实际",
        )
        axes[1].scatter(
            coverage_actual,
            y - 0.17,
            s=34,
            color="#0F766E",
            zorder=3,
        )
        for index, (actual, required) in enumerate(
            zip(coverage_actual, coverage_required, strict=True)
        ):
            axes[1].text(
                max(actual, required) + max(coverage_required) * 0.025,
                index,
                f"{actual}/{required}",
                va="center",
                ha="left",
            )
        axes[1].set_yticks(y, ["正负混合结构", "训练每类", "验证每类"])
        axes[1].set_xlim(0, max(coverage_required) * 1.22)
        axes[1].set_xlabel("实际数量 / 最低门槛")
        axes[1].set_title("可训练标签宽度门", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="lower right")
        axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.8)

        labels = ["语义匹配", "Sage", "CLAASP可导入", "gurobipy", "许可证"]
        colors = ["#0F766E" if value else "#DC2626" for value in p1_checks]
        axes[2].barh(np.arange(5), np.ones(5), color=colors, height=0.58)
        for index, value in enumerate(p1_checks):
            axes[2].text(
                0.5,
                index,
                "满足" if value else "缺失/未核验",
                ha="center",
                va="center",
                color="#FFFFFF",
                fontweight="bold",
                fontsize=8.6,
            )
        axes[2].set_yticks(np.arange(5), labels)
        axes[2].set_xlim(0, 1)
        axes[2].set_xticks([])
        axes[2].set_title("P1 完整 superpoly 执行门", loc="left", fontweight="bold")
        axes[2].invert_yaxis()
        axes[2].text(
            0.0,
            -0.14,
            "key-coefficient 仅覆盖零 offset，不能替代此门。",
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            color="#B91C1C",
            fontsize=8.8,
        )

        figure.text(
            0.065,
            0.145,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=10.0,
            color="#334155",
        )
        figure.text(
            0.065,
            0.085,
            "推荐：先补可验证的 P1/P2 标签提供者；五轮标签门通过前，不训练 GraphGPS、Transformer 或其他新网络。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.065,
            0.035,
            "证据范围：PRESENT-80 五轮严格标签覆盖审计；不是神经训练、五轮区分器、密码攻击或 SOTA 结果。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
