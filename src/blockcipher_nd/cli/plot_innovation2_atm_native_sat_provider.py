from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E58-A ATM native SAT audit.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_atm_native_sat_provider(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_atm_native_sat_provider(summary: dict[str, Any], output: Path) -> None:
    metrics = summary["calibration"]["metrics"]
    checks = summary["calibration"]["checks"]
    gate = summary["gate"]
    decisions = {
        "innovation2_atm_native_sat_mechanism_ready_for_r9_probe": (
            "原生SAT机制校准通过；只开放一个九轮relation mutation硬cap探针。"
        ),
        "innovation2_atm_native_sat_exact_calibration_failed": (
            "精确校准失败；禁止九轮探针。"
        ),
        "innovation2_atm_native_sat_source_or_environment_invalid": (
            "来源或运行环境无效；禁止解释provider结果。"
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
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.4))
        figure.subplots_adjust(
            left=0.07, right=0.975, top=0.70, bottom=0.28, wspace=0.42
        )
        figure.text(
            0.07,
            0.955,
            "创新2 E58-A：ATM作者原生PySAT能否生成严格key-monomial见证",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "校准对象：PRESENT S-box全部256个代数转移系数 + 一位函数 F_k(x)=x XOR k。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.848,
            "严格规则：只有具体非零key指数且trail计数为odd才是witness；达到cap必须返回unknown。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        nonzero = int(metrics["sbox_nonzero_coefficients"])
        zero = int(metrics["sbox_coefficients"]) - nonzero
        axes[0].bar(
            np.arange(3),
            [nonzero, zero, int(metrics["sbox_coefficients"]) - int(metrics["sbox_matches"])],
            color=["#0F766E", "#2563EB", "#DC2626"],
            width=0.62,
        )
        axes[0].set_xticks(np.arange(3), ["非零系数", "零系数", "不一致"])
        axes[0].set_ylabel("S-box (u,v) 坐标数")
        axes[0].set_title("256项真值逐项对拍", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate([nonzero, zero, 0]):
            axes[0].text(index, value + 4, str(value), ha="center", fontweight="bold")

        toy_labels = ["找到key项", "重放为odd", "常数项无witness", "cap为unknown"]
        toy_values = [
            checks["toy_key_term_returns_witness"],
            checks["toy_witness_replay_is_exactly_odd"],
            checks["toy_constant_term_has_no_key_witness"],
            checks["cap_exhaustion_is_unknown"],
        ]
        colors = ["#0F766E" if value else "#DC2626" for value in toy_values]
        axes[1].barh(np.arange(4), [1] * 4, color=colors, height=0.58)
        axes[1].set_yticks(np.arange(4), toy_labels)
        axes[1].set_xlim(0, 1.12)
        axes[1].set_xticks([])
        axes[1].invert_yaxis()
        axes[1].set_title("keyed toy严格见证契约", loc="left", fontweight="bold")
        for index, value in enumerate(toy_values):
            axes[1].text(0.5, index, "通过" if value else "失败", ha="center", va="center", color="#FFFFFF", fontweight="bold")

        scope_labels = ["作者commit", "Glucose4", "官方bitarrays", "真实80-bit调度"]
        scope_values = [
            gate["source_checks"]["atm_commit_matches_frozen_version"],
            gate["environment_checks"]["glucose4_available"],
            gate["environment_checks"]["bitarrays_extension_available"],
            False,
        ]
        scope_colors = ["#0F766E", "#0F766E", "#0F766E", "#94A3B8"]
        axes[2].barh(np.arange(4), [1] * 4, color=scope_colors, height=0.58)
        axes[2].set_yticks(np.arange(4), scope_labels)
        axes[2].set_xlim(0, 1.12)
        axes[2].set_xticks([])
        axes[2].invert_yaxis()
        axes[2].set_title("来源与可解释边界", loc="left", fontweight="bold")
        for index, value in enumerate(scope_values):
            label = "就绪" if value else "未建模"
            axes[2].text(0.5, index, label, ha="center", va="center", color="#FFFFFF", fontweight="bold")

        figure.text(
            0.07,
            0.17,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.07,
            0.108,
            "下一步：冻结一个九轮边际匹配mutation，60秒内只接受relation级odd key-monomial证书。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.07,
            0.048,
            "证据范围：低轮机制复现；不是九轮负类、PRESENT-80 master-key标签、神经结果或攻击。",
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
