from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager

from blockcipher_nd.tasks.innovation1.runtime_spn_structure_program_pretrain_k1bw import (
    CONFIG_PATH,
    ROOT,
    RUN_ID,
    load_and_validate_config,
    run_experiment,
)


DEFAULT_OUTPUT = ROOT / "outputs/local_diagnostic" / RUN_ID
CONTROL_LABELS = {
    "wrong_linear": "错误GF(2)连接",
    "wrong_sbox": "错误S盒语义",
    "wrong_order": "错误阶段顺序",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run K1-BW learnable Runtime-SPN structure-program gate."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_and_validate_config(args.config)
    payload = run_experiment(config, output_root=args.output_root, project_root=ROOT)
    report = render_k1bw_svg(
        payload["gate"],
        payload["results"],
        payload["history"],
        args.output_root / "curves.svg",
    )
    (args.output_root / "plot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if payload["gate"]["status"] == "invalid" else 0


def render_k1bw_svg(
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    _configure_chinese_font()
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle(
        "创新1 K1-BW：学习密码结构程序，而不是记住密码名称",
        fontsize=18,
        fontweight="bold",
    )

    loss_axis = axes[0, 0]
    for model_seed in sorted({int(row["model_seed"]) for row in history}):
        selected = [row for row in history if int(row["model_seed"]) == model_seed]
        loss_axis.plot(
            [int(row["epoch"]) for row in selected],
            [float(row["loss"]) for row in selected],
            linewidth=2.0,
            label=f"训练副本{model_seed}",
        )
    loss_axis.set_title("结构对比训练是否收敛")
    loss_axis.set_xlabel("训练轮次")
    loss_axis.set_ylabel("间隔损失")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(loc="upper right")

    holdout_axis = axes[0, 1]
    holdout_rows = [row for row in rows if row["scope"] == "holdout"]
    labels = []
    initial_values = []
    trained_values = []
    for model_seed in sorted({int(row["model_seed"]) for row in holdout_rows}):
        for control in CONTROL_LABELS:
            initial = [
                float(row["semantic_margin"])
                for row in holdout_rows
                if int(row["model_seed"]) == model_seed
                and row["control"] == control
                and row["phase"] == "initial"
            ]
            trained = [
                float(row["semantic_margin"])
                for row in holdout_rows
                if int(row["model_seed"]) == model_seed
                and row["control"] == control
                and row["phase"] == "trained"
            ]
            if not initial or not trained:
                continue
            labels.append(f"副本{model_seed}\n{CONTROL_LABELS[control]}")
            initial_values.append(min(initial))
            trained_values.append(min(trained))
    positions = list(range(len(labels)))
    holdout_axis.bar(
        [value - 0.18 for value in positions],
        initial_values,
        width=0.36,
        color="#A7B0B8",
        label="训练前",
    )
    holdout_axis.bar(
        [value + 0.18 for value in positions],
        trained_values,
        width=0.36,
        color="#007C83",
        label="训练后",
    )
    holdout_axis.axhline(0.02, color="#C4473A", linestyle="--", label="留出门槛0.02")
    holdout_axis.set_title("整密码留出：Dialga结构干预最小间隔")
    holdout_axis.set_ylabel("错误结构距离 - 重命名结构距离")
    holdout_axis.set_yscale("log")
    holdout_axis.set_ylim(1e-4, 1.0)
    holdout_axis.set_xticks(positions, labels)
    holdout_axis.grid(axis="y", which="both", alpha=0.25)
    holdout_axis.legend(loc="upper left")

    cipher_axis = axes[1, 0]
    cipher_labels = []
    cipher_values = []
    for cipher in sorted({str(row["cipher_key"]) for row in rows}):
        selected = [
            float(row["semantic_margin"])
            for row in rows
            if row["phase"] == "trained" and row["cipher_key"] == cipher
        ]
        cipher_labels.append(cipher.replace("64", "-64").replace("128", "-128"))
        cipher_values.append(min(selected))
    cipher_bars = cipher_axis.bar(cipher_labels, cipher_values, color="#35618D")
    cipher_axis.bar_label(
        cipher_bars,
        labels=[f"{value:.3f}" for value in cipher_values],
        padding=3,
        fontsize=8.5,
    )
    cipher_axis.set_title("七种SPN的训练后最弱结构间隔")
    cipher_axis.set_ylabel("最小语义间隔")
    cipher_axis.set_yscale("log")
    cipher_axis.set_ylim(1e-3, 1.2)
    cipher_axis.tick_params(axis="x", rotation=28)
    cipher_axis.grid(axis="y", which="both", alpha=0.25)

    summary_axis = axes[1, 1]
    summary_axis.axis("off")
    status_text = "通过" if gate["status"] == "pass" else "暂缓"
    summaries = gate["seed_summaries"]
    lines = [
        f"裁决：{status_text}",
        "",
        "输入：7份公开SPN结构说明书",
        "学习内容：S盒真值表、GF(2)实际连线、阶段顺序",
        "禁止输入：密码名称、密码ID、每密码独立参数",
        "留出密码：Dialga-128（训练时完全不可见）",
        "",
    ]
    for seed, values in summaries.items():
        lines.extend(
            [
                f"副本{seed}：",
                f"  留出最小间隔 {values['minimum_holdout_semantic_margin']:.4f}",
                f"  相对随机初始化增益 {values['minimum_holdout_margin_gain']:.4f}",
                f"  重命名一致性 {values['minimum_relabel_cosine']:.7f}",
            ]
        )
    lines.extend(
        [
            "",
            "边界：这是结构表示门，不是差分区分AUC。",
            "通过后才允许把冻结结构向量接入Runtime-E4。",
        ]
    )
    summary_axis.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11.5,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F4F5F2", "edgecolor": "#777777"},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", dpi=180)
    plt.close(figure)
    return {
        "run_id": gate["run_id"],
        "status": gate["status"],
        "panels": 4,
        "holdout": "dialga128",
        "output": str(output),
    }


def _configure_chinese_font() -> None:
    preferred = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in preferred if name in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False


__all__ = ["main", "parse_args", "render_k1bw_svg"]


if __name__ == "__main__":
    raise SystemExit(main())
