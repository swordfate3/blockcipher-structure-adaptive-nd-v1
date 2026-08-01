from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager

from blockcipher_nd.tasks.innovation1.runtime_spn_target_cell_program_k1bx0 import (
    CONFIG_PATH,
    CONTROL_NAMES,
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
    "wrong_edge_binding": "错误目标cell绑定",
}
CONTROL_TICK_LABELS = {
    "wrong_linear": "错线性连接",
    "wrong_sbox": "错S盒",
    "wrong_order": "错顺序",
    "wrong_edge_binding": "错绑定",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run K1-BX0 target-cell structure-program repair."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_and_validate_config(args.config)
    payload = run_experiment(config, output_root=args.output_root, project_root=ROOT)
    report = render_k1bx0_svg(
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


def render_k1bx0_svg(
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    _configure_chinese_font()
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle(
        "创新1 K1-BX0：GF(2)边先绑定目标cell，再组合结构程序",
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
    loss_axis.set_title("目标cell结构对比训练")
    loss_axis.set_xlabel("训练轮次")
    loss_axis.set_ylabel("间隔损失")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(loc="upper right")

    holdout_axis = axes[0, 1]
    holdout_rows = [row for row in rows if row["scope"] == "holdout"]
    labels = []
    initial_values = []
    trained_values = []
    for model_seed in (0, 1):
        for control in CONTROL_NAMES:
            initial = [
                float(row["semantic_margin"])
                for row in holdout_rows
                if int(row["model_seed"]) == model_seed
                and row["phase"] == "initial"
                and row["control"] == control
            ]
            trained = [
                float(row["semantic_margin"])
                for row in holdout_rows
                if int(row["model_seed"]) == model_seed
                and row["phase"] == "trained"
                and row["control"] == control
            ]
            if not initial or not trained:
                continue
            labels.append(f"副本{model_seed}\n{CONTROL_TICK_LABELS[control]}")
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
    holdout_axis.set_yscale("log")
    holdout_axis.set_ylim(1e-4, 1.5)
    holdout_axis.set_title("整密码留出：Dialga四类结构干预")
    holdout_axis.set_ylabel("正确结构与错误结构的余弦距离间隔")
    holdout_axis.set_xticks(positions, labels, fontsize=8.5)
    holdout_axis.grid(axis="y", which="both", alpha=0.25)
    holdout_axis.legend(loc="upper left")

    comparison_axis = axes[1, 0]
    comparison_labels = []
    anchor_values = []
    candidate_values = []
    for model_seed in (0, 1):
        summary = gate["seed_summaries"][str(model_seed)]
        for control in ("wrong_linear", "wrong_sbox", "wrong_order"):
            comparison_labels.append(
                f"副本{model_seed}\n{CONTROL_TICK_LABELS[control]}"
            )
            if control == "wrong_sbox":
                anchor = float(summary["k1bw_wrong_sbox_anchor"])
            else:
                anchor = _k1bw_reference(model_seed, control)
            anchor_values.append(anchor)
            candidate_values.append(float(summary["controls"][control]["minimum_margin"]))
    positions = list(range(len(comparison_labels)))
    comparison_axis.bar(
        [value - 0.18 for value in positions],
        anchor_values,
        width=0.36,
        color="#8C6D46",
        label="K1-BW全局池化",
    )
    comparison_axis.bar(
        [value + 0.18 for value in positions],
        candidate_values,
        width=0.36,
        color="#35618D",
        label="K1-BX0目标cell聚合",
    )
    comparison_axis.axhline(0.02, color="#C4473A", linestyle="--", label="门槛0.02")
    comparison_axis.set_yscale("log")
    comparison_axis.set_ylim(1e-3, 1.5)
    comparison_axis.set_title("K1-BX0是否修复K1-BW的线性与顺序弱点")
    comparison_axis.set_ylabel("Dialga最小语义间隔")
    comparison_axis.set_xticks(positions, comparison_labels, fontsize=8.5)
    comparison_axis.grid(axis="y", which="both", alpha=0.25)
    comparison_axis.legend(loc="upper left")

    summary_axis = axes[1, 1]
    summary_axis.axis("off")
    status_text = "通过" if gate["status"] == "pass" else "暂缓"
    lines = [
        f"裁决：{status_text}",
        "",
        "唯一变化：GF(2)边先按真实目标cell聚合",
        "保留：S盒token、7种结构、Dialga留出、160轮",
        "新增控制：正确token + 错误目标cell绑定",
        "禁止输入：密码名称、密码ID、每密码独立参数",
        "",
    ]
    for seed, summary in gate["seed_summaries"].items():
        lines.append(f"副本{seed}：重命名一致性 {summary['minimum_relabel_cosine']:.7f}")
        for control in CONTROL_NAMES:
            values = summary["controls"][control]
            lines.append(
                f"  {CONTROL_LABELS[control]}：{values['minimum_margin']:.4f} "
                f"(增益 {values['minimum_gain']:+.4f})"
            )
    lines.extend(
        [
            "",
            "边界：这是结构表示门，不是差分区分AUC。",
            "只有全部控制过门，才允许接入Runtime-E4。",
        ]
    )
    summary_axis.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10.5,
        linespacing=1.3,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#F4F5F2",
            "edgecolor": "#777777",
        },
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


def _k1bw_reference(model_seed: int, control: str) -> float:
    values = {
        (0, "wrong_linear"): 0.007066,
        (0, "wrong_order"): 0.001681,
        (1, "wrong_linear"): 0.002279,
        (1, "wrong_order"): 0.002347,
    }
    return values[(model_seed, control)]


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


__all__ = ["main", "parse_args", "render_k1bx0_svg"]


if __name__ == "__main__":
    raise SystemExit(main())
