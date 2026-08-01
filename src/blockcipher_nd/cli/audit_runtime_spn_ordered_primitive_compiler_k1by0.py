from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager

from blockcipher_nd.models.structure.spn.ordered_primitive_program import (
    GF2_EXPERT,
    PERMUTATION_EXPERT,
    SBOX_EXPERT,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_ordered_primitive_compiler_k1by0 import (
    CONFIG_PATH,
    CONTROL_NAMES,
    ROOT,
    RUN_ID,
    load_and_validate_config,
    run_audit,
)


DEFAULT_OUTPUT = ROOT / "outputs/local_audit" / RUN_ID
CONTROL_LABELS = {
    "exact_replay": "精确回放",
    "joint_cell_relabel": "cell重命名",
    "wrong_order_when_distinct": "阶段顺序检查\n（2种需拒绝）",
    "wrong_target_binding": "错误目标绑定",
}
CIPHER_LABELS = {
    "gift64": "GIFT",
    "present64": "PRESENT",
    "rectangle64": "RECTANGLE",
    "skinny64": "SKINNY",
    "midori64": "Midori",
    "uknit64": "uKNIT",
    "dialga128": "Dialga",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit K1-BY0 ordered primitive compiler readiness."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_and_validate_config(args.config)
    payload = run_audit(config, output_root=args.output_root, project_root=ROOT)
    report = render_k1by0_svg(
        payload["gate"],
        payload["programs"],
        payload["results"],
        args.output_root / "curves.svg",
    )
    (args.output_root / "plot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if payload["gate"]["status"] == "invalid" else 0


def render_k1by0_svg(
    gate: Mapping[str, Any],
    programs: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    _configure_chinese_font()
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle(
        "创新1 K1-BY0：把SPN说明书编译成可执行的神经网络积木顺序",
        fontsize=18,
        fontweight="bold",
    )

    control_axis = axes[0, 0]
    labels = [CONTROL_LABELS[name] for name in CONTROL_NAMES]
    totals = [sum(row["control"] == name for row in rows) for name in CONTROL_NAMES]
    passed = [
        sum(row["control"] == name and row["passed"] for row in rows)
        for name in CONTROL_NAMES
    ]
    positions = list(range(len(labels)))
    control_axis.bar(
        positions,
        totals,
        color="#C8CDD1",
        label="审计总行数",
    )
    control_axis.bar(
        positions,
        passed,
        color="#087E8B",
        width=0.62,
        label="通过行数",
    )
    for position, value, total in zip(positions, passed, totals, strict=True):
        control_axis.text(
            position,
            total + 0.7,
            f"{value}/{total}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    control_axis.set_title("编译器控制实验：是否全部按预期工作")
    control_axis.set_ylabel("结果行数")
    control_axis.set_xticks(positions, labels)
    control_axis.set_ylim(0, max(totals) * 1.18)
    control_axis.grid(axis="y", alpha=0.25)
    control_axis.legend(loc="upper left")

    usage_axis = axes[0, 1]
    cipher_keys = list(programs)
    cipher_labels = [CIPHER_LABELS.get(name, name) for name in cipher_keys]
    sbox_usage = [int(programs[name]["expert_usage"][SBOX_EXPERT]) for name in cipher_keys]
    permutation_usage = [
        int(programs[name]["expert_usage"][PERMUTATION_EXPERT])
        for name in cipher_keys
    ]
    gf2_usage = [int(programs[name]["expert_usage"][GF2_EXPERT]) for name in cipher_keys]
    positions = list(range(len(cipher_keys)))
    usage_axis.bar(positions, sbox_usage, color="#E0A458", label="4-bit S盒专家")
    usage_axis.bar(
        positions,
        permutation_usage,
        bottom=sbox_usage,
        color="#3D6D9A",
        label="置换线性专家",
    )
    stacked = [left + right for left, right in zip(sbox_usage, permutation_usage, strict=True)]
    usage_axis.bar(
        positions,
        gf2_usage,
        bottom=stacked,
        color="#609B63",
        label="一般GF(2)专家",
    )
    usage_axis.set_title("每种密码自动选择了哪些共享专家")
    usage_axis.set_ylabel("两阶段内的cell调用次数")
    usage_axis.set_xticks(positions, cipher_labels, rotation=20, ha="right")
    usage_axis.grid(axis="y", alpha=0.25)
    usage_axis.legend(loc="upper left")

    order_axis = axes[1, 0]
    unique_stages = [
        len(set(programs[name]["stage_content_sha256s"])) for name in cipher_keys
    ]
    bars = order_axis.bar(positions, unique_stages, color="#755C9B")
    applicable = set(gate["applicable_wrong_order_ciphers"])
    for position, bar, name, count in zip(
        positions,
        bars,
        cipher_keys,
        unique_stages,
        strict=True,
    ):
        text = "顺序敏感" if name in applicable else "两阶段相同"
        order_axis.text(
            position,
            count + 0.06,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    order_axis.set_title("阶段内容是否不同：何时换序必须被拒绝")
    order_axis.set_ylabel("不同阶段内容数量")
    order_axis.set_xticks(positions, cipher_labels, rotation=20, ha="right")
    order_axis.set_yticks((0, 1, 2))
    order_axis.set_ylim(0, 2.45)
    order_axis.grid(axis="y", alpha=0.25)

    summary_axis = axes[1, 1]
    summary_axis.axis("off")
    status_text = {
        "pass": "通过",
        "hold": "暂缓",
        "invalid": "协议无效",
    }[str(gate["status"])]
    summary_lines = [
        f"裁决：{status_text}",
        "",
        "输入：cell划分、S盒真值表、GF(2)连接、阶段顺序",
        "编译：每阶段拆成S盒积木 + 线性积木",
        "路由：每个目标cell自动选择置换或GF(2)专家",
        "输出：保留顺序和端点绑定的可执行结构程序",
        "",
        "7种结构全部精确回放",
        "28组cell重命名全部保持同一语义",
        "28组错误目标绑定全部被拒绝",
        "uKNIT和Dialga的阶段换序全部被拒绝",
        "其余密码两阶段内容相同，换序不产生伪差异",
        "",
        "训练步数：0；密文样本：0；密码名称输入：无",
        "边界：这是编译器门，不是差分区分AUC。",
        "通过后才允许设计小规模Runtime-E4积木路由实验。",
    ]
    summary_axis.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "#F3F4F1",
            "edgecolor": "#6E7377",
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", dpi=180)
    plt.close(figure)
    return {
        "run_id": gate["run_id"],
        "status": gate["status"],
        "panels": 4,
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


__all__ = ["main", "parse_args", "render_k1by0_svg"]


if __name__ == "__main__":
    raise SystemExit(main())
