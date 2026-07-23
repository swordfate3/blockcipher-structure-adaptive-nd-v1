from __future__ import annotations

import argparse
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

import numpy as np
from matplotlib import pyplot as plt

from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_readiness import (
    RUN_ID,
    SkinnyRuntimeReadinessConfig,
    run_skinny_runtime_readiness,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit RTG1-T2-A SKINNY general-GF(2) data readiness."
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    progress_path.unlink(missing_ok=True)
    _append_progress(progress_path, "run_start", {"run_id": args.run_id})
    result = run_skinny_runtime_readiness(
        SkinnyRuntimeReadinessConfig(run_id=args.run_id),
        cache_root=args.output_root / "cache",
        progress_callback=lambda event, payload: _append_progress(
            progress_path, event, payload
        ),
    )
    _write_jsonl(args.output_root / "results.jsonl", result["rows"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(args.output_root / "summary.json", result["summary"])
    _write_json(args.output_root / "gate.json", result["gate"])
    render_skinny_readiness_svg(result["summary"], args.output_root / "curves.svg")
    _append_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
        },
    )
    print(json.dumps(result["gate"], ensure_ascii=False, sort_keys=True))
    return 0 if result["gate"]["status"] == "pass" else 1


def render_skinny_readiness_svg(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    category_counts = gate["category_counts"]
    category_order = (
        "cipher_factory",
        "strict_dataset",
        "cache_replay",
        "runtime_model",
        "general_gf2",
    )
    category_labels = (
        "密码工厂",
        "严格差分数据",
        "缓存与回放",
        "运行时模型",
        "一般 GF(2)",
    )
    passed = [category_counts[name]["passed"] for name in category_order]
    totals = [category_counts[name]["total"] for name in category_order]
    degrees = np.asarray(summary["linear_row_degrees"], dtype=np.int64)
    degree_values, degree_counts = np.unique(degrees, return_counts=True)
    all_passed = gate["status"] == "pass"

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(14.2, 7.4))
        figure.subplots_adjust(
            left=0.10, right=0.97, top=0.72, bottom=0.18, wspace=0.34
        )
        figure.suptitle(
            "创新1 RTG1-T2-A：SKINNY 一般 GF(2) 数据通路就绪审计",
            x=0.08,
            y=0.96,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.08,
            0.895,
            "验证官方向量、严格负样本、磁盘缓存、参数几何和稀疏线性层；本轮不训练神经网络。",
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        conclusion = (
            "所有就绪门通过；下一步先冻结有信号的 SKINNY 差分协议，再允许本地三控制训练。"
            if all_passed
            else "存在就绪门失败；只修复失败的数据或结构适配项，禁止启动训练。"
        )
        figure.text(
            0.08,
            0.845,
            conclusion,
            ha="left",
            va="top",
            color="#047857" if all_passed else "#B42318",
            fontweight="bold",
            fontsize=10.3,
        )

        y = np.arange(len(category_labels))
        axes[0].barh(
            y,
            totals,
            color="#E2E8F0",
            height=0.62,
            label="全部检查",
        )
        passed_bars = axes[0].barh(
            y,
            passed,
            color="#059669" if all_passed else "#DC2626",
            height=0.62,
            label="已通过",
        )
        axes[0].set_yticks(y, labels=category_labels)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("检查项数量")
        axes[0].set_title("五类就绪门", loc="left", fontweight="bold")
        axes[0].grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        axes[0].bar_label(
            passed_bars,
            labels=[f"{ok}/{total}" for ok, total in zip(passed, totals, strict=True)],
            padding=5,
        )
        axes[0].legend(loc="lower right", frameon=False)

        axes[1].bar(
            degree_values,
            degree_counts,
            color="#2563EB",
            width=0.58,
        )
        for degree, count in zip(degree_values, degree_counts, strict=True):
            axes[1].text(
                degree + (0.08 if degree == 1 else 0.0),
                count + 0.6,
                str(count),
                ha="center",
                va="bottom",
            )
        axes[1].axvline(
            1.0,
            color="#DC2626",
            linestyle="--",
            linewidth=1.4,
            label="一对一 P 层：入度 1",
        )
        axes[1].set_xticks(sorted(set(degree_values.tolist() + [1])))
        axes[1].set_xlabel("每个输出 bit 依赖的输入 bit 数")
        axes[1].set_ylabel("输出 bit 数量")
        axes[1].set_title(
            "SKINNY ShiftRows + MixColumns 行度分布",
            loc="left",
            fontweight="bold",
        )
        axes[1].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(loc="upper right", frameon=False)

        figure.text(
            0.08,
            0.055,
            "证据范围：本地 64/class + 32/class 数据就绪与代数回放；不含 AUC、拓扑优越性、正式规模或 SOTA。",
            ha="left",
            va="bottom",
            color="#334155",
            fontsize=10.0,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


def _append_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
