from __future__ import annotations

import argparse
import csv
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

from blockcipher_nd.tasks.innovation1.runtime_spn_semantic_equivalence import (
    RUN_ID,
    RuntimeSpnSemanticEquivalenceConfig,
    audit_runtime_spn_semantic_equivalence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit same-weight semantics between GIFT R1d and runtime E4."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    _write_progress(
        progress,
        "run_start",
        {"run_id": args.run_id, "training_performed": False},
        mode="w",
    )
    audit = audit_runtime_spn_semantic_equivalence(
        RuntimeSpnSemanticEquivalenceConfig(run_id=args.run_id, seed=args.seed)
    )
    _write_jsonl(args.output_root / "results.jsonl", audit["rows"])
    _write_csv(args.output_root / "stage_errors.csv", audit["rows"])
    _write_json(args.output_root / "summary.json", audit["summary"])
    _write_json(args.output_root / "gate.json", audit["gate"])
    render_semantic_equivalence_svg(
        audit["rows"], audit["gate"], args.output_root / "curves.svg"
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass"
        if all(audit["gate"]["protocol_checks"].values())
        and all(audit["gate"]["execution_checks"].values())
        else "hold",
        "checks": {
            "required_artifacts_written": all(
                (args.output_root / name).is_file()
                for name in (
                    "results.jsonl",
                    "stage_errors.csv",
                    "summary.json",
                    "gate.json",
                    "curves.svg",
                )
            ),
            **audit["gate"]["protocol_checks"],
            **audit["gate"]["execution_checks"],
        },
    }
    _write_json(args.output_root / "validation.json", validation)
    _write_progress(
        progress,
        "run_done",
        {
            "run_id": args.run_id,
            "status": audit["gate"]["status"],
            "decision": audit["gate"]["decision"],
            "training_performed": False,
        },
    )
    print(
        json.dumps(
            {"gate": audit["gate"], "output_root": str(args.output_root)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def render_semantic_equivalence_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    labels = {
        "msb_to_lsb_conversion": "MSB/LSB 坐标转换",
        "current_delta_bits": "当前差分 bit",
        "inverse_linear_previous_bits": "逆线性层差分 bit",
        "current_delta_cells": "当前差分 cell",
        "inverse_linear_previous_cells": "逆线性层差分 cell",
        "typed_fusion": "当前/前态类型融合",
        "first_mixer_input": "第1个 mixer 输入",
        "mixer_1_output": "第1个 mixer 输出",
        "mixer_2_output": "第2个 mixer 输出",
        "final_mixer_output": "归一化后的 mixer 输出",
        "pair_embeddings": "每个密文对的嵌入",
        "pair_attention_weights": "密文对注意力权重",
        "pair_attention_result": "密文对注意力汇总",
        "final_logits": "最终区分 logits",
    }
    errors = np.array(
        [max(float(row["maximum_absolute_error"]), 1e-16) for row in rows]
    )
    y = np.arange(len(rows), dtype=np.float64)
    passed = [bool(row["within_tolerance"]) for row in rows]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(13.8, 8.8))
        figure.subplots_adjust(left=0.30, right=0.96, top=0.76, bottom=0.16)
        figure.suptitle(
            "创新1 RTG1：GIFT 专用主干与运行时主干逐阶段语义等价审计",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.905,
            "复制同一组共享权重，关闭 S盒晚期注入；比较从位序转换到最终 logits 的最大绝对误差。",
            ha="left",
            va="top",
            fontsize=10.5,
            color="#475569",
        )
        conclusion = (
            "全部阶段误差不超过 1e-6：确定性运行时表示不是 seed1 AUC 缺口来源。"
            if gate["status"] == "pass"
            else f"首个偏差位于 {gate['first_divergent_stage']}：只修这一阶段，不扩大训练。"
        )
        figure.text(
            0.07,
            0.855,
            conclusion,
            ha="left",
            va="top",
            fontsize=10.3,
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
        )
        colors = ["#059669" if value else "#DC2626" for value in passed]
        axis.hlines(y, 1e-16, errors, color=colors, linewidth=2.0, alpha=0.65)
        axis.scatter(errors, y, color=colors, s=48, zorder=3)
        axis.axvline(
            float(gate["tolerance"]),
            color="#1D4ED8",
            linewidth=1.8,
            linestyle="--",
            label="等价阈值 1e-6",
        )
        axis.set_xscale("log")
        axis.set_xlim(5e-17, 3e-6)
        axis.set_yticks(y, labels=[labels[str(row["stage"])] for row in rows])
        axis.invert_yaxis()
        axis.set_xlabel("最大绝对误差（对数坐标；精确 0 显示为 1e-16）")
        axis.set_title("逐阶段误差", loc="left", fontweight="bold", pad=10)
        axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        axis.grid(False, axis="y")
        axis.legend(loc="lower right", frameon=False)
        figure.text(
            0.07,
            0.055,
            "这是无训练的确定性审计，不代表 AUC、跨密码性能或论文级结果。",
            ha="left",
            va="bottom",
            fontsize=10.0,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _write_progress(
    path: Path,
    event: str,
    payload: dict[str, Any],
    *,
    mode: str = "a",
) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
