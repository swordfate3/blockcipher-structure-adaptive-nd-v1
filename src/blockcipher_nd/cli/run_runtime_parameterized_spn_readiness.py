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

from blockcipher_nd.tasks.innovation1.runtime_parameterized_spn_readiness import (
    RuntimeSpnReadinessConfig,
    run_runtime_spn_readiness,
)


DEFAULT_RUN_ID = "i1_rtg1_runtime_parameterized_spn_r0_readiness_seed0_20260723"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Innovation 1 RTG1 runtime-SPN implementation readiness gate."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root or Path("outputs/local_readiness") / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {"run_id": args.run_id, "training_performed": False},
        mode="w",
    )
    result = run_runtime_spn_readiness(
        RuntimeSpnReadinessConfig(run_id=args.run_id, seed=args.seed)
    )
    _write_jsonl(output_root / "results.jsonl", result["rows"])
    _write_csv(output_root / "cells.csv", result["cell_rows"])
    _write_json(output_root / "contract.json", result["contract"])
    _write_json(output_root / "summary.json", result["summary"])
    _write_json(output_root / "gate.json", result["gate"])
    render_runtime_spn_readiness_svg(
        result["rows"],
        result["gate"],
        output_root / "curves.svg",
    )
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
            "training_performed": False,
        },
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": result["gate"]["status"],
                "decision": result["gate"]["decision"],
                "checks": (
                    f"{result['gate']['checks_passed']}/"
                    f"{result['gate']['checks_total']}"
                ),
                "output_root": str(output_root),
                "next_action": result["gate"]["next_action"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] == "pass" else 1


def render_runtime_spn_readiness_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    structure_labels = {
        "PRESENT-64 permutation": "PRESENT\n64位 P层",
        "GIFT-64 permutation": "GIFT\n64位 P层",
        "SKINNY-64 sparse GF(2)": "SKINNY\n64位 GF(2)",
        "Synthetic-128 permutation": "合成结构\n128位 P层",
    }
    labels = [structure_labels[str(row["structure"])] for row in rows]
    block_bits = [int(row["block_bits"]) for row in rows]
    edge_counts = [int(row["linear_edges"]) // int(row["rounds"]) for row in rows]
    checks = list(gate["readiness_checks"].items())
    check_labels = {
        "four_runtime_structures_covered": "4种运行时结构",
        "shared_parameter_geometry_stable": "共享参数形状不变",
        "runtime_structure_absent_from_state": "结构不写入模型权重",
        "variable_width_and_pair_shapes_valid": "可变位宽与pair数",
        "finite_forward_backward_all_structures": "前向/反向数值有限",
        "exact_gf2_inverses_valid": "GF(2)逆矩阵正确",
        "permutation_and_general_gf2_supported": "P层与一般线性层",
        "permutation_gather_matches_gf2": "P层两种计算一致",
        "degree_preserving_corruption_valid": "打乱拓扑保持度数",
        "true_and_corrupted_logits_distinct": "正确/打乱结构可区分",
        "true_and_independent_logits_distinct": "正确/无结构可区分",
        "independent_mode_preserves_capacity": "无结构控制同容量",
        "sbox_descriptor_changes_context": "S盒描述确实生效",
        "cell_relabel_equivariance": "cell重编号等变",
        "invalid_contract_inputs_rejected": "非法结构快速拒绝",
    }
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
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(15.2, 8.4),
            gridspec_kw={"width_ratios": (1.0, 1.35)},
        )
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.76,
            bottom=0.17,
            wspace=0.34,
        )
        figure.suptitle(
            "创新1 RTG1：运行时结构参数化 SPN 区分器就绪审判",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=16.0,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.905,
            "同一组神经网络权重接收外部 cell、S盒和线性扩散描述；覆盖 P层与一般 GF(2) 线性层。",
            ha="left",
            va="top",
            fontsize=10.5,
            color="#475569",
        )
        figure.text(
            0.075,
            0.858,
            "注意：这是实现契约检查，不含训练或 AUC；初始化 logits 差异不代表正确拓扑性能更好。",
            ha="left",
            va="top",
            fontsize=10.2,
            color="#B42318",
            fontweight="bold",
        )

        coverage_axis, checks_axis = axes
        x = np.arange(len(rows), dtype=np.float64)
        bars = coverage_axis.bar(
            x,
            block_bits,
            color=["#2563EB", "#0891B2", "#DC2626", "#7C3AED"],
            width=0.64,
        )
        coverage_axis.bar_label(
            bars,
            labels=[
                f"{bits}位\n{edges}条边/轮"
                for bits, edges in zip(block_bits, edge_counts, strict=True)
            ],
            padding=5,
            fontsize=9.0,
        )
        coverage_axis.set_title("一套主干覆盖的结构", loc="left", fontweight="bold", pad=10)
        coverage_axis.set_ylabel("分组密码状态位数")
        coverage_axis.set_xticks(x, labels=labels)
        coverage_axis.set_ylim(0, max(block_bits) * 1.28)
        coverage_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        coverage_axis.grid(False, axis="x")

        y = np.arange(len(checks), dtype=np.float64)
        values = [1.0 if passed else 0.0 for _, passed in checks]
        colors = ["#059669" if passed else "#DC2626" for _, passed in checks]
        check_bars = checks_axis.barh(y, values, color=colors, height=0.60)
        checks_axis.bar_label(
            check_bars,
            labels=["通过" if passed else "失败" for _, passed in checks],
            padding=5,
            fontsize=8.8,
        )
        checks_axis.set_title(
            f"实现门：{gate['checks_passed']}/{gate['checks_total']} 项通过",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        checks_axis.set_yticks(
            y,
            labels=[check_labels.get(name, name) for name, _ in checks],
        )
        checks_axis.set_xlim(0.0, 1.16)
        checks_axis.set_xticks([])
        checks_axis.invert_yaxis()
        checks_axis.grid(False)

        figure.text(
            0.075,
            0.055,
            "门控通过后的下一步：本地同预算训练，比较正确拓扑、保持度数的打乱拓扑、无拓扑和固定 E4 锚点。",
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
    if not rows:
        raise ValueError("cells.csv requires at least one row")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
