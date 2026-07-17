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

from blockcipher_nd.tasks.innovation2.skinny_r8_geometry_diversity import (
    SkinnyR8GeometryDiversityConfig,
    run_skinny_r8_geometry_diversity_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SKINNY-64/64 r8 adjacent-pair kernel diversity."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discovery-keys", type=int, default=64)
    parser.add_argument("--validation-keys", type=int, default=64)
    parser.add_argument("--key-chunk-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SkinnyR8GeometryDiversityConfig(
        run_id=args.run_id,
        seed=args.seed,
        discovery_keys=args.discovery_keys,
        validation_keys=args.validation_keys,
        key_chunk_size=args.key_chunk_size,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "cipher": "SKINNY-64/64",
            "rounds": 8,
            "structures": 16,
            "discovery_keys": args.discovery_keys,
            "validation_keys": args.validation_keys,
            "plaintexts_per_structure_per_key": 256,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_skinny_r8_geometry_diversity_audit(
        config,
        progress_callback=progress_callback,
    )
    (args.output_root / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "kernel_basis.csv", result["basis_rows"])
    np.save(args.output_root / "keys.npy", result["keys"], allow_pickle=False)
    np.save(
        args.output_root / "parity_rows.npy",
        result["parity_rows"],
        allow_pickle=False,
    )
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(args.output_root / "gate.json", result["gate"])
    render_geometry_diversity_svg(
        result["rows"], result["gate"], args.output_root / "curves.svg"
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
                "metrics": result["gate"]["metrics"],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_geometry_diversity_svg(
    rows: list[dict[str, Any]], gate: dict[str, Any], output_path: Path
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["pair_index"]))
    x = np.arange(len(ordered), dtype=np.float64)
    labels = [str(row["active_cells"]).replace(",", "-") for row in ordered]
    discovery = [int(row["discovery_nullity"]) for row in ordered]
    validation = [int(row["validation_nullity"]) for row in ordered]
    joint = [int(row["joint_nullity"]) for row in ordered]
    signatures = [str(row["joint_kernel_signature"]) for row in ordered]
    nonempty_signatures = sorted({signature for signature in signatures if signature})
    signature_ids = {
        signature: index + 1 for index, signature in enumerate(nonempty_signatures)
    }
    signature_values = [signature_ids.get(signature, 0) for signature in signatures]
    point_colors = [
        "#DC2626"
        if bool(row["is_paper_anchor"])
        else "#64748B"
        if bool(row["is_same_budget_control"])
        else "#2563EB"
        for row in ordered
    ]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.6,
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
        figure, axes = plt.subplots(1, 2, figsize=(15.6, 7.2))
        figure.subplots_adjust(
            left=0.065,
            right=0.975,
            top=0.70,
            bottom=0.25,
            wspace=0.28,
        )
        figure.suptitle(
            "创新2 E22：SKINNY-64/64 8轮相邻活动 pair kernel 多样性",
            x=0.065,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.065,
            0.89,
            "16个循环相邻 pair；同一组64+64把新密钥；只把128-key joint kernel计入稳定多样性。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        dimension_axis, signature_axis = axes
        dimension_axis.plot(
            x,
            discovery,
            color="#2563EB",
            marker="o",
            markersize=4,
            linewidth=1.3,
            label="发现64把",
        )
        dimension_axis.plot(
            x,
            validation,
            color="#059669",
            marker="s",
            markersize=4,
            linewidth=1.3,
            label="验证64把",
        )
        dimension_axis.plot(
            x,
            joint,
            color="#DC2626",
            marker="D",
            markersize=4,
            linewidth=1.5,
            label="联合128把",
        )
        dimension_axis.axvline(
            0,
            color="#64748B",
            linestyle=":",
            linewidth=1.2,
            label="控制 0-1",
        )
        dimension_axis.axvline(
            14,
            color="#D97706",
            linestyle="--",
            linewidth=1.2,
            label="论文 anchor 14-15",
        )
        dimension_axis.set_title(
            "各活动 pair 的经验 kernel 维数",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        dimension_axis.set_xlabel("活动 plaintext cell pair")
        dimension_axis.set_ylabel("kernel 维数 / nullity")
        dimension_axis.set_xticks(x, labels, rotation=45, ha="right")
        dimension_axis.set_ylim(0, max(2.5, max(discovery + validation + joint) + 0.8))
        dimension_axis.grid(True, color="#E5E7EB", linewidth=0.8)
        dim_handles, dim_labels = dimension_axis.get_legend_handles_labels()
        figure.legend(
            dim_handles,
            dim_labels,
            loc="upper left",
            bbox_to_anchor=(0.065, 0.81),
            ncol=5,
            frameon=False,
            fontsize=8.2,
        )

        signature_axis.scatter(
            x,
            signature_values,
            s=52,
            c=point_colors,
            edgecolors="#FFFFFF",
            linewidths=0.7,
        )
        signature_axis.axvline(0, color="#64748B", linestyle=":", linewidth=1.2)
        signature_axis.axvline(14, color="#D97706", linestyle="--", linewidth=1.2)
        signature_axis.set_title(
            "128-key joint kernel 签名类别",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        signature_axis.set_xlabel("活动 plaintext cell pair")
        signature_axis.set_ylabel("签名类别（0表示空 kernel）")
        signature_axis.set_xticks(x, labels, rotation=45, ha="right")
        signature_axis.set_yticks(
            range(0, max(signature_values, default=0) + 1)
        )
        signature_axis.set_ylim(
            -0.3, max(1.3, max(signature_values, default=0) + 0.5)
        )
        signature_axis.grid(True, color="#E5E7EB", linewidth=0.8)
        signature_axis.scatter([], [], color="#DC2626", label="论文 anchor")
        signature_axis.scatter([], [], color="#64748B", label="同预算 control")
        signature_axis.scatter([], [], color="#2563EB", label="其他 pair")
        sig_handles, sig_labels = signature_axis.get_legend_handles_labels()
        figure.legend(
            sig_handles,
            sig_labels,
            loc="upper right",
            bbox_to_anchor=(0.975, 0.81),
            ncol=3,
            frameon=False,
            fontsize=8.2,
        )

        metrics = gate["metrics"]
        decision_labels = {
            "innovation2_skinny_r8_geometry_kernel_diversity_ready": (
                "稳定 kernel 与签名均达到多样性门，可构造结构-mask 标签"
            ),
            "innovation2_skinny_r8_geometry_kernel_not_diverse": (
                "稳定 kernel 或签名不足，停止循环相邻 pair 扩展"
            ),
            "innovation2_skinny_r8_geometry_protocol_invalid": (
                "论文 anchor、密钥所有权、缓存或 GF(2) 协议无效"
            ),
        }
        figure.text(
            0.065,
            0.085,
            (
                "多样性：非零 joint kernel 结构 "
                f"{metrics['nontrivial_joint_kernel_structures']}/16（门槛≥4）；"
                "不同签名 "
                f"{metrics['distinct_nontrivial_joint_kernel_signatures']}（门槛≥4）；"
                "平均 half 存活率 "
                f"{metrics['mean_discovery_basis_validation_survival_fraction']:.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        figure.text(
            0.065,
            0.04,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "这是本地 sampled-key kernel 审计，不是神经训练或新性质证明。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV output requires at least one row: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
