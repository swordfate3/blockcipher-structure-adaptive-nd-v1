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

from blockcipher_nd.tasks.innovation2.skinny_r8_bottom_row_closure import (
    SkinnyR8BottomRowClosureConfig,
    run_skinny_r8_bottom_row_closure_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SKINNY-64/64 r8 bottom-row pair kernel closure."
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
    config = SkinnyR8BottomRowClosureConfig(
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
            "bottom_row_structures": 6,
            "controls": 1,
            "discovery_keys": args.discovery_keys,
            "validation_keys": args.validation_keys,
            "plaintexts_per_structure_per_key": 256,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_skinny_r8_bottom_row_closure_audit(
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
    render_bottom_row_closure_svg(
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


def render_bottom_row_closure_svg(
    rows: list[dict[str, Any]], gate: dict[str, Any], output_path: Path
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["pair_index"]))
    x = np.arange(len(ordered), dtype=np.float64)
    labels = [str(row["active_cells"]).replace(",", "-") for row in ordered]
    discovery = [int(row["discovery_nullity"]) for row in ordered]
    validation = [int(row["validation_nullity"]) for row in ordered]
    joint = [int(row["joint_nullity"]) for row in ordered]
    signatures = [str(row["joint_kernel_signature"]) for row in ordered]
    nonempty = sorted({signature for signature in signatures if signature})
    signature_ids = {value: index + 1 for index, value in enumerate(nonempty)}
    signature_values = [signature_ids.get(signature, 0) for signature in signatures]
    point_colors = [
        "#D97706"
        if bool(row["is_known_e22_anchor"])
        else "#64748B"
        if bool(row["is_same_budget_control"])
        else "#2563EB"
        for row in ordered
    ]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.8,
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
        figure, axes = plt.subplots(1, 2, figsize=(15.4, 7.2))
        figure.subplots_adjust(
            left=0.07,
            right=0.975,
            top=0.70,
            bottom=0.25,
            wspace=0.28,
        )
        figure.suptitle(
            "创新2 E23：SKINNY-64/64 8轮底行活动 pair kernel 闭合审判",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.89,
            "底行6个无序 pair + 负对照0-1；64+64把全新密钥；只以128-key joint kernel裁决。",
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
            markersize=5,
            linewidth=1.4,
            label="发现64把",
        )
        dimension_axis.plot(
            x,
            validation,
            color="#059669",
            marker="s",
            markersize=5,
            linewidth=1.4,
            label="验证64把",
        )
        dimension_axis.plot(
            x,
            joint,
            color="#DC2626",
            marker="D",
            markersize=5,
            linewidth=1.6,
            label="联合128把",
        )
        dimension_axis.axvline(
            5.5,
            color="#64748B",
            linestyle="--",
            linewidth=1.2,
            label="底行 / 控制边界",
        )
        dimension_axis.set_title(
            "各活动 pair 的经验 kernel 维数",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        dimension_axis.set_xlabel("活动 plaintext cell pair")
        dimension_axis.set_ylabel("kernel 维数 / nullity")
        dimension_axis.set_xticks(x, labels)
        dimension_axis.set_ylim(0, max(2.5, max(discovery + validation + joint) + 0.8))
        dimension_axis.grid(True, color="#E5E7EB", linewidth=0.8)
        handles, legend_labels = dimension_axis.get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.07, 0.81),
            ncol=4,
            frameon=False,
            fontsize=8.4,
        )

        signature_axis.scatter(
            x,
            signature_values,
            s=64,
            c=point_colors,
            edgecolors="#FFFFFF",
            linewidths=0.8,
        )
        signature_axis.axvline(5.5, color="#64748B", linestyle="--", linewidth=1.2)
        signature_axis.set_title(
            "128-key joint kernel 签名类别",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        signature_axis.set_xlabel("活动 plaintext cell pair")
        signature_axis.set_ylabel("签名类别（0表示空 kernel）")
        signature_axis.set_xticks(x, labels)
        signature_axis.set_yticks(range(0, max(signature_values, default=0) + 1))
        signature_axis.set_ylim(
            -0.3, max(1.3, max(signature_values, default=0) + 0.5)
        )
        signature_axis.grid(True, color="#E5E7EB", linewidth=0.8)
        signature_axis.scatter([], [], color="#D97706", label="E22已知 anchor")
        signature_axis.scatter([], [], color="#2563EB", label="E23新 pair")
        signature_axis.scatter([], [], color="#64748B", label="负对照0-1")
        sig_handles, sig_labels = signature_axis.get_legend_handles_labels()
        figure.legend(
            sig_handles,
            sig_labels,
            loc="upper right",
            bbox_to_anchor=(0.975, 0.81),
            ncol=3,
            frameon=False,
            fontsize=8.4,
        )

        metrics = gate["metrics"]
        decision_labels = {
            "innovation2_skinny_r8_bottom_row_pair_family_ready": (
                "底行 kernel 家族达到标签多样性门，进入 E24 捷径审计"
            ),
            "innovation2_skinny_r8_bottom_row_anchor_not_reproduced": (
                "E22 已知方向未全部在全新密钥上复现，停止该几何路线"
            ),
            "innovation2_skinny_r8_bottom_row_pair_family_not_closed": (
                "底行多样性、稳定性或控制特异性不足，停止该几何路线"
            ),
            "innovation2_skinny_r8_bottom_row_protocol_invalid": (
                "结构、密钥、公开向量、缓存或 GF(2) 协议无效"
            ),
        }
        figure.text(
            0.07,
            0.085,
            (
                "底行多样性：非零 joint kernel "
                f"{metrics['bottom_row_nontrivial_joint_kernel_structures']}/6（门槛≥5）；"
                "不同签名 "
                f"{metrics['bottom_row_distinct_joint_kernel_signatures']}（门槛≥5）；"
                "平均 half 存活率 "
                f"{metrics['mean_discovery_basis_validation_survival_fraction']:.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        figure.text(
            0.07,
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
