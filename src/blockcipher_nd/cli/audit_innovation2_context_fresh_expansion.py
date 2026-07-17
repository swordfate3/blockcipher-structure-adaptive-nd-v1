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

from blockcipher_nd.tasks.innovation2.integral_context_fresh_expansion import (
    ContextFreshExpansionConfig,
    run_context_fresh_expansion_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PRESENT r7 fresh-key expanded-context kernel diversity."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--keys", type=int, default=128)
    parser.add_argument("--key-chunk-size", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ContextFreshExpansionConfig(
        run_id=args.run_id,
        seed=args.seed,
        contexts=args.contexts,
        keys=args.keys,
        key_chunk_size=args.key_chunk_size,
    )
    source_gate = _read_json(args.source_root / "gate.json")
    source_metadata = _read_json(args.source_root / "metadata.json")
    source_basis_rows = _read_csv(args.source_root / "kernel_basis.csv")
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source_run_id": source_gate.get("run_id"),
            "contexts": args.contexts,
            "fresh_keys_per_context": args.keys,
            "key_chunk_size": args.key_chunk_size,
            "plaintexts_per_context_per_key": 1 << 16,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_context_fresh_expansion_audit(
        config,
        source_gate=source_gate,
        source_metadata=source_metadata,
        source_basis_rows=source_basis_rows,
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
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_fresh_expansion_svg(
        result["rows"],
        result["gate"],
        args.output_root / "curves.svg",
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
                "reproduced_e16_context_signatures": result["gate"][
                    "reproduced_e16_context_signatures"
                ],
                "distinct_joint_kernel_signatures": result["gate"][
                    "distinct_joint_kernel_signatures"
                ],
                "contexts_with_directions_beyond_hwang": result["gate"][
                    "contexts_with_directions_beyond_hwang"
                ],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_fresh_expansion_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["context_id"]))
    context_ids = np.asarray(
        [int(row["context_id"]) for row in ordered], dtype=np.int64
    )
    discovery = [int(row["discovery_kernel_dimension"]) for row in ordered]
    validation = [int(row["validation_kernel_dimension"]) for row in ordered]
    joint = [int(row["joint_kernel_dimension"]) for row in ordered]
    signatures = [str(row["joint_basis_signature"]) for row in ordered]
    signature_ids = {
        signature: index + 1 for index, signature in enumerate(sorted(set(signatures)))
    }
    signature_values = [signature_ids[signature] for signature in signatures]
    colors = ["#2563EB" if index < 16 else "#059669" for index in context_ids]
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
        figure, axes = plt.subplots(1, 2, figsize=(15.4, 7.1))
        figure.subplots_adjust(
            left=0.07,
            right=0.975,
            top=0.70,
            bottom=0.24,
            wspace=0.30,
        )
        figure.suptitle(
            "创新2 E18：PRESENT 7轮64-context fresh-key kernel 扩展",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.89,
            "前16个为 E16 anchors，后48个为新 context；全部使用同一组128把全新密钥。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        dimension_axis, signature_axis = axes
        dimension_axis.plot(
            context_ids,
            discovery,
            color="#2563EB",
            linewidth=1.2,
            alpha=0.75,
            label="发现64把",
        )
        dimension_axis.plot(
            context_ids,
            validation,
            color="#059669",
            linewidth=1.2,
            alpha=0.75,
            label="验证64把",
        )
        dimension_axis.plot(
            context_ids,
            joint,
            color="#DC2626",
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            label="联合128把",
        )
        dimension_axis.axvline(
            15.5,
            color="#64748B",
            linestyle="--",
            linewidth=1.1,
            label="E16 anchor / 新 context 边界",
        )
        dimension_axis.set_title(
            "64个 context 的经验 kernel 维数",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        dimension_axis.set_xlabel("context 编号")
        dimension_axis.set_ylabel("kernel 维数")
        dimension_axis.set_xticks(np.arange(0, 64, 4))
        dimension_axis.grid(True, color="#E5E7EB", linewidth=0.8)
        dimension_handles, dimension_labels = (
            dimension_axis.get_legend_handles_labels()
        )
        figure.legend(
            dimension_handles,
            dimension_labels,
            loc="upper left",
            bbox_to_anchor=(0.07, 0.81),
            ncol=4,
            frameon=False,
            fontsize=8.1,
        )

        signature_axis.scatter(
            context_ids,
            signature_values,
            s=42,
            c=colors,
            edgecolors="#FFFFFF",
            linewidths=0.6,
        )
        signature_axis.axvline(
            15.5,
            color="#64748B",
            linestyle="--",
            linewidth=1.1,
        )
        signature_axis.set_title(
            "fresh-key joint kernel 签名类别",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        signature_axis.set_xlabel("context 编号")
        signature_axis.set_ylabel("签名类别编号")
        signature_axis.set_xticks(np.arange(0, 64, 4))
        signature_axis.set_yticks(sorted(set(signature_values)))
        signature_axis.grid(True, color="#E5E7EB", linewidth=0.8)
        signature_axis.scatter([], [], color="#2563EB", label="E16 anchor")
        signature_axis.scatter([], [], color="#059669", label="新 context")
        signature_handles, signature_labels = (
            signature_axis.get_legend_handles_labels()
        )
        figure.legend(
            signature_handles,
            signature_labels,
            loc="upper right",
            bbox_to_anchor=(0.975, 0.81),
            ncol=2,
            frameon=False,
            fontsize=8.3,
        )

        decision_labels = {
            "innovation2_fresh_expanded_context_kernel_ready": (
                "fresh-key稳定且 context 多样性充足，可重建64-context标签"
            ),
            "innovation2_context_kernel_fresh_key_unstable": (
                "E16 kernel签名未在全新密钥上复现，停止该分支"
            ),
            "innovation2_fresh_expanded_context_diversity_insufficient": (
                "fresh-key稳定但新增 context 多样性不足，停止该分支"
            ),
            "innovation2_fresh_context_protocol_invalid": (
                "fresh-key、source或Hwang协议校验无效"
            ),
        }
        figure.text(
            0.07,
            0.055,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "这是 fresh-key kernel 审计，不是神经训练，也不是全密钥证明。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
