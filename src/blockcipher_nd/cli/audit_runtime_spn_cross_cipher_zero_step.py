from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_zero_step import (
    EXPECTED_CONDITIONS,
    EXPECTED_SEEDS,
    adjudicate_zero_step_panel,
    evaluate_zero_step_seed,
    validate_target_result_rows,
)


RUN_ID = "i1_rtg1_gift_to_skinny_zero_step_topology_x1_seed0_seed1_20260724"
DEFAULT_GIFT_ROOTS = (
    Path(
        "outputs/local_diagnostic/"
        "i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0"
    ),
    Path(
        "outputs/local_diagnostic/"
        "i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed1"
    ),
)
DEFAULT_SKINNY_ROOTS = (
    Path(
        "outputs/local_diagnostic/"
        "i1_rtg1_skinny64_general_gf2_attribution_t2c_2048_seed0_20260724"
    ),
    Path(
        "outputs/local_diagnostic/"
        "i1_rtg1_skinny64_general_gf2_attribution_t2c_2048_seed1_20260724"
    ),
)
DEFAULT_OUTPUT_ROOT = Path("outputs/local_audits") / RUN_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit zero-step GIFT-to-SKINNY runtime-topology use with frozen "
            "checkpoints and validation caches."
        )
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gift-roots",
        nargs=2,
        type=Path,
        default=DEFAULT_GIFT_ROOTS,
        metavar=("SEED0", "SEED1"),
    )
    parser.add_argument(
        "--skinny-roots",
        nargs=2,
        type=Path,
        default=DEFAULT_SKINNY_ROOTS,
        metavar=("SEED0", "SEED1"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    progress_path.unlink(missing_ok=True)
    _append_progress(progress_path, "run_start", {"run_id": args.run_id})

    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        seed_rows = _evaluate_seed(args, seed)
        rows.extend(seed_rows)
        _append_progress(
            progress_path,
            "seed_done",
            {"run_id": args.run_id, "seed": seed, "rows": len(seed_rows)},
        )

    gate = adjudicate_zero_step_panel(run_id=args.run_id, rows=rows)
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "result_rows": len(rows),
        "expected_rows": len(EXPECTED_SEEDS) * len(EXPECTED_CONDITIONS),
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_gift_to_skinny_zero_step_topology_x1",
        "training_performed": False,
        "source_cipher": "GIFT-64 r6",
        "target_cipher": "SKINNY-64/64 r7",
        "target_validation_rows": 2048,
        "pairs_per_sample": 4,
        "seeds": list(EXPECTED_SEEDS),
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    render_zero_step_svg(gate, args.output_root / "curves.svg")
    _append_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _evaluate_seed(args: argparse.Namespace, seed: int) -> list[dict[str, Any]]:
    source_results_path = args.gift_roots[seed] / "results.jsonl"
    target_results_path = args.skinny_roots[seed] / "results.jsonl"
    source_rows = _read_jsonl(source_results_path)
    target_rows = _read_jsonl(target_results_path)
    target_validation_key = validate_target_result_rows(target_rows, seed)
    dataset, feature_path, label_path, metadata_path = _load_target_dataset(
        args.skinny_roots[seed], seed
    )
    return evaluate_zero_step_seed(
        seed=seed,
        source_rows=source_rows,
        dataset=dataset,
        feature_path=feature_path,
        label_path=label_path,
        metadata_path=metadata_path,
        target_results_path=target_results_path,
        target_validation_key=target_validation_key,
        batch_size=args.batch_size,
        device=args.device,
    )


def _load_target_dataset(
    target_root: Path, seed: int
) -> tuple[DiskDifferentialDataset, Path, Path, Path]:
    validation_root = target_root / "cache" / "skinny64" / "r7" / "validation"
    matches = list(validation_root.glob(f"seed-{10000 + seed}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one SKINNY validation cache for seed {seed}, got {len(matches)}"
        )
    cache_dir = matches[0]
    feature_path = cache_dir / "features.npy"
    label_path = cache_dir / "labels.npy"
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = DiskDifferentialDataset(
        features=np.load(feature_path, mmap_mode="r"),
        labels=np.load(label_path, mmap_mode="r"),
        metadata=metadata,
        cache_dir=cache_dir,
    )
    return dataset, feature_path, label_path, metadata_path


def render_zero_step_svg(gate: dict[str, Any], output: Path) -> None:
    condition_labels = (
        "正确源\n正确目标",
        "错误源\n正确目标",
        "正确源\n错误目标",
        "正确源\n无拓扑",
    )
    control_labels = ("错误源", "错误目标", "无拓扑")
    seed_colors = ("#059669", "#2563EB")
    seed_values = [
        [
            float(gate["seed_results"][str(seed)][field])
            for field in (
                "candidate_auc",
                "corrupted_source_auc",
                "corrupted_target_auc",
                "no_topology_auc",
            )
        ]
        for seed in EXPECTED_SEEDS
    ]
    margin_values = [
        [
            float(gate["seed_results"][str(seed)][field])
            for field in (
                "candidate_minus_source_auc",
                "candidate_minus_target_auc",
                "candidate_minus_no_topology_auc",
            )
        ]
        for seed in EXPECTED_SEEDS
    ]
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
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.3))
        figure.subplots_adjust(
            left=0.075, right=0.97, top=0.70, bottom=0.19, wspace=0.30
        )
        figure.suptitle(
            "创新1 X1：GIFT 权重零训练切换到 SKINNY 的拓扑审计",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.900,
            "同一主干参数形状；GIFT-64 r6 最佳权重；SKINNY-64/64 r7；每条样本含 4 对密文；每 seed 验证 2048 条。",
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        conclusion = {
            "pass": "两颗 seed 均通过：权重在零目标训练下仍有可归因的拓扑区分能力。",
            "hold": "控制会改变输出，但完整 AUC 门未过：仅证明拓扑敏感，不能称为零步跨密码迁移。",
            "fail": "协议或干预有效性检查失败：本结果不可解释，只允许修复审计。",
        }[gate["status"]]
        figure.text(
            0.075,
            0.845,
            conclusion,
            ha="left",
            va="top",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
            fontsize=10.3,
        )

        x = np.arange(len(condition_labels))
        width = 0.36
        all_auc_values = [value for values in seed_values for value in values]
        for seed, values in zip(EXPECTED_SEEDS, seed_values, strict=True):
            offset = (seed - 0.5) * width
            bars = axes[0].bar(
                x + offset,
                values,
                width,
                color=seed_colors[seed],
                label=f"seed{seed}",
            )
            axes[0].bar_label(
                bars,
                labels=[f"{value:.4f}" for value in values],
                padding=4,
                fontsize=8.6,
                rotation=90,
            )
        axes[0].axhline(
            0.52, color="#1D4ED8", linestyle="--", linewidth=1.4, label="候选门 0.52"
        )
        axes[0].axhline(
            0.50, color="#334155", linestyle=":", linewidth=1.2, label="随机基线 0.50"
        )
        axes[0].set_ylim(
            min(0.46, min(all_auc_values) - 0.025),
            max(0.56, max(all_auc_values) + 0.045),
        )
        axes[0].set_xticks(x, labels=condition_labels)
        axes[0].set_ylabel("SKINNY 验证 AUC")
        axes[0].set_title("四种源权重 / 目标拓扑组合", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(loc="upper right", frameon=False, ncols=2)

        margin_x = np.arange(len(control_labels))
        all_margins = [value for values in margin_values for value in values]
        for seed, values in zip(EXPECTED_SEEDS, margin_values, strict=True):
            offset = (seed - 0.5) * width
            bars = axes[1].bar(
                margin_x + offset,
                values,
                width,
                color=seed_colors[seed],
                label=f"seed{seed}",
            )
            axes[1].bar_label(
                bars,
                labels=[f"{value:+.4f}" for value in values],
                padding=4,
                fontsize=9.0,
            )
        axes[1].axhline(
            0.005,
            color="#DC2626",
            linestyle="--",
            linewidth=1.4,
            label="归因门 +0.005",
        )
        axes[1].axhline(0.0, color="#334155", linewidth=1.0)
        axes[1].set_ylim(
            min(-0.025, min(all_margins) - 0.015),
            max(0.05, max(all_margins) + 0.02),
        )
        axes[1].set_xticks(margin_x, labels=control_labels)
        axes[1].set_ylabel("候选 AUC - 控制 AUC")
        axes[1].set_title("候选相对三种控制的差值", loc="left", fontweight="bold")
        axes[1].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(loc="upper right", frameon=False, ncols=2)
        figure.text(
            0.075,
            0.055,
            "证据范围：本地零训练、同 checkpoint 跨密码推理审计；不是目标适配、正式规模、攻击、SOTA 或通用零样本迁移证明。",
            ha="left",
            va="bottom",
            color="#334155",
            fontsize=10.0,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
