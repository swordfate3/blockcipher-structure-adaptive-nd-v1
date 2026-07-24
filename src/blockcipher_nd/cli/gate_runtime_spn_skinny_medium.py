from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import torch

from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_medium import (
    adjudicate_runtime_spn_skinny_medium,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate a frozen SKINNY general-GF(2) topology scale phase."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory for local adjudication artifacts. Defaults to RUN_ROOT for "
            "backward compatibility; use a separate directory for immutable remote archives."
        ),
    )
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Optional retrieved checkpoint directory. When supplied, all three "
            "checkpoint payloads must strictly replay the result rows."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("rtg2a", "rtg2b", "rtg3a"),
        default="rtg2a",
        help="Frozen SKINNY sample-scale phase; defaults to historical RTG2-A.",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Progress JSONL to append; defaults to RUN_ROOT/progress.jsonl.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Defer Matplotlib rendering until verified local retrieval.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root or args.run_root
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.run_root / "results.jsonl"
    rows = _read_jsonl(results_path)
    gate = adjudicate_runtime_spn_skinny_medium(
        run_id=args.run_id,
        rows=rows,
        expected_seed=args.seed,
        phase=args.phase,
    )
    checkpoint_verification = None
    if args.checkpoint_dir is not None:
        checkpoint_verification = _verify_checkpoint_payloads(
            rows,
            args.checkpoint_dir,
        )
        checkpoint_passed = checkpoint_verification["status"] == "pass"
        gate["protocol_checks"][
            "retrieved_checkpoint_payloads_match_results"
        ] = checkpoint_passed
        gate["checkpoint_evidence"] = checkpoint_verification
        if not checkpoint_passed:
            gate["status"] = "fail"
            gate["decision"] = {
                "rtg2a": "innovation1_rtg2a_skinny_medium_protocol_invalid",
                "rtg2b": "innovation1_rtg2b_skinny_scale_protocol_invalid",
                "rtg3a": "innovation1_rtg3a_skinny_formal_protocol_invalid",
            }[args.phase]
            gate["next_action"] = (
                "repair retrieved checkpoint evidence without changing data, "
                "training, models, or thresholds"
            )
    train_samples_per_class = int(gate["samples_per_class"])
    validation_rows = int(gate["validation_rows"])
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "results": str(results_path),
    }
    summary = {
        "run_id": args.run_id,
        "task": f"innovation1_{args.phase}_skinny_general_gf2_scale_replication",
        "training_performed": True,
        "train_samples_per_class": train_samples_per_class,
        "train_samples_total": int(gate["train_rows"]),
        "validation_samples_per_class": validation_rows // 2,
        "validation_samples_total": validation_rows,
        "pairs_per_sample": 4,
        "epochs": 5,
        "seed": args.seed,
        "gate": gate,
    }
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "summary.json", summary)
    if checkpoint_verification is not None:
        _write_json(
            output_root / "checkpoint-verification.json",
            checkpoint_verification,
        )
    _write_history_csv(rows, output_root / "history.csv")
    if not args.no_plot:
        render_skinny_medium_svg(gate, output_root / "curves.svg")
    _append_progress(
        args.progress or output_root / "progress.jsonl",
        "gate_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
            "plot_deferred": bool(args.no_plot),
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _verify_checkpoint_payloads(
    rows: list[dict[str, Any]],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    expected_names: list[str] = []
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        model_name = str(row.get("model", ""))
        raw_path = row.get("training", {}).get("checkpoint_output")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"{model_name}: missing checkpoint_output")
            continue
        filename = PureWindowsPath(raw_path).name
        expected_names.append(filename)
        path = checkpoint_dir / filename
        checks: dict[str, bool] = {"file_exists": path.is_file()}
        entry: dict[str, Any] = {
            "model": model_name,
            "filename": filename,
            "checks": checks,
        }
        if not path.is_file():
            errors.append(f"{model_name}: missing {filename}")
            entries.append(entry)
            continue
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
            checks["payload_shape"] = bool(
                isinstance(payload, dict) and isinstance(state_dict, dict)
            )
            if not checks["payload_shape"]:
                raise ValueError("checkpoint payload lacks state_dict")
            model = build_model(
                model_name,
                input_bits=int(row.get("training", {}).get("input_bits", 512)),
                hidden_bits=64,
                pair_bits=int(row.get("training", {}).get("pair_bits", 128)),
                structure=str(row.get("structure", "SPN")),
                model_options=row.get("training", {}).get("model_options", {}),
            )
            model.load_state_dict(state_dict, strict=True)
            checks["strict_state_dict_load"] = True
            checks["parameter_count"] = (
                sum(parameter.numel() for parameter in model.parameters())
                == row.get("parameter_count")
            )
            checks["history_exact"] = payload.get("history") == row.get("history")
            checks["final_metrics_exact"] = (
                payload.get("final_metrics") == row.get("metrics")
            )
            checkpoint_metadata = payload.get("metadata", {})
            result_metadata = row.get("training", {})
            required_checkpoint_metadata = {
                "epochs",
                "batch_size",
                "learning_rate",
                "optimizer",
                "weight_decay",
                "lr_scheduler",
                "checkpoint_metric",
                "restore_best_checkpoint",
                "train_eval_interval",
                "loss",
                "best_epoch",
                "best_checkpoint_metric",
                "selected_checkpoint",
                "epochs_ran",
                "seed",
                "device",
                "checkpoint_output",
            }
            checks["metadata_core_complete"] = required_checkpoint_metadata.issubset(
                checkpoint_metadata
            )
            checks["metadata_payload_is_result_subset"] = all(
                result_metadata.get(key) == value
                for key, value in checkpoint_metadata.items()
            )
            checks["selected_best_checkpoint"] = bool(
                payload.get("metadata", {}).get("selected_checkpoint") == "best"
                and payload.get("metadata", {}).get("best_checkpoint_metric")
                == row.get("metrics", {}).get("auc")
            )
        except Exception as exc:
            checks.setdefault("strict_state_dict_load", False)
            errors.append(f"{model_name}: {type(exc).__name__}: {exc}")
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            errors.append(f"{model_name}: failed checks {failed_checks}")
        entries.append(entry)

    actual_names = sorted(path.name for path in checkpoint_dir.glob("*.pt"))
    file_set_exact = sorted(expected_names) == actual_names and len(actual_names) == 3
    if not file_set_exact:
        errors.append(
            f"checkpoint file set mismatch: expected={sorted(expected_names)} "
            f"actual={actual_names}"
        )
    return {
        "status": "pass" if len(entries) == 3 and file_set_exact and not errors else "fail",
        "checkpoint_dir": str(checkpoint_dir),
        "expected_files": sorted(expected_names),
        "actual_files": actual_names,
        "file_set_exact": file_set_exact,
        "entries": entries,
        "errors": errors,
    }


def render_skinny_medium_svg(gate: dict[str, Any], output: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt

    roles = ("true", "corrupted", "independent")
    labels = ("正确 GF(2) 拓扑", "确定性打乱拓扑", "无线性拓扑")
    values = [float(gate["aucs"][role]) for role in roles]
    margins = [
        float(gate["margins"]["true_minus_corrupted"]),
        float(gate["margins"]["true_minus_independent"]),
    ]
    seed = int(gate["seed"])
    phase = str(gate.get("phase", "rtg2a"))
    phase_label = phase.upper()
    samples_per_class = int(gate.get("samples_per_class", 65_536))
    validation_per_class = int(gate.get("validation_rows", 65_536)) // 2
    status = str(gate["status"])
    if status == "pass":
        if phase == "rtg2a":
            conclusion = (
                "正确拓扑同时通过信号门和两项归因门；可进入相同协议的 seed1 复验。"
                if seed == 0
                else "正确拓扑再次通过信号门和两项归因门；下一步汇总两颗种子后裁决扩样。"
            )
        else:
            conclusion = (
                "正确拓扑在 262144/类再次通过三门；可准备相同协议的 seed1 复验。"
                if seed == 0
                else "正确拓扑在 262144/类两颗种子通过；下一步汇总后再裁决正式规模。"
            )
    else:
        conclusion = {
            "hold": "中等规模优势未满足冻结门槛；停止扩样并审计方差或训练动态。",
            "fail": "协议证据不完整或不一致；仅修复失败检查，不解释 AUC。",
        }[status]
    status_color = {"pass": "#047857", "hold": "#B45309", "fail": "#B42318"}[status]
    finite_values = [value for value in values if math.isfinite(value)] or [0.5]
    finite_margins = [value for value in margins if math.isfinite(value)] or [0.0]

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
        figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.6))
        figure.subplots_adjust(
            left=0.075, right=0.97, top=0.72, bottom=0.18, wspace=0.32
        )
        figure.suptitle(
            f"创新1 {phase_label}：SKINNY 中等规模 GF(2) 拓扑复验",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.895,
            (
                f"seed{seed}；7轮；差分 0x2000；每条样本 4 个密文对；"
                f"训练 {samples_per_class}/class，验证 {validation_per_class}/class，5 epochs。"
            ),
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        figure.text(
            0.075,
            0.845,
            conclusion,
            ha="left",
            va="top",
            color=status_color,
            fontweight="bold",
            fontsize=10.3,
        )

        x = np.arange(3)
        bars = axes[0].bar(
            x,
            values,
            color=("#059669", "#DC2626", "#64748B"),
            width=0.62,
        )
        axes[0].bar_label(
            bars,
            labels=[f"{value:.6f}" for value in values],
            padding=5,
        )
        axes[0].axhline(
            0.55,
            color="#1D4ED8",
            linestyle="--",
            linewidth=1.4,
            label="正确拓扑信号门 0.55",
        )
        axes[0].axhline(
            0.5,
            color="#334155",
            linestyle=":",
            linewidth=1.2,
            label="随机基线 0.50",
        )
        axes[0].set_ylim(
            min(0.48, min(finite_values) - 0.015),
            max(0.56, max(finite_values) + 0.025),
        )
        axes[0].set_xticks(x, labels=labels)
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("同预算的三种运行时拓扑", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(loc="upper right", frameon=False)

        margin_x = np.arange(2)
        margin_bars = axes[1].bar(
            margin_x,
            margins,
            color=["#059669" if value >= 0.005 else "#DC2626" for value in margins],
            width=0.58,
        )
        axes[1].bar_label(
            margin_bars,
            labels=[f"{value:+.6f}" for value in margins],
            padding=5,
        )
        axes[1].axhline(
            0.005,
            color="#1D4ED8",
            linestyle="--",
            linewidth=1.4,
            label="归因门 +0.005",
        )
        axes[1].axhline(0.0, color="#334155", linewidth=1.0)
        axes[1].set_ylim(
            min(-0.01, min(finite_margins) - 0.01),
            max(0.02, max(finite_margins) + 0.015),
        )
        axes[1].set_xticks(
            margin_x, labels=("正确 - 打乱拓扑", "正确 - 无拓扑")
        )
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_title("正确拓扑相对控制组的优势", loc="left", fontweight="bold")
        axes[1].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(loc="upper left", frameon=False)
        figure.text(
            0.075,
            0.055,
            (
                f"证据范围：SKINNY-64/64 7轮、{samples_per_class}/class 中等规模架构/协议复验；"
                "不是正式规模、论文复现、攻击、SOTA 或突破。"
            ),
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_history_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = (
        "model",
        "seed",
        "epoch",
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_auc",
        "val_loss",
        "val_accuracy",
        "val_auc",
        "learning_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for history_row in row.get("history", []):
                writer.writerow(
                    {
                        "model": row.get("model"),
                        "seed": row.get("seed"),
                        **history_row,
                    }
                )


def _append_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
