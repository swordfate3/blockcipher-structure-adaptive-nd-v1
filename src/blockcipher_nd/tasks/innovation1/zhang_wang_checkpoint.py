from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_AUDIT_ROOT = Path("/tmp/zhang_wang2022_code_audit")
DEFAULT_CHECKPOINT = Path("DATA_Nm_good_trained_nets/present_best_7r_pairs16_distinguisher.h5")
DEFAULT_HISTORY = Path("present_hist7r_pairs16_nm.p")
REFERENCE_ACCURACY = 0.7205


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate or audit the official Zhang/Wang 2022 PRESENT Case2 r7/m16 "
            "checkpoint. TensorFlow is imported only when --run-eval is used."
        )
    )
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--diff", default="0x9")
    parser.add_argument(
        "--raw-pair-count",
        type=int,
        default=160_000,
        help=(
            "Raw basic-pair count passed to the official make_dataset_with_group_size. "
            "Grouped evaluation rows are raw_pair_count / pairs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_under(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def dependency_status() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "h5py": importlib.util.find_spec("h5py") is not None,
    }


def summarize_history(history_path: Path) -> dict[str, Any]:
    if not history_path.exists():
        return {"exists": False, "path": str(history_path)}
    with history_path.open("rb") as handle:
        history = pickle.load(handle)
    val_acc = [float(value) for value in history.get("val_acc", [])]
    val_loss = [float(value) for value in history.get("val_loss", [])]
    summary: dict[str, Any] = {
        "exists": True,
        "path": str(history_path),
        "keys": sorted(str(key) for key in history.keys()),
        "epochs": len(val_acc),
    }
    if val_acc:
        best_index = int(np.argmax(val_acc))
        summary.update(
            {
                "best_val_acc": val_acc[best_index],
                "best_val_acc_epoch": best_index + 1,
                "best_val_acc_minus_reference": round(val_acc[best_index] - REFERENCE_ACCURACY, 10),
            }
        )
    if val_loss:
        best_loss_index = int(np.argmin(val_loss))
        summary.update(
            {
                "best_val_loss": val_loss[best_loss_index],
                "best_val_loss_epoch": best_loss_index + 1,
            }
        )
    return summary


def build_audit_summary(
    *,
    audit_root: Path,
    checkpoint_path: Path,
    history_path: Path,
    rounds: int,
    pairs: int,
    diff: int,
    raw_pair_count: int,
) -> dict[str, Any]:
    grouped_rows = raw_pair_count // pairs if pairs else None
    return {
        "reference": {
            "paper": "Zhang/Wang 2022 Table 4",
            "cipher": "PRESENT-80",
            "task": "7-round Case2, m=16",
            "accuracy": REFERENCE_ACCURACY,
        },
        "protocol": {
            "rounds": rounds,
            "pairs": pairs,
            "raw_pair_count": raw_pair_count,
            "grouped_eval_rows": grouped_rows,
            "diff": hex(diff),
            "sample_source": "official present.py make_dataset_with_group_size",
            "checkpoint_source": "official DATA_Nm_good_trained_nets best val_loss checkpoint",
        },
        "audit_root": str(audit_root),
        "checkpoint": {
            "path": str(checkpoint_path),
            "exists": checkpoint_path.exists(),
            "size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
        },
        "history": summarize_history(history_path),
        "dependencies": dependency_status(),
    }


def run_checkpoint_eval(
    *,
    audit_root: Path,
    checkpoint_path: Path,
    rounds: int,
    pairs: int,
    diff: int,
    raw_pair_count: int,
    batch_size: int,
) -> dict[str, Any]:
    missing = [name for name, present in dependency_status().items() if not present]
    if missing:
        return {
            "status": "missing_dependencies",
            "missing_dependencies": missing,
            "note": "Install TensorFlow/Keras and h5py in an isolated reproduction environment before eval.",
        }
    if not checkpoint_path.exists():
        return {"status": "missing_checkpoint", "checkpoint": str(checkpoint_path)}

    sys.path.insert(0, str(audit_root))
    import present as official_present  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore

    model = load_model(checkpoint_path, compile=False)
    x_eval, y_eval = official_present.make_dataset_with_group_size(
        n=raw_pair_count,
        nr=rounds,
        diff=diff,
        group_size=pairs,
    )
    scores = model.predict(x_eval, batch_size=batch_size, verbose=0).reshape(-1)
    labels = np.asarray(y_eval).reshape(-1)
    predictions = scores >= 0.5
    accuracy = float(np.mean(predictions == labels))
    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    tpr = float(np.sum(predictions[labels == 1]) / n1) if n1 else None
    tnr = float(np.sum(~predictions[labels == 0]) / n0) if n0 else None
    mse = float(np.mean((labels - scores) ** 2))
    return {
        "status": "evaluated",
        "accuracy": accuracy,
        "accuracy_minus_reference": round(accuracy - REFERENCE_ACCURACY, 10),
        "mse": mse,
        "tpr": tpr,
        "tnr": tnr,
        "rows": int(len(labels)),
        "positive_rows": n1,
        "negative_rows": n0,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit_root = args.audit_root.resolve()
    checkpoint_path = resolve_under(audit_root, args.checkpoint).resolve()
    history_path = resolve_under(audit_root, args.history).resolve()
    diff = int(str(args.diff), 0)
    summary = build_audit_summary(
        audit_root=audit_root,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        rounds=args.rounds,
        pairs=args.pairs,
        diff=diff,
        raw_pair_count=args.raw_pair_count,
    )
    if args.run_eval:
        summary["evaluation"] = run_checkpoint_eval(
            audit_root=audit_root,
            checkpoint_path=checkpoint_path,
            rounds=args.rounds,
            pairs=args.pairs,
            diff=diff,
            raw_pair_count=args.raw_pair_count,
            batch_size=args.batch_size,
        )
    else:
        summary["evaluation"] = {
            "status": "not_run",
            "note": "Pass --run-eval inside a TensorFlow/Keras environment to evaluate the checkpoint.",
        }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
