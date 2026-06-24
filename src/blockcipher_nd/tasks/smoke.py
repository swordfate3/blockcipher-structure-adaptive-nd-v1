from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.ciphers import Speck32_64
from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.models.baseline import MlpDistinguisher
from blockcipher_nd.training import TrainingConfig, train_binary_classifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small innovation-one neural distinguisher smoke experiment."
    )
    parser.add_argument("--cipher", choices=["speck32"], default="speck32")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--input-difference", type=lambda value: int(value, 0), default=0x0040)
    parser.add_argument("--samples-per-class", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-bits", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", default="outputs/innovation_one_smoke_result.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cipher = Speck32_64(rounds=args.rounds, key=0x1918111009080100)
    train_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=args.input_difference,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
    )
    validation_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=args.input_difference,
            samples_per_class=max(16, args.samples_per_class // 2),
            seed=args.seed + 1,
        )
    )
    model = MlpDistinguisher(
        input_bits=train_dataset.features.shape[1],
        hidden_bits=args.hidden_bits,
    )
    result = train_binary_classifier(
        model,
        train_dataset,
        validation_dataset,
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
    )
    payload: dict[str, Any] = {
        "cipher": cipher.name,
        "structure": cipher.structure,
        "model": model.__class__.__name__,
        "rounds": args.rounds,
        "input_difference": args.input_difference,
        "samples_per_class": args.samples_per_class,
        "metrics": result.final_metrics,
        "history": result.history,
        "training": result.metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "accuracy={accuracy:.4f} auc={auc:.4f} advantage={advantage:.4f} output={output}".format(
            output=output,
            **result.final_metrics,
        )
    )


if __name__ == "__main__":
    main()
