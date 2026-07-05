from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.evaluation.neural_ensemble import (
    evaluate_frozen_score_ensemble,
    load_score_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed-rule ensemble over frozen neural score artifacts."
    )
    parser.add_argument("--artifacts", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = [load_score_artifact(path) for path in args.artifacts]
    summary = evaluate_frozen_score_ensemble(artifacts)
    summary["artifact_dirs"] = [str(path) for path in args.artifacts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
