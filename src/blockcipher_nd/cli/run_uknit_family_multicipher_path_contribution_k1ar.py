from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_path_contribution_k1ar import (
    CONFIG_PATH,
    run_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the zero-training K1-AR shared path-contribution audit."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_audit(
        config_path=args.config,
        output_root=args.output_root,
        device=args.device,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if report["validation"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args"]
