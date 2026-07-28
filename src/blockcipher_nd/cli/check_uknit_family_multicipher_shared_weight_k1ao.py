from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    CONFIG_PATH,
    ROOT,
    RUN_ID,
    load_and_validate_config,
    run_readiness,
)


DEFAULT_OUTPUT = ROOT / "outputs/local_readiness" / RUN_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the zero-training K1-AO multi-cipher shared-weight gate."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_and_validate_config(args.config)
    payload = run_readiness(config, output_root=args.output_root, project_root=ROOT)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["gate"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
