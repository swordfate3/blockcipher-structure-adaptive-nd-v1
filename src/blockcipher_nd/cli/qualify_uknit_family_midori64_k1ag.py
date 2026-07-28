from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.tasks.innovation1.uknit_family_midori64_qualification_k1ag import (
    CONFIG_PATH,
    ROOT,
    RUN_ID,
    load_and_validate_config,
    run_qualification,
    write_qualification_artifacts,
)


DEFAULT_OUTPUT = ROOT / "outputs/local_readiness" / RUN_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the zero-training Midori64 K1-AG qualification gate."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_and_validate_config(args.config)
    payload = run_qualification(config, project_root=ROOT)
    write_qualification_artifacts(payload, args.output_root)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["gate"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
