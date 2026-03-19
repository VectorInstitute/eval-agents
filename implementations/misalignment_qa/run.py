#!/usr/bin/env python3
"""CLI entrypoint for misalignment QA experiment configs."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys

# Allow running as a plain script: `python implementations/misalignment_qa/run.py ...`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implementations.misalignment_qa.experiment import load_experiment_config, run_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Langfuse-backed misalignment experiment runner (YAML config).")
    parser.add_argument("--config", required=True, type=str, help="Path to the YAML experiment config.")
    parser.add_argument("--log-level", default="INFO", type=str, help="Logging level (e.g. INFO, DEBUG).")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_experiment_config(Path(args.config))
    asyncio.run(run_experiment_config(config))


if __name__ == "__main__":
    main()

