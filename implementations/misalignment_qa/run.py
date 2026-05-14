#!/usr/bin/env python3
"""CLI entrypoint for misalignment QA experiment configs."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


# Allow running as a plain script: `python implementations/misalignment_qa/run.py ...`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implementations.misalignment_qa.experiment import load_experiment_config, run_experiment_config  # noqa: E402


def main() -> None:
    """Parse CLI arguments and run the specified experiment config."""
    parser = argparse.ArgumentParser(description="Langfuse-backed misalignment experiment runner (YAML config).")
    parser.add_argument("--config", required=True, type=str, help="Path to the YAML experiment config.")
    parser.add_argument(
        "--variant-id",
        action="append",
        default=None,
        help="Variant id to run. Repeat to run multiple variants; omitted runs all variants in the config.",
    )
    parser.add_argument("--log-level", default="INFO", type=str, help="Logging level (e.g. INFO, DEBUG).")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_experiment_config(Path(args.config))
    variant_ids = set(args.variant_id) if args.variant_id else None
    asyncio.run(run_experiment_config(config, variant_ids=variant_ids))


if __name__ == "__main__":
    main()
