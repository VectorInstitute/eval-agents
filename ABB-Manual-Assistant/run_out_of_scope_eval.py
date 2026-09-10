#!/usr/bin/env python3
"""Run the ABB Manual Assistant evaluation against the out-of-scope query dataset.

Usage example:
    python3 run_out_of_scope_eval.py --run-name out_of_scope_01
"""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET_NAME = "Out_of_Scope_Queries"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ABB Manual Assistant evaluation against the "
            "out-of-scope query dataset"
        )
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=f"Langfuse dataset name to evaluate (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name for this evaluation run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    run_eval_from_dataset = base_dir / "run_eval_from_dataset.py"

    if not run_eval_from_dataset.is_file():
        raise FileNotFoundError(
            f"Could not find run_eval_from_dataset.py at: {run_eval_from_dataset}"
        )

    cmd = [
        sys.executable,
        "-u",
        str(run_eval_from_dataset),
        "--dataset-name",
        args.dataset_name,
        "--run-name",
        args.run_name,
    ]

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    print("Running:", " ".join(cmd), flush=True)

    subprocess.run(
        cmd,
        cwd=base_dir,
        check=True,
    )


if __name__ == "__main__":
    main()
