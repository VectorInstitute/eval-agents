import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluations against a Langfuse dataset"
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Name of the Langfuse dataset",
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
    run_eval_path = base_dir / "run_eval.py"

    if not run_eval_path.is_file():
        raise FileNotFoundError(
            f"Could not find run_eval.py at: {run_eval_path}"
        )

    cmd = [
        sys.executable,
        "-u",  # Unbuffered stdout/stderr so logs appear immediately.
        str(run_eval_path),
        "--langfuse_dataset_name",
        args.dataset_name,
        "--run_name",
        args.run_name,
    ]

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    print("Running:", " ".join(cmd), flush=True)

    try:
        subprocess.run(
            cmd,
            cwd=base_dir,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"\nEvaluation run failed with exit code {exc.returncode}.",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()