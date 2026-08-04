import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_LATENCY_TARGET_SECONDS = 10.0
DEFAULT_COST_TARGET_USD = 0.05
DEFAULT_QUALITY_TARGET = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a Work Order evaluation dataset from the SME Q&A CSV."
    )
    parser.add_argument(
        "--input-file",
        default="SME-TestSet.csv",
        help="Path to the source SME CSV file.",
    )
    parser.add_argument(
        "--output-file",
        default="WO-TestSet-derived.csv",
        help="Path to the derived CSV dataset file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to convert.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_conversation(prompt: str, answer: str) -> str:
    return "\n".join(
        [
            f"User: I need help troubleshooting this ABB issue: {prompt}",
            f"Assistant: {answer}",
            "User: Please create a concise work order for a technician based only on this conversation.",
        ]
    )


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    input_path = (base_dir / args.input_file).resolve()
    output_path = (base_dir / args.output_file).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        raise RuntimeError("No rows found in the SME dataset.")

    fieldnames = [
        "id",
        "conversation",
        "source_question",
        "reference_answer",
        "safety_considerations",
        "quality_target",
        "latency_target_seconds",
        "cost_target_usd",
        "derived_from",
        "row_index",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(rows):
            prompt = clean_text(row.get("test_prompt"))
            answer = clean_text(row.get("expected_response"))

            if not prompt or not answer:
                continue

            writer.writerow(
                {
                    "id": f"wo-sme-{index:03}",
                    "conversation": build_conversation(prompt, answer),
                    "source_question": clean_text(row.get("test_prompt")),
                    "reference_answer": clean_text(row.get("expected_response")),
                    "safety_considerations": clean_text(
                        row.get("safety_considerations")
                    ),
                    "quality_target": DEFAULT_QUALITY_TARGET,
                    "latency_target_seconds": DEFAULT_LATENCY_TARGET_SECONDS,
                    "cost_target_usd": DEFAULT_COST_TARGET_USD,
                    "derived_from": input_path.name,
                    "row_index": index,
                }
            )

    print(f"Wrote {len(rows)} derived work order rows to: {output_path}")


if __name__ == "__main__":
    main()
