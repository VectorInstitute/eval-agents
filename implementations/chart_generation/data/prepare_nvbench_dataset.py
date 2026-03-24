import json
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

NVBENCH_PATH = Path("./nvBench/NVBench.json")
DB_PATH = Path("./implementations/chart_generation/data/hr_1.sqlite")
OUTPUT_PATH = Path("./implementations/chart_generation/data/NVBenchHR1ChartEval.json")


def prepare_dataset(
    nvbench_path: Path = NVBENCH_PATH,
    db_path: Path = DB_PATH,
    output_path: Path = OUTPUT_PATH,
) -> list[dict]:
    """Filter nvBench to hr_1 only and write a Langfuse-compatible JSON file."""
    with open(nvbench_path) as f:
        raw = json.load(f)

    output_records = []
    i = 0
    for key, item in raw.items():
        if item["db_id"] != "hr_1":
            continue

        gold_sql = item["vis_query"]["data_part"]["sql_part"].strip()
        chart_type = item["chart"].lower()
        x_name = item["vis_obj"]["x_name"]
        y_name = item["vis_obj"]["y_name"]

        for nl_query in item["nl_queries"]:
            i += 1
            output_records.append(
                {
                    "id": str(i),
                    "input": nl_query.strip(),
                    "expected_output": {
                        "final_chart": {
                            "chart_type": chart_type,
                            "x_name": x_name,
                            "y_name": y_name,
                            "gold_sql": gold_sql,
                            "filename": f"chart_{i}.png",
                        },
                        "trajectory": {
                            "actions": [
                                "get_schema_info",
                                "execute",
                                "write_chart",
                                "final_response",
                            ],
                            "description": [
                                "Inspect the hr_1 database schema",
                                f"Execute SQL to retrieve {x_name} vs {y_name}",
                                f"Generate a {chart_type} chart of the retrieved data",
                                "Return a downloadable link to the PNG chart",
                            ],
                        },
                    },
                    "metadata": {"nvbench_key": key},
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_records, f, indent=2)

    logger.info(f"Wrote {len(output_records)} records to {output_path}")
    return output_records


if __name__ == "__main__":
    prepare_dataset(NVBENCH_PATH, DB_PATH, OUTPUT_PATH)
