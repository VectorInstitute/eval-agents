import json
import csv

INPUT_FILE = "2023_processed.json"
OUTPUT_FILE = "2023_data.csv"
COLUMNS = ["title", "maintext", "description", "mentioned_companies", "named_entities"]

def load_json(filepath):
    with open(filepath, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def transform_record(record):
    return {
        "title": record.get("title", ""),
        "maintext": record.get("maintext", ""),
        "description": record.get("description", ""),
        "mentioned_companies": record.get("mentioned_companies", []),
        "named_entities": record.get("named_entities", []),
    }

def write_csv(records, filepath):
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

def main():
    print(f"Loading {INPUT_FILE}...")
    data = load_json(INPUT_FILE)

    if isinstance(data, dict):
        data = [data]

    print(f"Transforming {len(data)} records...")
    transformed = [transform_record(r) for r in data]

    filtered = [
        r for r in transformed
        if any(r[col] for col in COLUMNS)
    ]

    print(f"Writing {len(filtered)} records to {OUTPUT_FILE}...")
    write_csv(filtered, OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()