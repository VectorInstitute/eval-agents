"""Fetch all PDFs from the dataset and store them in the files directory.

Example
-------
$ python -m implementations.legislative_content_extraction.fetch_pdfs
"""

import json
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

IMPL_DIR = Path(__file__).resolve().parent
FILES_DIR = IMPL_DIR / "files"
DATASET_JSON = IMPL_DIR / "legislative_content_extraction_dataset.json"


def fetch_all_pdfs() -> None:
    """Download every PDF listed in the dataset that isn't already cached."""
    with open(DATASET_JSON) as f:
        dataset = json.load(f)

    total = len(dataset)
    skipped = 0
    downloaded = 0
    failed = 0

    for record in dataset:
        record_id = record.get("record_id", "")
        pdf_link = record.get("pdf_file_link", "")
        pdf_name = record.get("pdf_file_name", f"{record_id}.pdf")

        if not record_id or not pdf_link:
            print(f"  SKIP  {record_id or '(no id)'}: missing pdf_file_link")
            skipped += 1
            continue

        content_dir = FILES_DIR / record_id
        local_path = content_dir / pdf_name

        if local_path.exists():
            print(f"  OK    {record_id}: already exists")
            skipped += 1
            continue

        content_dir.mkdir(parents=True, exist_ok=True)

        try:
            req = urllib.request.Request(
                pdf_link,
                headers={"User-Agent": "Mozilla/5.0 (legislative-content-extraction-agent)"},
            )
            with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310
                local_path.write_bytes(response.read())
            print(f"  DONE  {record_id}: downloaded {pdf_name}")
            downloaded += 1
        except (HTTPError, URLError, OSError) as e:
            print(f"  FAIL  {record_id}: {e}")
            failed += 1

    print(f"\nTotal: {total} | Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    fetch_all_pdfs()
