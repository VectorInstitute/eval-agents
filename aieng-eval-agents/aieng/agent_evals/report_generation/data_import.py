"""
Import data from the Online Retail dataset into Weaviate.

https://archive.ics.uci.edu/dataset/352/online+retail
"""

import csv
import logging
import os
from copy import copy
from datetime import datetime
from typing import Any

from weaviate.classes.config import Configure, DataType, Property
from weaviate.client import WeaviateClient
from weaviate.collections.collection.sync import Collection
from weaviate.connect.helpers import connect_to_local


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASET_PATH = "/Users/marcelolotif/Downloads/Online Retail Sample 1k.csv"
COLLECTION_NAME = "OnlineRetail"


def is_numeric(value: str) -> bool:
    """Check if a string represents a numeric value.

    Args:
        value: The value to check.

    Returns
    -------
        True if the value is a numeric value, False otherwise.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def type_for_column_name(column_name: str) -> DataType:
    """Get the data type for a column name.

    Args:
        column_name: The name of the column.

    Returns
    -------
        The data type for the column.
    """
    if column_name in ["InvoiceNo", "Quantity", "CustomerID"]:
        return DataType.INT
    if column_name == "UnitPrice":
        return DataType.NUMBER
    if column_name == "InvoiceDate":
        return DataType.DATE
    return DataType.TEXT


def make_data_row(column_names: list[str], row: list[str]) -> dict[str, Any] | None:
    """Make a data row from a list of column names and a list of values.

    Args:
        column_names: The names of the columns.
        row: The values of the row.

    Returns
    -------
        The data row, or None if the row is invalid.
    """
    if len(row) == 0:
        logger.warning(f"Skipping row because it has no values: {row}")
        return None

    data_row: dict[str, Any] = {}
    for i in range(len(row)):
        column_name = column_names[i]
        value = row[i]

        try:
            data_type = type_for_column_name(column_name)
            if data_type == DataType.INT:
                data_row[column_name] = int(value)
            elif data_type == DataType.NUMBER:
                data_row[column_name] = float(value)
            elif data_type == DataType.DATE:
                data_row[column_name] = datetime.strptime(value, "%m/%d/%y %H:%M")
            else:
                data_row[column_name] = str(value)
        except Exception:
            logger.exception(f"Skipping row because of error: {row}")
            return None

    return data_row


def make_collection_with_column_names(weaviate_client: WeaviateClient, column_names: list[str]) -> Collection:
    """Make a collection with the column names of the rows.

    Args:
        weaviate_client: The Weaviate client.
        column_names: The names of the columns.

    Returns
    -------
        The weaviate collection object.
    """
    properties = []
    for column_name in column_names:
        properties.append(Property(name=column_name, data_type=type_for_column_name(column_name)))

    return weaviate_client.collections.create(
        name=COLLECTION_NAME,
        vector_config=Configure.Vectors.text2vec_openai(),
        properties=properties,
    )


if __name__ == "__main__":
    weaviate_client = connect_to_local(headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]})
    weaviate_collection = None

    with open(DATASET_PATH, "r") as file:
        csv_reader = csv.reader(file)

        column_names = None
        data_rows = []
        first_row = True
        logger.info("Collecting data points...")

        for r in csv_reader:
            row = copy(r)
            if first_row:
                first_row = False

                column_names = row

                if weaviate_client.collections.exists(COLLECTION_NAME):
                    weaviate_collection = weaviate_client.collections.use(COLLECTION_NAME)
                else:
                    weaviate_collection = make_collection_with_column_names(weaviate_client, column_names)

            else:
                if not is_numeric(row[0]):
                    logger.warning(f"Skipping row because it seems to be invalid: {row}")
                    continue

                assert column_names is not None, "Column names should not be None"
                data_row = make_data_row(column_names, row)
                if data_row is not None:
                    data_rows.append(data_row)

        assert weaviate_collection is not None, "Weaviate collection should not be None"

        logger.info("Importing data points...")
        with weaviate_collection.batch.fixed_size(batch_size=100, concurrent_requests=2) as batch:
            for data_row in data_rows:
                batch.add_object(properties=data_row)

        # Check for failed objects
        failed_objects = weaviate_collection.batch.failed_objects
        if failed_objects:
            print(f"Number of failed imports: {len(failed_objects)}")
            for failed in failed_objects[:3]:  # Show first 3 failures
                print(f"Failed object: {failed}")

        # Verify client-side batch import
        result = weaviate_collection.aggregate.over_all(total_count=True)
        logger.info(f"Client-side batch had {len(failed_objects)} failures")
        logger.info(f"Expected {len(data_rows)} objects, got {result.total_count}")
        logger.info(f"✓ Client-side batch: {result.total_count} objects imported successfully")

        weaviate_client.close()
