"""
Import data from the Online Retail dataset into Weaviate.

https://archive.ics.uci.edu/dataset/352/online+retail
"""

import asyncio
import csv
import logging
import os
from copy import copy
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.client import WeaviateClient
from weaviate.collections.collection.sync import Collection
from weaviate.connect.helpers import connect_to_local


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASET_PATH = "/Users/marcelolotif/Downloads/Online Retail Sample 1k.csv"
COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
SEACHEABLE_TEXT_COLUMN = "Description"


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

    assert COLLECTION_NAME is not None, "WEAVIATE_COLLECTION_NAME env var must be set"

    return weaviate_client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.dynamic(
            distance_metric=VectorDistances.COSINE,
            threshold=10_000,
            flat=Configure.VectorIndex.flat(quantizer=Configure.VectorIndex.Quantizer.bq(cache=True)),
            hnsw=Configure.VectorIndex.hnsw(
                quantizer=Configure.VectorIndex.Quantizer.pq(segments=128, training_limit=50_000),
            ),
        ),
        properties=properties,
    )


async def get_embeddings(texts: list[str], embedding_client: AsyncOpenAI, model_name: str) -> list[list[float]]:
    """Get embeddings for a list of texts.

    Args:
        texts: The texts to get embeddings for.
        embedding_client: The embedding client.
        model_name: The model name to use.

    Returns
    -------
        The embeddings for the texts.
    """
    try:
        embeddings = await embedding_client.embeddings.create(input=texts, model=model_name)
    except OpenAIError as e:
        if hasattr(e, "status_code") and e.status_code == 400 and "context" in str(e):
            embeddings = await embedding_client.embeddings.create(
                input=[text[:10000] for text in texts], model=model_name
            )
        else:
            raise
    return [embedding.embedding for embedding in embeddings.data]


def list_of_dicts_to_dict_of_lists(data: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert a list of dictionaries to a dictionary of lists.

    Args:
        data: List of dictionaries with the same keys

    Returns
    -------
        Dictionary where each key maps to a list of values
    """
    if not data:
        return {}

    keys = data[0].keys()
    return {key: [item[key] for item in data] for key in keys}


async def producer(
    dataset: list[dict[str, Any]],
    batch_size: int,
    embedding_client: AsyncOpenAI,
    model_name: str,
    obj_queue: asyncio.Queue,
) -> None:
    """Create batches of objects from the dataset with the vector included."""
    for i in range(0, len(dataset), batch_size):
        batch = list_of_dicts_to_dict_of_lists(dataset[i : i + batch_size])

        objects: dict[str, list[Any]] = {}

        # Filter out None or empty strings from the batch
        # Get index of empty or None text entries
        null_indices = [i for i, text in enumerate(batch[SEACHEABLE_TEXT_COLUMN]) if text is None or text == ""]
        # Remove empty or None text entries from the batch
        if null_indices:
            for key in batch:
                objects[key.replace("-", "_").replace(" ", "_")] = [
                    v for i, v in enumerate(batch[key]) if i not in null_indices
                ]
        else:
            objects = batch

        # Get embeddings for the batch
        embeddings = await get_embeddings(objects[SEACHEABLE_TEXT_COLUMN], embedding_client, model_name)

        # Rename "id" to "id_" to avoid conflict with Weaviate's reserved field
        if "id" in objects:
            objects["id_"] = objects.pop("id")

        objects["vector"] = embeddings

        await obj_queue.put(objects)


async def consumer(collection: Collection, obj_queue: asyncio.Queue) -> None:
    """Consume objects from the queue and add them to Weaviate."""
    while True:
        objects: dict[str, list[Any]] = await obj_queue.get()
        if objects is None:
            break  # Exit signal

        with collection.batch.fixed_size(batch_size=len(objects), concurrent_requests=1) as batch:
            vectors = objects.pop("vector")
            for i in range(len(objects[SEACHEABLE_TEXT_COLUMN])):
                obj = {k.replace("-", "_").replace(" ", "_"): v[i] for k, v in objects.items()}
                batch.add_object(obj, vector=vectors[i])

            # Flush the batch to Weaviate
            batch.flush()  # type: ignore[attr-defined]

        obj_queue.task_done()


async def main():
    """Import data from the Online Retail dataset into Weaviate."""
    weaviate_client = connect_to_local(headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]})
    weaviate_collection = None
    data_rows = []

    with open(DATASET_PATH, "r") as file:
        csv_reader = csv.reader(file)
        column_names = None
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

    batch_size = 5
    max_concurrent = 50

    embedding_client = AsyncOpenAI(
        api_key=os.environ["EMBEDDING_API_KEY"],
        base_url=os.environ["EMBEDDING_BASE_URL"],
        timeout=30,
        max_retries=5,
    )

    # Orchestrate producer and consumers
    obj_queue = asyncio.Queue(maxsize=max_concurrent * 5)
    producer_task = asyncio.create_task(
        producer(data_rows, batch_size, embedding_client, EMBEDDING_MODEL_NAME, obj_queue)
    )
    # Create consumers to process the queue
    consumer_tasks = [asyncio.create_task(consumer(weaviate_collection, obj_queue)) for _ in range(max_concurrent)]

    try:
        # Wait for the producer to finish
        await producer_task

        # Signal consumers to stop
        for _ in range(max_concurrent):
            await obj_queue.put(None)

        # Wait for all consumers to finish
        await asyncio.gather(*consumer_tasks)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)

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


if __name__ == "__main__":
    asyncio.run(main())
