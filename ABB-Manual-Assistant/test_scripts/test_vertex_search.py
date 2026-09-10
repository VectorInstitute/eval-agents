import asyncio
from pprint import pprint

from configs import Configs
from utils.tools.vertex_search import vertex_search


async def main():
    print("Loading config...")
    config = Configs()

    print("Config check:")
    print("vertex_datastore_id:", config.vertex_datastore_id)
    print("google_cloud_location:", config.google_cloud_location)
    print("default_worker_model:", config.default_worker_model)
    print("default_temperature:", config.default_temperature)

    print("\nRunning Vertex Search test...")
    result = await vertex_search("What does error code 10039 mean?")

    pprint(result)


if __name__ == "__main__":
    asyncio.run(main())
