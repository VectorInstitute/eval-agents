import weaviate
import openai
import os
from dotenv import load_dotenv
import json
from weaviate.classes.init import Auth

load_dotenv()

class Weaviate:
    def __init__(self, data_name=os.getenv("COLLECTION_NAME")):
        self.client = None
        self.data_name = data_name

        # Setup OpenAI client via Cloudflare
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL")
        )

    async def create_client(self) -> weaviate.WeaviateClient:
        cluster_url = os.getenv("WEAVIATE_HTTP_HOST")
        api_key = os.getenv("WEAVIATE_API_KEY")

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=Auth.api_key(api_key)
        )

        return client

    async def ensure_connected(self):
        if self.client is None:
            self.client = await self.create_client()

    async def get_knowledge(self, query: str) -> str:
        await self.ensure_connected()

        try:
            # Generate embedding
            embedding = self.openai_client.embeddings.create(
                model= os.getenv("EMBEDDING_MODEL_NAME"), input=query
            )

            # Perform hybrid search
            collection = self.client.collections.get(self.data_name)
            response = collection.query.hybrid(
                query=query,
                vector=embedding.data[0].embedding,
                return_metadata=["score"]
            )

            if not response.objects:
                return "No results found."

            # Format results
            formatted_results = [
                {
                    "Document Name": obj.properties.get("document_Name", ""),
                    "URL": obj.properties.get("uRL", ""),
                    "Page Number": obj.properties.get("page_number", ""),
                    "Full Text": obj.properties.get("full_text", "")
                }
                for obj in response.objects
            ]

            return json.dumps(formatted_results, indent=2, sort_keys=True)

        except Exception as e:
            return f"Search error: {e}"

    async def close(self):
        if self.client:
            self.client.close()
