import os
from qdrant_client import QdrantClient


def main() -> None:
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL env is required")
    api_key = os.getenv("QDRANT_API_KEY") or None
    collection = os.getenv("QDRANT_COLLECTION_TEXT", "insightino_text")

    client = QdrantClient(url=url, api_key=api_key)
    info = client.get_collection(collection)
    count = client.count(collection).count
    print({"collection": collection, "status": info.status, "points": count})


if __name__ == "__main__":
    main()


