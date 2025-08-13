import argparse
import os
import sys
import time
from typing import List, Dict, Any

import pandas as pd
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from enums.enum import Directories


def get_qdrant() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL env is required")
    api_key = os.getenv("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient, name: str, dim: int, truncate: bool) -> None:
    if truncate:
        client.recreate_collection(collection_name=name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        return
    try:
        client.get_collection(name)
    except Exception:
        client.recreate_collection(collection_name=name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))


def iter_source_documents() -> List[Document]:
    """
    Load authoritative source data from staging/RAW to reconstruct documents.
    This mirrors DataPipeline loaders to avoid FAISS as a data source.
    """
    docs: List[Document] = []

    # Instagram
    insta_parquet = Directories.STAGING_INSTAGRAM.value
    if os.path.exists(insta_parquet):
        df = pd.read_parquet(insta_parquet)
        for _, row in df.iterrows():
            text = str(row.to_dict())
            md = {
                "namespace": "insta",
                "type": "instagram",
                "date": str(row.get("date", "")),
                "post_id": str(row.get("post_id", "")),
                "media_type": str(row.get("media_type", "")),
            }
            docs.append(Document(text=text, metadata=md))

    # Sales
    sales_parquet = Directories.STAGING_SALES.value
    if os.path.exists(sales_parquet):
        df = pd.read_parquet(sales_parquet)
        for _, row in df.iterrows():
            text = str(row.to_dict())
            md = {
                "namespace": "sales",
                "type": "sales",
                "date": str(row.get("date", "")),
                "product_id": str(row.get("product_id", "")),
            }
            docs.append(Document(text=text, metadata=md))

    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate documents to Qdrant via LlamaIndex")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--truncate", action="store_true", help="Recreate the Qdrant text collection")
    args = parser.parse_args()

    dim = int(os.getenv("EMBED_DIM", "768"))
    q_collection = os.getenv("QDRANT_COLLECTION_TEXT", "insightino_text")

    client = get_qdrant()
    ensure_collection(client, q_collection, dim, truncate=args.truncate)

    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base", normalize=True)  # type: ignore[arg-type]
    vector_store = QdrantVectorStore(client=client, collection_name=q_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = iter_source_documents()
    total = len(docs)
    print(f"Found {total} documents to migrate.")
    if args.dry_run:
        print("Dry-run complete. No writes performed.")
        return

    failures = 0
    t0 = time.time()
    for i in range(0, total, args.batch_size):
        batch = docs[i : i + args.batch_size]
        try:
            VectorStoreIndex.from_documents(batch, storage_context=storage_context, show_progress=False)
        except Exception as e:
            failures += len(batch)
            print(f"Batch {i//args.batch_size} failed: {e}")

    dur = time.time() - t0
    print(f"Migration done in {dur:.1f}s. total={total}, failures={failures}")


if __name__ == "__main__":
    main()


