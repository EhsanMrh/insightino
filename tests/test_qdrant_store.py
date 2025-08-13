import os
import pytest

from llama_index.core import Document

from src.memory.store import MemoryStore
from src.common.logger import init_logger


@pytest.mark.skipif(not os.getenv("QDRANT_URL"), reason="Requires Qdrant URL")
def test_qdrant_upsert_and_retrieve(tmp_path):
    os.environ.setdefault("GROQ_API_KEY", "test_key")

    store = MemoryStore(base_dir=str(tmp_path / "data" / "memory_store"))
    log = init_logger(log_dir=str(tmp_path))

    docs = [
        Document(text="sample about product Z", metadata={"namespace": "sales", "product_id": "Z"}),
        Document(text="instagram caption for post 123", metadata={"namespace": "insta", "post_id": "123"}),
    ]
    store.add(namespace="sales", documents=[docs[0]])
    store.add(namespace="insta", documents=[docs[1]])

    retr = store.as_retriever(namespace="sales", similarity_top_k=2)
    nodes = retr.retrieve("product Z")
    assert nodes is not None


