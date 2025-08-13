import os
import sqlite3
import hashlib
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.memory.schemas import MemoryItem, ScoredItem, Filters


@dataclass
class MemoryStore:
    """
    Simple LlamaIndex-backed store using on-disk persistence per namespace.
    Avoids FAISS-specific APIs to keep dependencies minimal and tests portable.
    """

    base_dir: str
    dim: int = 768

    # Internal derived paths
    vector_store_path: str = field(init=False)
    meta_db_path: str = field(init=False)

    def __post_init__(self):
        self.vector_store_path = self.base_dir
        self.meta_db_path = os.path.join(self.base_dir, "meta.db")
        os.makedirs(self.vector_store_path, exist_ok=True)
        self._init_meta_db()

    # -----------------------------
    # Meta DB
    # -----------------------------
    def _init_meta_db(self):
        conn = sqlite3.connect(self.meta_db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                id INTEGER PRIMARY KEY,
                namespace TEXT,
                content TEXT,
                metadata TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def _meta_add(self, conn, id_int: int, namespace: str, content: str, metadata: str):
        conn.execute(
            "INSERT OR REPLACE INTO meta (id, namespace, content, metadata) VALUES (?, ?, ?, ?)",
            (id_int, namespace, content, metadata),
        )

    def _meta_get_by_label(self, conn, label: int) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
        cur = conn.execute("SELECT namespace, content, metadata FROM meta WHERE id=?", (label,))
        row = cur.fetchone()
        if row:
            ns, content, metadata_str = row
            import json

            metadata = json.loads(metadata_str) if metadata_str else {}
            return ns, content, metadata
        return None, None, None

    # -----------------------------
    # Helpers
    # -----------------------------
    def _ns_dir(self, namespace: str) -> str:
        return os.path.join(self.vector_store_path, namespace)

    def _ensure_embedder(self) -> None:
        try:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="intfloat/multilingual-e5-base",
                normalize=True,  # type: ignore[arg-type]
            )
        except TypeError:
            Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

    def _qdrant_client(self) -> QdrantClient:
        url = os.getenv("QDRANT_URL")
        if not url:
            raise RuntimeError("QDRANT_URL is required for Qdrant vector store")
        api_key = os.getenv("QDRANT_API_KEY") or None
        # Prefer gRPC on Windows to avoid intermittent HTTP 502 from httpx/local relays
        prefer_grpc_env = os.getenv("QDRANT_PREFER_GRPC", "1").strip()
        prefer_grpc = prefer_grpc_env not in {"0", "false", "False"}
        grpc_port_str = os.getenv("QDRANT_GRPC_PORT")
        grpc_port = int(grpc_port_str) if grpc_port_str and grpc_port_str.isdigit() else 6334
        return QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            grpc_port=grpc_port,
            timeout=60.0,
            check_compatibility=False,
        )

    def _ensure_collection(self, client: QdrantClient, collection: str, size: int) -> None:
        from qdrant_client.http import models as qmodels
        try:
            exists = client.get_collection(collection)
            if exists:
                return
        except Exception:
            pass
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )

    # -----------------------------
    # Upsert/Add
    # -----------------------------
    def upsert(self, namespace: str, items: List[MemoryItem]):
        self._ensure_embedder()
        # Switch to Qdrant vector store; use env collections per namespace
        client = self._qdrant_client()
        collection = os.getenv("QDRANT_COLLECTION_TEXT", "insightino_text")
        self._ensure_collection(client, collection, size=self.dim)

        vector_store = QdrantVectorStore(client=client, collection_name=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        docs: List[Document] = [Document(text=item.content, metadata={"namespace": namespace, **(item.metadata or {})}) for item in items]
        VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=False)

    def add(self, namespace: str, documents: List[Document]) -> None:
        self._ensure_embedder()
        client = self._qdrant_client()
        collection = os.getenv("QDRANT_COLLECTION_TEXT", "insightino_text")
        self._ensure_collection(client, collection, size=self.dim)
        vector_store = QdrantVectorStore(client=client, collection_name=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Normalize inputs to LlamaIndex Documents to guard against version differences
        normalized_docs: List[Document] = []
        for d in documents:
            try:
                # Prefer explicit text attribute
                text = getattr(d, "text", None)
                if text is None and callable(getattr(d, "get_text", None)):
                    text = d.get_text()  # type: ignore
                if text is None:
                    text = str(d)
                md = getattr(d, "metadata", {}) or {}
                normalized_docs.append(Document(text=str(text), metadata=md))
            except Exception:
                normalized_docs.append(Document(text=str(d), metadata={}))

        # Add namespace metadata if missing
        for d in normalized_docs:
            d.metadata = {"namespace": namespace, **(getattr(d, "metadata", {}) or {})}
        VectorStoreIndex.from_documents(normalized_docs, storage_context=storage_context, show_progress=False)

    # -----------------------------
    # Retrieval
    # -----------------------------
    def as_retriever(self, namespace: str, similarity_top_k: int = 10):
        # Ensure embedder
        self._ensure_embedder()
        client = self._qdrant_client()
        collection = os.getenv("QDRANT_COLLECTION_TEXT", "insightino_text")
        self._ensure_collection(client, collection, size=self.dim)
        vector_store = QdrantVectorStore(client=client, collection_name=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Build an empty index if needed implicitly
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)
        return index.as_retriever(similarity_top_k=similarity_top_k)

    # -----------------------------
    # Legacy compatibility stub
    # -----------------------------
    def search(
        self,
        namespace: str,
        query_vec: List[float],
        top_k: int = 5,
        filters: Optional[Filters] = None,
    ) -> List[ScoredItem]:
        # Not used in current flow.
        return []


