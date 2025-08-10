import os
import sqlite3
import hashlib
from typing import List, Optional, Tuple
from dataclasses import dataclass

import hnswlib
import numpy as np

from src.memory.schemas import MemoryItem, ScoredItem, Filters


@dataclass
class MemoryStore:
    vector_store_path: str
    meta_db_path: str
    dim: int
    space: str = "cosine"
    ef_construction: int = 200
    m: int = 32
    ef_search: int = 128
    max_elements: int = 200000

    def __post_init__(self):
        os.makedirs(self.vector_store_path, exist_ok=True)
        self._init_meta_db()

    # -----------------------------
    # Meta DB
    # -----------------------------
    def _init_meta_db(self):
        conn = sqlite3.connect(self.meta_db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            id INTEGER PRIMARY KEY,
            namespace TEXT,
            content TEXT,
            metadata TEXT
        )
        """)
        conn.commit()
        conn.close()

    def _meta_add(self, conn, id_int: int, namespace: str, content: str, metadata: str):
        conn.execute("INSERT OR REPLACE INTO meta (id, namespace, content, metadata) VALUES (?, ?, ?, ?)",
                     (id_int, namespace, content, metadata))

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
    # Index Management
    # -----------------------------
    def _index_path(self, namespace: str) -> str:
        return os.path.join(self.vector_store_path, f"{namespace}.hnsw")

    def _load_or_create_index(self, path: str) -> hnswlib.Index:
        index = hnswlib.Index(space=self.space, dim=self.dim)
        if os.path.exists(path):
            index.load_index(path)
        else:
            index.init_index(max_elements=self.max_elements,
                             ef_construction=self.ef_construction,
                             M=self.m)
        index.set_ef(self.ef_search)
        return index

    @staticmethod
    def _make_uid(namespace: str, content: str) -> str:
        return hashlib.sha256(f"{namespace}:{content}".encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_to_int(h: str) -> int:
        # 16 hex chars = 64 bits → mask to 63 bits to fit signed int64 (np.int64)
        return int(h[:16], 16) & ((1 << 63) - 1)

    @staticmethod
    def _int_to_hash(i: int) -> str:
        return f"{i:016x}".ljust(64, "0")

    # -----------------------------
    # Upsert
    # -----------------------------
    def upsert(self, namespace: str, items: List[MemoryItem]):
        path = self._index_path(namespace)
        index = self._load_or_create_index(path)

        conn = sqlite3.connect(self.meta_db_path)
        import json

        for item in items:
            uid = self._make_uid(namespace, item.content)
            label = self._hash_to_int(uid)
            vector = np.array(item.embedding, dtype=np.float32)

            if index.get_current_count() >= self.max_elements:
                raise RuntimeError(f"Index for {namespace} reached max_elements={self.max_elements}")

            if vector.shape != (self.dim,):
                raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {vector.shape}")

            try:
                index.add_items(vector.reshape(1, -1), ids=np.array([label], dtype=np.int64))
            except RuntimeError:
                # Already exists → skip
                continue

            self._meta_add(conn, label, namespace, item.content, json.dumps(item.metadata or {}))

        conn.commit()
        conn.close()
        index.save_index(path)

    # -----------------------------
    # Search
    # -----------------------------
    def search(self, namespace: str, query_vec: List[float], top_k: int = 5, filters: Optional[Filters] = None) -> List[ScoredItem]:
        path = self._index_path(namespace)
        if not os.path.exists(path):
            return []

        index = hnswlib.Index(space=self.space, dim=self.dim)
        index.load_index(path)

        count = index.get_current_count()
        if count == 0:
            return []

        safe_k = min(top_k * 3, count)  # oversample ×3
        index.set_ef(max(self.ef_search, safe_k))  # ef >= k

        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        labels, distances = index.knn_query(q, k=safe_k)

        results: List[ScoredItem] = []
        conn = sqlite3.connect(self.meta_db_path)

        for label, dist in zip(labels[0], distances[0]):
            ns, content, metadata = self._meta_get_by_label(conn, int(label))
            if not ns:
                continue
            if filters and not filters.match(metadata):
                continue
            score = 1.0 - float(dist) if self.space == "cosine" else -float(dist)
            mi = MemoryItem(namespace=ns, content=content, embedding=[], metadata=metadata)
            results.append(ScoredItem(score=score, item=mi))
            if len(results) >= top_k:
                break

        conn.close()
        return results
