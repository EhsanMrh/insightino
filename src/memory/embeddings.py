# src/memory/embeddings.py
#
# Embeddings via HuggingFace Sentence Transformers by default.
# - Caches results by SHA-256 of text
# - L2-normalizes vectors (recommended for cosine similarity)
# - Batch helper loops over single-call endpoint
#
# Requirements:
#   - Ollama running locally
#   - `ollama pull bge-m3`  (or your chosen embedding model)
#
# Env (optional):
#   OLLAMA_HOST=http://localhost:11434
#
# Model name is taken from enums.OllamaModels.EMBEDDING_MODEL by default.

import os
import hashlib
from typing import List, Optional, Dict, Any

import numpy as np
import requests

from enums.enum import RAGParams


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


class Embeddings:
    """
    Minimal embedding provider wrapper. For migration we keep a requests-based
    client signature but recommend using LlamaIndex HuggingFace embeddings at
    the index/query level (see MemoryStore._ensure_embedder()).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        normalize: bool = True,
        cache: Optional[Dict[str, List[float]]] = None,
        ollama_base_url: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.model_name = (model_name or "intfloat/multilingual-e5-base").strip()
        self.normalize = bool(normalize)
        self.cache = cache if cache is not None else {}
        self.ollama_base_url = (ollama_base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout_seconds = int(timeout_seconds)

    # -----------------------------
    # Public API
    # -----------------------------

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single string into a vector.
        """
        key = _sha256(query)
        if key in self.cache:
            return self.cache[key]

        emb = self._embed_ollama_single(query)

        if self.normalize:
            emb = self._ensure_normalized([emb])[0]

        self.cache[key] = emb
        return emb

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings. Uses cache and loops the Ollama endpoint.
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_idx: List[int] = []
        missing_texts: List[str] = []

        # Cache lookup
        for i, t in enumerate(texts):
            key = _sha256(t)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                missing_idx.append(i)
                missing_texts.append(t)

        # Compute missing via Ollama (one by one)
        if missing_texts:
            computed = [self._embed_ollama_single(t) for t in missing_texts]
            if self.normalize:
                computed = self._ensure_normalized(computed)
            for i, emb in zip(missing_idx, computed):
                results[i] = emb
                self.cache[_sha256(texts[i])] = emb

        # Finalize
        return [r for r in results]  # type: ignore

    # -----------------------------
    # Internals
    # -----------------------------

    def _embed_ollama_single(self, text: str) -> List[float]:
        url = f"{self.ollama_base_url}/api/embeddings"
        payload = {"model": self.model_name, "prompt": text}

        resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama embeddings error {resp.status_code}: {resp.text}")

        data: Dict[str, Any] = resp.json()
        emb = data.get("embedding")
        if not emb:
            raise RuntimeError("Ollama returned no 'embedding' field.")
        return [float(x) for x in emb]

    def _ensure_normalized(self, batch: List[List[float]]) -> List[List[float]]:
        arr = np.array(batch, dtype=float)
        arr = _l2_normalize(arr)
        return [row.tolist() for row in arr]


########################################################

# # src/memory/embeddings.py
# #
# # Dual backend embeddings:
# #   - Preferred: Ollama local embeddings API
# #   - Alternative: HuggingFace sentence-transformers
# #
# # Selection rule:
# #   - If model name contains '/', assume HuggingFace (e.g., 'BAAI/bge-m3')
# #   - Otherwise, assume Ollama (e.g., 'bge-m3')

# import os
# import hashlib
# from typing import List, Optional, Dict, Any, Literal

# import numpy as np
# import requests

# try:
#     from sentence_transformers import SentenceTransformer
#     _HAS_ST = True
# except ImportError:
#     _HAS_ST = False

# from enums.enum import OllamaModels, RAGParams


# def _sha256(text: str) -> str:
#     return hashlib.sha256(text.encode("utf-8")).hexdigest()


# def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#     norms[norms == 0] = 1
#     return vectors / norms


# class Embeddings:
#     def __init__(
#         self,
#         model_name: Optional[str] = None,
#         source: Literal["auto", "ollama", "hf"] = "auto",
#         normalize: bool = RAGParams.EMBED_NORMALIZE.value,
#         cache: Optional[Dict[str, List[float]]] = None,
#         ollama_base_url: Optional[str] = None,
#         hf_show_progress: bool = False
#     ) -> None:
#         self.model_name = (model_name or OllamaModels.EMBEDDING_MODEL.value).strip()
#         self.source = self._choose_source(source, self.model_name)
#         self.normalize = normalize
#         self.cache = cache if cache is not None else {}
#         self.ollama_base_url = (ollama_base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
#         self.hf_show_progress = hf_show_progress
#         self._st_model = None

#         if self.source == "hf":
#             if not _HAS_ST:
#                 raise RuntimeError("sentence-transformers is not installed for HuggingFace source.")
#             self._st_model = SentenceTransformer(self.model_name)

#     @staticmethod
#     def _choose_source(source: str, model_name: str) -> str:
#         if source != "auto":
#             return source
#         return "hf" if "/" in model_name else "ollama"

#     def embed_query(self, text: str) -> List[float]:
#         key = _sha256(text)
#         if key in self.cache:
#             return self.cache[key]

#         if self.source == "ollama":
#             emb = self._embed_ollama(text)
#         else:
#             emb = self._embed_hf([text])[0]

#         if self.normalize:
#             emb = self._normalize_batch([emb])[0]

#         self.cache[key] = emb
#         return emb

#     def embed_texts(self, texts: List[str]) -> List[List[float]]:
#         results: List[Optional[List[float]]] = [None] * len(texts)
#         missing_idx, missing_texts = [], []

#         for i, t in enumerate(texts):
#             key = _sha256(t)
#             if key in self.cache:
#                 results[i] = self.cache[key]
#             else:
#                 missing_idx.append(i)
#                 missing_texts.append(t)

#         if missing_texts:
#             if self.source == "ollama":
#                 computed = [self._embed_ollama(t) for t in missing_texts]
#             else:
#                 computed = self._embed_hf(missing_texts)

#             if self.normalize:
#                 computed = self._normalize_batch(computed)

#             for i, emb in zip(missing_idx, computed):
#                 results[i] = emb
#                 self.cache[_sha256(texts[i])] = emb

#         return results  # type: ignore

#     def _embed_ollama(self, text: str) -> List[float]:
#         url = f"{self.ollama_base_url}/api/embeddings"
#         payload = {"model": self.model_name, "prompt": text}
#         resp = requests.post(url, json=payload, timeout=60)
#         if resp.status_code != 200:
#             raise RuntimeError(f"Ollama embeddings error {resp.status_code}: {resp.text}")
#         data: Dict[str, Any] = resp.json()
#         emb = data.get("embedding")
#         if not emb:
#             raise RuntimeError("No embedding returned from Ollama.")
#         return [float(x) for x in emb]

#     def _embed_hf(self, texts: List[str]) -> List[List[float]]:
#         if self._st_model is None:
#             if not _HAS_ST:
#                 raise RuntimeError("sentence-transformers not available.")
#             self._st_model = SentenceTransformer(self.model_name)
#         vecs = self._st_model.encode(texts, convert_to_numpy=True, show_progress_bar=self.hf_show_progress)
#         return [row.astype(float).tolist() for row in np.array(vecs)]

#     def _normalize_batch(self, batch: List[List[float]]) -> List[List[float]]:
#         arr = np.array(batch, dtype=float)
#         arr = _l2_normalize(arr)
#         return [row.tolist() for row in arr]












###############################################


# # src/memory/embeddings.py

# import hashlib
# from typing import List, Optional

# import numpy as np
# from sentence_transformers import SentenceTransformer


# class Embeddings:
#     """
#     Handles text embedding using a pluggable SentenceTransformer model.
#     Supports local caching to avoid recomputing embeddings for the same text.
#     """

#     def __init__(
#         self,
#         model_name: str = "intfloat/multilingual-e5-small",
#         cache: Optional[dict] = None,
#         normalize: bool = True
#     ) -> None:
#         """
#         :param model_name: Name of the embedding model to load.
#         :param cache: Optional dict-like object for caching embeddings.
#         :param normalize: Whether to normalize embeddings to unit length.
#         """
#         self.model_name = model_name
#         self.cache = cache
#         self.normalize = normalize
#         self.model = self._load_model()

#     def _load_model(self) -> SentenceTransformer:
#         """
#         Loads the SentenceTransformer model.
#         """
#         return SentenceTransformer(self.model_name)

#     def _hash_text(self, text: str) -> str:
#         """
#         Returns a stable hash for the given text to use as a cache key.
#         """
#         return hashlib.sha256(text.encode("utf-8")).hexdigest()

#     def embed_texts(self, texts: List[str]) -> List[List[float]]:
#         """
#         Embeds a list of texts into vector representations.
#         Uses cache if available.

#         :param texts: List of text strings.
#         :return: List of embedding vectors.
#         """
#         results = []
#         to_compute = []
#         compute_indices = []

#         # First pass: try to load from cache
#         for idx, text in enumerate(texts):
#             key = self._hash_text(text)
#             if self.cache and key in self.cache:
#                 results.append(self.cache[key])
#             else:
#                 results.append(None)  # placeholder
#                 to_compute.append(text)
#                 compute_indices.append(idx)

#         # Compute embeddings for missing items
#         if to_compute:
#             new_embeddings = self.model.encode(to_compute, convert_to_numpy=True, show_progress_bar=False)
#             if self.normalize:
#                 new_embeddings = self._normalize_embeddings(new_embeddings)

#             for emb, idx in zip(new_embeddings, compute_indices):
#                 results[idx] = emb.tolist()
#                 if self.cache is not None:
#                     key = self._hash_text(texts[idx])
#                     self.cache[key] = results[idx]

#         return results

#     def embed_query(self, query: str) -> List[float]:
#         """
#         Embeds a single query text into a vector representation.
#         """
#         key = self._hash_text(query)
#         if self.cache and key in self.cache:
#             return self.cache[key]

#         emb = self.model.encode(query, convert_to_numpy=True)
#         if self.normalize:
#             emb = self._normalize_embeddings(np.array([emb]))[0]
#         emb_list = emb.tolist()

#         if self.cache is not None:
#             self.cache[key] = emb_list

#         return emb_list

#     def _normalize_embeddings(self, embs: np.ndarray) -> np.ndarray:
#         """
#         Normalizes embeddings to unit vectors.
#         """
#         norms = np.linalg.norm(embs, axis=1, keepdims=True)
#         norms[norms == 0] = 1.0
#         return embs / norms
