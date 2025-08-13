from __future__ import annotations

from typing import List, Optional

from llama_index.core.schema import NodeWithScore
try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
except Exception:  # package not installed
    FlagEmbeddingReranker = None  # type: ignore

from enums.enum import Flags, Models, RAGParams


def retrieve(
    question: str,
    store,
    namespace: str,
    top_k: Optional[int] = None,
    top_r: Optional[int] = None,
) -> List[NodeWithScore]:
    """
    Build a LlamaIndex retriever from Qdrant store and retrieve top_k nodes.
    Optionally rerank with FlagEmbeddingReranker to top_r.
    """
    k = int(top_k or RAGParams.TOP_K.value)
    r = int(top_r or RAGParams.TOP_R.value)

    retriever = store.as_retriever(namespace=namespace, similarity_top_k=k)
    try:
        nodes: List[NodeWithScore] = retriever.retrieve(question)
    except Exception:
        # If the namespace is empty or vector store cannot return nodes yet, return no results gracefully
        nodes = []

    if Flags.USE_RERANKER.value and r and r < len(nodes) and FlagEmbeddingReranker is not None:
        rr = FlagEmbeddingReranker(model=Models.RERANK_MODEL.value, top_n=r)  # type: ignore
        nodes = rr.postprocess_nodes(nodes, query_str=question)  # type: ignore

    return nodes




#########################################

# # src/memory/retriever.py

# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Callable, Tuple
# import math
# import numpy as np

# from .schemas import ScoredItem, Filters
# # store: expected to expose search(namespace, query, top_k, filters) -> List[ScoredItem]
# # embeddings: expected to expose embed_query(query: str) -> List[float]


# @dataclass
# class FusionConfig:
#     """
#     Configuration for cross-namespace fusion.
#     - method: 'wrr' (Weighted Reciprocal Rank) or 'zscore' (Z-Score fusion)
#     - weights: optional per-namespace weights, e.g. {'sales': 1.0, 'insta': 1.0}
#     - top_k_per_ns: how many items to pull from each namespace before fusion
#     - final_top_k: number of items to return after fusion
#     """
#     method: str = "wrr"
#     weights: Dict[str, float] = field(default_factory=dict)
#     top_k_per_ns: int = 8
#     final_top_k: int = 10


# @dataclass
# class RerankConfig:
#     """
#     Configuration for optional re-ranking.
#     - enabled: whether to re-rank after fusion
#     - weight: how much to mix in cosine similarity from embeddings (0..1)
#               final_score = (1 - weight) * fused_score + weight * cosine_sim
#     """
#     enabled: bool = True
#     weight: float = 0.35


# class Retriever:
#     """
#     High-level retrieval coordinator for multi-namespace RAG.
#     Orchestrates store searches, cross-namespace fusion, and optional re-ranking.
#     """

#     def __init__(self, store, embeddings, logger=None) -> None:
#         """
#         :param store: Vector/metadata store with a .search(...) API
#         :param embeddings: Embeddings provider with .embed_query(text) -> List[float]
#         :param logger: Optional logger (info/debug)
#         """
#         self.store = store
#         self.embeddings = embeddings
#         self.log = logger

#     # ---------------------------
#     # Public API
#     # ---------------------------

#     def retrieve_single(
#         self,
#         query: str,
#         namespace: str,
#         top_k: int = 8,
#         filters: Optional[Filters] = None,
#     ) -> List[ScoredItem]:
#         """
#         Run a single-namespace search.
#         """
#         if self.log:
#             self.log.debug(f"[Retriever] single namespace='{namespace}', top_k={top_k}, filters={filters}")
#         return self.store.search(namespace=namespace, query=query, top_k=top_k, filters=filters, embeddings=self.embeddings)

#     def retrieve_multi(
#         self,
#         query: str,
#         namespaces: List[str],
#         top_k_per_ns: int = 8,
#         filters_by_namespace: Optional[Dict[str, Filters]] = None,
#     ) -> Dict[str, List[ScoredItem]]:
#         """
#         Run searches across multiple namespaces. Returns raw per-namespace lists.
#         """
#         results: Dict[str, List[ScoredItem]] = {}
#         for ns in namespaces:
#             f = filters_by_namespace.get(ns) if filters_by_namespace else None
#             if self.log:
#                 self.log.debug(f"[Retriever] multi namespace='{ns}', top_k={top_k_per_ns}, filters={f}")
#             results[ns] = self.store.search(namespace=ns, query=query, top_k=top_k_per_ns, filters=f, embeddings=self.embeddings)
#         return results

#     def cross_retrieve(
#         self,
#         query: str,
#         namespaces: List[str],
#         fusion: FusionConfig = FusionConfig(),
#         filters_by_namespace: Optional[Dict[str, Filters]] = None,
#         rerank: Optional[RerankConfig] = RerankConfig(),
#     ) -> List[ScoredItem]:
#         """
#         End-to-end: search multiple namespaces, fuse results, optionally re-rank, and return final list.
#         """
#         # 1) Multi-namespace search
#         per_ns = self.retrieve_multi(
#             query=query,
#             namespaces=namespaces,
#             top_k_per_ns=fusion.top_k_per_ns,
#             filters_by_namespace=filters_by_namespace,
#         )

#         # 2) Fusion
#         fused = self._fuse(per_ns, method=fusion.method, weights=fusion.weights, final_top_k=fusion.final_top_k)

#         # 3) Optional rerank by cosine similarity to query embedding
#         if rerank and rerank.enabled and fused:
#             fused = self._rerank_with_query_cosine(query=query, items=fused, weight=rerank.weight)
#             # Keep top-K after rerank just in case
#             fused = fused[:fusion.final_top_k]

#         return fused

#     # ---------------------------
#     # Fusion methods
#     # ---------------------------

#     def _fuse(
#         self,
#         per_namespace: Dict[str, List[ScoredItem]],
#         method: str = "wrr",
#         weights: Optional[Dict[str, float]] = None,
#         final_top_k: int = 10,
#     ) -> List[ScoredItem]:
#         """
#         Fuse results from multiple namespaces.
#         Supported methods:
#           - 'wrr': Weighted Reciprocal Rank
#           - 'zscore': Z-score fusion over normalized scores
#         """
#         if method not in {"wrr", "zscore"}:
#             raise ValueError(f"Unsupported fusion method: {method}")

#         weights = weights or {}
#         if method == "wrr":
#             fused = self._fusion_wrr(per_namespace, weights=weights)
#         else:
#             fused = self._fusion_zscore(per_namespace, weights=weights)

#         # Sort by fused score (desc) and slice
#         fused.sort(key=lambda si: si.score, reverse=True)
#         return fused[:final_top_k]

#     def _fusion_wrr(
#         self,
#         per_namespace: Dict[str, List[ScoredItem]],
#         weights: Dict[str, float],
#     ) -> List[ScoredItem]:
#         """
#         Weighted Reciprocal Rank:
#           score = sum_ns ( w_ns * sum_{i in ranked list} ( 1 / (rank_i + c) ) )
#         We approximate by assigning 1/(rank+1) to each item and summing across namespaces.
#         Items are identified by (namespace, content, metadata hash) uniqueness proxy.
#         """
#         c = 1.0  # small constant to smooth rank denominator
#         agg: Dict[str, Tuple[float, ScoredItem]] = {}

#         for ns, items in per_namespace.items():
#             w = weights.get(ns, 1.0)
#             for rank, si in enumerate(items):
#                 # base reciprocal rank
#                 rr = 1.0 / (rank + 1.0 + c)
#                 score_contrib = w * rr

#                 key = self._result_key(ns, si)
#                 if key not in agg:
#                     # create a shallow copy with fused score
#                     agg[key] = (score_contrib, self._copy_scored_item(si))
#                 else:
#                     s, obj = agg[key]
#                     agg[key] = (s + score_contrib, obj)

#         fused = [ScoredItem(score=s, item=obj.item) for key, (s, obj) in agg.items()]
#         return fused

#     def _fusion_zscore(
#         self,
#         per_namespace: Dict[str, List[ScoredItem]],
#         weights: Dict[str, float],
#     ) -> List[ScoredItem]:
#         """
#         Z-score fusion:
#           - Normalize scores within each namespace via z = (x - mean) / std
#           - Weighted sum across namespaces
#         """
#         agg: Dict[str, Tuple[float, ScoredItem]] = {}

#         for ns, items in per_namespace.items():
#             w = weights.get(ns, 1.0)
#             if not items:
#                 continue

#             scores = np.array([si.score for si in items], dtype=float)
#             mu = float(scores.mean())
#             sigma = float(scores.std()) if float(scores.std()) > 1e-9 else 1.0
#             zscores = (scores - mu) / sigma

#             for si, z in zip(items, zscores):
#                 key = self._result_key(ns, si)
#                 if key not in agg:
#                     agg[key] = (w * float(z), self._copy_scored_item(si))
#                 else:
#                     s, obj = agg[key]
#                     agg[key] = (s + w * float(z), obj)

#         fused = [ScoredItem(score=s, item=obj.item) for key, (s, obj) in agg.items()]
#         return fused

#     # ---------------------------
#     # Re-ranking
#     # ---------------------------

#     def _rerank_with_query_cosine(
#         self,
#         query: str,
#         items: List[ScoredItem],
#         weight: float = 0.35,
#     ) -> List[ScoredItem]:
#         """
#         Mix fused scores with cosine similarity between query embedding and item embeddings.
#         Assumes embeddings are L2-normalized. If not, normalizes on the fly.
#         final_score = (1 - weight) * fused_score + weight * cosine_sim
#         """
#         q = np.array(self.embeddings.embed_query(query), dtype=float)
#         q = self._normalize_if_needed(q)

#         out: List[ScoredItem] = []
#         for si in items:
#             emb = np.array(si.item.embedding, dtype=float)
#             emb = self._normalize_if_needed(emb)
#             cos = float(np.dot(q, emb))
#             mixed = (1.0 - weight) * float(si.score) + weight * cos
#             out.append(ScoredItem(score=mixed, item=si.item))

#         # Sort descending by new score
#         out.sort(key=lambda x: x.score, reverse=True)
#         return out

#     # ---------------------------
#     # Utilities
#     # ---------------------------

#     @staticmethod
#     def _normalize_if_needed(vec: np.ndarray) -> np.ndarray:
#         """
#         Ensure the vector is L2-normalized.
#         """
#         norm = np.linalg.norm(vec)
#         if norm == 0.0:
#             return vec
#         # Consider a threshold to avoid repeated normalizations
#         if abs(norm - 1.0) < 1e-3:
#             return vec
#         return vec / norm

#     @staticmethod
#     def _result_key(namespace: str, si: ScoredItem) -> str:
#         """
#         Build a deterministic key to identify an item across namespaces and lists.
#         Useful when the same logical item appears multiple times.
#         """
#         md = si.item.metadata or {}
#         anchor = f"{namespace}::{md.get('product_id','')}::{md.get('post_id','')}::{md.get('date','')}::{si.item.content[:64]}"
#         return anchor

#     @staticmethod
#     def _copy_scored_item(si: ScoredItem) -> ScoredItem:
#         """
#         Create a lightweight copy (preserve original item, reset score; we recompute outside).
#         """
#         return ScoredItem(score=0.0, item=si.item)
