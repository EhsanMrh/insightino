# src/brains/analyst_brain.py

from typing import Dict, List, Optional

from src.memory.schemas import Filters, ScoredItem
from src.memory.retriever import Retriever, FusionConfig, RerankConfig
from src.memory import context_builder


class AnalystBrain:
    """
    Combines knowledge from multiple RAG namespaces (sales + insta),
    retrieves relevant evidence, builds context, and queries the LLM.
    """

    def __init__(self, retriever: Retriever, llm, logger=None):
        """
        :param retriever: Retriever instance (wraps store + embeddings)
        :param llm: LLM provider (e.g., OllamaProvider)
        :param logger: Optional logger
        """
        self.retriever = retriever
        self.llm = llm
        self.log = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        query: str,
        filters_by_namespace: Optional[Dict[str, Filters]] = None,
        fusion_cfg: Optional[FusionConfig] = None,
        rerank_cfg: Optional[RerankConfig] = None,
        context_cfg: Optional[context_builder.ContextConfig] = None,
    ) -> str:
        """
        Run full RAG pipeline: retrieve → fuse → build context → LLM synthesis.

        :param query: Natural language question or analysis request.
        :param filters_by_namespace: Optional dict of Filters keyed by namespace.
        :param fusion_cfg: Optional FusionConfig to control retrieval fusion.
        :param rerank_cfg: Optional RerankConfig for post-fusion reranking.
        :param context_cfg: Optional ContextConfig for context building.
        :return: LLM-generated answer.
        """
        namespaces = ["sales", "insta"]

        if self.log:
            self.log.info(f"[AnalystBrain] Starting analysis for query: {query}")

        # 1) Cross-namespace retrieval
        results = self.retriever.cross_retrieve(
            query=query,
            namespaces=namespaces,
            fusion=fusion_cfg or FusionConfig(),
            filters_by_namespace=filters_by_namespace,
            rerank=rerank_cfg or RerankConfig(),
        )

        if self.log:
            self.log.debug(f"[AnalystBrain] Retrieved {len(results)} fused results.")

        # 2) Build structured context for LLM
        context_str = context_builder.build(
            query=query,
            items=results,
            cfg=context_cfg or context_builder.ContextConfig(),
        )

        if self.log:
            self.log.debug("[AnalystBrain] Built context for LLM.")

        # 3) Ask LLM
        prompt = (
            "[SYSTEM]\nYou are a product and marketing analyst. "
            "Use the evidence to answer the question. "
            "Cite references by their (ref:...) tokens.\n\n"
            f"[EVIDENCE]\n{context_str}\n\n[QUESTION]\n{query}"
        )

        llm_response = self.llm.generate(prompt)

        if self.log:
            self.log.info("[AnalystBrain] Analysis complete.")

        return llm_response

    def retrieve_evidence(
        self,
        query: str,
        filters_by_namespace: Optional[Dict[str, Filters]] = None,
        fusion_cfg: Optional[FusionConfig] = None,
        rerank_cfg: Optional[RerankConfig] = None,
    ) -> List[ScoredItem]:
        """
        Retrieve and fuse evidence from multiple namespaces without LLM synthesis.
        Useful for debugging retrieval and context building.

        :param query: Search query.
        :param filters_by_namespace: Optional filters dict keyed by namespace.
        :param fusion_cfg: Fusion config.
        :param rerank_cfg: Rerank config.
        :return: List of ScoredItems.
        """
        namespaces = ["sales", "insta"]

        return self.retriever.cross_retrieve(
            query=query,
            namespaces=namespaces,
            fusion=fusion_cfg or FusionConfig(),
            filters_by_namespace=filters_by_namespace,
            rerank=rerank_cfg or RerankConfig(),
        )
