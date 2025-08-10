# src/pipelines/analysis_pipeline.py

from typing import Optional, Dict

from src.brains.analyst_brain import AnalystBrain
from src.memory.schemas import Filters
from src.memory.retriever import Retriever, FusionConfig, RerankConfig
from src.memory import context_builder


class AnalysisPipeline:
    """
    Orchestrates the analysis stage of the system.
    Uses AnalystBrain to retrieve evidence from RAG and synthesize answers via LLM.
    """

    def __init__(self, llm, embeddings, store, log):
        """
        :param llm: LLM provider (e.g., OllamaProvider)
        :param embeddings: Embeddings provider
        :param store: MemoryStore instance
        :param log: Logger instance
        """
        self.log = log
        self.embeddings = embeddings
        self.store = store
        self.llm = llm

        # Build retriever with injected components
        self.retriever = Retriever(store=self.store, embeddings=self.embeddings, logger=self.log)

        # Analyst brain
        self.analyst_brain = AnalystBrain(retriever=self.retriever, llm=self.llm, logger=self.log)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_query(
        self,
        query: str,
        filters_by_namespace: Optional[Dict[str, Filters]] = None,
        fusion_cfg: Optional[FusionConfig] = None,
        rerank_cfg: Optional[RerankConfig] = None,
        context_cfg: Optional[context_builder.ContextConfig] = None,
    ) -> str:
        """
        Execute a single analysis query.
        Returns an LLM-generated answer based on RAG evidence.
        """
        self.log.info(f"[AnalysisPipeline] Running analysis query: {query}")

        answer = self.analyst_brain.analyze(
            query=query,
            filters_by_namespace=filters_by_namespace,
            fusion_cfg=fusion_cfg,
            rerank_cfg=rerank_cfg,
            context_cfg=context_cfg,
        )

        self.log.info("[AnalysisPipeline] Query execution complete.")
        return answer

    def run_interactive(self) -> None:
        """
        Simple interactive loop for manual analysis queries.
        """
        self.log.info("[AnalysisPipeline] Starting interactive analysis mode. Type 'exit' to quit.")

        while True:
            query = input("\nEnter your analysis question: ").strip()
            if not query or query.lower() in {"exit", "quit"}:
                self.log.info("[AnalysisPipeline] Exiting interactive mode.")
                break

            answer = self.run_query(query)
            print("\n=== ANSWER ===\n")
            print(answer)
