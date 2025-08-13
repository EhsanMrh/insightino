# src/pipelines/analysis_pipeline.py

from typing import Optional, Dict

from enums.enum import RAGParams
from src.memory import context_builder
from src.memory.retriever import retrieve
from src.llm.provider import get_llm_provider_from_env, LLMProvider


class AnalysisPipeline:
    """
    Orchestrates the analysis stage of the system.
    Uses AnalystBrain to retrieve evidence from RAG and synthesize answers via LLM.
    """

    def __init__(self, store, log, llm: Optional[LLMProvider] = None):
        self.log = log
        self.store = store
        self.llm = llm or get_llm_provider_from_env()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        top_r: Optional[int] = None,
    ) -> Dict[str, object]:
        self.log.info(f"[AnalysisPipeline] Q: {question}")
        # Namespace is intentionally ignored for global retrieval; store-level retriever
        # does not filter by namespace metadata in the current implementation.
        nodes = retrieve(question=question, store=self.store, namespace="all", top_k=top_k, top_r=top_r)
        if not nodes:
            self.log.info("[AnalysisPipeline] No context found; answering without evidence.")
        ctx = context_builder.build(question, nodes, context_builder.ContextConfig(max_chars=int(RAGParams.MAX_CONTEXT_CHARS.value)))
        sys_fa = (
            "به پرسش فقط با تکیه بر زمینه داده‌شده پاسخ بده. اگر کافی نیست بگو نمی‌دانم. "
        )
        prompt = f"[SYSTEM]\n{sys_fa}\n\n[CONTEXT]\n{ctx}\n\n[QUESTION]\n{question}"
        ans = self.llm.generate(prompt)
        return {"answer": ans, "contexts": [getattr(n, "text", None) or n.node.get_text() for n in nodes]}

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

            result = self.answer(question=query)
            print("\n=== ANSWER ===\n")
            print(result["answer"])  # type: ignore
