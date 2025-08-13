import os
from pathlib import Path

from enums.enum import Directories
from src.common.logger import init_logger
from src.memory.store import MemoryStore
from src.pipelines.analysis_pipeline import AnalysisPipeline
import os
from llama_index.core import Document


def test_index_and_query(tmp_path):
    base = tmp_path / "data" / "memory_store" / "test"
    os.makedirs(base, exist_ok=True)

    store = MemoryStore(base_dir=str(tmp_path / "data" / "memory_store"))
    log = init_logger(log_dir=str(tmp_path))

    docs = [
        Document(text="جمله فارسی درباره فروش محصول A.", metadata={"namespace": "sales", "source": "a.txt"}),
        Document(text="English line about product A sales.", metadata={"namespace": "sales", "source": "b.txt"}),
    ]
    store.add(namespace="sales", documents=docs)

    # Use dummy Groq key for test environments that mock network
    os.environ.setdefault("GROQ_API_KEY", "test_key")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    ap = AnalysisPipeline(store=store, log=log)
    result = ap.answer(question="فروش محصول A چیست؟", top_k=5, top_r=3)
    assert result["answer"], "answer should not be empty"
    assert len(result["contexts"]) >= 1


