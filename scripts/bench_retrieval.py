import os
import time

from enums.enum import Directories
from src.common.logger import init_logger
from src.memory.store import MemoryStore
from src.pipelines.analysis_pipeline import AnalysisPipeline


def main() -> None:
    os.makedirs(Directories.LOGS.value, exist_ok=True)
    log = init_logger(log_dir=Directories.LOGS.value)

    store = MemoryStore(base_dir=Directories.MEMORY_STORE.value)
    ap = AnalysisPipeline(store=store, log=log)

    queries = [
        ("sales", "فروش محصول A در هفته گذشته چقدر بود؟"),
        ("insta", "کدام پست‌ها بیشترین لایک را گرفتند؟"),
    ]

    for ns, q in queries:
        t0 = time.time()
        result = ap.answer(namespace=ns, question=q, top_k=5, top_r=3)
        dur = time.time() - t0
        print({
            "namespace": ns,
            "latency_s": round(dur, 3),
            "contexts": len(result.get("contexts", [])),
            "answer_head": str(result.get("answer", ""))[:120],
        })


if __name__ == "__main__":
    main()


