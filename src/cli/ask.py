import argparse
import os

from enums.enum import Directories
from src.common.logger import init_logger
from src.memory.store import MemoryStore
from src.pipelines.analysis_pipeline import AnalysisPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_r", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(Directories.MEMORY_STORE.value, exist_ok=True)
    os.makedirs(Directories.LOGS.value, exist_ok=True)

    log = init_logger(log_dir=Directories.LOGS.value)
    store = MemoryStore(base_dir=Directories.MEMORY_STORE.value)
    ap = AnalysisPipeline(store=store, log=log)

    result = ap.answer(question=args.question, top_k=args.top_k, top_r=args.top_r)
    print(result["answer"])  # type: ignore


if __name__ == "__main__":
    main()


