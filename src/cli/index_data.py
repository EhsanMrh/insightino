import argparse
import os

from enums.enum import Directories
from src.common.logger import init_logger
from src.memory.store import MemoryStore
from src.pipelines.data_pipeline import DataPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Index raw files into Qdrant via LlamaIndex")
    parser.add_argument("--namespace", required=False, help="Namespace hint (optional; inferred from path)")
    parser.add_argument("--paths", nargs="+", required=False, help="Glob patterns to include (optional)")
    args = parser.parse_args()

    os.makedirs(Directories.MEMORY_STORE.value, exist_ok=True)
    os.makedirs(Directories.LOGS.value, exist_ok=True)

    log = init_logger(log_dir=Directories.LOGS.value)
    store = MemoryStore(base_dir=Directories.MEMORY_STORE.value)
    pipeline = DataPipeline(log=log, store=store)

    # If custom paths provided, extend the indexer input
    if args.paths:
        import glob
        from pathlib import Path
        files = []
        for pat in args.paths:
            files.extend(glob.glob(pat, recursive=True))
        files = sorted(set(files))
        for f in files:
            try:
                docs = pipeline._load_file_as_documents(f)
                ns = args.namespace or pipeline._infer_namespace(f)
                store.add(namespace=ns, documents=docs)
                pipeline._mark_indexed(f)
                log.info(f"Indexed {len(docs)} docs from {f} → ns={ns}")
            except Exception as e:
                log.error(f"Failed to index {f}: {e}")
        pipeline._save_index_log()
    else:
        pipeline.index_all()


if __name__ == "__main__":
    main()


