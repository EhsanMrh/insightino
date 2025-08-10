import os
from dotenv import load_dotenv

from enums.enum import Directories, OllamaModels, RAGParams
from src.common.logger import init_logger
from src.llm.ollama_provider import OllamaProvider
from src.memory.embeddings import Embeddings
from src.memory.store import MemoryStore
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.analysis_pipeline import AnalysisPipeline

# Optional ingestion
from src.ingest.instagram_fetcher import InstagramFetcher
from src.ingest.sales_ingestor import SalesIngestor


if __name__ == "__main__":
    load_dotenv()

    # Ensure directories
    os.makedirs(Directories.LOGS.value, exist_ok=True)
    os.makedirs(Directories.SESSIONS.value, exist_ok=True)
    os.makedirs(Directories.VECTOR_STORE.value, exist_ok=True)

    log = init_logger(log_dir=Directories.LOGS.value)

    # Embeddings (model name from enum, normalization from RAGParams)
    embeddings = Embeddings(
        # model_name=OllamaModels.EMBEDDING_MODEL.value,
        # normalize=OllamaModels.EMBED_NORMALIZE.value
    )

    # Vector/metadata store (all params from RAGParams)
    store = MemoryStore(
        vector_store_path=Directories.VECTOR_STORE.value,
        meta_db_path=Directories.MEMORY_DB.value,
        dim=RAGParams.VECTOR_DIM.value,
        space=RAGParams.VECTOR_SPACE.value,
        ef_construction=RAGParams.VECTOR_EF.value,
        m=RAGParams.VECTOR_M.value,
        ef_search=RAGParams.VECTOR_EF.value,   # همون ef برای سرچ
        max_elements=200000
    )

    # LLM provider (all from enum/env)
    llm = OllamaProvider(
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model=OllamaModels.TEXT_GENERATION_MODEL.value,
        embed_model=OllamaModels.EMBEDDING_MODEL.value,
        default_options={"num_ctx": 4096, "temperature": 0.3, "num_predict": 700},
    )

    data_pipeline = DataPipeline(log=log, embeddings=embeddings, store=store)
    analysis_pipeline = AnalysisPipeline(llm=llm, embeddings=embeddings, store=store, log=log)

    # Optional: fetch raw data
    # InstagramFetcher(log).full_refresh()
    # SalesIngestor(log).ingest_from_file("C:/path/to/your.xlsx", period_label="1404-05", move=False)

    # Build staging (from RAW)
    log.info("[Main] Building staging from RAW...")
    data_pipeline.build_sales_staging()
    data_pipeline.build_instagram_staging()

    # Indexing
    log.info("[Main] Starting RAG indexing...")
    data_pipeline.index_all()

    # Interactive analysis
    log.info("[Main] Starting interactive analysis mode...")
    analysis_pipeline.run_interactive()

    log.info("✅ Done.")
