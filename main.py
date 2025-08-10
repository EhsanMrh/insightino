import os
from dotenv import load_dotenv
from enums.enum import Directories, OllamaModels
from src.common.logger import init_logger
from src.llm.ollama_provider import OllamaProvider
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.analysis_pipeline import AnalysisPipeline


# اختیاری: اگر می‌خوای همین‌جا فچ کنی
from src.ingest.instagram_fetcher import InstagramFetcher
from src.ingest.sales_ingestor import SalesIngestor

if __name__ == "__main__":
    load_dotenv()
    os.makedirs(Directories.LOGS.value, exist_ok=True)
    os.makedirs(Directories.SESSIONS.value, exist_ok=True)
    log = init_logger(log_dir=Directories.LOGS.value)



    # InstagramFetcher(log).full_refresh()

    # فروشِ یک فایل مشخص را کپی/موو کن به data/raw/sales با نام استاندارد:
    # SalesIngestor(log).ingest_from_file("C:/path/to/your.xlsx", period_label="1404-05", move=False)

    # --- Stage (latest/history → staging parquet) ---
    data = DataPipeline(log)
    data.stage_instagram_latest()
    data.stage_sales_merged()

    # --- Analysis (profile → snapshot → insights) ---
    llm = OllamaProvider(
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model=OllamaModels.TEXT_GENERATION_MODEL.value,
        embed_model=OllamaModels.EMBEDDING_MODEL.value,
        default_options={"num_ctx": 4096, "temperature": 0.3, "num_predict": 700},
    )
    analysis = AnalysisPipeline(llm, log)
    analysis.run()

    log.info("✅ Done.")
