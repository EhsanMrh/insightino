import os
import argparse

from enums.enum import Directories
from src.common.logger import init_logger
from src.memory.store import MemoryStore
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.analysis_pipeline import AnalysisPipeline
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()

    # Ensure required dirs
    os.makedirs(Directories.LOGS.value, exist_ok=True)
    os.makedirs(Directories.SESSIONS.value, exist_ok=True)
    os.makedirs(Directories.MEMORY_STORE.value, exist_ok=True)

    parser = argparse.ArgumentParser(description="Insightino entrypoint")
    parser.add_argument("mode", choices=["interactive", "telegram"], help="Run mode")
    parser.add_argument("--skip-index", action="store_true", help="Skip staging build and indexing step")
    args = parser.parse_args()

    # Support common typo alias
    mode = args.mode

    log = init_logger(log_dir=Directories.LOGS.value)
    store = MemoryStore(base_dir=Directories.MEMORY_STORE.value)

    # Optional: fetch latest Instagram data if creds exist (disabled by default)
    # if os.getenv("INSTAGRAM_USERNAME") and os.getenv("INSTAGRAM_PASSWORD"):
    #     try:
    #         from src.ingest.instagram_fetcher import InstagramFetcher
    #         InstagramFetcher(logger=log, download_media=True).full_refresh()
    #     except Exception as e:
    #         log.warning(f"[Main] Instagram fetch skipped/failed: {e}")

    data_pipeline = DataPipeline(log=log, store=store)
    if not args.skip_index:
        # Build staging from RAW on every run (best-effort)
        try:
            data_pipeline.build_instagram_staging()
        except Exception as e:
            log.warning(f"[Main] Instagram staging build skipped/failed: {e}")
        try:
            data_pipeline.build_sales_staging()
        except Exception as e:
            log.warning(f"[Main] Sales staging build skipped/failed: {e}")

        log.info("[Main] Indexing any new/changed files under RAW...")
        data_pipeline.index_all()

    if mode == "interactive":
        analysis_pipeline = AnalysisPipeline(store=store, log=log)
        log.info("[Main] Starting interactive analysis mode...")
        analysis_pipeline.run_interactive()
        log.info("Done.")
    elif mode == "telegram":
        # Delegate to bot runner
        from src.bot.telegram_bot import main as run_bot
        run_bot()
