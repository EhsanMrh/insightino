import os
from typing import Optional

from dotenv import load_dotenv
import gradio as gr

from enums.enum import Directories, RAGParams
from src.common.logger import init_logger
from src.memory.store import MemoryStore
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.analysis_pipeline import AnalysisPipeline


def _truthy(env_value: Optional[str]) -> bool:
    if not env_value:
        return False
    return env_value.strip() in {"1", "true", "True", "yes", "on"}


def bootstrap() -> tuple[AnalysisPipeline, object]:
    # Load environment from .env if present
    load_dotenv()

    # Ensure required dirs
    os.makedirs(Directories.LOGS.value, exist_ok=True)
    os.makedirs(Directories.SESSIONS.value, exist_ok=True)
    os.makedirs(Directories.MEMORY_STORE.value, exist_ok=True)

    # Default to internal Qdrant when running in single container
    os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")

    log = init_logger(log_dir=Directories.LOGS.value)
    store = MemoryStore(base_dir=Directories.MEMORY_STORE.value)
    pipeline = DataPipeline(log=log, store=store)

    if not _truthy(os.getenv("SKIP_INDEX")):
        try:
            pipeline.build_instagram_staging()
        except Exception as e:  # noqa: BLE001 - best-effort staging
            log.warning(f"[Gradio] Instagram staging build skipped/failed: {e}")
        try:
            pipeline.build_sales_staging()
        except Exception as e:  # noqa: BLE001 - best-effort staging
            log.warning(f"[Gradio] Sales staging build skipped/failed: {e}")
        log.info("[Gradio] Indexing any new/changed files under RAW...")
        try:
            pipeline.index_all()
        except Exception as e:  # noqa: BLE001
            log.warning(f"[Gradio] Indexing skipped/failed: {e}")

    analysis = AnalysisPipeline(store=store, log=log)
    return analysis, log


analysis_pipeline, _log = bootstrap()


def answer_fn(question: str, top_k: int, top_r: int) -> str:
    if not question or not question.strip():
        return "لطفاً سوال خود را وارد کنید."
    try:
        res = analysis_pipeline.answer(
            question=question,
            top_k=top_k or None,
            top_r=top_r or None,
        )
        return str(res.get("answer", ""))  # type: ignore[no-any-return]
    except Exception as e:  # noqa: BLE001
        return f"خطایی رخ داد: {e}"


with gr.Blocks(title="Insightino") as demo:
    gr.Markdown("## Insightino — RAG UI (Gradio)")
    with gr.Row():
        top_k = gr.Slider(minimum=1, maximum=50, value=int(RAGParams.TOP_K.value), step=1, label="Top K")
        top_r = gr.Slider(minimum=1, maximum=50, value=int(RAGParams.TOP_R.value), step=1, label="Top R")
    q = gr.Textbox(lines=4, label="سوال")
    ask = gr.Button("بپرس")
    out = gr.Textbox(lines=12, label="پاسخ")

    ask.click(answer_fn, inputs=[q, top_k, top_r], outputs=out)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


