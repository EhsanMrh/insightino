import asyncio
import os
from functools import wraps
from typing import Dict

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from enums.enum import Directories
from src.common.logger import init_logger
from src.memory.store import MemoryStore
from src.pipelines.analysis_pipeline import AnalysisPipeline


RATE_LIMIT_PER_USER = 3  # requests in window
RATE_WINDOW_SEC = 30


class RateLimiter:
    def __init__(self) -> None:
        self._user_to_times: Dict[int, list[float]] = {}

    def allow(self, user_id: int, now: float) -> bool:
        times = self._user_to_times.setdefault(user_id, [])
        times[:] = [t for t in times if now - t < RATE_WINDOW_SEC]
        if len(times) >= RATE_LIMIT_PER_USER:
            return False
        times.append(now)
        return True


limiter = RateLimiter()
log = init_logger(log_dir=Directories.LOGS.value)
store = MemoryStore(base_dir=Directories.MEMORY_STORE.value)
pipeline = AnalysisPipeline(store=store, log=log)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("سوالت رو بفرست. برای نمونه /ask فروش وضعیت فروش محصول A در مرداد؟")


def _infer_namespace_from_text(text: str) -> str:
    t = (text or "").lower()
    # very simple heuristic keywords; default to sales
    insta_keys = ["اینستا", "پست", "لایک", "کامنت", "reach", "impressions", "instagram", "caption"]
    if any(k in t for k in insta_keys):
        return "insta"
    return "sales"


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    import time
    if update.effective_user is None or update.message is None:
        return
    if not limiter.allow(update.effective_user.id, time.time()):
        await update.message.reply_text("Rate limited. Try again shortly.")
        return

    if not context.args:
        await update.message.reply_text("Usage: /ask <question>")
        return
    q = " ".join(context.args)
    try:
        res = pipeline.answer(question=q)
        txt = res["answer"]  # type: ignore
        # split long
        chunks = [txt[i:i+3500] for i in range(0, len(txt), 3500)]
        for ch in chunks:
            await update.message.reply_text(ch)
    except Exception:
        log.exception("Failed to process /ask request")
        await update.message.reply_text("خطا در ارتباط با سرویس LLM. لطفاً کمی بعد دوباره تلاش کنید.")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # infer namespace based on message text
    if update.effective_user is None or update.message is None:
        return
    import time
    if not limiter.allow(update.effective_user.id, time.time()):
        await update.message.reply_text("Rate limited. Try again shortly.")
        return
    q = update.message.text.strip()
    try:
        res = pipeline.answer(question=q)
        txt = res["answer"]  # type: ignore
        chunks = [txt[i:i+3500] for i in range(0, len(txt), 3500)]
        for ch in chunks:
            await update.message.reply_text(ch)
    except Exception:
        log.exception("Failed to process text message")
        await update.message.reply_text("خطا در ارتباط با سرویس LLM. لطفاً کمی بعد دوباره تلاش کنید.")


def main() -> None:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment/.env")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling()


if __name__ == "__main__":
    main()


