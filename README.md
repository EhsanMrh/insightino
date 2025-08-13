# Insightino RAG Toolkit (CLI + Telegram)

Prereqs
- Python 3.10+
- Vector store: Qdrant (HTTP). Set `QDRANT_URL` (e.g., `http://localhost:6333`).
- LLM: Groq API (OpenAI-compatible). Set `GROQ_API_KEY` and optional `GROQ_MODEL`.

Install
```
pip install -r requirements.txt
```

Index data (examples)
```
python -m src.cli.index_data --paths "./data/raw/sales/*.xlsx" "./data/raw/instagram/**/*.pdf"
```

Ask
```
python -m src.cli.ask --namespace sales --question "فروش محصول A در مرداد؟" --top_k 10 --top_r 5
```

Generate PDF report
```
python -m src.cli.generate_report --namespace sales --prompt_file ./standard_prompts/monthly_farsi.txt --out ./data/marts/report_1404-07.pdf
```

Telegram bot
- Set environment variable `TELEGRAM_BOT_TOKEN`
```
python -m src.bot.telegram_bot
```

Notes
Migration to Qdrant:
- Collections: `QDRANT_COLLECTION_TEXT` (default `insightino_text`), `QDRANT_COLLECTION_IMAGE` (future use)
- Run `python scripts/migrate_faiss_to_qdrant.py --dry-run` then without `--dry-run`.
- Health: `python scripts/health_qdrant.py`
- Bench: `python scripts/bench_retrieval.py`
- Models/params/constants are configured in `enums/enum.py`.

Env (.env) example:
```
# Qdrant
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=
QDRANT_COLLECTION_TEXT=insightino_text
QDRANT_COLLECTION_IMAGE=insightino_image

# Groq (OpenAI-compatible)
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Providers
VECTOR_STORE=qdrant
LLM_PROVIDER=groq
```

Run Qdrant via Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 -v $PWD/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```


```bash
docker run --rm -p 7860:7860 -v C:\Users\Ehsan\Desktop\data_test_docker:/app/data -v C:\Users\Ehsan\Desktop\data_test_docker\qdrant:/qdrant/storage --env-file .env insightino-ui
```