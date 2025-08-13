# Migration to Qdrant + Groq

1) Configure env:
```
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_TEXT=insightino_text
GROQ_API_KEY=...
GROQ_MODEL=llama-3.1-8b-instant
VECTOR_STORE=qdrant
LLM_PROVIDER=groq
```

2) Start Qdrant (Docker):
```
docker run -p 6333:6333 -p 6334:6334 -v $PWD/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

3) Prepare staging data:
```
python -m src.cli.index_data
```

4) Migrate (dry-run first):
```
python scripts/migrate_faiss_to_qdrant.py --dry-run
python scripts/migrate_faiss_to_qdrant.py --truncate
```

5) Health + Bench:
```
python scripts/health_qdrant.py
python scripts/bench_retrieval.py
```


