from enum import Enum


# --------------------------------------------------------------------------------------
# New enums per RAG refactor (keep names stable; do not rely on .env)
# --------------------------------------------------------------------------------------

class Directories(Enum):
    RAW_BASE = "data/raw"
    RAW_SALES = "data/raw/sales"
    RAW_INSTAGRAM = "data/raw/instagram"
    MEMORY_STORE = "data/memory_store"
    LOGS = "data/logs"
    REPORTS = "data/marts"  # PDF outputs live here
    SESSIONS = "data/sessions"
    STAGING = "data/staging"
    # Staging file paths used by DataPipeline
    STAGING_INSTAGRAM = "data/staging/instagram_latest.parquet"
    STAGING_SALES = "data/staging/sales_merged.parquet"


class Models(Enum):
    GEN_MODEL = "llama-3.1-8b-instant"
    EMBED_MODEL = "intfloat/multilingual-e5-base"
    RERANK_MODEL = "BAAI/bge-reranker-base"


class RAGParams(Enum):
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    TOP_K = 10
    TOP_R = 5
    MAX_CONTEXT_CHARS = 6000
    NUM_CTX = 8192
    TEMPERATURE = 0.2
    TOP_P = 0.9


class Flags(Enum):
    USE_RERANKER = True


class Bot(Enum):
    # Token must come from env
    BOT_TOKEN_KEY = ""


# --------------------------------------------------------------------------------------
# Legacy enums kept for backward compatibility with existing modules (unused after refactor)
# --------------------------------------------------------------------------------------

class Providers(Enum):
    VECTOR_STORE = "qdrant"
    LLM_PROVIDER = "groq"

class QdrantDefaults(Enum):
    COLLECTION_TEXT = "insightino_text"
    COLLECTION_IMAGE = "insightino_image"


class Const(Enum):
    INSTAGRAM_POST_COUNT = 5
