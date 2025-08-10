from enum import Enum


class Directories(Enum):
    # Root
    ROOT = "data"

    # Raw data sources
    RAW_INSTAGRAM = "data/raw/instagram"                  # YYYYMMDD/instagram_posts.csv
    RAW_SALES = "data/raw/sales"                          # sales_data_YYYY-MM-DD.xlsx
    RAW_PRODUCT_MAP = "data/raw/product_media_map.xlsx"

    # Staging (processed/intermediate)
    STAGING_INSTAGRAM = "data/staging/instagram_latest.parquet"
    STAGING_SALES = "data/staging/sales_merged.parquet"

    # Data marts / aggregated outputs
    MART_PRODUCT_PROFILE = "data/marts/product_profile.parquet"

    # Logs / sessions
    LOGS = "data/logs"
    SESSIONS = "data/sessions"

    # Memory / Vector store (RAG)
    MEMORY_DB = "data/memory.sqlite"
    VECTOR_STORE = "data/memory_store"

    # Media download targets
    DOWNLOAD_MEDIA = "data/raw/instagram/downloads"       # generic root for media
    DOWNLOAD_PHOTOS = "data/raw/instagram/downloads/photos"
    DOWNLOAD_VIDEOS = "data/raw/instagram/downloads/videos"
    DOWNLOAD_CAROUSEL = "data/raw/instagram/downloads/carousel"


class OllamaModels(Enum):
    TEXT_GENERATION_MODEL = "qwen2.5"          # Default LLM for analysis
    EMBEDDING_MODEL = "bge-m3"           # Embedding model (Ollama or sentence-transformers)
    EMBED_NORMALIZE = True


class Const(Enum):
    INSTAGRAM_POST_COUNT = 5  # For optional fetch limits


class RAGParams(Enum):
    # Embedding & vector search parameters
    VECTOR_DIM = 1024           # must match embedding model output dimension
    VECTOR_SPACE = "cosine"    # 'cosine', 'l2', or 'ip'
    VECTOR_EF = 128            # HNSW search ef parameter
    VECTOR_M = 16              # HNSW construction parameter
    EMBED_NORMALIZE = True