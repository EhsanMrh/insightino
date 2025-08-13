from dataclasses import dataclass, field
from pathlib import Path
from enums.enum import Directories


@dataclass
class RAGConfig:
    """Configuration for Retrieval-Augmented Generation (RAG) memory system."""
    vector_dim: int = 1024
    vector_space: str = "cosine"
    vector_ef: int = 200
    vector_m: int = 16
    embed_normalize: bool = True
    namespace_sales: str = "sales"
    namespace_insta: str = "insta"


@dataclass
class Config:
    """Main application configuration."""
    # Paths (derived from Enums)
    logs_dir: Path = Path(Directories.LOGS.value)
    sessions_dir: Path = Path(Directories.SESSIONS.value)
    memory_db: Path = Path(Directories.MEMORY_DB.value)
    vector_store: Path = Path(Directories.VECTOR_STORE.value)
    raw_instagram_dir: Path = Path(Directories.RAW_INSTAGRAM.value)
    raw_sales_dir: Path = Path(Directories.RAW_SALES.value)
    raw_product_map: Path = Path(Directories.RAW_PRODUCT_MAP.value)
    staging_instagram: Path = Path(Directories.STAGING_INSTAGRAM.value)
    staging_sales: Path = Path(Directories.STAGING_SALES.value)
    mart_product_profile: Path = Path(Directories.MART_PRODUCT_PROFILE.value)

    # Vector/LLM provider settings (migrated to Qdrant + Groq)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_text: str = "insightino_text"
    qdrant_collection_image: str = "insightino_image"
    llm_provider: str = "groq"

    # RAG config (use default_factory to avoid mutable default)
    rag: RAGConfig = field(default_factory=RAGConfig)
