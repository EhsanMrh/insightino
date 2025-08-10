from enum import Enum
from pathlib import Path

class Directories(Enum):
    ROOT = "data"
    RAW_INSTAGRAM = "data/raw/instagram"        # YYYYMMDD/instagram_posts.csv
    RAW_SALES = "data/raw/sales"                # sales_data_YYYY-MM-DD.xlsx
    RAW_PRODUCT_MAP = "data/raw/product_media_map.xlsx"

    STAGING_INSTAGRAM = "data/staging/instagram_latest.parquet"
    STAGING_SALES = "data/staging/sales_merged.parquet"

    MART_PRODUCT_PROFILE = "data/marts/product_profile.parquet"

    LOGS = "data/logs"
    SESSIONS = "data/sessions"
    MEMORY_DB = "data/memory.sqlite"

class OllamaModels(Enum):
    TEXT_GENERATION_MODEL = "llama3.1" # qwen2.5
    EMBEDDING_MODEL = "bge-m3:latest"

class Const(Enum):
    INSTAGRAM_POST_COUNT = 5  # برای fetch اختیاری
