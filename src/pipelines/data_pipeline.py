# src/pipelines/data_pipeline.py

import os
import glob
from pathlib import Path
from typing import Optional, List

import pandas as pd

from enums.enum import Directories
from src.brains.sales_brain import SalesBrain
from src.brains.insta_brain import InstaBrain


class DataPipeline:
    """
    Orchestrates:
      1) Building staging files from RAW sources (instagram + sales)
      2) Loading staging DataFrames
      3) Indexing into the RAG store via SalesBrain / InstaBrain
    """

    def __init__(self, log, embeddings, store):
        self.log = log
        self.embeddings = embeddings
        self.store = store

        # Brains
        self.sales_brain = SalesBrain(embeddings=self.embeddings, store=self.store, logger=self.log)
        self.insta_brain = InstaBrain(embeddings=self.embeddings, store=self.store, logger=self.log)

    # ------------------------------------------------------------------
    # STAGING BUILDERS
    # ------------------------------------------------------------------

    def build_instagram_staging(self) -> Optional[pd.DataFrame]:
        """
        Build data/staging/instagram_latest.parquet from the latest RAW folder:
          RAW root: data/raw/instagram/YYYYMMDD/instagram_posts.csv

        Heuristics:
          - pick the latest YYYYMMDD folder by numeric sort
          - prefer file named 'instagram_posts.csv' otherwise the first *.csv
          - normalize a minimal set of columns for downstream use
        """
        raw_root = Path(Directories.RAW_INSTAGRAM.value)
        out_path = Path(Directories.STAGING_INSTAGRAM.value)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not raw_root.exists():
            self.log.warning(f"[DataPipeline] RAW instagram root not found: {raw_root}")
            return None

        # Find dated subfolders (YYYYMMDD)
        subdirs = sorted([p for p in raw_root.iterdir() if p.is_dir()], key=lambda p: p.name)
        if not subdirs:
            self.log.warning("[DataPipeline] No dated RAW instagram folders found.")
            return None

        latest = subdirs[-1]
        self.log.info(f"[DataPipeline] Using latest RAW instagram folder: {latest.name}")

        # Pick CSV file
        csv_path = latest / "instagram_posts.csv"
        if not csv_path.exists():
            csvs = list(latest.glob("*.csv"))
            if not csvs:
                self.log.warning(f"[DataPipeline] No CSV files in {latest}")
                return None
            csv_path = csvs[0]

        df = pd.read_csv(csv_path)

        # --- Normalize columns (best-effort) ---
        # Expected downstream minimal columns:
        # ['date', 'post_id', 'media_type', 'caption', 'like_count', 'comment_count',
        #  'share_count', 'save_count', 'reach', 'impressions']

        # try to infer post_id
        if "post_id" not in df.columns:
            if "code" in df.columns:
                df["post_id"] = df["code"]
            elif "id" in df.columns:
                df["post_id"] = df["id"].astype(str)
            else:
                df["post_id"] = df.index.astype(str)

        # try to infer date
        if "date" not in df.columns:
            # common fields in your code: 'taken_at' (string/datetime)
            cand = None
            for c in ["taken_at", "created_at", "timestamp"]:
                if c in df.columns:
                    cand = c
                    break
            if cand:
                df["date"] = pd.to_datetime(df[cand], errors="coerce").dt.strftime("%Y-%m-%d")
            else:
                df["date"] = pd.NaT
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # media_type: prefer human-readable if exists
        if "instagram_media_type" in df.columns:
            df["media_type"] = df["instagram_media_type"].fillna("UNKNOWN")
        elif "media_type" in df.columns:
            df["media_type"] = df["media_type"]
        elif "media_type_id" in df.columns:
            mt_map = {1: "PHOTO", 2: "VIDEO", 8: "CAROUSEL"}
            df["media_type"] = df["media_type_id"].map(mt_map).fillna("UNKNOWN")
        else:
            df["media_type"] = "UNKNOWN"

        # text-ish fields
        if "caption" not in df.columns:
            if "caption_text" in df.columns:
                df["caption"] = df["caption_text"]
            else:
                df["caption"] = ""

        # numeric engagement fields (best-effort)
        def ensure_int(colnames: List[str], out_col: str):
            for c in colnames:
                if c in df.columns:
                    df[out_col] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
                    return
            df[out_col] = 0

        ensure_int(["like_count", "likes"], "like_count")
        ensure_int(["comment_count", "comments"], "comment_count")
        ensure_int(["share_count", "shares"], "share_count")
        ensure_int(["save_count", "saves"], "save_count")
        ensure_int(["reach", "reach_count"], "reach")
        ensure_int(["impressions", "impressions_count"], "impressions")

        # keep minimal schema
        cols = [
            "date", "post_id", "media_type", "caption",
            "like_count", "comment_count", "share_count", "save_count",
            "reach", "impressions",
        ]
        df_out = df[cols].copy()

        df_out.to_parquet(out_path, index=False)
        self.log.info(f"[DataPipeline] Wrote staging instagram to: {out_path} ({len(df_out)} rows)")
        return df_out

    def build_sales_staging(self) -> Optional[pd.DataFrame]:
        """
        Build data/staging/sales_merged.parquet from RAW sales Excel files:
          RAW root: data/raw/sales/*.xlsx

        Heuristics:
          - read all xlsx files
          - normalize columns to: ['date', 'product_id', 'sales_qty', 'price', 'revenue', 'returns', 'discount_pct']
          - if revenue is missing, compute as sales_qty * price (when possible)
        """
        raw_glob = str(Path(Directories.RAW_SALES.value) / "*.xlsx")
        out_path = Path(Directories.STAGING_SALES.value)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        files = sorted(glob.glob(raw_glob))
        if not files:
            self.log.warning(f"[DataPipeline] No RAW sales Excel files found at {raw_glob}")
            return None

        frames = []
        for fp in files:
            try:
                df = pd.read_excel(fp)
                df_norm = self._normalize_sales_columns(df)
                frames.append(df_norm)
                self.log.info(f"[DataPipeline] Loaded RAW sales: {fp} ({len(df_norm)} rows)")
            except Exception as e:
                self.log.error(f"[DataPipeline] Failed reading {fp}: {e}")

        if not frames:
            self.log.warning("[DataPipeline] No sales frames after normalization.")
            return None

        merged = pd.concat(frames, ignore_index=True)
        # clean types
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        # sort for sanity
        merged = merged.sort_values(["date", "product_id"], kind="stable")

        merged.to_parquet(out_path, index=False)
        self.log.info(f"[DataPipeline] Wrote staging sales to: {out_path} ({len(merged)} rows)")
        return merged

    # ------------------------------------------------------------------
    # STAGING LOADERS (used by indexing)
    # ------------------------------------------------------------------

    def stage_instagram_latest(self) -> Optional[pd.DataFrame]:
        """
        Load and return the latest Instagram posts DataFrame from staging.
        """
        path = Path(Directories.STAGING_INSTAGRAM.value)
        if not path.exists():
            self.log.warning(f"[DataPipeline] Staging instagram not found: {path}")
            return None
        df = pd.read_parquet(path)
        self.log.info(f"[DataPipeline] Loaded Instagram staging: {len(df)} rows from {path}")
        return df

    def stage_sales_merged(self) -> Optional[pd.DataFrame]:
        """
        Load and return merged sales DataFrame from staging.
        """
        path = Path(Directories.STAGING_SALES.value)
        if not path.exists():
            self.log.warning(f"[DataPipeline] Staging sales not found: {path}")
            return None
        df = pd.read_parquet(path)
        self.log.info(f"[DataPipeline] Loaded sales staging: {len(df)} rows from {path}")
        return df

    # ------------------------------------------------------------------
    # INDEXING
    # ------------------------------------------------------------------

    def index_all(self) -> None:
        """
        Index both sales and Instagram staging DataFrames into the RAG store.
        """
        # Sales
        sales_df = self.stage_sales_merged()
        if sales_df is not None and not sales_df.empty:
            self.sales_brain.index_sales_data(sales_df)
        else:
            self.log.info("[DataPipeline] Skipping sales indexing (no staging data).")

        # Instagram
        insta_df = self.stage_instagram_latest()
        if insta_df is not None and not insta_df.empty:
            self.insta_brain.index_insta_data(insta_df)
        else:
            self.log.info("[DataPipeline] Skipping Instagram indexing (no staging data).")

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _normalize_sales_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize raw sales data into a consistent schema:
        ['date', 'product_id', 'sales_qty', 'price', 'revenue', 'returns', 'discount_pct']

        Assumptions for your current files:
        - product_name : product label (string)  → mapped to 'product_id'
        - sale_date    : sale date (YYYY-MM-DD) → mapped to 'date'
        - quantity     : units sold (int)       → mapped to 'sales_qty'

        Best-effort fallbacks are included for slightly different headers.
        """
        import pandas as pd

        # Lowercase and trim column names for robust matching
        dfc = df.copy()
        dfc.columns = [str(c).strip().lower() for c in dfc.columns]

        # Required base columns with fallbacks
        col_date = "sale_date" if "sale_date" in dfc.columns else ("date" if "date" in dfc.columns else None)
        col_product = "product_name" if "product_name" in dfc.columns else ("product_id" if "product_id" in dfc.columns else None)
        col_qty = "quantity" if "quantity" in dfc.columns else ("sales_qty" if "sales_qty" in dfc.columns else None)

        missing = [name for name, col in [("sale_date/date", col_date),
                                        ("product_name/product_id", col_product),
                                        ("quantity/sales_qty", col_qty)] if col is None]
        if missing:
            raise ValueError(f"Missing required sales columns: {', '.join(missing)}")

        out = pd.DataFrame()
        # Date → ISO string YYYY-MM-DD
        out["date"] = pd.to_datetime(dfc[col_date], errors="coerce").dt.strftime("%Y-%m-%d")

        # Product id (keep as text; you can later map names to IDs if needed)
        out["product_id"] = dfc[col_product].astype(str).fillna("")

        # Sales quantity (integer, non-negative)
        out["sales_qty"] = pd.to_numeric(dfc[col_qty], errors="coerce").fillna(0).astype(int)
        out.loc[out["sales_qty"] < 0, "sales_qty"] = 0

        # Optional columns (set to None/0 if not provided)
        # price (per-unit), revenue (total), returns (units), discount_pct (0-100)
        out["price"] = pd.NA
        out["revenue"] = pd.NA
        out["returns"] = 0
        out["discount_pct"] = pd.NA

        return out








# # src/pipelines/data_pipeline.py

# import pandas as pd
# from typing import Optional

# from enums.enum import Directories
# from src.brains.sales_brain import SalesBrain
# from src.brains.insta_brain import InstaBrain


# class DataPipeline:
#     """
#     Orchestrates data staging and indexing for RAG.
#     - Stages latest Instagram and Sales data into consistent DataFrames.
#     - Passes them to respective brains for indexing in the vector store.
#     """

#     def __init__(self, log, embeddings, store):
#         """
#         :param log: Logger instance
#         :param embeddings: Embedding provider (Embeddings)
#         :param store: MemoryStore instance
#         """
#         self.log = log
#         self.embeddings = embeddings
#         self.store = store

#         # Brains
#         self.sales_brain = SalesBrain(embeddings=self.embeddings, store=self.store, logger=self.log)
#         self.insta_brain = InstaBrain(embeddings=self.embeddings, store=self.store, logger=self.log)

#     # ------------------------------------------------------------------
#     # Staging methods (adapt these to your data formats)
#     # ------------------------------------------------------------------

#     def stage_instagram_latest(self) -> Optional[pd.DataFrame]:
#         """
#         Load and return the latest Instagram posts data as a DataFrame.
#         Expected columns: ['date', 'post_id', 'media_type', 'caption', 'like_count', ...]
#         """
#         # Example: load from staging parquet
#         try:
#             path = Directories.STAGING_INSTAGRAM.value
#             df = pd.read_parquet(path)
#             self.log.info(f"[DataPipeline] Loaded Instagram data: {len(df)} rows from {path}")
#             return df
#         except FileNotFoundError:
#             self.log.warning("[DataPipeline] No Instagram staging file found.")
#             return None

#     def stage_sales_merged(self) -> Optional[pd.DataFrame]:
#         """
#         Load and return merged sales data as a DataFrame.
#         Expected columns: ['date', 'product_id', 'sales_qty', 'revenue', ...]
#         """
#         try:
#             path = Directories.STAGING_SALES.value
#             df = pd.read_parquet(path)
#             self.log.info(f"[DataPipeline] Loaded sales data: {len(df)} rows from {path}")
#             return df
#         except FileNotFoundError:
#             self.log.warning("[DataPipeline] No sales staging file found.")
#             return None

#     # ------------------------------------------------------------------
#     # Indexing methods
#     # ------------------------------------------------------------------

#     def index_all(self) -> None:
#         """
#         Run RAG indexing for both sales and Instagram data.
#         """
#         # 1) Sales data
#         sales_df = self.stage_sales_merged()
#         if sales_df is not None and not sales_df.empty:
#             self.sales_brain.index_sales_data(sales_df)
#         else:
#             self.log.info("[DataPipeline] Skipping sales indexing (no data).")

#         # 2) Instagram data
#         insta_df = self.stage_instagram_latest()
#         if insta_df is not None and not insta_df.empty:
#             self.insta_brain.index_insta_data(insta_df)
#         else:
#             self.log.info("[DataPipeline] Skipping Instagram indexing (no data).")
