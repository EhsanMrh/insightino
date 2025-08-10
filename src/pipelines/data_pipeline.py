import os, glob, re, pandas as pd
from enums.enum import Directories
from ..data.io import write_parquet

def _latest_dated_subdir(root: str):
    subs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subs: return None
    subs.sort()
    return subs[-1]

def _period_key_from_name(name: str):
    # name: sales_data_1404-05.xlsx / sales_data_1404-05-01.xlsx
    m = re.match(r"sales_data_(\d{4})-(\d{2})(?:-\d{2})?\.xlsx$", name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

class DataPipeline:
    def __init__(self, logger):
        self.log = logger

    def stage_instagram_latest(self):
        raw_root = Directories.RAW_INSTAGRAM.value
        latest_dir = _latest_dated_subdir(raw_root)
        if not latest_dir:
            raise FileNotFoundError(f"No dated folder in {raw_root}")
        csv_path = os.path.join(latest_dir, "instagram_posts.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        out = Directories.STAGING_INSTAGRAM.value
        write_parquet(df, out)
        self.log.info(f"→ staged instagram_latest.parquet ({len(df)} rows)")


    def stage_sales_merged(self):
        sales_dir = Directories.RAW_SALES.value
        files = glob.glob(os.path.join(sales_dir, "sales_data_*.xlsx"))
        if not files:
            raise FileNotFoundError(f"No sales_data_*.xlsx in {sales_dir}")

        # sort by (year, month)
        files = sorted(files, key=lambda p: _period_key_from_name(os.path.basename(p)))

        df = pd.concat([pd.read_excel(p) for p in files], ignore_index=True)
        out = Directories.STAGING_SALES.value
        write_parquet(df, out)
        self.log.info(f"→ staged sales_merged.parquet ({len(df)} rows)")

