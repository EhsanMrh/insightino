import os, glob, pandas as pd

def ensure_dir_for(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_parquet(df: pd.DataFrame, path: str):
    ensure_dir_for(path); df.to_parquet(path, index=False)

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def merge_sales_dir(sales_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(sales_dir, "sales_*.xlsx")))
    if not files: raise FileNotFoundError(f"No sales files in {sales_dir}")
    return pd.concat([pd.read_excel(p) for p in files], ignore_index=True)
