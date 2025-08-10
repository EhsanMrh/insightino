import json, pandas as pd
from enums.enum import Directories, OllamaModels
from ..data.profile_builder import build_product_profile
from ..memory.store import MemoryStore

import os, glob, re

def _infer_period_from_sales() -> str:
    files = sorted(glob.glob(os.path.join(Directories.RAW_SALES.value, "sales_data_*.xlsx")))
    if not files:
        return "latest"

    # آخرین فایل بر اساس نام
    last = os.path.basename(files[-1])  # e.g. sales_data_1404-05.xlsx یا sales_data_1404-05-01.xlsx
    m = re.match(r"sales_data_(\d{4}-\d{2})(?:-\d{2})?\.xlsx$", last)
    return m.group(1) if m else "latest"


class AnalysisPipeline:
    def __init__(self, llm, logger):
        self.llm = llm
        self.log = logger
        self.mem = MemoryStore()

    def run(self):
        profile_df = build_product_profile()  # از Enum مسیرها را می‌خواند
        self.log.info(f"Built product_profile ({len(profile_df)} rows)")

        period = _infer_period_from_sales()
        self.mem.save_snapshot(f"product_profile_{period}", profile_df.to_dict(orient="records"))
        self.log.info(f"Snapshot saved for period {period}")

        for _, row in profile_df.iterrows():
            product = row["product_key"]
            prompt = f"""تحلیل کوتاه برای محصول: {product}
داده‌ها: {json.dumps(row.to_dict(), ensure_ascii=False)}
خروجی: 2-3 جمله خلاصه + 2 پیشنهاد عملی."""
            text = self.llm.generate(prompt)
            title = f"{product} — profile — {period}"   # idempotent per month
            self.mem.add_or_replace_insight(title, text, tags=["product_profile","auto"])
        self.log.info("Insights generated & stored.")