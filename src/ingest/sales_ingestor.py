import os
import shutil
import pandas as pd
from enums.enum import Directories

class SalesIngestor:
    """
    یک فایل فروش را به مسیر استاندارد data/raw/sales منتقل/کپی می‌کند.
    نام مقصد: sales_data_{period_label}.xlsx  ← مثل 1404-05 یا 1404-05-01
    """

    def __init__(self, logger):
        self.log = logger
        self.sales_dir = Directories.RAW_SALES.value
        os.makedirs(self.sales_dir, exist_ok=True)

    def ingest_from_file(self, src_excel_path: str, period_label: str, move: bool = False) -> str:
        """
        src_excel_path: مسیر فایل ورودی اکسل
        period_label : '1404-05' یا '1404-05-01'
        move         : اگر True باشد فایل را move می‌کند؛ در غیر این صورت copy
        """
        if not os.path.exists(src_excel_path):
            raise FileNotFoundError(src_excel_path)

        # اعتبارسنجی سریع ستون‌ها (Optional اما مفید)
        try:
            df_head = pd.read_excel(src_excel_path, nrows=5)
            required = {"product_name", "sale_date", "quantity"}
            if not required.issubset({str(c) for c in df_head.columns}):
                self.log.warning(f"Sales file missing required columns; got {list(df_head.columns)}")
        except Exception as e:
            self.log.warning(f"Could not read sales head for validation: {e}")

        dst = os.path.join(self.sales_dir, f"sales_data_{period_label}.xlsx")
        if move:
            shutil.move(src_excel_path, dst)
            self.log.info(f"Sales file moved → {dst}")
        else:
            shutil.copy2(src_excel_path, dst)
            self.log.info(f"Sales file copied → {dst}")
        return dst
