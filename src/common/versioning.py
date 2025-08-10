import os
from datetime import date

def latest_dated_subdir(root: str) -> str | None:
    subs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subs: return None
    subs.sort(); return subs[-1]

def dated_subdir(root: str, stamp: str | None = None) -> str:
    stamp = stamp or date.today().strftime("%Y%m%d")
    full = os.path.join(root, stamp); os.makedirs(full, exist_ok=True); return full
