import logging, os
from datetime import datetime

def init_logger(name: str = "insta_insight", log_dir: str = "data/logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(name); logger.setLevel(logging.INFO); logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger
