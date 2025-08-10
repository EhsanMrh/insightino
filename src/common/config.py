import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class AppConfig:
    data: Dict[str, Any]
    run: Dict[str, Any]
    llm: Dict[str, Any]

def load_config(path: str = "config/config.yaml") -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return AppConfig(data=y["data"], run=y["run"], llm=y["llm"])
