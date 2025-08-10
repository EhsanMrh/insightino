import requests
from typing import Any, Dict, List
from .provider import LLMProvider

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, embed_model: str, default_options: Dict[str, Any] | None = None):
        self.base = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.default_options = default_options or {}

    def generate(self, prompt: str, options: Dict[str, Any] | None = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {**self.default_options, **(options or {})}}
        r = requests.post(f"{self.base}/api/generate", json=payload, timeout=120); r.raise_for_status()
        data = r.json()
        if "response" not in data: raise RuntimeError(f"Ollama error: {data}")
        return data["response"].strip()

    def embed(self, text: str) -> List[float]:
        payload = {"model": self.embed_model, "prompt": text}
        r = requests.post(f"{self.base}/api/embeddings", json=payload, timeout=60); r.raise_for_status()
        return r.json()["embedding"]
