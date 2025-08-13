from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, options: Dict[str, Any] | None = None) -> str: ...

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> str: ...


def get_llm_provider_from_env() -> LLMProvider:
    """
    Factory that returns the configured LLM provider based on env var LLM_PROVIDER.
    Currently supports only 'groq' via OpenAI-compatible API.
    """
    provider = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    if provider != "groq":
        # Default to groq to remove any Ollama remnants/offline fallback
        provider = "groq"

    from .groq_client import GroqProvider

    return GroqProvider()
