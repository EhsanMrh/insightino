import os
import time
from typing import Any, Dict, List, Optional

from groq import Groq

from enums.enum import RAGParams
from .provider import LLMProvider


class GroqProvider(LLMProvider):
    """
    Groq SDK client for chat completions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.api_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required")
        self.model = (model or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
        self.temperature = float(temperature if temperature is not None else RAGParams.TEMPERATURE.value)
        self.max_tokens = int(max_tokens if max_tokens is not None else 512)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)

        self._client = Groq(api_key=self.api_key)

    # -----------------------------
    # Public API
    # -----------------------------
    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        messages = [
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages, options)

    def chat(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if options:
            payload.update({k: v for k, v in options.items() if v is not None})

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**payload)
                return resp.choices[0].message.content.strip()
            except Exception as e:  # retry on transient SDK/http errors
                last_err = e
                time.sleep(self.backoff_seconds * attempt)
                continue
        if last_err:
            raise last_err
        raise RuntimeError("Unknown error contacting Groq API")


