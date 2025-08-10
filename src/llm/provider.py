from abc import ABC, abstractmethod
from typing import Any, Dict, List

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, options: Dict[str, Any] | None = None) -> str: ...
    @abstractmethod
    def embed(self, text: str) -> List[float]: ...
