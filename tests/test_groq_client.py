import os
from unittest.mock import patch
from types import SimpleNamespace

from src.llm.groq_client import GroqProvider


def test_groq_chat_mocked():
    os.environ.setdefault("GROQ_API_KEY", "test_key")

    # Patch Groq SDK client used inside our provider
    with patch("src.llm.groq_client.Groq") as MockGroq:
        mock_client = MockGroq.return_value
        # Build a fake response structure: resp.choices[0].message.content -> "hello"
        fake_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
        )
        mock_client.chat.completions.create.return_value = fake_resp

        llm = GroqProvider()
        out = llm.generate("hi")
        assert out == "hello"


