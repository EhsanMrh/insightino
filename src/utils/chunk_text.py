# src/utils/chunk_text.py

from typing import List

try:
    from llama_index.core.node_parser import SentenceSplitter
except Exception:  # pragma: no cover
    SentenceSplitter = None  # type: ignore


def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> List[str]:
    if not text:
        return []
    if SentenceSplitter is None:
        # Fallback simple splitter
        clean_text = " ".join(text.split())
        if len(clean_text) <= max_chars:
            return [clean_text]
        chunks, start = [], 0
        while start < len(clean_text):
            end = start + max_chars
            chunk = clean_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += max_chars - overlap
        return chunks

    splitter = SentenceSplitter(chunk_size=max_chars, chunk_overlap=overlap)
    nodes = splitter.split_text(text)
    # nodes can be TextNodes or raw strings depending on version; normalize to str
    out: List[str] = []
    for n in nodes:
        if isinstance(n, str):
            out.append(n)
        else:
            t = getattr(n, "text", None)
            if t is None and callable(getattr(n, "get_text", None)):
                t = n.get_text()  # type: ignore[attr-defined]
            if t is None:
                t = str(n)
            out.append(str(t))
    return out


def chunk_paragraphs(paragraphs: List[str], max_chars: int = 500, overlap: int = 50) -> List[str]:
    result: List[str] = []
    for para in paragraphs:
        result.extend(chunk_text(para, max_chars=max_chars, overlap=overlap))
    return result
