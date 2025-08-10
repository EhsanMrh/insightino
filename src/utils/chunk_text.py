# src/utils/chunk_text.py

from typing import List


def chunk_text(
    text: str,
    max_chars: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split a text into overlapping chunks of up to `max_chars` characters.

    :param text: The raw text to split.
    :param max_chars: Maximum number of characters per chunk.
    :param overlap: Number of overlapping characters between consecutive chunks.
    :return: List of chunk strings.
    """
    if not text:
        return []

    # Normalize whitespace to avoid weird chunk breaks
    clean_text = " ".join(text.split())

    if len(clean_text) <= max_chars:
        return [clean_text]

    chunks = []
    start = 0
    end = max_chars

    while start < len(clean_text):
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward with overlap
        start += max_chars - overlap
        end = start + max_chars

    return chunks


def chunk_paragraphs(
    paragraphs: List[str],
    max_chars: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Chunk a list of paragraphs, preserving paragraph boundaries as much as possible.

    :param paragraphs: List of paragraph strings.
    :param max_chars: Maximum characters per chunk.
    :param overlap: Overlap between consecutive chunks.
    :return: List of chunk strings.
    """
    result = []
    for para in paragraphs:
        result.extend(chunk_text(para, max_chars=max_chars, overlap=overlap))
    return result
