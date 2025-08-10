# src/memory/schemas.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MemoryItem:
    """
    A single memory item to be stored in the Vector Store.
    - namespace: logical separation of data (e.g., 'sales', 'insta')
    - content: raw text content to be indexed
    - embedding: vector representation of the content
    - metadata: additional structured information (for filtering/citations)
    """
    namespace: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredItem:
    """
    A search result from the Vector Store.
    - score: similarity score (cosine similarity or distance)
    - item: the actual MemoryItem retrieved
    """
    score: float
    item: MemoryItem


@dataclass
class Filters:
    """
    Filters used for RAG search.
    All fields are optional.
    """
    product_ids: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None  # (YYYY-MM-DD, YYYY-MM-DD)
    media_types: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def match(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if given metadata matches the filter criteria.
        """
        # Product filter
        if self.product_ids and metadata.get("product_id") not in self.product_ids:
            return False
        # Date filter
        if self.date_range:
            start, end = self.date_range
            date_val = metadata.get("date")
            if date_val and not (start <= date_val <= end):
                return False
        # Media type filter
        if self.media_types and metadata.get("media_type") not in self.media_types:
            return False
        return True
