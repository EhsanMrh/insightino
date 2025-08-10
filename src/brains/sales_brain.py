# src/brains/sales_brain.py

import pandas as pd
from typing import Optional

from src.memory.schemas import MemoryItem
from src.utils.chunk_text import chunk_text


class SalesBrain:
    """
    Processes sales data, generates structured knowledge notes,
    and stores them in the RAG memory for later retrieval.
    """

    def __init__(self, embeddings, store, logger=None):
        """
        :param embeddings: Embedding provider (Embeddings class)
        :param store: MemoryStore instance
        :param logger: Optional logger
        """
        self.embeddings = embeddings
        self.store = store
        self.log = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_sales_data(self, df: pd.DataFrame) -> None:
        """
        Main entry point: takes raw sales dataframe and indexes into the vector store.
        Expects df to contain at least: ['date', 'product_id', 'sales_qty', ...]
        """
        if df.empty:
            if self.log:
                self.log.warning("[SalesBrain] Received empty sales DataFrame. Skipping indexing.")
            return

        # Build notes
        notes = self._build_sales_notes(df)

        # Chunk + embed + upsert
        items = self._notes_to_items(notes)
        if items:
            self.store.upsert(namespace="sales", items=items)
            if self.log:
                self.log.info(f"[SalesBrain] Indexed {len(items)} sales knowledge chunks.")
        else:
            if self.log:
                self.log.info("[SalesBrain] No sales items generated for indexing.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sales_notes(self, df: pd.DataFrame) -> list[dict]:
        """
        Aggregate sales data and produce one note per (product_id, date).
        Each note is a dict: { 'text': str, 'metadata': dict }
        """
        # Ensure date is string for metadata
        if not pd.api.types.is_string_dtype(df["date"]):
            df["date"] = df["date"].astype(str)

        notes = []

        grouped = df.groupby(["product_id", "date"], as_index=False)
        for _, grp in grouped:
            product_id = grp["product_id"].iloc[0]
            date = grp["date"].iloc[0]

            # Example metrics (adjust as needed)
            total_sales = grp["sales_qty"].sum()
            total_revenue = grp["revenue"].sum() if "revenue" in grp.columns else None
            avg_price = grp["price"].mean() if "price" in grp.columns else None
            returns = grp["returns"].sum() if "returns" in grp.columns else None
            discount = grp["discount_pct"].mean() if "discount_pct" in grp.columns else None

            # Build human-readable summary text
            parts = [
                f"Product {product_id} on {date}: sold {int(total_sales)} units."
            ]
            if total_revenue is not None:
                parts.append(f"Revenue {total_revenue:,.0f}.")
            if avg_price is not None:
                parts.append(f"Average price {avg_price:,.2f}.")
            if returns is not None:
                parts.append(f"Returns {int(returns)} units.")
            if discount is not None:
                parts.append(f"Average discount {discount:.1f}%.")
            # You can add more domain-specific metrics here

            note_text = " ".join(parts)

            metadata = {
                "product_id": product_id,
                "date": date,
                "total_sales": int(total_sales),
                "total_revenue": float(total_revenue) if total_revenue is not None else None,
                "avg_price": float(avg_price) if avg_price is not None else None,
                "returns": int(returns) if returns is not None else None,
                "discount_pct": float(discount) if discount is not None else None,
            }

            notes.append({"text": note_text, "metadata": metadata})

        return notes

    def _notes_to_items(self, notes: list[dict]) -> list[MemoryItem]:
        """
        Converts raw notes into MemoryItem chunks with embeddings.
        """
        all_chunks = []
        for n in notes:
            chunks = chunk_text(n["text"], max_chars=500, overlap=50)
            for ch in chunks:
                all_chunks.append({"text": ch, "metadata": n["metadata"]})

        if not all_chunks:
            return []

        embeddings = self.embeddings.embed_texts([c["text"] for c in all_chunks])

        items = [
            MemoryItem(
                namespace="sales",
                content=chunk["text"],
                embedding=emb,
                metadata=chunk["metadata"],
            )
            for chunk, emb in zip(all_chunks, embeddings)
        ]
        return items
