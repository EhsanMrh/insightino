# src/brains/insta_brain.py

import pandas as pd
from typing import Optional

from src.memory.schemas import MemoryItem
from src.utils.chunk_text import chunk_text


class InstaBrain:
    """
    Processes Instagram media data, generates structured knowledge notes,
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

    def index_insta_data(self, df: pd.DataFrame) -> None:
        """
        Main entry point: takes raw Instagram dataframe and indexes into the vector store.
        Expects df to contain at least: ['date', 'post_id', 'media_type', 'like_count', ...]
        """
        if df.empty:
            if self.log:
                self.log.warning("[InstaBrain] Received empty Instagram DataFrame. Skipping indexing.")
            return

        # Build notes
        notes = self._build_insta_notes(df)

        # Chunk + embed + upsert
        items = self._notes_to_items(notes)
        if items:
            self.store.upsert(namespace="insta", items=items)
            if self.log:
                self.log.info(f"[InstaBrain] Indexed {len(items)} Instagram knowledge chunks.")
        else:
            if self.log:
                self.log.info("[InstaBrain] No Instagram items generated for indexing.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_insta_notes(self, df: pd.DataFrame) -> list[dict]:
        """
        Aggregate Instagram data and produce one note per (post_id, date).
        Each note is a dict: { 'text': str, 'metadata': dict }
        """
        # Ensure date is string for metadata
        if not pd.api.types.is_string_dtype(df["date"]):
            df["date"] = df["date"].astype(str)

        notes = []

        grouped = df.groupby(["post_id", "date"], as_index=False)
        for _, grp in grouped:
            post_id = grp["post_id"].iloc[0]
            date = grp["date"].iloc[0]
            media_type = grp["media_type"].iloc[0] if "media_type" in grp.columns else None
            caption = grp["caption"].iloc[0] if "caption" in grp.columns else ""
            like_count = grp["like_count"].sum() if "like_count" in grp.columns else None
            comment_count = grp["comment_count"].sum() if "comment_count" in grp.columns else None
            share_count = grp["share_count"].sum() if "share_count" in grp.columns else None
            save_count = grp["save_count"].sum() if "save_count" in grp.columns else None
            reach = grp["reach"].sum() if "reach" in grp.columns else None
            impressions = grp["impressions"].sum() if "impressions" in grp.columns else None

            # Build human-readable summary text
            parts = [f"Instagram post {post_id} on {date}."]
            if media_type:
                parts.append(f"Type: {media_type}.")
            if like_count is not None:
                parts.append(f"Likes: {like_count}.")
            if comment_count is not None:
                parts.append(f"Comments: {comment_count}.")
            if share_count is not None:
                parts.append(f"Shares: {share_count}.")
            if save_count is not None:
                parts.append(f"Saves: {save_count}.")
            if reach is not None:
                parts.append(f"Reach: {reach}.")
            if impressions is not None:
                parts.append(f"Impressions: {impressions}.")
            if caption:
                parts.append(f"Caption: {caption}")

            note_text = " ".join(parts)

            metadata = {
                "post_id": post_id,
                "date": date,
                "media_type": media_type,
                "like_count": int(like_count) if like_count is not None else None,
                "comment_count": int(comment_count) if comment_count is not None else None,
                "share_count": int(share_count) if share_count is not None else None,
                "save_count": int(save_count) if save_count is not None else None,
                "reach": int(reach) if reach is not None else None,
                "impressions": int(impressions) if impressions is not None else None,
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
                namespace="insta",
                content=chunk["text"],
                embedding=emb,
                metadata=chunk["metadata"],
            )
            for chunk, emb in zip(all_chunks, embeddings)
        ]
        return items
