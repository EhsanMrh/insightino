import json, os, sqlite3, numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enums.enum import Directories, OllamaModels
from ..llm.ollama_provider import OllamaProvider

def _utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

class MemoryStore:
    def __init__(self):
        self.db_path = Directories.MEMORY_DB.value
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_db()
        self.provider = OllamaProvider(
            base_url=os.getenv("OLLAMA_HOST","http://localhost:11434"),
            model=OllamaModels.TEXT_GENERATION_MODEL.value,
            embed_model=OllamaModels.EMBEDDING_MODEL.value,
            default_options={"num_ctx":4096,"temperature":0.3}
        )

    def _ensure_db(self):
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS insights(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, content TEXT NOT NULL, tags TEXT,
                created_at TEXT NOT NULL, embedding BLOB NOT NULL, dim INTEGER NOT NULL)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS snapshots(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_insights_title ON insights(title)")
            con.commit()

    def _vec_to_blob(self, v: np.ndarray)->bytes: return v.astype(np.float32).tobytes()
    def _blob_to_vec(self, b: bytes, dim:int)->np.ndarray: return np.frombuffer(b, dtype=np.float32, count=dim)

    def add_or_replace_insight(self, title:str, content:str, tags:Optional[List[str]]=None):
        emb = np.array(self.provider.embed(f"{title}\n{content}"), dtype=np.float32)
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute("DELETE FROM insights WHERE title = ?", (title,))
            cur.execute("""INSERT INTO insights(title,content,tags,created_at,embedding,dim)
                           VALUES (?,?,?,?,?,?)""",
                        (title, content, json.dumps(tags or [], ensure_ascii=False),
                         _utc(), self._vec_to_blob(emb), emb.shape[0]))
            con.commit()

    def save_snapshot(self, label:str, payload:Any):
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute("INSERT INTO snapshots(label,payload,created_at) VALUES (?,?,?)",
                        (label, json.dumps(payload, ensure_ascii=False), _utc()))
            con.commit()
