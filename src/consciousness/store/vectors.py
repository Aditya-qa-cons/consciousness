"""Vector embedding store for semantic search over message history.

Uses ChromaDB for persistence + sentence-transformers for local embeddings.
No external API keys required — everything runs on-device.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from consciousness.models import Message, Role

_EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, ~90MB
_COLLECTION = "messages"
_CHUNK_SIZE = 512  # chars per chunk


@dataclass
class VectorHit:
    message_id: str
    conversation_id: str
    chunk_text: str
    score: float  # lower = more similar (L2 distance)


def _chunk_text(text: str, size: int = _CHUNK_SIZE) -> list[str]:
    """Split long messages into overlapping chunks for better recall."""
    if len(text) <= size:
        return [text]
    chunks = []
    step = size - 100  # 100-char overlap
    for i in range(0, len(text), step):
        chunks.append(text[i : i + size])
        if i + size >= len(text):
            break
    return chunks


class VectorStore:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._client: chromadb.PersistentClient | None = None
        self._collection = None
        self._model: SentenceTransformer | None = None

    def connect(self) -> "VectorStore":
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._data_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._model = SentenceTransformer(_EMBED_MODEL)
        return self

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        pass

    @property
    def collection(self):
        if not self._collection:
            raise RuntimeError("VectorStore not connected")
        return self._collection

    @property
    def model(self) -> SentenceTransformer:
        if not self._model:
            raise RuntimeError("VectorStore not connected")
        return self._model

    # ── write ──────────────────────────────────────────────────────────────

    def index_message(self, msg: Message):
        if not msg.content.strip():
            return

        chunks = _chunk_text(msg.content)
        ids = [f"{msg.id}::{i}" for i in range(len(chunks))]
        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()
        metadatas = [
            {
                "message_id": msg.id,
                "conversation_id": msg.conversation_id,
                "role": msg.role.value,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

    def index_messages_batch(self, messages: list[Message], progress=None):
        for msg in messages:
            self.index_message(msg)
            if progress:
                progress.advance(1)

    # ── search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 10,
        role_filter: Role | None = None,
        conversation_ids: list[str] | None = None,
    ) -> list[VectorHit]:
        where: dict = {}
        if role_filter:
            where["role"] = role_filter.value
        if conversation_ids:
            where["conversation_id"] = {"$in": conversation_ids}

        embedding = self.model.encode([query], show_progress_bar=False)[0].tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(limit, self.collection.count() or 1),
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                VectorHit(
                    message_id=meta["message_id"],
                    conversation_id=meta["conversation_id"],
                    chunk_text=doc,
                    score=dist,
                )
            )

        return hits

    def count(self) -> int:
        return self.collection.count()
