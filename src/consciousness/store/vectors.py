"""Vector embedding store — exchange-level chunking for semantic search.

Each indexed unit is a Q+A exchange (human question + assistant response),
giving the embeddings richer context than raw character-window chunks.
Human and assistant halves are indexed separately so role filtering still works,
but each carries its counterpart as context in the document text.

ChromaDB handles persistence; sentence-transformers provides local embeddings.
The encoder is lazy-loaded on first use and can be injected for tests.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb
import numpy as np
from chromadb.config import Settings

from consciousness.models import Conversation, Message, Role

_EMBED_MODEL = "all-MiniLM-L6-v2"
_COLLECTION = "messages"
_CHUNK_SIZE = 800       # chars per chunk within a single message
_SCORE_THRESHOLD = 0.8  # cosine distance; above this = likely irrelevant


class Encoder(Protocol):
    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray: ...


def _build_fake_encoder() -> Encoder:
    """Deterministic bag-of-words encoder for network-free environments."""
    import hashlib

    _dims = 384

    class _Fake:
        def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
            vecs = []
            for text in texts:
                vec = np.zeros(_dims, dtype=np.float32)
                for word in text.lower().split():
                    vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % _dims] += 1.0
                norm = np.linalg.norm(vec)
                vecs.append(vec / norm if norm > 0 else vec)
            return np.array(vecs)

    return _Fake()


@dataclass
class VectorHit:
    exchange_id: str
    conversation_id: str
    chunk_text: str
    score: float          # cosine distance: 0 = identical, lower = better
    human_message_id: str
    assistant_message_id: str | None = None

    @property
    def relevance_label(self) -> str:
        if self.score < 0.35:
            return "high"
        if self.score < 0.60:
            return "medium"
        return "low"

    @property
    def is_relevant(self) -> bool:
        return self.score < _SCORE_THRESHOLD


def _smart_chunk(text: str, size: int = _CHUNK_SIZE) -> list[str]:
    """Split at paragraph or sentence boundaries; fall back to hard split."""
    if len(text) <= size:
        return [text]

    # Try paragraph splits first
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paras:
        if current_len + len(para) > size and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_len = len(para)
        else:
            current_parts.append(para)
            current_len += len(para)
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # If any chunk is still over size, split at sentence boundaries
    refined: list[str] = []
    for chunk in chunks:
        if len(chunk) <= size:
            refined.append(chunk)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            buf, buf_len = [], 0
            for sent in sentences:
                if buf_len + len(sent) > size and buf:
                    refined.append(" ".join(buf))
                    buf, buf_len = [sent], len(sent)
                else:
                    buf.append(sent)
                    buf_len += len(sent)
            if buf:
                refined.append(" ".join(buf))

    return refined or [text[:size]]


def _pair_exchanges(messages: list[Message]) -> list[tuple[Message, Message | None]]:
    """Pair consecutive human+assistant messages into exchanges."""
    pairs: list[tuple[Message, Message | None]] = []
    i = 0
    while i < len(messages):
        human = messages[i]
        if human.role == Role.human:
            assistant = messages[i + 1] if i + 1 < len(messages) and messages[i + 1].role == Role.assistant else None
            pairs.append((human, assistant))
            i += 2 if assistant else 1
        else:
            # Orphaned assistant message (rare in practice)
            pairs.append((messages[i], None))
            i += 1
    return pairs


class VectorStore:
    def __init__(self, data_dir: Path, encoder: Encoder | None = None):
        self._data_dir = data_dir
        self._client: chromadb.PersistentClient | None = None
        self._collection: Any = None
        self._encoder: Encoder | None = encoder
        self._model_loaded = encoder is not None

    def connect(self) -> "VectorStore":
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._data_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=_COLLECTION, metadata={"hnsw:space": "cosine"})
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
    def model(self) -> Encoder:
        if not self._model_loaded:
            import os

            if os.environ.get("CONSCIOUSNESS_FAKE_ENCODER"):
                self._encoder = _build_fake_encoder()
            else:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(_EMBED_MODEL)
            self._model_loaded = True
        return self._encoder

    # ── write ──────────────────────────────────────────────────────────────────

    def index_conversation(self, conv: Conversation):
        """Index all exchanges in a conversation. This is the primary ingest method."""
        for ex_idx, (human, assistant) in enumerate(_pair_exchanges(conv.messages)):
            self._index_exchange(human, assistant, conv.title, ex_idx)

    def _index_exchange(self, human_msg: Message, assistant_msg: Message | None, conv_title: str, exchange_index: int):
        exchange_id = f"{human_msg.conversation_id}::ex{exchange_index}"
        asst_id = assistant_msg.id if assistant_msg else ""

        ids, docs, metadatas = [], [], []

        if human_msg.content.strip():
            q_text = f"Q: {human_msg.content}"
            for ci, chunk in enumerate(_smart_chunk(q_text)):
                ids.append(f"{exchange_id}::q{ci}")
                docs.append(chunk)
                metadatas.append({
                    "exchange_id": exchange_id,
                    "conversation_id": human_msg.conversation_id,
                    "role": "human",
                    "human_message_id": human_msg.id,
                    "assistant_message_id": asst_id,
                    "conv_title": conv_title,
                })

        if assistant_msg and assistant_msg.content.strip():
            ctx = f"[Q: {human_msg.content[:200]}]\n" if human_msg else ""
            a_text = f"{ctx}A: {assistant_msg.content}"
            for ci, chunk in enumerate(_smart_chunk(a_text)):
                ids.append(f"{exchange_id}::a{ci}")
                docs.append(chunk)
                metadatas.append({
                    "exchange_id": exchange_id,
                    "conversation_id": human_msg.conversation_id,
                    "role": "assistant",
                    "human_message_id": human_msg.id,
                    "assistant_message_id": asst_id,
                    "conv_title": conv_title,
                })

        if not ids:
            return

        embeddings = self.model.encode(docs, show_progress_bar=False).tolist()
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)

    def index_conversations_batch(self, conversations: list[Conversation], progress=None):
        for conv in conversations:
            self.index_conversation(conv)
            if progress:
                progress.advance(1)

    # ── search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 10,
        role_filter: Role | None = None,
        conversation_ids: list[str] | None = None,
        score_threshold: float = _SCORE_THRESHOLD,
    ) -> list[VectorHit]:
        where: dict = {}
        if role_filter:
            where["role"] = role_filter.value
        if conversation_ids:
            where["conversation_id"] = {"$in": conversation_ids}

        embedding = self.model.encode([query], show_progress_bar=False)[0].tolist()
        n = min(limit * 3, self.collection.count() or 1)  # over-fetch before threshold + dedup

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[VectorHit] = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        seen_exchanges: set[str] = set()
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            if dist >= score_threshold:
                continue
            ex_id = meta["exchange_id"]
            if ex_id in seen_exchanges:
                continue
            seen_exchanges.add(ex_id)
            hits.append(VectorHit(
                exchange_id=ex_id,
                conversation_id=meta["conversation_id"],
                chunk_text=doc,
                score=dist,
                human_message_id=meta["human_message_id"],
                assistant_message_id=meta.get("assistant_message_id") or None,
            ))
            if len(hits) >= limit:
                break

        return hits

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        """Delete all indexed data — used by rebuild-index."""
        self._client.delete_collection(_COLLECTION)
        self._collection = self._client.get_or_create_collection(name=_COLLECTION, metadata={"hnsw:space": "cosine"})
