"""
tools/rag_engine.py
Real RAG implementation using sentence-transformers + FAISS.

Architecture:
  Documents (.txt / .md) in knowledge_base/
      ↓  chunk()
  Text chunks (with source metadata)
      ↓  embed()   ← all-MiniLM-L6-v2 (fast, 384-dim)
  Numpy vectors
      ↓  faiss.IndexFlatIP (inner-product / cosine after normalisation)
  Persistent index saved to knowledge_base/.rag_index/
      ↓  search(query, top_k)
  Ranked results [{"content", "source", "score"}, ...]

Usage:
    engine = get_engine()          # builds/loads index automatically
    results = engine.search("your query", top_k=3)
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KB_DIR      = Path(__file__).parent.parent / "knowledge_base"
_INDEX_DIR   = _KB_DIR / ".rag_index"
_INDEX_FILE  = _INDEX_DIR / "faiss.index"
_META_FILE   = _INDEX_DIR / "metadata.pkl"
_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE    = 400   # characters per chunk
CHUNK_OVERLAP = 80    # overlap between consecutive chunks


class Chunk(TypedDict):
    content: str
    source:  str   # e.g. "rag:python313_performance.txt"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source: str) -> list[Chunk]:
    """Split a document into overlapping character-level windows."""
    text = text.strip()
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"content": chunk_text, "source": source})
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _load_documents(directory: Path) -> list[Chunk]:
    """Load all .txt and .md files from the knowledge base directory."""
    chunks: list[Chunk] = []
    supported = {".txt", ".md"}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in supported and path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
                source = f"rag:{path.name}"
                chunks.extend(_chunk_text(text, source))
            except Exception as exc:
                print(f"[RAG] Warning: could not read {path.name}: {exc}")
    return chunks


# ---------------------------------------------------------------------------
# Embedding — singleton model, shared by memory.py and rag_engine.py
# ---------------------------------------------------------------------------

def _get_model():
    """Lazy-load the sentence-transformer model (cached after first call)."""
    if not hasattr(_get_model, "_instance"):
        from sentence_transformers import SentenceTransformer
        print(f"[RAG] Loading embedding model: {_EMBED_MODEL} …", flush=True)
        _get_model._instance = SentenceTransformer(_EMBED_MODEL)
        print("[RAG] Model ready.", flush=True)
    return _get_model._instance


def _embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings (shape: N × D)."""
    model = _get_model()
    vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype("float32")


# ---------------------------------------------------------------------------
# FAISS index management
# ---------------------------------------------------------------------------

class RAGEngine:
    """In-process RAG engine backed by a FAISS flat inner-product index."""

    def __init__(self):
        self._index = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and build a fresh FAISS index."""
        import faiss

        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")

        print(f"[RAG] Embedding {len(chunks)} chunks …", flush=True)
        texts  = [c["content"] for c in chunks]
        vecs   = _embed(texts)
        dim    = vecs.shape[1]

        index  = faiss.IndexFlatIP(dim)
        index.add(vecs)

        self._index  = index
        self._chunks = chunks

        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(_INDEX_FILE))
        with open(_META_FILE, "wb") as f:
            pickle.dump(chunks, f)

        print(f"[RAG] Index built: {len(chunks)} chunks, dim={dim}. Saved to {_INDEX_DIR}", flush=True)

    def load(self) -> bool:
        """Try to load a persisted index. Returns True on success."""
        import faiss
        if _INDEX_FILE.exists() and _META_FILE.exists():
            try:
                self._index = faiss.read_index(str(_INDEX_FILE))
                with open(_META_FILE, "rb") as f:
                    self._chunks = pickle.load(f)
                print(f"[RAG] Loaded existing index: {len(self._chunks)} chunks.", flush=True)
                return True
            except Exception as exc:
                print(f"[RAG] Warning: could not load index ({exc}). Rebuilding …", flush=True)
        return False

    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top-k most similar chunks for a query."""
        if not self.is_ready():
            return []

        q_vec = _embed([query])
        k     = min(top_k, len(self._chunks))
        scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append({
                "content": chunk["content"],
                "source":  chunk["source"],
                "score":   round(float(score), 4),
            })
        return results

    def add_text(self, text: str, source: str) -> int:
        """Add a new document to the index at runtime."""
        import faiss
        new_chunks = _chunk_text(text, source)
        if not new_chunks:
            return 0

        vecs = _embed([c["content"] for c in new_chunks])

        if not self.is_ready():
            dim          = vecs.shape[1]
            self._index  = faiss.IndexFlatIP(dim)
            self._chunks = []

        self._index.add(vecs)
        self._chunks.extend(new_chunks)

        faiss.write_index(self._index, str(_INDEX_FILE))
        with open(_META_FILE, "wb") as f:
            pickle.dump(self._chunks, f)

        return len(new_chunks)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: RAGEngine | None = None


def get_engine(force_rebuild: bool = False) -> RAGEngine:
    """Return the shared RAGEngine instance, building the index if needed."""
    global _engine
    if _engine is None or force_rebuild:
        eng = RAGEngine()
        loaded = (not force_rebuild) and eng.load()
        if not loaded:
            docs = _load_documents(_KB_DIR)
            if docs:
                eng.build(docs)
            else:
                print(
                    "[RAG] Warning: knowledge_base/ is empty. "
                    "Add .txt or .md files and call get_engine(force_rebuild=True).",
                    flush=True,
                )
        _engine = eng
    return _engine
