"""Hybrid retrieval: dense embeddings + BM25, fused with reciprocal rank fusion,
optionally reranked by a cross-encoder.

`Hit.score` is the relevance used for missing-evidence decisions: the reranker
score when a reranker is configured, otherwise the dense cosine similarity.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking import Chunk

_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "be",
    "it", "this", "that", "as", "at", "by", "from", "i", "my", "me", "you", "your", "do", "does",
    "how", "what", "can", "should", "about", "so", "if", "not", "no",
}
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class Embedder(Protocol):
    name: str

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self._query_prefix = _BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self._model.encode(self._query_prefix + text, normalize_embeddings=True), dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words embedder. For tests and offline smoke runs only."""

    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        for t in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
            h = zlib.crc32(t.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def make_embedder(name: str) -> Embedder:
    return HashingEmbedder() if name == "hashing" else SentenceTransformerEmbedder(name)


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(query, p) for p in passages])]


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


@dataclass
class Hit:
    chunk: Chunk
    score: float  # relevance used for abstention
    dense: float
    bm25: float


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedder: Embedder,
        reranker: Reranker | None = None,
        mode: str = "hybrid",
        candidates: int = 20,
        top_k: int = 4,
        rrf_k: int = 60,
    ):
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        self.chunks, self.embeddings, self.embedder = chunks, embeddings, embedder
        self.reranker, self.mode = reranker, mode
        self.candidates, self.top_k, self.rrf_k = candidates, top_k, rrf_k
        self._bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])

    @classmethod
    def from_index(cls, index_dir, embedder: Embedder, **kwargs) -> "HybridRetriever":
        from rag.ingest import load_index

        chunks, embeddings, meta = load_index(index_dir)
        if meta.get("embedder") != embedder.name:
            raise RuntimeError(
                f"Index was built with embedder '{meta.get('embedder')}' but '{embedder.name}' is configured. "
                "Rebuild with: python -m rag.ingest"
            )
        return cls(chunks, embeddings, embedder, **kwargs)

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.top_k
        n = len(self.chunks)
        c = min(self.candidates, n)
        dense = self.embeddings @ self.embedder.encode_query(query)
        bm = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)

        dense_rank = [int(i) for i in np.argsort(-dense)[:c]]
        bm_rank = [int(i) for i in np.argsort(-bm)[:c] if bm[i] > 0]

        if self.mode == "dense":
            cand = dense_rank
        elif self.mode == "bm25":
            cand = bm_rank
        else:
            fused = rrf([dense_rank, bm_rank], self.rrf_k)
            cand = sorted(fused, key=fused.get, reverse=True)[:c]
        if not cand:
            return []

        if self.reranker:
            scores = self.reranker.score(query, [self.chunks[i].embed_text for i in cand])
            order = sorted(range(len(cand)), key=lambda j: scores[j], reverse=True)
            ranked = [(cand[j], float(scores[j])) for j in order]
        else:
            ranked = [(i, float(dense[i])) for i in cand]

        return [Hit(self.chunks[i], s, float(dense[i]), float(bm[i])) for i, s in ranked[:top_k]]
