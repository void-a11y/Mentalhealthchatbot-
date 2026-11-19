"""Builds the production Answerer once (lazily) from settings."""
from __future__ import annotations

from functools import lru_cache

from config import load_settings
from rag.answer import Answerer, GroqLLM
from rag.retrieval import CrossEncoderReranker, HybridRetriever, make_embedder


@lru_cache(maxsize=1)
def get_answerer() -> Answerer:
    s = load_settings()
    if not (s.index_dir / "chunks.jsonl").exists():
        raise RuntimeError("Index not found. Add documents to data/ and run: python -m rag.ingest")
    retriever = HybridRetriever.from_index(
        s.index_dir,
        make_embedder(s.embed_model),
        reranker=CrossEncoderReranker(s.rerank_model) if s.use_reranker else None,
        candidates=s.candidates,
        top_k=s.top_k,
    )
    llm = GroqLLM(s.groq_api_key, s.groq_model, s.groq_base_url)
    return Answerer(retriever, llm, s.min_relevance)
