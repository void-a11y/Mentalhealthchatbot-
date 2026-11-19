"""Central settings, read from environment (.env supported) at call time."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    index_dir: Path
    log_dir: Path
    embed_model: str
    rerank_model: str
    use_reranker: bool
    candidates: int
    top_k: int
    min_relevance: float
    chunk_strategy: str
    chunk_max_words: int
    chunk_overlap_words: int
    groq_api_key: str | None
    groq_model: str
    groq_base_url: str
    cors_origins: list[str]
    log_text: bool
    max_sessions: int


def load_settings() -> Settings:
    use_reranker = _bool("USE_RERANKER", True)
    # Cross-encoder logits and dense cosine live on different scales, so the
    # default threshold differs. 0.35 is the value found by `python eval/run_eval.py`
    # for the default models on the bundled corpus; re-tune if you change models or data.
    default_min = 0.35 if use_reranker else 0.5
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", ROOT / "data")),
        index_dir=Path(os.getenv("INDEX_DIR", ROOT / "index")),
        log_dir=Path(os.getenv("LOG_DIR", ROOT / "logs")),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        rerank_model=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        use_reranker=use_reranker,
        candidates=int(os.getenv("RETRIEVE_CANDIDATES", "20")),
        top_k=int(os.getenv("TOP_K", "4")),
        min_relevance=float(os.getenv("MIN_RELEVANCE", default_min)),
        chunk_strategy=os.getenv("CHUNK_STRATEGY", "structure"),
        chunk_max_words=int(os.getenv("CHUNK_MAX_WORDS", "220")),
        chunk_overlap_words=int(os.getenv("CHUNK_OVERLAP_WORDS", "40")),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        groq_base_url=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
        cors_origins=[
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
            if o.strip()
        ],
        log_text=_bool("LOG_TEXT", False),
        max_sessions=int(os.getenv("MAX_SESSIONS", "1000")),
    )
