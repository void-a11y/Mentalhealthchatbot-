"""Ingestion: load documents, chunk, embed, and persist the index.

Usage:  python -m rag.ingest [--data data] [--index index] [--strategy structure]

Supported inputs: .md and .txt (convert PDFs/web pages to markdown first so
headings survive; structure-aware chunking depends on them). Optional
`sources.json` in the data folder maps a file to {"title", "url", "license"}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from rag.chunking import Chunk, chunk_document  # noqa: E402
from rag.retrieval import make_embedder  # noqa: E402

_TEXT_EXT = {".md", ".txt"}
_SKIP = {"readme.md", "sources.md"}


def load_documents(data_dir: Path) -> list[dict]:
    data_dir = Path(data_dir)
    meta_path = data_dir / "sources.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in _TEXT_EXT or path.name.lower() in _SKIP:
            continue
        source = path.relative_to(data_dir).as_posix()
        docs.append(
            {
                "source": source,
                "text": path.read_text(encoding="utf-8"),
                "title": meta.get(source, {}).get("title"),
                "url": meta.get(source, {}).get("url", ""),
            }
        )
    return docs


def build_chunks(docs: list[dict], strategy: str, max_words: int, overlap_words: int) -> list[Chunk]:
    """Chunk all documents, dropping chunks whose text duplicates an earlier one
    (repeated boilerplate or a document ingested twice would otherwise fill the top-k)."""
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for d in docs:
        for c in chunk_document(d["text"], d["source"], strategy, max_words, overlap_words, d.get("title"), d.get("url", "")):
            key = " ".join(c.text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            chunks.append(c)
    return chunks


def build_index(data_dir, index_dir, embedder, strategy="structure", max_words=220, overlap_words=40) -> list[Chunk]:
    docs = load_documents(data_dir)
    if not docs:
        raise SystemExit(f"No .md/.txt documents found in {data_dir}. Add your corpus first.")
    chunks = build_chunks(docs, strategy, max_words, overlap_words)
    embeddings = embedder.encode_passages([c.embed_text for c in chunks])
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "meta.json").write_text(
        json.dumps({"embedder": embedder.name, "strategy": strategy, "chunks": len(chunks), "docs": len(docs)}, indent=2),
        encoding="utf-8",
    )
    return chunks


def load_index(index_dir) -> tuple[list[Chunk], np.ndarray, dict]:
    index_dir = Path(index_dir)
    chunks = [Chunk(**json.loads(line)) for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    embeddings = np.load(index_dir / "embeddings.npy")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return chunks, embeddings, meta


def main() -> None:
    s = load_settings()
    ap = argparse.ArgumentParser(description="Build the retrieval index.")
    ap.add_argument("--data", default=str(s.data_dir))
    ap.add_argument("--index", default=str(s.index_dir))
    ap.add_argument("--strategy", default=s.chunk_strategy, choices=["structure", "fixed"])
    ap.add_argument("--embedder", default=s.embed_model, help="model name, or 'hashing' for offline tests")
    args = ap.parse_args()
    chunks = build_index(args.data, args.index, make_embedder(args.embedder), args.strategy, s.chunk_max_words, s.chunk_overlap_words)
    print(f"Indexed {len(chunks)} chunks from {args.data} -> {args.index} ({args.strategy})")


if __name__ == "__main__":
    main()
