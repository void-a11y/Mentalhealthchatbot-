import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.ingest import build_chunks, load_documents  # noqa: E402
from rag.retrieval import HashingEmbedder, HybridRetriever  # noqa: E402

FIXTURE_DATA = ROOT / "tests" / "fixtures" / "data"


@pytest.fixture(scope="session")
def chunks():
    return build_chunks(load_documents(FIXTURE_DATA), "structure", 220, 40)


@pytest.fixture(scope="session")
def retriever(chunks):
    emb = HashingEmbedder()
    return HybridRetriever(chunks, emb.encode_passages([c.embed_text for c in chunks]), emb, top_k=3)
