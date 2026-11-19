import numpy as np

from rag.retrieval import HybridRetriever, rrf


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert max(fused, key=fused.get) == 1
    assert set(fused) == {1, 2, 3, 4}


def test_hybrid_finds_the_right_document(retriever):
    hits = retriever.search("how can I fall asleep without screens in bed")
    assert hits[0].chunk.source == "sleep.md"


def test_each_mode_returns_results(chunks):
    from rag.retrieval import HashingEmbedder

    emb = HashingEmbedder()
    vecs = emb.encode_passages([c.embed_text for c in chunks])
    for mode in ("dense", "bm25", "hybrid"):
        r = HybridRetriever(chunks, vecs, emb, mode=mode, top_k=2)
        assert r.search("grounding technique five things you can see")[0].chunk.source == "anxiety.md"


def test_reranker_can_reorder_and_sets_score(chunks):
    from rag.retrieval import HashingEmbedder

    class PreferBurnout:
        def score(self, query, passages):
            return [10.0 if "burnout" in p.lower() else -10.0 for p in passages]

    emb = HashingEmbedder()
    vecs = emb.encode_passages([c.embed_text for c in chunks])
    r = HybridRetriever(chunks, vecs, emb, reranker=PreferBurnout(), top_k=3)
    hits = r.search("tiredness and stress")
    assert hits[0].chunk.source == "burnout.md"
    assert hits[0].score == 10.0


def test_bm25_mode_returns_nothing_without_term_overlap(chunks):
    from rag.retrieval import HashingEmbedder

    emb = HashingEmbedder()
    vecs = emb.encode_passages([c.embed_text for c in chunks])
    r = HybridRetriever(chunks, vecs, emb, mode="bm25")
    assert r.search("xylophone quasar") == []


def test_index_roundtrip_and_embedder_mismatch(tmp_path, chunks):
    import pytest

    from rag.ingest import build_index
    from rag.retrieval import HashingEmbedder

    from tests.conftest import FIXTURE_DATA

    emb = HashingEmbedder()
    build_index(FIXTURE_DATA, tmp_path, emb)
    r = HybridRetriever.from_index(tmp_path, emb)
    assert len(r.chunks) == len(chunks)

    class Other(HashingEmbedder):
        name = "other"

    with pytest.raises(RuntimeError):
        HybridRetriever.from_index(tmp_path, Other())
    assert isinstance(r.embeddings, np.ndarray)
