from rag.chunking import chunk_document, parse_sections

DOC = """# Title Here

## First
Sentence one. Sentence two. Sentence three.

### Nested
Nested body text.

## Second
Second body.
"""


def test_title_and_heading_paths():
    title, sections = parse_sections(DOC, "fallback")
    assert title == "Title Here"
    assert [s for s, _ in sections] == ["First", "First > Nested", "Second"]


def test_chunks_carry_source_title_and_section():
    chunks = chunk_document(DOC, "dir/doc.md")
    assert {c.source for c in chunks} == {"dir/doc.md"}
    assert chunks[0].title == "Title Here"
    assert chunks[0].section == "First"
    assert chunks[0].id.startswith("dir-doc-")
    assert "Title Here > First" in chunks[0].embed_text


def test_chunks_never_cross_sections():
    chunks = chunk_document(DOC, "d.md")
    assert not any("Second body" in c.text and "Nested body" in c.text for c in chunks)


def test_size_limit_and_sentence_overlap():
    body = " ".join(f"Sentence number {i} has exactly seven words." for i in range(40))
    chunks = chunk_document(f"# T\n## S\n{body}", "d.md", max_words=50, overlap_words=16)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 50 for c in chunks)
    # the last sentence of a chunk is repeated at the start of the next one
    last_sentence = chunks[0].text.split(". ")[-1].rstrip(".")
    assert last_sentence in chunks[1].text


def test_fixed_strategy_ignores_headings():
    chunks = chunk_document(DOC, "d.md", strategy="fixed", max_words=10, overlap_words=2)
    assert all(c.section == "" for c in chunks)
    assert not any("#" in c.text for c in chunks)


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError):
        chunk_document(DOC, "d.md", strategy="nope")
