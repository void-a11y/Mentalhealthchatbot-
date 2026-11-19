from rag.ingest import build_chunks, load_documents

DOC = "# Title\n\n## Section\nThis is a reasonably long paragraph about coping with stress through exercise and sleep."


def test_duplicate_documents_are_indexed_once(tmp_path):
    (tmp_path / "a.md").write_text(DOC, encoding="utf-8")
    (tmp_path / "b.md").write_text(DOC, encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 2
    chunks = build_chunks(docs, "structure", 220, 40)
    assert len(chunks) == 1 and chunks[0].source == "a.md"


def test_sources_json_attaches_title_and_url(tmp_path):
    (tmp_path / "a.md").write_text("# Heading\n\n## S\nSome body text for the document here.", encoding="utf-8")
    (tmp_path / "sources.json").write_text('{"a.md": {"title": "Custom Title", "url": "https://x.org/a"}}', encoding="utf-8")
    chunks = build_chunks(load_documents(tmp_path), "structure", 220, 40)
    assert chunks[0].title == "Custom Title" and chunks[0].url == "https://x.org/a"


def test_readme_and_sources_md_are_skipped(tmp_path):
    (tmp_path / "SOURCES.md").write_text("# s", encoding="utf-8")
    (tmp_path / "README.md").write_text("# r", encoding="utf-8")
    (tmp_path / "doc.md").write_text("# d\n\n## s\nbody text", encoding="utf-8")
    assert [d["source"] for d in load_documents(tmp_path)] == ["doc.md"]
