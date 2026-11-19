"""Chunking: structure-aware (default) and a fixed-window baseline for comparison.

Structure-aware: split on markdown headings, then pack paragraphs/sentences into
chunks of at most `max_words`, carrying whole trailing sentences as overlap.
Every chunk keeps its document title and heading path, which is prepended when
embedding so a chunk like "Take a break every hour" still knows it is about burnout.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    section: str
    text: str
    url: str = ""

    @property
    def embed_text(self) -> str:
        head = f"{self.title} > {self.section}" if self.section else self.title
        return f"{head}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _n(text: str) -> int:
    return len(text.split())


def parse_sections(text: str, fallback_title: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (title, [(heading_path, body), ...]). The first H1 is the title."""
    title, title_found = fallback_title, False
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(h for _, h in stack), content))
        body.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if not m:
            body.append(line)
            continue
        flush()
        level, heading = len(m.group(1)), m.group(2).strip()
        if level == 1 and not title_found and not stack:
            title, title_found = heading, True
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
    flush()
    return title, sections


def _units(body: str, max_words: int) -> list[str]:
    units: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = " ".join(para.split())
        if not para:
            continue
        if _n(para) <= max_words:
            units.append(para)
        else:
            units.extend(s for s in _SENTENCE.split(para) if s)
    return units


def _pack(units: list[str], max_words: int, overlap_words: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for unit in units:
        w = _n(unit)
        if cur and cur_n + w > max_words:
            chunks.append(" ".join(cur))
            carry: list[str] = []
            carry_n = 0
            for prev in reversed(cur):
                if carry_n + _n(prev) > overlap_words:
                    break
                carry.insert(0, prev)
                carry_n += _n(prev)
            cur, cur_n = carry, carry_n
        cur.append(unit)
        cur_n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _fixed(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = re.sub(r"^#{1,6}\s*", "", text, flags=re.M).split()
    step = max(1, max_words - overlap_words)
    out: list[str] = []
    for i in range(0, len(words), step):
        out.append(" ".join(words[i : i + max_words]))
        if i + max_words >= len(words):
            break
    return out


def _slug(source: str) -> str:
    return re.sub(r"\W+", "-", source.rsplit(".", 1)[0]).strip("-").lower()


def chunk_document(
    text: str,
    source: str,
    strategy: str = "structure",
    max_words: int = 220,
    overlap_words: int = 40,
    title: str | None = None,
    url: str = "",
) -> list[Chunk]:
    fallback = title or Path(source).stem.replace("-", " ").replace("_", " ").title()
    doc_title, sections = parse_sections(text, fallback)
    doc_title = title or doc_title
    slug = _slug(source)
    chunks: list[Chunk] = []

    if strategy == "fixed":
        pieces = [("", p) for p in _fixed(text, max_words, overlap_words)]
    elif strategy == "structure":
        pieces = [
            (section, piece)
            for section, body in sections
            for piece in _pack(_units(body, max_words), max_words, overlap_words)
        ]
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    for i, (section, piece) in enumerate(pieces):
        chunks.append(Chunk(f"{slug}-{i:03d}", source, doc_title, section, piece, url))
    return chunks
