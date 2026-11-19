"""Download the pages listed in scripts/corpus_urls.txt and convert them to markdown.

Usage:  python scripts/fetch_corpus.py [--urls scripts/corpus_urls.txt] [--out data]

Writes data/<slug>.md, data/sources.json (titles + URLs for citations) and
data/SOURCES.md (origin/license table). sources.json and SOURCES.md are
regenerated on every run from the URL list. Failed URLs are reported and skipped;
pages with identical content are written once.

Navigation, clinical-trial links, repeated crisis-line text and other boilerplate
are removed (see clean_markdown). Review the generated files before indexing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "Mozilla/5.0 (research corpus builder; educational project)"}

# Verify each policy yourself before publishing the repo.
LICENSES = {
    "www.nimh.nih.gov": "US Government work, public domain unless noted (check NIMH reprint policy)",
    "www.cdc.gov": "US Government work, public domain unless noted",
}

_DROP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "button", "svg", "noscript", "iframe"]
_NOISE = re.compile(r"breadcrumb|social|share|skip|cookie|feedback|pagination|toolbar", re.I)

# Whole sections (heading and everything under it) that are navigation or repeated boilerplate.
_DROP_SECTIONS = re.compile(
    r"^(where can i learn more|explore clinical trials|share outreach|digital shareables|additional federal resources|"
    r"more resources|featured|learn more|reprints|for more information|find help and support|health hotlines|"
    r"what are clinical trials|how can i learn more)",
    re.I,
)
# Single lines to drop.
_DROP_LINES = re.compile(
    r"^\W*(download pdf|en español|esta página|la información sobre|view the related infographic|last reviewed|"
    r"unless otherwise specified|u\.s\. department of health|national institutes of health|nih publication|revised \d{4})",
    re.I,
)
# Crisis-line text is handled by the app's safety layer, not by retrieval.
_CRISIS_LINE = re.compile(r"988 suicide|988lifeline", re.I)


def clean_markdown(md: str) -> str:
    out: list[str] = []
    skip_level: int | None = None
    in_toc = False
    for line in md.splitlines():
        m = re.match(r"^(#{2,6})\s+(.*)", line)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and _DROP_SECTIONS.match(text):
                skip_level = level
            if skip_level is not None:
                continue
            in_toc = False
            out.append(line)
            continue
        if skip_level is not None:
            continue
        s = line.strip()
        if s.strip("*").strip().lower() == "on this page":
            in_toc = True
            continue
        if in_toc:
            if s == "" or s.startswith(("* ", "- ")):
                continue
            in_toc = False
        if _DROP_LINES.match(s) or _CRISIS_LINE.search(s):
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def html_to_markdown(html: str) -> tuple[str, str]:
    """Return (title, markdown) for the main content of a page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    for el in soup.find_all(class_=_NOISE) + soup.find_all(id=_NOISE):
        if not el.decomposed:
            el.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    h1 = root.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "Untitled")

    md = markdownify(str(root), heading_style="ATX", strip=["a", "img"])
    md = md.replace("\\_", "_").replace("\\*", "*")
    lines = [ln.rstrip() for ln in md.splitlines() if not re.match(r"^#\s", ln)]  # we add our own H1
    body = clean_markdown(re.sub(r"\n{3,}", "\n\n", "\n".join(lines)))
    if len(body) < 200:
        raise ValueError("extracted text is suspiciously short")
    return title, f"# {title}\n\n{body}\n"


def slugify(url: str, taken: set[str]) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p and not p.endswith(".shtml")]
    slug = re.sub(r"[^a-z0-9]+", "-", (parts[-1] if parts else "page").lower()).strip("-") or "page"
    base, n = slug, 2
    while slug in taken:
        slug, n = f"{base}-{n}", n + 1
    taken.add(slug)
    return slug


def read_urls(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def write_sources_table(out: Path, meta: dict) -> None:
    rows = ["# Corpus sources\n", f"Fetched {date.today().isoformat()}. Verify each license before redistributing.\n",
            "| File | Title | URL | License |", "|---|---|---|---|"]
    for name, m in sorted(meta.items()):
        rows.append(f"| {name} | {m['title']} | {m['url']} | {m['license']} |")
    (out / "SOURCES.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default=str(ROOT / "scripts" / "corpus_urls.txt"))
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    taken: set[str] = set()
    seen_hash: dict[str, str] = {}
    ok, failed = 0, []

    for url in read_urls(Path(args.urls)):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            title, md = html_to_markdown(resp.text)
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append((url, str(e)))
            print(f"FAIL  {url}  ({e})")
            continue
        digest = hashlib.sha256(md.encode()).hexdigest()
        if digest in seen_hash:
            print(f"SKIP  {url}  (identical to {seen_hash[digest]})")
            continue
        name = f"{slugify(url, taken)}.md"
        seen_hash[digest] = name
        (out / name).write_text(md, encoding="utf-8")
        meta[name] = {"title": title, "url": url, "license": LICENSES.get(urlparse(url).netloc, "CHECK LICENSE")}
        print(f"ok    {name:55s} {len(md.split()):5d} words")
        ok += 1
        time.sleep(args.delay)

    (out / "sources.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_sources_table(out, meta)
    print(f"\n{ok} written, {len(failed)} failed. Review data/*.md, then run: python -m rag.ingest")
    sys.exit(1 if failed and not ok else 0)


if __name__ == "__main__":
    main()
