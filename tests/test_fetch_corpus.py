import pytest

from scripts.fetch_corpus import html_to_markdown, read_urls, slugify

HTML = """
<html><head><title>Site | Anxiety</title></head><body>
<header><a href="/">Home</a> NAVIGATION</header>
<nav>Topics Menu</nav>
<main>
  <div class="breadcrumb">Home / Health</div>
  <h1>Anxiety Disorders</h1>
  <div class="social-share">Share on Facebook</div>
  <h2>Overview</h2>
  <p>Occasional anxiety is an expected part of life. People with <a href="/x">anxiety disorders</a>
  frequently have intense, excessive and persistent worry and fear about everyday situations.</p>
  <h2>Symptoms</h2>
  <ul><li>Feeling restless or on edge</li><li>Being easily fatigued</li><li>snake_case_term and 2*3</li></ul>
  <p>Treatments include psychotherapy and medication, and many people benefit from a combination of both approaches.</p>
</main>
<footer>Copyright footer text</footer>
</body></html>
"""


def test_extracts_main_content_and_title():
    title, md = html_to_markdown(HTML)
    assert title == "Anxiety Disorders"
    assert md.startswith("# Anxiety Disorders\n")
    assert md.count("\n# ") == 0  # exactly one H1
    assert "## Overview" in md and "## Symptoms" in md


def test_boilerplate_removed_and_links_flattened():
    _, md = html_to_markdown(HTML)
    for junk in ("NAVIGATION", "Topics Menu", "Share on Facebook", "Copyright footer", "Home / Health"):
        assert junk not in md
    assert "anxiety disorders" in md and "](" not in md


def test_no_stray_escapes():
    _, md = html_to_markdown(HTML)
    assert "snake_case_term" in md and "\\_" not in md and "2*3" in md


NIMH_STYLE = """
<html><body><main>
<h1>Depression</h1>
<ul><li>Download PDF</li><li>En español</li></ul>
<p><strong>On this page</strong></p>
<ul><li>What is depression?</li><li>Explore clinical trials</li></ul>
<h2>What is depression?</h2>
<p>Everyone feels sad or low sometimes, but depression is different. It can cause severe symptoms that
affect how you feel, think, and handle daily activities, such as sleeping, eating, or working.</p>
<p>If you or someone you know is struggling, call or text the 988 Suicide &amp; Crisis Lifeline at 988.</p>
<h2>How is depression treated?</h2>
<p>Treatment typically involves psychotherapy, medication, or both, chosen with a health care provider.</p>
<h3>Psychotherapy</h3>
<p>Cognitive behavioral therapy helps people question unhelpful thoughts and change behavior patterns.</p>
<h2>Explore clinical trials about depression</h2>
<p>Clinical trials are research studies. Visit clinicaltrials.gov.</p>
<h3>Digital shareables</h3>
<p>Social media graphics.</p>
<h2>Additional federal resources</h2>
<ul><li>MedlinePlus</li></ul>
<p><strong>Last Reviewed:</strong> December 2024</p>
<h2>Reprints</h2>
<p>The information in this publication is in the public domain.</p>
</main></body></html>
"""


def test_nimh_boilerplate_sections_and_lines_are_removed():
    _, md = html_to_markdown(NIMH_STYLE)
    for junk in ("On this page", "Download PDF", "En español", "clinical trials", "Digital shareables",
                 "MedlinePlus", "Last Reviewed", "Reprints", "988", "Social media graphics"):
        assert junk not in md, junk
    assert "## What is depression?" in md
    assert "## How is depression treated?" in md and "### Psychotherapy" in md
    assert "Cognitive behavioral therapy" in md


def test_too_short_page_is_rejected():
    with pytest.raises(ValueError):
        html_to_markdown("<html><body><main><h1>X</h1><p>tiny</p></main></body></html>")


def test_slugify_is_unique_and_clean():
    taken: set[str] = set()
    a = slugify("https://www.nimh.nih.gov/health/topics/anxiety-disorders", taken)
    b = slugify("https://other.org/x/anxiety-disorders", taken)
    assert a == "anxiety-disorders" and b == "anxiety-disorders-2"
    assert slugify("https://x.org/a/old-page.shtml", taken) == "a"


def test_read_urls_ignores_comments_and_blanks(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("# header\n\nhttps://a.org/1  # trailing\nhttps://a.org/2\n")
    assert read_urls(f) == ["https://a.org/1", "https://a.org/2"]
