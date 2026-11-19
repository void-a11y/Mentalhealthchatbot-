from rag.answer import NOT_FOUND_MESSAGE, Answerer


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, system, user):
        self.calls += 1
        self.last_user = user
        return self.reply


def make(retriever, reply, min_relevance=-1.0):
    llm = FakeLLM(reply)
    return Answerer(retriever, llm, min_relevance), llm


def test_grounded_answer_returns_citations(retriever):
    ans, _ = make(retriever, "Keep a regular bedtime [1].")
    out = ans.answer("how do I improve my sleep")
    assert not out.abstained and out.reason == "answered"
    assert [c.n for c in out.citations] == [1]
    assert out.citations[0].source == "sleep.md"


def test_low_relevance_abstains_without_calling_llm(retriever):
    ans, llm = make(retriever, "should not be used [1]", min_relevance=99.0)
    out = ans.answer("how do I improve my sleep")
    assert out.abstained and out.reason == "low_relevance"
    assert out.text == NOT_FOUND_MESSAGE
    assert llm.calls == 0


def test_llm_not_found_abstains(retriever):
    ans, _ = make(retriever, "NOT_FOUND")
    out = ans.answer("what is the capital of France")
    assert out.abstained and out.reason == "llm_not_found"


def test_answer_without_citation_is_rejected(retriever):
    ans, _ = make(retriever, "Sleep more and you will be fine.")
    out = ans.answer("how do I improve my sleep")
    assert out.abstained and out.reason == "no_valid_citation"


def test_invalid_citation_numbers_are_stripped_or_rejected(retriever):
    ans, _ = make(retriever, "Fake claim [99].")
    assert ans.answer("how do I improve my sleep").reason == "no_valid_citation"

    ans, _ = make(retriever, "Real claim [1] and fake claim [99].")
    out = ans.answer("how do I improve my sleep")
    assert "[99]" not in out.text and [c.n for c in out.citations] == [1]


def test_citation_formats_are_normalised(retriever):
    ans, _ = make(retriever, "Claim one [1, 2]. Claim two [Source 1]. Fake [2, 99].")
    out = ans.answer("how do I improve my sleep")
    assert not out.abstained
    assert [c.n for c in out.citations] == [1, 2]
    assert "[1][2]" in out.text and "[Source" not in out.text and "99" not in out.text


def test_fullwidth_bracket_citations_are_accepted(retriever):
    # Seen in real gpt-oss output: citations written as 【1】, sometimes with a 【1†L2-L4】 suffix.
    ans, _ = make(retriever, "Keep a regular bedtime【1】. Avoid caffeine late in the day【2†L3-L5】. Fake【99】.")
    out = ans.answer("how do I improve my sleep")
    assert not out.abstained and out.reason == "answered"
    assert [c.n for c in out.citations] == [1, 2]
    assert out.text.count("[1]") == 1 and "[2]" in out.text
    assert "【" not in out.text and "99" not in out.text


def test_raw_output_is_kept_for_debugging(retriever):
    ans, _ = make(retriever, "Sleep more.")
    out = ans.answer("how do I improve my sleep")
    assert out.abstained and out.raw == "Sleep more."


def test_prompt_marks_sources_as_data_and_numbers_them(retriever):
    ans, llm = make(retriever, "ok [1]")
    ans.answer("grounding technique")
    assert llm.last_user.startswith("Sources:\n[1] (")
