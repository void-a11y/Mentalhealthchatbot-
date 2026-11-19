import pytest
from fastapi.testclient import TestClient

import main
from rag.answer import Answerer
from sessions import SessionStore


class FakeLLM:
    def __init__(self, reply="Keep a regular bedtime [1]."):
        self.reply, self.calls = reply, 0

    def complete(self, system, user):
        self.calls += 1
        return self.reply


@pytest.fixture
def client(retriever, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    main.store = SessionStore()
    llm = FakeLLM()
    answerer = Answerer(retriever, llm, min_relevance=-1.0)
    main.app.dependency_overrides[main.answerer_provider] = lambda: (lambda: answerer)
    c = TestClient(main.app)
    c.llm, c.log_dir = llm, tmp_path
    yield c
    main.app.dependency_overrides.clear()


def post(client, query, sid="s1"):
    return client.post("/chat", json={"session_id": sid, "query": query})


def test_answered_route_returns_sources(client):
    r = post(client, "how do I improve my sleep").json()
    assert r["route"] == "answered" and r["sources"][0]["source"] == "sleep.md"


def test_crisis_bypasses_llm_and_retrieval(client):
    r = post(client, "I want to kill myself").json()
    assert r["route"] == "crisis" and "14416" in r["response"]
    assert client.llm.calls == 0


def test_crisis_works_even_if_pipeline_is_down(client):
    def broken():
        raise RuntimeError("Index not found")

    main.app.dependency_overrides[main.answerer_provider] = lambda: broken
    assert post(client, "I want to die").json()["route"] == "crisis"
    assert post(client, "how do I improve my sleep", sid="other").status_code == 503


def test_crisis_context_carries_over_within_session(client):
    post(client, "I want to kill myself")
    assert post(client, "what is a good bedtime routine").json()["route"] == "crisis"
    assert post(client, "what is a good bedtime routine", sid="fresh").json()["route"] == "answered"


def test_out_of_scope_redirect(client):
    r = post(client, "how many mg of sertraline should I take").json()
    assert r["route"] == "out_of_scope" and client.llm.calls == 0


def test_not_found_when_llm_says_so(client):
    client.llm.reply = "NOT_FOUND"
    r = post(client, "what is the capital of France").json()
    assert r["route"] == "not_found" and r["abstained"] and r["sources"] == []


def test_legacy_doc_chat_route_still_works(client):
    r = client.post("/doc-chat", json={"session_id": "s", "query": "how do I improve my sleep"})
    assert r.status_code == 200


def test_input_validation(client):
    assert client.post("/chat", json={"session_id": "s", "query": ""}).status_code == 422
    assert client.post("/chat", json={"session_id": "s", "query": "x" * 2001}).status_code == 422


def test_log_has_no_raw_text_by_default(client):
    post(client, "how do I improve my sleep")
    post(client, "I want to kill myself")
    text = (client.log_dir / "chat_log.jsonl").read_text()
    assert "sleep" not in text and "kill" not in text and "route" in text


def test_session_store_is_bounded():
    s = SessionStore(max_sessions=2, max_turns=2)
    for i in range(5):
        s.add(f"s{i}", "a")
    assert len(s) == 2
    for _ in range(5):
        s.add("s4", "b")
    assert len(s.recent("s4")) == 2
