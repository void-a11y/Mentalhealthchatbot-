import httpx
import pytest
from openai import RateLimitError

import rag.answer as answer_mod
from rag.answer import GroqLLM


def _rate_limit() -> RateLimitError:
    req = httpx.Request("POST", "http://x")
    return RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)


class _Completions:
    def __init__(self, failures: int):
        self.failures, self.calls = failures, 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise _rate_limit()
        msg = type("M", (), {"content": "ok [1]"})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


def _llm(failures: int) -> tuple[GroqLLM, _Completions]:
    llm = GroqLLM("key", "model", "http://x")
    comp = _Completions(failures)
    llm._client = type("Client", (), {"chat": type("Chat", (), {"completions": comp})()})()
    return llm, comp


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr(answer_mod.time, "sleep", lambda s: None)
    llm, comp = _llm(failures=2)
    assert llm.complete("sys", "user") == "ok [1]"
    assert comp.calls == 3


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(answer_mod.time, "sleep", lambda s: None)
    llm, comp = _llm(failures=99)
    with pytest.raises(RateLimitError):
        llm.complete("sys", "user")
    assert comp.calls == GroqLLM.MAX_ATTEMPTS


def test_missing_key_raises_clear_error():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqLLM(None, "m", "http://x").complete("s", "u")
