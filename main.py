from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import APIError, AuthenticationError, RateLimitError

import safety
from config import load_settings
from logger import log_event
from models import ChatRequest, ChatResponse, SourceOut
from pipeline import get_answerer
from sessions import SessionStore

settings = load_settings()
app = FastAPI(title="Mental health RAG chatbot")
store = SessionStore(max_sessions=settings.max_sessions)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_UI_DIR = Path(__file__).resolve().parent / "chatbot-ui"
if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")


def answerer_provider():
    """Returns a callable that builds the Answerer. Resolved lazily so safety
    responses still work if the index or LLM key is missing. Overridden in tests."""
    return get_answerer


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "ui": "/ui/"}


def _llm_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AuthenticationError):
        return HTTPException(status_code=401, detail="LLM API rejected the key. Check GROQ_API_KEY in .env.")
    if isinstance(exc, RateLimitError):
        return HTTPException(status_code=429, detail="LLM rate limit or quota exceeded.")
    return HTTPException(status_code=502, detail="LLM API error.")


@app.post("/chat", response_model=ChatResponse)
@app.post("/doc-chat", response_model=ChatResponse, include_in_schema=False)  # kept for the old UI
def chat(request: ChatRequest, get_ans=Depends(answerer_provider)) -> ChatResponse:
    history = store.recent(request.session_id)
    store.add(request.session_id, request.query)

    verdict = safety.check(request.query, history)
    if verdict.kind != "ok":
        log_event(request.session_id, verdict.kind, False, None, request.query, verdict.message)
        return ChatResponse(response=verdict.message, route=verdict.kind)

    try:
        result = get_ans().answer(request.query)
    except RuntimeError as e:  # missing index or API key
        raise HTTPException(status_code=503, detail=str(e)) from e
    except (AuthenticationError, RateLimitError, APIError) as e:
        raise _llm_http_error(e) from e

    route = "not_found" if result.abstained else "answered"
    log_event(request.session_id, route, result.abstained, result.top_score, request.query, result.text)
    return ChatResponse(
        response=result.text,
        route=route,
        abstained=result.abstained,
        sources=[SourceOut(**{k: getattr(c, k) for k in SourceOut.model_fields}) for c in result.citations],
    )
