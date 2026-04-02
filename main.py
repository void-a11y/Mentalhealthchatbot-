from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from chatengine import get_response
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from models import ChatRequest
from openai import APIError, AuthenticationError, RateLimitError
from crisis import contains_crisis_keywords, SAFETY_MESSAGE
from logger import log_chat
from doc_engine import query_documents
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_UI_DIR = Path(__file__).resolve().parent / "chatbot-ui"
if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")


@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "ui": "/ui/",
        "llm": "groq",
    }


def _http_from_llm_api(exc: Exception) -> HTTPException:
    if isinstance(exc, AuthenticationError):
        return HTTPException(
            status_code=401,
            detail="API rejected the key. Check GROQ_API_KEY in .env.",
        )
    if isinstance(exc, RateLimitError):
        return HTTPException(
            status_code=429,
            detail="LLM rate limit or quota exceeded. See https://console.groq.com/",
        )
    if isinstance(exc, APIError):
        return HTTPException(status_code=502, detail=str(exc) or "LLM API error.")
    return HTTPException(status_code=500, detail="Unexpected error while calling the model.")


@app.post("/chat")
def chat_with_memory(request: ChatRequest):
    session_id = request.session_id
    user_query = request.query

    if contains_crisis_keywords(user_query):
        log_chat(session_id, user_query, SAFETY_MESSAGE, is_crisis=True)
        return {"response": SAFETY_MESSAGE}

    try:
        response = get_response(session_id, user_query)
    except (AuthenticationError, RateLimitError, APIError) as e:
        raise _http_from_llm_api(e) from e

    log_chat(session_id, user_query, response, is_crisis=False)
    return {"response": response}


@app.post("/doc-chat")
def chat_with_documents(request: ChatRequest):
    try:
        response = query_documents(request.query)
    except (AuthenticationError, RateLimitError, APIError) as e:
        raise _http_from_llm_api(e) from e
    return {"response": response}
