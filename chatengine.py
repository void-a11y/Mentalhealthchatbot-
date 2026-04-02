import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to .env (https://console.groq.com/keys)")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

_llm = ChatOpenAI(
    model=GROQ_MODEL,
    openai_api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    temperature=0.7,
)

_SYSTEM = SystemMessage(
    content=(
        "You are a caring, supportive mental wellness chat assistant. "
        "You are not a doctor or therapist; encourage professional help when appropriate. "
        "Be concise, warm, and practical."
    )
)

_session_messages: dict[str, list] = {}


def get_response(session_id: str, user_query: str) -> str:
    if session_id not in _session_messages:
        _session_messages[session_id] = [_SYSTEM]

    messages = _session_messages[session_id]
    messages.append(HumanMessage(content=user_query))
    result = _llm.invoke(messages)
    text = result.content if isinstance(result.content, str) else str(result.content)
    messages.append(AIMessage(content=text))
    return text
