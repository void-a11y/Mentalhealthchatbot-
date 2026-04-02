from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

_DATA_DIR = Path(__file__).resolve().parent / "data"
_query_engine = None


def _ensure_data_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    placeholder = _DATA_DIR / "placeholder.txt"
    if not placeholder.exists():
        placeholder.write_text(
            "This is placeholder content for the document index. "
            "Replace or add files in the data folder for RAG.",
            encoding="utf-8",
        )


def _get_query_engine():
    global _query_engine
    if _query_engine is not None:
        return _query_engine

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai_like import OpenAILike

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to .env (https://console.groq.com/keys)")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_base = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

    _ensure_data_dir()
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = OpenAILike(
        model=model,
        api_key=api_key,
        api_base=api_base,
        is_chat_model=True,
        temperature=0.3,
    )
    documents = SimpleDirectoryReader(str(_DATA_DIR)).load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    _query_engine = index.as_query_engine(llm=llm)
    return _query_engine


def query_documents(user_query: str) -> str:
    engine = _get_query_engine()
    return str(engine.query(user_query))
