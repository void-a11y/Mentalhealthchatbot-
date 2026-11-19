"""Privacy-conscious event log (JSON lines).

By default only metadata is stored: hashed session id, route, relevance score and
query length. Raw text is logged only when LOG_TEXT=true, and never for crisis turns.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from config import load_settings


def log_event(
    session_id: str,
    route: str,
    abstained: bool,
    top_score: float | None,
    query: str = "",
    response: str = "",
) -> None:
    s = load_settings()
    s.log_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": hashlib.sha256(session_id.encode()).hexdigest()[:12],
        "route": route,
        "abstained": abstained,
        "top_score": None if top_score is None else round(top_score, 4),
        "query_chars": len(query),
    }
    if s.log_text and route != "crisis":
        event["query"], event["response"] = query, response
    with (s.log_dir / "chat_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
