"""Bounded in-memory session store. Keeps only recent user turns (used by the
crisis check) and evicts the least recently used sessions."""
from __future__ import annotations

import threading
from collections import OrderedDict, deque


class SessionStore:
    def __init__(self, max_sessions: int = 1000, max_turns: int = 6):
        self.max_sessions, self.max_turns = max_sessions, max_turns
        self._data: OrderedDict[str, deque[str]] = OrderedDict()
        self._lock = threading.Lock()

    def recent(self, session_id: str) -> list[str]:
        with self._lock:
            return list(self._data.get(session_id, ()))

    def add(self, session_id: str, text: str) -> None:
        with self._lock:
            turns = self._data.setdefault(session_id, deque(maxlen=self.max_turns))
            turns.append(text)
            self._data.move_to_end(session_id)
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
