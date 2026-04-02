"""Lightweight crisis keyword check — not a substitute for professional help."""

SAFETY_MESSAGE = (
    "If you are in immediate danger or thinking about hurting yourself, please contact "
    "local emergency services or a crisis line right away. In the U.S. you can call or text 988."
)

_CRISIS_KEYWORDS = (
    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "self-harm",
    "hurt myself",
)


def contains_crisis_keywords(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(k in lowered for k in _CRISIS_KEYWORDS)
