"""Safety layer. Runs BEFORE retrieval or any LLM call, and works without either.

Crisis detection is deliberately recall-first: normalised regex patterns over the
current message plus the last few user turns. After someone discloses distress we
stay in safety mode for a few turns rather than switching to a generic answer.
This is a keyword/pattern guardrail, not a clinical tool; the README states this.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

SAFETY_MESSAGE = (
    "I'm really sorry you're going through this, and I'm glad you said something. "
    "You deserve support from a real person right now.\n\n"
    "If you are in immediate danger, call your local emergency number (112 in India).\n"
    "- Tele-MANAS (India, 24/7, free): 14416 or 1800-891-4416\n"
    "- KIRAN mental health helpline (India): 1800-599-0019\n"
    "- United States: call or text 988\n\n"
    "If you can, reach out to someone you trust and tell them how you are feeling. "
    "I'm an AI assistant and can't provide crisis support."
)

SCOPE_MESSAGE = (
    "I can't diagnose conditions or give advice about medication or dosage. "
    "A doctor, psychiatrist or counsellor is the right person for that. "
    "I'm happy to share general information from my documents, for example about symptoms, "
    "coping strategies or how therapy works."
)

HISTORY_WINDOW = 3  # previous user turns considered alongside the current one

_BENIGN = re.compile(r"\bsuicide squad\b|\bsuicide mission\b")

_CRISIS_PATTERNS = [
    r"\bsuicid(e|al)\b",
    r"\bkill(ing)? my ?self\b",
    r"\b(end|ending|take|taking) (my|my own) (own )?life\b",
    r"\bend it all\b",
    r"\b(want|wanna|wish|wished|wanted)( to| i could)? (die|be dead|disappear|not wake up)\b",
    r"\b(don't|do not|dont|no longer) (want|wanna) to (live|be here|be alive|exist|go on|wake up)\b",
    r"\b(can't|cannot|cant) go on\b",
    r"\bbetter off (dead|without me)\b",
    r"\bno (reason|point) (to|in) (live|living|going on|being alive)\b",
    r"\bself ?harm(ing)?\b",
    r"\b(hurt|hurting|harm|harming|cut|cutting|burn|burning) my ?self\b",
]
_CRISIS = [re.compile(p) for p in _CRISIS_PATTERNS]

_SCOPE_PATTERNS = [
    r"\bdo i have (depression|anxiety|adhd|bipolar|ptsd|ocd|schizophrenia|a mental|an anxiety)\b",
    r"\bam i (depressed|bipolar|mentally ill|schizophrenic)\b",
    r"\b(diagnose|can you diagnose) me\b",
    r"\bwhat('s| is) wrong with me\b",
    r"\b(dosage|how much|how many)\b.*\b(mg|pills?|tablets?|medication|antidepressants?|sertraline|fluoxetine|alprazolam|xanax|prozac|lexapro|escitalopram|diazepam|valium)\b",
    r"\b(stop|quit|skip|reduce|change) (taking )?(my )?(medication|meds|antidepressants?|pills)\b",
]
_SCOPE = [re.compile(p) for p in _SCOPE_PATTERNS]


@dataclass
class Verdict:
    kind: str  # "crisis" | "out_of_scope" | "ok"
    message: str = ""


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_crisis(text: str) -> bool:
    t = _BENIGN.sub(" ", normalize(text))
    return any(p.search(t) for p in _CRISIS)


def check(query: str, recent_turns: list[str] | None = None) -> Verdict:
    window = (recent_turns or [])[-HISTORY_WINDOW:]
    # Joined, so a thought split across messages ("I just can't" / "go on") is still caught.
    if _is_crisis(" ".join(window + [query])):
        return Verdict("crisis", SAFETY_MESSAGE)
    q = normalize(query)
    if any(p.search(q) for p in _SCOPE):
        return Verdict("out_of_scope", SCOPE_MESSAGE)
    return Verdict("ok")
