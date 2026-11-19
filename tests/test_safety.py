import pytest

import safety
from safety import SAFETY_MESSAGE, check


@pytest.mark.parametrize(
    "text",
    [
        "I want to kill myself",
        "i've been thinking about suicide",
        "I DON'T WANT TO LIVE anymore",
        "dont want to be here any more",
        "I want to die",
        "everyone would be better off without me",
        "I keep wanting to hurt myself",
        "thoughts of self-harm",
        "I can't go on like this",
        "there is no point in living",
        "I want to end my life",
        "I’m thinking about taking my own life",
    ],
)
def test_crisis_detected(text):
    v = check(text)
    assert v.kind == "crisis" and v.message == SAFETY_MESSAGE


@pytest.mark.parametrize(
    "text",
    [
        "how do I improve my sleep",
        "I'm stressed about exams",
        "what are the symptoms of anxiety",
        "I loved the movie Suicide Squad",
        "my phone is dying",
        "I want to learn how to meditate",
    ],
)
def test_benign_not_flagged(text):
    assert check(text).kind == "ok"


def test_crisis_in_recent_history_keeps_safety_mode():
    history = ["I want to kill myself", "ok"]
    assert check("what is a good bedtime routine", history).kind == "crisis"


def test_old_history_expires():
    history = ["I want to kill myself", "a", "b", "c"]
    assert check("what is a good bedtime routine", history).kind == "ok"


def test_message_split_across_turns_is_caught():
    assert check("go on", ["I just can't"]).kind == "crisis"


@pytest.mark.parametrize(
    "text",
    [
        "Do I have depression?",
        "can you diagnose me",
        "how many mg of sertraline should I take",
        "should I stop taking my medication",
        "what's wrong with me",
    ],
)
def test_out_of_scope_requests_redirected(text):
    assert check(text).kind == "out_of_scope"


def test_crisis_takes_priority_over_scope():
    assert check("I want to kill myself, how many pills would it take").kind == "crisis"


def test_message_has_india_and_us_resources():
    assert "14416" in SAFETY_MESSAGE and "988" in SAFETY_MESSAGE


def test_normalize_handles_unicode_and_punctuation():
    assert safety.normalize("  I’M   fine!!! ") == "i'm fine"
