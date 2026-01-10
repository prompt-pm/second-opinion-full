import pytest

from ai_functions import clean_up_messages


@pytest.fixture
def system_message():
    return {"role": "system", "content": "System prompt"}


@pytest.fixture
def long_user_message():
    return {"role": "user", "content": "x" * 31000}


@pytest.fixture
def paired_messages(system_message):
    return [
        system_message,
        {"role": "user", "content": "u1" * 7500},
        {"role": "assistant", "content": "a1" * 6000},
        {"role": "user", "content": "u2" * 5500},
        {"role": "assistant", "content": "a2" * 50},
    ]


def test_truncates_long_messages(system_message, long_user_message):
    messages = [system_message, long_user_message]
    cleaned = clean_up_messages(messages.copy())
    assert len(cleaned) == 2
    assert len(cleaned[1]["content"]) == 2003
    assert cleaned[1]["content"].startswith("x" * 1000)
    assert cleaned[1]["content"].endswith("x" * 1000)


def test_pairing_removes_oldest_messages(paired_messages):
    cleaned = clean_up_messages(paired_messages.copy())
    assert cleaned[0]["role"] == "system"
    assert len(cleaned) == 3
    assert cleaned[1]["content"] == "u2" * 5500
    assert cleaned[2]["content"] == "a2" * 50


def test_preserves_structure(system_message):
    messages = [
        system_message,
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    cleaned = clean_up_messages(messages.copy())
    assert messages == cleaned
    for msg in cleaned:
        assert set(msg.keys()) == {"role", "content"}
        assert isinstance(msg["content"], str)
