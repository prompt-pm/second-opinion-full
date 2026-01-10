import pytest

from models import Message, MessageList


@pytest.fixture
def sample_messages():
    return [
        Message(role="system", content="System"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]


@pytest.fixture
def message_list(sample_messages):
    return MessageList(messages=list(sample_messages))


def test_add_and_prepend_message(message_list):
    new_msg = Message(role="user", content="New")
    message_list.add_message(new_msg)
    assert message_list.messages[-1] == new_msg

    first_msg = Message(role="assistant", content="First")
    message_list.prepend_message(first_msg)
    assert message_list.messages[0] == first_msg
    # Ensure total count updated correctly
    assert len(message_list.messages) == 5


def test_return_formats(message_list):
    openai_format = message_list.return_openai_format()
    assert openai_format == [
        {"role": msg.role, "content": msg.content} for msg in message_list.messages
    ]

    persona_format = message_list.return_persona_format()
    expected_lines = [
        "persona: System",
        "user: Hello",
        "persona: Hi",
    ]
    assert persona_format.split("\n") == expected_lines

    json_format = message_list.return_json()
    assert json_format == [msg.model_dump() for msg in message_list.messages]


def test_clear_and_str(message_list):
    # Check __str__ output before clearing
    s = str(message_list)
    assert s == "system: System\nuser: Hello\nassistant: Hi"

    message_list.clear()
    assert message_list.messages == []
