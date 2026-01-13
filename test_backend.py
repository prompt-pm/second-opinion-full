"""Tests for the Second Opinion backend API."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend import (
    Choice,
    ChoicesResponse,
    Message,
    PrioritiesResponse,
    app,
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a test response from the assistant."
    return mock_response


class TestServeIndex:
    """Tests for the index route."""

    def test_serve_index_returns_html(self, client):
        """GET / should return the index.html file."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestChatEndpoint:
    """Tests for the /api/chat endpoint."""

    def test_chat_with_valid_message(self, client, mock_openai_response):
        """POST /api/chat should return an assistant response."""
        with patch("backend.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_openai_response

            response = client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": "Should I take the job?"}]},
            )

            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert data["response"] == "This is a test response from the assistant."

    def test_chat_with_multiple_messages(self, client, mock_openai_response):
        """POST /api/chat should handle multiple messages in conversation."""
        with patch("backend.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_openai_response

            messages = [
                {"role": "user", "content": "Should I move to a new city?"},
                {"role": "assistant", "content": "What factors are most important to you?"},
                {"role": "user", "content": "Career opportunities and cost of living."},
            ]

            response = client.post("/api/chat", json={"messages": messages})

            assert response.status_code == 200
            # Verify the messages were passed to the OpenAI client
            call_args = mock_client.chat.completions.create.call_args
            assert len(call_args.kwargs["messages"]) == 4  # system + 3 user messages

    def test_chat_with_empty_messages(self, client, mock_openai_response):
        """POST /api/chat should handle empty messages list."""
        with patch("backend.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_openai_response

            response = client.post("/api/chat", json={"messages": []})

            assert response.status_code == 200

    def test_chat_missing_messages_field(self, client):
        """POST /api/chat should return 422 when messages field is missing."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_chat_invalid_message_format(self, client):
        """POST /api/chat should return 422 for invalid message format."""
        response = client.post(
            "/api/chat",
            json={"messages": [{"invalid": "format"}]},
        )
        assert response.status_code == 422


class TestPrioritiesEndpoint:
    """Tests for the /api/priorities endpoint."""

    def test_extract_priorities_returns_list(self, client):
        """POST /api/priorities should return a list of priorities."""
        mock_priorities = PrioritiesResponse(
            priorities=["Career growth", "Work-life balance", "Salary"]
        )

        with patch("backend.instructor") as mock_instructor:
            mock_structured_client = MagicMock()
            mock_instructor.from_openai.return_value = mock_structured_client
            mock_structured_client.chat.completions.create.return_value = mock_priorities

            response = client.post(
                "/api/priorities",
                json={
                    "messages": [
                        {"role": "user", "content": "Should I take this job offer?"},
                        {"role": "assistant", "content": "What matters most to you?"},
                        {"role": "user", "content": "I care about salary and growth."},
                    ]
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "priorities" in data
            assert len(data["priorities"]) == 3
            assert "Career growth" in data["priorities"]

    def test_extract_priorities_empty_conversation(self, client):
        """POST /api/priorities should handle empty conversation."""
        mock_priorities = PrioritiesResponse(priorities=[])

        with patch("backend.instructor") as mock_instructor:
            mock_structured_client = MagicMock()
            mock_instructor.from_openai.return_value = mock_structured_client
            mock_structured_client.chat.completions.create.return_value = mock_priorities

            response = client.post("/api/priorities", json={"messages": []})

            assert response.status_code == 200


class TestChoicesEndpoint:
    """Tests for the /api/choices endpoint."""

    def test_generate_choices_returns_structured_response(self, client):
        """POST /api/choices should return structured decision options."""
        mock_choices_json = {
            "title": "Should you take the new job?",
            "choices": [
                {
                    "name": "Take the job",
                    "best_case": "Career advancement and higher salary",
                    "worst_case": "Culture mismatch and burnout",
                },
                {
                    "name": "Stay at current job",
                    "best_case": "Stability and familiar environment",
                    "worst_case": "Miss growth opportunity",
                },
                {
                    "name": "Negotiate better terms",
                    "best_case": "Get the best of both worlds",
                    "worst_case": "Offer gets rescinded",
                },
            ],
            "uncertainties": [
                "How stable is the new company?",
                "Will the new role match expectations?",
            ],
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_choices_json)

        with patch("backend.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response

            response = client.post(
                "/api/choices",
                json={
                    "messages": [
                        {"role": "user", "content": "Should I take this job offer?"}
                    ],
                    "priorities": ["Career growth", "Salary"],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "title" in data
            assert "choices" in data
            assert "uncertainties" in data
            assert len(data["choices"]) == 3

    def test_generate_choices_without_priorities(self, client):
        """POST /api/choices should work without priorities."""
        mock_choices_json = {
            "title": "What should you do?",
            "choices": [
                {"name": "Option A", "best_case": "Good outcome", "worst_case": "Bad outcome"}
            ],
            "uncertainties": ["Key uncertainty?"],
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_choices_json)

        with patch("backend.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response

            response = client.post(
                "/api/choices",
                json={"messages": [{"role": "user", "content": "Help me decide"}]},
            )

            assert response.status_code == 200

    def test_generate_choices_includes_priorities_in_prompt(self, client):
        """POST /api/choices should include priorities in the prompt to OpenAI."""
        mock_choices_json = {
            "title": "Decision",
            "choices": [{"name": "Option", "best_case": "Good", "worst_case": "Bad"}],
            "uncertainties": [],
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_choices_json)

        with patch("backend.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response

            priorities = ["Stability", "Growth", "Compensation"]
            response = client.post(
                "/api/choices",
                json={
                    "messages": [{"role": "user", "content": "Should I switch jobs?"}],
                    "priorities": priorities,
                },
            )

            assert response.status_code == 200
            # Verify priorities were included in the system message
            call_args = mock_client.chat.completions.create.call_args
            system_message = call_args.kwargs["messages"][0]["content"]
            for priority in priorities:
                assert priority in system_message


class TestModels:
    """Tests for Pydantic models."""

    def test_message_model(self):
        """Message model should validate correctly."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_choice_model(self):
        """Choice model should validate correctly."""
        choice = Choice(
            name="Option A",
            best_case="Everything goes well",
            worst_case="Nothing works out",
        )
        assert choice.name == "Option A"
        assert choice.best_case == "Everything goes well"
        assert choice.worst_case == "Nothing works out"

    def test_priorities_response_model(self):
        """PrioritiesResponse model should validate correctly."""
        response = PrioritiesResponse(priorities=["A", "B", "C"])
        assert len(response.priorities) == 3

    def test_choices_response_model(self):
        """ChoicesResponse model should validate correctly."""
        response = ChoicesResponse(
            title="What to do?",
            choices=[
                Choice(name="Option 1", best_case="Good", worst_case="Bad"),
            ],
            uncertainties=["Will it work?"],
        )
        assert response.title == "What to do?"
        assert len(response.choices) == 1
        assert len(response.uncertainties) == 1
