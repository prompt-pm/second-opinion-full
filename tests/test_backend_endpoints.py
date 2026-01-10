import contextlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import importlib

from models import (
    RouterResponse,
    UIResponse,
    Prompt,
    MessageResponse,
    ChoicesResponse,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def setup_client(monkeypatch, tmp_path):
    # Setup in-memory database
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    # Import backend lazily so we can patch modal before import
    import modal
    import sys
    import types

    if not hasattr(modal, "Mount"):

        class DummyMount:
            @classmethod
            def from_local_dir(cls, *a, **k):
                return None

        monkeypatch.setattr(modal, "Mount", DummyMount, raising=False)

    # Stub out ai_agents with minimal implementations to satisfy backend import
    ai_agents_stub = types.ModuleType("ai_agents")
    from models import ChoicesResponse, MessageResponse, Choice
    from pydantic import BaseModel

    class PrioritiesOutput(BaseModel):
        decision: str = ""
        objectives: list = []

    class ObjectionsOutput(BaseModel):
        objections: list = []
        suggested_messages: list = []

    ai_agents_stub.ChoicesResponse = ChoicesResponse
    ai_agents_stub.ObjectionsOutput = ObjectionsOutput
    ai_agents_stub.MessageResponse = MessageResponse
    ai_agents_stub.PrioritiesOutput = PrioritiesOutput
    ai_agents_stub.Choice = Choice

    async def dummy_ask(
        messages, advisor="thought_partner", custom_prefix=None, context=None
    ):
        return None

    ai_agents_stub.ask = dummy_ask
    sys.modules["ai_agents"] = ai_agents_stub

    import backend

    importlib.reload(backend)

    backend.SessionLocal = TestingSessionLocal
    backend.engine = engine
    backend.Base.metadata.create_all(bind=engine)

    # Disable tracing and external project context
    monkeypatch.setattr(backend, "initiate_tracing", lambda: None)
    monkeypatch.setattr(
        backend, "using_project", lambda *a, **k: contextlib.nullcontext()
    )

    # Patch ai functions used by endpoints
    monkeypatch.setattr(
        backend,
        "cerebras_router",
        lambda history: RouterResponse(
            prompt=Prompt.NORMAL, response=UIResponse(text="router")
        ),
    )

    async def fake_ask(
        messages, advisor="thought_partner", custom_prefix=None, context=None
    ):
        return SimpleNamespace(
            final_output=MessageResponse(text="hi", suggested_messages=[])
        )

    monkeypatch.setattr(backend, "ask", fake_ask)
    monkeypatch.setattr(
        backend,
        "cerebras_generate_outcomes",
        lambda history: ChoicesResponse(
            choices=[], title="title", uncertainties=[], next_steps=[]
        ),
    )

    client = TestClient(backend.web_app)
    yield client


def test_read_root(setup_client):
    client = setup_client
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"Hello": "World"}


def test_initial_query_endpoint(setup_client):
    client = setup_client
    res = client.post("/api/initial_query/", json={"message_history": []})
    assert res.status_code == 200
    assert res.json() == {
        "prompt": "normal",
        "response": {"text": "router", "citations": [], "suggested_messages": []},
    }


def test_query_endpoint_returns_message(setup_client):
    client = setup_client
    res = client.post("/api/query/", json={"messages": []})
    assert res.status_code == 200
    data = res.json()
    assert data["response_type"] == "message"
    assert data["agent"] == "Conversation Agent"
    assert data["response"]["text"] == "hi"


def test_choices_endpoint(setup_client):
    client = setup_client
    res = client.post("/api/choices", json={"message_history": []})
    assert res.status_code == 200
    assert res.json() == {
        "choices": [],
        "title": "title",
        "uncertainties": [],
        "next_steps": [],
    }


def test_save_and_get_decision(setup_client):
    client = setup_client
    history = [{"role": "user", "content": "hi"}]
    res = client.post("/api/save_decision/", json={"message_history": history})
    assert res.status_code == 200
    decision_id = res.json()["id"]

    res2 = client.get(f"/api/get_decision/{decision_id}")
    assert res2.status_code == 200
    assert res2.json() == {"message_history": history}
