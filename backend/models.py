"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class PrioritiesRequest(BaseModel):
    messages: list[Message]


class ChoicesRequest(BaseModel):
    messages: list[Message]
    priorities: list[str] = []


class PrioritiesResponse(BaseModel):
    priorities: list[str] = Field(description="3-5 key priorities for this decision")


class Choice(BaseModel):
    name: str = Field(description="Short name for this option, e.g. 'Take the job' or 'Stay put'")
    best_case: str = Field(
        description="Best case scenario, e.g. 'Career takes off, double salary'"
    )
    worst_case: str = Field(
        description="Worst case scenario, e.g. 'Hate the new role, regret leaving'"
    )


class ChoicesResponse(BaseModel):
    title: str = Field(description="A question summarizing the decision")
    choices: list[Choice]
    uncertainties: list[str] = Field(description="1-2 key uncertainties as questions")
