"""
Second Opinion - A decision-making assistant
Simplified starter for teaching purposes
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
import instructor

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter client - set OPENROUTER_API_KEY in your environment
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
MODEL = "google/gemini-2.0-flash-001"

# --- Prompts ---
SYSTEM_PROMPT = """You are a decision-making assistant that helps people think through choices.
Analyze trade-offs, consider uncertainties, and help users see different perspectives.
Be warm but concise. Ask clarifying questions when needed. Use 1-2 sentences max."""

PRIORITIES_PROMPT = """Based on this conversation, identify 3-5 priorities or objectives
that matter most to this person's decision. Return them as a simple list."""

CHOICES_PROMPT = """Based on this conversation and the user's priorities, generate 3 possible
choices for the user's decision.

For EACH choice, you must provide:
- name: A short name for the option (2-5 words)
- best_case: The best case scenario if they choose this (under 10 words)
- worst_case: The worst case scenario if they choose this (under 10 words)

Also provide 1-2 key uncertainties as questions."""


# --- Models ---
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class PrioritiesRequest(BaseModel):
    messages: List[Message]


class ChoicesRequest(BaseModel):
    messages: List[Message]
    priorities: List[str] = []


class PrioritiesResponse(BaseModel):
    priorities: List[str] = Field(description="3-5 key priorities for this decision")


class Choice(BaseModel):
    name: str = Field(description="Short name for this option, e.g. 'Take the job' or 'Stay put'")
    best_case: str = Field(description="Best case scenario, e.g. 'Career takes off, double salary'")
    worst_case: str = Field(description="Worst case scenario, e.g. 'Hate the new role, regret leaving'")


class ChoicesResponse(BaseModel):
    title: str = Field(description="A question summarizing the decision")
    choices: List[Choice]
    uncertainties: List[str] = Field(description="1-2 key uncertainties as questions")


# --- Endpoints ---
@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.post("/api/chat")
def chat(request: ChatRequest):
    """Simple chat endpoint for conversation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [m.model_dump() for m in request.messages]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return {"response": response.choices[0].message.content}


@app.post("/api/priorities")
def extract_priorities(request: PrioritiesRequest):
    """Extract priorities from the conversation."""
    structured_client = instructor.from_openai(client)
    messages = [{"role": "system", "content": PRIORITIES_PROMPT}]
    messages += [m.model_dump() for m in request.messages]
    response = structured_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=PrioritiesResponse,
    )
    return response.model_dump()


@app.post("/api/choices")
def generate_choices(request: ChoicesRequest):
    """Generate structured decision options from conversation and priorities."""
    import json

    # Include priorities in the prompt if provided
    prompt = CHOICES_PROMPT
    if request.priorities:
        prompt += f"\n\nUser's ranked priorities (most important first):\n"
        for i, p in enumerate(request.priorities, 1):
            prompt += f"{i}. {p}\n"

    prompt += """

Respond with valid JSON in exactly this format:
{
  "title": "Question summarizing the decision",
  "choices": [
    {"name": "Option 1 name", "best_case": "Best outcome", "worst_case": "Worst outcome"},
    {"name": "Option 2 name", "best_case": "Best outcome", "worst_case": "Worst outcome"},
    {"name": "Option 3 name", "best_case": "Best outcome", "worst_case": "Worst outcome"}
  ],
  "uncertainties": ["Key uncertainty 1?", "Key uncertainty 2?"]
}"""

    messages = [{"role": "system", "content": prompt}]
    messages += [m.model_dump() for m in request.messages]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
