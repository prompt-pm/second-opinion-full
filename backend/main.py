"""
Second Opinion - A decision-making assistant
Simplified starter for teaching purposes
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()


import instructor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI

from backend.models import (
    ChatRequest,
    ChoicesRequest,
    PrioritiesRequest,
    PrioritiesResponse,
)
from backend.prompts import CHOICES_PROMPT, PRIORITIES_PROMPT, SYSTEM_PROMPT

app = FastAPI()

# CORS configuration - use environment variable for production
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# OpenRouter client - set OPENROUTER_API_KEY in your environment
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
MODEL = "google/gemini-2.0-flash-001"


@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")


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
    # Include priorities in the prompt if provided
    prompt = CHOICES_PROMPT
    if request.priorities:
        prompt += "\n\nUser's ranked priorities (most important first):\n"
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

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from LLM: {e}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
