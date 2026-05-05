"""
Second Opinion - A decision-making assistant
Uses DeepAgents with Gemini for intelligent widget orchestration.
"""

import json
import os
from typing import Literal

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
    StoryExtractionRequest,
)
from backend.prompts import CHOICES_PROMPT, PRIORITIES_PROMPT, SYSTEM_PROMPT
from backend.tools import CARD_SETS

app = FastAPI()

# CORS configuration
ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:7000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Gemini client via OpenAI-compatible API (for structured output endpoints)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"

_gemini_client = None


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)
    return _gemini_client


# ---------------------------------------------------------------------------
# DeepAgents widget tool functions
# ---------------------------------------------------------------------------


def show_story_prompts() -> dict:
    """Show best-case/worst-case story prompts to help user discover what matters.
    Use when the user can't articulate their priorities, says things like
    'I don't know what I want', or seems confused about what matters."""
    return {"type": "show_story_prompts", "params": {}}


def show_card_sort(
    decision_type: Literal["job", "housing", "relationship", "purchase", "other"],
) -> dict:
    """Show common priorities for this decision type as selectable cards.
    Use when the user is slow to identify priorities, lists too many vague
    factors, or would benefit from seeing common options.

    Args:
        decision_type: The type of decision to show relevant priority cards for.
    """
    cards = CARD_SETS.get(decision_type, CARD_SETS["other"])
    return {
        "type": "show_card_sort",
        "params": {"decision_type": decision_type, "cards": cards},
    }


def show_priority_tournament(priorities: list[str]) -> dict:
    """Show pairwise comparisons to help rank priorities.
    Use when the user says 'everything is equally important', can't decide
    on ranking, or is paralyzed trying to order their priorities.

    Args:
        priorities: The list of priorities to rank through pairwise comparison.
    """
    return {"type": "show_priority_tournament", "params": {"priorities": priorities}}


def show_budget_allocation(priorities: list[str]) -> dict:
    """Show sliders to allocate 100 points across priorities.
    Use to confirm a ranking, when the user wants to express precise weights,
    or after a tournament to fine-tune.

    Args:
        priorities: The list of priorities to allocate points to.
    """
    return {"type": "show_budget_allocation", "params": {"priorities": priorities}}


def show_elimination_game(priorities: list[str]) -> dict:
    """Show priorities and ask user to eliminate the least important one at a time.
    Use when the user has too many priorities (5+) and is overwhelmed,
    or needs help narrowing down.

    Args:
        priorities: The list of priorities to eliminate from.
    """
    return {"type": "show_elimination_game", "params": {"priorities": priorities}}


def show_recommendation(options: list[dict], priorities: list[dict]) -> dict:
    """Show a scored recommendation comparing options against the user's priorities.
    Use when you have enough information about both the user's priorities and
    their options to make a recommendation.

    Args:
        options: The options to compare. Each dict has 'name' (str), 'score' (number), and 'scores' (dict of per-priority scores).
        priorities: The priorities with weights. Each dict has 'name' (str) and 'weight' (number, should sum to 1.0).
    """
    return {
        "type": "show_recommendation",
        "params": {"options": options, "priorities": priorities},
    }


def show_tradeoff_acknowledgment(
    choice: str, gaining: list[str], sacrificing: list[str]
) -> dict:
    """Show what the user gains vs sacrifices with their choice.
    Use before finalizing a decision to ensure the user explicitly accepts
    the tradeoffs.

    Args:
        choice: The option the user is choosing.
        gaining: What the user gains by making this choice.
        sacrificing: What the user gives up by making this choice.
    """
    return {
        "type": "show_tradeoff_acknowledgment",
        "params": {
            "choice": choice,
            "gaining": gaining,
            "sacrificing": sacrificing,
        },
    }


def show_premortem(choice: str, worst_case: str) -> dict:
    """Show the worst realistic outcome and ask if the user can live with it.
    Use when the user is anxious about committing, keeps second-guessing,
    or needs to confront the downside.

    Args:
        choice: The option the user is considering.
        worst_case: A realistic worst-case scenario if they make this choice.
    """
    return {
        "type": "show_premortem",
        "params": {"choice": choice, "worst_case": worst_case},
    }


WIDGET_TOOLS = [
    show_story_prompts,
    show_card_sort,
    show_priority_tournament,
    show_budget_allocation,
    show_elimination_game,
    show_recommendation,
    show_tradeoff_acknowledgment,
    show_premortem,
]

# ---------------------------------------------------------------------------
# Agent creation (lazy, cached per process)
# ---------------------------------------------------------------------------

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from deepagents import create_deep_agent

        _agent = create_deep_agent(
            model="google_genai:gemini-2.5-flash",
            tools=WIDGET_TOOLS,
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent


def _extract_response(result: dict) -> tuple[str, dict | None]:
    """Extract text response and widget from DeepAgent result."""
    from langchain_core.messages import AIMessage, ToolMessage

    text = ""
    widget = None

    # Get text response: last AIMessage that has content and no tool_calls
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            text = msg.content
            break

    # Get widget: look for ToolMessage with widget data
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                content = (
                    json.loads(msg.content)
                    if isinstance(msg.content, str)
                    else msg.content
                )
                if isinstance(content, dict) and content.get("type", "").startswith(
                    "show_"
                ):
                    widget = content
            except (json.JSONDecodeError, TypeError):
                pass

    return text, widget


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")


@app.post("/api/chat")
def chat(request: ChatRequest):
    """Chat endpoint with DeepAgents widget orchestration."""
    input_messages = [m.model_dump() for m in request.messages]
    agent = get_agent()
    result = agent.invoke({"messages": input_messages})
    text, widget = _extract_response(result)

    response = {"response": text}
    if widget:
        response["widget"] = widget
    return response


@app.post("/api/extract-priorities")
def extract_priorities_from_stories(request: StoryExtractionRequest):
    """Extract priorities from best/worst case narratives."""
    prompt = f"""From these scenarios, extract 3-6 priorities that matter to this person.
Focus on what they value, not specific details.

BEST CASE (what they want): {request.best_case}
WORST CASE (what they fear): {request.worst_case}

Return as JSON: {{"priorities": ["priority 1", "priority 2", ...]}}
Keep each priority to 2-4 words."""

    response = get_gemini_client().chat.completions.create(
        model=GEMINI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON from LLM: {e}"
        ) from e


@app.post("/api/priorities")
def extract_priorities(request: PrioritiesRequest):
    """Extract priorities from the conversation."""
    structured_client = instructor.from_openai(get_gemini_client())
    messages = [{"role": "system", "content": PRIORITIES_PROMPT}]
    messages += [m.model_dump() for m in request.messages]
    response = structured_client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=messages,
        response_model=PrioritiesResponse,
    )
    return response.model_dump()


@app.post("/api/choices")
def generate_choices(request: ChoicesRequest):
    """Generate structured decision options from conversation and priorities."""
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

    response = get_gemini_client().chat.completions.create(
        model=GEMINI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON from LLM: {e}"
        ) from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
