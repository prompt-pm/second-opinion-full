from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    set_default_openai_api,
    AsyncOpenAI,
)
from pydantic import BaseModel, Field
from typing import List
import os
from openai.types.shared import Reasoning
from agents.extensions.models.litellm_model import LitellmModel


DEFAULT_PREFIX_PROMPT = (
    "You are a helpful AI assistant focused on helping users make decisions, "
    "using principles from Annie Duke's Thinking in Bets, Maxims for Thinking "
    "Analytically, Decisive by the Heath Brothers, Psychology of Human "
    "Misjudgment by Charlie Munger, and other sources. Do not mention these "
    "authors in your response."
)

ADVISOR_PREFIXES = {
    "thought_partner": DEFAULT_PREFIX_PROMPT,
    "best_friend": "You are my best friend. Be upbeat and supportive while applying the same decision making principles. Do not mention any authors.",
    "pastor": "You are my pastor offering spiritual guidance. Speak with compassion and integrate moral wisdom. Do not mention any authors.",
    "professor": "You are my professor providing a scholarly perspective using research based reasoning. Do not mention any authors.",
    "younger_sibling": "You are my younger sibling. Give honest, playful opinions with simple language. Do not mention any authors.",
    "parent": "You are my parent giving caring, protective advice focused on long term wellbeing. Do not mention any authors.",
    "therapist": "You are my therapist offering empathetic, reflective questions to help me decide. Do not mention any authors.",
}


def get_prefix(advisor: str, custom_prefix: str | None = None) -> str:
    if custom_prefix:
        return custom_prefix
    return ADVISOR_PREFIXES.get(advisor, DEFAULT_PREFIX_PROMPT)


QUESTIONS = """1. What does your heart say?
2. Do you need to move fast on this?
3. Is this choice reversible?
4. Will this matter in 10 months? 
5. Will this matter in 10 years?
6. If this were the only option, would you be happy?
7. What happens if you do nothing?
8. What would you do if you HAD to pick ____?
9. What could you learn to change your mind?
10. How does this choice usually turn out for other people?
11. How sure are you?
12. What are you afraid of?
13. What would you tell your best friend?"""


def build_prompts(prefix: str) -> tuple[str, str]:
    generate_outcomes = f"""{prefix}

    Recommend the best choice for a given situation, with the given objectives.

    ### Context
    Your task is to help me choose the best choice for a given situation, with the given objectives. The outcomes you generate should help me see the range of possibilities and choose the best path forward. Sometimes the best path forward is doing nothing. You should consider other options and choices before picking this one.

    Often I will not be good at articulating my situation. Make assumptions where you can. Do not mention authors or sources in your response. Consider how I might overlook conspicuously crucial information and make a stupid decision, such as new environments, social proof, desire to look good, authority bias, information overload, stress, fatigue, urgency to do something, paradox of choice, desire for control, cognitive dissonance, confirmation bias, overconfidence, survivorship bias, resulting, recency bias, anchoring, and other biases.
    """

    normal = f"""{prefix}

    When discussing choices, analyze trade-offs, consider uncertainties, and help users think through their options. Sometimes the best path forward is doing nothing. Often I will not be good at articulating the situation or objectives. Make assumptions where you can. Consider how I might overlook conspicuously crucial information and make a stupid decision, such as new environments, social proof, desire to look good, authority bias, information overload, stress, fatigue, urgency to do something, paradox of choice, desire for control, cognitive dissonance, confirmation bias, overconfidence, survivorship bias, resulting, recency bias, anchoring, and other biases.

    Respond with clarity, brevity, politeness, and helpfulness using fewer words and a warm and encouraging tone. Ask at MOST 1 question if you are going to ask me a question. Do not use the word "and" in your response.

    You can keep these questions in mind when helping me with my decision making:{QUESTIONS}
    """
    return generate_outcomes, normal


GENERATE_OUTCOMES_PROMPT = f"""{DEFAULT_PREFIX_PROMPT}

Recommend the best choice for a given situation, with the given objectives.

### Context
Your task is to help me choose the best choice for a given situation, with the given objectives. The outcomes you generate should help me see the range of possibilities and choose the best path forward. Sometimes the best path forward is doing nothing. You should consider other options and choices before picking this one.

Often I will not be good at articulating my situation. Make assumptions where you can. Do not mention authors or sources in your response. Consider how I might overlook conspicuously crucial information and make a stupid decision, such as new environments, social proof, desire to look good, authority bias, information overload, stress, fatigue, urgency to do something, paradox of choice, desire for control, cognitive dissonance, confirmation bias, overconfidence, survivorship bias, resulting, recency bias, anchoring, and other biases.
"""


NORMAL_PROMPT = f"""{DEFAULT_PREFIX_PROMPT}

When discussing choices, analyze trade-offs, consider uncertainties, and help users think through their options. Sometimes the best path forward is doing nothing. Often I will not be good at articulating the situation or objectives. Make assumptions where you can. Consider how I might overlook conspicuously crucial information and make a stupid decision, such as new environments, social proof, desire to look good, authority bias, information overload, stress, fatigue, urgency to do something, paradox of choice, desire for control, cognitive dissonance, confirmation bias, overconfidence, survivorship bias, resulting, recency bias, anchoring, and other biases.

Respond with clarity, brevity, politeness, and helpfulness using fewer words and a warm and encouraging tone. Ask at MOST 1 question if you are going to ask me a question. Do not use the word "and" in your response.

You can keep these questions in mind when helping me with my decision making:
{QUESTIONS}
"""


class MessageResponse(BaseModel):
    text: str = Field(description="Your response to the conversation.")
    suggested_messages: List[str] = Field(
        ...,
        description="A list of suggested messages that the user could send back to the response you generate. The messages should be from the user's point of view in the conversation. If you are asking me a question in your response, the suggested messages should be answers to that question. The suggested messages should be 5-10 words long. There should be at most 3 suggested messages.",
    )


class Choice(BaseModel):
    name: str = Field(
        ...,
        description="The name of the choice that this outcome is for. This should be a single word or phrase. It should be an action that I can take.",
    )
    assumptions: List[str] = Field(
        ...,
        description="A list of 1 or 2 assumptions that were made to generate this outcome. These assumptions must materially affect the outcome of the choice. If the assumption does not affect whether I make this choice, then do not include it. Use at most 8 words for each assumption.",
    )

    best_case_scenario: str = Field(
        ..., description="The best case scenario for this outcome. Use at most 8 words."
    )
    worst_case_scenario: str = Field(
        ...,
        description="The worst case scenario for this outcome. Use at most 8 words.",
    )

    def __str__(self):
        return (
            f"## {self.name}\n"
            f"### Scenarios:\n"
            f"- Best: {self.best_case_scenario}\n"
            f"- Worst: {self.worst_case_scenario}\n\n"
        )


class ChoicesResponse(BaseModel):
    choices: List[Choice] = Field(description="A list of choices.")
    title: str = Field(
        description="A question that summarizes the decision being considered. The choices should be answers to this question. An example: 'Should I stay in my current job?'"
    )
    uncertainties: List[str] = Field(
        ...,
        description="A list of 1 or 2 uncertainties about the decision being considered. These uncertainties must materially affect the quality of the choice. If the uncertainty does not affect whether I make this choice, then do not include it. A good uncertainty captures what would need to be true for this to be the right decision or wrong decision, depending on the answer. Each uncertainty should be a question that ends with a question mark. Use at most 8 words for each uncertainty.",
    )
    next_steps: List[str] = Field(
        ...,
        description="A list of 1 or 2 next steps for this decision. These next steps should be actions that I can take or information that I can gather to reduce my uncertainty about the decision. Use at most 8 words for each next step.",
    )


class PrioritiesOutput(BaseModel):
    decision: str = Field(..., description="The decision being considered")
    objectives: List[str] = Field(
        ...,
        description="Three to five key preferences or objectives related to the decision. Use at most 3 words for each objective. Examples: access to nature, work life balance, time with family, financial stability, etc.",
    )


class Objection(BaseModel):
    objection: str = Field(
        ..., description="An objection to the decision being considered"
    )
    mitigation_strategies: List[str] = Field(
        ...,
        description="A list of the best mitigation strategies for the objection. Use at most 8 words for each mitigation strategy. The mitigation strategies should be actions that the user can take to address the objection. There should be at most 3 mitigation strategies.",
    )


class ObjectionsOutput(BaseModel):
    objections: List[Objection] = Field(
        ..., description="A list of objections to the decision being considered"
    )
    suggested_messages: List[str] = Field(
        ...,
        description="A list of suggested messages that the user could send back to the response you generate. The messages should be from the user's point of view in the conversation. If you are asking me a question in your response, the suggested messages should be answers to that question. The suggested messages should be 5-10 words long. There should be at most 3 suggested messages.",
    )


set_default_openai_api("chat_completions")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL_PRO = "gemini-2.5-pro"
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-0"
CLAUDE_BASE_URL = "https://api.anthropic.com/v1/"
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)
claude_client = AsyncOpenAI(base_url=CLAUDE_BASE_URL, api_key=CLAUDE_API_KEY)


def create_orchestrator_agent(prefix: str) -> Agent:
    generate_outcomes_prompt, normal_prompt = build_prompts(prefix)

    conversation_agent = Agent(
        name="Conversation Agent",
        handoff_description="This agent makes general conversation with the user to try to help them clarify their situation and figure out what they want to do.",
        instructions=normal_prompt,
        output_type=MessageResponse,
        model=OpenAIChatCompletionsModel(
            model=GEMINI_MODEL, openai_client=gemini_client
        ),
        model_settings=ModelSettings(reasoning=Reasoning(effort=None)),
    )

    priorities_agent = Agent(
        name="Priorities Agent",
        handoff_description="This agent asks the user for their priorities for the decision being considered.",
        instructions=prefix
        + "\nGenerate the decision being considered and five key objectives",
        output_type=PrioritiesOutput,
        model=OpenAIChatCompletionsModel(
            model=GEMINI_MODEL, openai_client=gemini_client
        ),
        model_settings=ModelSettings(reasoning=Reasoning(effort=None)),
    )

    recommendation_agent = Agent(
        name="Recommendation Agent",
        handoff_description="This agent generates the recommended choice to make for the decision being considered. Use this once you've gathered enough information about the user's situation, such as what they care about, why they care about it, and what they're willing to do to achieve it.",
        instructions=generate_outcomes_prompt,
        output_type=ChoicesResponse,
        model=OpenAIChatCompletionsModel(
            model=GEMINI_MODEL, openai_client=gemini_client
        ),
        model_settings=ModelSettings(reasoning=Reasoning(effort=None)),
    )

    objections_agent = Agent(
        name="Objections Agent",
        handoff_description="This agent generates mitigation strategies for any top objections to the decision being considered. Use this when the user has expressed concerns about a recommended choice.",
        instructions=prefix
        + "\nGiven the conversation and priorities, generate mitigation strategies for any top objections.",
        output_type=ObjectionsOutput,
        model=OpenAIChatCompletionsModel(
            model=GEMINI_MODEL, openai_client=gemini_client
        ),
        model_settings=ModelSettings(reasoning=Reasoning(effort=None)),
    )

    orchestrator_agent = Agent(
        name="Orchestrator Agent",
        instructions=prefix
        + "\nRoute the user input to the correct agent based on what the user needs in the conversation. If you don't know what the user needs, ask them using the conversation agent. After getting more information, use the priorities agent to get their priorities. Only ask for their priorities if they haven't already provided them. Finally, use the recommendation agent to generate a recommended choice. If the user has any objections to the recommended choice, use the objections agent to generate mitigation strategies.",
        handoffs=[
            conversation_agent,
            priorities_agent,
            recommendation_agent,
            objections_agent,
        ],
        model=OpenAIChatCompletionsModel(
            model=GEMINI_MODEL, openai_client=gemini_client
        ),
        model_settings=ModelSettings(
            tool_choice="required", reasoning=Reasoning(effort=None)
        ),
    )

    return orchestrator_agent


DEFAULT_ORCHESTRATOR_AGENT = create_orchestrator_agent(DEFAULT_PREFIX_PROMPT)


async def ask(
    messages,
    advisor: str = "thought_partner",
    custom_prefix: str | None = None,
    context=None,
):
    prefix = get_prefix(advisor, custom_prefix)
    agent = (
        DEFAULT_ORCHESTRATOR_AGENT
        if prefix == DEFAULT_PREFIX_PROMPT
        else create_orchestrator_agent(prefix)
    )
    input_items = [{"content": msg["content"], "role": msg["role"]} for msg in messages]
    return await Runner.run(agent, input_items, context=context)
