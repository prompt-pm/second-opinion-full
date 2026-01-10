================================================================================

PROJECT DIRECTORY STRUCTURE

================================================================================

+-- .
+-- Makefile
+-- __init__.py
+-- ai_functions.py
+-- backend.py
+-- database.py
+-- decision_records.db
+-- dump_codebase.sh
+-- frontend/
|   +-- capacitor.config.json
|   +-- index.html
|   +-- package-lock.json
|   +-- package.json
|   +-- postcss.config.js
|   +-- src/
|   |   +-- actions.js
|   |   +-- decision-helper.js
|   |   +-- main.js
|   |   +-- styles.css
|   |   +-- tailwind.css
|   +-- tailwind.config.js
|   +-- vite.config.js
+-- instrumentor.py
+-- models.py
+-- prompts.py
+-- requirements.txt
+-- scripts/
|   +-- generate_sitemap.py

================================================================================

FILE CONTENTS

================================================================================



================================================================================
FILE: ./backend.py
================================================================================
from fastapi import Request, FastAPI, Depends
import modal
import os
import json
from modal import App, Image, asgi_app, Mount
from ai_functions import (
    cerebras_choose_option,
    cerebras_generate_outcomes,
    cerebras_generate_alternative,
    cerebras_router,
    generate_question,
    cerebras_generate_next_steps,
    cerebras_suggest_additional_action,
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from instrumentor import set_easy_tracing_instrumentation
from phoenix.trace import using_project
import sentry_sdk
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database import Base, DecisionRecord

tracing_initialized = False


def initiate_sentry():
    sentry_sdk.init(
        dsn="https://37b97329990b0c3eecf0bb5eefb64136@o190156.ingest.us.sentry.io/4508088928567296",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )


def initiate_tracing():
    global tracing_initialized
    if not tracing_initialized:
        set_easy_tracing_instrumentation()
        # set_hosted_phoenix_instrumentation()
        initiate_sentry()
        tracing_initialized = True
        print("Tracing instrumentation initialized")


# image = (
#     Image.debian_slim()
#     .pip_install("uv")
#     .run_commands("uv pip install --system --compile-bytecode ./requirements.txt")
# )

image = Image.debian_slim().pip_install_from_requirements("./requirements.txt").add_local_python_source("ai_functions", "instrumentor", "models", "prompts", "database")
mount = Mount.from_local_dir("./assets", remote_path="/assets")
app = App(image=image, mounts=[mount])
web_app = FastAPI()

# Database setup
# Use a path from the root directory to assets/data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "assets", "data")
# Create the data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)
db_path = os.path.join(data_dir, "decision_records.db")
DATABASE_URL = f"sqlite:///{db_path}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Dependency to provide a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5500",
    "http://localhost:5501",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501",
    "http://choices.dev",
    "https://choices.dev",
    "http://oksayless.com",
    "https://oksayless.com",
    "http://overthinking.app",
    "https://overthinking.app",
    "http://overthinking.help",
    "https://overthinking.help",
]

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.post("/api/initial_query/")
async def initial_query(request: Request):
    initiate_tracing()
    data = await request.json()
    message_history = data.get("message_history", [])

    with using_project("say-less"):
        response = cerebras_router(message_history)
        response_json = {
            "prompt": response.prompt,
            "response": response.response.model_dump(),
        }
        return JSONResponse(content=response_json)


@web_app.post("/api/query/")
async def query(request: Request):
    initiate_tracing()
    data = await request.json()
    messages = data.get("messages", [])
    with using_project("say-less"):
        response = cerebras_router(messages)
        response_json = {
            "prompt": response.prompt,
            "response": response.response.model_dump(),
        }
        return JSONResponse(content=response_json)


@web_app.post("/api/questions/")
async def questions(request: Request):
    initiate_tracing()
    data = await request.json()
    situation = data.get("situation")
    results = data.get("results")
    questions = data.get("questions")
    with using_project("say-less"):
        response_message = generate_question(situation, results, questions)

    response_json = {
        "uncertainties": [response_message],
    }
    return JSONResponse(content=response_json)


@web_app.post("/api/choices")
async def simulate_choices(request: Request):
    initiate_tracing()
    data = await request.json()
    message_history = data.get("message_history", [])

    with using_project("say-less"):
        response = cerebras_generate_outcomes(message_history)

    response_json = {
        "choices": [choice.model_dump() for choice in response.choices],
        "title": response.title,
        "uncertainties": response.uncertainties,
        "next_steps": response.next_steps,
    }
    return JSONResponse(content=response_json)


@web_app.get("/")
def read_root():
    initiate_tracing()
    return {"Hello": "World"}


@web_app.get("/sentry-debug")
async def trigger_error():
    initiate_tracing()
    division_by_zero = 1 / 0


@app.function(
    image=image,
    mounts=[mount],
    allow_concurrent_inputs=50,
    keep_warm=1,
    secrets=[
        modal.Secret.from_name("openai-key"),
        modal.Secret.from_name("cerebras-key"),
    ],
)
@asgi_app()
def fastapi_app():
    return web_app


@web_app.post("/api/add_alternative/")
async def add_alternative(request: Request):
    initiate_tracing()
    data = await request.json()
    situation = data.get("situation", "")
    results = data.get("results")

    with using_project("say-less"):
        response = cerebras_generate_alternative(situation, results)

    response_json = {
        "new_alternative": response.model_dump(),
    }

    return JSONResponse(content=response_json)


@web_app.post("/api/choose/")
async def choose_option(request: Request):
    initiate_tracing()
    data = await request.json()
    situation = data.get("situation", "")
    results = data.get("results")
    selected_index = data.get("selectedIndex")

    with using_project("say-less"):
        response = cerebras_choose_option(situation, results, selected_index)

    response_json = {
        "chosen_index": response.chosen_index,
        "explanation": response.explanation,
    }

    return JSONResponse(content=response_json)


@web_app.post("/api/next_steps")
async def next_steps(request: Request):
    initiate_tracing()
    data = await request.json()
    situation = data.get("situation", "")
    choice_name = data.get("choice_name", "")
    choice_index = data.get("choice_index")
    results = data.get("results")

    with using_project("say-less"):
        response = cerebras_generate_next_steps(situation, choice_name, results, choice_index)

    response_json = {
        "next_steps": response.next_steps,
    }

    return JSONResponse(content=response_json)


@web_app.post("/api/suggest_additional_action")
async def suggest_additional_action(request: Request):
    initiate_tracing()
    data = await request.json()
    situation = data.get("situation", "")
    existing_next_steps = data.get("existing_next_steps", [])
    results = data.get("results", {})

    with using_project("say-less"):
        response = cerebras_suggest_additional_action(situation, existing_next_steps, results)

    response_json = {
        "additional_action": response.additional_action,
    }
    return JSONResponse(content=response_json)


@web_app.post("/api/save_decision/")
async def save_decision(request: Request, db: Session = Depends(get_db)):
    initiate_tracing()
    data = await request.json()
    message_history = data.get("message_history", [])  # List of message objects
    if not message_history:
        return JSONResponse(content={"error": "Message history is empty"}, status_code=400)
    
    unique_id = str(uuid4())  # Generate a unique ID
    message_history_json = json.dumps(message_history)  # Convert to JSON string
    decision_record = DecisionRecord(id=unique_id, message_history=message_history_json)
    
    db.add(decision_record)
    db.commit()
    return JSONResponse(content={"id": unique_id})


@web_app.get("/api/get_decision/{decision_id}")
async def get_decision(decision_id: str, db: Session = Depends(get_db)):
    initiate_tracing()
    record = db.query(DecisionRecord).filter(DecisionRecord.id == decision_id).first()
    if record is None:
        return JSONResponse(content={"error": "Decision not found"}, status_code=404)
    message_history = json.loads(record.message_history)
    return JSONResponse(content={"message_history": message_history})


================================================================================
FILE: ./models.py
================================================================================
from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class Prompt(str, Enum):
    NORMAL = "normal"
    WEB_SEARCH = "web_search"
    # CHOOSE_FOR_ME = "choose_for_me"
    # GENERATE_FEEDBACK = "generate_feedback"
    # DECISION_CHAT = "decision_chat"

    def to_json(self):
        return self.value

    def __json__(self):
        return self.value


class PromptSelection(BaseModel):
    prompt: str = Field(
        description="The prompt to use to answer the user's question most effectively. Return the prompt title and nothing else."
    )


class RoleplayScenarioResponse(BaseModel):
    response: str = Field(description="The response of the narrator.")
    suggested_choices: List[str] = Field(
        description="A list of 3 choices that the user could make to move forward in the scenario. Keep these choices to less than 10 words long. Be as specific as possible.",
    )


class MessageResponse(BaseModel):
    text: str = Field(
        description="Your response to the conversation."
    )
    suggested_messages: List[str] = Field(
        ...,
        description="A list of suggested messages that the user could send back to the response you generate. The messages should be from the user's point of view in the conversation. If you are asking me a question in your response, the suggested messages should be answers.",
    )


class RouterResponse(BaseModel):
    response: BaseModel = Field(description="The response of the router.")
    prompt: Prompt = Field(
        description="The prompt that was used to generate the response."
    )


class UIResponse(BaseModel):
    text: str = Field(description="The text to display to the user.")
    citations: List[str] = Field(
        default=[],
        description="A list of citations for the response. These should be links to the sources of the information in the response.",
    )
    suggested_messages: List[str] = Field(
        default=[],
        description="A list of suggested follow-up messages that the user might want to send.",
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


class ChooseForMeResponse(BaseModel):
    explanation: str = Field(
        description="A short explanation of why this choice could be a good decision. Use at most 30 words. Write as though you are speaking to me, and use words that I would expect to hear from a friend as a suggestion. The explanation should be a single complete sentence."
    )
    chosen_index: int = Field(
        description="The index of the chosen option. The index starts at 0. It can go up to the number of options minus 1."
    )


class Character(BaseModel):
    name: str
    age: int
    gender: str
    location: str
    occupation: str
    mood: str
    facts: List[str] = Field(..., description="A list of 3 facts about the character")
    goals: List[str] = Field(
        ..., description="A list of 3 goals the character is trying to achieve"
    )
    fears: List[str] = Field(
        ..., description="A list of 3 fears the character is experiencing"
    )
    desires: List[str] = Field(
        ..., description="A list of 3 desires the character is experiencing"
    )
    joys: List[str] = Field(
        ..., description="A list of 3 joys the character is experiencing"
    )
    problems: List[str] = Field(
        ..., description="A list of 3 problems the character is facing"
    )


class Message(BaseModel):
    role: str
    content: str


class MessageList(BaseModel):
    messages: List[Message] = Field(description="A list of messages.")

    def __str__(self):
        return "\n".join(
            [f"{message.role}: {message.content}" for message in self.messages]
        )

    def add_message(self, message: Message):
        self.messages.append(message)

    def prepend_message(self, message: Message):
        self.messages.insert(0, message)

    def clear(self):
        self.messages = []

    def return_openai_format(self):
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]

    def return_persona_format(self):
        message_list = []
        for message in self.messages:
            if message.role == "user":
                message_list.append(f"user: {message.content}")
            else:
                message_list.append(f"persona: {message.content}")
        return "\n".join(message_list)

    def return_json(self):
        return [message.model_dump() for message in self.messages]


class NextStepsResponse(BaseModel):
    next_steps: List[str] = Field(
        ...,
        min_items=1,  # Ensures at least 1 step
        max_items=2,  # Limits to no more than 2 steps
        description="A list of 1-2 specific, actionable next steps for the user. Each step should be clear, concrete, and start with a verb. Each step should be 5-15 words and directly applicable to the situation."
    )


class AdditionalActionResponse(BaseModel):
    additional_action: str = Field(
        description="A single additional action suggestion that complements the user's existing actions. The action should be specific, actionable, start with a verb, and be 5-15 words long."
    )


================================================================================
FILE: ./ai_functions.py
================================================================================
import instructor
import os
from openai import OpenAI

from cerebras.cloud.sdk import Cerebras
from groq import Groq
from models import (
    ChoicesResponse,
    ChooseForMeResponse,
    MessageResponse,
    Prompt,
    PromptSelection,
    Choice,
    RouterResponse,
    UIResponse,
    NextStepsResponse,
    AdditionalActionResponse,
)

from prompts import (
    CHOOSE_OPTION_PROMPT,
    GENERATE_ALTERNATIVE_PROMPT,
    GENERATE_FEEDBACK_PROMPT,
    GENERATE_OUTCOMES_PROMPT,
    GENERATE_QUESTIONS_PROMPT,
    NORMAL_PROMPT,
    PROMPT_SELECTION_PROMPT,
    WEB_SEARCH_PROMPT,
    PRE_DECISION_NEXT_STEPS_PROMPT,
    POST_DECISION_NEXT_STEPS_PROMPT,
    SUGGEST_ADDITIONAL_ACTION_PROMPT,
)
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

CEREBRAS_MODEL = "llama-3.3-70b"
GROQ_MODEL = "llama-3.3-70b-specdec"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PPLX_API_KEY = os.environ.get("PPLX_API_KEY")


def groq_or_cerebras(messages, temperature, response_model=None):
    """
    Try Groq first, if that fails, try Cerebras.
    """
    try:
        if response_model is not None:
            client = instructor.from_groq(Groq(api_key=GROQ_API_KEY))
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                response_model=response_model,
                max_retries=2
            )    
        else:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_retries=2
            )
    except Exception as e:
        print(f"Error with Groq: {e}")
        if response_model is not None:
            client = instructor.from_cerebras(Cerebras(), mode=instructor.Mode.CEREBRAS_JSON)
            response = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                response_model=response_model,
                max_retries=2
            )
        else:
            client = Cerebras()
            response = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_retries=2
            )
    return response


def cerebras_router(message_history):
    """
    Route to the appropriate ai prompt based on the input messages.

    Returns:
    {
        "prompt": Prompt,
        "response": Object or JSON dictionary
    }

    The response object will either be an object like ChoicesResponse, Choice, or a JSON dictionary with the following keys:
    {
        "text": str,
        "citations": list[str]
    }
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_router") as span:
        cleaned_messages = clean_up_messages(message_history)
        messages = [
            {"role": "user", "content": PROMPT_SELECTION_PROMPT.format(message_history=str(cleaned_messages))}
        ]
        router_response = groq_or_cerebras(messages, 0.2, PromptSelection)
        prompt = router_response.prompt
        last_message = cleaned_messages[-1]["content"]
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.INPUT_VALUE, str(last_message))
        # Map prompts to their handler functions
        handlers = {
            Prompt.WEB_SEARCH: lambda: perplexity_web_search(cleaned_messages),
            Prompt.NORMAL: lambda: cerebras_normal(cleaned_messages),
        }
        # Check if we have a handler for this prompt type
        if prompt not in handlers:
            return cerebras_normal(cleaned_messages)  # Default to normal chat if no specific handler

        # Execute the handler and return results
        full_response = handlers[prompt]()
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(full_response))
        return RouterResponse(prompt=prompt, response=full_response)


def perplexity_web_search(message_history):
    """
    Perform a web search for a given situation using Perplexity.
    """
    pplx_client = OpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")

    # Extract the last message
    last_message = message_history[-1]
    # Get all previous messages for context
    previous_messages = message_history[:-1]
    
    # Add the final message with the web search prompt
    previous_messages.append({
        "role": "user",
        "content": WEB_SEARCH_PROMPT.format(conversation=last_message.get("content", ""))
    })

    response = pplx_client.chat.completions.create(
        model="sonar",
        messages=previous_messages,
    )
    if hasattr(response, "citations"):
        return UIResponse(
            text=response.choices[0].message.content, citations=response.citations
        )
    else:
        return UIResponse(text=response.choices[0].message.content)


def cerebras_normal(message_history):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_normal") as span:
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, NORMAL_PROMPT)
        span.set_attribute(SpanAttributes.INPUT_VALUE, str(message_history))
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        messages = [
            {"role": "system", "content": NORMAL_PROMPT}
        ] + message_history
        response = groq_or_cerebras(messages, 0.2, MessageResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return UIResponse(text=response.text, suggested_messages=response.suggested_messages)


def cerebras_generate_outcomes(message_history):
    """
    Generate 3 outcomes for a given situation.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_generate_outcomes") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, GENERATE_OUTCOMES_PROMPT)
        messages = [
            {"role": "system", "content": GENERATE_OUTCOMES_PROMPT}
        ] + message_history
        response = groq_or_cerebras(messages, 0.2, ChoicesResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response


def cerebras_generate_alternative(situation, results):
    """
    Generate a new, unique alternative that wasn't previously considered. The alternative should be realistic and relevant to the situation.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_generate_alternative") as span:
        # Extract situation context from results if situation is empty
        if not situation and results and "title" in results:
            situation = results.get("title", "")
            
        prompt = GENERATE_ALTERNATIVE_PROMPT.format(
            situation=situation, results=results
        )
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE, GENERATE_ALTERNATIVE_PROMPT
        )
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
            str({"situation": situation, "results": results}),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, situation)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 0.2, Choice)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response



def cerebras_choose_option(situation, results, current_selected_index=None):
    """
    Choose an option based on the situation and results.
    If current_selected_index is provided, choose a different option.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_choose_option") as span:
        # Extract situation context from results if situation is empty
        if not situation and results and "title" in results:
            situation = results.get("title", "")

        total_options = len(results.get("choices", []))
        # Update prompt to include current selection if it exists
        if current_selected_index is not None:
            prompt = CHOOSE_OPTION_PROMPT.format(
                situation=situation,
                results=results,
                current_choice=current_selected_index,
                total_options=total_options,
            )
        else:
            prompt = CHOOSE_OPTION_PROMPT.format(
                situation=situation, results=results, current_choice="None", total_options=total_options
            )
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, CHOOSE_OPTION_PROMPT)
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
            str(
                {
                    "situation": situation,
                    "results": results,
                    "current_selected_index": current_selected_index,
                }
            ),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, situation)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 1, ChooseForMeResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response


def generate_question(situation, results, questions=None):
    """
    Generate a question for a given situation and results.
    """
    tracer = trace.get_tracer(__name__)
    if not situation and results and "title" in results:
        situation = results.get("title", "")

    with tracer.start_as_current_span("groq_generate_questions") as span:
        prompt = GENERATE_QUESTIONS_PROMPT.format(situation=situation, results=results, questions=questions)
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, GENERATE_QUESTIONS_PROMPT)
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 0.9)
        response_message = response.choices[0].message.content
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response_message))
    return response_message


def clean_up_messages(messages):
    # Keep track of total characters
    total_chars = sum(len(msg["content"]) for msg in messages)

    # Target is ~30K chars (roughly 8K tokens)
    while total_chars > 30000:
        if len(messages) <= 2:
            # If only system prompt and last message remain, truncate last message
            last_msg = messages[-1]["content"]
            # Keep first 1000 chars and last 1000 chars to maintain context
            if len(last_msg) > 2000:
                messages[-1]["content"] = last_msg[:1000] + "..." + last_msg[-1000:]
        else:
            # Remove oldest non-system messages in pairs
            for i in range(1, len(messages) - 1):
                if messages[i]["role"] == "user":
                    # Remove this user message and next assistant message if it exists
                    messages.pop(i)
                    if i < len(messages) and messages[i]["role"] == "assistant":
                        messages.pop(i)
                    break

        total_chars = sum(len(msg["content"]) for msg in messages)

    # Ensure messages only have role and content attributes that are STRINGS
    cleaned_messages = []
    for msg in messages:
        cleaned_messages.append({
            "role": msg["role"],
            "content": str(msg["content"])
        })
    messages = cleaned_messages
    return messages


def cerebras_generate_next_steps(situation, choice_name=None, results=None, choice_index=None):
    """
    Generate actionable next steps for a situation, either before or after a decision.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_generate_next_steps") as span:
        # Determine if it's post-decision (choice provided) or pre-decision (no choice)
        if choice_name or choice_index is not None:
            # Post-decision scenario
            if choice_index is not None and results and "choices" in results:
                try:
                    selected_choice = results["choices"][choice_index]
                except (IndexError, TypeError):
                    selected_choice = {"name": choice_name} if choice_name else {}
            else:
                selected_choice = {"name": choice_name} if choice_name else {}
            
            prompt = POST_DECISION_NEXT_STEPS_PROMPT.format(
                situation=situation,
                results=results if results else {},
                choice_name=choice_name if choice_name else "",
                selected_choice=selected_choice
            )
            prompt_template = POST_DECISION_NEXT_STEPS_PROMPT
        else:
            # Pre-decision scenario
            prompt = PRE_DECISION_NEXT_STEPS_PROMPT.format(
                situation=situation,
                results=results if results else {}
            )
            prompt_template = PRE_DECISION_NEXT_STEPS_PROMPT
        
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE, 
            prompt_template
        )
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
            str({
                "situation": situation, 
                "choice_name": choice_name,
                "results": results,
                "choice_index": choice_index
            }),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 0.2, NextStepsResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response


def cerebras_suggest_additional_action(situation, existing_next_steps, results=None):
    """
    Generate a single additional action that complements existing actions.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_suggest_additional_action") as span:
        prompt = SUGGEST_ADDITIONAL_ACTION_PROMPT.format(
            situation=situation,
            existing_next_steps=existing_next_steps,
            results=results if results else {}
        )
        
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, SUGGEST_ADDITIONAL_ACTION_PROMPT)
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, str({
            "situation": situation,
            "existing_next_steps": existing_next_steps,
            "results": results
        }))
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 0.2, AdditionalActionResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response

================================================================================
FILE: ./database.py
================================================================================
from sqlalchemy import Column, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DecisionRecord(Base):
    __tablename__ = 'decision_records'
    id = Column(String, primary_key=True)  # Unique ID for the decision
    message_history = Column(Text)         # Conversation thread as JSON string

================================================================================
FILE: ./__init__.py
================================================================================


================================================================================
FILE: ./prompts.py
================================================================================
PREFIX_PROMPT = """You are a helpful AI assistant focused on helping users make decisions, using principles from Annie Duke's Thinking in Bets, Maxims for Thinking Analytically, Decisive by the Heath Brothers, Psychology of Human Misjudgment by Charlie Munger, and other sources. Do not mention these authors in your response."""

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

PROMPT_SELECTION_PROMPT = f"""{PREFIX_PROMPT}

Based on the conversation, you will choose the best prompt to answer my question best. The LAST user message is most important. 

Return the prompt title and nothing else.

Here are your prompt choices:
- normal: for general conversation and questions about decision making, use this.
- web_search: for web research for latest events, such as weather, news, facts, statistics, and more.

[BEGIN MESSAGE HISTORY]
{{message_history}}
[END MESSAGE HISTORY]
"""

GENERATE_OUTCOMES_PROMPT = f"""{PREFIX_PROMPT}

Generate 3 outcomes for a given situation, with the given objectives.

### Context
Your task is to help me widen my options by suggesting options. The outcomes you generate should help me see the range of possibilities and choose the best path forward. Sometimes the best path forward is doing nothing. 

Often I will not be good at articulating my situation. Make assumptions where you can. Do not mention authors or sources in your response. Consider how I might overlook conspicuously crucial information and make a stupid decision, such as new environments, social proof, desire to look good, authority bias, information overload, stress, fatigue, urgency to do something, paradox of choice, desire for control, cognitive dissonance, confirmation bias, overconfidence, survivorship bias, resulting, recency bias, anchoring, and other biases.
"""


GENERATE_ROLEPLAY_SCENARIO_PROMPT = """I have made the following choice for a given scenario.
### BEGIN SCENARIO
{scenario}
### END SCENARIO

You are the narrator of this scenario. You will observe different facts, feelings, and observations about the scenario. I will continually make different choices, and you will observe the consequences of those choices. Add variance to the scenario to make it more engaging. This is the {best_or_worst} case scenario.
"""

GENERATE_FEEDBACK_PROMPT = """You are a decision making expert, using principles from Annie Duke's Thinking in Bets, Maxims for Thinking Analytically, Decisive by the Heath Brothers, Psychology of Human Misjudgment by Charlie Munger, and other sources. I have a question or message for you, and I want you to help me with a given situation, objectives, proposed options, and message. Use at most 50 words.

### Context
The situation is the decision that I need help with and what's important to me. The proposed options are the possibilities that I am considering. Your job is to help me see the range of possibilities and choose the best path forward. Sometimes the best path forward is doing nothing. Often I will not be good at articulating the situation or objectives. Make assumptions where you can. Do not mention authors or sources in your response. Consider how I might overlook conspicuously crucial information and make a stupid decision, such as new environments, social proof, desire to look good, authority bias, information overload, stress, fatigue, urgency to do something, paradox of choice, desire for control, cognitive dissonance, confirmation bias, overconfidence, survivorship bias, resulting, recency bias, anchoring, and other biases.

### Response format
Use at most 50 words. Respond in markdown format. For every line break, use two newlines. Use line breaks often. Respond with a warm and encouraging tone.

### Situation
{situation}

### Proposed options
{results}

### Message
{message}
"""


GENERATE_ALTERNATIVE_PROMPT = f"""{PREFIX_PROMPT}

I will provide you with my situation and the current proposed options. Your task is to generate a new, unique alternative decision. The alternative should be realistic and relevant to my situation. The alternative must be different than any of the current proposed options.

### Situation
{{situation}}

### Current proposed options
{{results}}
"""

CHOOSE_OPTION_PROMPT = f"""{PREFIX_PROMPT}

I will provide you with my situation and the potential options. Your task is to choose an option for me. Select an option completely at random, and provide a plausible explanation for why it could be a good decision. If a current selected option is provided, choose a different option.

### Situation
{{situation}}

### The total number of options
{{total_options}}

### Potential options
{{results}}

### Current selected option (0-indexed)
{{current_choice}}
"""

WEB_SEARCH_PROMPT = f"""

I am missing some information that I need to search the web for. I will provide you with our current conversation, and you should respond to my current inquiry with clarity, brevity, politeness, and helpfulness with fewer words.

[BEGIN CONVERSATION]
{{conversation}}
[END CONVERSATION]
"""

NORMAL_PROMPT = f"""{PREFIX_PROMPT}

When discussing choices, analyze trade-offs, consider uncertainties, and help users think through their options. Sometimes the best path forward is doing nothing. Often I will not be good at articulating the situation or objectives. Make assumptions where you can. Consider how I might overlook conspicuously crucial information and make a stupid decision, such as new environments, social proof, desire to look good, authority bias, information overload, stress, fatigue, urgency to do something, paradox of choice, desire for control, cognitive dissonance, confirmation bias, overconfidence, survivorship bias, resulting, recency bias, anchoring, and other biases.

Respond with clarity, brevity, politeness, and helpfulness using fewer words and a warm and encouraging tone. Ask at MOST 1 question if you are going to ask me a question. Do not use the word "and" in your response.

You can keep these questions in mind when helping me with my decision making:
{QUESTIONS}
"""


GENERATE_QUESTIONS_PROMPT = f"""{PREFIX_PROMPT}

Your task is to generate the most pertinent question to ask me about my given situation, which is the greatest uncertainty I am facing. 

You should respond with the best question that is not the current question being asked. Your response should be one sentence with a max of 20 words.

I will also provide you with a list of questions as inspiration which are really useful questions for decision making, and why they are useful.

### Situation
{{situation}}

### Current questions
{{questions}}

### Proposed options
{{results}}

### Questions as inspiration
{QUESTIONS}
"""

PRE_DECISION_NEXT_STEPS_PROMPT = f"""{PREFIX_PROMPT}

Your task is to suggest 1 or 2 specific actions I can take to make progress in my situation or to gather more information that will help me make a decision.

### Context
I am considering my options and need guidance on what to do next to better understand my situation or the potential choices.

### Guidelines for suggestions:
1. Be specific and actionable - start with a verb
2. Focus on information gathering, reducing uncertainty, or preparatory steps
3. Keep each suggestion concise (5-15 words)
4. Consider what information or actions would be most helpful in making the decision
5. Think about potential obstacles or uncertainties and how to address them

### Situation
{{situation}}

### Current analysis (if available)
{{results}}
"""

POST_DECISION_NEXT_STEPS_PROMPT = f"""{PREFIX_PROMPT}

Your task is to generate 1 or 2 specific, actionable next steps for me based on the decision I've made. These steps should help me implement my decision effectively.

### Context
I have decided on a specific option for my situation. Now I need concrete actions to take to move forward with this decision.

### Guidelines for next steps:
1. Be specific and concrete - avoid vague suggestions
2. Make each step actionable - start with a verb
3. Keep steps concise (5-15 words each)
4. Focus on immediate actions I can take within the next few days/weeks

### Situation
{{situation}}

### Decision Context
{{results}}

### My chosen option
{{choice_name}}

### Selected choice details
{{selected_choice}}
"""

# Add new prompt for suggesting an additional action
SUGGEST_ADDITIONAL_ACTION_PROMPT = f"""{PREFIX_PROMPT}

Your task is to suggest one additional specific action I can take, based on my current situation and the actions I already have planned.

### Context
I have already planned some actions and need one more suggestion to complement or add to them.

### Guidelines for the suggestion:
1. Be specific and actionable - start with a verb
2. Ensure it is different from the existing actions
3. Keep it concise (5-15 words)
4. Focus on what would be most helpful next

### Situation
{{situation}}

### Existing actions
{{existing_next_steps}}

### Current analysis (if available)
{{results}}
"""

================================================================================
FILE: ./instrumentor.py
================================================================================
from phoenix.otel import register
from openinference.instrumentation.groq import GroqInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.instructor import InstructorInstrumentor
import os


def set_easy_tracing_instrumentation():
    """Set tracing instrumentation for Phoenix and Arize"""
    # tracer_provider = register(batch=True)
    tracer_provider = register(
        batch=True, endpoint="http://phoenix-kd03.onrender.com/v1/traces"
    )
    # InstructorInstrumentor().instrument(tracer_provider=tracer_provider)
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    print("Tracing instrumentation set")


def set_hosted_phoenix_instrumentation():
    """Set tracing instrumentation for Phoenix and Arize"""
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
    os.environ["PHOENIX_CLIENT_HEADERS"] = (
        f"api_key={os.environ.get('PHOENIX_API_KEY')}"
    )
    # Setup OTEL tracing for hosted Phoenix. The register function will automatically detect the endpoint and headers from your environment variables.
    tracer_provider = register(batch=True)

    # Turn on instrumentation for OpenAI
    # InstructorInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    print("Tracing instrumentation set")


================================================================================
FILE: ./scripts/generate_sitemap.py
================================================================================
import requests
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

# Ghost API settings
GHOST_CONTENT_API_KEY = os.getenv(
    "GHOST_CONTENT_API_KEY"
)  # Replace with your Content API key
DOMAIN = "https://oksayless.com"
GHOST_URL = "https://always-be-optimizing.ghost.io"  # Update with your Ghost URL
API_VERSION = "v5.0"


def fetch_posts():
    url = f"{GHOST_URL}/ghost/api/content/posts/"
    params = {"key": GHOST_CONTENT_API_KEY, "fields": "slug,updated_at", "limit": "all"}

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["posts"]
    else:
        raise Exception(f"Failed to fetch posts: {response.status_code}")


def generate_sitemap():
    # Create the root element
    urlset = ET.Element("urlset")
    urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    # Add homepage
    home_url = ET.SubElement(urlset, "url")
    ET.SubElement(home_url, "loc").text = DOMAIN
    ET.SubElement(home_url, "changefreq").text = "monthly"
    ET.SubElement(home_url, "priority").text = "1.0"

    # Add blog index
    blog_url = ET.SubElement(urlset, "url")
    ET.SubElement(blog_url, "loc").text = f"{DOMAIN}/blog"
    ET.SubElement(blog_url, "changefreq").text = "daily"
    ET.SubElement(blog_url, "priority").text = "0.8"

    # Add all blog posts
    posts = fetch_posts()
    for post in posts:
        url = ET.SubElement(urlset, "url")
        ET.SubElement(url, "loc").text = f"{DOMAIN}/blog/{post['slug']}"
        ET.SubElement(url, "lastmod").text = post["updated_at"][
            :10
        ]  # YYYY-MM-DD format
        ET.SubElement(url, "changefreq").text = "never"
        ET.SubElement(url, "priority").text = "0.6"

    # Create the XML string with pretty printing
    xmlstr = minidom.parseString(ET.tostring(urlset)).toprettyxml(indent="   ")

    # Write to file
    with open("frontend/sitemap.xml", "w", encoding="utf-8") as f:
        f.write(xmlstr)


if __name__ == "__main__":
    generate_sitemap()


================================================================================
FILE: ./frontend/tailwind.config.js
================================================================================
module.exports = {
    content: [
      './index.html',
      './src/**/*.{js,ts,jsx,tsx}',
    ],
    darkMode: 'class',
    theme: {
      extend: {}
    },
    plugins: []
  }

================================================================================
FILE: ./frontend/vite.config.js
================================================================================
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html')
      },
    }
  },
  optimizeDeps: {
    esbuildOptions: {
      define: {
        global: 'globalThis'
      }
    },
    include: ['@capacitor/core', '@revenuecat/purchases-capacitor']
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src')
    }
  },
  server: {
    port: 3000
  }
});

================================================================================
FILE: ./frontend/postcss.config.js
================================================================================
export default {
    plugins: {
      tailwindcss: {},
      autoprefixer: {},
    }
}

================================================================================
FILE: ./frontend/src/decision-helper.js
================================================================================
import { marked } from 'marked';

// Configure marked to handle line breaks properly
marked.setOptions({
  breaks: true,  // Convert line breaks to <br>
  gfm: true,     // Enable GitHub Flavored Markdown
});

export function decisionHelper() {
    return {
        situation: '',
        isLoading: false,
        refreshingQuestions: false,
        results: null,
        selectedExample: '',
        aiChoice: null,
        selectedOptionIndex: null,
        currentView: 'input', // Can be 'input' or 'conversation'
        error: '',
        showToast: false,
        toastMessage: '',
        toastIcon: 'info', // Add this new property to track the toast icon type
        showUndoButton: false,
        _toastTimeout: null, // For tracking the toast timeout
        darkMode: localStorage.getItem('darkMode') === 'true' || false,
        messages: [], // Array to store conversation messages
        userMessage: '', // For the input box at the bottom
        followUpQuestion: '', // Current follow-up question being displayed
        latestChoicesIndex: -1, // Track the index of the most recent choices message
        conversationHistory: [], // Array to store conversation history
        showHistoryModal: false, // Controls visibility of history modal
        currentConversationId: null, // ID of the current conversation
        // Removed todo discussions filtering
        currentChoices: [], // Track the currently visible choices (for removal feature)
        activeButtonLoading: null, // Track which button is currently loading
        isProcessingAction: false, // Controls visibility of CTAs during action processing
        sharingDecision: false, // Track when sharing is in progress
        
        baseUrl: (() => {
            if (window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost') {
                return 'http://localhost:8000';
            } else {
                return 'https://prompt-pm--values-fastapi-app.modal.run';
            }
        })(),
        
        examples: {
            newJob: {
                situation: "I'm considering a new role at work."
            },
            moveCity: {
                situation: "I'm thinking about a new place to live."
            },
            vacation: {
                situation: "I'm thinking about a new vacation spot."
            },
            school: {
                situation: "I'm considering going back to school."
            },
            date: {
                situation: "I'm considering going on a new date."
            }
        },

        init() {
            // Check for dark mode preference
            const savedDarkMode = localStorage.getItem('darkMode');
            
            if (savedDarkMode === null) {
                // If no preference is saved, check system preference
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                this.darkMode = prefersDark;
                localStorage.setItem('darkMode', prefersDark);
            } else {
                this.darkMode = savedDarkMode === 'true';
            }
            
            // Apply dark mode class if needed
            if (this.darkMode) {
                document.documentElement.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
            }

            // Load conversation history from localStorage
            this.loadConversationHistory();

            // Restore current view state from localStorage
            const savedView = localStorage.getItem('currentView');
            if (savedView) {
                // Only restore 'conversation' view if there are messages
                if (savedView === 'conversation' && this.messages.length > 0) {
                    this.currentView = 'conversation';
                }
            }

            // Listen for system dark mode changes
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
                if (localStorage.getItem('darkMode') === null) {
                    this.darkMode = e.matches;
                    localStorage.setItem('darkMode', e.matches);
                    
                    if (e.matches) {
                        document.documentElement.classList.add('dark');
                    } else {
                        document.documentElement.classList.remove('dark');
                    }
                }
            });

            // Set up visualViewport listener for keyboard events
            if (window.visualViewport) {
                window.visualViewport.addEventListener('resize', () => {
                    // If we're in the conversation view and the keyboard is open
                    if (this.currentView === 'conversation') {
                        const viewportHeight = window.innerHeight;
                        const keyboardHeight = viewportHeight - window.visualViewport.height;
                        
                        // If keyboard is opening (height > 50px)
                        if (keyboardHeight > 50) {
                            // Add a small delay to let the keyboard fully open
                            setTimeout(() => {
                                // Find the input element
                                const inputElement = document.getElementById('message-input');
                                if (inputElement) {
                                    // Scroll to make sure the input is visible
                                    inputElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                }
                            }, 300);
                        }
                    }
                });
            }

            // Focus the situation input
            this.$nextTick(() => {
                const textarea = document.getElementById('situation');
                if (textarea) {
                    textarea.focus();
                }
            });
        },

        loadConversationHistory() {
            try {
                const savedHistory = localStorage.getItem('conversationHistory');
                if (savedHistory) {
                    this.conversationHistory = JSON.parse(savedHistory);
                    // Sort by timestamp, most recent first
                    this.conversationHistory.sort((a, b) => b.timestamp - a.timestamp);
                }
            } catch (error) {
                console.error('Error loading conversation history:', error);
                this.conversationHistory = [];
            }
        },

        saveConversationHistory() {
            try {
                localStorage.setItem('conversationHistory', JSON.stringify(this.conversationHistory));
            } catch (error) {
                console.error('Error saving conversation history:', error);
                this.showToast = true;
                this.toastMessage = 'Error saving conversation history';
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
            }
        },

        saveCurrentConversation() {
            // Don't save empty conversations
            if (!this.messages.length) return;
            
            // Generate a title from the first user message
            const firstUserMessage = this.messages.find(m => m.role === 'user');
            const title = firstUserMessage ? 
                (firstUserMessage.content.length > 50 ? 
                    firstUserMessage.content.substring(0, 50) + '...' : 
                    firstUserMessage.content) : 
                'Untitled Decision';
            
            // Create a conversation object
            const conversation = {
                id: this.currentConversationId || this.generateId(),
                title: title,
                messages: this.messages,
                situation: this.situation,
                timestamp: Date.now(),
                latestChoicesIndex: this.latestChoicesIndex,
                selectedOptionIndex: this.selectedOptionIndex,
                results: this.results
            };
            
            // Set the current conversation ID
            this.currentConversationId = conversation.id;
            
            // Check if this conversation already exists in history
            const existingIndex = this.conversationHistory.findIndex(c => c.id === conversation.id);
            
            if (existingIndex !== -1) {
                // Update existing conversation
                this.conversationHistory[existingIndex] = conversation;
            } else {
                // Add new conversation to history
                this.conversationHistory.unshift(conversation);
            }
            
            // Save to localStorage
            this.saveConversationHistory();
        },
        
        // Todo related functions removed

        loadConversation(id) {
            const conversation = this.conversationHistory.find(c => c.id === id);
            if (!conversation) return;
            
            // Load conversation data
            this.messages = JSON.parse(JSON.stringify(conversation.messages));
            this.results = conversation.results ? JSON.parse(JSON.stringify(conversation.results)) : null;
            this.selectedOptionIndex = conversation.selectedOptionIndex;
            this.currentConversationId = conversation.id;
            this.currentView = 'conversation';
            localStorage.setItem('currentView', 'conversation');
            
            // Find the latest choices message in the loaded conversation
            this.latestChoicesIndex = -1;
            for (let i = this.messages.length - 1; i >= 0; i--) {
                if (this.messages[i].type === 'choices') {
                    this.latestChoicesIndex = i;
                    this.results = this.messages[i].content;
                    break;
                }
            }
            
            // Initialize the current choices from results
            this.initializeCurrentChoices();
            
            // Format markdown in assistant messages
            this.messages.forEach(message => {
                if (message.role === 'assistant' && !message.type && typeof message.content === 'string') {
                    message.content = marked.parse(message.content);
                }
            });
            
            // Close the history modal
            this.showHistoryModal = false;
            
            // Scroll to bottom
            this.$nextTick(() => {
                this.scrollToMessage();
            });
        },

        deleteConversation(id) {
            // Remove the conversation from history
            this.conversationHistory = this.conversationHistory.filter(c => c.id !== id);
            
            // Save updated history
            this.saveConversationHistory();
            
            // If we deleted the current conversation, reset the form
            if (this.currentConversationId === id) {
                this.resetForm();
            }
        },

        startNewDecision() {
            // Save current conversation before starting a new one
            this.saveCurrentConversation();
            
            // Reset the form
            this.resetForm();
        },

        openHistoryModal() {
            // Save current conversation before opening history
            this.saveCurrentConversation();
            
            // Show the history modal
            this.showHistoryModal = true;
        },

        generateId() {
            return Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
        },

        formatDate(timestamp) {
            const date = new Date(timestamp);
            const now = new Date();
            const isToday = date.toDateString() === now.toDateString();
            
            // Create a new date for yesterday without modifying 'now'
            const yesterday = new Date(now);
            yesterday.setDate(yesterday.getDate() - 1);
            const isYesterday = yesterday.toDateString() === date.toDateString();
            
            // Format time as "1:30 PM"
            const timeString = date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
            
            if (isToday) {
                return `Today ${timeString}`;
            } else if (isYesterday) {
                return `Yesterday ${timeString}`;
            } else {
                // Format date as "Feb 25"
                const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                const month = monthNames[date.getMonth()];
                const day = date.getDate();
                return `${month} ${day} ${timeString}`;
            }
        },

        toggleDarkMode() {
            this.darkMode = !this.darkMode;
            localStorage.setItem('darkMode', this.darkMode);
            
            if (this.darkMode) {
                document.documentElement.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
            }
        },

        fillExampleData() {
            if (this.selectedExample in this.examples) {
                this.situation = this.examples[this.selectedExample].situation;
                // Add this timeout to allow the model to update before adjusting height
                setTimeout(() => {
                    const textarea = document.getElementById('situation');
                    if (textarea) {
                        textarea.style.height = '';
                        textarea.style.height = Math.max(textarea.scrollHeight, 56) + 'px';
                    }
                }, 0);
            } else {
                this.situation = '';
            }
        },

        resetForm() {
            this.situation = '';
            this.currentView = 'input';
            localStorage.setItem('currentView', 'input');
            this.results = null;
            this.aiChoice = null;
            this.selectedOptionIndex = null;
            this.messages = [];
            this.userMessage = '';
            this.followUpQuestion = '';
            this.currentConversationId = null; // Clear current conversation ID
            
            // Reset textarea height
            const textarea = document.getElementById('situation');
            if (textarea) {
                textarea.style.height = '56px'; // Reset to minimum height
            }
        },

        /**
         * Improved scroll function that scrolls to show the latest message while keeping context
         * @param {number} [offset=100] - Optional offset from the bottom of the message
         */
        scrollToMessage(offset = 100) {
            this.$nextTick(() => {
                // Find the message container
                const messageContainer = document.querySelector('.overflow-y-auto');
                if (!messageContainer) return;
                
                // Find the last message element
                const messages = messageContainer.querySelectorAll('.animate__fadeIn');
                if (messages.length === 0) return;
                
                const lastMessage = messages[messages.length - 1];
                
                // Calculate position to scroll to
                const rect = lastMessage.getBoundingClientRect();
                
                // Get viewport height
                const viewportHeight = window.innerHeight;
                
                // Calculate how much of the message is visible
                const visibleHeight = Math.min(
                    rect.bottom,
                    viewportHeight
                ) - Math.max(rect.top, 0);
                
                // If less than 70% of the message is visible, scroll to make it fully visible
                if (visibleHeight < rect.height * 0.7) {
                    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                    
                    // Calculate a position that ensures the message is visible with some context above
                    // We want the top of the message to be 'offset' pixels from the top of the viewport
                    const targetPosition = scrollTop + rect.top - offset;
                    
                    // Get keyboard height if visualViewport API is available
                    let keyboardHeight = 0;
                    if (window.visualViewport) {
                        keyboardHeight = viewportHeight - window.visualViewport.height;
                    }
                    
                    // Scroll with smooth behavior
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                    
                    // If keyboard is open, add extra scroll after a delay
                    if (keyboardHeight > 50) {
                        setTimeout(() => {
                            window.scrollTo({
                                top: targetPosition + keyboardHeight,
                                behavior: 'smooth'
                            });
                        }, 300);
                    }
                }
            });
        },

        sendMessage(isInitialSubmission = false) {
            // For initial submission, use situation; for follow-ups, use userMessage
            const messageContent = isInitialSubmission ? this.situation : this.userMessage;
            
            if (!messageContent.trim() || this.isLoading) return;
            
            // Add user message to conversation
            this.messages.push({
                role: 'user',
                content: messageContent,
                isEditing: false,
                editedContent: messageContent
            });
            
            // Clear input field for follow-up messages
            if (!isInitialSubmission) {
                this.userMessage = '';
            } else {
                // Initial submission specific actions
                this.results = null;
                this.aiChoice = null;
                this.currentView = 'conversation'; // Change to conversation view
                localStorage.setItem('currentView', 'conversation');
                
                // Trigger Google Ads conversion tracking for initial submissions
                if (typeof gtag_report_conversion === 'function') {
                    gtag_report_conversion();
                }
            }
            
            this.isLoading = true;
            this.error = '';

            fetch(`${this.baseUrl}/api/query/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: this.messages
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const responseData = data.response;
                const prompt = data.prompt;
                
                if (responseData && prompt === "generate_outcomes" && responseData.choices && Array.isArray(responseData.choices)) {
                    
                    this.results = {
                        title: responseData.title || "Your options",
                        choices: responseData.choices,
                        uncertainties: responseData.uncertainties || [],
                        next_steps: responseData.next_steps || []
                    };
                    
                    // Add the choices as a special message type in the conversation
                    this.messages.push({
                        role: 'assistant',
                        content: this.results,
                        type: 'choices',
                        choices: this.results
                    });
                    
                    // Update the latest choices index
                    this.latestChoicesIndex = this.messages.length - 1;
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                } 
                else if (responseData && ["normal", "generate_feedback", "web_search", "decision_chat"].includes(prompt) && responseData.text) {
                    let formattedText = responseData.text;
                    
                    // Parse markdown to HTML
                    formattedText = marked.parse(formattedText);
                    
                    // Handle citations if they exist
                    if (responseData.citations) {
                        responseData.citations.forEach((citation, index) => {
                            const refNumber = index + 1;
                            const refTag = `[${refNumber}]`;
                            formattedText = formattedText.replaceAll(
                                refTag,
                                `<a href="${citation}" target="_blank" class="text-sky-500 hover:text-sky-700">[${refNumber}]</a>`
                            );
                        });
                    }
                    
                    // Add AI response to conversation with suggested messages if they exist
                    this.messages.push({
                        role: 'assistant',
                        content: formattedText,
                        suggested_messages: responseData.suggested_messages || []
                    });
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                }
                // Handle any other type of response
                else {
                    // Add error message to conversation
                    this.messages.push({
                        role: 'assistant',
                        content: "Sorry, I couldn't process your request due to an error in our system.",
                        suggested_messages: []
                    });
                }
            })
            .catch(error => {
                console.error('Error details:', error);
                this.error = 'We couldn\'t submit your message, please try again.';
                this.showToast = true;
                this.toastMessage = this.error;
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
                

            })
            .finally(() => {
                this.isLoading = false;
                
                // Save the conversation to history
                this.saveCurrentConversation();
                
                // Focus on the message input at the bottom
                this.$nextTick(() => {
                    const messageInput = document.getElementById('message-input');
                    if (messageInput) {
                        messageInput.focus();
                    }
                });
            });
        },

        // Alias methods for backward compatibility and clarity
        submitChoices() {
            this.sendMessage(true);
        },

        sendFollowUpMessage() {
            if (!this.userMessage.trim() || this.isLoading) return;
            
            // Add user message to conversation
            this.messages.push({
                role: 'user',
                content: this.userMessage,
                isEditing: false,
                editedContent: this.userMessage
            });
            
            // Clear input field
            this.userMessage = '';
            
            // Scroll to show the user's message immediately
            this.scrollToMessage(150);
            
            // Then proceed with the API call
            this.isLoading = true;
            this.error = '';

            fetch(`${this.baseUrl}/api/query/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: this.messages
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const responseData = data.response;
                const prompt = data.prompt;
                
                if (responseData && prompt === "generate_outcomes" && responseData.choices && Array.isArray(responseData.choices)) {
                    
                    this.results = {
                        title: responseData.title || "Your options",
                        choices: responseData.choices,
                        uncertainties: responseData.uncertainties || [],
                        next_steps: responseData.next_steps || []
                    };
                    
                    // Add the choices as a special message type in the conversation
                    this.messages.push({
                        role: 'assistant',
                        content: this.results,
                        type: 'choices',
                        choices: this.results
                    });
                    
                    // Update the latest choices index
                    this.latestChoicesIndex = this.messages.length - 1;
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                } 
                else if (responseData && ["normal", "generate_feedback", "web_search", "decision_chat"].includes(prompt) && responseData.text) {
                    let formattedText = responseData.text;
                    
                    // Parse markdown to HTML
                    formattedText = marked.parse(formattedText);
                    
                    // Handle citations if they exist
                    if (responseData.citations) {
                        responseData.citations.forEach((citation, index) => {
                            const refNumber = index + 1;
                            const refTag = `[${refNumber}]`;
                            formattedText = formattedText.replaceAll(
                                refTag,
                                `<a href="${citation}" target="_blank" class="text-sky-500 hover:text-sky-700">[${refNumber}]</a>`
                            );
                        });
                    }
                    
                    // Add AI response to conversation with suggested messages if they exist
                    this.messages.push({
                        role: 'assistant',
                        content: formattedText,
                        suggested_messages: responseData.suggested_messages || []
                    });
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                }
                // Handle any other type of response
                else {
                    // Add error message to conversation
                    this.messages.push({
                        role: 'assistant',
                        content: "Sorry, I couldn't process your request due to an error in our system.",
                        suggested_messages: []
                    });
                }
            })
            .catch(error => {
                console.error('Error details:', error);
                this.error = 'We couldn\'t submit your message, please try again.';
                this.showToast = true;
                this.toastMessage = this.error;
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
            })
            .finally(() => {
                this.isLoading = false;
                
                // Save the conversation to history
                this.saveCurrentConversation();
                
                // Focus on the message input at the bottom
                this.$nextTick(() => {
                    const messageInput = document.getElementById('message-input');
                    if (messageInput) {
                        messageInput.focus();
                    }
                });
            });
        },

        addAlternative() {
            if (!this.results || !this.results.choices || this.isLoading) return;
            
            this.isLoading = true;
            this.isProcessingAction = true; // Set flag to hide CTAs
            this.activeButtonLoading = 'add';
            this.error = '';

            fetch(`${this.baseUrl}/api/add_alternative/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    results: this.results
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.new_alternative) {
                    // Add the new alternative to the results object
                    this.results.choices.push(data.new_alternative);
                    
                    // Find and update the choices message in the messages array
                    if (this.latestChoicesIndex !== -1) {
                        // Create a new copy of the choices to ensure reactivity
                        const updatedChoices = {...this.messages[this.latestChoicesIndex].choices};
                        updatedChoices.choices = [...updatedChoices.choices, data.new_alternative];
                        this.messages[this.latestChoicesIndex].choices = updatedChoices;
                        
                        // Also add to currentChoices for the UI
                        this.currentChoices.push(data.new_alternative);
                    }
                    
                    // Show notification
                    this.showToast = true;
                    this.toastMessage = "Added a new option";
                    setTimeout(() => {
                        this.showToast = false;
                    }, 3000);
                    
                    // Save the conversation with the new option
                    this.saveCurrentConversation();
                } else {
                    this.error = 'Failed to generate a new alternative.';
                    this.showToast = true;
                    this.toastMessage = this.error;
                    this.toastIcon = 'error'; // Set the error icon
                    setTimeout(() => {
                        this.showToast = false;
                    }, 3000);
                }
            })
            .catch(error => {
                this.error = 'An error occurred while fetching data. Please try again.';
                this.showToast = true;
                this.toastMessage = this.error;
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
                console.error('Error:', error);
            })
            .finally(() => {
                this.isLoading = false;
                this.isProcessingAction = false; // Reset flag to show CTAs on the latest message
                this.activeButtonLoading = null;
            });
        },

        chooseForMe() {
            if (!this.results || !this.results.choices || this.isLoading) return;
            
            this.isLoading = true;
            this.isProcessingAction = true; // Set flag to hide CTAs
            this.activeButtonLoading = 'choose';
            this.aiChoice = null;
            
            fetch(`${this.baseUrl}/api/choose/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    situation: this.messages.length > 0 ? this.messages[0].content : this.situation,
                    results: this.results,
                    selectedIndex: this.selectedOptionIndex !== null ? this.selectedOptionIndex : undefined
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                this.aiChoice = data;
                this.selectedOptionIndex = data.chosen_index;
                
                // Update the choices in the results object
                if (this.results.choices[data.chosen_index]) {
                    this.results.choices[data.chosen_index].explanation = data.explanation;
                }
                
                // Update currentChoices array (if it exists)
                if (this.currentChoices && this.currentChoices.length > 0 && 
                    data.chosen_index < this.currentChoices.length) {
                    this.currentChoices[data.chosen_index].explanation = data.explanation;
                }
                
                // Update the choices message to show the selected option with explanation
                const choicesMessageIndex = this.latestChoicesIndex;
                if (choicesMessageIndex !== -1) {
                    // Create a new copy of the choices to ensure reactivity
                    const updatedChoices = {...this.messages[choicesMessageIndex].choices};
                    if (updatedChoices.choices[data.chosen_index]) {
                        updatedChoices.choices[data.chosen_index].explanation = data.explanation;
                        this.messages[choicesMessageIndex].choices = updatedChoices;
                    }
                }
                
                // Show a toast notification
                this.showToastMessage(`Selected option: ${this.results.choices[data.chosen_index].name}`);
                
                // Use improved scroll function
                this.scrollToMessage();
                
                // Save the conversation with the recommendation
                this.saveCurrentConversation();
            })
            .catch(error => {
                this.error = 'An error occurred while getting AI recommendation. Please try again.';
                this.showToast = true;
                this.toastMessage = this.error;
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
                console.error('Error:', error);
            })
            .finally(() => {
                this.isLoading = false;
                this.isProcessingAction = false; // Reset flag to show CTAs on the latest message
                this.activeButtonLoading = null;
            });
        },

        refreshQuestions(decisionId) {
            if (this.refreshingQuestions) return;
            
            // Set the refreshing questions state
            this.refreshingQuestions = true;
            
            fetch(`${this.baseUrl}/api/questions/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    results: this.results,
                    questions: this.results.uncertainties || []
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.uncertainties && data.uncertainties.length > 0) {
                    // Update the uncertainties in the results object
                    this.results.uncertainties = data.uncertainties;
                    
                    // Find and update the choices message in the messages array
                    if (this.latestChoicesIndex !== -1) {
                        const updatedContent = {...this.messages[this.latestChoicesIndex].content};
                        updatedContent.uncertainties = data.uncertainties;
                        this.messages[this.latestChoicesIndex].content = updatedContent;
                    }
                    
                    // Show notification
                    this.showToast = true;
                    this.toastMessage = `${data.uncertainties.length} new questions generated`;
                    setTimeout(() => {
                        this.showToast = false;
                    }, 3000);
                    
                    // Save the conversation with the new questions
                    this.saveCurrentConversation();
                } else {
                    this.showToast = true;
                    this.toastMessage = "Couldn't generate new questions";
                    this.toastIcon = 'error'; // Set the error icon
                    setTimeout(() => {
                        this.showToast = false;
                    }, 3000);
                }
            })
            .catch(error => {
                this.showToast = true;
                this.toastMessage = "Error refreshing questions";
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
                console.error('Error:', error);
            })
            .finally(() => {
                this.refreshingQuestions = false;
            });
        },

        generateNextSteps() {
            this.isLoading = true;
            this.isProcessingAction = true; // Set flag to hide CTAs
            this.activeButtonLoading = 'nextsteps';
            
            // Create payload for API request
            const payload = {
                situation: this.messages.length > 0 ? this.messages[0].content : this.situation,
                results: this.results || {}
            };
            
            // Check if we have a selected option - if so, add it to payload for post-decision steps
            // Note: We'll still work if no option is selected
            if (this.selectedOptionIndex !== null && this.results && this.results.choices) {
                const selectedChoice = this.results.choices[this.selectedOptionIndex];
                payload.choice_name = selectedChoice.name;
                payload.choice_index = this.selectedOptionIndex;
                if (this.results.id) {
                    payload.decision_id = this.results.id;
                }
            }
            
            fetch(`${this.baseUrl}/api/next_steps`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data.next_steps && data.next_steps.length > 0) {
                    // Check if there's already a next_steps message
                    const existingNextStepsIndex = this.messages.findIndex(msg => msg.type === 'next_steps');
                    
                    // Create title based on whether we're in pre-decision or post-decision mode
                    const title = this.selectedOptionIndex !== null && this.results && this.results.choices
                        ? `Next Steps for "${this.results.choices[this.selectedOptionIndex].name}"`
                        : "Next Steps";
                    
                    const newNextStepsMessage = {
                        role: 'assistant',
                        type: 'next_steps',
                        content: {
                            title: title,
                            steps: data.next_steps.map(step => ({
                                text: step
                            }))
                        }
                    };
                    
                    if (existingNextStepsIndex !== -1) {
                        // Replace the existing next_steps message
                        this.messages[existingNextStepsIndex] = newNextStepsMessage;
                    } else {
                        // Add a new message with the next steps
                        this.messages.push(newNextStepsMessage);
                    }
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                    
                    // Save the current conversation
                    this.saveCurrentConversation();
                } else {
                    this.showToastMessage('No action items generated', false, 'error');
                }
            })
            .catch(error => {
                console.error('Error generating next steps:', error);
                this.showToastMessage('Error generating next steps: ' + error.message, false, 'error');
            })
            .finally(() => {
                this.isLoading = false;
                this.isProcessingAction = false; // Reset flag to show CTAs on the latest message
                this.activeButtonLoading = null;
            });
        },
        
        suggestAdditionalAction(message) {
            if (!message || !message.content || !message.content.steps) {
                this.showToastMessage('No next steps found', false, 'error');
                return;
            }
            
            this.isLoading = true;
            
            // Get the situation from the first message or the situation field
            const situation = this.messages.length > 0 ? this.messages[0].content : this.situation;
            
            // Extract existing next steps as plain text array
            const existingNextSteps = message.content.steps.map(step => step.text);
            
            fetch(`${this.baseUrl}/api/suggest_additional_action`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    situation: situation,
                    existing_next_steps: existingNextSteps,
                    results: this.results || {}
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data.additional_action) {
                    // Add the new action to the steps array
                    message.content.steps.push({
                        text: data.additional_action
                    });
                    
                    // Save the conversation
                    this.saveCurrentConversation();
                    
                    // Scroll to show the updated list
                    this.scrollToMessage();
                    
                    // Show success message
                    this.showToastMessage('Added new action');
                } else {
                    this.showToastMessage('Could not generate a new action', false, 'error');
                }
            })
            .catch(error => {
                console.error('Error suggesting additional action:', error);
                this.showToastMessage('Error suggesting action: ' + error.message, false, 'error');
            })
            .finally(() => {
                this.isLoading = false;
            });
        },

        showToastMessage(message, showUndo = false, icon = 'info') {
            // Clear any existing timeout to prevent premature hiding
            if (this._toastTimeout) {
                clearTimeout(this._toastTimeout);
                this._toastTimeout = null;
            }
            
            this.toastMessage = message;
            this.showUndoButton = showUndo;
            this.toastIcon = icon; // Set the icon type
            this.showToast = true;
            
            this._toastTimeout = setTimeout(() => {
                this.showToast = false;
                this._toastTimeout = null;
            }, 5000);
        },
        
        switchToDecisions() {
            this.currentView = this.messages.length > 0 ? 'conversation' : 'input';
            localStorage.setItem('currentView', this.currentView);
        },
        
        resubmitFromMessage(index) {
            if (index < 0 || index >= this.messages.length || this.isLoading) return;
            
            // Keep messages up to and including the selected message
            const messagesToKeep = this.messages.slice(0, index + 1);
            const selectedMessage = this.messages[index];
            
            // Reset the conversation state
            this.messages = messagesToKeep;
            this.selectedOptionIndex = null;
            this.results = null;
            this.aiChoice = null;
            this.latestChoicesIndex = -1;
            
            // Find the latest choices message if it exists in the kept messages
            for (let i = messagesToKeep.length - 1; i >= 0; i--) {
                if (messagesToKeep[i].type === 'choices') {
                    this.latestChoicesIndex = i;
                    this.results = messagesToKeep[i].content;
                    break;
                }
            }
            
            // If the selected message is from the user, we need to send it again to get a new response
            if (selectedMessage.role === 'user') {
                // Remove the last message (user message) as we'll add it back when sending
                this.messages.pop();
                
                // Set the user message to the content of the selected message
                this.userMessage = selectedMessage.content;
                
                // Send the message
                this.$nextTick(() => {
                    this.sendFollowUpMessage();
                });
            }
            
            // Show a toast notification
            this.showToastMessage(selectedMessage.role === 'user' 
                ? 'Resubmitting message...' 
                : 'Conversation reset to this point');
            
            // Save the updated conversation
            this.saveCurrentConversation();
        },

        editMessage(index) {
            if (index < 0 || index >= this.messages.length || this.isLoading) return;
            
            const selectedMessage = this.messages[index];
            
            // Only allow editing user messages
            if (selectedMessage.role !== 'user') return;
            
            // Set the message as being edited
            selectedMessage.isEditing = true;
            selectedMessage.editedContent = selectedMessage.content;
            
            // Focus the editable content in the next tick
            this.$nextTick(() => {
                const editableElement = document.querySelector(`[data-message-index="${index}"] .editable-message`);
                if (editableElement) {
                    editableElement.focus();
                    // Place cursor at the end
                    const range = document.createRange();
                    const selection = window.getSelection();
                    range.selectNodeContents(editableElement);
                    range.collapse(false);
                    selection.removeAllRanges();
                    selection.addRange(range);
                }
            });
        },
        
        saveEditedMessage(index) {
            if (index < 0 || index >= this.messages.length) return;
            
            const selectedMessage = this.messages[index];
            
            // Only proceed if the message is being edited
            if (!selectedMessage.isEditing) return;
            
            // Get the edited content
            const editedContent = selectedMessage.editedContent.trim();
            
            // Only proceed if the content is not empty and has changed
            if (!editedContent || editedContent === selectedMessage.content) {
                selectedMessage.isEditing = false;
                return;
            }
            
            // Update the message content
            selectedMessage.content = editedContent;
            selectedMessage.isEditing = false;
            
            // Keep messages up to and including the edited message
            const messagesToKeep = this.messages.slice(0, index + 1);
            
            // Reset the conversation state
            this.messages = messagesToKeep;
            this.selectedOptionIndex = null;
            this.results = null;
            this.aiChoice = null;
            this.latestChoicesIndex = -1;
            
            // Find the latest choices message if it exists in the kept messages
            for (let i = messagesToKeep.length - 1; i >= 0; i--) {
                if (messagesToKeep[i].type === 'choices') {
                    this.latestChoicesIndex = i;
                    this.results = messagesToKeep[i].content;
                    break;
                }
            }
            
            // Send the edited message directly without adding a new user message
            this.isLoading = true;
            this.error = '';

            fetch(`${this.baseUrl}/api/query/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: this.messages
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const responseData = data.response;
                const prompt = data.prompt;
                
                if (responseData && prompt === "generate_outcomes" && responseData.choices && Array.isArray(responseData.choices)) {
                    
                    this.results = {
                        title: responseData.title || "Your options",
                        choices: responseData.choices,
                        uncertainties: responseData.uncertainties || [],
                        next_steps: responseData.next_steps || []
                    };
                    
                    // Add the choices as a special message type in the conversation
                    this.messages.push({
                        role: 'assistant',
                        content: this.results,
                        type: 'choices',
                        choices: this.results
                    });
                    
                    // Update the latest choices index
                    this.latestChoicesIndex = this.messages.length - 1;
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                } 
                else if (responseData && ["normal", "generate_feedback", "web_search", "decision_chat"].includes(prompt) && responseData.text) {
                    let formattedText = responseData.text;
                    
                    // Parse markdown to HTML
                    formattedText = marked.parse(formattedText);
                    
                    // Handle citations if they exist
                    if (responseData.citations) {
                        responseData.citations.forEach((citation, index) => {
                            const refNumber = index + 1;
                            const refTag = `[${refNumber}]`;
                            formattedText = formattedText.replaceAll(
                                refTag,
                                `<a href="${citation}" target="_blank" class="text-sky-500 hover:text-sky-700">[${refNumber}]</a>`
                            );
                        });
                    }
                    
                    // Add AI response to conversation with suggested messages if they exist
                    this.messages.push({
                        role: 'assistant',
                        content: formattedText,
                        suggested_messages: responseData.suggested_messages || []
                    });
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                }
                // Handle any other type of response
                else {
                    // Add error message to conversation
                    this.messages.push({
                        role: 'assistant',
                        content: "Sorry, I couldn't process your request due to an error in our system.",
                        suggested_messages: []
                    });
                }
            })
            .catch(error => {
                console.error('Error details:', error);
                this.error = 'We couldn\'t submit your message, please try again.';
                this.showToast = true;
                this.toastMessage = this.error;
                this.toastIcon = 'error'; // Set the error icon
                setTimeout(() => {
                    this.showToast = false;
                }, 3000);
            })
            .finally(() => {
                this.isLoading = false;
                
                // Save the conversation to history
                this.saveCurrentConversation();
                
                // Focus on the message input at the bottom
                this.$nextTick(() => {
                    const messageInput = document.getElementById('message-input');
                    if (messageInput) {
                        messageInput.focus();
                    }
                });
            });
        },
        
        cancelEditMessage(index) {
            if (index < 0 || index >= this.messages.length) return;
            
            const selectedMessage = this.messages[index];
            
            // Only proceed if the message is being edited
            if (!selectedMessage.isEditing) return;
            
            // Reset the editing state
            selectedMessage.isEditing = false;
            selectedMessage.editedContent = selectedMessage.content;
        },

        // Todo related functions removed
        
        // Generate options based on the entire conversation history
        generateChoices() {
            if (this.isLoading) return;
            
            this.isLoading = true;
            
            // Remove any existing choice messages from the chat history
            if (this.latestChoicesIndex !== -1) {
                this.messages = this.messages.filter(msg => msg.type !== 'choices');
                // Reset the latest choices index
                this.latestChoicesIndex = -1;
                // Reset selection
                this.selectedOptionIndex = null;
            }
            
            // Format messages for API
            const formattedMessages = this.messages.map(msg => ({
                role: msg.role,
                content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
            }));
            
            this.error = '';

            fetch(`${this.baseUrl}/api/choices/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message_history: formattedMessages
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.choices && Array.isArray(data.choices)) {
                    this.results = {
                        title: data.title || "Your options",
                        choices: data.choices,
                        uncertainties: data.uncertainties || [],
                        next_steps: data.next_steps || []
                    };
                    
                    // Add the choices as a special message type in the conversation
                    this.messages.push({
                        role: 'assistant',
                        content: this.results,
                        type: 'choices',
                        choices: this.results
                    });
                    
                    // Update the latest choices index
                    this.latestChoicesIndex = this.messages.length - 1;
                    
                    // Initialize the current choices from results
                    this.initializeCurrentChoices();
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                    
                    // Save the conversation history
                    this.saveCurrentConversation();
                } else {
                    // Handle error
                    this.showToastMessage('Failed to generate options', false, 'error');
                }
            })
            .catch(error => {
                console.error('Error generating options:', error);
                this.showToastMessage('Error generating options: ' + error.message, false, 'error');
            })
            .finally(() => {
                this.isLoading = false;
            });
        },
        
        // Initialize or update the currentChoices array from results
        initializeCurrentChoices() {
            if (this.results && this.results.choices) {
                // Create a deep copy of the choices array to avoid reference issues
                this.currentChoices = JSON.parse(JSON.stringify(this.results.choices));
            } else {
                this.currentChoices = [];
            }
        },
        
        // Remove a choice from the currentChoices array
        removeChoice(index) {
            if (index >= 0 && index < this.currentChoices.length) {
                // If the removed choice was selected, reset selection
                if (this.selectedOptionIndex === index) {
                    this.selectedOptionIndex = null;
                } 
                // If the removed choice was before the selected one, adjust selectedOptionIndex
                else if (this.selectedOptionIndex > index) {
                    this.selectedOptionIndex--;
                }
                
                // Remove the choice at the specified index
                this.currentChoices.splice(index, 1);
                
                // Show toast notification
                this.showToastMessage("Option removed from view");
            }
        },

        // Generate options from any point in the conversation
        generateOptionsFromMessage(messageIndex) {
            if (messageIndex < 0 || messageIndex >= this.messages.length || this.isLoading) return;
            
            // Set both flags to hide CTAs immediately
            this.isProcessingAction = true;
            this.isLoading = true;
            
            // Remove any existing choice messages from the chat history
            if (this.latestChoicesIndex !== -1) {
                this.messages = this.messages.filter(msg => msg.type !== 'choices');
                // Reset the latest choices index
                this.latestChoicesIndex = -1;
                // Reset selection
                this.selectedOptionIndex = null;
            }
            
            // Include conversation context up to this message
            const contextMessages = this.messages.slice(0, messageIndex + 1);
            
            // The backend expects a list of message objects, not a string
            // Convert to the format expected by the backend
            const formattedMessages = contextMessages.map(msg => ({
                role: msg.role,
                content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
            }));
            
            // If there are no messages, create a default one
            if (formattedMessages.length === 0) {
                formattedMessages.push({
                    role: 'user',
                    content: 'Help me make a decision'
                });
            }
            
            this.error = '';

            fetch(`${this.baseUrl}/api/choices/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message_history: formattedMessages  // Send as array of message objects
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.choices && Array.isArray(data.choices)) {
                    this.results = {
                        title: data.title || "Your options",
                        choices: data.choices,
                        uncertainties: data.uncertainties || [],
                        next_steps: data.next_steps || []
                    };
                    
                    // Add the choices as a special message type in the conversation
                    this.messages.push({
                        role: 'assistant',
                        content: this.results,
                        type: 'choices',
                        choices: this.results
                    });
                    
                    // Update the latest choices index
                    this.latestChoicesIndex = this.messages.length - 1;
                    
                    // Use improved scroll function
                    this.scrollToMessage();
                    
                    // Initialize the current choices from results
                    this.initializeCurrentChoices();
                    
                    // Save the conversation history
                    this.saveCurrentConversation();
                } else {
                    // Handle error
                    this.showToastMessage('Failed to generate options', false, 'error');
                }
            })
            .catch(error => {
                console.error('Error generating options:', error);
                this.showToastMessage('Error generating options: ' + error.message, false, 'error');
            })
            .finally(() => {
                this.isLoading = false;
                this.isProcessingAction = false; // Reset processing flag to allow CTAs on the new message
            });
        },

        // Generate next steps from a message context
        generateNextStepsFromMessage(messageIndex) {
            // Redirect to generateOptionsFromMessage to consolidate behavior
            this.generateOptionsFromMessage(messageIndex);
        },

        // Share the current decision 
        shareDecision() {
            if (this.messages.length === 0 || this.sharingDecision) return;
            
            this.sharingDecision = true;
            
            fetch(`${this.baseUrl}/api/save_decision/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message_history: this.messages
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.id) {
                    // Create shareable link using the ID
                    const shareLink = `${window.location.origin}/decision/${data.id}`;
                    
                    // Copy to clipboard
                    navigator.clipboard.writeText(shareLink).then(() => {
                        this.showToastMessage('Link copied to clipboard', false, 'info');
                    }, () => {
                        // Clipboard write failed, just show the link
                        this.showToastMessage('Share link created', false, 'info');
                    });

                    // Create a prompt to show the shareable link
                    const modal = document.createElement('div');
                    modal.className = 'fixed z-50 inset-0 overflow-y-auto flex items-center justify-center bg-black bg-opacity-50';
                    modal.innerHTML = `
                        <div class="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
                            <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-4">Share Your Decision</h3>
                            <p class="text-gray-600 dark:text-gray-300 mb-4">Share this link with others to show them your decision:</p>
                            <div class="flex items-center gap-2 mb-6">
                                <input 
                                    type="text" 
                                    value="${shareLink}" 
                                    class="w-full bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white p-2 rounded-md" 
                                    readonly
                                    onclick="this.select();"
                                />
                                <button 
                                    class="bg-sky-500 hover:bg-sky-600 text-white p-2 rounded-md"
                                    onclick="navigator.clipboard.writeText('${shareLink}').then(() => { 
                                        const copyBtn = document.getElementById('copy-btn');
                                        if (copyBtn) {
                                            copyBtn.innerHTML = '<i class=\\'fas fa-check\\'></i>';
                                            setTimeout(() => { copyBtn.innerHTML = '<i class=\\'fas fa-copy\\'></i>'; }, 2000);
                                        }
                                    })"
                                    id="copy-btn"
                                >
                                    <i class="fas fa-copy"></i>
                                </button>
                            </div>
                            <div class="flex justify-end">
                                <button 
                                    class="bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-white px-4 py-2 rounded-md"
                                    onclick="this.closest('.fixed').remove();"
                                >
                                    Close
                                </button>
                            </div>
                        </div>
                    `;
                    
                    document.body.appendChild(modal);
                    
                    // Add event listener to close on background click
                    modal.addEventListener('click', (e) => {
                        if (e.target === modal) {
                            modal.remove();
                        }
                    });
                    
                } else {
                    this.showToastMessage('Failed to create share link', false, 'error');
                }
            })
            .catch(error => {
                console.error('Error sharing decision:', error);
                this.showToastMessage('Error creating share link', false, 'error');
            })
            .finally(() => {
                this.sharingDecision = false;
            });
        },

        // Load a shared decision
        loadSharedDecision(decisionId) {
            if (!decisionId) return;
            
            this.isLoading = true;
            this.error = '';
            
            fetch(`${this.baseUrl}/api/get_decision/${decisionId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Decision not found (status: ${response.status})`);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.message_history && Array.isArray(data.message_history)) {
                        // Reset current state
                        this.resetForm();
                        
                        // Load the shared decision messages
                        this.messages = data.message_history;
                        
                        // Set to conversation view
                        this.currentView = 'conversation';
                        localStorage.setItem('currentView', 'conversation');
                        
                        // Find the latest choices message in the loaded conversation
                        this.latestChoicesIndex = -1;
                        for (let i = this.messages.length - 1; i >= 0; i--) {
                            if (this.messages[i].type === 'choices') {
                                this.latestChoicesIndex = i;
                                this.results = this.messages[i].content;
                                break;
                            }
                        }
                        
                        // Initialize the current choices from results
                        this.initializeCurrentChoices();
                        
                        // Format markdown in assistant messages
                        this.messages.forEach(message => {
                            if (message.role === 'assistant' && !message.type && typeof message.content === 'string') {
                                message.content = marked.parse(message.content);
                            }
                        });
                        
                        // Set title from first user message
                        const firstUserMessage = this.messages.find(m => m.role === 'user');
                        const title = firstUserMessage ? 
                            (firstUserMessage.content.length > 50 ? 
                                firstUserMessage.content.substring(0, 50) + '...' : 
                                firstUserMessage.content) : 
                            'Shared Decision';
                        
                        document.title = `Shared: ${title} | say less`;
                        
                        // Show notification
                        this.showToastMessage('Viewing a shared decision', false, 'info');
                    } else {
                        throw new Error('Invalid decision data');
                    }
                })
                .catch(error => {
                    console.error('Error loading shared decision:', error);
                    this.error = error.message || 'Failed to load shared decision';
                    this.showToastMessage(this.error, false, 'error');
                })
                .finally(() => {
                    this.isLoading = false;
                    
                    // Scroll to show the conversation
                    this.$nextTick(() => {
                        this.scrollToMessage();
                    });
                });
        }
    };
}

================================================================================
FILE: ./frontend/src/actions.js
================================================================================
// Actions/Todo list functionality
export function actionsHelper() {
    return {
        todos: [],
        completedTodos: [],
        newTodo: '',
        draggedTodoId: null,
        touchStartY: null,
        touchStartX: null,
        touchElement: null,
        longPressTimer: null,
        isDragging: false,
        currentTouchTarget: null,
        dragClone: null,
        touchOffsetX: 0,
        touchOffsetY: 0,
        originalPosition: null,
        placeholderElement: null,
        lastDeletedTodo: null,
        lastDeletedTodoWasCompleted: false,
        _toastTimeout: null, // For tracking the toast timeout

        init() {
            this.loadTodos();
        },

        loadTodos() {
            const savedTodos = localStorage.getItem('todos');
            const savedCompletedTodos = localStorage.getItem('completedTodos');
            
            if (savedTodos) {
                this.todos = JSON.parse(savedTodos);
            }
            
            if (savedCompletedTodos) {
                this.completedTodos = JSON.parse(savedCompletedTodos);
            }
        },
        
        saveTodos() {
            localStorage.setItem('todos', JSON.stringify(this.todos));
            localStorage.setItem('completedTodos', JSON.stringify(this.completedTodos));
        },
        
        addTodo() {
            if (this.newTodo.trim() === '') return;
            
            this.todos.unshift({
                id: this.generateId(),
                text: this.newTodo.trim(),
                createdAt: new Date().toISOString(),
                hasDiscussion: false,
                discussionId: null
            });
            
            this.newTodo = '';
            this.saveTodos();
            this.showToastMessage('Todo added');
        },
        
        // Drag and drop methods
        startDrag(event, todoId) {
            this.draggedTodoId = todoId;
            event.dataTransfer.effectAllowed = 'move';
            
            // Find the parent li element
            const listItem = event.target.closest('li');
            
            // Add a class to the list item for styling
            listItem.classList.add('dragging');
            
            // Set the drag image to be the list item
            event.dataTransfer.setDragImage(listItem, 20, 20);
        },
        
        endDrag(event) {
            this.draggedTodoId = null;
            
            // Find the parent li element
            const listItem = event.target.closest('li');
            
            // Remove the dragging class
            if (listItem) {
                listItem.classList.remove('dragging');
            }
        },
        
        onDragOver(event) {
            event.preventDefault();
            
            // Simplified - directly add the class without conditional check
            event.currentTarget.classList.add('drag-over');
            
            return false;
        },
        
        onDragLeave(event) {
            // Remove the drag-over class when leaving the element
            event.currentTarget.classList.remove('drag-over');
        },
        
        onDrop(event, targetTodoId) {
            event.preventDefault();
            
            // Remove the drag-over class
            event.currentTarget.classList.remove('drag-over');
            
            // Don't do anything if we're dropping onto the same item
            if (this.draggedTodoId === targetTodoId) {
                return;
            }
            
            // Find the indices of the dragged and target todos
            const draggedIndex = this.todos.findIndex(todo => todo.id === this.draggedTodoId);
            const targetIndex = this.todos.findIndex(todo => todo.id === targetTodoId);
            
            if (draggedIndex !== -1 && targetIndex !== -1) {
                // Remove the dragged item
                const draggedTodo = this.todos.splice(draggedIndex, 1)[0];
                
                // Insert it at the target position
                this.todos.splice(targetIndex, 0, draggedTodo);
                
                // Save the updated order
                this.saveTodos();
                this.showToastMessage('Todo order updated');
            }
        },
        
        completeTodo(todoId) {
            const todoIndex = this.todos.findIndex(todo => todo.id === todoId);
            
            if (todoIndex !== -1) {
                const completedTodo = this.todos.splice(todoIndex, 1)[0];
                
                // Clear any pending auto-save timer
                if (completedTodo._autoSaveTimer) {
                    clearTimeout(completedTodo._autoSaveTimer);
                    delete completedTodo._autoSaveTimer;
                }
                
                completedTodo.completedAt = new Date().toISOString();
                this.completedTodos.unshift(completedTodo);
                this.saveTodos();
                this.showToastMessage('Todo completed');
            }
        },
        
        deleteTodo(todoId, isCompleted = false) {
            let deletedTodo = null;
            
            if (isCompleted) {
                // Find the todo before removing it to clear any timers
                const todoIndex = this.completedTodos.findIndex(todo => todo.id === todoId);
                if (todoIndex !== -1) {
                    deletedTodo = { ...this.completedTodos[todoIndex] };
                    const todo = this.completedTodos[todoIndex];
                    if (todo._autoSaveTimer) {
                        clearTimeout(todo._autoSaveTimer);
                    }
                }
                
                this.completedTodos = this.completedTodos.filter(todo => todo.id !== todoId);
            } else {
                // Find the todo before removing it to clear any timers
                const todoIndex = this.todos.findIndex(todo => todo.id === todoId);
                if (todoIndex !== -1) {
                    deletedTodo = { ...this.todos[todoIndex] };
                    const todo = this.todos[todoIndex];
                    if (todo._autoSaveTimer) {
                        clearTimeout(todo._autoSaveTimer);
                    }
                }
                
                this.todos = this.todos.filter(todo => todo.id !== todoId);
            }
            
            // Save the deleted todo for potential undo
            if (deletedTodo) {
                this.lastDeletedTodo = deletedTodo;
                this.lastDeletedTodoWasCompleted = isCompleted;
            }
            
            this.saveTodos();
            this.showToastMessage('Todo deleted', true);
        },
        
        undoLastAction() {
            if (this.lastDeletedTodo) {
                // Restore the deleted todo
                if (this.lastDeletedTodoWasCompleted) {
                    this.completedTodos.push(this.lastDeletedTodo);
                } else {
                    this.todos.push(this.lastDeletedTodo);
                }
                
                // Clear the last deleted todo
                this.lastDeletedTodo = null;
                this.lastDeletedTodoWasCompleted = false;
                
                this.saveTodos();
                this.showToastMessage('Todo restored', false);
            }
        },
        
        restoreTodo(todoId) {
            const todoIndex = this.completedTodos.findIndex(todo => todo.id === todoId);
            
            if (todoIndex !== -1) {
                const restoredTodo = this.completedTodos.splice(todoIndex, 1)[0];
                
                // Clear any pending auto-save timer
                if (restoredTodo._autoSaveTimer) {
                    clearTimeout(restoredTodo._autoSaveTimer);
                    delete restoredTodo._autoSaveTimer;
                }
                
                delete restoredTodo.completedAt;
                this.todos.push(restoredTodo);
                this.saveTodos();
                this.showToastMessage('Todo restored');
            }
        },

        // Helper methods that need to be shared with the main app
        generateId() {
            return Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
        },
        
        showToastMessage(message, showUndo = false) {
            if (this.$parent && typeof this.$parent.showToastMessage === 'function') {
                this.$parent.showToastMessage(message, showUndo);
            } else {
                // Clear any existing timeout to prevent premature hiding
                if (this._toastTimeout) {
                    clearTimeout(this._toastTimeout);
                    this._toastTimeout = null;
                }
                
                this.toastMessage = message;
                this.showUndoButton = showUndo;
                this.showToast = true;
                
                this._toastTimeout = setTimeout(() => {
                    this.showToast = false;
                    this._toastTimeout = null;
                }, 5000);
            }
        },

        startInlineEdit(todo) {
            // Store the original text in case we need to cancel
            todo._originalText = todo.text;
            
            // Clear any existing auto-save timer for this todo
            if (todo._autoSaveTimer) {
                clearTimeout(todo._autoSaveTimer);
                todo._autoSaveTimer = null;
            }
        },
        
        updateTodoText(event, todo) {
            // Update the todo text in real-time as the user types
            todo.text = event.target.textContent;
            
            // Clear any existing auto-save timer
            if (todo._autoSaveTimer) {
                clearTimeout(todo._autoSaveTimer);
            }
            
            // Set a new timer to auto-save after 2 seconds of inactivity
            todo._autoSaveTimer = setTimeout(() => {
                this.autoSaveTodo(event, todo);
            }, 2000);
        },
        
        autoSaveTodo(event, todo) {
            const newText = event.target.textContent.trim();
            
            // Don't save empty todos
            if (newText === '') {
                return;
            }
            
            // Only save if the text has actually changed
            if (newText !== todo._originalText) {
                todo.text = newText;
                delete todo._originalText;
                delete todo._autoSaveTimer;
                this.saveTodos();
                
                // Show a subtle indicator that the todo was saved
                this.showToastMessage('Todo saved');
            }
        },
        
        saveTodoText(event, todo) {
            // Clear any pending auto-save timer
            if (todo._autoSaveTimer) {
                clearTimeout(todo._autoSaveTimer);
                todo._autoSaveTimer = null;
            }
            
            const newText = event.target.textContent.trim();
            
            // Don't save empty todos
            if (newText === '') {
                event.target.textContent = todo._originalText || '';
                return;
            }
            
            todo.text = newText;
            delete todo._originalText;
            this.saveTodos();
            
            // Remove focus from the element
            event.target.blur();
        },
        
        cancelTodoEdit(event, todo) {
            // Clear any pending auto-save timer
            if (todo._autoSaveTimer) {
                clearTimeout(todo._autoSaveTimer);
                todo._autoSaveTimer = null;
            }
            
            // Restore the original text
            event.target.textContent = todo._originalText || todo.text;
            delete todo._originalText;
            
            // Remove focus from the element
            event.target.blur();
        },

        // Mobile touch handlers
        handleTouchStart(event, todoId) {
            // Store the initial touch position
            this.touchStartY = event.touches[0].clientY;
            this.touchStartX = event.touches[0].clientX;
            
            // Find the grip element and the parent li element
            this.touchElement = event.currentTarget.closest('li');
            this.currentTouchTarget = todoId;
            
            // Calculate the offset from the touch point to the element's top-left corner
            const rect = this.touchElement.getBoundingClientRect();
            this.touchOffsetX = this.touchStartX - rect.left;
            this.touchOffsetY = this.touchStartY - rect.top;
            
            // Store the original position for creating a placeholder later
            this.originalPosition = {
                width: rect.width,
                height: rect.height,
                left: rect.left,
                top: rect.top
            };
            
            // Set a timer for long press (500ms)
            this.longPressTimer = setTimeout(() => {
                this.isDragging = true;
                this.draggedTodoId = todoId;
                
                // Create a clone of the element to move with the finger
                this.createDragClone();
                
                // Create a placeholder in the original position
                this.touchElement.classList.add('drag-placeholder');
                this.placeholderElement = this.touchElement;
                
                // Vibrate if supported
                if (navigator.vibrate) {
                    navigator.vibrate(50);
                }
                
                // Show toast to indicate drag mode
                this.showToastMessage('Drag to reorder');
            }, 300); // Reduced from 500ms to 300ms for faster response
        },
        
        createDragClone() {
            // Create a clone of the element
            this.dragClone = this.touchElement.cloneNode(true);
            
            // Style the clone
            this.dragClone.classList.add('touch-dragging');
            this.dragClone.classList.remove('drag-placeholder');
            
            // Position the clone at the original position
            this.dragClone.style.position = 'fixed';
            this.dragClone.style.left = `${this.originalPosition.left}px`;
            this.dragClone.style.top = `${this.originalPosition.top}px`;
            this.dragClone.style.width = `${this.originalPosition.width}px`;
            this.dragClone.style.margin = '0';
            this.dragClone.style.zIndex = '9999';
            
            // Add the clone to the document body
            document.body.appendChild(this.dragClone);
            
            // Move the clone to the current touch position
            this.updateDragClonePosition(this.touchStartX, this.touchStartY);
        },
        
        updateDragClonePosition(touchX, touchY) {
            if (!this.dragClone) return;
            
            // Calculate the new position based on the touch position and the initial offset
            const left = touchX - this.touchOffsetX;
            const top = touchY - this.touchOffsetY;
            
            // Update the clone's position
            this.dragClone.style.left = `${left}px`;
            this.dragClone.style.top = `${top}px`;
        },
        
        handleTouchMove(event) {
            if (!this.isDragging) {
                // If not in dragging mode yet, check if we should cancel the long press
                const touchY = event.touches[0].clientY;
                const touchX = event.touches[0].clientX;
                
                // If moved more than 10px in any direction before long press activated, cancel it
                if (Math.abs(touchY - this.touchStartY) > 10 || Math.abs(touchX - this.touchStartX) > 10) {
                    clearTimeout(this.longPressTimer);
                }
                return;
            }
            
            // Prevent default only when we're actually dragging
            event.preventDefault();
            event.stopPropagation();
            
            // Get the current touch position
            const touchX = event.touches[0].clientX;
            const touchY = event.touches[0].clientY;
            
            // Update the position of the drag clone
            this.updateDragClonePosition(touchX, touchY);
            
            // Get the element under the touch point (excluding the clone)
            const elementsUnderTouch = document.elementsFromPoint(touchX, touchY);
            const targetLi = elementsUnderTouch.find(el => 
                el.tagName === 'LI' && 
                !el.classList.contains('touch-dragging') && 
                !el.classList.contains('drag-placeholder')
            );
            
            if (targetLi) {
                // Get the todo ID from the element
                const targetTodoId = targetLi.getAttribute('data-id');
                
                if (targetTodoId && targetTodoId !== this.draggedTodoId) {
                    // Add visual feedback
                    document.querySelectorAll('li').forEach(li => {
                        if (li !== this.placeholderElement) {
                            li.classList.remove('drag-over');
                        }
                    });
                    targetLi.classList.add('drag-over');
                    
                    // Update the current touch target
                    this.currentTouchTarget = targetTodoId;
                }
            }
        },
        
        handleTouchEnd(event, todoId) {
            // Clear the long press timer
            clearTimeout(this.longPressTimer);
            
            // If we were dragging
            if (this.isDragging) {
                // Remove the drag clone
                if (this.dragClone) {
                    this.dragClone.remove();
                    this.dragClone = null;
                }
                
                // Remove the placeholder class
                if (this.placeholderElement) {
                    this.placeholderElement.classList.remove('drag-placeholder');
                    this.placeholderElement = null;
                }
                
                // Remove visual feedback from all items
                document.querySelectorAll('li').forEach(li => {
                    li.classList.remove('drag-over');
                });
                
                // If we have a valid drop target
                if (this.currentTouchTarget && this.currentTouchTarget !== this.draggedTodoId) {
                    this.onDrop(event, this.currentTouchTarget);
                }
                
                // Reset state
                this.isDragging = false;
            }
            
            this.draggedTodoId = null;
            this.touchStartY = null;
            this.touchStartX = null;
            this.touchElement = null;
            this.currentTouchTarget = null;
            this.touchOffsetX = 0;
            this.touchOffsetY = 0;
            this.originalPosition = null;
        }
    };
} 

================================================================================
FILE: ./frontend/src/main.js
================================================================================
import './styles.css';
import Alpine from 'alpinejs'
import { decisionHelper } from './decision-helper';
import { actionsHelper } from './actions';
import { marked } from 'marked';

// Browser history navigation functions
// Navigate to a specific path and update the browser's history
function navigateTo(path, state = {}) {
    // Make sure we're using the full path for consistent history management
    console.log('Navigating to:', path);
    const stateObj = { path, timestamp: Date.now(), ...state };
    history.pushState(stateObj, '', path);
}

// Initialize the page based on the URL
function initBrowserNavigation(app) {
    // Handle initial load
    window.addEventListener('load', () => {
        const path = window.location.pathname;
        
        // Set initial state without creating a new history entry
        history.replaceState({ path }, '', path);
        
        // Handle initial page path
        handlePathChange(path, app);
    });
    
    // Handle back/forward navigation
    window.addEventListener('popstate', (event) => {
        console.log('Navigation event:', event.state);
        if (event.state && event.state.path) {
            // We need to make sure we're always accessing the full path
            const fullPath = window.location.pathname;
            handlePathChange(fullPath, app);
        } else {
            // Default to home if no state
            handlePathChange('/', app);
        }
    });
}

// Handle path changes and update app state accordingly
function handlePathChange(path, app) {
    // Extract path segments
    const segments = path.split('/').filter(segment => segment);
    
    if (segments.length === 0 || segments[0] === '') {
        // Home path: "/"
        app._isHandlingHistoryNavigation = true;
        app.resetForm();
        setTimeout(() => {
            app._isHandlingHistoryNavigation = false;
        }, 0);
    /* Disabled actions view
    } else if (segments[0] === 'actions') {
        // Actions path: "/actions"
        app._isHandlingHistoryNavigation = true;
        app.switchToTodos();
        setTimeout(() => {
            app._isHandlingHistoryNavigation = false;
        }, 0);
    */
    } else if (segments[0] === 'conversation' && segments.length > 1) {
        // Conversation path: "/conversation/:id"
        const conversationId = segments[1];
        console.log('Loading conversation from URL:', conversationId);
        
        // Use the loadConversation method but make sure we don't create a new history entry
        const conversation = app.conversationHistory.find(c => c.id === conversationId);
        
        if (conversation) {
            // Set a flag to prevent duplicate history entries
            app._isHandlingHistoryNavigation = true;
            
            // Call the app's loadConversation method directly
            // This will handle loading all the necessary data
            app.loadConversation(conversationId);
            
            // Clear the flag
            setTimeout(() => {
                app._isHandlingHistoryNavigation = false;
            }, 0);
        } else {
            // Conversation not found, go to home
            app.currentView = 'input';
            localStorage.setItem('currentView', 'input');
            history.replaceState({ path: '/' }, '', '/');
        }
    } else if (segments[0] === 'decision' && segments.length > 1) {
        // Shared decision path: "/decision/:id"
        const decisionId = segments[1];
        console.log('Loading shared decision from URL:', decisionId);
        
        // Set a flag to prevent duplicate history entries
        app._isHandlingHistoryNavigation = true;
        
        // Load the shared decision from the server
        app.loadSharedDecision(decisionId);
        
        // Clear the flag
        setTimeout(() => {
            app._isHandlingHistoryNavigation = false;
        }, 0);
    } else if (segments[0] === 'history') {
        // History path: "/history"
        app._isHandlingHistoryNavigation = true;
        app.openHistoryModal();
        setTimeout(() => {
            app._isHandlingHistoryNavigation = false;
        }, 0);
    } else {
        // Default to home for unrecognized paths
        app._isHandlingHistoryNavigation = true;
        app.resetForm();
        setTimeout(() => {
            app._isHandlingHistoryNavigation = false;
        }, 0);
    }
}

// Combine the helpers into a single app
function appData() {
    const decisions = decisionHelper();
    const actions = actionsHelper();
    
    // Create a merged object with all properties and methods
    const merged = {
        ...decisions,
        ...actions,
        _isHandlingHistoryNavigation: false, // Flag to prevent history duplication
        
        // Override init to call both inits
        init() {
            decisions.init.call(this);
            actions.init.call(this);
            
            // Initialize browser navigation
            initBrowserNavigation(this);
        },
        
        // Override navigation methods to use browser history
        switchToTodos() {
            if (!this._isHandlingHistoryNavigation) {
                navigateTo('/actions');
            }
            decisions.switchToTodos.call(this);
        },
        
        switchToDecisions() {
            if (!this._isHandlingHistoryNavigation) {
                navigateTo('/' + (this.messages.length > 0 ? 'conversation/' + this.currentConversationId : ''));
            }
            decisions.switchToDecisions.call(this);
        },
        
        resetForm() {
            if (!this._isHandlingHistoryNavigation) {
                navigateTo('/');
            }
            decisions.resetForm.call(this);
        },
        
        openHistoryModal() {
            if (!this._isHandlingHistoryNavigation) {
                navigateTo('/history');
            }
            decisions.openHistoryModal.call(this);
        },
        
        loadConversation(id) {
            const conversation = this.conversationHistory.find(c => c.id === id);
            if (!conversation) return;
            
            // Check if this was triggered by our browser history handler
            if (!this._isHandlingHistoryNavigation) {
                // Only update history if this wasn't triggered by a popstate event
                navigateTo('/conversation/' + id, { conversationId: id });
            }
            
            // Always load the conversation content
            decisions.loadConversation.call(this, id);
        },
        
        startNewDecision() {
            if (!this._isHandlingHistoryNavigation) {
                navigateTo('/');
            }
            decisions.startNewDecision.call(this);
        },
        
        // Override showToastMessage to call the decision helper's method
        showToastMessage(message, showUndo = false) {
            decisions.showToastMessage.call(this, message, showUndo);
        },
        
        // Add undoLastAction to call the actions helper's method
        undoLastAction() {
            actions.undoLastAction.call(this);
        },
        
        // Make sure the decision helper can use the actions helper's methods
        saveTodos() {
            actions.saveTodos.call(this);
        }
    };
    
    return merged;
}

// Register the combined app with Alpine
Alpine.data('app', appData);

Alpine.start();
import 'htmx.org';
import '@fortawesome/fontawesome-free/css/all.min.css';
import 'animate.css';
import './tailwind.css';
import { CapacitorUpdater } from '@capgo/capacitor-updater'
CapacitorUpdater.notifyAppReady()


================================================================================
FILE: ./frontend/index.html
================================================================================
<!DOCTYPE html>
<html lang="en" class="dark:bg-gray-900">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, 
    viewport-fit=cover, user-scalable=no">
    <meta property="og:title" content="say less - choose with confidence">
    <meta property="og:description" content="say less - choose with confidence">
    <meta property="og:url" content="https://oksayless.com">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="say less - choose with confidence">
    <meta name="twitter:description" content="say less - choose with confidence">
    <title>say less</title>
    <link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Import local modules -->
    <script type="module" src="/src/main.js"></script>
    <script>
        !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey getNextSurveyStep identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property getSessionProperty createPersonProfile opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing debug".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);
        posthog.init('phc_A0E8UGmsyTtCZuNhafJ5ywDXMp4gSRHESjjr5xDyVDn', {
            api_host:'https://us.i.posthog.com',
            person_profiles: 'identified_only' // or 'always' to create profiles for anonymous users as well
        })
    </script>
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=AW-11420446469"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'AW-11420446469');
    </script>
    <!-- Event snippet for Sign-up conversion page -->
    <script>
        function gtag_report_conversion(url) {
            var callback = function () {
                if (typeof(url) != 'undefined') {
                    window.location = url;
                }
            };
            gtag('event', 'conversion', {
                'send_to': 'AW-11420446469/aKL7CPjJmOsZEIXe2MUq',
                'event_callback': callback
            });
            return false;
        }
    </script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        // Polyfill for document.elementsFromPoint if not supported
        if (!document.elementsFromPoint) {
            document.elementsFromPoint = function(x, y) {
                const elements = [];
                const element = document.elementFromPoint(x, y);
                
                if (element) {
                    elements.push(element);
                    
                    // Temporarily hide the element to get the one underneath
                    const originalVisibility = element.style.visibility;
                    element.style.visibility = 'hidden';
                    
                    // Recursively get elements underneath
                    const elementsUnderneath = document.elementsFromPoint(x, y);
                    
                    // Restore visibility
                    element.style.visibility = originalVisibility;
                    
                    elements.push(...elementsUnderneath);
                }
                
                return elements;
            };
        }
    </script>
    <script
        async
        crossorigin="anonymous"
        data-clerk-publishable-key="pk_test_aW50ZW5zZS10YWRwb2xlLTIyLmNsZXJrLmFjY291bnRzLmRldiQ"
        src="https://intense-tadpole-22.clerk.accounts.dev/npm/@clerk/clerk-js@latest/dist/clerk.browser.js"
        type="text/javascript"
    ></script>
    <style>
        /* Drag and drop styles */
        .dragging {
            opacity: 0.5;
            transform: scale(1.02);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            z-index: 10;
            transition: all 0.05s ease;
        }
        
        /* Mobile dragging style - no transition for smooth movement */
        .touch-dragging {
            position: absolute;
            opacity: 0.9;
            transform: scale(1.05);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
            z-index: 100;
            pointer-events: none;
            width: calc(100% - 2rem); /* Account for padding */
            max-width: 100%;
            transition: none; /* Disable transitions for direct movement */
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 0.5rem;
        }
        
        .dark .touch-dragging {
            background-color: rgba(31, 41, 55, 0.95);
        }
        
        /* Drag over indicator */
        .drag-over {
            border: 2px dashed #3b82f6 !important;
            background-color: rgba(59, 130, 246, 0.05) !important;
            transition: all 0.05s ease-out !important;
        }
        
        /* Placeholder for the dragged item's original position */
        .drag-placeholder {
            background-color: rgba(203, 213, 225, 0.3) !important;
            border: 2px dashed #cbd5e1 !important;
            box-shadow: none !important;
            height: 60px;
            opacity: 0.6 !important;
            transform: none !important;
            transition: all 0.05s ease-out !important;
        }
        
        /* Transition for smooth animations */
        .bg-white, .bg-gray-800 {
            transition: all 0.05s ease;
        }
        
        /* Faster transition for drag interactions */
        li[draggable="true"] {
            transition: all 0.05s ease;
        }
        
        /* Hide grip icon by default and show on hover */
        .todo-grip {
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        
        .group:hover .todo-grip {
            opacity: 1;
        }

        /* Mobile touch styles */
        @media (hover: none) and (pointer: coarse) {
            /* Always show grip on touch devices */
            .todo-grip {
                opacity: 0.7;
                min-width: 30px;
                min-height: 30px;
                padding: 5px;
                margin: -5px;
                background-color: rgba(0, 0, 0, 0.03);
                border-radius: 4px;
            }
            
            .dark .todo-grip {
                background-color: rgba(255, 255, 255, 0.05);
            }
            
            /* Enhance visual feedback during touch drag */
            .dragging {
                transform: scale(1.03);
                box-shadow: 0 15px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            }
            
            /* Larger touch targets for mobile */
            .todo-grip i {
                font-size: 1.2rem;
            }
            
            /* Always show edit button on mobile */
            .group button[title="Edit message"] {
                opacity: 0.7 !important;
                background-color: rgba(0, 0, 0, 0.05);
                border-radius: 50%;
                width: 36px;
                height: 36px;
                display: flex;
                align-items: center;
                justify-content: center;
                top: 8px;
                right: 8px;
            }
            
            .group button[title="Edit message"] i {
                font-size: 1.1rem;
            }
            
            .dark .group button[title="Edit message"] {
                background-color: rgba(255, 255, 255, 0.1);
            }
            
            /* Make editable area more touch-friendly */
            .editable-message {
                min-height: 60px;
                padding: 10px !important;
                font-size: 16px !important; /* Prevent zoom on iOS */
            }
        }

        /* Desktop styles for edit button */
        @media (hover: hover) {
            .group button[title="Edit message"] {
                opacity: 0 !important;
            }
            
            .group:hover button[title="Edit message"] {
                opacity: 0.7 !important;
            }
            
            /* Smooth transition for editable area */
            .editable-message {
                transition: all 0.2s ease;
            }
        }
    </style>
</head>
<body class="bg-white dark:bg-gray-900 min-h-screen flex flex-col" x-data="app" x-init="init">
    <!-- Add sign in button container -->
    <div class="fixed top-0 left-0 right-0 z-50 bg-white dark:bg-gray-900 p-2 sm:p-4 flex justify-between items-center">
        <!-- Logo -->
        <div class="flex items-center pl-2 sm:pl-4 cursor-pointer" @click="resetForm()">
            <h1 class="text-lg sm:text-xl font-bold text-gray-900 dark:text-white">say less</h1>
            <!-- Dark mode toggle moved next to logo -->
            <button 
                @click.stop="toggleDarkMode()" 
                class="flex items-center text-gray-700 dark:text-white hover:text-gray-900 dark:hover:text-gray-200 font-medium ml-2 sm:ml-3 text-sm sm:text-base"
                title="Toggle Dark Mode"
            >
                <i class="fas text-lg sm:text-xl" :class="darkMode ? 'fa-moon' : 'fa-sun'"></i>
            </button>
        </div>
        
        <!-- Sign in and user buttons -->
        <div class="flex items-center pr-2 sm:pr-4">
            <button 
                @click="startNewDecision()" 
                class="flex items-center text-gray-700 dark:text-white hover:text-gray-900 dark:hover:text-gray-200 font-medium mr-2 sm:mr-4 text-sm sm:text-base"
                title="New Decision"
                x-show="currentView === 'conversation'"
            >
                <i class="fas fa-plus text-lg sm:text-xl"></i>
                <span class="ml-1 sm:ml-2">new</span>
            </button>
            <button 
                @click="shareDecision()" 
                class="flex items-center text-gray-700 dark:text-white hover:text-gray-900 dark:hover:text-gray-200 font-medium mr-2 sm:mr-4 text-sm sm:text-base"
                title="Share Decision"
                x-show="currentView === 'conversation' && messages.length > 0"
            >
                <i class="fas fa-share-alt text-lg sm:text-xl"></i>
                <span class="ml-1 sm:ml-2">share</span>
            </button>
            <button 
                @click="openHistoryModal()" 
                class="flex items-center text-gray-700 dark:text-white hover:text-gray-900 dark:hover:text-gray-200 font-medium mr-2 sm:mr-4 text-sm sm:text-base"
                title="History"
            >
                <i class="fas fa-history text-lg sm:text-xl"></i>
                <span class="ml-1 sm:ml-2">history</span>
            </button>
            <!-- Actions button removed -->
            <!-- Decisions button removed -->
            
            <!--div id="sign-in-button"></div>
            <div id="user-button"></div-->
        </div>
    </div>

    <!-- Main content area -->
    <div class="flex-1 flex flex-col pt-12 sm:pt-16">
        <!-- Initial Input UI - Only visible before first submission -->
        <div x-show="currentView === 'input'" class="flex-1 flex items-center justify-center p-4">
            <div class="max-w-3xl w-full mx-auto bg-gray-100 dark:bg-gray-800 rounded-lg shadow-md p-6">
                <div class="mb-4 sm:mb-6">
                    <h1 class="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mb-2 sm:mb-0">get an instant second opinion</h1>
                </div>
                
                <ul class="list-none mb-4 text-gray-700 dark:text-gray-300 text-sm sm:text-base space-y-2 sm:space-y-0 sm:flex sm:space-x-4">
                    <li class="flex items-center">ü§î figure out what you want</li>
                    <li class="flex items-center">‚ö° decide faster</li>
                    <li class="flex items-center">‚úÖ take action</li>
                </ul>

                <div class="mb-4">
                    <div class="relative">
                        <textarea 
                            id="situation"
                            x-model="situation"
                            class="w-full bg-white dark:bg-gray-700 text-gray-900 dark:text-white p-4 pr-16 rounded-lg border-2 border-gray-300 dark:border-gray-600 focus:border-sky-500 outline-none resize-none overflow-hidden min-h-[56px] leading-[1.5] flex items-center"
                            placeholder="what's your situation?"
                            @keydown.enter.prevent="submitChoices"
                            rows="1"
                            style="padding-top: 14px; padding-bottom: 14px;"
                            x-init="
                                $el.style.height = ''; 
                                $el.style.height = Math.max($el.scrollHeight, 56) + 'px';
                                $el.focus();
                            "
                            @input="$nextTick(() => { $el.style.height = ''; $el.style.height = Math.max($el.scrollHeight, 56) + 'px' })"
                        ></textarea>
                        <div class="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-2 bg-white dark:bg-gray-700 pl-2">
                            <button
                                @click="submitChoices"
                                :disabled="!situation.trim() || isLoading"
                                class="bg-sky-500 hover:bg-sky-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-sky-500"
                            >
                                <!-- Arrow icon for all screens -->
                                <svg x-show="!isLoading" class="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                                </svg>
                                <!-- Loading indicator -->
                                <svg x-show="isLoading" class="animate-spin w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Examples section -->
                <div class="flex items-center justify-end mt-4">
                    <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 w-full">
                        <button 
                            @click="selectedExample = 'newJob'; fillExampleData(); submitChoices()"
                            class="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-left border border-gray-200 dark:border-gray-700"
                        >
                            <div class="flex items-center gap-2">
                                <span class="text-lg">üíº</span>
                                <span class="text-xs text-gray-600 dark:text-gray-400">new role at work</span>
                            </div>
                        </button>
                        <button 
                            @click="selectedExample = 'moveCity'; fillExampleData(); submitChoices()"
                            class="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-left border border-gray-200 dark:border-gray-700"
                        >
                            <div class="flex items-center gap-2">
                                <span class="text-lg">üè†</span>
                                <span class="text-xs text-gray-600 dark:text-gray-400">new place to live</span>
                            </div>
                        </button>
                        <button 
                            @click="selectedExample = 'vacation'; fillExampleData(); submitChoices()"
                            class="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-left border border-gray-200 dark:border-gray-700"
                        >
                            <div class="flex items-center gap-2">
                                <span class="text-lg">‚úàÔ∏è</span>
                                <span class="text-xs text-gray-600 dark:text-gray-400">new vacation spot</span>
                            </div>
                        </button>
                        <button 
                            @click="selectedExample = 'date'; fillExampleData(); submitChoices()"
                            class="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-left border border-gray-200 dark:border-gray-700"
                        >
                            <div class="flex items-center gap-2">
                                <span class="text-lg">‚ô•Ô∏è</span>
                                <span class="text-xs text-gray-600 dark:text-gray-400">new date</span>
                            </div>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Conversation View - Shows after first submission -->
        <div x-show="currentView === 'conversation'" class="flex-1 flex flex-col">
            <!-- Message display area -->
            <div class="flex-1 overflow-y-auto p-2 sm:p-4 pb-32 sm:pb-40">
                <!-- Messages in chronological order -->
                <template x-for="(message, index) in messages" :key="index">
                    <div class="mb-3 sm:mb-4 animate__animated animate__fadeIn">
                        <!-- User message -->
                        <div x-show="message.role === 'user'" class="max-w-3xl mx-auto bg-gray-200 dark:bg-gray-700 rounded-lg p-3 sm:p-4 text-gray-900 dark:text-white relative group" :data-message-index="index">
                            <div class="pr-6">
                                <!-- Regular message display (when not editing) -->
                                <p x-show="!message.isEditing" x-text="message.content"></p>
                                
                                <!-- Editable message (when editing) -->
                                <div 
                                    x-show="message.isEditing" 
                                    class="editable-message bg-white dark:bg-gray-600 p-2 rounded-md outline-none border border-sky-500 dark:border-sky-400" 
                                    contenteditable="true"
                                    @input="message.editedContent = $event.target.textContent"
                                    @keydown.enter.prevent="saveEditedMessage(index)"
                                    @keydown.escape.prevent="cancelEditMessage(index)"
                                    x-init="$el.textContent = message.content"
                                ></div>
                                
                                <!-- Edit controls (only shown when editing) -->
                                <div x-show="message.isEditing" class="flex justify-end mt-2 space-x-2">
                                    <button 
                                        @click="cancelEditMessage(index)" 
                                        class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-sm"
                                    >
                                        Cancel
                                    </button>
                                    <button 
                                        @click="saveEditedMessage(index)" 
                                        class="bg-sky-500 hover:bg-sky-600 text-white px-3 py-1 rounded-md text-sm"
                                    >
                                        Save
                                    </button>
                                </div>
                            </div>
                            
                            <!-- Edit icon (only shown when not editing) -->
                            <button 
                                x-show="!message.isEditing"
                                @click="editMessage(index)"
                                class="absolute right-2 top-2 text-gray-500 hover:text-sky-600 dark:text-gray-400 dark:hover:text-sky-400 transition-opacity flex items-center justify-center"
                                title="Edit message"
                            >
                                <i class="fas fa-edit text-sm"></i>
                            </button>
                        </div>
                        
                        <!-- AI message -->
                        <div x-show="message.role === 'assistant' && !message.type" class="max-w-3xl mx-auto text-gray-900 dark:text-white">
                            <div x-html="message.content" class="markdown-content"></div>
                            
                            <!-- Generate buttons removed - using single button at bottom -->
                            
                            <!-- Suggested messages - only shown for the last assistant message and only when no options are generated -->
                            <div x-show="message.suggested_messages && message.suggested_messages.length > 0 && index === (messages.findLastIndex(msg => msg.role === 'assistant' && !msg.type)) && latestChoicesIndex === -1" class="mt-4">
                                <div class="text-sm text-gray-500 dark:text-gray-400 mb-2">Suggested replies:</div>
                                <div class="flex flex-wrap gap-2">
                                    <template x-for="(suggestion, i) in message.suggested_messages" :key="i">
                                        <button 
                                            @click="userMessage = suggestion; isProcessingAction = true; $nextTick(() => { sendFollowUpMessage(); })"
                                            class="bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-white px-3 py-2 rounded-full text-sm transition-colors flex items-center"
                                        >
                                            <i class="fas fa-reply text-xs mr-2 text-gray-500 dark:text-gray-400"></i>
                                            <span x-text="suggestion"></span>
                                        </button>
                                    </template>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Choices UI as a special message type -->
                        <div x-show="message.role === 'assistant' && message.type === 'choices'" class="max-w-3xl mx-auto">
                            <!-- Options section -->
                            <div class="mb-6 relative" :class="{'opacity-70 pointer-events-none': index !== latestChoicesIndex}">
                                <!-- Inactive overlay for old decision results -->
                                <div x-show="index !== latestChoicesIndex" class="absolute inset-0 flex items-center justify-center z-10">
                                    <div class="bg-gray-200 dark:bg-gray-700 px-3 py-1 rounded-md text-sm text-gray-700 dark:text-gray-300">
                                        Previous decision
                                    </div>
                                </div>
                                
                                <div class="flex sm:flex-row sm:items-center mb-2">
                                    <h2 class="text-lg sm:text-xl font-medium text-gray-900 dark:text-white" x-text="message.content?.title || 'Your Options'"></h2>
                                </div>
                                
                                <!-- Key questions - moved below title -->
                                <div class="mb-4 text-gray-700 dark:text-gray-300 text-sm flex flex-wrap gap-1">
                                    <template x-for="(uncertainty, index) in message.content?.uncertainties || []" :key="index">
                                        <span x-text="uncertainty"></span>
                                    </template>
                                    <button 
                                        @click="refreshQuestions(message.content?.id)"
                                        :disabled="refreshingQuestions || index !== latestChoicesIndex"
                                        class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                                        title="Refresh questions"
                                        x-show="index === latestChoicesIndex"
                                    >
                                        <i class="fas fa-sync-alt text-xs" :class="{'animate-spin': refreshingQuestions}"></i>
                                    </button>
                                </div>
                                
                                <!-- Options cards -->
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    <template x-for="(choice, choiceIndex) in index === latestChoicesIndex ? currentChoices : (message.content?.choices || [])" :key="choiceIndex">
                                        <div 
                                            class="bg-stone-100 dark:bg-gray-800 rounded-lg p-3 relative"
                                            :class="{
                                                'cursor-pointer': index === latestChoicesIndex,
                                                'cursor-default': index !== latestChoicesIndex,
                                                'ring-2 ring-sky-500': index === latestChoicesIndex && choiceIndex === selectedOptionIndex
                                            }"
                                        >
                                            <!-- Remove button (x) - only visible for the latest choices view -->
                                            <button 
                                                x-show="index === latestChoicesIndex"
                                                @click.stop="removeChoice(choiceIndex)"
                                                class="absolute top-2 right-2 h-6 w-6 flex items-center justify-center text-red-500 hover:text-red-700 bg-white dark:bg-gray-700 rounded-full opacity-70 hover:opacity-100 transition-opacity z-10"
                                                title="Remove this option"
                                            >
                                                <i class="fas fa-times text-xs"></i>
                                            </button>
                                            
                                            <!-- Main content - clickable for selection -->
                                            <div 
                                                @click="index === latestChoicesIndex && (selectedOptionIndex = choiceIndex)"
                                                class="h-full"
                                            >
                                                <!-- Selected option indicator -->
                                                <span 
                                                    x-show="choiceIndex === selectedOptionIndex" 
                                                    class="inline-block mb-2 text-sm bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-300 px-2 py-1 rounded-md"
                                                >
                                                    <i class="fas fa-check-circle mr-1"></i> Selected option
                                                </span>
                                                <h3 class="text-base sm:text-lg font-semibold mb-1 text-gray-900 dark:text-white pr-6" x-text="choice?.name"></h3>
                                                <div class="space-y-1 text-xs sm:text-sm">
                                                    <div class="flex items-start justify-between">
                                                        <div class="flex items-start flex-1 mr-2">
                                                            <span class="text-green-500 mr-1 flex-shrink-0">‚úì</span>
                                                            <span class="text-gray-700 dark:text-gray-300" x-text="choice?.best_case_scenario"></span>
                                                        </div>
                                                        <div class="flex items-start flex-1">
                                                            <span class="text-red-500 mr-1 flex-shrink-0">‚úó</span>
                                                            <span class="text-gray-700 dark:text-gray-300" x-text="choice?.worst_case_scenario"></span>
                                                        </div>
                                                    </div>
                                                    <!-- Display explanation if it exists -->
                                                    <div x-show="choice?.explanation" class="mt-2 pt-2 border-t border-gray-300 dark:border-gray-600">
                                                        <div class="text-gray-700 dark:text-gray-300 text-xs sm:text-sm">
                                                            <p><strong>Why it could be good:</strong> <span x-text="choice?.explanation"></span></p>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </template>
                                </div>
                                
                                <!-- Choose for me button - only shown for the last choices message -->
                                <div class="mt-3" x-show="index === (messages.findLastIndex(msg => msg.role === 'assistant' && msg.type === 'choices'))">
                                    <button
                                        @click="isProcessingAction = true; chooseForMe(); activeButtonLoading = 'choose'"
                                        class="bg-sky-600 hover:bg-sky-700 text-white font-medium py-1.5 px-3 text-sm sm:text-base sm:py-2 sm:px-4 rounded inline-flex items-center"
                                    >
                                        <i class="fas fa-check-circle mr-1"></i> Choose for me
                                    </button>
                                    <button 
                                        @click="isProcessingAction = true; addAlternative(); activeButtonLoading = 'add'"
                                        class="ml-3 bg-sky-600 hover:bg-sky-700 text-white font-medium py-1.5 px-3 text-sm sm:text-base sm:py-2 sm:px-4 rounded inline-flex items-center"
                                    >
                                        <i class="fas fa-plus mr-1"></i> Add Option
                                    </button>
                                    <button 
                                        @click="isProcessingAction = true; generateNextSteps(); activeButtonLoading = 'nextsteps'"
                                        class="ml-3 bg-sky-600 hover:bg-sky-700 text-white font-medium py-1.5 px-3 text-sm sm:text-base sm:py-2 sm:px-4 rounded inline-flex items-center"
                                        title="Generate action items for your decision"
                                    >
                                        <i class="fas fa-tasks mr-1"></i> Next Steps
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Next Steps UI as a special message type -->
                        <div x-show="message.role === 'assistant' && message.type === 'next_steps'" class="max-w-3xl mx-auto bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700 animate__animated animate__fadeIn">
                            <div class="mb-3">
                                <h2 class="text-lg sm:text-xl font-medium text-gray-900 dark:text-white" x-text="message.content?.title || 'Next Steps'"></h2>
                            </div>
                            
                            <!-- Next steps list -->
                            <div class="space-y-2 mb-4">
                                <template x-for="(step, stepIndex) in message.content?.steps || []" :key="stepIndex">
                                    <div class="flex items-start justify-between bg-gray-50 dark:bg-gray-700 p-2 rounded">
                                        <div class="text-sm font-medium text-gray-900 dark:text-gray-300" x-text="step.text"></div>
                                        <button 
                                            @click="message.content.steps.splice(stepIndex, 1)" 
                                            class="text-gray-500 hover:text-red-500 dark:text-gray-400 dark:hover:text-red-400 ml-2"
                                            title="Remove"
                                        >
                                            <i class="fas fa-times"></i>
                                        </button>
                                    </div>
                                </template>
                            </div>
                            
                            <!-- Action buttons - only shown for the last next steps message -->
                            <div class="flex flex-wrap gap-2" x-show="index === (messages.findLastIndex(msg => msg.role === 'assistant' && msg.type === 'next_steps'))">
                                <button 
                                    @click="suggestAdditionalAction(message)"
                                    class="bg-sky-600 hover:bg-sky-700 text-white font-medium py-1.5 px-3 text-sm sm:text-base sm:py-2 sm:px-4 rounded"
                                >
                                    <i class="fas fa-plus-circle mr-1"></i> Suggest Action
                                </button>
                            </div>
                        </div>
                    </div>
                </template>
                
                <!-- Loading indicator -->
                <div x-show="isLoading" class="flex justify-center my-4">
                    <div class="animate-pulse flex space-x-2">
                        <div class="h-2 w-2 bg-gray-400 rounded-full"></div>
                        <div class="h-2 w-2 bg-gray-400 rounded-full"></div>
                        <div class="h-2 w-2 bg-gray-400 rounded-full"></div>
                    </div>
                </div>
            </div>
            
            <!-- Fixed bottom section with input -->
            <div class="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 p-2 sm:p-4">
                <div class="max-w-3xl mx-auto">
                    <!-- Generate Options Button -->
                    <div class="generate-options-container flex justify-center mb-2">
                        <button 
                            @click="generateChoices()" 
                            :disabled="isLoading"
                            class="bg-sky-600 hover:bg-sky-700 text-white font-bold py-2 px-4 rounded-full disabled:opacity-50 disabled:cursor-not-allowed text-sm sm:text-base"
                        >
                            <i class="fas fa-list-ul mr-1"></i> Generate Options
                        </button>
                    </div>
                    
                    <!-- Message Input -->
                    <div class="relative">
                        <input 
                            id="message-input"
                            x-model="userMessage"
                            class="w-full bg-white dark:bg-gray-700 text-gray-900 dark:text-white p-3 sm:p-4 pr-12 sm:pr-16 rounded-full border border-gray-300 dark:border-gray-600 focus:border-sky-500 outline-none text-sm sm:text-base"
                            placeholder="Type your message"
                            @keydown.enter.prevent="sendFollowUpMessage"
                            @focus="setTimeout(() => { window.scrollTo({ top: window.scrollY + 100, behavior: 'smooth' }); }, 300)"
                        />
                        <button
                            @click="sendFollowUpMessage"
                            :disabled="!userMessage.trim() || isLoading"
                            class="absolute right-1.5 sm:right-2 top-1/2 transform -translate-y-1/2 bg-sky-500 hover:bg-sky-700 text-white p-1.5 sm:p-2 rounded-full disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-sky-500"
                        >
                            <svg x-show="!isLoading" class="w-4 h-4 sm:w-5 sm:h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                            </svg>
                            <svg x-show="isLoading" class="animate-spin w-4 h-4 sm:w-5 sm:h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Todo List View removed -->
    </div>

    <!-- Toast Message -->
    <div 
        x-show="showToast" 
        x-transition:enter="transition ease-out duration-300"
        x-transition:enter-start="opacity-0 transform translate-y-2"
        x-transition:enter-end="opacity-100 transform translate-y-0"
        x-transition:leave="transition ease-in duration-200"
        x-transition:leave-start="opacity-100 transform translate-y-0"
        x-transition:leave-end="opacity-0 transform translate-y-2"
        class="fixed bottom-20 inset-x-0 pb-safe z-50 flex justify-center"
    >
        <div 
            class="bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white px-6 py-3 rounded-lg border border-gray-300 dark:border-gray-600 text-center max-w-sm mx-4 flex items-center justify-between shadow-lg cursor-pointer"
            :class="{'border-sky-500 dark:border-sky-400': showUndoButton}"
            @click="showUndoButton ? undoLastAction() : (showToast = false, _toastTimeout && clearTimeout(_toastTimeout) && (_toastTimeout = null))"
        >
            <div class="flex items-center">
                <template x-if="toastIcon === 'error'">
                    <span class="mr-2 text-red-500">ü•∫</span>
                </template>
                <template x-if="toastIcon === 'info'">
                    <i 
                        class="fas fa-info-circle mr-2 text-sky-600 dark:text-sky-400"
                        :class="{'fa-undo': showUndoButton}"
                    ></i>
                </template>
                <span x-text="toastMessage"></span>
            </div>
            <button 
                x-show="showUndoButton"
                @click.stop="undoLastAction()"
                class="ml-3 text-sky-600 dark:text-sky-400 font-medium text-sm hover:text-sky-800 dark:hover:text-sky-300 bg-sky-100 dark:bg-sky-900/30 px-3 py-1 rounded-full"
            >
                Undo
            </button>
        </div>
    </div>

    <!-- History Modal -->
    <div 
        x-show="showHistoryModal" 
        x-transition:enter="transition ease-out duration-300"
        x-transition:enter-start="opacity-0"
        x-transition:enter-end="opacity-100"
        x-transition:leave="transition ease-in duration-200"
        x-transition:leave-start="opacity-100"
        x-transition:leave-end="opacity-0"
        class="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        @click.self="showHistoryModal = false"
    >
        <div 
            class="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-lg w-full max-h-[80vh] flex flex-col"
            @click.away="showHistoryModal = false"
        >
            <div class="flex justify-between items-center border-b border-gray-200 dark:border-gray-700 p-4">
                <h2 class="text-xl font-semibold text-gray-900 dark:text-white">Decision History</h2>
                <div class="flex items-center">
                    <!-- Action filter removed -->
                    <button @click="showHistoryModal = false" class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
            
            <div class="overflow-y-auto flex-1 p-4">
                <template x-if="conversationHistory.length === 0">
                    <div class="text-center text-gray-500 dark:text-gray-400 py-8">
                        <p>No previous decisions found</p>
                    </div>
                </template>
                
                <template x-for="(history, index) in conversationHistory" :key="index">
                    <div 
                        @click="loadConversation(history.id)"
                        class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 mb-4 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                    >
                        <div class="flex justify-between items-start">
                            <div>
                                <h3 class="font-medium text-gray-900 dark:text-white">
                                    <span x-text="history.title || 'Untitled Decision'"></span>
                                </h3>
                                <p class="text-sm text-gray-500 dark:text-gray-400 mt-1" x-text="formatDate(history.timestamp)"></p>
                            </div>
                            <button 
                                @click.stop="deleteConversation(history.id)" 
                                class="text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400"
                                title="Delete"
                            >
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                        <p class="text-gray-700 dark:text-gray-300 text-sm mt-2 line-clamp-2" x-text="history.situation"></p>
                        <!-- Task association removed -->
                    </div>
                </template>
            </div>
        </div>
    </div>

    <!-- Sign in script -->
    <!--script>
        // Update the Clerk load event handler
        window.addEventListener('load', async function () {
            if (typeof Clerk !== 'undefined') {
                await Clerk.load();
                
                if (!Clerk.user) {
                    // Add sign-in link if user is not signed in
                    const signInLink = document.createElement('a');
                    signInLink.className = 'text-gray-700 dark:text-white hover:text-gray-900 dark:hover:text-gray-200 font-medium';
                    signInLink.textContent = 'Sign in';
                    signInLink.href = '#';
                    signInLink.onclick = (e) => {
                        e.preventDefault();
                        Clerk.openSignIn();
                    };
                    
                    document.getElementById('sign-in-button').appendChild(signInLink);
                } else {
                    // Add user button if signed in
                    Clerk.mountUserButton(document.getElementById('user-button'), {
                        afterSignOutUrl: window.location.href,
                        appearance: {
                            elements: {
                                userButtonAvatarBox: {
                                    width: '2.5rem',
                                    height: '2.5rem'
                                }
                            }
                        }
                    });
                }
            }
        });
    </script-->
</body>
</html>

================================================================================
FILE: ./frontend/src/styles.css
================================================================================
body {
    /*  background-color: #f7fafc; bg-gray-100 */
    font-family: 'Quicksand', sans-serif !important; /* font-sans */
}

.bg-white {
    background-color: #fffefa;
}

.pt-safe {
    padding-top: max(1rem, env(safe-area-inset-top));
}

@viewport {
    viewport-fit: cover;
}

.navbar {
    padding-top: max(1rem, env(safe-area-inset-top));
}

.custom-select {
    appearance: none;
    background-color: transparent;
    padding-right: 2rem; /* pr-8 */
    padding-top: 0.25rem; /* py-1 */
    padding-bottom: 0.25rem; /* py-1 */
    color: #4a5568; /* text-gray-700 */
    font-weight: 600; /* font-semibold */
    outline: none; /* focus:outline-none */
}

.custom-svg-container {
    pointer-events: none;
    position: absolute;
    right: 0;
    display: flex;
    align-items: center;
    padding-left: 0.5rem; /* px-2 */
    color: #4a5568; /* text-gray-700 */
}

.custom-svg {
    fill: currentColor;
    height: 1rem; /* h-4 */
    width: 1rem; /* w-4 */
}

.icon-button {
    color: #718096; /* text-gray-600 */
    transition: color 0.2s;
}

.icon-button:hover {
    color: #2d3748; /* hover:text-gray-800 */
}

.verse-number {
    color: #a0aec0; /* text-gray-500 */
    font-size: smaller;
    vertical-align: super;
}

.sticky-buttons {
    position: sticky;
    bottom: 0;
    background: white;
    z-index: 20;
    padding: 1rem; /* Add padding to sticky buttons */
}

.main-content {
    flex: 1; /* Take remaining space */
}

.chat-container {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 144px - env(safe-area-inset-bottom)); /* Account for bottom nav, input container and safe area */
    position: relative;
}

.chat-bubble {
    max-width: 80%;
    margin-bottom: 1rem;
    word-break: break-word;
}

.chat-bubble p {
    margin: 0;
    white-space: pre-wrap;
}

#chatMessages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    padding-bottom: calc(8rem + env(safe-area-inset-bottom)); /* Add extra padding at the bottom */
    scroll-behavior: smooth;
    display: flex;
    flex-direction: column;
}

.chat-start .chat-bubble {
    background-color: #f3f4f6;
    color: #1f2937;
}

.chat-end .chat-bubble {
    background-color: #2563eb;
    color: white;
}

.btm-nav {
    height: calc(64px + env(safe-area-inset-bottom));
    padding-bottom: env(safe-area-inset-bottom);
    background-color: white;
}

.btm-nav button {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 64px; /* Fixed height for the button content */
}

.btm-nav button.active {
    border-top: 2px solid #2563eb;
}

.btm-nav-label {
    font-size: 12px;
    margin-top: 4px;
}

.suggested-questions {
    margin-top: 0.5rem;
}

.suggested-question-btn {
    transition: all 0.2s;
    font-size: 0.875rem;
}

.suggested-question-btn:hover {
    background-color: #f3f4f6;
}

.btn-circle {
    border-radius: 50%;    /* Makes the button circular */
    width: 32px;          /* Fixed width */
    height: 32px;         /* Fixed height */
    display: flex;        /* Center the icon */
    align-items: center;  /* Center vertically */
    justify-content: center; /* Center horizontally */
    padding: 0;          /* Remove padding */
    border: 1px solid #e2e8f0; /* border-gray-200 */
}

.btn-sm {
    font-size: 0.875rem;  /* Smaller font size */
}

.translate-y-full {
    transform: translateY(100%);
}

.translate-y-0 {
    transform: translateY(0);
}

.translate-y-0 {
    transform: translateY(0);
}

.-translate-y-full {
    transform: translateY(-100%);
}

.transition-all {
    transition-property: all;
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}

.duration-300 {
    transition-duration: 300ms;
}

.paywall-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  padding-top: max(1rem, env(safe-area-inset-top));
  padding-bottom: env(safe-area-inset-bottom);
}

.paywall-card {
  background: white;
  border-radius: 1rem;
  padding: 2rem;
  width: 100%;
  max-width: 400px;
  text-align: center;
  margin-top: max(0px, env(safe-area-inset-top));
}

.subscription-option {
  border: 2px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 1rem;
  margin: 1rem 0;
  cursor: pointer;
  transition: all 0.2s;
}

.subscription-option.selected {
  border-color: #3b82f6;
  background-color: #eff6ff;
}

.pb-safe {
    padding-bottom: env(safe-area-inset-bottom);
}

.chat-input-container {
    padding-bottom: env(safe-area-inset-bottom);
    bottom: calc(4rem + env(safe-area-inset-bottom));
}

.bible-text {
    font-family: 'Goudy Bookletter 1911', serif;
    font-size: 1.125rem; /* text-lg */
    line-height: 1.75;
}

.bible-verse-link {
    color: #2563eb;
    cursor: pointer;
}

.bible-verse-link:hover {
    color: #1d4ed8;
}

.transition {
    transition-property: opacity, transform;
}

.duration-200 {
    transition-duration: 200ms;
}

.duration-300 {
    transition-duration: 300ms;
}

.ease-in {
    transition-timing-function: cubic-bezier(0.4, 0, 1, 1);
}

.ease-out {
    transition-timing-function: cubic-bezier(0, 0, 0.2, 1);
}

.transform {
    transform: translateY(0);
}

.translate-y-0 {
    transform: translateY(0);
}

.translate-y-2 {
    transform: translateY(0.5rem);
}

.opacity-0 {
    opacity: 0;
}

.opacity-100 {
    opacity: 1;
}

.animate-bounce {
    animation: bounce 1s infinite;
}

@keyframes bounce {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-4px);
    }
}

.dark .prose {
    color: #e5e7eb;
}

.dark .prose h1,
.dark .prose h2,
.dark .prose h3,
.dark .prose h4,
.dark .prose h5,
.dark .prose h6 {
    color: #f3f4f6;
}

.dark .prose strong {
    color: #f3f4f6;
}

.dark .prose a {
    color: #38bdf8;
}

.dark .prose blockquote {
    color: #d1d5db;
    border-left-color: #4b5563;
}

.dark .prose code {
    color: #f3f4f6;
    background-color: #374151;
}

.dark .prose pre {
    background-color: #1f2937;
    color: #f3f4f6;
}

.dark .prose ol > li::before {
    color: #9ca3af;
}

.dark .prose ul > li::before {
    background-color: #6b7280;
}

.dark .prose hr {
    border-color: #4b5563;
}

.dark .prose thead {
    color: #f3f4f6;
    border-bottom-color: #4b5563;
}

.dark .prose tbody tr {
    border-bottom-color: #4b5563;
}

.prose {
    max-width: none;
}

.prose p {
    margin-top: 1em;
    margin-bottom: 1em;
}

.prose h1, .prose h2, .prose h3, .prose h4 {
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

.prose ul, .prose ol {
    margin-top: 1em;
    margin-bottom: 1em;
    padding-left: 1.5em;
}

.prose li {
    margin-top: 0.5em;
    margin-bottom: 0.5em;
}

.prose blockquote {
    margin-top: 1em;
    margin-bottom: 1em;
    padding-left: 1em;
    border-left-width: 4px;
    border-left-color: #e5e7eb;
    font-style: italic;
}

.prose code {
    font-size: 0.875em;
    font-weight: 600;
    padding: 0.2em 0.4em;
    background-color: #f3f4f6;
    border-radius: 0.25em;
}

.prose pre {
    margin-top: 1em;
    margin-bottom: 1em;
    padding: 1em;
    background-color: #f3f4f6;
    border-radius: 0.375em;
    overflow-x: auto;
}

.prose pre code {
    background-color: transparent;
    padding: 0;
    font-weight: 400;
}

.prose a {
    color: #0ea5e9;
    text-decoration: underline;
    font-weight: 500;
}

.prose a:hover {
    color: #0284c7;
}

.prose table {
    width: 100%;
    margin-top: 1em;
    margin-bottom: 1em;
    border-collapse: collapse;
}

.prose thead {
    border-bottom-width: 1px;
    border-bottom-color: #e5e7eb;
}

.prose th {
    padding: 0.5em;
    text-align: left;
    font-weight: 600;
}

.prose td {
    padding: 0.5em;
    border-bottom-width: 1px;
    border-bottom-color: #e5e7eb;
}

.animate-spin {
    animation: spin 1s linear infinite;
}

@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

.transition {
    transition-property: all;
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}

.duration-200 {
    transition-duration: 200ms;
}

.duration-300 {
    transition-duration: 300ms;
}

.ease-in {
    transition-timing-function: cubic-bezier(0.4, 0, 1, 1);
}

.ease-out {
    transition-timing-function: cubic-bezier(0, 0, 0.2, 1);
}

.transform {
    transform: translateX(var(--tw-translate-x)) translateY(var(--tw-translate-y)) rotate(var(--tw-rotate)) skewX(var(--tw-skew-x)) skewY(var(--tw-skew-y)) scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y));
}

.translate-y-0 {
    --tw-translate-y: 0px;
}

.translate-y-2 {
    --tw-translate-y: 0.5rem;
}

.opacity-0 {
    opacity: 0;
}

.opacity-100 {
    opacity: 1;
}

.pb-safe {
    padding-bottom: env(safe-area-inset-bottom);
}

@media (prefers-color-scheme: dark) {
    .dark\:bg-gray-800 {
        background-color: #1f2937;
    }
    
    .dark\:bg-gray-700 {
        background-color: #374151;
    }
    
    .dark\:bg-gray-600 {
        background-color: #4b5563;
    }
    
    .dark\:text-white {
        color: #ffffff;
    }
    
    .dark\:text-gray-300 {
        color: #d1d5db;
    }
    
    .dark\:border-gray-600 {
        border-color: #4b5563;
    }
    
    .dark\:hover\:bg-gray-600:hover {
        background-color: #4b5563;
    }
    
    .dark\:hover\:bg-gray-500:hover {
        background-color: #6b7280;
    }
}

/* Markdown content styling */
.markdown-content {
  line-height: 1.6;
}

.markdown-content h1 {
  font-size: 1.8rem;
  font-weight: 600;
  margin-top: 1.5rem;
  margin-bottom: 1rem;
}

.markdown-content h2 {
  font-size: 1.5rem;
  font-weight: 600;
  margin-top: 1.5rem;
  margin-bottom: 0.75rem;
}

.markdown-content h3 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-top: 1.25rem;
  margin-bottom: 0.5rem;
}

.markdown-content p {
  margin-bottom: 1rem;
}

.markdown-content ul, .markdown-content ol {
  margin-left: 1.5rem;
  margin-bottom: 1rem;
}

.markdown-content ul {
  list-style-type: disc;
}

.markdown-content ol {
  list-style-type: decimal;
}

.markdown-content li {
  margin-bottom: 0.5rem;
}

.markdown-content code {
  font-family: monospace;
  background-color: rgba(0, 0, 0, 0.05);
  padding: 0.2rem 0.4rem;
  border-radius: 0.25rem;
}

.markdown-content pre {
  background-color: rgba(0, 0, 0, 0.05);
  padding: 1rem;
  border-radius: 0.5rem;
  overflow-x: auto;
  margin-bottom: 1rem;
}

.markdown-content pre code {
  background-color: transparent;
  padding: 0;
}

.markdown-content blockquote {
  border-left: 4px solid #e2e8f0;
  padding-left: 1rem;
  margin-left: 0;
  margin-bottom: 1rem;
  font-style: italic;
}

.markdown-content a {
  color: #3b82f6;
  text-decoration: underline;
}

.markdown-content a:hover {
  color: #2563eb;
}

.dark .markdown-content code {
  background-color: rgba(255, 255, 255, 0.1);
}

.dark .markdown-content pre {
  background-color: rgba(255, 255, 255, 0.1);
}

.dark .markdown-content blockquote {
  border-left-color: #4b5563;
}

.dark .markdown-content a {
  color: #60a5fa;
}

.dark .markdown-content a:hover {
  color: #93c5fd;
}

================================================================================
FILE: ./frontend/src/tailwind.css
================================================================================
@import 'tailwindcss/base';
@import 'tailwindcss/components';
@import 'tailwindcss/utilities';

================================================================================
FILE: ./frontend/package-lock.json
================================================================================
{
  "name": "say-less",
  "version": "1.0.0",
  "lockfileVersion": 3,
  "requires": true,
  "packages": {
    "": {
      "name": "say-less",
      "version": "1.0.0",
      "license": "ISC",
      "dependencies": {
        "@capacitor-community/in-app-review": "^6.0.0",
        "@capacitor/android": "^6.2.0",
        "@capacitor/app": "^6.0.2",
        "@capacitor/assets": "^3.0.5",
        "@capacitor/cli": "^6.2.0",
        "@capacitor/core": "^6.2.0",
        "@capacitor/ios": "^6.2.0",
        "@capgo/capacitor-updater": "^6.3.8",
        "@fortawesome/fontawesome-free": "^6.7.2",
        "@revenuecat/purchases-capacitor": "^9.0.9",
        "alpinejs": "^3.14.7",
        "animate.css": "^4.1.1",
        "htmx.org": "^1.9.12",
        "marked": "^15.0.4",
        "tailwindcss": "^3.4.1"
      },
      "devDependencies": {
        "autoprefixer": "^10.4.20",
        "postcss": "^8.4.49",
        "serve": "^14.2.4",
        "vite": "^6.0.5"
      }
    },
    "node_modules/@alloc/quick-lru": {
      "version": "5.2.0",
      "resolved": "https://registry.npmjs.org/@alloc/quick-lru/-/quick-lru-5.2.0.tgz",
      "integrity": "sha512-UrcABB+4bUrFABwbluTIBErXwvbsU/V7TZWfmbgJfbkwiBuziS9gxdODUyuiecfdGQ85jglMW6juS3+z5TsKLw==",
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/@babel/code-frame": {
      "version": "7.26.2",
      "resolved": "https://registry.npmjs.org/@babel/code-frame/-/code-frame-7.26.2.tgz",
      "integrity": "sha512-RJlIHRueQgwWitWgF8OdFYGZX328Ax5BCemNGlqHfplnRT9ESi8JkFlvaVYbS+UubVY6dpv87Fs2u5M29iNFVQ==",
      "dependencies": {
        "@babel/helper-validator-identifier": "^7.25.9",
        "js-tokens": "^4.0.0",
        "picocolors": "^1.0.0"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/helper-validator-identifier": {
      "version": "7.25.9",
      "resolved": "https://registry.npmjs.org/@babel/helper-validator-identifier/-/helper-validator-identifier-7.25.9.tgz",
      "integrity": "sha512-Ed61U6XJc3CVRfkERJWDz4dJwKe7iLmmJsbOGu9wSloNSFttHV0I8g6UAgb7qnK5ly5bGLPd4oXZlxCdANBOWQ==",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@capacitor-community/in-app-review": {
      "version": "6.0.0",
      "resolved": "https://registry.npmjs.org/@capacitor-community/in-app-review/-/in-app-review-6.0.0.tgz",
      "integrity": "sha512-k4GxGepsNV7YFf/GdAG0+M05U4Ew3SbzvrWVNUut+0tdTqxvLPCMzVBA41dolDSgRW7iDt923AKcRkKTJcv5FA==",
      "peerDependencies": {
        "@capacitor/core": "^6.0.0"
      }
    },
    "node_modules/@capacitor/android": {
      "version": "6.2.0",
      "resolved": "https://registry.npmjs.org/@capacitor/android/-/android-6.2.0.tgz",
      "integrity": "sha512-3YIDPylV0Q2adEQ/H568p496QdYG0jK/XGMdx7OGSqdBZen92ciAsYdyhLtyl91UVsN1lBhDi5H6j3T2KS6aJg==",
      "peerDependencies": {
        "@capacitor/core": "^6.2.0"
      }
    },
    "node_modules/@capacitor/app": {
      "version": "6.0.2",
      "resolved": "https://registry.npmjs.org/@capacitor/app/-/app-6.0.2.tgz",
      "integrity": "sha512-SiGTGgslK4TbWJVImCUL1odul7/YFkVfkYtAYS9AAEzQpxBECBeRnuN3FFBcfZ9eiN1XxFBFchhiwpxtx/c7yQ==",
      "peerDependencies": {
        "@capacitor/core": "^6.0.0"
      }
    },
    "node_modules/@capacitor/assets": {
      "version": "3.0.5",
      "resolved": "https://registry.npmjs.org/@capacitor/assets/-/assets-3.0.5.tgz",
      "integrity": "sha512-ohz/OUq61Y1Fc6aVSt0uDrUdeOA7oTH4pkWDbv/8I3UrPjH7oPkzYhShuDRUjekNp9RBi198VSFdt0CetpEOzw==",
      "dependencies": {
        "@capacitor/cli": "^5.3.0",
        "@ionic/utils-array": "2.1.6",
        "@ionic/utils-fs": "3.1.7",
        "@trapezedev/project": "^7.0.10",
        "commander": "8.3.0",
        "debug": "4.3.4",
        "fs-extra": "10.1.0",
        "node-fetch": "2.7.0",
        "node-html-parser": "5.4.2",
        "sharp": "0.32.6",
        "tslib": "2.6.2",
        "yargs": "17.7.2"
      },
      "bin": {
        "capacitor-assets": "bin/capacitor-assets"
      },
      "engines": {
        "node": ">=10.3.0"
      }
    },
    "node_modules/@capacitor/assets/node_modules/@capacitor/cli": {
      "version": "5.7.8",
      "resolved": "https://registry.npmjs.org/@capacitor/cli/-/cli-5.7.8.tgz",
      "integrity": "sha512-qN8LDlREMhrYhOvVXahoJVNkP8LP55/YPRJrzTAFrMqlNJC18L3CzgWYIblFPnuwfbH/RxbfoZT/ydkwgVpMrw==",
      "dependencies": {
        "@ionic/cli-framework-output": "^2.2.5",
        "@ionic/utils-fs": "^3.1.6",
        "@ionic/utils-subprocess": "^2.1.11",
        "@ionic/utils-terminal": "^2.3.3",
        "commander": "^9.3.0",
        "debug": "^4.3.4",
        "env-paths": "^2.2.0",
        "kleur": "^4.1.4",
        "native-run": "^2.0.0",
        "open": "^8.4.0",
        "plist": "^3.0.5",
        "prompts": "^2.4.2",
        "rimraf": "^4.4.1",
        "semver": "^7.3.7",
        "tar": "^6.1.11",
        "tslib": "^2.4.0",
        "xml2js": "^0.5.0"
      },
      "bin": {
        "cap": "bin/capacitor",
        "capacitor": "bin/capacitor"
      },
      "engines": {
        "node": ">=16.0.0"
      }
    },
    "node_modules/@capacitor/assets/node_modules/@capacitor/cli/node_modules/commander": {
      "version": "9.5.0",
      "resolved": "https://registry.npmjs.org/commander/-/commander-9.5.0.tgz",
      "integrity": "sha512-KRs7WVDKg86PWiuAqhDrAQnTXZKraVcCc6vFdL14qrZ/DcWwuRo7VoiYXalXO7S5GKpqYiVEwCbgFDfxNHKJBQ==",
      "engines": {
        "node": "^12.20.0 || >=14"
      }
    },
    "node_modules/@capacitor/cli": {
      "version": "6.2.0",
      "resolved": "https://registry.npmjs.org/@capacitor/cli/-/cli-6.2.0.tgz",
      "integrity": "sha512-EWcXG39mZh35zrHhOqzN1ILeSyMRyEqWVtQDXqMGjCXYRH6b6p5TvyvLDN8ZNy26tbhI3i79gfrgirt+mNwuuw==",
      "dependencies": {
        "@ionic/cli-framework-output": "^2.2.5",
        "@ionic/utils-fs": "^3.1.6",
        "@ionic/utils-subprocess": "2.1.11",
        "@ionic/utils-terminal": "^2.3.3",
        "commander": "^9.3.0",
        "debug": "^4.3.4",
        "env-paths": "^2.2.0",
        "kleur": "^4.1.4",
        "native-run": "^2.0.0",
        "open": "^8.4.0",
        "plist": "^3.0.5",
        "prompts": "^2.4.2",
        "rimraf": "^4.4.1",
        "semver": "^7.3.7",
        "tar": "^6.1.11",
        "tslib": "^2.4.0",
        "xml2js": "^0.5.0"
      },
      "bin": {
        "cap": "bin/capacitor",
        "capacitor": "bin/capacitor"
      },
      "engines": {
        "node": ">=18.0.0"
      }
    },
    "node_modules/@capacitor/cli/node_modules/commander": {
      "version": "9.5.0",
      "resolved": "https://registry.npmjs.org/commander/-/commander-9.5.0.tgz",
      "integrity": "sha512-KRs7WVDKg86PWiuAqhDrAQnTXZKraVcCc6vFdL14qrZ/DcWwuRo7VoiYXalXO7S5GKpqYiVEwCbgFDfxNHKJBQ==",
      "engines": {
        "node": "^12.20.0 || >=14"
      }
    },
    "node_modules/@capacitor/core": {
      "version": "6.2.0",
      "resolved": "https://registry.npmjs.org/@capacitor/core/-/core-6.2.0.tgz",
      "integrity": "sha512-B9IlJtDpUqhhYb+T8+cp2Db/3RETX36STgjeU2kQZBs/SLAcFiMama227o+msRjLeo3DO+7HJjWVA1+XlyyPEg==",
      "dependencies": {
        "tslib": "^2.1.0"
      }
    },
    "node_modules/@capacitor/ios": {
      "version": "6.2.0",
      "resolved": "https://registry.npmjs.org/@capacitor/ios/-/ios-6.2.0.tgz",
      "integrity": "sha512-gisvZBIrKT1siiumgpLPY63HmJe69Ed/dOmfQQ+U1MIJmOR5gWGWvfO7QSj/FMatVZS4Xt/8jCoUgzDD1U6kSw==",
      "peerDependencies": {
        "@capacitor/core": "^6.2.0"
      }
    },
    "node_modules/@capgo/capacitor-updater": {
      "version": "6.14.12",
      "resolved": "https://registry.npmjs.org/@capgo/capacitor-updater/-/capacitor-updater-6.14.12.tgz",
      "integrity": "sha512-z13JtGWHIncx0footBYHBzv/Lp//JhquEI+l8hxM02YMokq+kTGu9vcXd9CS8AeDoe6Rtjs3B9QhnbdZM++Sqg==",
      "peerDependencies": {
        "@capacitor/core": "^6.0.0"
      }
    },
    "node_modules/@cspotcode/source-map-support": {
      "version": "0.8.1",
      "resolved": "https://registry.npmjs.org/@cspotcode/source-map-support/-/source-map-support-0.8.1.tgz",
      "integrity": "sha512-IchNf6dN4tHoMFIn/7OE8LWZ19Y6q/67Bmf6vnGREv8RSbBVb9LPJxEcnwrcwX6ixSvaiGoomAUvu4YSxXrVgw==",
      "dependencies": {
        "@jridgewell/trace-mapping": "0.3.9"
      },
      "engines": {
        "node": ">=12"
      }
    },
    "node_modules/@cspotcode/source-map-support/node_modules/@jridgewell/trace-mapping": {
      "version": "0.3.9",
      "resolved": "https://registry.npmjs.org/@jridgewell/trace-mapping/-/trace-mapping-0.3.9.tgz",
      "integrity": "sha512-3Belt6tdc8bPgAtbcmdtNJlirVoTmEb5e2gC94PnkwEW9jI6CAHUeoG85tjWP5WquqfavoMtMwiG4P926ZKKuQ==",
      "dependencies": {
        "@jridgewell/resolve-uri": "^3.0.3",
        "@jridgewell/sourcemap-codec": "^1.4.10"
      }
    },
    "node_modules/@esbuild/aix-ppc64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/aix-ppc64/-/aix-ppc64-0.24.0.tgz",
      "integrity": "sha512-WtKdFM7ls47zkKHFVzMz8opM7LkcsIp9amDUBIAWirg70RM71WRSjdILPsY5Uv1D42ZpUfaPILDlfactHgsRkw==",
      "cpu": [
        "ppc64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "aix"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/android-arm": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/android-arm/-/android-arm-0.24.0.tgz",
      "integrity": "sha512-arAtTPo76fJ/ICkXWetLCc9EwEHKaeya4vMrReVlEIUCAUncH7M4bhMQ+M9Vf+FFOZJdTNMXNBrWwW+OXWpSew==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/android-arm64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/android-arm64/-/android-arm64-0.24.0.tgz",
      "integrity": "sha512-Vsm497xFM7tTIPYK9bNTYJyF/lsP590Qc1WxJdlB6ljCbdZKU9SY8i7+Iin4kyhV/KV5J2rOKsBQbB77Ab7L/w==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/android-x64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/android-x64/-/android-x64-0.24.0.tgz",
      "integrity": "sha512-t8GrvnFkiIY7pa7mMgJd7p8p8qqYIz1NYiAoKc75Zyv73L3DZW++oYMSHPRarcotTKuSs6m3hTOa5CKHaS02TQ==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/darwin-arm64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/darwin-arm64/-/darwin-arm64-0.24.0.tgz",
      "integrity": "sha512-CKyDpRbK1hXwv79soeTJNHb5EiG6ct3efd/FTPdzOWdbZZfGhpbcqIpiD0+vwmpu0wTIL97ZRPZu8vUt46nBSw==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/darwin-x64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/darwin-x64/-/darwin-x64-0.24.0.tgz",
      "integrity": "sha512-rgtz6flkVkh58od4PwTRqxbKH9cOjaXCMZgWD905JOzjFKW+7EiUObfd/Kav+A6Gyud6WZk9w+xu6QLytdi2OA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/freebsd-arm64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/freebsd-arm64/-/freebsd-arm64-0.24.0.tgz",
      "integrity": "sha512-6Mtdq5nHggwfDNLAHkPlyLBpE5L6hwsuXZX8XNmHno9JuL2+bg2BX5tRkwjyfn6sKbxZTq68suOjgWqCicvPXA==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "freebsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/freebsd-x64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/freebsd-x64/-/freebsd-x64-0.24.0.tgz",
      "integrity": "sha512-D3H+xh3/zphoX8ck4S2RxKR6gHlHDXXzOf6f/9dbFt/NRBDIE33+cVa49Kil4WUjxMGW0ZIYBYtaGCa2+OsQwQ==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "freebsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-arm": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-arm/-/linux-arm-0.24.0.tgz",
      "integrity": "sha512-gJKIi2IjRo5G6Glxb8d3DzYXlxdEj2NlkixPsqePSZMhLudqPhtZ4BUrpIuTjJYXxvF9njql+vRjB2oaC9XpBw==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-arm64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-arm64/-/linux-arm64-0.24.0.tgz",
      "integrity": "sha512-TDijPXTOeE3eaMkRYpcy3LarIg13dS9wWHRdwYRnzlwlA370rNdZqbcp0WTyyV/k2zSxfko52+C7jU5F9Tfj1g==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-ia32": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-ia32/-/linux-ia32-0.24.0.tgz",
      "integrity": "sha512-K40ip1LAcA0byL05TbCQ4yJ4swvnbzHscRmUilrmP9Am7//0UjPreh4lpYzvThT2Quw66MhjG//20mrufm40mA==",
      "cpu": [
        "ia32"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-loong64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-loong64/-/linux-loong64-0.24.0.tgz",
      "integrity": "sha512-0mswrYP/9ai+CU0BzBfPMZ8RVm3RGAN/lmOMgW4aFUSOQBjA31UP8Mr6DDhWSuMwj7jaWOT0p0WoZ6jeHhrD7g==",
      "cpu": [
        "loong64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-mips64el": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-mips64el/-/linux-mips64el-0.24.0.tgz",
      "integrity": "sha512-hIKvXm0/3w/5+RDtCJeXqMZGkI2s4oMUGj3/jM0QzhgIASWrGO5/RlzAzm5nNh/awHE0A19h/CvHQe6FaBNrRA==",
      "cpu": [
        "mips64el"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-ppc64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-ppc64/-/linux-ppc64-0.24.0.tgz",
      "integrity": "sha512-HcZh5BNq0aC52UoocJxaKORfFODWXZxtBaaZNuN3PUX3MoDsChsZqopzi5UupRhPHSEHotoiptqikjN/B77mYQ==",
      "cpu": [
        "ppc64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-riscv64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-riscv64/-/linux-riscv64-0.24.0.tgz",
      "integrity": "sha512-bEh7dMn/h3QxeR2KTy1DUszQjUrIHPZKyO6aN1X4BCnhfYhuQqedHaa5MxSQA/06j3GpiIlFGSsy1c7Gf9padw==",
      "cpu": [
        "riscv64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-s390x": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-s390x/-/linux-s390x-0.24.0.tgz",
      "integrity": "sha512-ZcQ6+qRkw1UcZGPyrCiHHkmBaj9SiCD8Oqd556HldP+QlpUIe2Wgn3ehQGVoPOvZvtHm8HPx+bH20c9pvbkX3g==",
      "cpu": [
        "s390x"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-x64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-x64/-/linux-x64-0.24.0.tgz",
      "integrity": "sha512-vbutsFqQ+foy3wSSbmjBXXIJ6PL3scghJoM8zCL142cGaZKAdCZHyf+Bpu/MmX9zT9Q0zFBVKb36Ma5Fzfa8xA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/netbsd-x64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/netbsd-x64/-/netbsd-x64-0.24.0.tgz",
      "integrity": "sha512-hjQ0R/ulkO8fCYFsG0FZoH+pWgTTDreqpqY7UnQntnaKv95uP5iW3+dChxnx7C3trQQU40S+OgWhUVwCjVFLvg==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "netbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/openbsd-arm64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/openbsd-arm64/-/openbsd-arm64-0.24.0.tgz",
      "integrity": "sha512-MD9uzzkPQbYehwcN583yx3Tu5M8EIoTD+tUgKF982WYL9Pf5rKy9ltgD0eUgs8pvKnmizxjXZyLt0z6DC3rRXg==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "openbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/openbsd-x64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/openbsd-x64/-/openbsd-x64-0.24.0.tgz",
      "integrity": "sha512-4ir0aY1NGUhIC1hdoCzr1+5b43mw99uNwVzhIq1OY3QcEwPDO3B7WNXBzaKY5Nsf1+N11i1eOfFcq+D/gOS15Q==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "openbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/sunos-x64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/sunos-x64/-/sunos-x64-0.24.0.tgz",
      "integrity": "sha512-jVzdzsbM5xrotH+W5f1s+JtUy1UWgjU0Cf4wMvffTB8m6wP5/kx0KiaLHlbJO+dMgtxKV8RQ/JvtlFcdZ1zCPA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "sunos"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/win32-arm64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/win32-arm64/-/win32-arm64-0.24.0.tgz",
      "integrity": "sha512-iKc8GAslzRpBytO2/aN3d2yb2z8XTVfNV0PjGlCxKo5SgWmNXx82I/Q3aG1tFfS+A2igVCY97TJ8tnYwpUWLCA==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/win32-ia32": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/win32-ia32/-/win32-ia32-0.24.0.tgz",
      "integrity": "sha512-vQW36KZolfIudCcTnaTpmLQ24Ha1RjygBo39/aLkM2kmjkWmZGEJ5Gn9l5/7tzXA42QGIoWbICfg6KLLkIw6yw==",
      "cpu": [
        "ia32"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/win32-x64": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/@esbuild/win32-x64/-/win32-x64-0.24.0.tgz",
      "integrity": "sha512-7IAFPrjSQIJrGsK6flwg7NFmwBoSTyF3rl7If0hNUFQU4ilTsEPL6GuMuU9BfIWVVGuRnuIidkSMC+c0Otu8IA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@fortawesome/fontawesome-free": {
      "version": "6.7.2",
      "resolved": "https://registry.npmjs.org/@fortawesome/fontawesome-free/-/fontawesome-free-6.7.2.tgz",
      "integrity": "sha512-JUOtgFW6k9u4Y+xeIaEiLr3+cjoUPiAuLXoyKOJSia6Duzb7pq+A76P9ZdPDoAoxHdHzq6gE9/jKBGXlZT8FbA==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/@hutson/parse-repository-url": {
      "version": "3.0.2",
      "resolved": "https://registry.npmjs.org/@hutson/parse-repository-url/-/parse-repository-url-3.0.2.tgz",
      "integrity": "sha512-H9XAx3hc0BQHY6l+IFSWHDySypcXsvsuLhgYLUGywmJ5pswRVQJUHpOsobnLYp2ZUaUlKiKDrgWWhosOwAEM8Q==",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@ionic/cli-framework-output": {
      "version": "2.2.8",
      "resolved": "https://registry.npmjs.org/@ionic/cli-framework-output/-/cli-framework-output-2.2.8.tgz",
      "integrity": "sha512-TshtaFQsovB4NWRBydbNFawql6yul7d5bMiW1WYYf17hd99V6xdDdk3vtF51bw6sLkxON3bDQpWsnUc9/hVo3g==",
      "dependencies": {
        "@ionic/utils-terminal": "2.3.5",
        "debug": "^4.0.0",
        "tslib": "^2.0.1"
      },
      "engines": {
        "node": ">=16.0.0"
      }
    },
    "node_modules/@ionic/utils-array": {
      "version": "2.1.6",
      "resolved": "https://registry.npmjs.org/@ionic/utils-array/-/utils-array-2.1.6.tgz",
      "integrity": "sha512-0JZ1Zkp3wURnv8oq6Qt7fMPo5MpjbLoUoa9Bu2Q4PJuSDWM8H8gwF3dQO7VTeUj3/0o1IB1wGkFWZZYgUXZMUg==",
      "dependencies": {
        "debug": "^4.0.0",
        "tslib": "^2.0.1"
      },
      "engines": {
        "node": ">=16.0.0"
      }
    },
    "node_modules/@ionic/utils-fs": {
      "version": "3.1.7",
      "resolved": "https://registry.npmjs.org/@ionic/utils-fs/-/utils-fs-3.1.7.tgz",
      "integrity": "sha512-2EknRvMVfhnyhL1VhFkSLa5gOcycK91VnjfrTB0kbqkTFCOXyXgVLI5whzq7SLrgD9t1aqos3lMMQyVzaQ5gVA==",
      "dependencies": {
        "@types/fs-extra": "^8.0.0",
        "debug": "^4.0.0",
        "fs-extra": "^9.0.0",
        "tslib": "^2.0.1"
      },
      "engines": {
        "node": ">=16.0.0"
      }
    },
    "node_modules/@ionic/utils-fs/node_modules/fs-extra": {
      "version": "9.1.0",
      "resolved": "https://registry.npmjs.org/fs-extra/-/fs-extra-9.1.0.tgz",
      "integrity": "sha512-hcg3ZmepS30/7BSFqRvoo3DOMQu7IjqxO5nCDt+zM9XWjb33Wg7ziNT+Qvqbuc3+gWpzO02JubVyk2G4Zvo1OQ==",
      "dependencies": {
        "at-least-node": "^1.0.0",
        "graceful-fs": "^4.2.0",
        "jsonfile": "^6.0.1",
        "universalify": "^2.0.0"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/@ionic/utils-object": {
      "version": "2.1.5",
      "resolved": "https://registry.npmjs.org/@ionic/utils-object/-/utils-object-2.1.5.tgz",
      "integrity": "sha512-XnYNSwfewUqxq+yjER1hxTKggftpNjFLJH0s37jcrNDwbzmbpFTQTVAp4ikNK4rd9DOebX/jbeZb8jfD86IYxw==",
      "dependencies": {
        "debug": "^4.0.0",
        "tslib": "^2.0.1"
      },
      "engines": {
        "node": ">=10.3.0"
      }
    },
    "node_modules/@ionic/utils-process": {
      "version": "2.1.10",
      "resolved": "https://registry.npmjs.org/@ionic/utils-process/-/utils-process-2.1.10.tgz",
      "integrity": "sha512-mZ7JEowcuGQK+SKsJXi0liYTcXd2bNMR3nE0CyTROpMECUpJeAvvaBaPGZf5ERQUPeWBVuwqAqjUmIdxhz5bxw==",
      "dependencies": {
        "@ionic/utils-object": "2.1.5",
        "@ionic/utils-terminal": "2.3.3",
        "debug": "^4.0.0",
        "signal-exit": "^3.0.3",
        "tree-kill": "^1.2.2",
        "tslib": "^2.0.1"
      },
      "engines": {
        "node": ">=10.3.0"
      }
    },
    "node_modules/@ionic/utils-process/node_modules/@ionic/utils-terminal": {
      "version": "2.3.3",
      "resolved": "https://registry.npmjs.org/@ionic/utils-terminal/-/utils-terminal-2.3.3.tgz",
      "integrity": "sha512-RnuSfNZ5fLEyX3R5mtcMY97cGD1A0NVBbarsSQ6yMMfRJ5YHU7hHVyUfvZeClbqkBC/pAqI/rYJuXKCT9YeMCQ==",
      "dependencies": {
        "@types/slice-ansi": "^4.0.0",
        "debug": "^4.0.0",
        "signal-exit": "^3.0.3",
        "slice-ansi": "^4.0.0",
        "string-width": "^4.1.0",
        "strip-ansi": "^6.0.0",
        "tslib": "^2.0.1",
        "untildify": "^4.0.0",
        "wrap-ansi": "^7.0.0"
      },
      "engines": {
        "node": ">=10.3.0"
      }
    },
    "node_modules/@ionic/utils-stream": {
      "version": "3.1.5",
      "resolved": "https://registry.npmjs.org/@ionic/utils-stream/-/utils-stream-3.1.5.tgz",
      "integrity": "sha512-hkm46uHvEC05X/8PHgdJi4l4zv9VQDELZTM+Kz69odtO9zZYfnt8DkfXHJqJ+PxmtiE5mk/ehJWLnn/XAczTUw==",
      "dependencies": {
        "debug": "^4.0.0",
        "tslib": "^2.0.1"
      },
      "engines": {
        "node": ">=10.3.0"
      }
    },
    "node_modules/@ionic/utils-subprocess": {
      "version": "2.1.11",
      "resolved": "https://registry.npmjs.org/@ionic/utils-subprocess/-/utils-subprocess-2.1.11.tgz",
      "integrity": "sha512-6zCDixNmZCbMCy5np8klSxOZF85kuDyzZSTTQKQP90ZtYNCcPYmuFSzaqDwApJT4r5L3MY3JrqK1gLkc6xiUPw==",
      "dependencies": {
        "@ionic/utils-array": "2.1.5",
        "@ionic/utils-fs": "3.1.6",
        "@ionic/utils-process": "2.1.10",
        "@ionic/utils-stream": "3.1.5",
        "@ionic/utils-terminal": "2.3.3",
        "cross-spawn": "^7.0.3",
        "debug": "^4.0.0",
        "tslib": "^2.0.1"
      },
      "engines": {
        "node": ">=10.3.0"
      }
    },
    "node_modules/@ionic/utils-subprocess/node_modules/@ionic/utils-array": {
      "version": "2.1.5",
      "resolved": "https://registry.npmjs.org/@ionic/utils-array/-/utils-array-2.1.5.tgz",
      "integrity": "sha512-HD72a71IQVBmQckDwmA8RxNVMTbxnaLbgFOl+dO5tbvW9CkkSFCv41h6fUuNsSEVgngfkn0i98HDuZC8mk+lTA==",
      "dependencies": {
        "debug": "^4.0.0",
        "tslib": "^2.0.1"
      },
      "engines": {
        "node": ">=10.3.0"
      }
    },
    "node_modules/@ionic/utils-subprocess/node_modules/@ionic/utils-fs": {
      "version": "3.1.6",
      "resolved": "https://registry.npmjs.org/@ionic/utils-fs/-/utils-fs-3.1.6.tgz",
      "integrity": "sha512-eikrNkK89CfGPmexjTfSWl4EYqsPSBh0Ka7by4F0PLc1hJZYtJxUZV3X4r5ecA8ikjicUmcbU7zJmAjmqutG/w==",
      "dependencies": {
        "@types/fs-extra": "^8.0.0",
        "debug": "^4.0.0",
        "fs-extra": "^9.0.0",
        "tslib": "^2.0.1"
      },
      "engines": {
        "node": ">=10.3.0"
      }
    },
    "node_modules/@ionic/utils-subprocess/node_modules/@ionic/utils-terminal": {
      "version": "2.3.3",
      "resolved": "https://registry.npmjs.org/@ionic/utils-terminal/-/utils-terminal-2.3.3.tgz",
      "integrity": "sha512-RnuSfNZ5fLEyX3R5mtcMY97cGD1A0NVBbarsSQ6yMMfRJ5YHU7hHVyUfvZeClbqkBC/pAqI/rYJuXKCT9YeMCQ==",
      "dependencies": {
        "@types/slice-ansi": "^4.0.0",
        "debug": "^4.0.0",
        "signal-exit": "^3.0.3",
        "slice-ansi": "^4.0.0",
        "string-width": "^4.1.0",
        "strip-ansi": "^6.0.0",
        "tslib": "^2.0.1",
        "untildify": "^4.0.0",
        "wrap-ansi": "^7.0.0"
      },
      "engines": {
        "node": ">=10.3.0"
      }
    },
    "node_modules/@ionic/utils-subprocess/node_modules/fs-extra": {
      "version": "9.1.0",
      "resolved": "https://registry.npmjs.org/fs-extra/-/fs-extra-9.1.0.tgz",
      "integrity": "sha512-hcg3ZmepS30/7BSFqRvoo3DOMQu7IjqxO5nCDt+zM9XWjb33Wg7ziNT+Qvqbuc3+gWpzO02JubVyk2G4Zvo1OQ==",
      "dependencies": {
        "at-least-node": "^1.0.0",
        "graceful-fs": "^4.2.0",
        "jsonfile": "^6.0.1",
        "universalify": "^2.0.0"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/@ionic/utils-terminal": {
      "version": "2.3.5",
      "resolved": "https://registry.npmjs.org/@ionic/utils-terminal/-/utils-terminal-2.3.5.tgz",
      "integrity": "sha512-3cKScz9Jx2/Pr9ijj1OzGlBDfcmx7OMVBt4+P1uRR0SSW4cm1/y3Mo4OY3lfkuaYifMNBW8Wz6lQHbs1bihr7A==",
      "dependencies": {
        "@types/slice-ansi": "^4.0.0",
        "debug": "^4.0.0",
        "signal-exit": "^3.0.3",
        "slice-ansi": "^4.0.0",
        "string-width": "^4.1.0",
        "strip-ansi": "^6.0.0",
        "tslib": "^2.0.1",
        "untildify": "^4.0.0",
        "wrap-ansi": "^7.0.0"
      },
      "engines": {
        "node": ">=16.0.0"
      }
    },
    "node_modules/@isaacs/cliui": {
      "version": "8.0.2",
      "resolved": "https://registry.npmjs.org/@isaacs/cliui/-/cliui-8.0.2.tgz",
      "integrity": "sha512-O8jcjabXaleOG9DQ0+ARXWZBTfnP4WNAqzuiJK7ll44AmxGKv/J2M4TPjxjY3znBCfvBXFzucm1twdyFybFqEA==",
      "dependencies": {
        "string-width": "^5.1.2",
        "string-width-cjs": "npm:string-width@^4.2.0",
        "strip-ansi": "^7.0.1",
        "strip-ansi-cjs": "npm:strip-ansi@^6.0.1",
        "wrap-ansi": "^8.1.0",
        "wrap-ansi-cjs": "npm:wrap-ansi@^7.0.0"
      },
      "engines": {
        "node": ">=12"
      }
    },
    "node_modules/@isaacs/cliui/node_modules/ansi-regex": {
      "version": "6.1.0",
      "resolved": "https://registry.npmjs.org/ansi-regex/-/ansi-regex-6.1.0.tgz",
      "integrity": "sha512-7HSX4QQb4CspciLpVFwyRe79O3xsIZDDLER21kERQ71oaPodF8jL725AgJMFAYbooIqolJoRLuM81SpeUkpkvA==",
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/ansi-regex?sponsor=1"
      }
    },
    "node_modules/@isaacs/cliui/node_modules/ansi-styles": {
      "version": "6.2.1",
      "resolved": "https://registry.npmjs.org/ansi-styles/-/ansi-styles-6.2.1.tgz",
      "integrity": "sha512-bN798gFfQX+viw3R7yrGWRqnrN2oRkEkUjjl4JNn4E8GxxbjtG3FbrEIIY3l8/hrwUwIeCZvi4QuOTP4MErVug==",
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/ansi-styles?sponsor=1"
      }
    },
    "node_modules/@isaacs/cliui/node_modules/emoji-regex": {
      "version": "9.2.2",
      "resolved": "https://registry.npmjs.org/emoji-regex/-/emoji-regex-9.2.2.tgz",
      "integrity": "sha512-L18DaJsXSUk2+42pv8mLs5jJT2hqFkFE4j21wOmgbUqsZ2hL72NsUU785g9RXgo3s0ZNgVl42TiHp3ZtOv/Vyg=="
    },
    "node_modules/@isaacs/cliui/node_modules/string-width": {
      "version": "5.1.2",
      "resolved": "https://registry.npmjs.org/string-width/-/string-width-5.1.2.tgz",
      "integrity": "sha512-HnLOCR3vjcY8beoNLtcjZ5/nxn2afmME6lhrDrebokqMap+XbeW8n9TXpPDOqdGK5qcI3oT0GKTW6wC7EMiVqA==",
      "dependencies": {
        "eastasianwidth": "^0.2.0",
        "emoji-regex": "^9.2.2",
        "strip-ansi": "^7.0.1"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/@isaacs/cliui/node_modules/strip-ansi": {
      "version": "7.1.0",
      "resolved": "https://registry.npmjs.org/strip-ansi/-/strip-ansi-7.1.0.tgz",
      "integrity": "sha512-iq6eVVI64nQQTRYq2KtEg2d2uU7LElhTJwsH4YzIHZshxlgZms/wIc4VoDQTlG/IvVIrBKG06CrZnp0qv7hkcQ==",
      "dependencies": {
        "ansi-regex": "^6.0.1"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/strip-ansi?sponsor=1"
      }
    },
    "node_modules/@isaacs/cliui/node_modules/wrap-ansi": {
      "version": "8.1.0",
      "resolved": "https://registry.npmjs.org/wrap-ansi/-/wrap-ansi-8.1.0.tgz",
      "integrity": "sha512-si7QWI6zUMq56bESFvagtmzMdGOtoxfR+Sez11Mobfc7tm+VkUckk9bW2UeffTGVUbOksxmSw0AA2gs8g71NCQ==",
      "dependencies": {
        "ansi-styles": "^6.1.0",
        "string-width": "^5.0.1",
        "strip-ansi": "^7.0.1"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/wrap-ansi?sponsor=1"
      }
    },
    "node_modules/@jridgewell/gen-mapping": {
      "version": "0.3.8",
      "resolved": "https://registry.npmjs.org/@jridgewell/gen-mapping/-/gen-mapping-0.3.8.tgz",
      "integrity": "sha512-imAbBGkb+ebQyxKgzv5Hu2nmROxoDOXHh80evxdoXNOrvAnVx7zimzc1Oo5h9RlfV4vPXaE2iM5pOFbvOCClWA==",
      "dependencies": {
        "@jridgewell/set-array": "^1.2.1",
        "@jridgewell/sourcemap-codec": "^1.4.10",
        "@jridgewell/trace-mapping": "^0.3.24"
      },
      "engines": {
        "node": ">=6.0.0"
      }
    },
    "node_modules/@jridgewell/resolve-uri": {
      "version": "3.1.2",
      "resolved": "https://registry.npmjs.org/@jridgewell/resolve-uri/-/resolve-uri-3.1.2.tgz",
      "integrity": "sha512-bRISgCIjP20/tbWSPWMEi54QVPRZExkuD9lJL+UIxUKtwVJA8wW1Trb1jMs1RFXo1CBTNZ/5hpC9QvmKWdopKw==",
      "engines": {
        "node": ">=6.0.0"
      }
    },
    "node_modules/@jridgewell/set-array": {
      "version": "1.2.1",
      "resolved": "https://registry.npmjs.org/@jridgewell/set-array/-/set-array-1.2.1.tgz",
      "integrity": "sha512-R8gLRTZeyp03ymzP/6Lil/28tGeGEzhx1q2k703KGWRAI1VdvPIXdG70VJc2pAMw3NA6JKL5hhFu1sJX0Mnn/A==",
      "engines": {
        "node": ">=6.0.0"
      }
    },
    "node_modules/@jridgewell/source-map": {
      "version": "0.3.6",
      "resolved": "https://registry.npmjs.org/@jridgewell/source-map/-/source-map-0.3.6.tgz",
      "integrity": "sha512-1ZJTZebgqllO79ue2bm3rIGud/bOe0pP5BjSRCRxxYkEZS8STV7zN84UBbiYu7jy+eCKSnVIUgoWWE/tt+shMQ==",
      "dev": true,
      "optional": true,
      "peer": true,
      "dependencies": {
        "@jridgewell/gen-mapping": "^0.3.5",
        "@jridgewell/trace-mapping": "^0.3.25"
      }
    },
    "node_modules/@jridgewell/sourcemap-codec": {
      "version": "1.5.0",
      "resolved": "https://registry.npmjs.org/@jridgewell/sourcemap-codec/-/sourcemap-codec-1.5.0.tgz",
      "integrity": "sha512-gv3ZRaISU3fjPAgNsriBRqGWQL6quFx04YMPW/zD8XMLsU32mhCCbfbO6KZFLjvYpCZ8zyDEgqsgf+PwPaM7GQ=="
    },
    "node_modules/@jridgewell/trace-mapping": {
      "version": "0.3.25",
      "resolved": "https://registry.npmjs.org/@jridgewell/trace-mapping/-/trace-mapping-0.3.25.tgz",
      "integrity": "sha512-vNk6aEwybGtawWmy/PzwnGDOjCkLWSD2wqvjGGAgOAwCGWySYXfYoxt00IJkTF+8Lb57DwOb3Aa0o9CApepiYQ==",
      "dependencies": {
        "@jridgewell/resolve-uri": "^3.1.0",
        "@jridgewell/sourcemap-codec": "^1.4.14"
      }
    },
    "node_modules/@nodelib/fs.scandir": {
      "version": "2.1.5",
      "resolved": "https://registry.npmjs.org/@nodelib/fs.scandir/-/fs.scandir-2.1.5.tgz",
      "integrity": "sha512-vq24Bq3ym5HEQm2NKCr3yXDwjc7vTsEThRDnkp2DK9p1uqLR+DHurm/NOTo0KG7HYHU7eppKZj3MyqYuMBf62g==",
      "dependencies": {
        "@nodelib/fs.stat": "2.0.5",
        "run-parallel": "^1.1.9"
      },
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/@nodelib/fs.stat": {
      "version": "2.0.5",
      "resolved": "https://registry.npmjs.org/@nodelib/fs.stat/-/fs.stat-2.0.5.tgz",
      "integrity": "sha512-RkhPPp2zrqDAQA/2jNhnztcPAlv64XdhIp7a7454A5ovI7Bukxgt7MX7udwAu3zg1DcpPU0rz3VV1SeaqvY4+A==",
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/@nodelib/fs.walk": {
      "version": "1.2.8",
      "resolved": "https://registry.npmjs.org/@nodelib/fs.walk/-/fs.walk-1.2.8.tgz",
      "integrity": "sha512-oGB+UxlgWcgQkgwo8GcEGwemoTFt3FIO9ababBmaGwXIoBKZ+GTy0pP185beGg7Llih/NSHSV2XAs1lnznocSg==",
      "dependencies": {
        "@nodelib/fs.scandir": "2.1.5",
        "fastq": "^1.6.0"
      },
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/@pkgjs/parseargs": {
      "version": "0.11.0",
      "resolved": "https://registry.npmjs.org/@pkgjs/parseargs/-/parseargs-0.11.0.tgz",
      "integrity": "sha512-+1VkjdD0QBLPodGrJUeqarH8VAIvQODIbwh9XpP5Syisf7YoQgsJKPNFoqqLQlu+VQ/tVSshMR6loPMn8U+dPg==",
      "optional": true,
      "engines": {
        "node": ">=14"
      }
    },
    "node_modules/@prettier/plugin-xml": {
      "version": "2.2.0",
      "resolved": "https://registry.npmjs.org/@prettier/plugin-xml/-/plugin-xml-2.2.0.tgz",
      "integrity": "sha512-UWRmygBsyj4bVXvDiqSccwT1kmsorcwQwaIy30yVh8T+Gspx4OlC0shX1y+ZuwXZvgnafmpRYKks0bAu9urJew==",
      "dependencies": {
        "@xml-tools/parser": "^1.0.11",
        "prettier": ">=2.4.0"
      }
    },
    "node_modules/@revenuecat/purchases-capacitor": {
      "version": "9.2.1",
      "resolved": "https://registry.npmjs.org/@revenuecat/purchases-capacitor/-/purchases-capacitor-9.2.1.tgz",
      "integrity": "sha512-Cfm5cTHYKAtp53k0g49c9jfv7j8mcv5frX4MN9hPUKrJ5Gsxs1GXj+OwCQna7QEH/w9UETSH4X0TEzREXRtWsg==",
      "dependencies": {
        "@revenuecat/purchases-typescript-internal-esm": "13.15.2"
      },
      "peerDependencies": {
        "@capacitor/core": "^6.0.0"
      }
    },
    "node_modules/@revenuecat/purchases-typescript-internal-esm": {
      "version": "13.15.2",
      "resolved": "https://registry.npmjs.org/@revenuecat/purchases-typescript-internal-esm/-/purchases-typescript-internal-esm-13.15.2.tgz",
      "integrity": "sha512-dM4oNpjP6+rdlHIYWd0REulChcllXsMBQwhhRFKKsrj2IXHnF/+5ugIscYgXQN6LMvrg038hn5fWflg2nokD5w=="
    },
    "node_modules/@rollup/rollup-android-arm-eabi": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-android-arm-eabi/-/rollup-android-arm-eabi-4.29.1.tgz",
      "integrity": "sha512-ssKhA8RNltTZLpG6/QNkCSge+7mBQGUqJRisZ2MDQcEGaK93QESEgWK2iOpIDZ7k9zPVkG5AS3ksvD5ZWxmItw==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "android"
      ]
    },
    "node_modules/@rollup/rollup-android-arm64": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-android-arm64/-/rollup-android-arm64-4.29.1.tgz",
      "integrity": "sha512-CaRfrV0cd+NIIcVVN/jx+hVLN+VRqnuzLRmfmlzpOzB87ajixsN/+9L5xNmkaUUvEbI5BmIKS+XTwXsHEb65Ew==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "android"
      ]
    },
    "node_modules/@rollup/rollup-darwin-arm64": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-darwin-arm64/-/rollup-darwin-arm64-4.29.1.tgz",
      "integrity": "sha512-2ORr7T31Y0Mnk6qNuwtyNmy14MunTAMx06VAPI6/Ju52W10zk1i7i5U3vlDRWjhOI5quBcrvhkCHyF76bI7kEw==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "darwin"
      ]
    },
    "node_modules/@rollup/rollup-darwin-x64": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-darwin-x64/-/rollup-darwin-x64-4.29.1.tgz",
      "integrity": "sha512-j/Ej1oanzPjmN0tirRd5K2/nncAhS9W6ICzgxV+9Y5ZsP0hiGhHJXZ2JQ53iSSjj8m6cRY6oB1GMzNn2EUt6Ng==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "darwin"
      ]
    },
    "node_modules/@rollup/rollup-freebsd-arm64": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-freebsd-arm64/-/rollup-freebsd-arm64-4.29.1.tgz",
      "integrity": "sha512-91C//G6Dm/cv724tpt7nTyP+JdN12iqeXGFM1SqnljCmi5yTXriH7B1r8AD9dAZByHpKAumqP1Qy2vVNIdLZqw==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "freebsd"
      ]
    },
    "node_modules/@rollup/rollup-freebsd-x64": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-freebsd-x64/-/rollup-freebsd-x64-4.29.1.tgz",
      "integrity": "sha512-hEioiEQ9Dec2nIRoeHUP6hr1PSkXzQaCUyqBDQ9I9ik4gCXQZjJMIVzoNLBRGet+hIUb3CISMh9KXuCcWVW/8w==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "freebsd"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm-gnueabihf": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm-gnueabihf/-/rollup-linux-arm-gnueabihf-4.29.1.tgz",
      "integrity": "sha512-Py5vFd5HWYN9zxBv3WMrLAXY3yYJ6Q/aVERoeUFwiDGiMOWsMs7FokXihSOaT/PMWUty/Pj60XDQndK3eAfE6A==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm-musleabihf": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm-musleabihf/-/rollup-linux-arm-musleabihf-4.29.1.tgz",
      "integrity": "sha512-RiWpGgbayf7LUcuSNIbahr0ys2YnEERD4gYdISA06wa0i8RALrnzflh9Wxii7zQJEB2/Eh74dX4y/sHKLWp5uQ==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm64-gnu": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm64-gnu/-/rollup-linux-arm64-gnu-4.29.1.tgz",
      "integrity": "sha512-Z80O+taYxTQITWMjm/YqNoe9d10OX6kDh8X5/rFCMuPqsKsSyDilvfg+vd3iXIqtfmp+cnfL1UrYirkaF8SBZA==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm64-musl": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm64-musl/-/rollup-linux-arm64-musl-4.29.1.tgz",
      "integrity": "sha512-fOHRtF9gahwJk3QVp01a/GqS4hBEZCV1oKglVVq13kcK3NeVlS4BwIFzOHDbmKzt3i0OuHG4zfRP0YoG5OF/rA==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-loongarch64-gnu": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-loongarch64-gnu/-/rollup-linux-loongarch64-gnu-4.29.1.tgz",
      "integrity": "sha512-5a7q3tnlbcg0OodyxcAdrrCxFi0DgXJSoOuidFUzHZ2GixZXQs6Tc3CHmlvqKAmOs5eRde+JJxeIf9DonkmYkw==",
      "cpu": [
        "loong64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-powerpc64le-gnu": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-powerpc64le-gnu/-/rollup-linux-powerpc64le-gnu-4.29.1.tgz",
      "integrity": "sha512-9b4Mg5Yfz6mRnlSPIdROcfw1BU22FQxmfjlp/CShWwO3LilKQuMISMTtAu/bxmmrE6A902W2cZJuzx8+gJ8e9w==",
      "cpu": [
        "ppc64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-riscv64-gnu": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-riscv64-gnu/-/rollup-linux-riscv64-gnu-4.29.1.tgz",
      "integrity": "sha512-G5pn0NChlbRM8OJWpJFMX4/i8OEU538uiSv0P6roZcbpe/WfhEO+AT8SHVKfp8qhDQzaz7Q+1/ixMy7hBRidnQ==",
      "cpu": [
        "riscv64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-s390x-gnu": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-s390x-gnu/-/rollup-linux-s390x-gnu-4.29.1.tgz",
      "integrity": "sha512-WM9lIkNdkhVwiArmLxFXpWndFGuOka4oJOZh8EP3Vb8q5lzdSCBuhjavJsw68Q9AKDGeOOIHYzYm4ZFvmWez5g==",
      "cpu": [
        "s390x"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-x64-gnu": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-x64-gnu/-/rollup-linux-x64-gnu-4.29.1.tgz",
      "integrity": "sha512-87xYCwb0cPGZFoGiErT1eDcssByaLX4fc0z2nRM6eMtV9njAfEE6OW3UniAoDhX4Iq5xQVpE6qO9aJbCFumKYQ==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-x64-musl": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-x64-musl/-/rollup-linux-x64-musl-4.29.1.tgz",
      "integrity": "sha512-xufkSNppNOdVRCEC4WKvlR1FBDyqCSCpQeMMgv9ZyXqqtKBfkw1yfGMTUTs9Qsl6WQbJnsGboWCp7pJGkeMhKA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-win32-arm64-msvc": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-win32-arm64-msvc/-/rollup-win32-arm64-msvc-4.29.1.tgz",
      "integrity": "sha512-F2OiJ42m77lSkizZQLuC+jiZ2cgueWQL5YC9tjo3AgaEw+KJmVxHGSyQfDUoYR9cci0lAywv2Clmckzulcq6ig==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@rollup/rollup-win32-ia32-msvc": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-win32-ia32-msvc/-/rollup-win32-ia32-msvc-4.29.1.tgz",
      "integrity": "sha512-rYRe5S0FcjlOBZQHgbTKNrqxCBUmgDJem/VQTCcTnA2KCabYSWQDrytOzX7avb79cAAweNmMUb/Zw18RNd4mng==",
      "cpu": [
        "ia32"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@rollup/rollup-win32-x64-msvc": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-win32-x64-msvc/-/rollup-win32-x64-msvc-4.29.1.tgz",
      "integrity": "sha512-+10CMg9vt1MoHj6x1pxyjPSMjHTIlqs8/tBztXvPAx24SKs9jwVnKqHJumlH/IzhaPUaj3T6T6wfZr8okdXaIg==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@trapezedev/gradle-parse": {
      "version": "7.1.3",
      "resolved": "https://registry.npmjs.org/@trapezedev/gradle-parse/-/gradle-parse-7.1.3.tgz",
      "integrity": "sha512-WQVF5pEJ5o/mUyvfGTG9nBKx9Te/ilKM3r2IT69GlbaooItT5ao7RyF1MUTBNjHLPk/xpGUY3c6PyVnjDlz0Vw=="
    },
    "node_modules/@trapezedev/project": {
      "version": "7.1.3",
      "resolved": "https://registry.npmjs.org/@trapezedev/project/-/project-7.1.3.tgz",
      "integrity": "sha512-GANh8Ey73MechZrryfJoILY9hBnWqzS6AdB53zuWBCBbaiImyblXT41fWdN6pB2f5+cNI2FAUxGfVhl+LeEVbQ==",
      "dependencies": {
        "@ionic/utils-fs": "^3.1.5",
        "@ionic/utils-subprocess": "^2.1.8",
        "@prettier/plugin-xml": "^2.2.0",
        "@trapezedev/gradle-parse": "7.1.3",
        "@xmldom/xmldom": "^0.7.5",
        "conventional-changelog": "^3.1.4",
        "cross-spawn": "^7.0.3",
        "diff": "^5.1.0",
        "env-paths": "^3.0.0",
        "gradle-to-js": "^2.0.0",
        "ini": "^2.0.0",
        "kleur": "^4.1.5",
        "lodash": "^4.17.21",
        "mergexml": "^1.2.3",
        "plist": "^3.0.4",
        "prettier": "^2.7.1",
        "prompts": "^2.4.2",
        "replace": "^1.1.0",
        "tempy": "^1.0.1",
        "tmp": "^0.2.1",
        "ts-node": "^10.2.1",
        "xcode": "^3.0.1",
        "xml-js": "^1.6.11",
        "xpath": "^0.0.32",
        "yargs": "^17.2.1"
      }
    },
    "node_modules/@trapezedev/project/node_modules/env-paths": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/env-paths/-/env-paths-3.0.0.tgz",
      "integrity": "sha512-dtJUTepzMW3Lm/NPxRf3wP4642UWhjL2sQxc+ym2YMj1m/H2zDNQOlezafzkHwn6sMstjHTwG6iQQsctDW/b1A==",
      "engines": {
        "node": "^12.20.0 || ^14.13.1 || >=16.0.0"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/@tsconfig/node10": {
      "version": "1.0.11",
      "resolved": "https://registry.npmjs.org/@tsconfig/node10/-/node10-1.0.11.tgz",
      "integrity": "sha512-DcRjDCujK/kCk/cUe8Xz8ZSpm8mS3mNNpta+jGCA6USEDfktlNvm1+IuZ9eTcDbNk41BHwpHHeW+N1lKCz4zOw=="
    },
    "node_modules/@tsconfig/node12": {
      "version": "1.0.11",
      "resolved": "https://registry.npmjs.org/@tsconfig/node12/-/node12-1.0.11.tgz",
      "integrity": "sha512-cqefuRsh12pWyGsIoBKJA9luFu3mRxCA+ORZvA4ktLSzIuCUtWVxGIuXigEwO5/ywWFMZ2QEGKWvkZG1zDMTag=="
    },
    "node_modules/@tsconfig/node14": {
      "version": "1.0.3",
      "resolved": "https://registry.npmjs.org/@tsconfig/node14/-/node14-1.0.3.tgz",
      "integrity": "sha512-ysT8mhdixWK6Hw3i1V2AeRqZ5WfXg1G43mqoYlM2nc6388Fq5jcXyr5mRsqViLx/GJYdoL0bfXD8nmF+Zn/Iow=="
    },
    "node_modules/@tsconfig/node16": {
      "version": "1.0.4",
      "resolved": "https://registry.npmjs.org/@tsconfig/node16/-/node16-1.0.4.tgz",
      "integrity": "sha512-vxhUy4J8lyeyinH7Azl1pdd43GJhZH/tP2weN8TntQblOY+A0XbT8DJk1/oCPuOOyg/Ja757rG0CgHcWC8OfMA=="
    },
    "node_modules/@types/estree": {
      "version": "1.0.6",
      "resolved": "https://registry.npmjs.org/@types/estree/-/estree-1.0.6.tgz",
      "integrity": "sha512-AYnb1nQyY49te+VRAVgmzfcgjYS91mY5P0TKUDCLEM+gNnA+3T6rWITXRLYCpahpqSQbN5cE+gHpnPyXjHWxcw==",
      "dev": true
    },
    "node_modules/@types/fs-extra": {
      "version": "8.1.5",
      "resolved": "https://registry.npmjs.org/@types/fs-extra/-/fs-extra-8.1.5.tgz",
      "integrity": "sha512-0dzKcwO+S8s2kuF5Z9oUWatQJj5Uq/iqphEtE3GQJVRRYm/tD1LglU2UnXi2A8jLq5umkGouOXOR9y0n613ZwQ==",
      "dependencies": {
        "@types/node": "*"
      }
    },
    "node_modules/@types/minimist": {
      "version": "1.2.5",
      "resolved": "https://registry.npmjs.org/@types/minimist/-/minimist-1.2.5.tgz",
      "integrity": "sha512-hov8bUuiLiyFPGyFPE1lwWhmzYbirOXQNNo40+y3zow8aFVTeyn3VWL0VFFfdNddA8S4Vf0Tc062rzyNr7Paag=="
    },
    "node_modules/@types/node": {
      "version": "22.10.2",
      "resolved": "https://registry.npmjs.org/@types/node/-/node-22.10.2.tgz",
      "integrity": "sha512-Xxr6BBRCAOQixvonOye19wnzyDiUtTeqldOOmj3CkeblonbccA12PFwlufvRdrpjXxqnmUaeiU5EOA+7s5diUQ==",
      "dependencies": {
        "undici-types": "~6.20.0"
      }
    },
    "node_modules/@types/normalize-package-data": {
      "version": "2.4.4",
      "resolved": "https://registry.npmjs.org/@types/normalize-package-data/-/normalize-package-data-2.4.4.tgz",
      "integrity": "sha512-37i+OaWTh9qeK4LSHPsyRC7NahnGotNuZvjLSgcPzblpHB3rrCJxAOgI5gCdKm7coonsaX1Of0ILiTcnZjbfxA=="
    },
    "node_modules/@types/slice-ansi": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/@types/slice-ansi/-/slice-ansi-4.0.0.tgz",
      "integrity": "sha512-+OpjSaq85gvlZAYINyzKpLeiFkSC4EsC6IIiT6v6TLSU5k5U83fHGj9Lel8oKEXM0HqgrMVCjXPDPVICtxF7EQ=="
    },
    "node_modules/@vue/reactivity": {
      "version": "3.1.5",
      "resolved": "https://registry.npmjs.org/@vue/reactivity/-/reactivity-3.1.5.tgz",
      "integrity": "sha512-1tdfLmNjWG6t/CsPldh+foumYFo3cpyCHgBYQ34ylaMsJ+SNHQ1kApMIa8jN+i593zQuaw3AdWH0nJTARzCFhg==",
      "dependencies": {
        "@vue/shared": "3.1.5"
      }
    },
    "node_modules/@vue/shared": {
      "version": "3.1.5",
      "resolved": "https://registry.npmjs.org/@vue/shared/-/shared-3.1.5.tgz",
      "integrity": "sha512-oJ4F3TnvpXaQwZJNF3ZK+kLPHKarDmJjJ6jyzVNDKH9md1dptjC7lWR//jrGuLdek/U6iltWxqAnYOu8gCiOvA=="
    },
    "node_modules/@xml-tools/parser": {
      "version": "1.0.11",
      "resolved": "https://registry.npmjs.org/@xml-tools/parser/-/parser-1.0.11.tgz",
      "integrity": "sha512-aKqQ077XnR+oQtHJlrAflaZaL7qZsulWc/i/ZEooar5JiWj1eLt0+Wg28cpa+XLney107wXqneC+oG1IZvxkTA==",
      "dependencies": {
        "chevrotain": "7.1.1"
      }
    },
    "node_modules/@xmldom/xmldom": {
      "version": "0.7.13",
      "resolved": "https://registry.npmjs.org/@xmldom/xmldom/-/xmldom-0.7.13.tgz",
      "integrity": "sha512-lm2GW5PkosIzccsaZIz7tp8cPADSIlIHWDFTR1N0SzfinhhYgeIQjFMz4rYzanCScr3DqQLeomUDArp6MWKm+g==",
      "deprecated": "this version is no longer supported, please update to at least 0.8.*",
      "engines": {
        "node": ">=10.0.0"
      }
    },
    "node_modules/@zeit/schemas": {
      "version": "2.36.0",
      "resolved": "https://registry.npmjs.org/@zeit/schemas/-/schemas-2.36.0.tgz",
      "integrity": "sha512-7kjMwcChYEzMKjeex9ZFXkt1AyNov9R5HZtjBKVsmVpw7pa7ZtlCGvCBC2vnnXctaYN+aRI61HjIqeetZW5ROg==",
      "dev": true
    },
    "node_modules/accepts": {
      "version": "1.3.8",
      "resolved": "https://registry.npmjs.org/accepts/-/accepts-1.3.8.tgz",
      "integrity": "sha512-PYAthTa2m2VKxuvSD3DPC/Gy+U+sOA1LAuT8mkmRuvw+NACSaeXEQ+NHcVF7rONl6qcaxV3Uuemwawk+7+SJLw==",
      "dev": true,
      "dependencies": {
        "mime-types": "~2.1.34",
        "negotiator": "0.6.3"
      },
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/acorn": {
      "version": "8.14.0",
      "resolved": "https://registry.npmjs.org/acorn/-/acorn-8.14.0.tgz",
      "integrity": "sha512-cl669nCJTZBsL97OF4kUQm5g5hC2uihk0NxY3WENAC0TYdILVkAyHymAntgxGkl7K+t0cXIrH5siy5S4XkFycA==",
      "bin": {
        "acorn": "bin/acorn"
      },
      "engines": {
        "node": ">=0.4.0"
      }
    },
    "node_modules/acorn-walk": {
      "version": "8.3.4",
      "resolved": "https://registry.npmjs.org/acorn-walk/-/acorn-walk-8.3.4.tgz",
      "integrity": "sha512-ueEepnujpqee2o5aIYnvHU6C0A42MNdsIDeqy5BydrkuC5R1ZuUFnm27EeFJGoEHJQgn3uleRvmTXaJgfXbt4g==",
      "dependencies": {
        "acorn": "^8.11.0"
      },
      "engines": {
        "node": ">=0.4.0"
      }
    },
    "node_modules/add-stream": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/add-stream/-/add-stream-1.0.0.tgz",
      "integrity": "sha512-qQLMr+8o0WC4FZGQTcJiKBVC59JylcPSrTtk6usvmIDFUOCKegapy1VHQwRbFMOFyb/inzUVqHs+eMYKDM1YeQ=="
    },
    "node_modules/aggregate-error": {
      "version": "3.1.0",
      "resolved": "https://registry.npmjs.org/aggregate-error/-/aggregate-error-3.1.0.tgz",
      "integrity": "sha512-4I7Td01quW/RpocfNayFdFVk1qSuoh0E7JrbRJ16nH01HhKFQ88INq9Sd+nd72zqRySlr9BmDA8xlEJ6vJMrYA==",
      "dependencies": {
        "clean-stack": "^2.0.0",
        "indent-string": "^4.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/ajv": {
      "version": "8.12.0",
      "resolved": "https://registry.npmjs.org/ajv/-/ajv-8.12.0.tgz",
      "integrity": "sha512-sRu1kpcO9yLtYxBKvqfTeh9KzZEwO3STyX1HT+4CaDzC6HpTGYhIhPIzj9XuKU7KYDwnaeh5hcOwjy1QuJzBPA==",
      "dev": true,
      "dependencies": {
        "fast-deep-equal": "^3.1.1",
        "json-schema-traverse": "^1.0.0",
        "require-from-string": "^2.0.2",
        "uri-js": "^4.2.2"
      },
      "funding": {
        "type": "github",
        "url": "https://github.com/sponsors/epoberezkin"
      }
    },
    "node_modules/alpinejs": {
      "version": "3.14.7",
      "resolved": "https://registry.npmjs.org/alpinejs/-/alpinejs-3.14.7.tgz",
      "integrity": "sha512-ScnbydNBcWVnCiVupD3wWUvoMPm8244xkvDNMxVCspgmap9m4QuJ7pjc+77UtByU+1+Ejg0wzYkP4mQaOMcvng==",
      "dependencies": {
        "@vue/reactivity": "~3.1.1"
      }
    },
    "node_modules/animate.css": {
      "version": "4.1.1",
      "resolved": "https://registry.npmjs.org/animate.css/-/animate.css-4.1.1.tgz",
      "integrity": "sha512-+mRmCTv6SbCmtYJCN4faJMNFVNN5EuCTTprDTAo7YzIGji2KADmakjVA3+8mVDkZ2Bf09vayB35lSQIex2+QaQ=="
    },
    "node_modules/ansi-align": {
      "version": "3.0.1",
      "resolved": "https://registry.npmjs.org/ansi-align/-/ansi-align-3.0.1.tgz",
      "integrity": "sha512-IOfwwBF5iczOjp/WeY4YxyjqAFMQoZufdQWDd19SEExbVLNXqvpzSJ/M7Za4/sCPmQ0+GRquoA7bGcINcxew6w==",
      "dev": true,
      "dependencies": {
        "string-width": "^4.1.0"
      }
    },
    "node_modules/ansi-regex": {
      "version": "5.0.1",
      "resolved": "https://registry.npmjs.org/ansi-regex/-/ansi-regex-5.0.1.tgz",
      "integrity": "sha512-quJQXlTSUGL2LH9SUXo8VwsY4soanhgo6LNSm84E1LBcE8s3O0wpdiRzyR9z/ZZJMlMWv37qOOb9pdJlMUEKFQ==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/ansi-styles": {
      "version": "4.3.0",
      "resolved": "https://registry.npmjs.org/ansi-styles/-/ansi-styles-4.3.0.tgz",
      "integrity": "sha512-zbB9rCJAT1rbjiVDb2hqKFHNYLxgtk8NURxZ3IZwD3F6NtxbXZQCnnSi1Lkx+IDohdPlFp222wVALIheZJQSEg==",
      "dependencies": {
        "color-convert": "^2.0.1"
      },
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/chalk/ansi-styles?sponsor=1"
      }
    },
    "node_modules/any-promise": {
      "version": "1.3.0",
      "resolved": "https://registry.npmjs.org/any-promise/-/any-promise-1.3.0.tgz",
      "integrity": "sha512-7UvmKalWRt1wgjL1RrGxoSJW/0QZFIegpeGvZG9kjp8vrRu55XTHbwnqq2GpXm9uLbcuhxm3IqX9OB4MZR1b2A=="
    },
    "node_modules/anymatch": {
      "version": "3.1.3",
      "resolved": "https://registry.npmjs.org/anymatch/-/anymatch-3.1.3.tgz",
      "integrity": "sha512-KMReFUr0B4t+D+OBkjR3KYqvocp2XaSzO55UcB6mgQMd3KbcE+mWTyvVV7D/zsdEbNnV6acZUutkiHQXvTr1Rw==",
      "dependencies": {
        "normalize-path": "^3.0.0",
        "picomatch": "^2.0.4"
      },
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/arch": {
      "version": "2.2.0",
      "resolved": "https://registry.npmjs.org/arch/-/arch-2.2.0.tgz",
      "integrity": "sha512-Of/R0wqp83cgHozfIYLbBMnej79U/SVGOOyuB3VVFv1NRM/PSFMK12x9KVtiYzJqmnU5WR2qp0Z5rHb7sWGnFQ==",
      "dev": true,
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/feross"
        },
        {
          "type": "patreon",
          "url": "https://www.patreon.com/feross"
        },
        {
          "type": "consulting",
          "url": "https://feross.org/support"
        }
      ]
    },
    "node_modules/arg": {
      "version": "5.0.2",
      "resolved": "https://registry.npmjs.org/arg/-/arg-5.0.2.tgz",
      "integrity": "sha512-PYjyFOLKQ9y57JvQ6QLo8dAgNqswh8M1RMJYdQduT6xbWSgK36P/Z/v+p888pM69jMMfS8Xd8F6I1kQ/I9HUGg=="
    },
    "node_modules/array-ify": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/array-ify/-/array-ify-1.0.0.tgz",
      "integrity": "sha512-c5AMf34bKdvPhQ7tBGhqkgKNUzMr4WUs+WDtC2ZUGOUncbxKMTvqxYctiseW3+L4bA8ec+GcZ6/A/FW4m8ukng=="
    },
    "node_modules/array-union": {
      "version": "2.1.0",
      "resolved": "https://registry.npmjs.org/array-union/-/array-union-2.1.0.tgz",
      "integrity": "sha512-HGyxoOTYUyCM6stUe6EJgnd4EoewAI7zMdfqO+kGjnlZmBDz/cR5pf8r/cR4Wq60sL/p0IkcjUEEPwS3GFrIyw==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/arrify": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/arrify/-/arrify-1.0.1.tgz",
      "integrity": "sha512-3CYzex9M9FGQjCGMGyi6/31c8GJbgb0qGyrx5HWxPd0aCwh4cB2YjMb2Xf9UuoogrMrlO9cTqnB5rI5GHZTcUA==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/asap": {
      "version": "2.0.6",
      "resolved": "https://registry.npmjs.org/asap/-/asap-2.0.6.tgz",
      "integrity": "sha512-BSHWgDSAiKs50o2Re8ppvp3seVHXSRM44cdSsT9FfNEUUZLOGWVCsiWaRPWM1Znn+mqZ1OfVZ3z3DWEzSp7hRA=="
    },
    "node_modules/astral-regex": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/astral-regex/-/astral-regex-2.0.0.tgz",
      "integrity": "sha512-Z7tMw1ytTXt5jqMcOP+OQteU1VuNK9Y02uuJtKQ1Sv69jXQKKg5cibLwGJow8yzZP+eAc18EmLGPal0bp36rvQ==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/at-least-node": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/at-least-node/-/at-least-node-1.0.0.tgz",
      "integrity": "sha512-+q/t7Ekv1EDY2l6Gda6LLiX14rU9TV20Wa3ofeQmwPFZbOMo9DXrLbOjFaaclkXKWidIaopwAObQDqwWtGUjqg==",
      "engines": {
        "node": ">= 4.0.0"
      }
    },
    "node_modules/autoprefixer": {
      "version": "10.4.20",
      "resolved": "https://registry.npmjs.org/autoprefixer/-/autoprefixer-10.4.20.tgz",
      "integrity": "sha512-XY25y5xSv/wEoqzDyXXME4AFfkZI0P23z6Fs3YgymDnKJkCGOnkL0iTxCa85UTqaSgfcqyf3UA6+c7wUvx/16g==",
      "dev": true,
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/postcss/"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/autoprefixer"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "dependencies": {
        "browserslist": "^4.23.3",
        "caniuse-lite": "^1.0.30001646",
        "fraction.js": "^4.3.7",
        "normalize-range": "^0.1.2",
        "picocolors": "^1.0.1",
        "postcss-value-parser": "^4.2.0"
      },
      "bin": {
        "autoprefixer": "bin/autoprefixer"
      },
      "engines": {
        "node": "^10 || ^12 || >=14"
      },
      "peerDependencies": {
        "postcss": "^8.1.0"
      }
    },
    "node_modules/b4a": {
      "version": "1.6.7",
      "resolved": "https://registry.npmjs.org/b4a/-/b4a-1.6.7.tgz",
      "integrity": "sha512-OnAYlL5b7LEkALw87fUVafQw5rVR9RjwGd4KUwNQ6DrrNmaVaUCgLipfVlzrPQ4tWOR9P0IXGNOx50jYCCdSJg=="
    },
    "node_modules/balanced-match": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/balanced-match/-/balanced-match-1.0.2.tgz",
      "integrity": "sha512-3oSeUO0TMV67hN1AmbXsK4yaqU7tjiHlbxRDZOpH0KW9+CeX4bRAaX0Anxt0tx2MrpRpWwQaPwIlISEJhYU5Pw=="
    },
    "node_modules/bare-events": {
      "version": "2.5.4",
      "resolved": "https://registry.npmjs.org/bare-events/-/bare-events-2.5.4.tgz",
      "integrity": "sha512-+gFfDkR8pj4/TrWCGUGWmJIkBwuxPS5F+a5yWjOHQt2hHvNZd5YLzadjmDUtFmMM4y429bnKLa8bYBMHcYdnQA==",
      "optional": true
    },
    "node_modules/bare-fs": {
      "version": "4.0.1",
      "resolved": "https://registry.npmjs.org/bare-fs/-/bare-fs-4.0.1.tgz",
      "integrity": "sha512-ilQs4fm/l9eMfWY2dY0WCIUplSUp7U0CT1vrqMg1MUdeZl4fypu5UP0XcDBK5WBQPJAKP1b7XEodISmekH/CEg==",
      "optional": true,
      "dependencies": {
        "bare-events": "^2.0.0",
        "bare-path": "^3.0.0",
        "bare-stream": "^2.0.0"
      },
      "engines": {
        "bare": ">=1.7.0"
      }
    },
    "node_modules/bare-os": {
      "version": "3.4.0",
      "resolved": "https://registry.npmjs.org/bare-os/-/bare-os-3.4.0.tgz",
      "integrity": "sha512-9Ous7UlnKbe3fMi7Y+qh0DwAup6A1JkYgPnjvMDNOlmnxNRQvQ/7Nst+OnUQKzk0iAT0m9BisbDVp9gCv8+ETA==",
      "optional": true,
      "engines": {
        "bare": ">=1.6.0"
      }
    },
    "node_modules/bare-path": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/bare-path/-/bare-path-3.0.0.tgz",
      "integrity": "sha512-tyfW2cQcB5NN8Saijrhqn0Zh7AnFNsnczRcuWODH0eYAXBsJ5gVxAUuNr7tsHSC6IZ77cA0SitzT+s47kot8Mw==",
      "optional": true,
      "dependencies": {
        "bare-os": "^3.0.1"
      }
    },
    "node_modules/bare-stream": {
      "version": "2.6.5",
      "resolved": "https://registry.npmjs.org/bare-stream/-/bare-stream-2.6.5.tgz",
      "integrity": "sha512-jSmxKJNJmHySi6hC42zlZnq00rga4jjxcgNZjY9N5WlOe/iOoGRtdwGsHzQv2RlH2KOYMwGUXhf2zXd32BA9RA==",
      "optional": true,
      "dependencies": {
        "streamx": "^2.21.0"
      },
      "peerDependencies": {
        "bare-buffer": "*",
        "bare-events": "*"
      },
      "peerDependenciesMeta": {
        "bare-buffer": {
          "optional": true
        },
        "bare-events": {
          "optional": true
        }
      }
    },
    "node_modules/base64-js": {
      "version": "1.5.1",
      "resolved": "https://registry.npmjs.org/base64-js/-/base64-js-1.5.1.tgz",
      "integrity": "sha512-AKpaYlHn8t4SVbOHCy+b5+KKgvR4vrsD8vbvrbiQJps7fKDTkjkDry6ji0rUJjC0kzbNePLwzxq8iypo41qeWA==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/feross"
        },
        {
          "type": "patreon",
          "url": "https://www.patreon.com/feross"
        },
        {
          "type": "consulting",
          "url": "https://feross.org/support"
        }
      ]
    },
    "node_modules/big-integer": {
      "version": "1.6.52",
      "resolved": "https://registry.npmjs.org/big-integer/-/big-integer-1.6.52.tgz",
      "integrity": "sha512-QxD8cf2eVqJOOz63z6JIN9BzvVs/dlySa5HGSBH5xtR8dPteIRQnBxxKqkNTiT6jbDTF6jAfrd4oMcND9RGbQg==",
      "engines": {
        "node": ">=0.6"
      }
    },
    "node_modules/binary-extensions": {
      "version": "2.3.0",
      "resolved": "https://registry.npmjs.org/binary-extensions/-/binary-extensions-2.3.0.tgz",
      "integrity": "sha512-Ceh+7ox5qe7LJuLHoY0feh3pHuUDHAcRUeyL2VYghZwfpkNIy/+8Ocg0a3UuSoYzavmylwuLWQOf3hl0jjMMIw==",
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/bl": {
      "version": "4.1.0",
      "resolved": "https://registry.npmjs.org/bl/-/bl-4.1.0.tgz",
      "integrity": "sha512-1W07cM9gS6DcLperZfFSj+bWLtaPGSOHWhPiGzXmvVJbRLdG82sH/Kn8EtW1VqWVA54AKf2h5k5BbnIbwF3h6w==",
      "dependencies": {
        "buffer": "^5.5.0",
        "inherits": "^2.0.4",
        "readable-stream": "^3.4.0"
      }
    },
    "node_modules/boolbase": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/boolbase/-/boolbase-1.0.0.tgz",
      "integrity": "sha512-JZOSA7Mo9sNGB8+UjSgzdLtokWAky1zbztM3WRLCbZ70/3cTANmQmOdR7y2g+J0e2WXywy1yS468tY+IruqEww=="
    },
    "node_modules/boxen": {
      "version": "7.0.0",
      "resolved": "https://registry.npmjs.org/boxen/-/boxen-7.0.0.tgz",
      "integrity": "sha512-j//dBVuyacJbvW+tvZ9HuH03fZ46QcaKvvhZickZqtB271DxJ7SNRSNxrV/dZX0085m7hISRZWbzWlJvx/rHSg==",
      "dev": true,
      "dependencies": {
        "ansi-align": "^3.0.1",
        "camelcase": "^7.0.0",
        "chalk": "^5.0.1",
        "cli-boxes": "^3.0.0",
        "string-width": "^5.1.2",
        "type-fest": "^2.13.0",
        "widest-line": "^4.0.1",
        "wrap-ansi": "^8.0.1"
      },
      "engines": {
        "node": ">=14.16"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/boxen/node_modules/ansi-regex": {
      "version": "6.1.0",
      "resolved": "https://registry.npmjs.org/ansi-regex/-/ansi-regex-6.1.0.tgz",
      "integrity": "sha512-7HSX4QQb4CspciLpVFwyRe79O3xsIZDDLER21kERQ71oaPodF8jL725AgJMFAYbooIqolJoRLuM81SpeUkpkvA==",
      "dev": true,
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/ansi-regex?sponsor=1"
      }
    },
    "node_modules/boxen/node_modules/ansi-styles": {
      "version": "6.2.1",
      "resolved": "https://registry.npmjs.org/ansi-styles/-/ansi-styles-6.2.1.tgz",
      "integrity": "sha512-bN798gFfQX+viw3R7yrGWRqnrN2oRkEkUjjl4JNn4E8GxxbjtG3FbrEIIY3l8/hrwUwIeCZvi4QuOTP4MErVug==",
      "dev": true,
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/ansi-styles?sponsor=1"
      }
    },
    "node_modules/boxen/node_modules/emoji-regex": {
      "version": "9.2.2",
      "resolved": "https://registry.npmjs.org/emoji-regex/-/emoji-regex-9.2.2.tgz",
      "integrity": "sha512-L18DaJsXSUk2+42pv8mLs5jJT2hqFkFE4j21wOmgbUqsZ2hL72NsUU785g9RXgo3s0ZNgVl42TiHp3ZtOv/Vyg==",
      "dev": true
    },
    "node_modules/boxen/node_modules/string-width": {
      "version": "5.1.2",
      "resolved": "https://registry.npmjs.org/string-width/-/string-width-5.1.2.tgz",
      "integrity": "sha512-HnLOCR3vjcY8beoNLtcjZ5/nxn2afmME6lhrDrebokqMap+XbeW8n9TXpPDOqdGK5qcI3oT0GKTW6wC7EMiVqA==",
      "dev": true,
      "dependencies": {
        "eastasianwidth": "^0.2.0",
        "emoji-regex": "^9.2.2",
        "strip-ansi": "^7.0.1"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/boxen/node_modules/strip-ansi": {
      "version": "7.1.0",
      "resolved": "https://registry.npmjs.org/strip-ansi/-/strip-ansi-7.1.0.tgz",
      "integrity": "sha512-iq6eVVI64nQQTRYq2KtEg2d2uU7LElhTJwsH4YzIHZshxlgZms/wIc4VoDQTlG/IvVIrBKG06CrZnp0qv7hkcQ==",
      "dev": true,
      "dependencies": {
        "ansi-regex": "^6.0.1"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/strip-ansi?sponsor=1"
      }
    },
    "node_modules/boxen/node_modules/wrap-ansi": {
      "version": "8.1.0",
      "resolved": "https://registry.npmjs.org/wrap-ansi/-/wrap-ansi-8.1.0.tgz",
      "integrity": "sha512-si7QWI6zUMq56bESFvagtmzMdGOtoxfR+Sez11Mobfc7tm+VkUckk9bW2UeffTGVUbOksxmSw0AA2gs8g71NCQ==",
      "dev": true,
      "dependencies": {
        "ansi-styles": "^6.1.0",
        "string-width": "^5.0.1",
        "strip-ansi": "^7.0.1"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/wrap-ansi?sponsor=1"
      }
    },
    "node_modules/bplist-creator": {
      "version": "0.1.0",
      "resolved": "https://registry.npmjs.org/bplist-creator/-/bplist-creator-0.1.0.tgz",
      "integrity": "sha512-sXaHZicyEEmY86WyueLTQesbeoH/mquvarJaQNbjuOQO+7gbFcDEWqKmcWA4cOTLzFlfgvkiVxolk1k5bBIpmg==",
      "dependencies": {
        "stream-buffers": "2.2.x"
      }
    },
    "node_modules/bplist-parser": {
      "version": "0.3.2",
      "resolved": "https://registry.npmjs.org/bplist-parser/-/bplist-parser-0.3.2.tgz",
      "integrity": "sha512-apC2+fspHGI3mMKj+dGevkGo/tCqVB8jMb6i+OX+E29p0Iposz07fABkRIfVUPNd5A5VbuOz1bZbnmkKLYF+wQ==",
      "dependencies": {
        "big-integer": "1.6.x"
      },
      "engines": {
        "node": ">= 5.10.0"
      }
    },
    "node_modules/brace-expansion": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/brace-expansion/-/brace-expansion-2.0.1.tgz",
      "integrity": "sha512-XnAIvQ8eM+kC6aULx6wuQiwVsnzsi9d3WxzV3FpWTGA19F621kwdbsAcFKXgKUHZWsy+mY6iL1sHTxWEFCytDA==",
      "dependencies": {
        "balanced-match": "^1.0.0"
      }
    },
    "node_modules/braces": {
      "version": "3.0.3",
      "resolved": "https://registry.npmjs.org/braces/-/braces-3.0.3.tgz",
      "integrity": "sha512-yQbXgO/OSZVD2IsiLlro+7Hf6Q18EJrKSEsdoMzKePKXct3gvD8oLcOQdIzGupr5Fj+EDe8gO/lxc1BzfMpxvA==",
      "dependencies": {
        "fill-range": "^7.1.1"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/browserslist": {
      "version": "4.24.3",
      "resolved": "https://registry.npmjs.org/browserslist/-/browserslist-4.24.3.tgz",
      "integrity": "sha512-1CPmv8iobE2fyRMV97dAcMVegvvWKxmq94hkLiAkUGwKVTyDLw33K+ZxiFrREKmmps4rIw6grcCFCnTMSZ/YiA==",
      "dev": true,
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/browserslist"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/browserslist"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "dependencies": {
        "caniuse-lite": "^1.0.30001688",
        "electron-to-chromium": "^1.5.73",
        "node-releases": "^2.0.19",
        "update-browserslist-db": "^1.1.1"
      },
      "bin": {
        "browserslist": "cli.js"
      },
      "engines": {
        "node": "^6 || ^7 || ^8 || ^9 || ^10 || ^11 || ^12 || >=13.7"
      }
    },
    "node_modules/buffer": {
      "version": "5.7.1",
      "resolved": "https://registry.npmjs.org/buffer/-/buffer-5.7.1.tgz",
      "integrity": "sha512-EHcyIPBQ4BSGlvjB16k5KgAJ27CIsHY/2JBmCRReo48y9rQ3MaUzWX3KVlBa4U7MyX02HdVj0K7C3WaB3ju7FQ==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/feross"
        },
        {
          "type": "patreon",
          "url": "https://www.patreon.com/feross"
        },
        {
          "type": "consulting",
          "url": "https://feross.org/support"
        }
      ],
      "dependencies": {
        "base64-js": "^1.3.1",
        "ieee754": "^1.1.13"
      }
    },
    "node_modules/buffer-crc32": {
      "version": "0.2.13",
      "resolved": "https://registry.npmjs.org/buffer-crc32/-/buffer-crc32-0.2.13.tgz",
      "integrity": "sha512-VO9Ht/+p3SN7SKWqcrgEzjGbRSJYTx+Q1pTQC0wrWqHx0vpJraQ6GtHx8tvcg1rlK1byhU5gccxgOgj7B0TDkQ==",
      "engines": {
        "node": "*"
      }
    },
    "node_modules/buffer-from": {
      "version": "1.1.2",
      "resolved": "https://registry.npmjs.org/buffer-from/-/buffer-from-1.1.2.tgz",
      "integrity": "sha512-E+XQCRwSbaaiChtv6k6Dwgc+bx+Bs6vuKJHHl5kox/BaKbhiXzqQOwK4cO22yElGp2OCmjwVhT3HmxgyPGnJfQ==",
      "dev": true,
      "optional": true,
      "peer": true
    },
    "node_modules/bytes": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/bytes/-/bytes-3.0.0.tgz",
      "integrity": "sha512-pMhOfFDPiv9t5jjIXkHosWmkSyQbvsgEVNkz0ERHbuLh2T/7j4Mqqpz523Fe8MVY89KC6Sh/QfS2sM+SjgFDcw==",
      "dev": true,
      "engines": {
        "node": ">= 0.8"
      }
    },
    "node_modules/camelcase": {
      "version": "7.0.1",
      "resolved": "https://registry.npmjs.org/camelcase/-/camelcase-7.0.1.tgz",
      "integrity": "sha512-xlx1yCK2Oc1APsPXDL2LdlNP6+uu8OCDdhOBSVT279M/S+y75O30C2VuD8T2ogdePBBl7PfPF4504tnLgX3zfw==",
      "dev": true,
      "engines": {
        "node": ">=14.16"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/camelcase-css": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/camelcase-css/-/camelcase-css-2.0.1.tgz",
      "integrity": "sha512-QOSvevhslijgYwRx6Rv7zKdMF8lbRmx+uQGx2+vDc+KI/eBnsy9kit5aj23AgGu3pa4t9AgwbnXWqS+iOY+2aA==",
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/camelcase-keys": {
      "version": "6.2.2",
      "resolved": "https://registry.npmjs.org/camelcase-keys/-/camelcase-keys-6.2.2.tgz",
      "integrity": "sha512-YrwaA0vEKazPBkn0ipTiMpSajYDSe+KjQfrjhcBMxJt/znbvlHd8Pw/Vamaz5EB4Wfhs3SUR3Z9mwRu/P3s3Yg==",
      "dependencies": {
        "camelcase": "^5.3.1",
        "map-obj": "^4.0.0",
        "quick-lru": "^4.0.1"
      },
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/camelcase-keys/node_modules/camelcase": {
      "version": "5.3.1",
      "resolved": "https://registry.npmjs.org/camelcase/-/camelcase-5.3.1.tgz",
      "integrity": "sha512-L28STB170nwWS63UjtlEOE3dldQApaJXZkOI1uMFfzf3rRuPegHaHesyee+YxQ+W6SvRDQV6UrdOdRiR153wJg==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/caniuse-lite": {
      "version": "1.0.30001690",
      "resolved": "https://registry.npmjs.org/caniuse-lite/-/caniuse-lite-1.0.30001690.tgz",
      "integrity": "sha512-5ExiE3qQN6oF8Clf8ifIDcMRCRE/dMGcETG/XGMD8/XiXm6HXQgQTh1yZYLXXpSOsEUlJm1Xr7kGULZTuGtP/w==",
      "dev": true,
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/browserslist"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/caniuse-lite"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ]
    },
    "node_modules/chalk": {
      "version": "5.0.1",
      "resolved": "https://registry.npmjs.org/chalk/-/chalk-5.0.1.tgz",
      "integrity": "sha512-Fo07WOYGqMfCWHOzSXOt2CxDbC6skS/jO9ynEcmpANMoPrD+W1r1K6Vx7iNm+AQmETU1Xr2t+n8nzkV9t6xh3w==",
      "dev": true,
      "engines": {
        "node": "^12.17.0 || ^14.13 || >=16.0.0"
      },
      "funding": {
        "url": "https://github.com/chalk/chalk?sponsor=1"
      }
    },
    "node_modules/chalk-template": {
      "version": "0.4.0",
      "resolved": "https://registry.npmjs.org/chalk-template/-/chalk-template-0.4.0.tgz",
      "integrity": "sha512-/ghrgmhfY8RaSdeo43hNXxpoHAtxdbskUHjPpfqUWGttFgycUhYPGx3YZBCnUCvOa7Doivn1IZec3DEGFoMgLg==",
      "dev": true,
      "dependencies": {
        "chalk": "^4.1.2"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/chalk-template?sponsor=1"
      }
    },
    "node_modules/chalk-template/node_modules/chalk": {
      "version": "4.1.2",
      "resolved": "https://registry.npmjs.org/chalk/-/chalk-4.1.2.tgz",
      "integrity": "sha512-oKnbhFyRIXpUuez8iBMmyEa4nbj4IOQyuhc/wy9kY7/WVPcwIO9VA668Pu8RkO7+0G76SLROeyw9CpQ061i4mA==",
      "dev": true,
      "dependencies": {
        "ansi-styles": "^4.1.0",
        "supports-color": "^7.1.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/chalk/chalk?sponsor=1"
      }
    },
    "node_modules/chevrotain": {
      "version": "7.1.1",
      "resolved": "https://registry.npmjs.org/chevrotain/-/chevrotain-7.1.1.tgz",
      "integrity": "sha512-wy3mC1x4ye+O+QkEinVJkPf5u2vsrDIYW9G7ZuwFl6v/Yu0LwUuT2POsb+NUWApebyxfkQq6+yDfRExbnI5rcw==",
      "dependencies": {
        "regexp-to-ast": "0.5.0"
      }
    },
    "node_modules/chokidar": {
      "version": "3.6.0",
      "resolved": "https://registry.npmjs.org/chokidar/-/chokidar-3.6.0.tgz",
      "integrity": "sha512-7VT13fmjotKpGipCW9JEQAusEPE+Ei8nl6/g4FBAmIm0GOOLMua9NDDo/DWp0ZAxCr3cPq5ZpBqmPAQgDda2Pw==",
      "dependencies": {
        "anymatch": "~3.1.2",
        "braces": "~3.0.2",
        "glob-parent": "~5.1.2",
        "is-binary-path": "~2.1.0",
        "is-glob": "~4.0.1",
        "normalize-path": "~3.0.0",
        "readdirp": "~3.6.0"
      },
      "engines": {
        "node": ">= 8.10.0"
      },
      "funding": {
        "url": "https://paulmillr.com/funding/"
      },
      "optionalDependencies": {
        "fsevents": "~2.3.2"
      }
    },
    "node_modules/chokidar/node_modules/glob-parent": {
      "version": "5.1.2",
      "resolved": "https://registry.npmjs.org/glob-parent/-/glob-parent-5.1.2.tgz",
      "integrity": "sha512-AOIgSQCepiJYwP3ARnGx+5VnTu2HBYdzbGP45eLw1vr3zB3vZLeyed1sC9hnbcOc9/SrMyM5RPQrkGz4aS9Zow==",
      "dependencies": {
        "is-glob": "^4.0.1"
      },
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/chownr": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/chownr/-/chownr-2.0.0.tgz",
      "integrity": "sha512-bIomtDF5KGpdogkLd9VspvFzk9KfpyyGlS8YFVZl7TGPBHL5snIOnxeshwVgPteQ9b4Eydl+pVbIyE1DcvCWgQ==",
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/clean-stack": {
      "version": "2.2.0",
      "resolved": "https://registry.npmjs.org/clean-stack/-/clean-stack-2.2.0.tgz",
      "integrity": "sha512-4diC9HaTE+KRAMWhDhrGOECgWZxoevMc5TlkObMqNSsVU62PYzXZ/SMTjzyGAFF1YusgxGcSWTEXBhp0CPwQ1A==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/cli-boxes": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/cli-boxes/-/cli-boxes-3.0.0.tgz",
      "integrity": "sha512-/lzGpEWL/8PfI0BmBOPRwp0c/wFNX1RdUML3jK/RcSBA9T8mZDdQpqYBKtCFTOfQbwPqWEOpjqW+Fnayc0969g==",
      "dev": true,
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/clipboardy": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/clipboardy/-/clipboardy-3.0.0.tgz",
      "integrity": "sha512-Su+uU5sr1jkUy1sGRpLKjKrvEOVXgSgiSInwa/qeID6aJ07yh+5NWc3h2QfjHjBnfX4LhtFcuAWKUsJ3r+fjbg==",
      "dev": true,
      "dependencies": {
        "arch": "^2.2.0",
        "execa": "^5.1.1",
        "is-wsl": "^2.2.0"
      },
      "engines": {
        "node": "^12.20.0 || ^14.13.1 || >=16.0.0"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/cliui": {
      "version": "8.0.1",
      "resolved": "https://registry.npmjs.org/cliui/-/cliui-8.0.1.tgz",
      "integrity": "sha512-BSeNnyus75C4//NQ9gQt1/csTXyo/8Sb+afLAkzAptFuMsod9HFokGNudZpi/oQV73hnVK+sR+5PVRMd+Dr7YQ==",
      "dependencies": {
        "string-width": "^4.2.0",
        "strip-ansi": "^6.0.1",
        "wrap-ansi": "^7.0.0"
      },
      "engines": {
        "node": ">=12"
      }
    },
    "node_modules/color": {
      "version": "4.2.3",
      "resolved": "https://registry.npmjs.org/color/-/color-4.2.3.tgz",
      "integrity": "sha512-1rXeuUUiGGrykh+CeBdu5Ie7OJwinCgQY0bc7GCRxy5xVHy+moaqkpL/jqQq0MtQOeYcrqEz4abc5f0KtU7W4A==",
      "dependencies": {
        "color-convert": "^2.0.1",
        "color-string": "^1.9.0"
      },
      "engines": {
        "node": ">=12.5.0"
      }
    },
    "node_modules/color-convert": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/color-convert/-/color-convert-2.0.1.tgz",
      "integrity": "sha512-RRECPsj7iu/xb5oKYcsFHSppFNnsj/52OVTRKb4zP5onXwVF3zVmmToNcOfGC+CRDpfK/U584fMg38ZHCaElKQ==",
      "dependencies": {
        "color-name": "~1.1.4"
      },
      "engines": {
        "node": ">=7.0.0"
      }
    },
    "node_modules/color-name": {
      "version": "1.1.4",
      "resolved": "https://registry.npmjs.org/color-name/-/color-name-1.1.4.tgz",
      "integrity": "sha512-dOy+3AuW3a2wNbZHIuMZpTcgjGuLU/uBL/ubcZF9OXbDo8ff4O8yVp5Bf0efS8uEoYo5q4Fx7dY9OgQGXgAsQA=="
    },
    "node_modules/color-string": {
      "version": "1.9.1",
      "resolved": "https://registry.npmjs.org/color-string/-/color-string-1.9.1.tgz",
      "integrity": "sha512-shrVawQFojnZv6xM40anx4CkoDP+fZsw/ZerEMsW/pyzsRbElpsL/DBVW7q3ExxwusdNXI3lXpuhEZkzs8p5Eg==",
      "dependencies": {
        "color-name": "^1.0.0",
        "simple-swizzle": "^0.2.2"
      }
    },
    "node_modules/commander": {
      "version": "8.3.0",
      "resolved": "https://registry.npmjs.org/commander/-/commander-8.3.0.tgz",
      "integrity": "sha512-OkTL9umf+He2DZkUq8f8J9of7yL6RJKI24dVITBmNfZBmri9zYZQrKkuXiKhyfPSu8tUhnVBB1iKXevvnlR4Ww==",
      "engines": {
        "node": ">= 12"
      }
    },
    "node_modules/compare-func": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/compare-func/-/compare-func-2.0.0.tgz",
      "integrity": "sha512-zHig5N+tPWARooBnb0Zx1MFcdfpyJrfTJ3Y5L+IFvUm8rM74hHz66z0gw0x4tijh5CorKkKUCnW82R2vmpeCRA==",
      "dependencies": {
        "array-ify": "^1.0.0",
        "dot-prop": "^5.1.0"
      }
    },
    "node_modules/compressible": {
      "version": "2.0.18",
      "resolved": "https://registry.npmjs.org/compressible/-/compressible-2.0.18.tgz",
      "integrity": "sha512-AF3r7P5dWxL8MxyITRMlORQNaOA2IkAFaTr4k7BUumjPtRpGDTZpl0Pb1XCO6JeDCBdp126Cgs9sMxqSjgYyRg==",
      "dev": true,
      "dependencies": {
        "mime-db": ">= 1.43.0 < 2"
      },
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/compression": {
      "version": "1.7.4",
      "resolved": "https://registry.npmjs.org/compression/-/compression-1.7.4.tgz",
      "integrity": "sha512-jaSIDzP9pZVS4ZfQ+TzvtiWhdpFhE2RDHz8QJkpX9SIpLq88VueF5jJw6t+6CUQcAoA6t+x89MLrWAqpfDE8iQ==",
      "dev": true,
      "dependencies": {
        "accepts": "~1.3.5",
        "bytes": "3.0.0",
        "compressible": "~2.0.16",
        "debug": "2.6.9",
        "on-headers": "~1.0.2",
        "safe-buffer": "5.1.2",
        "vary": "~1.1.2"
      },
      "engines": {
        "node": ">= 0.8.0"
      }
    },
    "node_modules/compression/node_modules/debug": {
      "version": "2.6.9",
      "resolved": "https://registry.npmjs.org/debug/-/debug-2.6.9.tgz",
      "integrity": "sha512-bC7ElrdJaJnPbAP+1EotYvqZsb3ecl5wi6Bfi6BJTUcNowp6cvspg0jXznRTKDjm/E7AdgFBVeAPVMNcKGsHMA==",
      "dev": true,
      "dependencies": {
        "ms": "2.0.0"
      }
    },
    "node_modules/compression/node_modules/ms": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/ms/-/ms-2.0.0.tgz",
      "integrity": "sha512-Tpp60P6IUJDTuOq/5Z8cdskzJujfwqfOTkrwIwj7IRISpnkJnT6SyJ4PCPnGMoFjC9ddhal5KVIYtAt97ix05A==",
      "dev": true
    },
    "node_modules/concat-map": {
      "version": "0.0.1",
      "resolved": "https://registry.npmjs.org/concat-map/-/concat-map-0.0.1.tgz",
      "integrity": "sha512-/Srv4dswyQNBfohGpz9o6Yb3Gz3SrUDqBH5rTuhGR7ahtlbYKnVxw2bCFMRljaA7EXHaXZ8wsHdodFvbkhKmqg=="
    },
    "node_modules/content-disposition": {
      "version": "0.5.2",
      "resolved": "https://registry.npmjs.org/content-disposition/-/content-disposition-0.5.2.tgz",
      "integrity": "sha512-kRGRZw3bLlFISDBgwTSA1TMBFN6J6GWDeubmDE3AF+3+yXL8hTWv8r5rkLbqYXY4RjPk/EzHnClI3zQf1cFmHA==",
      "dev": true,
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/conventional-changelog": {
      "version": "3.1.25",
      "resolved": "https://registry.npmjs.org/conventional-changelog/-/conventional-changelog-3.1.25.tgz",
      "integrity": "sha512-ryhi3fd1mKf3fSjbLXOfK2D06YwKNic1nC9mWqybBHdObPd8KJ2vjaXZfYj1U23t+V8T8n0d7gwnc9XbIdFbyQ==",
      "dependencies": {
        "conventional-changelog-angular": "^5.0.12",
        "conventional-changelog-atom": "^2.0.8",
        "conventional-changelog-codemirror": "^2.0.8",
        "conventional-changelog-conventionalcommits": "^4.5.0",
        "conventional-changelog-core": "^4.2.1",
        "conventional-changelog-ember": "^2.0.9",
        "conventional-changelog-eslint": "^3.0.9",
        "conventional-changelog-express": "^2.0.6",
        "conventional-changelog-jquery": "^3.0.11",
        "conventional-changelog-jshint": "^2.0.9",
        "conventional-changelog-preset-loader": "^2.3.4"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-angular": {
      "version": "5.0.13",
      "resolved": "https://registry.npmjs.org/conventional-changelog-angular/-/conventional-changelog-angular-5.0.13.tgz",
      "integrity": "sha512-i/gipMxs7s8L/QeuavPF2hLnJgH6pEZAttySB6aiQLWcX3puWDL3ACVmvBhJGxnAy52Qc15ua26BufY6KpmrVA==",
      "dependencies": {
        "compare-func": "^2.0.0",
        "q": "^1.5.1"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-atom": {
      "version": "2.0.8",
      "resolved": "https://registry.npmjs.org/conventional-changelog-atom/-/conventional-changelog-atom-2.0.8.tgz",
      "integrity": "sha512-xo6v46icsFTK3bb7dY/8m2qvc8sZemRgdqLb/bjpBsH2UyOS8rKNTgcb5025Hri6IpANPApbXMg15QLb1LJpBw==",
      "dependencies": {
        "q": "^1.5.1"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-codemirror": {
      "version": "2.0.8",
      "resolved": "https://registry.npmjs.org/conventional-changelog-codemirror/-/conventional-changelog-codemirror-2.0.8.tgz",
      "integrity": "sha512-z5DAsn3uj1Vfp7po3gpt2Boc+Bdwmw2++ZHa5Ak9k0UKsYAO5mH1UBTN0qSCuJZREIhX6WU4E1p3IW2oRCNzQw==",
      "dependencies": {
        "q": "^1.5.1"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-conventionalcommits": {
      "version": "4.6.3",
      "resolved": "https://registry.npmjs.org/conventional-changelog-conventionalcommits/-/conventional-changelog-conventionalcommits-4.6.3.tgz",
      "integrity": "sha512-LTTQV4fwOM4oLPad317V/QNQ1FY4Hju5qeBIM1uTHbrnCE+Eg4CdRZ3gO2pUeR+tzWdp80M2j3qFFEDWVqOV4g==",
      "dependencies": {
        "compare-func": "^2.0.0",
        "lodash": "^4.17.15",
        "q": "^1.5.1"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-core": {
      "version": "4.2.4",
      "resolved": "https://registry.npmjs.org/conventional-changelog-core/-/conventional-changelog-core-4.2.4.tgz",
      "integrity": "sha512-gDVS+zVJHE2v4SLc6B0sLsPiloR0ygU7HaDW14aNJE1v4SlqJPILPl/aJC7YdtRE4CybBf8gDwObBvKha8Xlyg==",
      "dependencies": {
        "add-stream": "^1.0.0",
        "conventional-changelog-writer": "^5.0.0",
        "conventional-commits-parser": "^3.2.0",
        "dateformat": "^3.0.0",
        "get-pkg-repo": "^4.0.0",
        "git-raw-commits": "^2.0.8",
        "git-remote-origin-url": "^2.0.0",
        "git-semver-tags": "^4.1.1",
        "lodash": "^4.17.15",
        "normalize-package-data": "^3.0.0",
        "q": "^1.5.1",
        "read-pkg": "^3.0.0",
        "read-pkg-up": "^3.0.0",
        "through2": "^4.0.0"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-ember": {
      "version": "2.0.9",
      "resolved": "https://registry.npmjs.org/conventional-changelog-ember/-/conventional-changelog-ember-2.0.9.tgz",
      "integrity": "sha512-ulzIReoZEvZCBDhcNYfDIsLTHzYHc7awh+eI44ZtV5cx6LVxLlVtEmcO+2/kGIHGtw+qVabJYjdI5cJOQgXh1A==",
      "dependencies": {
        "q": "^1.5.1"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-eslint": {
      "version": "3.0.9",
      "resolved": "https://registry.npmjs.org/conventional-changelog-eslint/-/conventional-changelog-eslint-3.0.9.tgz",
      "integrity": "sha512-6NpUCMgU8qmWmyAMSZO5NrRd7rTgErjrm4VASam2u5jrZS0n38V7Y9CzTtLT2qwz5xEChDR4BduoWIr8TfwvXA==",
      "dependencies": {
        "q": "^1.5.1"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-express": {
      "version": "2.0.6",
      "resolved": "https://registry.npmjs.org/conventional-changelog-express/-/conventional-changelog-express-2.0.6.tgz",
      "integrity": "sha512-SDez2f3iVJw6V563O3pRtNwXtQaSmEfTCaTBPCqn0oG0mfkq0rX4hHBq5P7De2MncoRixrALj3u3oQsNK+Q0pQ==",
      "dependencies": {
        "q": "^1.5.1"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-jquery": {
      "version": "3.0.11",
      "resolved": "https://registry.npmjs.org/conventional-changelog-jquery/-/conventional-changelog-jquery-3.0.11.tgz",
      "integrity": "sha512-x8AWz5/Td55F7+o/9LQ6cQIPwrCjfJQ5Zmfqi8thwUEKHstEn4kTIofXub7plf1xvFA2TqhZlq7fy5OmV6BOMw==",
      "dependencies": {
        "q": "^1.5.1"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-jshint": {
      "version": "2.0.9",
      "resolved": "https://registry.npmjs.org/conventional-changelog-jshint/-/conventional-changelog-jshint-2.0.9.tgz",
      "integrity": "sha512-wMLdaIzq6TNnMHMy31hql02OEQ8nCQfExw1SE0hYL5KvU+JCTuPaDO+7JiogGT2gJAxiUGATdtYYfh+nT+6riA==",
      "dependencies": {
        "compare-func": "^2.0.0",
        "q": "^1.5.1"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-preset-loader": {
      "version": "2.3.4",
      "resolved": "https://registry.npmjs.org/conventional-changelog-preset-loader/-/conventional-changelog-preset-loader-2.3.4.tgz",
      "integrity": "sha512-GEKRWkrSAZeTq5+YjUZOYxdHq+ci4dNwHvpaBC3+ENalzFWuCWa9EZXSuZBpkr72sMdKB+1fyDV4takK1Lf58g==",
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-writer": {
      "version": "5.0.1",
      "resolved": "https://registry.npmjs.org/conventional-changelog-writer/-/conventional-changelog-writer-5.0.1.tgz",
      "integrity": "sha512-5WsuKUfxW7suLblAbFnxAcrvf6r+0b7GvNaWUwUIk0bXMnENP/PEieGKVUQrjPqwPT4o3EPAASBXiY6iHooLOQ==",
      "dependencies": {
        "conventional-commits-filter": "^2.0.7",
        "dateformat": "^3.0.0",
        "handlebars": "^4.7.7",
        "json-stringify-safe": "^5.0.1",
        "lodash": "^4.17.15",
        "meow": "^8.0.0",
        "semver": "^6.0.0",
        "split": "^1.0.0",
        "through2": "^4.0.0"
      },
      "bin": {
        "conventional-changelog-writer": "cli.js"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-changelog-writer/node_modules/semver": {
      "version": "6.3.1",
      "resolved": "https://registry.npmjs.org/semver/-/semver-6.3.1.tgz",
      "integrity": "sha512-BR7VvDCVHO+q2xBEWskxS6DJE1qRnb7DxzUrogb71CWoSficBxYsiAGd+Kl0mmq/MprG9yArRkyrQxTO6XjMzA==",
      "bin": {
        "semver": "bin/semver.js"
      }
    },
    "node_modules/conventional-commits-filter": {
      "version": "2.0.7",
      "resolved": "https://registry.npmjs.org/conventional-commits-filter/-/conventional-commits-filter-2.0.7.tgz",
      "integrity": "sha512-ASS9SamOP4TbCClsRHxIHXRfcGCnIoQqkvAzCSbZzTFLfcTqJVugB0agRgsEELsqaeWgsXv513eS116wnlSSPA==",
      "dependencies": {
        "lodash.ismatch": "^4.4.0",
        "modify-values": "^1.0.0"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/conventional-commits-parser": {
      "version": "3.2.4",
      "resolved": "https://registry.npmjs.org/conventional-commits-parser/-/conventional-commits-parser-3.2.4.tgz",
      "integrity": "sha512-nK7sAtfi+QXbxHCYfhpZsfRtaitZLIA6889kFIouLvz6repszQDgxBu7wf2WbU+Dco7sAnNCJYERCwt54WPC2Q==",
      "dependencies": {
        "is-text-path": "^1.0.1",
        "JSONStream": "^1.0.4",
        "lodash": "^4.17.15",
        "meow": "^8.0.0",
        "split2": "^3.0.0",
        "through2": "^4.0.0"
      },
      "bin": {
        "conventional-commits-parser": "cli.js"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/core-util-is": {
      "version": "1.0.3",
      "resolved": "https://registry.npmjs.org/core-util-is/-/core-util-is-1.0.3.tgz",
      "integrity": "sha512-ZQBvi1DcpJ4GDqanjucZ2Hj3wEO5pZDS89BWbkcrvdxksJorwUDDZamX9ldFkp9aw2lmBDLgkObEA4DWNJ9FYQ=="
    },
    "node_modules/create-require": {
      "version": "1.1.1",
      "resolved": "https://registry.npmjs.org/create-require/-/create-require-1.1.1.tgz",
      "integrity": "sha512-dcKFX3jn0MpIaXjisoRvexIJVEKzaq7z2rZKxf+MSr9TkdmHmsU4m2lcLojrj/FHl8mk5VxMmYA+ftRkP/3oKQ=="
    },
    "node_modules/cross-spawn": {
      "version": "7.0.6",
      "resolved": "https://registry.npmjs.org/cross-spawn/-/cross-spawn-7.0.6.tgz",
      "integrity": "sha512-uV2QOWP2nWzsy2aMp8aRibhi9dlzF5Hgh5SHaB9OiTGEyDTiJJyx0uy51QXdyWbtAHNua4XJzUKca3OzKUd3vA==",
      "dependencies": {
        "path-key": "^3.1.0",
        "shebang-command": "^2.0.0",
        "which": "^2.0.1"
      },
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/crypto-random-string": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/crypto-random-string/-/crypto-random-string-2.0.0.tgz",
      "integrity": "sha512-v1plID3y9r/lPhviJ1wrXpLeyUIGAZ2SHNYTEapm7/8A9nLPoyvVp3RK/EPFqn5kEznyWgYZNsRtYYIWbuG8KA==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/css-select": {
      "version": "4.3.0",
      "resolved": "https://registry.npmjs.org/css-select/-/css-select-4.3.0.tgz",
      "integrity": "sha512-wPpOYtnsVontu2mODhA19JrqWxNsfdatRKd64kmpRbQgh1KtItko5sTnEpPdpSaJszTOhEMlF/RPz28qj4HqhQ==",
      "dependencies": {
        "boolbase": "^1.0.0",
        "css-what": "^6.0.1",
        "domhandler": "^4.3.1",
        "domutils": "^2.8.0",
        "nth-check": "^2.0.1"
      },
      "funding": {
        "url": "https://github.com/sponsors/fb55"
      }
    },
    "node_modules/css-what": {
      "version": "6.1.0",
      "resolved": "https://registry.npmjs.org/css-what/-/css-what-6.1.0.tgz",
      "integrity": "sha512-HTUrgRJ7r4dsZKU6GjmpfRK1O76h97Z8MfS1G0FozR+oF2kG6Vfe8JE6zwrkbxigziPHinCJ+gCPjA9EaBDtRw==",
      "engines": {
        "node": ">= 6"
      },
      "funding": {
        "url": "https://github.com/sponsors/fb55"
      }
    },
    "node_modules/cssesc": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/cssesc/-/cssesc-3.0.0.tgz",
      "integrity": "sha512-/Tb/JcjK111nNScGob5MNtsntNM1aCNUDipB/TkwZFhyDrrE47SOx/18wF2bbjgc3ZzCSKW1T5nt5EbFoAz/Vg==",
      "bin": {
        "cssesc": "bin/cssesc"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/dargs": {
      "version": "7.0.0",
      "resolved": "https://registry.npmjs.org/dargs/-/dargs-7.0.0.tgz",
      "integrity": "sha512-2iy1EkLdlBzQGvbweYRFxmFath8+K7+AKB0TlhHWkNuH+TmovaMH/Wp7V7R4u7f4SnX3OgLsU9t1NI9ioDnUpg==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/dateformat": {
      "version": "3.0.3",
      "resolved": "https://registry.npmjs.org/dateformat/-/dateformat-3.0.3.tgz",
      "integrity": "sha512-jyCETtSl3VMZMWeRo7iY1FL19ges1t55hMo5yaam4Jrsm5EPL89UQkoQRyiI+Yf4k8r2ZpdngkV8hr1lIdjb3Q==",
      "engines": {
        "node": "*"
      }
    },
    "node_modules/debug": {
      "version": "4.3.4",
      "resolved": "https://registry.npmjs.org/debug/-/debug-4.3.4.tgz",
      "integrity": "sha512-PRWFHuSU3eDtQJPvnNY7Jcket1j0t5OuOsFzPPzsekD52Zl8qUfFIPEiswXqIvHWGVHOgX+7G/vCNNhehwxfkQ==",
      "dependencies": {
        "ms": "2.1.2"
      },
      "engines": {
        "node": ">=6.0"
      },
      "peerDependenciesMeta": {
        "supports-color": {
          "optional": true
        }
      }
    },
    "node_modules/decamelize": {
      "version": "1.2.0",
      "resolved": "https://registry.npmjs.org/decamelize/-/decamelize-1.2.0.tgz",
      "integrity": "sha512-z2S+W9X73hAUUki+N+9Za2lBlun89zigOyGrsax+KUQ6wKW4ZoWpEYBkGhQjwAjjDCkWxhY0VKEhk8wzY7F5cA==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/decamelize-keys": {
      "version": "1.1.1",
      "resolved": "https://registry.npmjs.org/decamelize-keys/-/decamelize-keys-1.1.1.tgz",
      "integrity": "sha512-WiPxgEirIV0/eIOMcnFBA3/IJZAZqKnwAwWyvvdi4lsr1WCN22nhdf/3db3DoZcUjTV2SqfzIwNyp6y2xs3nmg==",
      "dependencies": {
        "decamelize": "^1.1.0",
        "map-obj": "^1.0.0"
      },
      "engines": {
        "node": ">=0.10.0"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/decamelize-keys/node_modules/map-obj": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/map-obj/-/map-obj-1.0.1.tgz",
      "integrity": "sha512-7N/q3lyZ+LVCp7PzuxrJr4KMbBE2hW7BT7YNia330OFxIf4d3r5zVpicP2650l7CPN6RM9zOJRl3NGpqSiw3Eg==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/decompress-response": {
      "version": "6.0.0",
      "resolved": "https://registry.npmjs.org/decompress-response/-/decompress-response-6.0.0.tgz",
      "integrity": "sha512-aW35yZM6Bb/4oJlZncMH2LCoZtJXTRxES17vE3hoRiowU2kWHaJKFkSBDnDR+cm9J+9QhXmREyIfv0pji9ejCQ==",
      "dependencies": {
        "mimic-response": "^3.1.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/deep-extend": {
      "version": "0.6.0",
      "resolved": "https://registry.npmjs.org/deep-extend/-/deep-extend-0.6.0.tgz",
      "integrity": "sha512-LOHxIOaPYdHlJRtCQfDIVZtfw/ufM8+rVj649RIHzcm/vGwQRXFt6OPqIFWsm2XEMrNIEtWR64sY1LEKD2vAOA==",
      "engines": {
        "node": ">=4.0.0"
      }
    },
    "node_modules/define-lazy-prop": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/define-lazy-prop/-/define-lazy-prop-2.0.0.tgz",
      "integrity": "sha512-Ds09qNh8yw3khSjiJjiUInaGX9xlqZDY7JVryGxdxV7NPeuqQfplOpQ66yJFZut3jLa5zOwkXw1g9EI2uKh4Og==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/del": {
      "version": "6.1.1",
      "resolved": "https://registry.npmjs.org/del/-/del-6.1.1.tgz",
      "integrity": "sha512-ua8BhapfP0JUJKC/zV9yHHDW/rDoDxP4Zhn3AkA6/xT6gY7jYXJiaeyBZznYVujhZZET+UgcbZiQ7sN3WqcImg==",
      "dependencies": {
        "globby": "^11.0.1",
        "graceful-fs": "^4.2.4",
        "is-glob": "^4.0.1",
        "is-path-cwd": "^2.2.0",
        "is-path-inside": "^3.0.2",
        "p-map": "^4.0.0",
        "rimraf": "^3.0.2",
        "slash": "^3.0.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/del/node_modules/brace-expansion": {
      "version": "1.1.11",
      "resolved": "https://registry.npmjs.org/brace-expansion/-/brace-expansion-1.1.11.tgz",
      "integrity": "sha512-iCuPHDFgrHX7H2vEI/5xpz07zSHB00TpugqhmYtVmMO6518mCuRMoOYFldEBl0g187ufozdaHgWKcYFb61qGiA==",
      "dependencies": {
        "balanced-match": "^1.0.0",
        "concat-map": "0.0.1"
      }
    },
    "node_modules/del/node_modules/glob": {
      "version": "7.2.3",
      "resolved": "https://registry.npmjs.org/glob/-/glob-7.2.3.tgz",
      "integrity": "sha512-nFR0zLpU2YCaRxwoCJvL6UvCH2JFyFVIvwTLsIf21AuHlMskA1hhTdk+LlYJtOlYt9v6dvszD2BGRqBL+iQK9Q==",
      "deprecated": "Glob versions prior to v9 are no longer supported",
      "dependencies": {
        "fs.realpath": "^1.0.0",
        "inflight": "^1.0.4",
        "inherits": "2",
        "minimatch": "^3.1.1",
        "once": "^1.3.0",
        "path-is-absolute": "^1.0.0"
      },
      "engines": {
        "node": "*"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/del/node_modules/minimatch": {
      "version": "3.1.2",
      "resolved": "https://registry.npmjs.org/minimatch/-/minimatch-3.1.2.tgz",
      "integrity": "sha512-J7p63hRiAjw1NDEww1W7i37+ByIrOWO5XQQAzZ3VOcL0PNybwpfmV/N05zFAzwQ9USyEcX6t3UO+K5aqBQOIHw==",
      "dependencies": {
        "brace-expansion": "^1.1.7"
      },
      "engines": {
        "node": "*"
      }
    },
    "node_modules/del/node_modules/rimraf": {
      "version": "3.0.2",
      "resolved": "https://registry.npmjs.org/rimraf/-/rimraf-3.0.2.tgz",
      "integrity": "sha512-JZkJMZkAGFFPP2YqXZXPbMlMBgsxzE8ILs4lMIX/2o0L9UBw9O/Y3o6wFw/i9YLapcUJWwqbi3kdxIPdC62TIA==",
      "deprecated": "Rimraf versions prior to v4 are no longer supported",
      "dependencies": {
        "glob": "^7.1.3"
      },
      "bin": {
        "rimraf": "bin.js"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/detect-libc": {
      "version": "2.0.3",
      "resolved": "https://registry.npmjs.org/detect-libc/-/detect-libc-2.0.3.tgz",
      "integrity": "sha512-bwy0MGW55bG41VqxxypOsdSdGqLwXPI/focwgTYCFMbdUiBAxLg9CFzG08sz2aqzknwiX7Hkl0bQENjg8iLByw==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/dezalgo": {
      "version": "1.0.4",
      "resolved": "https://registry.npmjs.org/dezalgo/-/dezalgo-1.0.4.tgz",
      "integrity": "sha512-rXSP0bf+5n0Qonsb+SVVfNfIsimO4HEtmnIpPHY8Q1UCzKlQrDMfdobr8nJOOsRgWCyMRqeSBQzmWUMq7zvVig==",
      "dependencies": {
        "asap": "^2.0.0",
        "wrappy": "1"
      }
    },
    "node_modules/didyoumean": {
      "version": "1.2.2",
      "resolved": "https://registry.npmjs.org/didyoumean/-/didyoumean-1.2.2.tgz",
      "integrity": "sha512-gxtyfqMg7GKyhQmb056K7M3xszy/myH8w+B4RT+QXBQsvAOdc3XymqDDPHx1BgPgsdAA5SIifona89YtRATDzw=="
    },
    "node_modules/diff": {
      "version": "5.2.0",
      "resolved": "https://registry.npmjs.org/diff/-/diff-5.2.0.tgz",
      "integrity": "sha512-uIFDxqpRZGZ6ThOk84hEfqWoHx2devRFvpTZcTHur85vImfaxUbTW9Ryh4CpCuDnToOP1CEtXKIgytHBPVff5A==",
      "engines": {
        "node": ">=0.3.1"
      }
    },
    "node_modules/dir-glob": {
      "version": "3.0.1",
      "resolved": "https://registry.npmjs.org/dir-glob/-/dir-glob-3.0.1.tgz",
      "integrity": "sha512-WkrWp9GR4KXfKGYzOLmTuGVi1UWFfws377n9cc55/tb6DuqyF6pcQ5AbiHEshaDpY9v6oaSr2XCDidGmMwdzIA==",
      "dependencies": {
        "path-type": "^4.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/dir-glob/node_modules/path-type": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/path-type/-/path-type-4.0.0.tgz",
      "integrity": "sha512-gDKb8aZMDeD/tZWs9P6+q0J9Mwkdl6xMV8TjnGP3qJVJ06bdMgkbBlLU8IdfOsIsFz2BW1rNVT3XuNEl8zPAvw==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/dlv": {
      "version": "1.1.3",
      "resolved": "https://registry.npmjs.org/dlv/-/dlv-1.1.3.tgz",
      "integrity": "sha512-+HlytyjlPKnIG8XuRG8WvmBP8xs8P71y+SKKS6ZXWoEgLuePxtDoUEiH7WkdePWrQ5JBpE6aoVqfZfJUQkjXwA=="
    },
    "node_modules/dom-serializer": {
      "version": "1.4.1",
      "resolved": "https://registry.npmjs.org/dom-serializer/-/dom-serializer-1.4.1.tgz",
      "integrity": "sha512-VHwB3KfrcOOkelEG2ZOfxqLZdfkil8PtJi4P8N2MMXucZq2yLp75ClViUlOVwyoHEDjYU433Aq+5zWP61+RGag==",
      "dependencies": {
        "domelementtype": "^2.0.1",
        "domhandler": "^4.2.0",
        "entities": "^2.0.0"
      },
      "funding": {
        "url": "https://github.com/cheeriojs/dom-serializer?sponsor=1"
      }
    },
    "node_modules/domelementtype": {
      "version": "2.3.0",
      "resolved": "https://registry.npmjs.org/domelementtype/-/domelementtype-2.3.0.tgz",
      "integrity": "sha512-OLETBj6w0OsagBwdXnPdN0cnMfF9opN69co+7ZrbfPGrdpPVNBUj02spi6B1N7wChLQiPn4CSH/zJvXw56gmHw==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/fb55"
        }
      ]
    },
    "node_modules/domhandler": {
      "version": "4.3.1",
      "resolved": "https://registry.npmjs.org/domhandler/-/domhandler-4.3.1.tgz",
      "integrity": "sha512-GrwoxYN+uWlzO8uhUXRl0P+kHE4GtVPfYzVLcUxPL7KNdHKj66vvlhiweIHqYYXWlw+T8iLMp42Lm67ghw4WMQ==",
      "dependencies": {
        "domelementtype": "^2.2.0"
      },
      "engines": {
        "node": ">= 4"
      },
      "funding": {
        "url": "https://github.com/fb55/domhandler?sponsor=1"
      }
    },
    "node_modules/domutils": {
      "version": "2.8.0",
      "resolved": "https://registry.npmjs.org/domutils/-/domutils-2.8.0.tgz",
      "integrity": "sha512-w96Cjofp72M5IIhpjgobBimYEfoPjx1Vx0BSX9P30WBdZW2WIKU0T1Bd0kz2eNZ9ikjKgHbEyKx8BB6H1L3h3A==",
      "dependencies": {
        "dom-serializer": "^1.0.1",
        "domelementtype": "^2.2.0",
        "domhandler": "^4.2.0"
      },
      "funding": {
        "url": "https://github.com/fb55/domutils?sponsor=1"
      }
    },
    "node_modules/dot-prop": {
      "version": "5.3.0",
      "resolved": "https://registry.npmjs.org/dot-prop/-/dot-prop-5.3.0.tgz",
      "integrity": "sha512-QM8q3zDe58hqUqjraQOmzZ1LIH9SWQJTlEKCH4kJ2oQvLZk7RbQXvtDM2XEq3fwkV9CCvvH4LA0AV+ogFsBM2Q==",
      "dependencies": {
        "is-obj": "^2.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/eastasianwidth": {
      "version": "0.2.0",
      "resolved": "https://registry.npmjs.org/eastasianwidth/-/eastasianwidth-0.2.0.tgz",
      "integrity": "sha512-I88TYZWc9XiYHRQ4/3c5rjjfgkjhLyW2luGIheGERbNQ6OY7yTybanSpDXZa8y7VUP9YmDcYa+eyq4ca7iLqWA=="
    },
    "node_modules/electron-to-chromium": {
      "version": "1.5.75",
      "resolved": "https://registry.npmjs.org/electron-to-chromium/-/electron-to-chromium-1.5.75.tgz",
      "integrity": "sha512-Lf3++DumRE/QmweGjU+ZcKqQ+3bKkU/qjaKYhIJKEOhgIO9Xs6IiAQFkfFoj+RhgDk4LUeNsLo6plExHqSyu6Q==",
      "dev": true
    },
    "node_modules/elementtree": {
      "version": "0.1.7",
      "resolved": "https://registry.npmjs.org/elementtree/-/elementtree-0.1.7.tgz",
      "integrity": "sha512-wkgGT6kugeQk/P6VZ/f4T+4HB41BVgNBq5CDIZVbQ02nvTVqAiVTbskxxu3eA/X96lMlfYOwnLQpN2v5E1zDEg==",
      "dependencies": {
        "sax": "1.1.4"
      },
      "engines": {
        "node": ">= 0.4.0"
      }
    },
    "node_modules/emoji-regex": {
      "version": "8.0.0",
      "resolved": "https://registry.npmjs.org/emoji-regex/-/emoji-regex-8.0.0.tgz",
      "integrity": "sha512-MSjYzcWNOA0ewAHpz0MxpYFvwg6yjy1NG3xteoqz644VCo/RPgnr1/GGt+ic3iJTzQ8Eu3TdM14SawnVUmGE6A=="
    },
    "node_modules/end-of-stream": {
      "version": "1.4.4",
      "resolved": "https://registry.npmjs.org/end-of-stream/-/end-of-stream-1.4.4.tgz",
      "integrity": "sha512-+uw1inIHVPQoaVuHzRyXd21icM+cnt4CzD5rW+NC1wjOUSTOs+Te7FOv7AhN7vS9x/oIyhLP5PR1H+phQAHu5Q==",
      "dependencies": {
        "once": "^1.4.0"
      }
    },
    "node_modules/entities": {
      "version": "2.2.0",
      "resolved": "https://registry.npmjs.org/entities/-/entities-2.2.0.tgz",
      "integrity": "sha512-p92if5Nz619I0w+akJrLZH0MX0Pb5DX39XOwQTtXSdQQOaYH03S1uIQp4mhOZtAXrxq4ViO67YTiLBo2638o9A==",
      "funding": {
        "url": "https://github.com/fb55/entities?sponsor=1"
      }
    },
    "node_modules/env-paths": {
      "version": "2.2.1",
      "resolved": "https://registry.npmjs.org/env-paths/-/env-paths-2.2.1.tgz",
      "integrity": "sha512-+h1lkLKhZMTYjog1VEpJNG7NZJWcuc2DDk/qsqSTRRCOXiLjeQ1d1/udrUGhqMxUgAlwKNZ0cf2uqan5GLuS2A==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/error-ex": {
      "version": "1.3.2",
      "resolved": "https://registry.npmjs.org/error-ex/-/error-ex-1.3.2.tgz",
      "integrity": "sha512-7dFHNmqeFSEt2ZBsCriorKnn3Z2pj+fd9kmI6QoWw4//DL+icEBfc0U7qJCisqrTsKTjw4fNFy2pW9OqStD84g==",
      "dependencies": {
        "is-arrayish": "^0.2.1"
      }
    },
    "node_modules/esbuild": {
      "version": "0.24.0",
      "resolved": "https://registry.npmjs.org/esbuild/-/esbuild-0.24.0.tgz",
      "integrity": "sha512-FuLPevChGDshgSicjisSooU0cemp/sGXR841D5LHMB7mTVOmsEHcAxaH3irL53+8YDIeVNQEySh4DaYU/iuPqQ==",
      "dev": true,
      "hasInstallScript": true,
      "bin": {
        "esbuild": "bin/esbuild"
      },
      "engines": {
        "node": ">=18"
      },
      "optionalDependencies": {
        "@esbuild/aix-ppc64": "0.24.0",
        "@esbuild/android-arm": "0.24.0",
        "@esbuild/android-arm64": "0.24.0",
        "@esbuild/android-x64": "0.24.0",
        "@esbuild/darwin-arm64": "0.24.0",
        "@esbuild/darwin-x64": "0.24.0",
        "@esbuild/freebsd-arm64": "0.24.0",
        "@esbuild/freebsd-x64": "0.24.0",
        "@esbuild/linux-arm": "0.24.0",
        "@esbuild/linux-arm64": "0.24.0",
        "@esbuild/linux-ia32": "0.24.0",
        "@esbuild/linux-loong64": "0.24.0",
        "@esbuild/linux-mips64el": "0.24.0",
        "@esbuild/linux-ppc64": "0.24.0",
        "@esbuild/linux-riscv64": "0.24.0",
        "@esbuild/linux-s390x": "0.24.0",
        "@esbuild/linux-x64": "0.24.0",
        "@esbuild/netbsd-x64": "0.24.0",
        "@esbuild/openbsd-arm64": "0.24.0",
        "@esbuild/openbsd-x64": "0.24.0",
        "@esbuild/sunos-x64": "0.24.0",
        "@esbuild/win32-arm64": "0.24.0",
        "@esbuild/win32-ia32": "0.24.0",
        "@esbuild/win32-x64": "0.24.0"
      }
    },
    "node_modules/escalade": {
      "version": "3.2.0",
      "resolved": "https://registry.npmjs.org/escalade/-/escalade-3.2.0.tgz",
      "integrity": "sha512-WUj2qlxaQtO4g6Pq5c29GTcWGDyd8itL8zTlipgECz3JesAiiOKotd8JU6otB3PACgG6xkJUyVhboMS+bje/jA==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/escape-string-regexp": {
      "version": "1.0.5",
      "resolved": "https://registry.npmjs.org/escape-string-regexp/-/escape-string-regexp-1.0.5.tgz",
      "integrity": "sha512-vbRorB5FUQWvla16U8R/qgaFIya2qGzwDrNmCZuYKrbdSUMG6I1ZCGQRefkRVhuOkIGVne7BQ35DSfo1qvJqFg==",
      "engines": {
        "node": ">=0.8.0"
      }
    },
    "node_modules/execa": {
      "version": "5.1.1",
      "resolved": "https://registry.npmjs.org/execa/-/execa-5.1.1.tgz",
      "integrity": "sha512-8uSpZZocAZRBAPIEINJj3Lo9HyGitllczc27Eh5YYojjMFMn8yHMDMaUHE2Jqfq05D/wucwI4JGURyXt1vchyg==",
      "dev": true,
      "dependencies": {
        "cross-spawn": "^7.0.3",
        "get-stream": "^6.0.0",
        "human-signals": "^2.1.0",
        "is-stream": "^2.0.0",
        "merge-stream": "^2.0.0",
        "npm-run-path": "^4.0.1",
        "onetime": "^5.1.2",
        "signal-exit": "^3.0.3",
        "strip-final-newline": "^2.0.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sindresorhus/execa?sponsor=1"
      }
    },
    "node_modules/expand-template": {
      "version": "2.0.3",
      "resolved": "https://registry.npmjs.org/expand-template/-/expand-template-2.0.3.tgz",
      "integrity": "sha512-XYfuKMvj4O35f/pOXLObndIRvyQ+/+6AhODh+OKWj9S9498pHHn/IMszH+gt0fBCRWMNfk1ZSp5x3AifmnI2vg==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/fast-deep-equal": {
      "version": "3.1.3",
      "resolved": "https://registry.npmjs.org/fast-deep-equal/-/fast-deep-equal-3.1.3.tgz",
      "integrity": "sha512-f3qQ9oQy9j2AhBe/H9VC91wLmKBCCU/gDOnKNAYG5hswO7BLKj09Hc5HYNz9cGI++xlpDCIgDaitVs03ATR84Q==",
      "dev": true
    },
    "node_modules/fast-fifo": {
      "version": "1.3.2",
      "resolved": "https://registry.npmjs.org/fast-fifo/-/fast-fifo-1.3.2.tgz",
      "integrity": "sha512-/d9sfos4yxzpwkDkuN7k2SqFKtYNmCTzgfEpz82x34IM9/zc8KGxQoXg1liNC/izpRM/MBdt44Nmx41ZWqk+FQ=="
    },
    "node_modules/fast-glob": {
      "version": "3.3.2",
      "resolved": "https://registry.npmjs.org/fast-glob/-/fast-glob-3.3.2.tgz",
      "integrity": "sha512-oX2ruAFQwf/Orj8m737Y5adxDQO0LAB7/S5MnxCdTNDd4p6BsyIVsv9JQsATbTSq8KHRpLwIHbVlUNatxd+1Ow==",
      "dependencies": {
        "@nodelib/fs.stat": "^2.0.2",
        "@nodelib/fs.walk": "^1.2.3",
        "glob-parent": "^5.1.2",
        "merge2": "^1.3.0",
        "micromatch": "^4.0.4"
      },
      "engines": {
        "node": ">=8.6.0"
      }
    },
    "node_modules/fast-glob/node_modules/glob-parent": {
      "version": "5.1.2",
      "resolved": "https://registry.npmjs.org/glob-parent/-/glob-parent-5.1.2.tgz",
      "integrity": "sha512-AOIgSQCepiJYwP3ARnGx+5VnTu2HBYdzbGP45eLw1vr3zB3vZLeyed1sC9hnbcOc9/SrMyM5RPQrkGz4aS9Zow==",
      "dependencies": {
        "is-glob": "^4.0.1"
      },
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/fastq": {
      "version": "1.17.1",
      "resolved": "https://registry.npmjs.org/fastq/-/fastq-1.17.1.tgz",
      "integrity": "sha512-sRVD3lWVIXWg6By68ZN7vho9a1pQcN/WBFaAAsDDFzlJjvoGx0P8z7V1t72grFJfJhu3YPZBuu25f7Kaw2jN1w==",
      "dependencies": {
        "reusify": "^1.0.4"
      }
    },
    "node_modules/fd-slicer": {
      "version": "1.1.0",
      "resolved": "https://registry.npmjs.org/fd-slicer/-/fd-slicer-1.1.0.tgz",
      "integrity": "sha512-cE1qsB/VwyQozZ+q1dGxR8LBYNZeofhEdUNGSMbQD3Gw2lAzX9Zb3uIU6Ebc/Fmyjo9AWWfnn0AUCHqtevs/8g==",
      "dependencies": {
        "pend": "~1.2.0"
      }
    },
    "node_modules/fill-range": {
      "version": "7.1.1",
      "resolved": "https://registry.npmjs.org/fill-range/-/fill-range-7.1.1.tgz",
      "integrity": "sha512-YsGpe3WHLK8ZYi4tWDg2Jy3ebRz2rXowDxnld4bkQB00cc/1Zw9AWnC0i9ztDJitivtQvaI9KaLyKrc+hBW0yg==",
      "dependencies": {
        "to-regex-range": "^5.0.1"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/find-up": {
      "version": "2.1.0",
      "resolved": "https://registry.npmjs.org/find-up/-/find-up-2.1.0.tgz",
      "integrity": "sha512-NWzkk0jSJtTt08+FBFMvXoeZnOJD+jTtsRmBYbAIzJdX6l7dLgR7CTubCM5/eDdPUBvLCeVasP1brfVR/9/EZQ==",
      "dependencies": {
        "locate-path": "^2.0.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/foreground-child": {
      "version": "3.3.0",
      "resolved": "https://registry.npmjs.org/foreground-child/-/foreground-child-3.3.0.tgz",
      "integrity": "sha512-Ld2g8rrAyMYFXBhEqMz8ZAHBi4J4uS1i/CxGMDnjyFWddMXLVcDp051DZfu+t7+ab7Wv6SMqpWmyFIj5UbfFvg==",
      "dependencies": {
        "cross-spawn": "^7.0.0",
        "signal-exit": "^4.0.1"
      },
      "engines": {
        "node": ">=14"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/foreground-child/node_modules/signal-exit": {
      "version": "4.1.0",
      "resolved": "https://registry.npmjs.org/signal-exit/-/signal-exit-4.1.0.tgz",
      "integrity": "sha512-bzyZ1e88w9O1iNJbKnOlvYTrWPDl46O1bG0D3XInv+9tkPrxrN8jUUTiFlDkkmKWgn1M6CfIA13SuGqOa9Korw==",
      "engines": {
        "node": ">=14"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/formidable": {
      "version": "3.5.2",
      "resolved": "https://registry.npmjs.org/formidable/-/formidable-3.5.2.tgz",
      "integrity": "sha512-Jqc1btCy3QzRbJaICGwKcBfGWuLADRerLzDqi2NwSt/UkXLsHJw2TVResiaoBufHVHy9aSgClOHCeJsSsFLTbg==",
      "dependencies": {
        "dezalgo": "^1.0.4",
        "hexoid": "^2.0.0",
        "once": "^1.4.0"
      },
      "funding": {
        "url": "https://ko-fi.com/tunnckoCore/commissions"
      }
    },
    "node_modules/fraction.js": {
      "version": "4.3.7",
      "resolved": "https://registry.npmjs.org/fraction.js/-/fraction.js-4.3.7.tgz",
      "integrity": "sha512-ZsDfxO51wGAXREY55a7la9LScWpwv9RxIrYABrlvOFBlH/ShPnrtsXeuUIfXKKOVicNxQ+o8JTbJvjS4M89yew==",
      "dev": true,
      "engines": {
        "node": "*"
      },
      "funding": {
        "type": "patreon",
        "url": "https://github.com/sponsors/rawify"
      }
    },
    "node_modules/fs-constants": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/fs-constants/-/fs-constants-1.0.0.tgz",
      "integrity": "sha512-y6OAwoSIf7FyjMIv94u+b5rdheZEjzR63GTyZJm5qh4Bi+2YgwLCcI/fPFZkL5PSixOt6ZNKm+w+Hfp/Bciwow=="
    },
    "node_modules/fs-extra": {
      "version": "10.1.0",
      "resolved": "https://registry.npmjs.org/fs-extra/-/fs-extra-10.1.0.tgz",
      "integrity": "sha512-oRXApq54ETRj4eMiFzGnHWGy+zo5raudjuxN0b8H7s/RU2oW0Wvsx9O0ACRN/kRq9E8Vu/ReskGB5o3ji+FzHQ==",
      "dependencies": {
        "graceful-fs": "^4.2.0",
        "jsonfile": "^6.0.1",
        "universalify": "^2.0.0"
      },
      "engines": {
        "node": ">=12"
      }
    },
    "node_modules/fs-minipass": {
      "version": "2.1.0",
      "resolved": "https://registry.npmjs.org/fs-minipass/-/fs-minipass-2.1.0.tgz",
      "integrity": "sha512-V/JgOLFCS+R6Vcq0slCuaeWEdNC3ouDlJMNIsacH2VtALiu9mV4LPrHc5cDl8k5aw6J8jwgWWpiTo5RYhmIzvg==",
      "dependencies": {
        "minipass": "^3.0.0"
      },
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/fs-minipass/node_modules/minipass": {
      "version": "3.3.6",
      "resolved": "https://registry.npmjs.org/minipass/-/minipass-3.3.6.tgz",
      "integrity": "sha512-DxiNidxSEK+tHG6zOIklvNOwm3hvCrbUrdtzY74U6HKTJxvIDfOUL5W5P2Ghd3DTkhhKPYGqeNUIh5qcM4YBfw==",
      "dependencies": {
        "yallist": "^4.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/fs.realpath": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/fs.realpath/-/fs.realpath-1.0.0.tgz",
      "integrity": "sha512-OO0pH2lK6a0hZnAdau5ItzHPI6pUlvI7jMVnxUQRtw4owF2wk8lOSabtGDCTP4Ggrg2MbGnWO9X8K1t4+fGMDw=="
    },
    "node_modules/fsevents": {
      "version": "2.3.3",
      "resolved": "https://registry.npmjs.org/fsevents/-/fsevents-2.3.3.tgz",
      "integrity": "sha512-5xoDfX+fL7faATnagmWPpbFtwh/R77WmMMqqHGS65C3vvB0YHrgF+B1YmZ3441tMj5n63k0212XNoJwzlhffQw==",
      "hasInstallScript": true,
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": "^8.16.0 || ^10.6.0 || >=11.0.0"
      }
    },
    "node_modules/function-bind": {
      "version": "1.1.2",
      "resolved": "https://registry.npmjs.org/function-bind/-/function-bind-1.1.2.tgz",
      "integrity": "sha512-7XHNxH7qX9xG5mIwxkhumTox/MIRNcOgDrxWsMt2pAr23WHp6MrRlN7FBSFpCpr+oVO0F744iUgR82nJMfG2SA==",
      "funding": {
        "url": "https://github.com/sponsors/ljharb"
      }
    },
    "node_modules/get-caller-file": {
      "version": "2.0.5",
      "resolved": "https://registry.npmjs.org/get-caller-file/-/get-caller-file-2.0.5.tgz",
      "integrity": "sha512-DyFP3BM/3YHTQOCUL/w0OZHR0lpKeGrxotcHWcqNEdnltqFwXVfhEBQ94eIo34AfQpo0rGki4cyIiftY06h2Fg==",
      "engines": {
        "node": "6.* || 8.* || >= 10.*"
      }
    },
    "node_modules/get-pkg-repo": {
      "version": "4.2.1",
      "resolved": "https://registry.npmjs.org/get-pkg-repo/-/get-pkg-repo-4.2.1.tgz",
      "integrity": "sha512-2+QbHjFRfGB74v/pYWjd5OhU3TDIC2Gv/YKUTk/tCvAz0pkn/Mz6P3uByuBimLOcPvN2jYdScl3xGFSrx0jEcA==",
      "dependencies": {
        "@hutson/parse-repository-url": "^3.0.0",
        "hosted-git-info": "^4.0.0",
        "through2": "^2.0.0",
        "yargs": "^16.2.0"
      },
      "bin": {
        "get-pkg-repo": "src/cli.js"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/get-pkg-repo/node_modules/cliui": {
      "version": "7.0.4",
      "resolved": "https://registry.npmjs.org/cliui/-/cliui-7.0.4.tgz",
      "integrity": "sha512-OcRE68cOsVMXp1Yvonl/fzkQOyjLSu/8bhPDfQt0e0/Eb283TKP20Fs2MqoPsr9SwA595rRCA+QMzYc9nBP+JQ==",
      "dependencies": {
        "string-width": "^4.2.0",
        "strip-ansi": "^6.0.0",
        "wrap-ansi": "^7.0.0"
      }
    },
    "node_modules/get-pkg-repo/node_modules/readable-stream": {
      "version": "2.3.8",
      "resolved": "https://registry.npmjs.org/readable-stream/-/readable-stream-2.3.8.tgz",
      "integrity": "sha512-8p0AUk4XODgIewSi0l8Epjs+EVnWiK7NoDIEGU0HhE7+ZyY8D1IMY7odu5lRrFXGg71L15KG8QrPmum45RTtdA==",
      "dependencies": {
        "core-util-is": "~1.0.0",
        "inherits": "~2.0.3",
        "isarray": "~1.0.0",
        "process-nextick-args": "~2.0.0",
        "safe-buffer": "~5.1.1",
        "string_decoder": "~1.1.1",
        "util-deprecate": "~1.0.1"
      }
    },
    "node_modules/get-pkg-repo/node_modules/string_decoder": {
      "version": "1.1.1",
      "resolved": "https://registry.npmjs.org/string_decoder/-/string_decoder-1.1.1.tgz",
      "integrity": "sha512-n/ShnvDi6FHbbVfviro+WojiFzv+s8MPMHBczVePfUpDJLwoLT0ht1l4YwBCbi8pJAveEEdnkHyPyTP/mzRfwg==",
      "dependencies": {
        "safe-buffer": "~5.1.0"
      }
    },
    "node_modules/get-pkg-repo/node_modules/through2": {
      "version": "2.0.5",
      "resolved": "https://registry.npmjs.org/through2/-/through2-2.0.5.tgz",
      "integrity": "sha512-/mrRod8xqpA+IHSLyGCQ2s8SPHiCDEeQJSep1jqLYeEUClOFG2Qsh+4FU6G9VeqpZnGW/Su8LQGc4YKni5rYSQ==",
      "dependencies": {
        "readable-stream": "~2.3.6",
        "xtend": "~4.0.1"
      }
    },
    "node_modules/get-pkg-repo/node_modules/yargs": {
      "version": "16.2.0",
      "resolved": "https://registry.npmjs.org/yargs/-/yargs-16.2.0.tgz",
      "integrity": "sha512-D1mvvtDG0L5ft/jGWkLpG1+m0eQxOfaBvTNELraWj22wSVUMWxZUvYgJYcKh6jGGIkJFhH4IZPQhR4TKpc8mBw==",
      "dependencies": {
        "cliui": "^7.0.2",
        "escalade": "^3.1.1",
        "get-caller-file": "^2.0.5",
        "require-directory": "^2.1.1",
        "string-width": "^4.2.0",
        "y18n": "^5.0.5",
        "yargs-parser": "^20.2.2"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/get-stream": {
      "version": "6.0.1",
      "resolved": "https://registry.npmjs.org/get-stream/-/get-stream-6.0.1.tgz",
      "integrity": "sha512-ts6Wi+2j3jQjqi70w5AlN8DFnkSwC+MqmxEzdEALB2qXZYV3X/b1CTfgPLGJNMeAWxdPfU8FO1ms3NUfaHCPYg==",
      "dev": true,
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/git-raw-commits": {
      "version": "2.0.11",
      "resolved": "https://registry.npmjs.org/git-raw-commits/-/git-raw-commits-2.0.11.tgz",
      "integrity": "sha512-VnctFhw+xfj8Va1xtfEqCUD2XDrbAPSJx+hSrE5K7fGdjZruW7XV+QOrN7LF/RJyvspRiD2I0asWsxFp0ya26A==",
      "dependencies": {
        "dargs": "^7.0.0",
        "lodash": "^4.17.15",
        "meow": "^8.0.0",
        "split2": "^3.0.0",
        "through2": "^4.0.0"
      },
      "bin": {
        "git-raw-commits": "cli.js"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/git-remote-origin-url": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/git-remote-origin-url/-/git-remote-origin-url-2.0.0.tgz",
      "integrity": "sha512-eU+GGrZgccNJcsDH5LkXR3PB9M958hxc7sbA8DFJjrv9j4L2P/eZfKhM+QD6wyzpiv+b1BpK0XrYCxkovtjSLw==",
      "dependencies": {
        "gitconfiglocal": "^1.0.0",
        "pify": "^2.3.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/git-semver-tags": {
      "version": "4.1.1",
      "resolved": "https://registry.npmjs.org/git-semver-tags/-/git-semver-tags-4.1.1.tgz",
      "integrity": "sha512-OWyMt5zBe7xFs8vglMmhM9lRQzCWL3WjHtxNNfJTMngGym7pC1kh8sP6jevfydJ6LP3ZvGxfb6ABYgPUM0mtsA==",
      "dependencies": {
        "meow": "^8.0.0",
        "semver": "^6.0.0"
      },
      "bin": {
        "git-semver-tags": "cli.js"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/git-semver-tags/node_modules/semver": {
      "version": "6.3.1",
      "resolved": "https://registry.npmjs.org/semver/-/semver-6.3.1.tgz",
      "integrity": "sha512-BR7VvDCVHO+q2xBEWskxS6DJE1qRnb7DxzUrogb71CWoSficBxYsiAGd+Kl0mmq/MprG9yArRkyrQxTO6XjMzA==",
      "bin": {
        "semver": "bin/semver.js"
      }
    },
    "node_modules/gitconfiglocal": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/gitconfiglocal/-/gitconfiglocal-1.0.0.tgz",
      "integrity": "sha512-spLUXeTAVHxDtKsJc8FkFVgFtMdEN9qPGpL23VfSHx4fP4+Ds097IXLvymbnDH8FnmxX5Nr9bPw3A+AQ6mWEaQ==",
      "dependencies": {
        "ini": "^1.3.2"
      }
    },
    "node_modules/gitconfiglocal/node_modules/ini": {
      "version": "1.3.8",
      "resolved": "https://registry.npmjs.org/ini/-/ini-1.3.8.tgz",
      "integrity": "sha512-JV/yugV2uzW5iMRSiZAyDtQd+nxtUnjeLt0acNdw98kKLrvuRVyB80tsREOE7yvGVgalhZ6RNXCmEHkUKBKxew=="
    },
    "node_modules/github-from-package": {
      "version": "0.0.0",
      "resolved": "https://registry.npmjs.org/github-from-package/-/github-from-package-0.0.0.tgz",
      "integrity": "sha512-SyHy3T1v2NUXn29OsWdxmK6RwHD+vkj3v8en8AOBZ1wBQ/hCAQ5bAQTD02kW4W9tUp/3Qh6J8r9EvntiyCmOOw=="
    },
    "node_modules/glob": {
      "version": "9.3.5",
      "resolved": "https://registry.npmjs.org/glob/-/glob-9.3.5.tgz",
      "integrity": "sha512-e1LleDykUz2Iu+MTYdkSsuWX8lvAjAcs0Xef0lNIu0S2wOAzuTxCJtcd9S3cijlwYF18EsU3rzb8jPVobxDh9Q==",
      "dependencies": {
        "fs.realpath": "^1.0.0",
        "minimatch": "^8.0.2",
        "minipass": "^4.2.4",
        "path-scurry": "^1.6.1"
      },
      "engines": {
        "node": ">=16 || 14 >=14.17"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/glob-parent": {
      "version": "6.0.2",
      "resolved": "https://registry.npmjs.org/glob-parent/-/glob-parent-6.0.2.tgz",
      "integrity": "sha512-XxwI8EOhVQgWp6iDL+3b0r86f4d6AX6zSU55HfB4ydCEuXLXc5FcYeOu+nnGftS4TEju/11rt4KJPTMgbfmv4A==",
      "dependencies": {
        "is-glob": "^4.0.3"
      },
      "engines": {
        "node": ">=10.13.0"
      }
    },
    "node_modules/glob/node_modules/minimatch": {
      "version": "8.0.4",
      "resolved": "https://registry.npmjs.org/minimatch/-/minimatch-8.0.4.tgz",
      "integrity": "sha512-W0Wvr9HyFXZRGIDgCicunpQ299OKXs9RgZfaukz4qAW/pJhcpUfupc9c+OObPOFueNy8VSrZgEmDtk6Kh4WzDA==",
      "dependencies": {
        "brace-expansion": "^2.0.1"
      },
      "engines": {
        "node": ">=16 || 14 >=14.17"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/globby": {
      "version": "11.1.0",
      "resolved": "https://registry.npmjs.org/globby/-/globby-11.1.0.tgz",
      "integrity": "sha512-jhIXaOzy1sb8IyocaruWSn1TjmnBVs8Ayhcy83rmxNJ8q2uWKCAj3CnJY+KpGSXCueAPc0i05kVvVKtP1t9S3g==",
      "dependencies": {
        "array-union": "^2.1.0",
        "dir-glob": "^3.0.1",
        "fast-glob": "^3.2.9",
        "ignore": "^5.2.0",
        "merge2": "^1.4.1",
        "slash": "^3.0.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/graceful-fs": {
      "version": "4.2.11",
      "resolved": "https://registry.npmjs.org/graceful-fs/-/graceful-fs-4.2.11.tgz",
      "integrity": "sha512-RbJ5/jmFcNNCcDV5o9eTnBLJ/HszWV0P73bc+Ff4nS/rJj+YaS6IGyiOL0VoBYX+l1Wrl3k63h/KrH+nhJ0XvQ=="
    },
    "node_modules/gradle-to-js": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/gradle-to-js/-/gradle-to-js-2.0.1.tgz",
      "integrity": "sha512-is3hDn9zb8XXnjbEeAEIqxTpLHUiGBqjegLmXPuyMBfKAggpadWFku4/AP8iYAGBX6qR9/5UIUIp47V0XI3aMw==",
      "dependencies": {
        "lodash.merge": "^4.6.2"
      },
      "bin": {
        "gradle-to-js": "cli.js"
      }
    },
    "node_modules/handlebars": {
      "version": "4.7.8",
      "resolved": "https://registry.npmjs.org/handlebars/-/handlebars-4.7.8.tgz",
      "integrity": "sha512-vafaFqs8MZkRrSX7sFVUdo3ap/eNiLnb4IakshzvP56X5Nr1iGKAIqdX6tMlm6HcNRIkr6AxO5jFEoJzzpT8aQ==",
      "dependencies": {
        "minimist": "^1.2.5",
        "neo-async": "^2.6.2",
        "source-map": "^0.6.1",
        "wordwrap": "^1.0.0"
      },
      "bin": {
        "handlebars": "bin/handlebars"
      },
      "engines": {
        "node": ">=0.4.7"
      },
      "optionalDependencies": {
        "uglify-js": "^3.1.4"
      }
    },
    "node_modules/hard-rejection": {
      "version": "2.1.0",
      "resolved": "https://registry.npmjs.org/hard-rejection/-/hard-rejection-2.1.0.tgz",
      "integrity": "sha512-VIZB+ibDhx7ObhAe7OVtoEbuP4h/MuOTHJ+J8h/eBXotJYl0fBgR72xDFCKgIh22OJZIOVNxBMWuhAr10r8HdA==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/has-flag": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/has-flag/-/has-flag-4.0.0.tgz",
      "integrity": "sha512-EykJT/Q1KjTWctppgIAgfSO0tKVuZUjhgMr17kqTumMl6Afv3EISleU7qZUzoXDFTAHTDC4NOoG/ZxU3EvlMPQ==",
      "dev": true,
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/hasown": {
      "version": "2.0.2",
      "resolved": "https://registry.npmjs.org/hasown/-/hasown-2.0.2.tgz",
      "integrity": "sha512-0hJU9SCPvmMzIBdZFqNPXWa6dqh7WdH0cII9y+CyS8rG3nL48Bclra9HmKhVVUHyPWNH5Y7xDwAB7bfgSjkUMQ==",
      "dependencies": {
        "function-bind": "^1.1.2"
      },
      "engines": {
        "node": ">= 0.4"
      }
    },
    "node_modules/he": {
      "version": "1.2.0",
      "resolved": "https://registry.npmjs.org/he/-/he-1.2.0.tgz",
      "integrity": "sha512-F/1DnUGPopORZi0ni+CvrCgHQ5FyEAHRLSApuYWMmrbSwoN2Mn/7k+Gl38gJnR7yyDZk6WLXwiGod1JOWNDKGw==",
      "bin": {
        "he": "bin/he"
      }
    },
    "node_modules/hexoid": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/hexoid/-/hexoid-2.0.0.tgz",
      "integrity": "sha512-qlspKUK7IlSQv2o+5I7yhUd7TxlOG2Vr5LTa3ve2XSNVKAL/n/u/7KLvKmFNimomDIKvZFXWHv0T12mv7rT8Aw==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/hosted-git-info": {
      "version": "4.1.0",
      "resolved": "https://registry.npmjs.org/hosted-git-info/-/hosted-git-info-4.1.0.tgz",
      "integrity": "sha512-kyCuEOWjJqZuDbRHzL8V93NzQhwIB71oFWSyzVo+KPZI+pnQPPxucdkrOZvkLRnrf5URsQM+IJ09Dw29cRALIA==",
      "dependencies": {
        "lru-cache": "^6.0.0"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/htmx.org": {
      "version": "1.9.12",
      "resolved": "https://registry.npmjs.org/htmx.org/-/htmx.org-1.9.12.tgz",
      "integrity": "sha512-VZAohXyF7xPGS52IM8d1T1283y+X4D+Owf3qY1NZ9RuBypyu9l8cGsxUMAG5fEAb/DhT7rDoJ9Hpu5/HxFD3cw=="
    },
    "node_modules/human-signals": {
      "version": "2.1.0",
      "resolved": "https://registry.npmjs.org/human-signals/-/human-signals-2.1.0.tgz",
      "integrity": "sha512-B4FFZ6q/T2jhhksgkbEW3HBvWIfDW85snkQgawt07S7J5QXTk6BkNV+0yAeZrM5QpMAdYlocGoljn0sJ/WQkFw==",
      "dev": true,
      "engines": {
        "node": ">=10.17.0"
      }
    },
    "node_modules/ieee754": {
      "version": "1.2.1",
      "resolved": "https://registry.npmjs.org/ieee754/-/ieee754-1.2.1.tgz",
      "integrity": "sha512-dcyqhDvX1C46lXZcVqCpK+FtMRQVdIMN6/Df5js2zouUsqG7I6sFxitIC+7KYK29KdXOLHdu9zL4sFnoVQnqaA==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/feross"
        },
        {
          "type": "patreon",
          "url": "https://www.patreon.com/feross"
        },
        {
          "type": "consulting",
          "url": "https://feross.org/support"
        }
      ]
    },
    "node_modules/ignore": {
      "version": "5.3.2",
      "resolved": "https://registry.npmjs.org/ignore/-/ignore-5.3.2.tgz",
      "integrity": "sha512-hsBTNUqQTDwkWtcdYI2i06Y/nUBEsNEDJKjWdigLvegy8kDuJAS8uRlpkkcQpyEXL0Z/pjDy5HBmMjRCJ2gq+g==",
      "engines": {
        "node": ">= 4"
      }
    },
    "node_modules/indent-string": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/indent-string/-/indent-string-4.0.0.tgz",
      "integrity": "sha512-EdDDZu4A2OyIK7Lr/2zG+w5jmbuk1DVBnEwREQvBzspBJkCEbRa8GxU1lghYcaGJCnRWibjDXlq779X1/y5xwg==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/inflight": {
      "version": "1.0.6",
      "resolved": "https://registry.npmjs.org/inflight/-/inflight-1.0.6.tgz",
      "integrity": "sha512-k92I/b08q4wvFscXCLvqfsHCrjrF7yiXsQuIVvVE7N82W3+aqpzuUdBbfhWcy/FZR3/4IgflMgKLOsvPDrGCJA==",
      "deprecated": "This module is not supported, and leaks memory. Do not use it. Check out lru-cache if you want a good and tested way to coalesce async requests by a key value, which is much more comprehensive and powerful.",
      "dependencies": {
        "once": "^1.3.0",
        "wrappy": "1"
      }
    },
    "node_modules/inherits": {
      "version": "2.0.4",
      "resolved": "https://registry.npmjs.org/inherits/-/inherits-2.0.4.tgz",
      "integrity": "sha512-k/vGaX4/Yla3WzyMCvTQOXYeIHvqOKtnqBduzTHpzpQZzAskKMhZ2K+EnBiSM9zGSoIFeMpXKxa4dYeZIQqewQ=="
    },
    "node_modules/ini": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/ini/-/ini-2.0.0.tgz",
      "integrity": "sha512-7PnF4oN3CvZF23ADhA5wRaYEQpJ8qygSkbtTXWBeXWXmEVRXK+1ITciHWwHhsjv1TmW0MgacIv6hEi5pX5NQdA==",
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/is-arrayish": {
      "version": "0.2.1",
      "resolved": "https://registry.npmjs.org/is-arrayish/-/is-arrayish-0.2.1.tgz",
      "integrity": "sha512-zz06S8t0ozoDXMG+ube26zeCTNXcKIPJZJi8hBrF4idCLms4CG9QtK7qBl1boi5ODzFpjswb5JPmHCbMpjaYzg=="
    },
    "node_modules/is-binary-path": {
      "version": "2.1.0",
      "resolved": "https://registry.npmjs.org/is-binary-path/-/is-binary-path-2.1.0.tgz",
      "integrity": "sha512-ZMERYes6pDydyuGidse7OsHxtbI7WVeUEozgR/g7rd0xUimYNlvZRE/K2MgZTjWy725IfelLeVcEM97mmtRGXw==",
      "dependencies": {
        "binary-extensions": "^2.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/is-core-module": {
      "version": "2.16.1",
      "resolved": "https://registry.npmjs.org/is-core-module/-/is-core-module-2.16.1.tgz",
      "integrity": "sha512-UfoeMA6fIJ8wTYFEUjelnaGI67v6+N7qXJEvQuIGa99l4xsCruSYOVSQ0uPANn4dAzm8lkYPaKLrrijLq7x23w==",
      "dependencies": {
        "hasown": "^2.0.2"
      },
      "engines": {
        "node": ">= 0.4"
      },
      "funding": {
        "url": "https://github.com/sponsors/ljharb"
      }
    },
    "node_modules/is-docker": {
      "version": "2.2.1",
      "resolved": "https://registry.npmjs.org/is-docker/-/is-docker-2.2.1.tgz",
      "integrity": "sha512-F+i2BKsFrH66iaUFc0woD8sLy8getkwTwtOBjvs56Cx4CgJDeKQeqfz8wAYiSb8JOprWhHH5p77PbmYCvvUuXQ==",
      "bin": {
        "is-docker": "cli.js"
      },
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/is-extglob": {
      "version": "2.1.1",
      "resolved": "https://registry.npmjs.org/is-extglob/-/is-extglob-2.1.1.tgz",
      "integrity": "sha512-SbKbANkN603Vi4jEZv49LeVJMn4yGwsbzZworEoyEiutsN3nJYdbO36zfhGJ6QEDpOZIFkDtnq5JRxmvl3jsoQ==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/is-fullwidth-code-point": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/is-fullwidth-code-point/-/is-fullwidth-code-point-3.0.0.tgz",
      "integrity": "sha512-zymm5+u+sCsSWyD9qNaejV3DFvhCKclKdizYaJUuHA83RLjb7nSuGnddCHGv0hk+KY7BMAlsWeK4Ueg6EV6XQg==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/is-glob": {
      "version": "4.0.3",
      "resolved": "https://registry.npmjs.org/is-glob/-/is-glob-4.0.3.tgz",
      "integrity": "sha512-xelSayHH36ZgE7ZWhli7pW34hNbNl8Ojv5KVmkJD4hBdD3th8Tfk9vYasLM+mXWOZhFkgZfxhLSnrwRr4elSSg==",
      "dependencies": {
        "is-extglob": "^2.1.1"
      },
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/is-number": {
      "version": "7.0.0",
      "resolved": "https://registry.npmjs.org/is-number/-/is-number-7.0.0.tgz",
      "integrity": "sha512-41Cifkg6e8TylSpdtTpeLVMqvSBEVzTttHvERD741+pnZ8ANv0004MRL43QKPDlK9cGvNp6NZWZUBlbGXYxxng==",
      "engines": {
        "node": ">=0.12.0"
      }
    },
    "node_modules/is-obj": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/is-obj/-/is-obj-2.0.0.tgz",
      "integrity": "sha512-drqDG3cbczxxEJRoOXcOjtdp1J/lyp1mNn0xaznRs8+muBhgQcrnbspox5X5fOw0HnMnbfDzvnEMEtqDEJEo8w==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/is-path-cwd": {
      "version": "2.2.0",
      "resolved": "https://registry.npmjs.org/is-path-cwd/-/is-path-cwd-2.2.0.tgz",
      "integrity": "sha512-w942bTcih8fdJPJmQHFzkS76NEP8Kzzvmw92cXsazb8intwLqPibPPdXf4ANdKV3rYMuuQYGIWtvz9JilB3NFQ==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/is-path-inside": {
      "version": "3.0.3",
      "resolved": "https://registry.npmjs.org/is-path-inside/-/is-path-inside-3.0.3.tgz",
      "integrity": "sha512-Fd4gABb+ycGAmKou8eMftCupSir5lRxqf4aD/vd0cD2qc4HL07OjCeuHMr8Ro4CoMaeCKDB0/ECBOVWjTwUvPQ==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/is-plain-obj": {
      "version": "1.1.0",
      "resolved": "https://registry.npmjs.org/is-plain-obj/-/is-plain-obj-1.1.0.tgz",
      "integrity": "sha512-yvkRyxmFKEOQ4pNXCmJG5AEQNlXJS5LaONXo5/cLdTZdWvsZ1ioJEonLGAosKlMWE8lwUy/bJzMjcw8az73+Fg==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/is-port-reachable": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/is-port-reachable/-/is-port-reachable-4.0.0.tgz",
      "integrity": "sha512-9UoipoxYmSk6Xy7QFgRv2HDyaysmgSG75TFQs6S+3pDM7ZhKTF/bskZV+0UlABHzKjNVhPjYCLfeZUEg1wXxig==",
      "dev": true,
      "engines": {
        "node": "^12.20.0 || ^14.13.1 || >=16.0.0"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/is-stream": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/is-stream/-/is-stream-2.0.1.tgz",
      "integrity": "sha512-hFoiJiTl63nn+kstHGBtewWSKnQLpyb155KHheA1l39uvtO9nWIop1p3udqPcUd/xbF1VLMO4n7OI6p7RbngDg==",
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/is-text-path": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/is-text-path/-/is-text-path-1.0.1.tgz",
      "integrity": "sha512-xFuJpne9oFz5qDaodwmmG08e3CawH/2ZV8Qqza1Ko7Sk8POWbkRdwIoAWVhqvq0XeUzANEhKo2n0IXUGBm7A/w==",
      "dependencies": {
        "text-extensions": "^1.0.0"
      },
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/is-wsl": {
      "version": "2.2.0",
      "resolved": "https://registry.npmjs.org/is-wsl/-/is-wsl-2.2.0.tgz",
      "integrity": "sha512-fKzAra0rGJUUBwGBgNkHZuToZcn+TtXHpeCgmkMJMMYx1sQDYaCSyjJBSCa2nH1DGm7s3n1oBnohoVTBaN7Lww==",
      "dependencies": {
        "is-docker": "^2.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/isarray": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/isarray/-/isarray-1.0.0.tgz",
      "integrity": "sha512-VLghIWNM6ELQzo7zwmcg0NmTVyWKYjvIeM83yjp0wRDTmUnrM678fQbcKBo6n2CJEF0szoG//ytg+TKla89ALQ=="
    },
    "node_modules/isexe": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/isexe/-/isexe-2.0.0.tgz",
      "integrity": "sha512-RHxMLp9lnKHGHRng9QFhRCMbYAcVpn69smSGcq3f36xjgVVWThj4qqLbTLlq7Ssj8B+fIQ1EuCEGI2lKsyQeIw=="
    },
    "node_modules/jackspeak": {
      "version": "3.4.3",
      "resolved": "https://registry.npmjs.org/jackspeak/-/jackspeak-3.4.3.tgz",
      "integrity": "sha512-OGlZQpz2yfahA/Rd1Y8Cd9SIEsqvXkLVoSw/cgwhnhFMDbsQFeZYoJJ7bIZBS9BcamUW96asq/npPWugM+RQBw==",
      "dependencies": {
        "@isaacs/cliui": "^8.0.2"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      },
      "optionalDependencies": {
        "@pkgjs/parseargs": "^0.11.0"
      }
    },
    "node_modules/jiti": {
      "version": "1.21.7",
      "resolved": "https://registry.npmjs.org/jiti/-/jiti-1.21.7.tgz",
      "integrity": "sha512-/imKNG4EbWNrVjoNC/1H5/9GFy+tqjGBHCaSsN+P2RnPqjsLmv6UD3Ej+Kj8nBWaRAwyk7kK5ZUc+OEatnTR3A==",
      "bin": {
        "jiti": "bin/jiti.js"
      }
    },
    "node_modules/js-tokens": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/js-tokens/-/js-tokens-4.0.0.tgz",
      "integrity": "sha512-RdJUflcE3cUzKiMqQgsCu06FPu9UdIJO0beYbPhHN4k6apgJtifcoCtT9bcxOpYBtpD2kCM6Sbzg4CausW/PKQ=="
    },
    "node_modules/json-parse-better-errors": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/json-parse-better-errors/-/json-parse-better-errors-1.0.2.tgz",
      "integrity": "sha512-mrqyZKfX5EhL7hvqcV6WG1yYjnjeuYDzDhhcAAUrq8Po85NBQBJP+ZDUT75qZQ98IkUoBqdkExkukOU7Ts2wrw=="
    },
    "node_modules/json-parse-even-better-errors": {
      "version": "2.3.1",
      "resolved": "https://registry.npmjs.org/json-parse-even-better-errors/-/json-parse-even-better-errors-2.3.1.tgz",
      "integrity": "sha512-xyFwyhro/JEof6Ghe2iz2NcXoj2sloNsWr/XsERDK/oiPCfaNhl5ONfp+jQdAZRQQ0IJWNzH9zIZF7li91kh2w=="
    },
    "node_modules/json-schema-traverse": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/json-schema-traverse/-/json-schema-traverse-1.0.0.tgz",
      "integrity": "sha512-NM8/P9n3XjXhIZn1lLhkFaACTOURQXjWhV4BA/RnOv8xvgqtqpAX9IO4mRQxSx1Rlo4tqzeqb0sOlruaOy3dug==",
      "dev": true
    },
    "node_modules/json-stringify-safe": {
      "version": "5.0.1",
      "resolved": "https://registry.npmjs.org/json-stringify-safe/-/json-stringify-safe-5.0.1.tgz",
      "integrity": "sha512-ZClg6AaYvamvYEE82d3Iyd3vSSIjQ+odgjaTzRuO3s7toCdFKczob2i0zCh7JE8kWn17yvAWhUVxvqGwUalsRA=="
    },
    "node_modules/jsonfile": {
      "version": "6.1.0",
      "resolved": "https://registry.npmjs.org/jsonfile/-/jsonfile-6.1.0.tgz",
      "integrity": "sha512-5dgndWOriYSm5cnYaJNhalLNDKOqFwyDB/rr1E9ZsGciGvKPs8R2xYGCacuf3z6K1YKDz182fd+fY3cn3pMqXQ==",
      "dependencies": {
        "universalify": "^2.0.0"
      },
      "optionalDependencies": {
        "graceful-fs": "^4.1.6"
      }
    },
    "node_modules/jsonparse": {
      "version": "1.3.1",
      "resolved": "https://registry.npmjs.org/jsonparse/-/jsonparse-1.3.1.tgz",
      "integrity": "sha512-POQXvpdL69+CluYsillJ7SUhKvytYjW9vG/GKpnf+xP8UWgYEM/RaMzHHofbALDiKbbP1W8UEYmgGl39WkPZsg==",
      "engines": [
        "node >= 0.2.0"
      ]
    },
    "node_modules/JSONStream": {
      "version": "1.3.5",
      "resolved": "https://registry.npmjs.org/JSONStream/-/JSONStream-1.3.5.tgz",
      "integrity": "sha512-E+iruNOY8VV9s4JEbe1aNEm6MiszPRr/UfcHMz0TQh1BXSxHK+ASV1R6W4HpjBhSeS+54PIsAMCBmwD06LLsqQ==",
      "dependencies": {
        "jsonparse": "^1.2.0",
        "through": ">=2.2.7 <3"
      },
      "bin": {
        "JSONStream": "bin.js"
      },
      "engines": {
        "node": "*"
      }
    },
    "node_modules/kind-of": {
      "version": "6.0.3",
      "resolved": "https://registry.npmjs.org/kind-of/-/kind-of-6.0.3.tgz",
      "integrity": "sha512-dcS1ul+9tmeD95T+x28/ehLgd9mENa3LsvDTtzm3vyBEO7RPptvAD+t44WVXaUjTBRcrpFeFlC8WCruUR456hw==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/kleur": {
      "version": "4.1.5",
      "resolved": "https://registry.npmjs.org/kleur/-/kleur-4.1.5.tgz",
      "integrity": "sha512-o+NO+8WrRiQEE4/7nwRJhN1HWpVmJm511pBHUxPLtp0BUISzlBplORYSmTclCnJvQq2tKu/sgl3xVpkc7ZWuQQ==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/lilconfig": {
      "version": "3.1.3",
      "resolved": "https://registry.npmjs.org/lilconfig/-/lilconfig-3.1.3.tgz",
      "integrity": "sha512-/vlFKAoH5Cgt3Ie+JLhRbwOsCQePABiU3tJ1egGvyQ+33R/vcwM2Zl2QR/LzjsBeItPt3oSVXapn+m4nQDvpzw==",
      "engines": {
        "node": ">=14"
      },
      "funding": {
        "url": "https://github.com/sponsors/antonk52"
      }
    },
    "node_modules/lines-and-columns": {
      "version": "1.2.4",
      "resolved": "https://registry.npmjs.org/lines-and-columns/-/lines-and-columns-1.2.4.tgz",
      "integrity": "sha512-7ylylesZQ/PV29jhEDl3Ufjo6ZX7gCqJr5F7PKrqc93v7fzSymt1BpwEU8nAUXs8qzzvqhbjhK5QZg6Mt/HkBg=="
    },
    "node_modules/load-json-file": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/load-json-file/-/load-json-file-4.0.0.tgz",
      "integrity": "sha512-Kx8hMakjX03tiGTLAIdJ+lL0htKnXjEZN6hk/tozf/WOuYGdZBJrZ+rCJRbVCugsjB3jMLn9746NsQIf5VjBMw==",
      "dependencies": {
        "graceful-fs": "^4.1.2",
        "parse-json": "^4.0.0",
        "pify": "^3.0.0",
        "strip-bom": "^3.0.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/load-json-file/node_modules/pify": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/pify/-/pify-3.0.0.tgz",
      "integrity": "sha512-C3FsVNH1udSEX48gGX1xfvwTWfsYWj5U+8/uK15BGzIGrKoUpghX8hWZwa/OFnakBiiVNmBvemTJR5mcy7iPcg==",
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/locate-path": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/locate-path/-/locate-path-2.0.0.tgz",
      "integrity": "sha512-NCI2kiDkyR7VeEKm27Kda/iQHyKJe1Bu0FlTbYp3CqJu+9IFe9bLyAjMxf5ZDDbEg+iMPzB5zYyUTSm8wVTKmA==",
      "dependencies": {
        "p-locate": "^2.0.0",
        "path-exists": "^3.0.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/lodash": {
      "version": "4.17.21",
      "resolved": "https://registry.npmjs.org/lodash/-/lodash-4.17.21.tgz",
      "integrity": "sha512-v2kDEe57lecTulaDIuNTPy3Ry4gLGJ6Z1O3vE1krgXZNrsQ+LFTGHVxVjcXPs17LhbZVGedAJv8XZ1tvj5FvSg=="
    },
    "node_modules/lodash.ismatch": {
      "version": "4.4.0",
      "resolved": "https://registry.npmjs.org/lodash.ismatch/-/lodash.ismatch-4.4.0.tgz",
      "integrity": "sha512-fPMfXjGQEV9Xsq/8MTSgUf255gawYRbjwMyDbcvDhXgV7enSZA0hynz6vMPnpAb5iONEzBHBPsT+0zes5Z301g=="
    },
    "node_modules/lodash.merge": {
      "version": "4.6.2",
      "resolved": "https://registry.npmjs.org/lodash.merge/-/lodash.merge-4.6.2.tgz",
      "integrity": "sha512-0KpjqXRVvrYyCsX1swR/XTK0va6VQkQM6MNo7PqW77ByjAhoARA8EfrP1N4+KlKj8YS0ZUCtRT/YUuhyYDujIQ=="
    },
    "node_modules/lru-cache": {
      "version": "6.0.0",
      "resolved": "https://registry.npmjs.org/lru-cache/-/lru-cache-6.0.0.tgz",
      "integrity": "sha512-Jo6dJ04CmSjuznwJSS3pUeWmd/H0ffTlkXXgwZi+eq1UCmqQwCh+eLsYOYCwY991i2Fah4h1BEMCx4qThGbsiA==",
      "dependencies": {
        "yallist": "^4.0.0"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/make-error": {
      "version": "1.3.6",
      "resolved": "https://registry.npmjs.org/make-error/-/make-error-1.3.6.tgz",
      "integrity": "sha512-s8UhlNe7vPKomQhC1qFelMokr/Sc3AgNbso3n74mVPA5LTZwkB9NlXf4XPamLxJE8h0gh73rM94xvwRT2CVInw=="
    },
    "node_modules/map-obj": {
      "version": "4.3.0",
      "resolved": "https://registry.npmjs.org/map-obj/-/map-obj-4.3.0.tgz",
      "integrity": "sha512-hdN1wVrZbb29eBGiGjJbeP8JbKjq1urkHJ/LIP/NY48MZ1QVXUsQBV1G1zvYFHn1XE06cwjBsOI2K3Ulnj1YXQ==",
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/marked": {
      "version": "15.0.4",
      "resolved": "https://registry.npmjs.org/marked/-/marked-15.0.4.tgz",
      "integrity": "sha512-TCHvDqmb3ZJ4PWG7VEGVgtefA5/euFmsIhxtD0XsBxI39gUSKL81mIRFdt0AiNQozUahd4ke98ZdirExd/vSEw==",
      "bin": {
        "marked": "bin/marked.js"
      },
      "engines": {
        "node": ">= 18"
      }
    },
    "node_modules/meow": {
      "version": "8.1.2",
      "resolved": "https://registry.npmjs.org/meow/-/meow-8.1.2.tgz",
      "integrity": "sha512-r85E3NdZ+mpYk1C6RjPFEMSE+s1iZMuHtsHAqY0DT3jZczl0diWUZ8g6oU7h0M9cD2EL+PzaYghhCLzR0ZNn5Q==",
      "dependencies": {
        "@types/minimist": "^1.2.0",
        "camelcase-keys": "^6.2.2",
        "decamelize-keys": "^1.1.0",
        "hard-rejection": "^2.1.0",
        "minimist-options": "4.1.0",
        "normalize-package-data": "^3.0.0",
        "read-pkg-up": "^7.0.1",
        "redent": "^3.0.0",
        "trim-newlines": "^3.0.0",
        "type-fest": "^0.18.0",
        "yargs-parser": "^20.2.3"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/meow/node_modules/find-up": {
      "version": "4.1.0",
      "resolved": "https://registry.npmjs.org/find-up/-/find-up-4.1.0.tgz",
      "integrity": "sha512-PpOwAdQ/YlXQ2vj8a3h8IipDuYRi3wceVQQGYWxNINccq40Anw7BlsEXCMbt1Zt+OLA6Fq9suIpIWD0OsnISlw==",
      "dependencies": {
        "locate-path": "^5.0.0",
        "path-exists": "^4.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/meow/node_modules/hosted-git-info": {
      "version": "2.8.9",
      "resolved": "https://registry.npmjs.org/hosted-git-info/-/hosted-git-info-2.8.9.tgz",
      "integrity": "sha512-mxIDAb9Lsm6DoOJ7xH+5+X4y1LU/4Hi50L9C5sIswK3JzULS4bwk1FvjdBgvYR4bzT4tuUQiC15FE2f5HbLvYw=="
    },
    "node_modules/meow/node_modules/locate-path": {
      "version": "5.0.0",
      "resolved": "https://registry.npmjs.org/locate-path/-/locate-path-5.0.0.tgz",
      "integrity": "sha512-t7hw9pI+WvuwNJXwk5zVHpyhIqzg2qTlklJOf0mVxGSbe3Fp2VieZcduNYjaLDoy6p9uGpQEGWG87WpMKlNq8g==",
      "dependencies": {
        "p-locate": "^4.1.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/meow/node_modules/p-limit": {
      "version": "2.3.0",
      "resolved": "https://registry.npmjs.org/p-limit/-/p-limit-2.3.0.tgz",
      "integrity": "sha512-//88mFWSJx8lxCzwdAABTJL2MyWB12+eIY7MDL2SqLmAkeKU9qxRvWuSyTjm3FUmpBEMuFfckAIqEaVGUDxb6w==",
      "dependencies": {
        "p-try": "^2.0.0"
      },
      "engines": {
        "node": ">=6"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/meow/node_modules/p-locate": {
      "version": "4.1.0",
      "resolved": "https://registry.npmjs.org/p-locate/-/p-locate-4.1.0.tgz",
      "integrity": "sha512-R79ZZ/0wAxKGu3oYMlz8jy/kbhsNrS7SKZ7PxEHBgJ5+F2mtFW2fK2cOtBh1cHYkQsbzFV7I+EoRKe6Yt0oK7A==",
      "dependencies": {
        "p-limit": "^2.2.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/meow/node_modules/p-try": {
      "version": "2.2.0",
      "resolved": "https://registry.npmjs.org/p-try/-/p-try-2.2.0.tgz",
      "integrity": "sha512-R4nPAVTAU0B9D35/Gk3uJf/7XYbQcyohSKdvAxIRSNghFl4e71hVoGnBNQz9cWaXxO2I10KTC+3jMdvvoKw6dQ==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/meow/node_modules/parse-json": {
      "version": "5.2.0",
      "resolved": "https://registry.npmjs.org/parse-json/-/parse-json-5.2.0.tgz",
      "integrity": "sha512-ayCKvm/phCGxOkYRSCM82iDwct8/EonSEgCSxWxD7ve6jHggsFl4fZVQBPRNgQoKiuV/odhFrGzQXZwbifC8Rg==",
      "dependencies": {
        "@babel/code-frame": "^7.0.0",
        "error-ex": "^1.3.1",
        "json-parse-even-better-errors": "^2.3.0",
        "lines-and-columns": "^1.1.6"
      },
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/meow/node_modules/path-exists": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/path-exists/-/path-exists-4.0.0.tgz",
      "integrity": "sha512-ak9Qy5Q7jYb2Wwcey5Fpvg2KoAc/ZIhLSLOSBmRmygPsGwkVVt0fZa0qrtMz+m6tJTAHfZQ8FnmB4MG4LWy7/w==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/meow/node_modules/read-pkg": {
      "version": "5.2.0",
      "resolved": "https://registry.npmjs.org/read-pkg/-/read-pkg-5.2.0.tgz",
      "integrity": "sha512-Ug69mNOpfvKDAc2Q8DRpMjjzdtrnv9HcSMX+4VsZxD1aZ6ZzrIE7rlzXBtWTyhULSMKg076AW6WR5iZpD0JiOg==",
      "dependencies": {
        "@types/normalize-package-data": "^2.4.0",
        "normalize-package-data": "^2.5.0",
        "parse-json": "^5.0.0",
        "type-fest": "^0.6.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/meow/node_modules/read-pkg-up": {
      "version": "7.0.1",
      "resolved": "https://registry.npmjs.org/read-pkg-up/-/read-pkg-up-7.0.1.tgz",
      "integrity": "sha512-zK0TB7Xd6JpCLmlLmufqykGE+/TlOePD6qKClNW7hHDKFh/J7/7gCWGR7joEQEW1bKq3a3yUZSObOoWLFQ4ohg==",
      "dependencies": {
        "find-up": "^4.1.0",
        "read-pkg": "^5.2.0",
        "type-fest": "^0.8.1"
      },
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/meow/node_modules/read-pkg-up/node_modules/type-fest": {
      "version": "0.8.1",
      "resolved": "https://registry.npmjs.org/type-fest/-/type-fest-0.8.1.tgz",
      "integrity": "sha512-4dbzIzqvjtgiM5rw1k5rEHtBANKmdudhGyBEajN01fEyhaAIhsoKNy6y7+IN93IfpFtwY9iqi7kD+xwKhQsNJA==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/meow/node_modules/read-pkg/node_modules/normalize-package-data": {
      "version": "2.5.0",
      "resolved": "https://registry.npmjs.org/normalize-package-data/-/normalize-package-data-2.5.0.tgz",
      "integrity": "sha512-/5CMN3T0R4XTj4DcGaexo+roZSdSFW/0AOOTROrjxzCG1wrWXEsGbRKevjlIL+ZDE4sZlJr5ED4YW0yqmkK+eA==",
      "dependencies": {
        "hosted-git-info": "^2.1.4",
        "resolve": "^1.10.0",
        "semver": "2 || 3 || 4 || 5",
        "validate-npm-package-license": "^3.0.1"
      }
    },
    "node_modules/meow/node_modules/read-pkg/node_modules/type-fest": {
      "version": "0.6.0",
      "resolved": "https://registry.npmjs.org/type-fest/-/type-fest-0.6.0.tgz",
      "integrity": "sha512-q+MB8nYR1KDLrgr4G5yemftpMC7/QLqVndBmEEdqzmNj5dcFOO4Oo8qlwZE3ULT3+Zim1F8Kq4cBnikNhlCMlg==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/meow/node_modules/semver": {
      "version": "5.7.2",
      "resolved": "https://registry.npmjs.org/semver/-/semver-5.7.2.tgz",
      "integrity": "sha512-cBznnQ9KjJqU67B52RMC65CMarK2600WFnbkcaiwWq3xy/5haFJlshgnpjovMVJ+Hff49d8GEn0b87C5pDQ10g==",
      "bin": {
        "semver": "bin/semver"
      }
    },
    "node_modules/meow/node_modules/type-fest": {
      "version": "0.18.1",
      "resolved": "https://registry.npmjs.org/type-fest/-/type-fest-0.18.1.tgz",
      "integrity": "sha512-OIAYXk8+ISY+qTOwkHtKqzAuxchoMiD9Udx+FSGQDuiRR+PJKJHc2NJAXlbhkGwTt/4/nKZxELY1w3ReWOL8mw==",
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/merge-stream": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/merge-stream/-/merge-stream-2.0.0.tgz",
      "integrity": "sha512-abv/qOcuPfk3URPfDzmZU1LKmuw8kT+0nIHvKrKgFrwifol/doWcdA4ZqsWQ8ENrFKkd67Mfpo/LovbIUsbt3w==",
      "dev": true
    },
    "node_modules/merge2": {
      "version": "1.4.1",
      "resolved": "https://registry.npmjs.org/merge2/-/merge2-1.4.1.tgz",
      "integrity": "sha512-8q7VEgMJW4J8tcfVPy8g09NcQwZdbwFEqhe/WZkoIzjn/3TGDwtOCYtXGxA3O8tPzpczCCDgv+P2P5y00ZJOOg==",
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/mergexml": {
      "version": "1.2.4",
      "resolved": "https://registry.npmjs.org/mergexml/-/mergexml-1.2.4.tgz",
      "integrity": "sha512-yiOlDqcVCz7AG1eSboonc18FTlfqDEKYfGoAV3Lul98u6YRV/s0kjtf4bjk47t0hLTFJR0BSYMd6BpmX3xDjNQ==",
      "dependencies": {
        "@xmldom/xmldom": "^0.7.0",
        "formidable": "^3.5.1",
        "xpath": "0.0.27"
      }
    },
    "node_modules/mergexml/node_modules/xpath": {
      "version": "0.0.27",
      "resolved": "https://registry.npmjs.org/xpath/-/xpath-0.0.27.tgz",
      "integrity": "sha512-fg03WRxtkCV6ohClePNAECYsmpKKTv5L8y/X3Dn1hQrec3POx2jHZ/0P2qQ6HvsrU1BmeqXcof3NGGueG6LxwQ==",
      "engines": {
        "node": ">=0.6.0"
      }
    },
    "node_modules/micromatch": {
      "version": "4.0.8",
      "resolved": "https://registry.npmjs.org/micromatch/-/micromatch-4.0.8.tgz",
      "integrity": "sha512-PXwfBhYu0hBCPw8Dn0E+WDYb7af3dSLVWKi3HGv84IdF4TyFoC0ysxFd0Goxw7nSv4T/PzEJQxsYsEiFCKo2BA==",
      "dependencies": {
        "braces": "^3.0.3",
        "picomatch": "^2.3.1"
      },
      "engines": {
        "node": ">=8.6"
      }
    },
    "node_modules/mime-db": {
      "version": "1.53.0",
      "resolved": "https://registry.npmjs.org/mime-db/-/mime-db-1.53.0.tgz",
      "integrity": "sha512-oHlN/w+3MQ3rba9rqFr6V/ypF10LSkdwUysQL7GkXoTgIWeV+tcXGA852TBxH+gsh8UWoyhR1hKcoMJTuWflpg==",
      "dev": true,
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/mime-types": {
      "version": "2.1.35",
      "resolved": "https://registry.npmjs.org/mime-types/-/mime-types-2.1.35.tgz",
      "integrity": "sha512-ZDY+bPm5zTTF+YpCrAU9nK0UgICYPT0QtT1NZWFv4s++TNkcgVaT0g6+4R2uI4MjQjzysHB1zxuWL50hzaeXiw==",
      "dev": true,
      "dependencies": {
        "mime-db": "1.52.0"
      },
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/mime-types/node_modules/mime-db": {
      "version": "1.52.0",
      "resolved": "https://registry.npmjs.org/mime-db/-/mime-db-1.52.0.tgz",
      "integrity": "sha512-sPU4uV7dYlvtWJxwwxHD0PuihVNiE7TyAbQ5SWxDCB9mUYvOgroQOwYQQOKPJ8CIbE+1ETVlOoK1UC2nU3gYvg==",
      "dev": true,
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/mimic-fn": {
      "version": "2.1.0",
      "resolved": "https://registry.npmjs.org/mimic-fn/-/mimic-fn-2.1.0.tgz",
      "integrity": "sha512-OqbOk5oEQeAZ8WXWydlu9HJjz9WVdEIvamMCcXmuqUYjTknH/sqsWvhQ3vgwKFRR1HpjvNBKQ37nbJgYzGqGcg==",
      "dev": true,
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/mimic-response": {
      "version": "3.1.0",
      "resolved": "https://registry.npmjs.org/mimic-response/-/mimic-response-3.1.0.tgz",
      "integrity": "sha512-z0yWI+4FDrrweS8Zmt4Ej5HdJmky15+L2e6Wgn3+iK5fWzb6T3fhNFq2+MeTRb064c6Wr4N/wv0DzQTjNzHNGQ==",
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/min-indent": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/min-indent/-/min-indent-1.0.1.tgz",
      "integrity": "sha512-I9jwMn07Sy/IwOj3zVkVik2JTvgpaykDZEigL6Rx6N9LbMywwUSMtxET+7lVoDLLd3O3IXwJwvuuns8UB/HeAg==",
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/minimatch": {
      "version": "3.0.5",
      "resolved": "https://registry.npmjs.org/minimatch/-/minimatch-3.0.5.tgz",
      "integrity": "sha512-tUpxzX0VAzJHjLu0xUfFv1gwVp9ba3IOuRAVH2EGuRW8a5emA2FlACLqiT/lDVtS1W+TGNwqz3sWaNyLgDJWuw==",
      "dependencies": {
        "brace-expansion": "^1.1.7"
      },
      "engines": {
        "node": "*"
      }
    },
    "node_modules/minimatch/node_modules/brace-expansion": {
      "version": "1.1.11",
      "resolved": "https://registry.npmjs.org/brace-expansion/-/brace-expansion-1.1.11.tgz",
      "integrity": "sha512-iCuPHDFgrHX7H2vEI/5xpz07zSHB00TpugqhmYtVmMO6518mCuRMoOYFldEBl0g187ufozdaHgWKcYFb61qGiA==",
      "dependencies": {
        "balanced-match": "^1.0.0",
        "concat-map": "0.0.1"
      }
    },
    "node_modules/minimist": {
      "version": "1.2.8",
      "resolved": "https://registry.npmjs.org/minimist/-/minimist-1.2.8.tgz",
      "integrity": "sha512-2yyAR8qBkN3YuheJanUpWC5U3bb5osDywNB8RzDVlDwDHbocAJveqqj1u8+SVD7jkWT4yvsHCpWqqWqAxb0zCA==",
      "funding": {
        "url": "https://github.com/sponsors/ljharb"
      }
    },
    "node_modules/minimist-options": {
      "version": "4.1.0",
      "resolved": "https://registry.npmjs.org/minimist-options/-/minimist-options-4.1.0.tgz",
      "integrity": "sha512-Q4r8ghd80yhO/0j1O3B2BjweX3fiHg9cdOwjJd2J76Q135c+NDxGCqdYKQ1SKBuFfgWbAUzBfvYjPUEeNgqN1A==",
      "dependencies": {
        "arrify": "^1.0.1",
        "is-plain-obj": "^1.1.0",
        "kind-of": "^6.0.3"
      },
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/minipass": {
      "version": "4.2.8",
      "resolved": "https://registry.npmjs.org/minipass/-/minipass-4.2.8.tgz",
      "integrity": "sha512-fNzuVyifolSLFL4NzpF+wEF4qrgqaaKX0haXPQEdQ7NKAN+WecoKMHV09YcuL/DHxrUsYQOK3MiuDf7Ip2OXfQ==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/minizlib": {
      "version": "2.1.2",
      "resolved": "https://registry.npmjs.org/minizlib/-/minizlib-2.1.2.tgz",
      "integrity": "sha512-bAxsR8BVfj60DWXHE3u30oHzfl4G7khkSuPW+qvpd7jFRHm7dLxOjUk1EHACJ/hxLY8phGJ0YhYHZo7jil7Qdg==",
      "dependencies": {
        "minipass": "^3.0.0",
        "yallist": "^4.0.0"
      },
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/minizlib/node_modules/minipass": {
      "version": "3.3.6",
      "resolved": "https://registry.npmjs.org/minipass/-/minipass-3.3.6.tgz",
      "integrity": "sha512-DxiNidxSEK+tHG6zOIklvNOwm3hvCrbUrdtzY74U6HKTJxvIDfOUL5W5P2Ghd3DTkhhKPYGqeNUIh5qcM4YBfw==",
      "dependencies": {
        "yallist": "^4.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/mkdirp": {
      "version": "1.0.4",
      "resolved": "https://registry.npmjs.org/mkdirp/-/mkdirp-1.0.4.tgz",
      "integrity": "sha512-vVqVZQyf3WLx2Shd0qJ9xuvqgAyKPLAiqITEtqW0oIUjzo3PePDd6fW9iFz30ef7Ysp/oiWqbhszeGWW2T6Gzw==",
      "bin": {
        "mkdirp": "bin/cmd.js"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/mkdirp-classic": {
      "version": "0.5.3",
      "resolved": "https://registry.npmjs.org/mkdirp-classic/-/mkdirp-classic-0.5.3.tgz",
      "integrity": "sha512-gKLcREMhtuZRwRAfqP3RFW+TK4JqApVBtOIftVgjuABpAtpxhPGaDcfvbhNvD0B8iD1oUr/txX35NjcaY6Ns/A=="
    },
    "node_modules/modify-values": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/modify-values/-/modify-values-1.0.1.tgz",
      "integrity": "sha512-xV2bxeN6F7oYjZWTe/YPAy6MN2M+sL4u/Rlm2AHCIVGfo2p1yGmBHQ6vHehl4bRTZBdHu3TSkWdYgkwpYzAGSw==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/ms": {
      "version": "2.1.2",
      "resolved": "https://registry.npmjs.org/ms/-/ms-2.1.2.tgz",
      "integrity": "sha512-sGkPx+VjMtmA6MX27oA4FBFELFCZZ4S4XqeGOXCv68tT+jb3vk/RyaKWP0PTKyWtmLSM0b+adUTEvbs1PEaH2w=="
    },
    "node_modules/mz": {
      "version": "2.7.0",
      "resolved": "https://registry.npmjs.org/mz/-/mz-2.7.0.tgz",
      "integrity": "sha512-z81GNO7nnYMEhrGh9LeymoE4+Yr0Wn5McHIZMK5cfQCl+NDX08sCZgUc9/6MHni9IWuFLm1Z3HTCXu2z9fN62Q==",
      "dependencies": {
        "any-promise": "^1.0.0",
        "object-assign": "^4.0.1",
        "thenify-all": "^1.0.0"
      }
    },
    "node_modules/nanoid": {
      "version": "3.3.8",
      "resolved": "https://registry.npmjs.org/nanoid/-/nanoid-3.3.8.tgz",
      "integrity": "sha512-WNLf5Sd8oZxOm+TzppcYk8gVOgP+l58xNy58D0nbUnOxOWRWvlcCV4kUF7ltmI6PsrLl/BgKEyS4mqsGChFN0w==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "bin": {
        "nanoid": "bin/nanoid.cjs"
      },
      "engines": {
        "node": "^10 || ^12 || ^13.7 || ^14 || >=15.0.1"
      }
    },
    "node_modules/napi-build-utils": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/napi-build-utils/-/napi-build-utils-2.0.0.tgz",
      "integrity": "sha512-GEbrYkbfF7MoNaoh2iGG84Mnf/WZfB0GdGEsM8wz7Expx/LlWf5U8t9nvJKXSp3qr5IsEbK04cBGhol/KwOsWA=="
    },
    "node_modules/native-run": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/native-run/-/native-run-2.0.1.tgz",
      "integrity": "sha512-XfG1FBZLM50J10xH9361whJRC9SHZ0Bub4iNRhhI61C8Jv0e1ud19muex6sNKB51ibQNUJNuYn25MuYET/rE6w==",
      "dependencies": {
        "@ionic/utils-fs": "^3.1.7",
        "@ionic/utils-terminal": "^2.3.4",
        "bplist-parser": "^0.3.2",
        "debug": "^4.3.4",
        "elementtree": "^0.1.7",
        "ini": "^4.1.1",
        "plist": "^3.1.0",
        "split2": "^4.2.0",
        "through2": "^4.0.2",
        "tslib": "^2.6.2",
        "yauzl": "^2.10.0"
      },
      "bin": {
        "native-run": "bin/native-run"
      },
      "engines": {
        "node": ">=16.0.0"
      }
    },
    "node_modules/native-run/node_modules/ini": {
      "version": "4.1.3",
      "resolved": "https://registry.npmjs.org/ini/-/ini-4.1.3.tgz",
      "integrity": "sha512-X7rqawQBvfdjS10YU1y1YVreA3SsLrW9dX2CewP2EbBJM4ypVNLDkO5y04gejPwKIY9lR+7r9gn3rFPt/kmWFg==",
      "engines": {
        "node": "^14.17.0 || ^16.13.0 || >=18.0.0"
      }
    },
    "node_modules/native-run/node_modules/split2": {
      "version": "4.2.0",
      "resolved": "https://registry.npmjs.org/split2/-/split2-4.2.0.tgz",
      "integrity": "sha512-UcjcJOWknrNkF6PLX83qcHM6KHgVKNkV62Y8a5uYDVv9ydGQVwAHMKqHdJje1VTWpljG0WYpCDhrCdAOYH4TWg==",
      "engines": {
        "node": ">= 10.x"
      }
    },
    "node_modules/negotiator": {
      "version": "0.6.3",
      "resolved": "https://registry.npmjs.org/negotiator/-/negotiator-0.6.3.tgz",
      "integrity": "sha512-+EUsqGPLsM+j/zdChZjsnX51g4XrHFOIXwfnCVPGlQk/k5giakcKsuxCObBRu6DSm9opw/O6slWbJdghQM4bBg==",
      "dev": true,
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/neo-async": {
      "version": "2.6.2",
      "resolved": "https://registry.npmjs.org/neo-async/-/neo-async-2.6.2.tgz",
      "integrity": "sha512-Yd3UES5mWCSqR+qNT93S3UoYUkqAZ9lLg8a7g9rimsWmYGK8cVToA4/sF3RrshdyV3sAGMXVUmpMYOw+dLpOuw=="
    },
    "node_modules/node-abi": {
      "version": "3.74.0",
      "resolved": "https://registry.npmjs.org/node-abi/-/node-abi-3.74.0.tgz",
      "integrity": "sha512-c5XK0MjkGBrQPGYG24GBADZud0NCbznxNx0ZkS+ebUTrmV1qTDxPxSL8zEAPURXSbLRWVexxmP4986BziahL5w==",
      "dependencies": {
        "semver": "^7.3.5"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/node-addon-api": {
      "version": "6.1.0",
      "resolved": "https://registry.npmjs.org/node-addon-api/-/node-addon-api-6.1.0.tgz",
      "integrity": "sha512-+eawOlIgy680F0kBzPUNFhMZGtJ1YmqM6l4+Crf4IkImjYrO/mqPwRMh352g23uIaQKFItcQ64I7KMaJxHgAVA=="
    },
    "node_modules/node-fetch": {
      "version": "2.7.0",
      "resolved": "https://registry.npmjs.org/node-fetch/-/node-fetch-2.7.0.tgz",
      "integrity": "sha512-c4FRfUm/dbcWZ7U+1Wq0AwCyFL+3nt2bEw05wfxSz+DWpWsitgmSgYmy2dQdWyKC1694ELPqMs/YzUSNozLt8A==",
      "dependencies": {
        "whatwg-url": "^5.0.0"
      },
      "engines": {
        "node": "4.x || >=6.0.0"
      },
      "peerDependencies": {
        "encoding": "^0.1.0"
      },
      "peerDependenciesMeta": {
        "encoding": {
          "optional": true
        }
      }
    },
    "node_modules/node-html-parser": {
      "version": "5.4.2",
      "resolved": "https://registry.npmjs.org/node-html-parser/-/node-html-parser-5.4.2.tgz",
      "integrity": "sha512-RaBPP3+51hPne/OolXxcz89iYvQvKOydaqoePpOgXcrOKZhjVIzmpKZz+Hd/RBO2/zN2q6CNJhQzucVz+u3Jyw==",
      "dependencies": {
        "css-select": "^4.2.1",
        "he": "1.2.0"
      }
    },
    "node_modules/node-releases": {
      "version": "2.0.19",
      "resolved": "https://registry.npmjs.org/node-releases/-/node-releases-2.0.19.tgz",
      "integrity": "sha512-xxOWJsBKtzAq7DY0J+DTzuz58K8e7sJbdgwkbMWQe8UYB6ekmsQ45q0M/tJDsGaZmbC+l7n57UV8Hl5tHxO9uw==",
      "dev": true
    },
    "node_modules/normalize-package-data": {
      "version": "3.0.3",
      "resolved": "https://registry.npmjs.org/normalize-package-data/-/normalize-package-data-3.0.3.tgz",
      "integrity": "sha512-p2W1sgqij3zMMyRC067Dg16bfzVH+w7hyegmpIvZ4JNjqtGOVAIvLmjBx3yP7YTe9vKJgkoNOPjwQGogDoMXFA==",
      "dependencies": {
        "hosted-git-info": "^4.0.1",
        "is-core-module": "^2.5.0",
        "semver": "^7.3.4",
        "validate-npm-package-license": "^3.0.1"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/normalize-path": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/normalize-path/-/normalize-path-3.0.0.tgz",
      "integrity": "sha512-6eZs5Ls3WtCisHWp9S2GUy8dqkpGi4BVSz3GaqiE6ezub0512ESztXUwUB6C6IKbQkY2Pnb/mD4WYojCRwcwLA==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/normalize-range": {
      "version": "0.1.2",
      "resolved": "https://registry.npmjs.org/normalize-range/-/normalize-range-0.1.2.tgz",
      "integrity": "sha512-bdok/XvKII3nUpklnV6P2hxtMNrCboOjAcyBuQnWEhO665FwrSNRxU+AqpsyvO6LgGYPspN+lu5CLtw4jPRKNA==",
      "dev": true,
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/npm-run-path": {
      "version": "4.0.1",
      "resolved": "https://registry.npmjs.org/npm-run-path/-/npm-run-path-4.0.1.tgz",
      "integrity": "sha512-S48WzZW777zhNIrn7gxOlISNAqi9ZC/uQFnRdbeIHhZhCA6UqpkOT8T1G7BvfdgP4Er8gF4sUbaS0i7QvIfCWw==",
      "dev": true,
      "dependencies": {
        "path-key": "^3.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/nth-check": {
      "version": "2.1.1",
      "resolved": "https://registry.npmjs.org/nth-check/-/nth-check-2.1.1.tgz",
      "integrity": "sha512-lqjrjmaOoAnWfMmBPL+XNnynZh2+swxiX3WUE0s4yEHI6m+AwrK2UZOimIRl3X/4QctVqS8AiZjFqyOGrMXb/w==",
      "dependencies": {
        "boolbase": "^1.0.0"
      },
      "funding": {
        "url": "https://github.com/fb55/nth-check?sponsor=1"
      }
    },
    "node_modules/object-assign": {
      "version": "4.1.1",
      "resolved": "https://registry.npmjs.org/object-assign/-/object-assign-4.1.1.tgz",
      "integrity": "sha512-rJgTQnkUnH1sFw8yT6VSU3zD3sWmu6sZhIseY8VX+GRu3P6F7Fu+JNDoXfklElbLJSnc3FUQHVe4cU5hj+BcUg==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/object-hash": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/object-hash/-/object-hash-3.0.0.tgz",
      "integrity": "sha512-RSn9F68PjH9HqtltsSnqYC1XXoWe9Bju5+213R98cNGttag9q9yAOTzdbsqvIa7aNm5WffBZFpWYr2aWrklWAw==",
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/on-headers": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/on-headers/-/on-headers-1.0.2.tgz",
      "integrity": "sha512-pZAE+FJLoyITytdqK0U5s+FIpjN0JP3OzFi/u8Rx+EV5/W+JTWGXG8xFzevE7AjBfDqHv/8vL8qQsIhHnqRkrA==",
      "dev": true,
      "engines": {
        "node": ">= 0.8"
      }
    },
    "node_modules/once": {
      "version": "1.4.0",
      "resolved": "https://registry.npmjs.org/once/-/once-1.4.0.tgz",
      "integrity": "sha512-lNaJgI+2Q5URQBkccEKHTQOPaXdUxnZZElQTZY0MFUAuaEqe1E+Nyvgdz/aIyNi6Z9MzO5dv1H8n58/GELp3+w==",
      "dependencies": {
        "wrappy": "1"
      }
    },
    "node_modules/onetime": {
      "version": "5.1.2",
      "resolved": "https://registry.npmjs.org/onetime/-/onetime-5.1.2.tgz",
      "integrity": "sha512-kbpaSSGJTWdAY5KPVeMOKXSrPtr8C8C7wodJbcsd51jRnmD+GZu8Y0VoU6Dm5Z4vWr0Ig/1NKuWRKf7j5aaYSg==",
      "dev": true,
      "dependencies": {
        "mimic-fn": "^2.1.0"
      },
      "engines": {
        "node": ">=6"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/open": {
      "version": "8.4.2",
      "resolved": "https://registry.npmjs.org/open/-/open-8.4.2.tgz",
      "integrity": "sha512-7x81NCL719oNbsq/3mh+hVrAWmFuEYUqrq/Iw3kUzH8ReypT9QQ0BLoJS7/G9k6N81XjW4qHWtjWwe/9eLy1EQ==",
      "dependencies": {
        "define-lazy-prop": "^2.0.0",
        "is-docker": "^2.1.1",
        "is-wsl": "^2.2.0"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/p-limit": {
      "version": "1.3.0",
      "resolved": "https://registry.npmjs.org/p-limit/-/p-limit-1.3.0.tgz",
      "integrity": "sha512-vvcXsLAJ9Dr5rQOPk7toZQZJApBl2K4J6dANSsEuh6QI41JYcsS/qhTGa9ErIUUgK3WNQoJYvylxvjqmiqEA9Q==",
      "dependencies": {
        "p-try": "^1.0.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/p-locate": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/p-locate/-/p-locate-2.0.0.tgz",
      "integrity": "sha512-nQja7m7gSKuewoVRen45CtVfODR3crN3goVQ0DDZ9N3yHxgpkuBhZqsaiotSQRrADUrne346peY7kT3TSACykg==",
      "dependencies": {
        "p-limit": "^1.1.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/p-map": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/p-map/-/p-map-4.0.0.tgz",
      "integrity": "sha512-/bjOqmgETBYB5BoEeGVea8dmvHb2m9GLy1E9W43yeyfP6QQCZGFNa+XRceJEuDB6zqr+gKpIAmlLebMpykw/MQ==",
      "dependencies": {
        "aggregate-error": "^3.0.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/p-try": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/p-try/-/p-try-1.0.0.tgz",
      "integrity": "sha512-U1etNYuMJoIz3ZXSrrySFjsXQTWOx2/jdi86L+2pRvph/qMKL6sbcCYdH23fqsbm8TH2Gn0OybpT4eSFlCVHww==",
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/package-json-from-dist": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/package-json-from-dist/-/package-json-from-dist-1.0.1.tgz",
      "integrity": "sha512-UEZIS3/by4OC8vL3P2dTXRETpebLI2NiI5vIrjaD/5UtrkFX/tNbwjTSRAGC/+7CAo2pIcBaRgWmcBBHcsaCIw=="
    },
    "node_modules/parse-json": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/parse-json/-/parse-json-4.0.0.tgz",
      "integrity": "sha512-aOIos8bujGN93/8Ox/jPLh7RwVnPEysynVFE+fQZyg6jKELEHwzgKdLRFHUgXJL6kylijVSBC4BvN9OmsB48Rw==",
      "dependencies": {
        "error-ex": "^1.3.1",
        "json-parse-better-errors": "^1.0.1"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/path-exists": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/path-exists/-/path-exists-3.0.0.tgz",
      "integrity": "sha512-bpC7GYwiDYQ4wYLe+FA8lhRjhQCMcQGuSgGGqDkg/QerRWw9CmGRT0iSOVRSZJ29NMLZgIzqaljJ63oaL4NIJQ==",
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/path-is-absolute": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/path-is-absolute/-/path-is-absolute-1.0.1.tgz",
      "integrity": "sha512-AVbw3UJ2e9bq64vSaS9Am0fje1Pa8pbGqTTsmXfaIiMpnr5DlDhfJOuLj9Sf95ZPVDAUerDfEk88MPmPe7UCQg==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/path-is-inside": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/path-is-inside/-/path-is-inside-1.0.2.tgz",
      "integrity": "sha512-DUWJr3+ULp4zXmol/SZkFf3JGsS9/SIv+Y3Rt93/UjPpDpklB5f1er4O3POIbUuUJ3FXgqte2Q7SrU6zAqwk8w==",
      "dev": true
    },
    "node_modules/path-key": {
      "version": "3.1.1",
      "resolved": "https://registry.npmjs.org/path-key/-/path-key-3.1.1.tgz",
      "integrity": "sha512-ojmeN0qd+y0jszEtoY48r0Peq5dwMEkIlCOu6Q5f41lfkswXuKtYrhgoTpLnyIcHm24Uhqx+5Tqm2InSwLhE6Q==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/path-parse": {
      "version": "1.0.7",
      "resolved": "https://registry.npmjs.org/path-parse/-/path-parse-1.0.7.tgz",
      "integrity": "sha512-LDJzPVEEEPR+y48z93A0Ed0yXb8pAByGWo/k5YYdYgpY2/2EsOsksJrq7lOHxryrVOn1ejG6oAp8ahvOIQD8sw=="
    },
    "node_modules/path-scurry": {
      "version": "1.11.1",
      "resolved": "https://registry.npmjs.org/path-scurry/-/path-scurry-1.11.1.tgz",
      "integrity": "sha512-Xa4Nw17FS9ApQFJ9umLiJS4orGjm7ZzwUrwamcGQuHSzDyth9boKDaycYdDcZDuqYATXw4HFXgaqWTctW/v1HA==",
      "dependencies": {
        "lru-cache": "^10.2.0",
        "minipass": "^5.0.0 || ^6.0.2 || ^7.0.0"
      },
      "engines": {
        "node": ">=16 || 14 >=14.18"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/path-scurry/node_modules/lru-cache": {
      "version": "10.4.3",
      "resolved": "https://registry.npmjs.org/lru-cache/-/lru-cache-10.4.3.tgz",
      "integrity": "sha512-JNAzZcXrCt42VGLuYz0zfAzDfAvJWW6AfYlDBQyDV5DClI2m5sAmK+OIO7s59XfsRsWHp02jAJrRadPRGTt6SQ=="
    },
    "node_modules/path-scurry/node_modules/minipass": {
      "version": "7.1.2",
      "resolved": "https://registry.npmjs.org/minipass/-/minipass-7.1.2.tgz",
      "integrity": "sha512-qOOzS1cBTWYF4BH8fVePDBOO9iptMnGUEZwNc/cMWnTV2nVLZ7VoNWEPHkYczZA0pdoA7dl6e7FL659nX9S2aw==",
      "engines": {
        "node": ">=16 || 14 >=14.17"
      }
    },
    "node_modules/path-to-regexp": {
      "version": "3.3.0",
      "resolved": "https://registry.npmjs.org/path-to-regexp/-/path-to-regexp-3.3.0.tgz",
      "integrity": "sha512-qyCH421YQPS2WFDxDjftfc1ZR5WKQzVzqsp4n9M2kQhVOo/ByahFoUNJfl58kOcEGfQ//7weFTDhm+ss8Ecxgw==",
      "dev": true
    },
    "node_modules/path-type": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/path-type/-/path-type-3.0.0.tgz",
      "integrity": "sha512-T2ZUsdZFHgA3u4e5PfPbjd7HDDpxPnQb5jN0SrDsjNSuVXHJqtwTnWqG0B1jZrgmJ/7lj1EmVIByWt1gxGkWvg==",
      "dependencies": {
        "pify": "^3.0.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/path-type/node_modules/pify": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/pify/-/pify-3.0.0.tgz",
      "integrity": "sha512-C3FsVNH1udSEX48gGX1xfvwTWfsYWj5U+8/uK15BGzIGrKoUpghX8hWZwa/OFnakBiiVNmBvemTJR5mcy7iPcg==",
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/pend": {
      "version": "1.2.0",
      "resolved": "https://registry.npmjs.org/pend/-/pend-1.2.0.tgz",
      "integrity": "sha512-F3asv42UuXchdzt+xXqfW1OGlVBe+mxa2mqI0pg5yAHZPvFmY3Y6drSf/GQ1A86WgWEN9Kzh/WrgKa6iGcHXLg=="
    },
    "node_modules/picocolors": {
      "version": "1.1.1",
      "resolved": "https://registry.npmjs.org/picocolors/-/picocolors-1.1.1.tgz",
      "integrity": "sha512-xceH2snhtb5M9liqDsmEw56le376mTZkEX/jEb/RxNFyegNul7eNslCXP9FDj/Lcu0X8KEyMceP2ntpaHrDEVA=="
    },
    "node_modules/picomatch": {
      "version": "2.3.1",
      "resolved": "https://registry.npmjs.org/picomatch/-/picomatch-2.3.1.tgz",
      "integrity": "sha512-JU3teHTNjmE2VCGFzuY8EXzCDVwEqB2a8fsIvwaStHhAWJEeVd1o1QD80CU6+ZdEXXSLbSsuLwJjkCBWqRQUVA==",
      "engines": {
        "node": ">=8.6"
      },
      "funding": {
        "url": "https://github.com/sponsors/jonschlinkert"
      }
    },
    "node_modules/pify": {
      "version": "2.3.0",
      "resolved": "https://registry.npmjs.org/pify/-/pify-2.3.0.tgz",
      "integrity": "sha512-udgsAY+fTnvv7kI7aaxbqwWNb0AHiB0qBO89PZKPkoTmGOgdbrHDKD+0B2X4uTfJ/FT1R09r9gTsjUjNJotuog==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/pirates": {
      "version": "4.0.6",
      "resolved": "https://registry.npmjs.org/pirates/-/pirates-4.0.6.tgz",
      "integrity": "sha512-saLsH7WeYYPiD25LDuLRRY/i+6HaPYr6G1OUlN39otzkSTxKnubR9RTxS3/Kk50s1g2JTgFwWQDQyplC5/SHZg==",
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/plist": {
      "version": "3.1.0",
      "resolved": "https://registry.npmjs.org/plist/-/plist-3.1.0.tgz",
      "integrity": "sha512-uysumyrvkUX0rX/dEVqt8gC3sTBzd4zoWfLeS29nb53imdaXVvLINYXTI2GNqzaMuvacNx4uJQ8+b3zXR0pkgQ==",
      "dependencies": {
        "@xmldom/xmldom": "^0.8.8",
        "base64-js": "^1.5.1",
        "xmlbuilder": "^15.1.1"
      },
      "engines": {
        "node": ">=10.4.0"
      }
    },
    "node_modules/plist/node_modules/@xmldom/xmldom": {
      "version": "0.8.10",
      "resolved": "https://registry.npmjs.org/@xmldom/xmldom/-/xmldom-0.8.10.tgz",
      "integrity": "sha512-2WALfTl4xo2SkGCYRt6rDTFfk9R1czmBvUQy12gK2KuRKIpWEhcbbzy8EZXtz/jkRqHX8bFEc6FC1HjX4TUWYw==",
      "engines": {
        "node": ">=10.0.0"
      }
    },
    "node_modules/postcss": {
      "version": "8.4.49",
      "resolved": "https://registry.npmjs.org/postcss/-/postcss-8.4.49.tgz",
      "integrity": "sha512-OCVPnIObs4N29kxTjzLfUryOkvZEq+pf8jTF0lg8E7uETuWHA+v7j3c/xJmiqpX450191LlmZfUKkXxkTry7nA==",
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/postcss/"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/postcss"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "dependencies": {
        "nanoid": "^3.3.7",
        "picocolors": "^1.1.1",
        "source-map-js": "^1.2.1"
      },
      "engines": {
        "node": "^10 || ^12 || >=14"
      }
    },
    "node_modules/postcss-import": {
      "version": "15.1.0",
      "resolved": "https://registry.npmjs.org/postcss-import/-/postcss-import-15.1.0.tgz",
      "integrity": "sha512-hpr+J05B2FVYUAXHeK1YyI267J/dDDhMU6B6civm8hSY1jYJnBXxzKDKDswzJmtLHryrjhnDjqqp/49t8FALew==",
      "dependencies": {
        "postcss-value-parser": "^4.0.0",
        "read-cache": "^1.0.0",
        "resolve": "^1.1.7"
      },
      "engines": {
        "node": ">=14.0.0"
      },
      "peerDependencies": {
        "postcss": "^8.0.0"
      }
    },
    "node_modules/postcss-js": {
      "version": "4.0.1",
      "resolved": "https://registry.npmjs.org/postcss-js/-/postcss-js-4.0.1.tgz",
      "integrity": "sha512-dDLF8pEO191hJMtlHFPRa8xsizHaM82MLfNkUHdUtVEV3tgTp5oj+8qbEqYM57SLfc74KSbw//4SeJma2LRVIw==",
      "dependencies": {
        "camelcase-css": "^2.0.1"
      },
      "engines": {
        "node": "^12 || ^14 || >= 16"
      },
      "funding": {
        "type": "opencollective",
        "url": "https://opencollective.com/postcss/"
      },
      "peerDependencies": {
        "postcss": "^8.4.21"
      }
    },
    "node_modules/postcss-load-config": {
      "version": "4.0.2",
      "resolved": "https://registry.npmjs.org/postcss-load-config/-/postcss-load-config-4.0.2.tgz",
      "integrity": "sha512-bSVhyJGL00wMVoPUzAVAnbEoWyqRxkjv64tUl427SKnPrENtq6hJwUojroMz2VB+Q1edmi4IfrAPpami5VVgMQ==",
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/postcss/"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "dependencies": {
        "lilconfig": "^3.0.0",
        "yaml": "^2.3.4"
      },
      "engines": {
        "node": ">= 14"
      },
      "peerDependencies": {
        "postcss": ">=8.0.9",
        "ts-node": ">=9.0.0"
      },
      "peerDependenciesMeta": {
        "postcss": {
          "optional": true
        },
        "ts-node": {
          "optional": true
        }
      }
    },
    "node_modules/postcss-nested": {
      "version": "6.2.0",
      "resolved": "https://registry.npmjs.org/postcss-nested/-/postcss-nested-6.2.0.tgz",
      "integrity": "sha512-HQbt28KulC5AJzG+cZtj9kvKB93CFCdLvog1WFLf1D+xmMvPGlBstkpTEZfK5+AN9hfJocyBFCNiqyS48bpgzQ==",
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/postcss/"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "dependencies": {
        "postcss-selector-parser": "^6.1.1"
      },
      "engines": {
        "node": ">=12.0"
      },
      "peerDependencies": {
        "postcss": "^8.2.14"
      }
    },
    "node_modules/postcss-selector-parser": {
      "version": "6.1.2",
      "resolved": "https://registry.npmjs.org/postcss-selector-parser/-/postcss-selector-parser-6.1.2.tgz",
      "integrity": "sha512-Q8qQfPiZ+THO/3ZrOrO0cJJKfpYCagtMUkXbnEfmgUjwXg6z/WBeOyS9APBBPCTSiDV+s4SwQGu8yFsiMRIudg==",
      "dependencies": {
        "cssesc": "^3.0.0",
        "util-deprecate": "^1.0.2"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/postcss-value-parser": {
      "version": "4.2.0",
      "resolved": "https://registry.npmjs.org/postcss-value-parser/-/postcss-value-parser-4.2.0.tgz",
      "integrity": "sha512-1NNCs6uurfkVbeXG4S8JFT9t19m45ICnif8zWLd5oPSZ50QnwMfK+H3jv408d4jw/7Bttv5axS5IiHoLaVNHeQ=="
    },
    "node_modules/prebuild-install": {
      "version": "7.1.3",
      "resolved": "https://registry.npmjs.org/prebuild-install/-/prebuild-install-7.1.3.tgz",
      "integrity": "sha512-8Mf2cbV7x1cXPUILADGI3wuhfqWvtiLA1iclTDbFRZkgRQS0NqsPZphna9V+HyTEadheuPmjaJMsbzKQFOzLug==",
      "dependencies": {
        "detect-libc": "^2.0.0",
        "expand-template": "^2.0.3",
        "github-from-package": "0.0.0",
        "minimist": "^1.2.3",
        "mkdirp-classic": "^0.5.3",
        "napi-build-utils": "^2.0.0",
        "node-abi": "^3.3.0",
        "pump": "^3.0.0",
        "rc": "^1.2.7",
        "simple-get": "^4.0.0",
        "tar-fs": "^2.0.0",
        "tunnel-agent": "^0.6.0"
      },
      "bin": {
        "prebuild-install": "bin.js"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/prebuild-install/node_modules/chownr": {
      "version": "1.1.4",
      "resolved": "https://registry.npmjs.org/chownr/-/chownr-1.1.4.tgz",
      "integrity": "sha512-jJ0bqzaylmJtVnNgzTeSOs8DPavpbYgEr/b0YL8/2GO3xJEhInFmhKMUnEJQjZumK7KXGFhUy89PrsJWlakBVg=="
    },
    "node_modules/prebuild-install/node_modules/tar-fs": {
      "version": "2.1.2",
      "resolved": "https://registry.npmjs.org/tar-fs/-/tar-fs-2.1.2.tgz",
      "integrity": "sha512-EsaAXwxmx8UB7FRKqeozqEPop69DXcmYwTQwXvyAPF352HJsPdkVhvTaDPYqfNgruveJIJy3TA2l+2zj8LJIJA==",
      "dependencies": {
        "chownr": "^1.1.1",
        "mkdirp-classic": "^0.5.2",
        "pump": "^3.0.0",
        "tar-stream": "^2.1.4"
      }
    },
    "node_modules/prebuild-install/node_modules/tar-stream": {
      "version": "2.2.0",
      "resolved": "https://registry.npmjs.org/tar-stream/-/tar-stream-2.2.0.tgz",
      "integrity": "sha512-ujeqbceABgwMZxEJnk2HDY2DlnUZ+9oEcb1KzTVfYHio0UE6dG71n60d8D2I4qNvleWrrXpmjpt7vZeF1LnMZQ==",
      "dependencies": {
        "bl": "^4.0.3",
        "end-of-stream": "^1.4.1",
        "fs-constants": "^1.0.0",
        "inherits": "^2.0.3",
        "readable-stream": "^3.1.1"
      },
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/prettier": {
      "version": "2.8.8",
      "resolved": "https://registry.npmjs.org/prettier/-/prettier-2.8.8.tgz",
      "integrity": "sha512-tdN8qQGvNjw4CHbY+XXk0JgCXn9QiF21a55rBe5LJAU+kDyC4WQn4+awm2Xfk2lQMk5fKup9XgzTZtGkjBdP9Q==",
      "bin": {
        "prettier": "bin-prettier.js"
      },
      "engines": {
        "node": ">=10.13.0"
      },
      "funding": {
        "url": "https://github.com/prettier/prettier?sponsor=1"
      }
    },
    "node_modules/process-nextick-args": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/process-nextick-args/-/process-nextick-args-2.0.1.tgz",
      "integrity": "sha512-3ouUOpQhtgrbOa17J7+uxOTpITYWaGP7/AhoR3+A+/1e9skrzelGi/dXzEYyvbxubEF6Wn2ypscTKiKJFFn1ag=="
    },
    "node_modules/prompts": {
      "version": "2.4.2",
      "resolved": "https://registry.npmjs.org/prompts/-/prompts-2.4.2.tgz",
      "integrity": "sha512-NxNv/kLguCA7p3jE8oL2aEBsrJWgAakBpgmgK6lpPWV+WuOmY6r2/zbAVnP+T8bQlA0nzHXSJSJW0Hq7ylaD2Q==",
      "dependencies": {
        "kleur": "^3.0.3",
        "sisteransi": "^1.0.5"
      },
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/prompts/node_modules/kleur": {
      "version": "3.0.3",
      "resolved": "https://registry.npmjs.org/kleur/-/kleur-3.0.3.tgz",
      "integrity": "sha512-eTIzlVOSUR+JxdDFepEYcBMtZ9Qqdef+rnzWdRZuMbOywu5tO2w2N7rqjoANZ5k9vywhL6Br1VRjUIgTQx4E8w==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/pump": {
      "version": "3.0.2",
      "resolved": "https://registry.npmjs.org/pump/-/pump-3.0.2.tgz",
      "integrity": "sha512-tUPXtzlGM8FE3P0ZL6DVs/3P58k9nk8/jZeQCurTJylQA8qFYzHFfhBJkuqyE0FifOsQ0uKWekiZ5g8wtr28cw==",
      "dependencies": {
        "end-of-stream": "^1.1.0",
        "once": "^1.3.1"
      }
    },
    "node_modules/punycode": {
      "version": "2.3.1",
      "resolved": "https://registry.npmjs.org/punycode/-/punycode-2.3.1.tgz",
      "integrity": "sha512-vYt7UD1U9Wg6138shLtLOvdAu+8DsC/ilFtEVHcH+wydcSpNE20AfSOduf6MkRFahL5FY7X1oU7nKVZFtfq8Fg==",
      "dev": true,
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/q": {
      "version": "1.5.1",
      "resolved": "https://registry.npmjs.org/q/-/q-1.5.1.tgz",
      "integrity": "sha512-kV/CThkXo6xyFEZUugw/+pIOywXcDbFYgSct5cT3gqlbkBE1SJdwy6UQoZvodiWF/ckQLZyDE/Bu1M6gVu5lVw==",
      "deprecated": "You or someone you depend on is using Q, the JavaScript Promise library that gave JavaScript developers strong feelings about promises. They can almost certainly migrate to the native JavaScript promise now. Thank you literally everyone for joining me in this bet against the odds. Be excellent to each other.\n\n(For a CapTP with native promises, see @endo/eventual-send and @endo/captp)",
      "engines": {
        "node": ">=0.6.0",
        "teleport": ">=0.2.0"
      }
    },
    "node_modules/queue-microtask": {
      "version": "1.2.3",
      "resolved": "https://registry.npmjs.org/queue-microtask/-/queue-microtask-1.2.3.tgz",
      "integrity": "sha512-NuaNSa6flKT5JaSYQzJok04JzTL1CA6aGhv5rfLW3PgqA+M2ChpZQnAC8h8i4ZFkBS8X5RqkDBHA7r4hej3K9A==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/feross"
        },
        {
          "type": "patreon",
          "url": "https://www.patreon.com/feross"
        },
        {
          "type": "consulting",
          "url": "https://feross.org/support"
        }
      ]
    },
    "node_modules/quick-lru": {
      "version": "4.0.1",
      "resolved": "https://registry.npmjs.org/quick-lru/-/quick-lru-4.0.1.tgz",
      "integrity": "sha512-ARhCpm70fzdcvNQfPoy49IaanKkTlRWF2JMzqhcJbhSFRZv7nPTvZJdcY7301IPmvW+/p0RgIWnQDLJxifsQ7g==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/range-parser": {
      "version": "1.2.0",
      "resolved": "https://registry.npmjs.org/range-parser/-/range-parser-1.2.0.tgz",
      "integrity": "sha512-kA5WQoNVo4t9lNx2kQNFCxKeBl5IbbSNBl1M/tLkw9WCn+hxNBAW5Qh8gdhs63CJnhjJ2zQWFoqPJP2sK1AV5A==",
      "dev": true,
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/rc": {
      "version": "1.2.8",
      "resolved": "https://registry.npmjs.org/rc/-/rc-1.2.8.tgz",
      "integrity": "sha512-y3bGgqKj3QBdxLbLkomlohkvsA8gdAiUQlSBJnBhfn+BPxg4bc62d8TcBW15wavDfgexCgccckhcZvywyQYPOw==",
      "dependencies": {
        "deep-extend": "^0.6.0",
        "ini": "~1.3.0",
        "minimist": "^1.2.0",
        "strip-json-comments": "~2.0.1"
      },
      "bin": {
        "rc": "cli.js"
      }
    },
    "node_modules/rc/node_modules/ini": {
      "version": "1.3.8",
      "resolved": "https://registry.npmjs.org/ini/-/ini-1.3.8.tgz",
      "integrity": "sha512-JV/yugV2uzW5iMRSiZAyDtQd+nxtUnjeLt0acNdw98kKLrvuRVyB80tsREOE7yvGVgalhZ6RNXCmEHkUKBKxew=="
    },
    "node_modules/read-cache": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/read-cache/-/read-cache-1.0.0.tgz",
      "integrity": "sha512-Owdv/Ft7IjOgm/i0xvNDZ1LrRANRfew4b2prF3OWMQLxLfu3bS8FVhCsrSCMK4lR56Y9ya+AThoTpDCTxCmpRA==",
      "dependencies": {
        "pify": "^2.3.0"
      }
    },
    "node_modules/read-pkg": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/read-pkg/-/read-pkg-3.0.0.tgz",
      "integrity": "sha512-BLq/cCO9two+lBgiTYNqD6GdtK8s4NpaWrl6/rCO9w0TUS8oJl7cmToOZfRYllKTISY6nt1U7jQ53brmKqY6BA==",
      "dependencies": {
        "load-json-file": "^4.0.0",
        "normalize-package-data": "^2.3.2",
        "path-type": "^3.0.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/read-pkg-up": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/read-pkg-up/-/read-pkg-up-3.0.0.tgz",
      "integrity": "sha512-YFzFrVvpC6frF1sz8psoHDBGF7fLPc+llq/8NB43oagqWkx8ar5zYtsTORtOjw9W2RHLpWP+zTWwBvf1bCmcSw==",
      "dependencies": {
        "find-up": "^2.0.0",
        "read-pkg": "^3.0.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/read-pkg/node_modules/hosted-git-info": {
      "version": "2.8.9",
      "resolved": "https://registry.npmjs.org/hosted-git-info/-/hosted-git-info-2.8.9.tgz",
      "integrity": "sha512-mxIDAb9Lsm6DoOJ7xH+5+X4y1LU/4Hi50L9C5sIswK3JzULS4bwk1FvjdBgvYR4bzT4tuUQiC15FE2f5HbLvYw=="
    },
    "node_modules/read-pkg/node_modules/normalize-package-data": {
      "version": "2.5.0",
      "resolved": "https://registry.npmjs.org/normalize-package-data/-/normalize-package-data-2.5.0.tgz",
      "integrity": "sha512-/5CMN3T0R4XTj4DcGaexo+roZSdSFW/0AOOTROrjxzCG1wrWXEsGbRKevjlIL+ZDE4sZlJr5ED4YW0yqmkK+eA==",
      "dependencies": {
        "hosted-git-info": "^2.1.4",
        "resolve": "^1.10.0",
        "semver": "2 || 3 || 4 || 5",
        "validate-npm-package-license": "^3.0.1"
      }
    },
    "node_modules/read-pkg/node_modules/semver": {
      "version": "5.7.2",
      "resolved": "https://registry.npmjs.org/semver/-/semver-5.7.2.tgz",
      "integrity": "sha512-cBznnQ9KjJqU67B52RMC65CMarK2600WFnbkcaiwWq3xy/5haFJlshgnpjovMVJ+Hff49d8GEn0b87C5pDQ10g==",
      "bin": {
        "semver": "bin/semver"
      }
    },
    "node_modules/readable-stream": {
      "version": "3.6.2",
      "resolved": "https://registry.npmjs.org/readable-stream/-/readable-stream-3.6.2.tgz",
      "integrity": "sha512-9u/sniCrY3D5WdsERHzHE4G2YCXqoG5FTHUiCC4SIbr6XcLZBY05ya9EKjYek9O5xOAwjGq+1JdGBAS7Q9ScoA==",
      "dependencies": {
        "inherits": "^2.0.3",
        "string_decoder": "^1.1.1",
        "util-deprecate": "^1.0.1"
      },
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/readdirp": {
      "version": "3.6.0",
      "resolved": "https://registry.npmjs.org/readdirp/-/readdirp-3.6.0.tgz",
      "integrity": "sha512-hOS089on8RduqdbhvQ5Z37A0ESjsqz6qnRcffsMU3495FuTdqSm+7bhJ29JvIOsBDEEnan5DPu9t3To9VRlMzA==",
      "dependencies": {
        "picomatch": "^2.2.1"
      },
      "engines": {
        "node": ">=8.10.0"
      }
    },
    "node_modules/redent": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/redent/-/redent-3.0.0.tgz",
      "integrity": "sha512-6tDA8g98We0zd0GvVeMT9arEOnTw9qM03L9cJXaCjrip1OO764RDBLBfrB4cwzNGDj5OA5ioymC9GkizgWJDUg==",
      "dependencies": {
        "indent-string": "^4.0.0",
        "strip-indent": "^3.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/regexp-to-ast": {
      "version": "0.5.0",
      "resolved": "https://registry.npmjs.org/regexp-to-ast/-/regexp-to-ast-0.5.0.tgz",
      "integrity": "sha512-tlbJqcMHnPKI9zSrystikWKwHkBqu2a/Sgw01h3zFjvYrMxEDYHzzoMZnUrbIfpTFEsoRnnviOXNCzFiSc54Qw=="
    },
    "node_modules/registry-auth-token": {
      "version": "3.3.2",
      "resolved": "https://registry.npmjs.org/registry-auth-token/-/registry-auth-token-3.3.2.tgz",
      "integrity": "sha512-JL39c60XlzCVgNrO+qq68FoNb56w/m7JYvGR2jT5iR1xBrUA3Mfx5Twk5rqTThPmQKMWydGmq8oFtDlxfrmxnQ==",
      "dev": true,
      "dependencies": {
        "rc": "^1.1.6",
        "safe-buffer": "^5.0.1"
      }
    },
    "node_modules/registry-url": {
      "version": "3.1.0",
      "resolved": "https://registry.npmjs.org/registry-url/-/registry-url-3.1.0.tgz",
      "integrity": "sha512-ZbgR5aZEdf4UKZVBPYIgaglBmSF2Hi94s2PcIHhRGFjKYu+chjJdYfHn4rt3hB6eCKLJ8giVIIfgMa1ehDfZKA==",
      "dev": true,
      "dependencies": {
        "rc": "^1.0.1"
      },
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/replace": {
      "version": "1.2.2",
      "resolved": "https://registry.npmjs.org/replace/-/replace-1.2.2.tgz",
      "integrity": "sha512-C4EDifm22XZM2b2JOYe6Mhn+lBsLBAvLbK8drfUQLTfD1KYl/n3VaW/CDju0Ny4w3xTtegBpg8YNSpFJPUDSjA==",
      "dependencies": {
        "chalk": "2.4.2",
        "minimatch": "3.0.5",
        "yargs": "^15.3.1"
      },
      "bin": {
        "replace": "bin/replace.js",
        "search": "bin/search.js"
      },
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/replace/node_modules/ansi-styles": {
      "version": "3.2.1",
      "resolved": "https://registry.npmjs.org/ansi-styles/-/ansi-styles-3.2.1.tgz",
      "integrity": "sha512-VT0ZI6kZRdTh8YyJw3SMbYm/u+NqfsAxEpWO0Pf9sq8/e94WxxOpPKx9FR1FlyCtOVDNOQ+8ntlqFxiRc+r5qA==",
      "dependencies": {
        "color-convert": "^1.9.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/replace/node_modules/camelcase": {
      "version": "5.3.1",
      "resolved": "https://registry.npmjs.org/camelcase/-/camelcase-5.3.1.tgz",
      "integrity": "sha512-L28STB170nwWS63UjtlEOE3dldQApaJXZkOI1uMFfzf3rRuPegHaHesyee+YxQ+W6SvRDQV6UrdOdRiR153wJg==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/replace/node_modules/chalk": {
      "version": "2.4.2",
      "resolved": "https://registry.npmjs.org/chalk/-/chalk-2.4.2.tgz",
      "integrity": "sha512-Mti+f9lpJNcwF4tWV8/OrTTtF1gZi+f8FqlyAdouralcFWFQWF2+NgCHShjkCb+IFBLq9buZwE1xckQU4peSuQ==",
      "dependencies": {
        "ansi-styles": "^3.2.1",
        "escape-string-regexp": "^1.0.5",
        "supports-color": "^5.3.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/replace/node_modules/cliui": {
      "version": "6.0.0",
      "resolved": "https://registry.npmjs.org/cliui/-/cliui-6.0.0.tgz",
      "integrity": "sha512-t6wbgtoCXvAzst7QgXxJYqPt0usEfbgQdftEPbLL/cvv6HPE5VgvqCuAIDR0NgU52ds6rFwqrgakNLrHEjCbrQ==",
      "dependencies": {
        "string-width": "^4.2.0",
        "strip-ansi": "^6.0.0",
        "wrap-ansi": "^6.2.0"
      }
    },
    "node_modules/replace/node_modules/color-convert": {
      "version": "1.9.3",
      "resolved": "https://registry.npmjs.org/color-convert/-/color-convert-1.9.3.tgz",
      "integrity": "sha512-QfAUtd+vFdAtFQcC8CCyYt1fYWxSqAiK2cSD6zDB8N3cpsEBAvRxp9zOGg6G/SHHJYAT88/az/IuDGALsNVbGg==",
      "dependencies": {
        "color-name": "1.1.3"
      }
    },
    "node_modules/replace/node_modules/color-name": {
      "version": "1.1.3",
      "resolved": "https://registry.npmjs.org/color-name/-/color-name-1.1.3.tgz",
      "integrity": "sha512-72fSenhMw2HZMTVHeCA9KCmpEIbzWiQsjN+BHcBbS9vr1mtt+vJjPdksIBNUmKAW8TFUDPJK5SUU3QhE9NEXDw=="
    },
    "node_modules/replace/node_modules/find-up": {
      "version": "4.1.0",
      "resolved": "https://registry.npmjs.org/find-up/-/find-up-4.1.0.tgz",
      "integrity": "sha512-PpOwAdQ/YlXQ2vj8a3h8IipDuYRi3wceVQQGYWxNINccq40Anw7BlsEXCMbt1Zt+OLA6Fq9suIpIWD0OsnISlw==",
      "dependencies": {
        "locate-path": "^5.0.0",
        "path-exists": "^4.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/replace/node_modules/has-flag": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/has-flag/-/has-flag-3.0.0.tgz",
      "integrity": "sha512-sKJf1+ceQBr4SMkvQnBDNDtf4TXpVhVGateu0t918bl30FnbE2m4vNLX+VWe/dpjlb+HugGYzW7uQXH98HPEYw==",
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/replace/node_modules/locate-path": {
      "version": "5.0.0",
      "resolved": "https://registry.npmjs.org/locate-path/-/locate-path-5.0.0.tgz",
      "integrity": "sha512-t7hw9pI+WvuwNJXwk5zVHpyhIqzg2qTlklJOf0mVxGSbe3Fp2VieZcduNYjaLDoy6p9uGpQEGWG87WpMKlNq8g==",
      "dependencies": {
        "p-locate": "^4.1.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/replace/node_modules/p-limit": {
      "version": "2.3.0",
      "resolved": "https://registry.npmjs.org/p-limit/-/p-limit-2.3.0.tgz",
      "integrity": "sha512-//88mFWSJx8lxCzwdAABTJL2MyWB12+eIY7MDL2SqLmAkeKU9qxRvWuSyTjm3FUmpBEMuFfckAIqEaVGUDxb6w==",
      "dependencies": {
        "p-try": "^2.0.0"
      },
      "engines": {
        "node": ">=6"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/replace/node_modules/p-locate": {
      "version": "4.1.0",
      "resolved": "https://registry.npmjs.org/p-locate/-/p-locate-4.1.0.tgz",
      "integrity": "sha512-R79ZZ/0wAxKGu3oYMlz8jy/kbhsNrS7SKZ7PxEHBgJ5+F2mtFW2fK2cOtBh1cHYkQsbzFV7I+EoRKe6Yt0oK7A==",
      "dependencies": {
        "p-limit": "^2.2.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/replace/node_modules/p-try": {
      "version": "2.2.0",
      "resolved": "https://registry.npmjs.org/p-try/-/p-try-2.2.0.tgz",
      "integrity": "sha512-R4nPAVTAU0B9D35/Gk3uJf/7XYbQcyohSKdvAxIRSNghFl4e71hVoGnBNQz9cWaXxO2I10KTC+3jMdvvoKw6dQ==",
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/replace/node_modules/path-exists": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/path-exists/-/path-exists-4.0.0.tgz",
      "integrity": "sha512-ak9Qy5Q7jYb2Wwcey5Fpvg2KoAc/ZIhLSLOSBmRmygPsGwkVVt0fZa0qrtMz+m6tJTAHfZQ8FnmB4MG4LWy7/w==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/replace/node_modules/supports-color": {
      "version": "5.5.0",
      "resolved": "https://registry.npmjs.org/supports-color/-/supports-color-5.5.0.tgz",
      "integrity": "sha512-QjVjwdXIt408MIiAqCX4oUKsgU2EqAGzs2Ppkm4aQYbjm+ZEWEcW4SfFNTr4uMNZma0ey4f5lgLrkB0aX0QMow==",
      "dependencies": {
        "has-flag": "^3.0.0"
      },
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/replace/node_modules/wrap-ansi": {
      "version": "6.2.0",
      "resolved": "https://registry.npmjs.org/wrap-ansi/-/wrap-ansi-6.2.0.tgz",
      "integrity": "sha512-r6lPcBGxZXlIcymEu7InxDMhdW0KDxpLgoFLcguasxCaJ/SOIZwINatK9KY/tf+ZrlywOKU0UDj3ATXUBfxJXA==",
      "dependencies": {
        "ansi-styles": "^4.0.0",
        "string-width": "^4.1.0",
        "strip-ansi": "^6.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/replace/node_modules/wrap-ansi/node_modules/ansi-styles": {
      "version": "4.3.0",
      "resolved": "https://registry.npmjs.org/ansi-styles/-/ansi-styles-4.3.0.tgz",
      "integrity": "sha512-zbB9rCJAT1rbjiVDb2hqKFHNYLxgtk8NURxZ3IZwD3F6NtxbXZQCnnSi1Lkx+IDohdPlFp222wVALIheZJQSEg==",
      "dependencies": {
        "color-convert": "^2.0.1"
      },
      "engines": {
        "node": ">=8"
      },
      "funding": {
        "url": "https://github.com/chalk/ansi-styles?sponsor=1"
      }
    },
    "node_modules/replace/node_modules/wrap-ansi/node_modules/color-convert": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/color-convert/-/color-convert-2.0.1.tgz",
      "integrity": "sha512-RRECPsj7iu/xb5oKYcsFHSppFNnsj/52OVTRKb4zP5onXwVF3zVmmToNcOfGC+CRDpfK/U584fMg38ZHCaElKQ==",
      "dependencies": {
        "color-name": "~1.1.4"
      },
      "engines": {
        "node": ">=7.0.0"
      }
    },
    "node_modules/replace/node_modules/wrap-ansi/node_modules/color-name": {
      "version": "1.1.4",
      "resolved": "https://registry.npmjs.org/color-name/-/color-name-1.1.4.tgz",
      "integrity": "sha512-dOy+3AuW3a2wNbZHIuMZpTcgjGuLU/uBL/ubcZF9OXbDo8ff4O8yVp5Bf0efS8uEoYo5q4Fx7dY9OgQGXgAsQA=="
    },
    "node_modules/replace/node_modules/y18n": {
      "version": "4.0.3",
      "resolved": "https://registry.npmjs.org/y18n/-/y18n-4.0.3.tgz",
      "integrity": "sha512-JKhqTOwSrqNA1NY5lSztJ1GrBiUodLMmIZuLiDaMRJ+itFd+ABVE8XBjOvIWL+rSqNDC74LCSFmlb/U4UZ4hJQ=="
    },
    "node_modules/replace/node_modules/yargs": {
      "version": "15.4.1",
      "resolved": "https://registry.npmjs.org/yargs/-/yargs-15.4.1.tgz",
      "integrity": "sha512-aePbxDmcYW++PaqBsJ+HYUFwCdv4LVvdnhBy78E57PIor8/OVvhMrADFFEDh8DHDFRv/O9i3lPhsENjO7QX0+A==",
      "dependencies": {
        "cliui": "^6.0.0",
        "decamelize": "^1.2.0",
        "find-up": "^4.1.0",
        "get-caller-file": "^2.0.1",
        "require-directory": "^2.1.1",
        "require-main-filename": "^2.0.0",
        "set-blocking": "^2.0.0",
        "string-width": "^4.2.0",
        "which-module": "^2.0.0",
        "y18n": "^4.0.0",
        "yargs-parser": "^18.1.2"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/replace/node_modules/yargs-parser": {
      "version": "18.1.3",
      "resolved": "https://registry.npmjs.org/yargs-parser/-/yargs-parser-18.1.3.tgz",
      "integrity": "sha512-o50j0JeToy/4K6OZcaQmW6lyXXKhq7csREXcDwk2omFPJEwUNOVtJKvmDr9EI1fAJZUyZcRF7kxGBWmRXudrCQ==",
      "dependencies": {
        "camelcase": "^5.0.0",
        "decamelize": "^1.2.0"
      },
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/require-directory": {
      "version": "2.1.1",
      "resolved": "https://registry.npmjs.org/require-directory/-/require-directory-2.1.1.tgz",
      "integrity": "sha512-fGxEI7+wsG9xrvdjsrlmL22OMTTiHRwAMroiEeMgq8gzoLC/PQr7RsRDSTLUg/bZAZtF+TVIkHc6/4RIKrui+Q==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/require-from-string": {
      "version": "2.0.2",
      "resolved": "https://registry.npmjs.org/require-from-string/-/require-from-string-2.0.2.tgz",
      "integrity": "sha512-Xf0nWe6RseziFMu+Ap9biiUbmplq6S9/p+7w7YXP/JBHhrUDDUhwa+vANyubuqfZWTveU//DYVGsDG7RKL/vEw==",
      "dev": true,
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/require-main-filename": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/require-main-filename/-/require-main-filename-2.0.0.tgz",
      "integrity": "sha512-NKN5kMDylKuldxYLSUfrbo5Tuzh4hd+2E8NPPX02mZtn1VuREQToYe/ZdlJy+J3uCpfaiGF05e7B8W0iXbQHmg=="
    },
    "node_modules/resolve": {
      "version": "1.22.10",
      "resolved": "https://registry.npmjs.org/resolve/-/resolve-1.22.10.tgz",
      "integrity": "sha512-NPRy+/ncIMeDlTAsuqwKIiferiawhefFJtkNSW0qZJEqMEb+qBt/77B/jGeeek+F0uOeN05CDa6HXbbIgtVX4w==",
      "dependencies": {
        "is-core-module": "^2.16.0",
        "path-parse": "^1.0.7",
        "supports-preserve-symlinks-flag": "^1.0.0"
      },
      "bin": {
        "resolve": "bin/resolve"
      },
      "engines": {
        "node": ">= 0.4"
      },
      "funding": {
        "url": "https://github.com/sponsors/ljharb"
      }
    },
    "node_modules/reusify": {
      "version": "1.0.4",
      "resolved": "https://registry.npmjs.org/reusify/-/reusify-1.0.4.tgz",
      "integrity": "sha512-U9nH88a3fc/ekCF1l0/UP1IosiuIjyTh7hBvXVMHYgVcfGvt897Xguj2UOLDeI5BG2m7/uwyaLVT6fbtCwTyzw==",
      "engines": {
        "iojs": ">=1.0.0",
        "node": ">=0.10.0"
      }
    },
    "node_modules/rimraf": {
      "version": "4.4.1",
      "resolved": "https://registry.npmjs.org/rimraf/-/rimraf-4.4.1.tgz",
      "integrity": "sha512-Gk8NlF062+T9CqNGn6h4tls3k6T1+/nXdOcSZVikNVtlRdYpA7wRJJMoXmuvOnLW844rPjdQ7JgXCYM6PPC/og==",
      "dependencies": {
        "glob": "^9.2.0"
      },
      "bin": {
        "rimraf": "dist/cjs/src/bin.js"
      },
      "engines": {
        "node": ">=14"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/rollup": {
      "version": "4.29.1",
      "resolved": "https://registry.npmjs.org/rollup/-/rollup-4.29.1.tgz",
      "integrity": "sha512-RaJ45M/kmJUzSWDs1Nnd5DdV4eerC98idtUOVr6FfKcgxqvjwHmxc5upLF9qZU9EpsVzzhleFahrT3shLuJzIw==",
      "dev": true,
      "dependencies": {
        "@types/estree": "1.0.6"
      },
      "bin": {
        "rollup": "dist/bin/rollup"
      },
      "engines": {
        "node": ">=18.0.0",
        "npm": ">=8.0.0"
      },
      "optionalDependencies": {
        "@rollup/rollup-android-arm-eabi": "4.29.1",
        "@rollup/rollup-android-arm64": "4.29.1",
        "@rollup/rollup-darwin-arm64": "4.29.1",
        "@rollup/rollup-darwin-x64": "4.29.1",
        "@rollup/rollup-freebsd-arm64": "4.29.1",
        "@rollup/rollup-freebsd-x64": "4.29.1",
        "@rollup/rollup-linux-arm-gnueabihf": "4.29.1",
        "@rollup/rollup-linux-arm-musleabihf": "4.29.1",
        "@rollup/rollup-linux-arm64-gnu": "4.29.1",
        "@rollup/rollup-linux-arm64-musl": "4.29.1",
        "@rollup/rollup-linux-loongarch64-gnu": "4.29.1",
        "@rollup/rollup-linux-powerpc64le-gnu": "4.29.1",
        "@rollup/rollup-linux-riscv64-gnu": "4.29.1",
        "@rollup/rollup-linux-s390x-gnu": "4.29.1",
        "@rollup/rollup-linux-x64-gnu": "4.29.1",
        "@rollup/rollup-linux-x64-musl": "4.29.1",
        "@rollup/rollup-win32-arm64-msvc": "4.29.1",
        "@rollup/rollup-win32-ia32-msvc": "4.29.1",
        "@rollup/rollup-win32-x64-msvc": "4.29.1",
        "fsevents": "~2.3.2"
      }
    },
    "node_modules/run-parallel": {
      "version": "1.2.0",
      "resolved": "https://registry.npmjs.org/run-parallel/-/run-parallel-1.2.0.tgz",
      "integrity": "sha512-5l4VyZR86LZ/lDxZTR6jqL8AFE2S0IFLMP26AbjsLVADxHdhB/c0GUsH+y39UfCi3dzz8OlQuPmnaJOMoDHQBA==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/feross"
        },
        {
          "type": "patreon",
          "url": "https://www.patreon.com/feross"
        },
        {
          "type": "consulting",
          "url": "https://feross.org/support"
        }
      ],
      "dependencies": {
        "queue-microtask": "^1.2.2"
      }
    },
    "node_modules/safe-buffer": {
      "version": "5.1.2",
      "resolved": "https://registry.npmjs.org/safe-buffer/-/safe-buffer-5.1.2.tgz",
      "integrity": "sha512-Gd2UZBJDkXlY7GbJxfsE8/nvKkUEU1G38c1siN6QP6a9PT9MmHB8GnpscSmMJSoF8LOIrt8ud/wPtojys4G6+g=="
    },
    "node_modules/sax": {
      "version": "1.1.4",
      "resolved": "https://registry.npmjs.org/sax/-/sax-1.1.4.tgz",
      "integrity": "sha512-5f3k2PbGGp+YtKJjOItpg3P99IMD84E4HOvcfleTb5joCHNXYLsR9yWFPOYGgaeMPDubQILTCMdsFb2OMeOjtg=="
    },
    "node_modules/semver": {
      "version": "7.7.1",
      "resolved": "https://registry.npmjs.org/semver/-/semver-7.7.1.tgz",
      "integrity": "sha512-hlq8tAfn0m/61p4BVRcPzIGr6LKiMwo4VM6dGi6pt4qcRkmNzTcWq6eCEjEh+qXjkMDvPlOFFSGwQjoEa6gyMA==",
      "bin": {
        "semver": "bin/semver.js"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/serve": {
      "version": "14.2.4",
      "resolved": "https://registry.npmjs.org/serve/-/serve-14.2.4.tgz",
      "integrity": "sha512-qy1S34PJ/fcY8gjVGszDB3EXiPSk5FKhUa7tQe0UPRddxRidc2V6cNHPNewbE1D7MAkgLuWEt3Vw56vYy73tzQ==",
      "dev": true,
      "dependencies": {
        "@zeit/schemas": "2.36.0",
        "ajv": "8.12.0",
        "arg": "5.0.2",
        "boxen": "7.0.0",
        "chalk": "5.0.1",
        "chalk-template": "0.4.0",
        "clipboardy": "3.0.0",
        "compression": "1.7.4",
        "is-port-reachable": "4.0.0",
        "serve-handler": "6.1.6",
        "update-check": "1.5.4"
      },
      "bin": {
        "serve": "build/main.js"
      },
      "engines": {
        "node": ">= 14"
      }
    },
    "node_modules/serve-handler": {
      "version": "6.1.6",
      "resolved": "https://registry.npmjs.org/serve-handler/-/serve-handler-6.1.6.tgz",
      "integrity": "sha512-x5RL9Y2p5+Sh3D38Fh9i/iQ5ZK+e4xuXRd/pGbM4D13tgo/MGwbttUk8emytcr1YYzBYs+apnUngBDFYfpjPuQ==",
      "dev": true,
      "dependencies": {
        "bytes": "3.0.0",
        "content-disposition": "0.5.2",
        "mime-types": "2.1.18",
        "minimatch": "3.1.2",
        "path-is-inside": "1.0.2",
        "path-to-regexp": "3.3.0",
        "range-parser": "1.2.0"
      }
    },
    "node_modules/serve-handler/node_modules/brace-expansion": {
      "version": "1.1.11",
      "resolved": "https://registry.npmjs.org/brace-expansion/-/brace-expansion-1.1.11.tgz",
      "integrity": "sha512-iCuPHDFgrHX7H2vEI/5xpz07zSHB00TpugqhmYtVmMO6518mCuRMoOYFldEBl0g187ufozdaHgWKcYFb61qGiA==",
      "dev": true,
      "dependencies": {
        "balanced-match": "^1.0.0",
        "concat-map": "0.0.1"
      }
    },
    "node_modules/serve-handler/node_modules/mime-db": {
      "version": "1.33.0",
      "resolved": "https://registry.npmjs.org/mime-db/-/mime-db-1.33.0.tgz",
      "integrity": "sha512-BHJ/EKruNIqJf/QahvxwQZXKygOQ256myeN/Ew+THcAa5q+PjyTTMMeNQC4DZw5AwfvelsUrA6B67NKMqXDbzQ==",
      "dev": true,
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/serve-handler/node_modules/mime-types": {
      "version": "2.1.18",
      "resolved": "https://registry.npmjs.org/mime-types/-/mime-types-2.1.18.tgz",
      "integrity": "sha512-lc/aahn+t4/SWV/qcmumYjymLsWfN3ELhpmVuUFjgsORruuZPVSwAQryq+HHGvO/SI2KVX26bx+En+zhM8g8hQ==",
      "dev": true,
      "dependencies": {
        "mime-db": "~1.33.0"
      },
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/serve-handler/node_modules/minimatch": {
      "version": "3.1.2",
      "resolved": "https://registry.npmjs.org/minimatch/-/minimatch-3.1.2.tgz",
      "integrity": "sha512-J7p63hRiAjw1NDEww1W7i37+ByIrOWO5XQQAzZ3VOcL0PNybwpfmV/N05zFAzwQ9USyEcX6t3UO+K5aqBQOIHw==",
      "dev": true,
      "dependencies": {
        "brace-expansion": "^1.1.7"
      },
      "engines": {
        "node": "*"
      }
    },
    "node_modules/set-blocking": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/set-blocking/-/set-blocking-2.0.0.tgz",
      "integrity": "sha512-KiKBS8AnWGEyLzofFfmvKwpdPzqiy16LvQfK3yv/fVH7Bj13/wl3JSR1J+rfgRE9q7xUJK4qvgS8raSOeLUehw=="
    },
    "node_modules/sharp": {
      "version": "0.32.6",
      "resolved": "https://registry.npmjs.org/sharp/-/sharp-0.32.6.tgz",
      "integrity": "sha512-KyLTWwgcR9Oe4d9HwCwNM2l7+J0dUQwn/yf7S0EnTtb0eVS4RxO0eUSvxPtzT4F3SY+C4K6fqdv/DO27sJ/v/w==",
      "hasInstallScript": true,
      "dependencies": {
        "color": "^4.2.3",
        "detect-libc": "^2.0.2",
        "node-addon-api": "^6.1.0",
        "prebuild-install": "^7.1.1",
        "semver": "^7.5.4",
        "simple-get": "^4.0.1",
        "tar-fs": "^3.0.4",
        "tunnel-agent": "^0.6.0"
      },
      "engines": {
        "node": ">=14.15.0"
      },
      "funding": {
        "url": "https://opencollective.com/libvips"
      }
    },
    "node_modules/shebang-command": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/shebang-command/-/shebang-command-2.0.0.tgz",
      "integrity": "sha512-kHxr2zZpYtdmrN1qDjrrX/Z1rR1kG8Dx+gkpK1G4eXmvXswmcE1hTWBWYUzlraYw1/yZp6YuDY77YtvbN0dmDA==",
      "dependencies": {
        "shebang-regex": "^3.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/shebang-regex": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/shebang-regex/-/shebang-regex-3.0.0.tgz",
      "integrity": "sha512-7++dFhtcx3353uBaq8DDR4NuxBetBzC7ZQOhmTQInHEd6bSrXdiEyzCvG07Z44UYdLShWUyXt5M/yhz8ekcb1A==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/signal-exit": {
      "version": "3.0.7",
      "resolved": "https://registry.npmjs.org/signal-exit/-/signal-exit-3.0.7.tgz",
      "integrity": "sha512-wnD2ZE+l+SPC/uoS0vXeE9L1+0wuaMqKlfz9AMUo38JsyLSBWSFcHR1Rri62LZc12vLr1gb3jl7iwQhgwpAbGQ=="
    },
    "node_modules/simple-concat": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/simple-concat/-/simple-concat-1.0.1.tgz",
      "integrity": "sha512-cSFtAPtRhljv69IK0hTVZQ+OfE9nePi/rtJmw5UjHeVyVroEqJXP1sFztKUy1qU+xvz3u/sfYJLa947b7nAN2Q==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/feross"
        },
        {
          "type": "patreon",
          "url": "https://www.patreon.com/feross"
        },
        {
          "type": "consulting",
          "url": "https://feross.org/support"
        }
      ]
    },
    "node_modules/simple-get": {
      "version": "4.0.1",
      "resolved": "https://registry.npmjs.org/simple-get/-/simple-get-4.0.1.tgz",
      "integrity": "sha512-brv7p5WgH0jmQJr1ZDDfKDOSeWWg+OVypG99A/5vYGPqJ6pxiaHLy8nxtFjBA7oMa01ebA9gfh1uMCFqOuXxvA==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/feross"
        },
        {
          "type": "patreon",
          "url": "https://www.patreon.com/feross"
        },
        {
          "type": "consulting",
          "url": "https://feross.org/support"
        }
      ],
      "dependencies": {
        "decompress-response": "^6.0.0",
        "once": "^1.3.1",
        "simple-concat": "^1.0.0"
      }
    },
    "node_modules/simple-plist": {
      "version": "1.3.1",
      "resolved": "https://registry.npmjs.org/simple-plist/-/simple-plist-1.3.1.tgz",
      "integrity": "sha512-iMSw5i0XseMnrhtIzRb7XpQEXepa9xhWxGUojHBL43SIpQuDQkh3Wpy67ZbDzZVr6EKxvwVChnVpdl8hEVLDiw==",
      "dependencies": {
        "bplist-creator": "0.1.0",
        "bplist-parser": "0.3.1",
        "plist": "^3.0.5"
      }
    },
    "node_modules/simple-plist/node_modules/bplist-parser": {
      "version": "0.3.1",
      "resolved": "https://registry.npmjs.org/bplist-parser/-/bplist-parser-0.3.1.tgz",
      "integrity": "sha512-PyJxiNtA5T2PlLIeBot4lbp7rj4OadzjnMZD/G5zuBNt8ei/yCU7+wW0h2bag9vr8c+/WuRWmSxbqAl9hL1rBA==",
      "dependencies": {
        "big-integer": "1.6.x"
      },
      "engines": {
        "node": ">= 5.10.0"
      }
    },
    "node_modules/simple-swizzle": {
      "version": "0.2.2",
      "resolved": "https://registry.npmjs.org/simple-swizzle/-/simple-swizzle-0.2.2.tgz",
      "integrity": "sha512-JA//kQgZtbuY83m+xT+tXJkmJncGMTFT+C+g2h2R9uxkYIrE2yy9sgmcLhCnw57/WSD+Eh3J97FPEDFnbXnDUg==",
      "dependencies": {
        "is-arrayish": "^0.3.1"
      }
    },
    "node_modules/simple-swizzle/node_modules/is-arrayish": {
      "version": "0.3.2",
      "resolved": "https://registry.npmjs.org/is-arrayish/-/is-arrayish-0.3.2.tgz",
      "integrity": "sha512-eVRqCvVlZbuw3GrM63ovNSNAeA1K16kaR/LRY/92w0zxQ5/1YzwblUX652i4Xs9RwAGjW9d9y6X88t8OaAJfWQ=="
    },
    "node_modules/sisteransi": {
      "version": "1.0.5",
      "resolved": "https://registry.npmjs.org/sisteransi/-/sisteransi-1.0.5.tgz",
      "integrity": "sha512-bLGGlR1QxBcynn2d5YmDX4MGjlZvy2MRBDRNHLJ8VI6l6+9FUiyTFNJ0IveOSP0bcXgVDPRcfGqA0pjaqUpfVg=="
    },
    "node_modules/slash": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/slash/-/slash-3.0.0.tgz",
      "integrity": "sha512-g9Q1haeby36OSStwb4ntCGGGaKsaVSjQ68fBxoQcutl5fS1vuY18H3wSt3jFyFtrkx+Kz0V1G85A4MyAdDMi2Q==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/slice-ansi": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/slice-ansi/-/slice-ansi-4.0.0.tgz",
      "integrity": "sha512-qMCMfhY040cVHT43K9BFygqYbUPFZKHOg7K73mtTWJRb8pyP3fzf4Ixd5SzdEJQ6MRUg/WBnOLxghZtKKurENQ==",
      "dependencies": {
        "ansi-styles": "^4.0.0",
        "astral-regex": "^2.0.0",
        "is-fullwidth-code-point": "^3.0.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/chalk/slice-ansi?sponsor=1"
      }
    },
    "node_modules/source-map": {
      "version": "0.6.1",
      "resolved": "https://registry.npmjs.org/source-map/-/source-map-0.6.1.tgz",
      "integrity": "sha512-UjgapumWlbMhkBgzT7Ykc5YXUT46F0iKu8SGXq0bcwP5dz/h0Plj6enJqjz1Zbq2l5WaqYnrVbwWOWMyF3F47g==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/source-map-js": {
      "version": "1.2.1",
      "resolved": "https://registry.npmjs.org/source-map-js/-/source-map-js-1.2.1.tgz",
      "integrity": "sha512-UXWMKhLOwVKb728IUtQPXxfYU+usdybtUrK/8uGE8CQMvrhOpwvzDBwj0QhSL7MQc7vIsISBG8VQ8+IDQxpfQA==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/source-map-support": {
      "version": "0.5.21",
      "resolved": "https://registry.npmjs.org/source-map-support/-/source-map-support-0.5.21.tgz",
      "integrity": "sha512-uBHU3L3czsIyYXKX88fdrGovxdSCoTGDRZ6SYXtSRxLZUzHg5P/66Ht6uoUlHu9EZod+inXhKo3qQgwXUT/y1w==",
      "dev": true,
      "optional": true,
      "peer": true,
      "dependencies": {
        "buffer-from": "^1.0.0",
        "source-map": "^0.6.0"
      }
    },
    "node_modules/spdx-correct": {
      "version": "3.2.0",
      "resolved": "https://registry.npmjs.org/spdx-correct/-/spdx-correct-3.2.0.tgz",
      "integrity": "sha512-kN9dJbvnySHULIluDHy32WHRUu3Og7B9sbY7tsFLctQkIqnMh3hErYgdMjTYuqmcXX+lK5T1lnUt3G7zNswmZA==",
      "dependencies": {
        "spdx-expression-parse": "^3.0.0",
        "spdx-license-ids": "^3.0.0"
      }
    },
    "node_modules/spdx-exceptions": {
      "version": "2.5.0",
      "resolved": "https://registry.npmjs.org/spdx-exceptions/-/spdx-exceptions-2.5.0.tgz",
      "integrity": "sha512-PiU42r+xO4UbUS1buo3LPJkjlO7430Xn5SVAhdpzzsPHsjbYVflnnFdATgabnLude+Cqu25p6N+g2lw/PFsa4w=="
    },
    "node_modules/spdx-expression-parse": {
      "version": "3.0.1",
      "resolved": "https://registry.npmjs.org/spdx-expression-parse/-/spdx-expression-parse-3.0.1.tgz",
      "integrity": "sha512-cbqHunsQWnJNE6KhVSMsMeH5H/L9EpymbzqTQ3uLwNCLZ1Q481oWaofqH7nO6V07xlXwY6PhQdQ2IedWx/ZK4Q==",
      "dependencies": {
        "spdx-exceptions": "^2.1.0",
        "spdx-license-ids": "^3.0.0"
      }
    },
    "node_modules/spdx-license-ids": {
      "version": "3.0.21",
      "resolved": "https://registry.npmjs.org/spdx-license-ids/-/spdx-license-ids-3.0.21.tgz",
      "integrity": "sha512-Bvg/8F5XephndSK3JffaRqdT+gyhfqIPwDHpX80tJrF8QQRYMo8sNMeaZ2Dp5+jhwKnUmIOyFFQfHRkjJm5nXg=="
    },
    "node_modules/split": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/split/-/split-1.0.1.tgz",
      "integrity": "sha512-mTyOoPbrivtXnwnIxZRFYRrPNtEFKlpB2fvjSnCQUiAA6qAZzqwna5envK4uk6OIeP17CsdF3rSBGYVBsU0Tkg==",
      "dependencies": {
        "through": "2"
      },
      "engines": {
        "node": "*"
      }
    },
    "node_modules/split2": {
      "version": "3.2.2",
      "resolved": "https://registry.npmjs.org/split2/-/split2-3.2.2.tgz",
      "integrity": "sha512-9NThjpgZnifTkJpzTZ7Eue85S49QwpNhZTq6GRJwObb6jnLFNGB7Qm73V5HewTROPyxD0C29xqmaI68bQtV+hg==",
      "dependencies": {
        "readable-stream": "^3.0.0"
      }
    },
    "node_modules/stream-buffers": {
      "version": "2.2.0",
      "resolved": "https://registry.npmjs.org/stream-buffers/-/stream-buffers-2.2.0.tgz",
      "integrity": "sha512-uyQK/mx5QjHun80FLJTfaWE7JtwfRMKBLkMne6udYOmvH0CawotVa7TfgYHzAnpphn4+TweIx1QKMnRIbipmUg==",
      "engines": {
        "node": ">= 0.10.0"
      }
    },
    "node_modules/streamx": {
      "version": "2.22.0",
      "resolved": "https://registry.npmjs.org/streamx/-/streamx-2.22.0.tgz",
      "integrity": "sha512-sLh1evHOzBy/iWRiR6d1zRcLao4gGZr3C1kzNz4fopCOKJb6xD9ub8Mpi9Mr1R6id5o43S+d93fI48UC5uM9aw==",
      "dependencies": {
        "fast-fifo": "^1.3.2",
        "text-decoder": "^1.1.0"
      },
      "optionalDependencies": {
        "bare-events": "^2.2.0"
      }
    },
    "node_modules/string_decoder": {
      "version": "1.3.0",
      "resolved": "https://registry.npmjs.org/string_decoder/-/string_decoder-1.3.0.tgz",
      "integrity": "sha512-hkRX8U1WjJFd8LsDJ2yQ/wWWxaopEsABU1XfkM8A+j0+85JAGppt16cr1Whg6KIbb4okU6Mql6BOj+uup/wKeA==",
      "dependencies": {
        "safe-buffer": "~5.2.0"
      }
    },
    "node_modules/string_decoder/node_modules/safe-buffer": {
      "version": "5.2.1",
      "resolved": "https://registry.npmjs.org/safe-buffer/-/safe-buffer-5.2.1.tgz",
      "integrity": "sha512-rp3So07KcdmmKbGvgaNxQSJr7bGVSVk5S9Eq1F+ppbRo70+YeaDxkw5Dd8NPN+GD6bjnYm2VuPuCXmpuYvmCXQ==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/feross"
        },
        {
          "type": "patreon",
          "url": "https://www.patreon.com/feross"
        },
        {
          "type": "consulting",
          "url": "https://feross.org/support"
        }
      ]
    },
    "node_modules/string-width": {
      "version": "4.2.3",
      "resolved": "https://registry.npmjs.org/string-width/-/string-width-4.2.3.tgz",
      "integrity": "sha512-wKyQRQpjJ0sIp62ErSZdGsjMJWsap5oRNihHhu6G7JVO/9jIB6UyevL+tXuOqrng8j/cxKTWyWUwvSTriiZz/g==",
      "dependencies": {
        "emoji-regex": "^8.0.0",
        "is-fullwidth-code-point": "^3.0.0",
        "strip-ansi": "^6.0.1"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/string-width-cjs": {
      "name": "string-width",
      "version": "4.2.3",
      "resolved": "https://registry.npmjs.org/string-width/-/string-width-4.2.3.tgz",
      "integrity": "sha512-wKyQRQpjJ0sIp62ErSZdGsjMJWsap5oRNihHhu6G7JVO/9jIB6UyevL+tXuOqrng8j/cxKTWyWUwvSTriiZz/g==",
      "dependencies": {
        "emoji-regex": "^8.0.0",
        "is-fullwidth-code-point": "^3.0.0",
        "strip-ansi": "^6.0.1"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/strip-ansi": {
      "version": "6.0.1",
      "resolved": "https://registry.npmjs.org/strip-ansi/-/strip-ansi-6.0.1.tgz",
      "integrity": "sha512-Y38VPSHcqkFrCpFnQ9vuSXmquuv5oXOKpGeT6aGrr3o3Gc9AlVa6JBfUSOCnbxGGZF+/0ooI7KrPuUSztUdU5A==",
      "dependencies": {
        "ansi-regex": "^5.0.1"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/strip-ansi-cjs": {
      "name": "strip-ansi",
      "version": "6.0.1",
      "resolved": "https://registry.npmjs.org/strip-ansi/-/strip-ansi-6.0.1.tgz",
      "integrity": "sha512-Y38VPSHcqkFrCpFnQ9vuSXmquuv5oXOKpGeT6aGrr3o3Gc9AlVa6JBfUSOCnbxGGZF+/0ooI7KrPuUSztUdU5A==",
      "dependencies": {
        "ansi-regex": "^5.0.1"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/strip-bom": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/strip-bom/-/strip-bom-3.0.0.tgz",
      "integrity": "sha512-vavAMRXOgBVNF6nyEEmL3DBK19iRpDcoIwW+swQ+CbGiu7lju6t+JklA1MHweoWtadgt4ISVUsXLyDq34ddcwA==",
      "engines": {
        "node": ">=4"
      }
    },
    "node_modules/strip-final-newline": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/strip-final-newline/-/strip-final-newline-2.0.0.tgz",
      "integrity": "sha512-BrpvfNAE3dcvq7ll3xVumzjKjZQ5tI1sEUIKr3Uoks0XUl45St3FlatVqef9prk4jRDzhW6WZg+3bk93y6pLjA==",
      "dev": true,
      "engines": {
        "node": ">=6"
      }
    },
    "node_modules/strip-indent": {
      "version": "3.0.0",
      "resolved": "https://registry.npmjs.org/strip-indent/-/strip-indent-3.0.0.tgz",
      "integrity": "sha512-laJTa3Jb+VQpaC6DseHhF7dXVqHTfJPCRDaEbid/drOhgitgYku/letMUqOXFoWV0zIIUbjpdH2t+tYj4bQMRQ==",
      "dependencies": {
        "min-indent": "^1.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/strip-json-comments": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/strip-json-comments/-/strip-json-comments-2.0.1.tgz",
      "integrity": "sha512-4gB8na07fecVVkOI6Rs4e7T6NOTki5EmL7TUduTs6bu3EdnSycntVJ4re8kgZA+wx9IueI2Y11bfbgwtzuE0KQ==",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/sucrase": {
      "version": "3.35.0",
      "resolved": "https://registry.npmjs.org/sucrase/-/sucrase-3.35.0.tgz",
      "integrity": "sha512-8EbVDiu9iN/nESwxeSxDKe0dunta1GOlHufmSSXxMD2z2/tMZpDMpvXQGsc+ajGo8y2uYUmixaSRUc/QPoQ0GA==",
      "dependencies": {
        "@jridgewell/gen-mapping": "^0.3.2",
        "commander": "^4.0.0",
        "glob": "^10.3.10",
        "lines-and-columns": "^1.1.6",
        "mz": "^2.7.0",
        "pirates": "^4.0.1",
        "ts-interface-checker": "^0.1.9"
      },
      "bin": {
        "sucrase": "bin/sucrase",
        "sucrase-node": "bin/sucrase-node"
      },
      "engines": {
        "node": ">=16 || 14 >=14.17"
      }
    },
    "node_modules/sucrase/node_modules/commander": {
      "version": "4.1.1",
      "resolved": "https://registry.npmjs.org/commander/-/commander-4.1.1.tgz",
      "integrity": "sha512-NOKm8xhkzAjzFx8B2v5OAHT+u5pRQc2UCa2Vq9jYL/31o2wi9mxBA7LIFs3sV5VSC49z6pEhfbMULvShKj26WA==",
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/sucrase/node_modules/glob": {
      "version": "10.4.5",
      "resolved": "https://registry.npmjs.org/glob/-/glob-10.4.5.tgz",
      "integrity": "sha512-7Bv8RF0k6xjo7d4A/PxYLbUCfb6c+Vpd2/mB2yRDlew7Jb5hEXiCD9ibfO7wpk8i4sevK6DFny9h7EYbM3/sHg==",
      "dependencies": {
        "foreground-child": "^3.1.0",
        "jackspeak": "^3.1.2",
        "minimatch": "^9.0.4",
        "minipass": "^7.1.2",
        "package-json-from-dist": "^1.0.0",
        "path-scurry": "^1.11.1"
      },
      "bin": {
        "glob": "dist/esm/bin.mjs"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/sucrase/node_modules/minimatch": {
      "version": "9.0.5",
      "resolved": "https://registry.npmjs.org/minimatch/-/minimatch-9.0.5.tgz",
      "integrity": "sha512-G6T0ZX48xgozx7587koeX9Ys2NYy6Gmv//P89sEte9V9whIapMNF4idKxnW2QtCcLiTWlb/wfCabAtAFWhhBow==",
      "dependencies": {
        "brace-expansion": "^2.0.1"
      },
      "engines": {
        "node": ">=16 || 14 >=14.17"
      },
      "funding": {
        "url": "https://github.com/sponsors/isaacs"
      }
    },
    "node_modules/sucrase/node_modules/minipass": {
      "version": "7.1.2",
      "resolved": "https://registry.npmjs.org/minipass/-/minipass-7.1.2.tgz",
      "integrity": "sha512-qOOzS1cBTWYF4BH8fVePDBOO9iptMnGUEZwNc/cMWnTV2nVLZ7VoNWEPHkYczZA0pdoA7dl6e7FL659nX9S2aw==",
      "engines": {
        "node": ">=16 || 14 >=14.17"
      }
    },
    "node_modules/supports-color": {
      "version": "7.2.0",
      "resolved": "https://registry.npmjs.org/supports-color/-/supports-color-7.2.0.tgz",
      "integrity": "sha512-qpCAvRl9stuOHveKsn7HncJRvv501qIacKzQlO/+Lwxc9+0q2wLyv4Dfvt80/DPn2pqOBsJdDiogXGR9+OvwRw==",
      "dev": true,
      "dependencies": {
        "has-flag": "^4.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/supports-preserve-symlinks-flag": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/supports-preserve-symlinks-flag/-/supports-preserve-symlinks-flag-1.0.0.tgz",
      "integrity": "sha512-ot0WnXS9fgdkgIcePe6RHNk1WA8+muPa6cSjeR3V8K27q9BB1rTE3R1p7Hv0z1ZyAc8s6Vvv8DIyWf681MAt0w==",
      "engines": {
        "node": ">= 0.4"
      },
      "funding": {
        "url": "https://github.com/sponsors/ljharb"
      }
    },
    "node_modules/tailwindcss": {
      "version": "3.4.17",
      "resolved": "https://registry.npmjs.org/tailwindcss/-/tailwindcss-3.4.17.tgz",
      "integrity": "sha512-w33E2aCvSDP0tW9RZuNXadXlkHXqFzSkQew/aIa2i/Sj8fThxwovwlXHSPXTbAHwEIhBFXAedUhP2tueAKP8Og==",
      "dependencies": {
        "@alloc/quick-lru": "^5.2.0",
        "arg": "^5.0.2",
        "chokidar": "^3.6.0",
        "didyoumean": "^1.2.2",
        "dlv": "^1.1.3",
        "fast-glob": "^3.3.2",
        "glob-parent": "^6.0.2",
        "is-glob": "^4.0.3",
        "jiti": "^1.21.6",
        "lilconfig": "^3.1.3",
        "micromatch": "^4.0.8",
        "normalize-path": "^3.0.0",
        "object-hash": "^3.0.0",
        "picocolors": "^1.1.1",
        "postcss": "^8.4.47",
        "postcss-import": "^15.1.0",
        "postcss-js": "^4.0.1",
        "postcss-load-config": "^4.0.2",
        "postcss-nested": "^6.2.0",
        "postcss-selector-parser": "^6.1.2",
        "resolve": "^1.22.8",
        "sucrase": "^3.35.0"
      },
      "bin": {
        "tailwind": "lib/cli.js",
        "tailwindcss": "lib/cli.js"
      },
      "engines": {
        "node": ">=14.0.0"
      }
    },
    "node_modules/tar": {
      "version": "6.2.1",
      "resolved": "https://registry.npmjs.org/tar/-/tar-6.2.1.tgz",
      "integrity": "sha512-DZ4yORTwrbTj/7MZYq2w+/ZFdI6OZ/f9SFHR+71gIVUZhOQPHzVCLpvRnPgyaMpfWxxk/4ONva3GQSyNIKRv6A==",
      "dependencies": {
        "chownr": "^2.0.0",
        "fs-minipass": "^2.0.0",
        "minipass": "^5.0.0",
        "minizlib": "^2.1.1",
        "mkdirp": "^1.0.3",
        "yallist": "^4.0.0"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/tar-fs": {
      "version": "3.0.8",
      "resolved": "https://registry.npmjs.org/tar-fs/-/tar-fs-3.0.8.tgz",
      "integrity": "sha512-ZoROL70jptorGAlgAYiLoBLItEKw/fUxg9BSYK/dF/GAGYFJOJJJMvjPAKDJraCXFwadD456FCuvLWgfhMsPwg==",
      "dependencies": {
        "pump": "^3.0.0",
        "tar-stream": "^3.1.5"
      },
      "optionalDependencies": {
        "bare-fs": "^4.0.1",
        "bare-path": "^3.0.0"
      }
    },
    "node_modules/tar-stream": {
      "version": "3.1.7",
      "resolved": "https://registry.npmjs.org/tar-stream/-/tar-stream-3.1.7.tgz",
      "integrity": "sha512-qJj60CXt7IU1Ffyc3NJMjh6EkuCFej46zUqJ4J7pqYlThyd9bO0XBTmcOIhSzZJVWfsLks0+nle/j538YAW9RQ==",
      "dependencies": {
        "b4a": "^1.6.4",
        "fast-fifo": "^1.2.0",
        "streamx": "^2.15.0"
      }
    },
    "node_modules/tar/node_modules/minipass": {
      "version": "5.0.0",
      "resolved": "https://registry.npmjs.org/minipass/-/minipass-5.0.0.tgz",
      "integrity": "sha512-3FnjYuehv9k6ovOEbyOswadCDPX1piCfhV8ncmYtHOjuPwylVWsghTLo7rabjC3Rx5xD4HDx8Wm1xnMF7S5qFQ==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/temp-dir": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/temp-dir/-/temp-dir-2.0.0.tgz",
      "integrity": "sha512-aoBAniQmmwtcKp/7BzsH8Cxzv8OL736p7v1ihGb5e9DJ9kTwGWHrQrVB5+lfVDzfGrdRzXch+ig7LHaY1JTOrg==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/tempy": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/tempy/-/tempy-1.0.1.tgz",
      "integrity": "sha512-biM9brNqxSc04Ee71hzFbryD11nX7VPhQQY32AdDmjFvodsRFz/3ufeoTZ6uYkRFfGo188tENcASNs3vTdsM0w==",
      "dependencies": {
        "del": "^6.0.0",
        "is-stream": "^2.0.0",
        "temp-dir": "^2.0.0",
        "type-fest": "^0.16.0",
        "unique-string": "^2.0.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/tempy/node_modules/type-fest": {
      "version": "0.16.0",
      "resolved": "https://registry.npmjs.org/type-fest/-/type-fest-0.16.0.tgz",
      "integrity": "sha512-eaBzG6MxNzEn9kiwvtre90cXaNLkmadMWa1zQMs3XORCXNbsH/OewwbxC5ia9dCxIxnTAsSxXJaa/p5y8DlvJg==",
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/terser": {
      "version": "5.37.0",
      "resolved": "https://registry.npmjs.org/terser/-/terser-5.37.0.tgz",
      "integrity": "sha512-B8wRRkmre4ERucLM/uXx4MOV5cbnOlVAqUst+1+iLKPI0dOgFO28f84ptoQt9HEI537PMzfYa/d+GEPKTRXmYA==",
      "dev": true,
      "optional": true,
      "peer": true,
      "dependencies": {
        "@jridgewell/source-map": "^0.3.3",
        "acorn": "^8.8.2",
        "commander": "^2.20.0",
        "source-map-support": "~0.5.20"
      },
      "bin": {
        "terser": "bin/terser"
      },
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/terser/node_modules/commander": {
      "version": "2.20.3",
      "resolved": "https://registry.npmjs.org/commander/-/commander-2.20.3.tgz",
      "integrity": "sha512-GpVkmM8vF2vQUkj2LvZmD35JxeJOLCwJ9cUkugyk2nuhbv3+mJvpLYYt+0+USMxE+oj+ey/lJEnhZw75x/OMcQ==",
      "dev": true,
      "optional": true,
      "peer": true
    },
    "node_modules/text-decoder": {
      "version": "1.2.3",
      "resolved": "https://registry.npmjs.org/text-decoder/-/text-decoder-1.2.3.tgz",
      "integrity": "sha512-3/o9z3X0X0fTupwsYvR03pJ/DjWuqqrfwBgTQzdWDiQSm9KitAyz/9WqsT2JQW7KV2m+bC2ol/zqpW37NHxLaA==",
      "dependencies": {
        "b4a": "^1.6.4"
      }
    },
    "node_modules/text-extensions": {
      "version": "1.9.0",
      "resolved": "https://registry.npmjs.org/text-extensions/-/text-extensions-1.9.0.tgz",
      "integrity": "sha512-wiBrwC1EhBelW12Zy26JeOUkQ5mRu+5o8rpsJk5+2t+Y5vE7e842qtZDQ2g1NpX/29HdyFeJ4nSIhI47ENSxlQ==",
      "engines": {
        "node": ">=0.10"
      }
    },
    "node_modules/thenify": {
      "version": "3.3.1",
      "resolved": "https://registry.npmjs.org/thenify/-/thenify-3.3.1.tgz",
      "integrity": "sha512-RVZSIV5IG10Hk3enotrhvz0T9em6cyHBLkH/YAZuKqd8hRkKhSfCGIcP2KUY0EPxndzANBmNllzWPwak+bheSw==",
      "dependencies": {
        "any-promise": "^1.0.0"
      }
    },
    "node_modules/thenify-all": {
      "version": "1.6.0",
      "resolved": "https://registry.npmjs.org/thenify-all/-/thenify-all-1.6.0.tgz",
      "integrity": "sha512-RNxQH/qI8/t3thXJDwcstUO4zeqo64+Uy/+sNVRBx4Xn2OX+OZ9oP+iJnNFqplFra2ZUVeKCSa2oVWi3T4uVmA==",
      "dependencies": {
        "thenify": ">= 3.1.0 < 4"
      },
      "engines": {
        "node": ">=0.8"
      }
    },
    "node_modules/through": {
      "version": "2.3.8",
      "resolved": "https://registry.npmjs.org/through/-/through-2.3.8.tgz",
      "integrity": "sha512-w89qg7PI8wAdvX60bMDP+bFoD5Dvhm9oLheFp5O4a2QF0cSBGsBX4qZmadPMvVqlLJBBci+WqGGOAPvcDeNSVg=="
    },
    "node_modules/through2": {
      "version": "4.0.2",
      "resolved": "https://registry.npmjs.org/through2/-/through2-4.0.2.tgz",
      "integrity": "sha512-iOqSav00cVxEEICeD7TjLB1sueEL+81Wpzp2bY17uZjZN0pWZPuo4suZ/61VujxmqSGFfgOcNuTZ85QJwNZQpw==",
      "dependencies": {
        "readable-stream": "3"
      }
    },
    "node_modules/tmp": {
      "version": "0.2.3",
      "resolved": "https://registry.npmjs.org/tmp/-/tmp-0.2.3.tgz",
      "integrity": "sha512-nZD7m9iCPC5g0pYmcaxogYKggSfLsdxl8of3Q/oIbqCqLLIO9IAF0GWjX1z9NZRHPiXv8Wex4yDCaZsgEw0Y8w==",
      "engines": {
        "node": ">=14.14"
      }
    },
    "node_modules/to-regex-range": {
      "version": "5.0.1",
      "resolved": "https://registry.npmjs.org/to-regex-range/-/to-regex-range-5.0.1.tgz",
      "integrity": "sha512-65P7iz6X5yEr1cwcgvQxbbIw7Uk3gOy5dIdtZ4rDveLqhrdJP+Li/Hx6tyK0NEb+2GCyneCMJiGqrADCSNk8sQ==",
      "dependencies": {
        "is-number": "^7.0.0"
      },
      "engines": {
        "node": ">=8.0"
      }
    },
    "node_modules/tr46": {
      "version": "0.0.3",
      "resolved": "https://registry.npmjs.org/tr46/-/tr46-0.0.3.tgz",
      "integrity": "sha512-N3WMsuqV66lT30CrXNbEjx4GEwlow3v6rr4mCcv6prnfwhS01rkgyFdjPNBYd9br7LpXV1+Emh01fHnq2Gdgrw=="
    },
    "node_modules/tree-kill": {
      "version": "1.2.2",
      "resolved": "https://registry.npmjs.org/tree-kill/-/tree-kill-1.2.2.tgz",
      "integrity": "sha512-L0Orpi8qGpRG//Nd+H90vFB+3iHnue1zSSGmNOOCh1GLJ7rUKVwV2HvijphGQS2UmhUZewS9VgvxYIdgr+fG1A==",
      "bin": {
        "tree-kill": "cli.js"
      }
    },
    "node_modules/trim-newlines": {
      "version": "3.0.1",
      "resolved": "https://registry.npmjs.org/trim-newlines/-/trim-newlines-3.0.1.tgz",
      "integrity": "sha512-c1PTsA3tYrIsLGkJkzHF+w9F2EyxfXGo4UyJc4pFL++FMjnq0HJS69T3M7d//gKrFKwy429bouPescbjecU+Zw==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/ts-interface-checker": {
      "version": "0.1.13",
      "resolved": "https://registry.npmjs.org/ts-interface-checker/-/ts-interface-checker-0.1.13.tgz",
      "integrity": "sha512-Y/arvbn+rrz3JCKl9C4kVNfTfSm2/mEp5FSz5EsZSANGPSlQrpRI5M4PKF+mJnE52jOO90PnPSc3Ur3bTQw0gA=="
    },
    "node_modules/ts-node": {
      "version": "10.9.2",
      "resolved": "https://registry.npmjs.org/ts-node/-/ts-node-10.9.2.tgz",
      "integrity": "sha512-f0FFpIdcHgn8zcPSbf1dRevwt047YMnaiJM3u2w2RewrB+fob/zePZcrOyQoLMMO7aBIddLcQIEK5dYjkLnGrQ==",
      "dependencies": {
        "@cspotcode/source-map-support": "^0.8.0",
        "@tsconfig/node10": "^1.0.7",
        "@tsconfig/node12": "^1.0.7",
        "@tsconfig/node14": "^1.0.0",
        "@tsconfig/node16": "^1.0.2",
        "acorn": "^8.4.1",
        "acorn-walk": "^8.1.1",
        "arg": "^4.1.0",
        "create-require": "^1.1.0",
        "diff": "^4.0.1",
        "make-error": "^1.1.1",
        "v8-compile-cache-lib": "^3.0.1",
        "yn": "3.1.1"
      },
      "bin": {
        "ts-node": "dist/bin.js",
        "ts-node-cwd": "dist/bin-cwd.js",
        "ts-node-esm": "dist/bin-esm.js",
        "ts-node-script": "dist/bin-script.js",
        "ts-node-transpile-only": "dist/bin-transpile.js",
        "ts-script": "dist/bin-script-deprecated.js"
      },
      "peerDependencies": {
        "@swc/core": ">=1.2.50",
        "@swc/wasm": ">=1.2.50",
        "@types/node": "*",
        "typescript": ">=2.7"
      },
      "peerDependenciesMeta": {
        "@swc/core": {
          "optional": true
        },
        "@swc/wasm": {
          "optional": true
        }
      }
    },
    "node_modules/ts-node/node_modules/arg": {
      "version": "4.1.3",
      "resolved": "https://registry.npmjs.org/arg/-/arg-4.1.3.tgz",
      "integrity": "sha512-58S9QDqG0Xx27YwPSt9fJxivjYl432YCwfDMfZ+71RAqUrZef7LrKQZ3LHLOwCS4FLNBplP533Zx895SeOCHvA=="
    },
    "node_modules/ts-node/node_modules/diff": {
      "version": "4.0.2",
      "resolved": "https://registry.npmjs.org/diff/-/diff-4.0.2.tgz",
      "integrity": "sha512-58lmxKSA4BNyLz+HHMUzlOEpg09FV+ev6ZMe3vJihgdxzgcwZ8VoEEPmALCZG9LmqfVoNMMKpttIYTVG6uDY7A==",
      "engines": {
        "node": ">=0.3.1"
      }
    },
    "node_modules/tslib": {
      "version": "2.6.2",
      "resolved": "https://registry.npmjs.org/tslib/-/tslib-2.6.2.tgz",
      "integrity": "sha512-AEYxH93jGFPn/a2iVAwW87VuUIkR1FVUKB77NwMF7nBTDkDrrT/Hpt/IrCJ0QXhW27jTBDcf5ZY7w6RiqTMw2Q=="
    },
    "node_modules/tunnel-agent": {
      "version": "0.6.0",
      "resolved": "https://registry.npmjs.org/tunnel-agent/-/tunnel-agent-0.6.0.tgz",
      "integrity": "sha512-McnNiV1l8RYeY8tBgEpuodCC1mLUdbSN+CYBL7kJsJNInOP8UjDDEwdk6Mw60vdLLrr5NHKZhMAOSrR2NZuQ+w==",
      "dependencies": {
        "safe-buffer": "^5.0.1"
      },
      "engines": {
        "node": "*"
      }
    },
    "node_modules/type-fest": {
      "version": "2.19.0",
      "resolved": "https://registry.npmjs.org/type-fest/-/type-fest-2.19.0.tgz",
      "integrity": "sha512-RAH822pAdBgcNMAfWnCBU3CFZcfZ/i1eZjwFU/dsLKumyuuP3niueg2UAukXYF0E2AAoc82ZSSf9J0WQBinzHA==",
      "dev": true,
      "engines": {
        "node": ">=12.20"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/typescript": {
      "version": "5.7.2",
      "resolved": "https://registry.npmjs.org/typescript/-/typescript-5.7.2.tgz",
      "integrity": "sha512-i5t66RHxDvVN40HfDd1PsEThGNnlMCMT3jMUuoh9/0TaqWevNontacunWyN02LA9/fIbEWlcHZcgTKb9QoaLfg==",
      "peer": true,
      "bin": {
        "tsc": "bin/tsc",
        "tsserver": "bin/tsserver"
      },
      "engines": {
        "node": ">=14.17"
      }
    },
    "node_modules/uglify-js": {
      "version": "3.19.3",
      "resolved": "https://registry.npmjs.org/uglify-js/-/uglify-js-3.19.3.tgz",
      "integrity": "sha512-v3Xu+yuwBXisp6QYTcH4UbH+xYJXqnq2m/LtQVWKWzYc1iehYnLixoQDN9FH6/j9/oybfd6W9Ghwkl8+UMKTKQ==",
      "optional": true,
      "bin": {
        "uglifyjs": "bin/uglifyjs"
      },
      "engines": {
        "node": ">=0.8.0"
      }
    },
    "node_modules/undici-types": {
      "version": "6.20.0",
      "resolved": "https://registry.npmjs.org/undici-types/-/undici-types-6.20.0.tgz",
      "integrity": "sha512-Ny6QZ2Nju20vw1SRHe3d9jVu6gJ+4e3+MMpqu7pqE5HT6WsTSlce++GQmK5UXS8mzV8DSYHrQH+Xrf2jVcuKNg=="
    },
    "node_modules/unique-string": {
      "version": "2.0.0",
      "resolved": "https://registry.npmjs.org/unique-string/-/unique-string-2.0.0.tgz",
      "integrity": "sha512-uNaeirEPvpZWSgzwsPGtU2zVSTrn/8L5q/IexZmH0eH6SA73CmAA5U4GwORTxQAZs95TAXLNqeLoPPNO5gZfWg==",
      "dependencies": {
        "crypto-random-string": "^2.0.0"
      },
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/universalify": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/universalify/-/universalify-2.0.1.tgz",
      "integrity": "sha512-gptHNQghINnc/vTGIk0SOFGFNXw7JVrlRUtConJRlvaw6DuX0wO5Jeko9sWrMBhh+PsYAZ7oXAiOnf/UKogyiw==",
      "engines": {
        "node": ">= 10.0.0"
      }
    },
    "node_modules/untildify": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/untildify/-/untildify-4.0.0.tgz",
      "integrity": "sha512-KK8xQ1mkzZeg9inewmFVDNkg3l5LUhoq9kN6iWYB/CC9YMG8HA+c1Q8HwDe6dEX7kErrEVNVBO3fWsVq5iDgtw==",
      "engines": {
        "node": ">=8"
      }
    },
    "node_modules/update-browserslist-db": {
      "version": "1.1.1",
      "resolved": "https://registry.npmjs.org/update-browserslist-db/-/update-browserslist-db-1.1.1.tgz",
      "integrity": "sha512-R8UzCaa9Az+38REPiJ1tXlImTJXlVfgHZsglwBD/k6nj76ctsH1E3q4doGrukiLQd3sGQYu56r5+lo5r94l29A==",
      "dev": true,
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/browserslist"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/browserslist"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "dependencies": {
        "escalade": "^3.2.0",
        "picocolors": "^1.1.0"
      },
      "bin": {
        "update-browserslist-db": "cli.js"
      },
      "peerDependencies": {
        "browserslist": ">= 4.21.0"
      }
    },
    "node_modules/update-check": {
      "version": "1.5.4",
      "resolved": "https://registry.npmjs.org/update-check/-/update-check-1.5.4.tgz",
      "integrity": "sha512-5YHsflzHP4t1G+8WGPlvKbJEbAJGCgw+Em+dGR1KmBUbr1J36SJBqlHLjR7oob7sco5hWHGQVcr9B2poIVDDTQ==",
      "dev": true,
      "dependencies": {
        "registry-auth-token": "3.3.2",
        "registry-url": "3.1.0"
      }
    },
    "node_modules/uri-js": {
      "version": "4.4.1",
      "resolved": "https://registry.npmjs.org/uri-js/-/uri-js-4.4.1.tgz",
      "integrity": "sha512-7rKUyy33Q1yc98pQ1DAmLtwX109F7TIfWlW1Ydo8Wl1ii1SeHieeh0HHfPeL2fMXK6z0s8ecKs9frCuLJvndBg==",
      "dev": true,
      "dependencies": {
        "punycode": "^2.1.0"
      }
    },
    "node_modules/util-deprecate": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/util-deprecate/-/util-deprecate-1.0.2.tgz",
      "integrity": "sha512-EPD5q1uXyFxJpCrLnCc1nHnq3gOa6DZBocAIiI2TaSCA7VCJ1UJDMagCzIkXNsUYfD1daK//LTEQ8xiIbrHtcw=="
    },
    "node_modules/uuid": {
      "version": "7.0.3",
      "resolved": "https://registry.npmjs.org/uuid/-/uuid-7.0.3.tgz",
      "integrity": "sha512-DPSke0pXhTZgoF/d+WSt2QaKMCFSfx7QegxEWT+JOuHF5aWrKEn0G+ztjuJg/gG8/ItK+rbPCD/yNv8yyih6Cg==",
      "bin": {
        "uuid": "dist/bin/uuid"
      }
    },
    "node_modules/v8-compile-cache-lib": {
      "version": "3.0.1",
      "resolved": "https://registry.npmjs.org/v8-compile-cache-lib/-/v8-compile-cache-lib-3.0.1.tgz",
      "integrity": "sha512-wa7YjyUGfNZngI/vtK0UHAN+lgDCxBPCylVXGp0zu59Fz5aiGtNXaq3DhIov063MorB+VfufLh3JlF2KdTK3xg=="
    },
    "node_modules/validate-npm-package-license": {
      "version": "3.0.4",
      "resolved": "https://registry.npmjs.org/validate-npm-package-license/-/validate-npm-package-license-3.0.4.tgz",
      "integrity": "sha512-DpKm2Ui/xN7/HQKCtpZxoRWBhZ9Z0kqtygG8XCgNQ8ZlDnxuQmWhj566j8fN4Cu3/JmbhsDo7fcAJq4s9h27Ew==",
      "dependencies": {
        "spdx-correct": "^3.0.0",
        "spdx-expression-parse": "^3.0.0"
      }
    },
    "node_modules/vary": {
      "version": "1.1.2",
      "resolved": "https://registry.npmjs.org/vary/-/vary-1.1.2.tgz",
      "integrity": "sha512-BNGbWLfd0eUPabhkXUVm0j8uuvREyTh5ovRa/dyow/BqAbZJyC+5fU+IzQOzmAKzYqYRAISoRhdQr3eIZ/PXqg==",
      "dev": true,
      "engines": {
        "node": ">= 0.8"
      }
    },
    "node_modules/vite": {
      "version": "6.0.5",
      "resolved": "https://registry.npmjs.org/vite/-/vite-6.0.5.tgz",
      "integrity": "sha512-akD5IAH/ID5imgue2DYhzsEwCi0/4VKY31uhMLEYJwPP4TiUp8pL5PIK+Wo7H8qT8JY9i+pVfPydcFPYD1EL7g==",
      "dev": true,
      "dependencies": {
        "esbuild": "0.24.0",
        "postcss": "^8.4.49",
        "rollup": "^4.23.0"
      },
      "bin": {
        "vite": "bin/vite.js"
      },
      "engines": {
        "node": "^18.0.0 || ^20.0.0 || >=22.0.0"
      },
      "funding": {
        "url": "https://github.com/vitejs/vite?sponsor=1"
      },
      "optionalDependencies": {
        "fsevents": "~2.3.3"
      },
      "peerDependencies": {
        "@types/node": "^18.0.0 || ^20.0.0 || >=22.0.0",
        "jiti": ">=1.21.0",
        "less": "*",
        "lightningcss": "^1.21.0",
        "sass": "*",
        "sass-embedded": "*",
        "stylus": "*",
        "sugarss": "*",
        "terser": "^5.16.0",
        "tsx": "^4.8.1",
        "yaml": "^2.4.2"
      },
      "peerDependenciesMeta": {
        "@types/node": {
          "optional": true
        },
        "jiti": {
          "optional": true
        },
        "less": {
          "optional": true
        },
        "lightningcss": {
          "optional": true
        },
        "sass": {
          "optional": true
        },
        "sass-embedded": {
          "optional": true
        },
        "stylus": {
          "optional": true
        },
        "sugarss": {
          "optional": true
        },
        "terser": {
          "optional": true
        },
        "tsx": {
          "optional": true
        },
        "yaml": {
          "optional": true
        }
      }
    },
    "node_modules/webidl-conversions": {
      "version": "3.0.1",
      "resolved": "https://registry.npmjs.org/webidl-conversions/-/webidl-conversions-3.0.1.tgz",
      "integrity": "sha512-2JAn3z8AR6rjK8Sm8orRC0h/bcl/DqL7tRPdGZ4I1CjdF+EaMLmYxBHyXuKL849eucPFhvBoxMsflfOb8kxaeQ=="
    },
    "node_modules/whatwg-url": {
      "version": "5.0.0",
      "resolved": "https://registry.npmjs.org/whatwg-url/-/whatwg-url-5.0.0.tgz",
      "integrity": "sha512-saE57nupxk6v3HY35+jzBwYa0rKSy0XR8JSxZPwgLr7ys0IBzhGviA1/TUGJLmSVqs8pb9AnvICXEuOHLprYTw==",
      "dependencies": {
        "tr46": "~0.0.3",
        "webidl-conversions": "^3.0.0"
      }
    },
    "node_modules/which": {
      "version": "2.0.2",
      "resolved": "https://registry.npmjs.org/which/-/which-2.0.2.tgz",
      "integrity": "sha512-BLI3Tl1TW3Pvl70l3yq3Y64i+awpwXqsGBYWkkqMtnbXgrMD+yj7rhW0kuEDxzJaYXGjEW5ogapKNMEKNMjibA==",
      "dependencies": {
        "isexe": "^2.0.0"
      },
      "bin": {
        "node-which": "bin/node-which"
      },
      "engines": {
        "node": ">= 8"
      }
    },
    "node_modules/which-module": {
      "version": "2.0.1",
      "resolved": "https://registry.npmjs.org/which-module/-/which-module-2.0.1.tgz",
      "integrity": "sha512-iBdZ57RDvnOR9AGBhML2vFZf7h8vmBjhoaZqODJBFWHVtKkDmKuHai3cx5PgVMrX5YDNp27AofYbAwctSS+vhQ=="
    },
    "node_modules/widest-line": {
      "version": "4.0.1",
      "resolved": "https://registry.npmjs.org/widest-line/-/widest-line-4.0.1.tgz",
      "integrity": "sha512-o0cyEG0e8GPzT4iGHphIOh0cJOV8fivsXxddQasHPHfoZf1ZexrfeA21w2NaEN1RHE+fXlfISmOE8R9N3u3Qig==",
      "dev": true,
      "dependencies": {
        "string-width": "^5.0.1"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/widest-line/node_modules/ansi-regex": {
      "version": "6.1.0",
      "resolved": "https://registry.npmjs.org/ansi-regex/-/ansi-regex-6.1.0.tgz",
      "integrity": "sha512-7HSX4QQb4CspciLpVFwyRe79O3xsIZDDLER21kERQ71oaPodF8jL725AgJMFAYbooIqolJoRLuM81SpeUkpkvA==",
      "dev": true,
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/ansi-regex?sponsor=1"
      }
    },
    "node_modules/widest-line/node_modules/emoji-regex": {
      "version": "9.2.2",
      "resolved": "https://registry.npmjs.org/emoji-regex/-/emoji-regex-9.2.2.tgz",
      "integrity": "sha512-L18DaJsXSUk2+42pv8mLs5jJT2hqFkFE4j21wOmgbUqsZ2hL72NsUU785g9RXgo3s0ZNgVl42TiHp3ZtOv/Vyg==",
      "dev": true
    },
    "node_modules/widest-line/node_modules/string-width": {
      "version": "5.1.2",
      "resolved": "https://registry.npmjs.org/string-width/-/string-width-5.1.2.tgz",
      "integrity": "sha512-HnLOCR3vjcY8beoNLtcjZ5/nxn2afmME6lhrDrebokqMap+XbeW8n9TXpPDOqdGK5qcI3oT0GKTW6wC7EMiVqA==",
      "dev": true,
      "dependencies": {
        "eastasianwidth": "^0.2.0",
        "emoji-regex": "^9.2.2",
        "strip-ansi": "^7.0.1"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/sponsors/sindresorhus"
      }
    },
    "node_modules/widest-line/node_modules/strip-ansi": {
      "version": "7.1.0",
      "resolved": "https://registry.npmjs.org/strip-ansi/-/strip-ansi-7.1.0.tgz",
      "integrity": "sha512-iq6eVVI64nQQTRYq2KtEg2d2uU7LElhTJwsH4YzIHZshxlgZms/wIc4VoDQTlG/IvVIrBKG06CrZnp0qv7hkcQ==",
      "dev": true,
      "dependencies": {
        "ansi-regex": "^6.0.1"
      },
      "engines": {
        "node": ">=12"
      },
      "funding": {
        "url": "https://github.com/chalk/strip-ansi?sponsor=1"
      }
    },
    "node_modules/wordwrap": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/wordwrap/-/wordwrap-1.0.0.tgz",
      "integrity": "sha512-gvVzJFlPycKc5dZN4yPkP8w7Dc37BtP1yczEneOb4uq34pXZcvrtRTmWV8W+Ume+XCxKgbjM+nevkyFPMybd4Q=="
    },
    "node_modules/wrap-ansi": {
      "version": "7.0.0",
      "resolved": "https://registry.npmjs.org/wrap-ansi/-/wrap-ansi-7.0.0.tgz",
      "integrity": "sha512-YVGIj2kamLSTxw6NsZjoBxfSwsn0ycdesmc4p+Q21c5zPuZ1pl+NfxVdxPtdHvmNVOQ6XSYG4AUtyt/Fi7D16Q==",
      "dependencies": {
        "ansi-styles": "^4.0.0",
        "string-width": "^4.1.0",
        "strip-ansi": "^6.0.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/chalk/wrap-ansi?sponsor=1"
      }
    },
    "node_modules/wrap-ansi-cjs": {
      "name": "wrap-ansi",
      "version": "7.0.0",
      "resolved": "https://registry.npmjs.org/wrap-ansi/-/wrap-ansi-7.0.0.tgz",
      "integrity": "sha512-YVGIj2kamLSTxw6NsZjoBxfSwsn0ycdesmc4p+Q21c5zPuZ1pl+NfxVdxPtdHvmNVOQ6XSYG4AUtyt/Fi7D16Q==",
      "dependencies": {
        "ansi-styles": "^4.0.0",
        "string-width": "^4.1.0",
        "strip-ansi": "^6.0.0"
      },
      "engines": {
        "node": ">=10"
      },
      "funding": {
        "url": "https://github.com/chalk/wrap-ansi?sponsor=1"
      }
    },
    "node_modules/wrappy": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/wrappy/-/wrappy-1.0.2.tgz",
      "integrity": "sha512-l4Sp/DRseor9wL6EvV2+TuQn63dMkPjZ/sp9XkghTEbV9KlPS1xUsZ3u7/IQO4wxtcFB4bgpQPRcR3QCvezPcQ=="
    },
    "node_modules/xcode": {
      "version": "3.0.1",
      "resolved": "https://registry.npmjs.org/xcode/-/xcode-3.0.1.tgz",
      "integrity": "sha512-kCz5k7J7XbJtjABOvkc5lJmkiDh8VhjVCGNiqdKCscmVpdVUpEAyXv1xmCLkQJ5dsHqx3IPO4XW+NTDhU/fatA==",
      "dependencies": {
        "simple-plist": "^1.1.0",
        "uuid": "^7.0.3"
      },
      "engines": {
        "node": ">=10.0.0"
      }
    },
    "node_modules/xml-js": {
      "version": "1.6.11",
      "resolved": "https://registry.npmjs.org/xml-js/-/xml-js-1.6.11.tgz",
      "integrity": "sha512-7rVi2KMfwfWFl+GpPg6m80IVMWXLRjO+PxTq7V2CDhoGak0wzYzFgUY2m4XJ47OGdXd8eLE8EmwfAmdjw7lC1g==",
      "dependencies": {
        "sax": "^1.2.4"
      },
      "bin": {
        "xml-js": "bin/cli.js"
      }
    },
    "node_modules/xml-js/node_modules/sax": {
      "version": "1.4.1",
      "resolved": "https://registry.npmjs.org/sax/-/sax-1.4.1.tgz",
      "integrity": "sha512-+aWOz7yVScEGoKNd4PA10LZ8sk0A/z5+nXQG5giUO5rprX9jgYsTdov9qCchZiPIZezbZH+jRut8nPodFAX4Jg=="
    },
    "node_modules/xml2js": {
      "version": "0.5.0",
      "resolved": "https://registry.npmjs.org/xml2js/-/xml2js-0.5.0.tgz",
      "integrity": "sha512-drPFnkQJik/O+uPKpqSgr22mpuFHqKdbS835iAQrUC73L2F5WkboIRd63ai/2Yg6I1jzifPFKH2NTK+cfglkIA==",
      "dependencies": {
        "sax": ">=0.6.0",
        "xmlbuilder": "~11.0.0"
      },
      "engines": {
        "node": ">=4.0.0"
      }
    },
    "node_modules/xml2js/node_modules/xmlbuilder": {
      "version": "11.0.1",
      "resolved": "https://registry.npmjs.org/xmlbuilder/-/xmlbuilder-11.0.1.tgz",
      "integrity": "sha512-fDlsI/kFEx7gLvbecc0/ohLG50fugQp8ryHzMTuW9vSa1GJ0XYWKnhsUx7oie3G98+r56aTQIUB4kht42R3JvA==",
      "engines": {
        "node": ">=4.0"
      }
    },
    "node_modules/xmlbuilder": {
      "version": "15.1.1",
      "resolved": "https://registry.npmjs.org/xmlbuilder/-/xmlbuilder-15.1.1.tgz",
      "integrity": "sha512-yMqGBqtXyeN1e3TGYvgNgDVZ3j84W4cwkOXQswghol6APgZWaff9lnbvN7MHYJOiXsvGPXtjTYJEiC9J2wv9Eg==",
      "engines": {
        "node": ">=8.0"
      }
    },
    "node_modules/xpath": {
      "version": "0.0.32",
      "resolved": "https://registry.npmjs.org/xpath/-/xpath-0.0.32.tgz",
      "integrity": "sha512-rxMJhSIoiO8vXcWvSifKqhvV96GjiD5wYb8/QHdoRyQvraTpp4IEv944nhGausZZ3u7dhQXteZuZbaqfpB7uYw==",
      "engines": {
        "node": ">=0.6.0"
      }
    },
    "node_modules/xtend": {
      "version": "4.0.2",
      "resolved": "https://registry.npmjs.org/xtend/-/xtend-4.0.2.tgz",
      "integrity": "sha512-LKYU1iAXJXUgAXn9URjiu+MWhyUXHsvfp7mcuYm9dSUKK0/CjtrUwFAxD82/mCWbtLsGjFIad0wIsod4zrTAEQ==",
      "engines": {
        "node": ">=0.4"
      }
    },
    "node_modules/y18n": {
      "version": "5.0.8",
      "resolved": "https://registry.npmjs.org/y18n/-/y18n-5.0.8.tgz",
      "integrity": "sha512-0pfFzegeDWJHJIAmTLRP2DwHjdF5s7jo9tuztdQxAhINCdvS+3nGINqPd00AphqJR/0LhANUS6/+7SCb98YOfA==",
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/yallist": {
      "version": "4.0.0",
      "resolved": "https://registry.npmjs.org/yallist/-/yallist-4.0.0.tgz",
      "integrity": "sha512-3wdGidZyq5PB084XLES5TpOSRA3wjXAlIWMhum2kRcv/41Sn2emQ0dycQW4uZXLejwKvg6EsvbdlVL+FYEct7A=="
    },
    "node_modules/yaml": {
      "version": "2.6.1",
      "resolved": "https://registry.npmjs.org/yaml/-/yaml-2.6.1.tgz",
      "integrity": "sha512-7r0XPzioN/Q9kXBro/XPnA6kznR73DHq+GXh5ON7ZozRO6aMjbmiBuKste2wslTFkC5d1dw0GooOCepZXJ2SAg==",
      "bin": {
        "yaml": "bin.mjs"
      },
      "engines": {
        "node": ">= 14"
      }
    },
    "node_modules/yargs": {
      "version": "17.7.2",
      "resolved": "https://registry.npmjs.org/yargs/-/yargs-17.7.2.tgz",
      "integrity": "sha512-7dSzzRQ++CKnNI/krKnYRV7JKKPUXMEh61soaHKg9mrWEhzFWhFnxPxGl+69cD1Ou63C13NUPCnmIcrvqCuM6w==",
      "dependencies": {
        "cliui": "^8.0.1",
        "escalade": "^3.1.1",
        "get-caller-file": "^2.0.5",
        "require-directory": "^2.1.1",
        "string-width": "^4.2.3",
        "y18n": "^5.0.5",
        "yargs-parser": "^21.1.1"
      },
      "engines": {
        "node": ">=12"
      }
    },
    "node_modules/yargs-parser": {
      "version": "20.2.9",
      "resolved": "https://registry.npmjs.org/yargs-parser/-/yargs-parser-20.2.9.tgz",
      "integrity": "sha512-y11nGElTIV+CT3Zv9t7VKl+Q3hTQoT9a1Qzezhhl6Rp21gJ/IVTW7Z3y9EWXhuUBC2Shnf+DX0antecpAwSP8w==",
      "engines": {
        "node": ">=10"
      }
    },
    "node_modules/yargs/node_modules/yargs-parser": {
      "version": "21.1.1",
      "resolved": "https://registry.npmjs.org/yargs-parser/-/yargs-parser-21.1.1.tgz",
      "integrity": "sha512-tVpsJW7DdjecAiFpbIB1e3qxIQsE6NoPc5/eTdrbbIC4h0LVsWhnoa3g+m2HclBIujHzsxZ4VJVA+GUuc2/LBw==",
      "engines": {
        "node": ">=12"
      }
    },
    "node_modules/yauzl": {
      "version": "2.10.0",
      "resolved": "https://registry.npmjs.org/yauzl/-/yauzl-2.10.0.tgz",
      "integrity": "sha512-p4a9I6X6nu6IhoGmBqAcbJy1mlC4j27vEPZX9F4L4/vZT3Lyq1VkFHw/V/PUcB9Buo+DG3iHkT0x3Qya58zc3g==",
      "dependencies": {
        "buffer-crc32": "~0.2.3",
        "fd-slicer": "~1.1.0"
      }
    },
    "node_modules/yn": {
      "version": "3.1.1",
      "resolved": "https://registry.npmjs.org/yn/-/yn-3.1.1.tgz",
      "integrity": "sha512-Ux4ygGWsu2c7isFWe8Yu1YluJmqVhxqK2cLXNQA5AcC3QfbGNpM7fu0Y8b/z16pXLnFxZYvWhd3fhBY9DLmC6Q==",
      "engines": {
        "node": ">=6"
      }
    }
  }
}


================================================================================
FILE: ./frontend/package.json
================================================================================
{
  "name": "say-less",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "sync": "npx cap sync",
    "postbuild": "npx cap sync"
  },
  "keywords": ["decision", "ai", "choices"],
  "author": "",
  "license": "ISC",
  "description": "Make decisions with confidence",
  "dependencies": {
    "@capacitor-community/in-app-review": "^6.0.0",
    "@capacitor/android": "^6.2.0",
    "@capacitor/app": "^6.0.2",
    "@capacitor/assets": "^3.0.5",
    "@capacitor/cli": "^6.2.0",
    "@capacitor/core": "^6.2.0",
    "@capacitor/ios": "^6.2.0",
    "@capgo/capacitor-updater": "^6.3.8",
    "@fortawesome/fontawesome-free": "^6.7.2",
    "@revenuecat/purchases-capacitor": "^9.0.9",
    "alpinejs": "^3.14.7",
    "animate.css": "^4.1.1",
    "htmx.org": "^1.9.12",
    "marked": "^15.0.4",
    "tailwindcss": "^3.4.1"
  },
  "devDependencies": {
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.49",
    "serve": "^14.2.4",
    "vite": "^6.0.5"
  }
}


================================================================================
FILE: ./frontend/capacitor.config.json
================================================================================
{
  "appId": "com.bible.genius",
  "appName": "Bible Genius",
  "webDir": "dist",
  "server": {
    "androidScheme": "https",
    "cleartext": true
  },
  "plugins": {
    "PurchasesPlugin": {
      "apiKey": {
        "android": "goog_iAMZTjuqpJNDPznmrJLCQKBsUKp",
        "ios": "appl_vxDzATFBkVKsVdSrBaMzpmMSYpK"
      }
    },
    "CapacitorUpdater": {
      "appReadyTimeout": 5000,
      "responseTimeout": 10,
      "version": "1.1.0",
      "autoUpdate": true,
      "directUpdate": false,
      "appId": "com.bible.genius",
      "updateUrl": "https://prompt-pm--bloom-updater-fastapi-app.modal.run/check_update",
      "statsUrl": "https://prompt-pm--bloom-updater-fastapi-app.modal.run/log_update"
    }
  }
}


================================================================================
FILE: ./backend.py
================================================================================
from fastapi import Request, FastAPI, Depends
import modal
import os
import json
from modal import App, Image, asgi_app, Mount
from ai_functions import (
    cerebras_choose_option,
    cerebras_generate_outcomes,
    cerebras_generate_alternative,
    cerebras_router,
    generate_question,
    cerebras_generate_next_steps,
    cerebras_suggest_additional_action,
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from instrumentor import set_easy_tracing_instrumentation
from phoenix.trace import using_project
import sentry_sdk
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database import Base, DecisionRecord

tracing_initialized = False


def initiate_sentry():
    sentry_sdk.init(
        dsn="https://37b97329990b0c3eecf0bb5eefb64136@o190156.ingest.us.sentry.io/4508088928567296",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )


def initiate_tracing():
    global tracing_initialized
    if not tracing_initialized:
        set_easy_tracing_instrumentation()
        # set_hosted_phoenix_instrumentation()
        initiate_sentry()
        tracing_initialized = True
        print("Tracing instrumentation initialized")


# image = (
#     Image.debian_slim()
#     .pip_install("uv")
#     .run_commands("uv pip install --system --compile-bytecode ./requirements.txt")
# )

image = Image.debian_slim().pip_install_from_requirements("./requirements.txt").add_local_python_source("ai_functions", "instrumentor", "models", "prompts", "database")
mount = Mount.from_local_dir("./assets", remote_path="/assets")
app = App(image=image, mounts=[mount])
web_app = FastAPI()

# Database setup
# Use a path from the root directory to assets/data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "assets", "data")
# Create the data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)
db_path = os.path.join(data_dir, "decision_records.db")
DATABASE_URL = f"sqlite:///{db_path}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Dependency to provide a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5500",
    "http://localhost:5501",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501",
    "http://choices.dev",
    "https://choices.dev",
    "http://oksayless.com",
    "https://oksayless.com",
    "http://overthinking.app",
    "https://overthinking.app",
    "http://overthinking.help",
    "https://overthinking.help",
]

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.post("/api/initial_query/")
async def initial_query(request: Request):
    initiate_tracing()
    data = await request.json()
    message_history = data.get("message_history", [])

    with using_project("say-less"):
        response = cerebras_router(message_history)
        response_json = {
            "prompt": response.prompt,
            "response": response.response.model_dump(),
        }
        return JSONResponse(content=response_json)


@web_app.post("/api/query/")
async def query(request: Request):
    initiate_tracing()
    data = await request.json()
    messages = data.get("messages", [])
    with using_project("say-less"):
        response = cerebras_router(messages)
        response_json = {
            "prompt": response.prompt,
            "response": response.response.model_dump(),
        }
        return JSONResponse(content=response_json)


@web_app.post("/api/questions/")
async def questions(request: Request):
    initiate_tracing()
    data = await request.json()
    situation = data.get("situation")
    results = data.get("results")
    questions = data.get("questions")
    with using_project("say-less"):
        response_message = generate_question(situation, results, questions)

    response_json = {
        "uncertainties": [response_message],
    }
    return JSONResponse(content=response_json)


@web_app.post("/api/choices")
async def simulate_choices(request: Request):
    initiate_tracing()
    data = await request.json()
    message_history = data.get("message_history", [])

    with using_project("say-less"):
        response = cerebras_generate_outcomes(message_history)

    response_json = {
        "choices": [choice.model_dump() for choice in response.choices],
        "title": response.title,
        "uncertainties": response.uncertainties,
        "next_steps": response.next_steps,
    }
    return JSONResponse(content=response_json)


@web_app.get("/")
def read_root():
    initiate_tracing()
    return {"Hello": "World"}


@web_app.get("/sentry-debug")
async def trigger_error():
    initiate_tracing()
    division_by_zero = 1 / 0


@app.function(
    image=image,
    mounts=[mount],
    allow_concurrent_inputs=50,
    keep_warm=1,
    secrets=[
        modal.Secret.from_name("openai-key"),
        modal.Secret.from_name("cerebras-key"),
    ],
)
@asgi_app()
def fastapi_app():
    return web_app


@web_app.post("/api/add_alternative/")
async def add_alternative(request: Request):
    initiate_tracing()
    data = await request.json()
    situation = data.get("situation", "")
    results = data.get("results")

    with using_project("say-less"):
        response = cerebras_generate_alternative(situation, results)

    response_json = {
        "new_alternative": response.model_dump(),
    }

    return JSONResponse(content=response_json)


@web_app.post("/api/choose/")
async def choose_option(request: Request):
    initiate_tracing()
    data = await request.json()
    situation = data.get("situation", "")
    results = data.get("results")
    selected_index = data.get("selectedIndex")

    with using_project("say-less"):
        response = cerebras_choose_option(situation, results, selected_index)

    response_json = {
        "chosen_index": response.chosen_index,
        "explanation": response.explanation,
    }

    return JSONResponse(content=response_json)


@web_app.post("/api/next_steps")
async def next_steps(request: Request):
    initiate_tracing()
    data = await request.json()
    situation = data.get("situation", "")
    choice_name = data.get("choice_name", "")
    choice_index = data.get("choice_index")
    results = data.get("results")

    with using_project("say-less"):
        response = cerebras_generate_next_steps(situation, choice_name, results, choice_index)

    response_json = {
        "next_steps": response.next_steps,
    }

    return JSONResponse(content=response_json)


@web_app.post("/api/suggest_additional_action")
async def suggest_additional_action(request: Request):
    initiate_tracing()
    data = await request.json()
    situation = data.get("situation", "")
    existing_next_steps = data.get("existing_next_steps", [])
    results = data.get("results", {})

    with using_project("say-less"):
        response = cerebras_suggest_additional_action(situation, existing_next_steps, results)

    response_json = {
        "additional_action": response.additional_action,
    }
    return JSONResponse(content=response_json)


@web_app.post("/api/save_decision/")
async def save_decision(request: Request, db: Session = Depends(get_db)):
    initiate_tracing()
    data = await request.json()
    message_history = data.get("message_history", [])  # List of message objects
    if not message_history:
        return JSONResponse(content={"error": "Message history is empty"}, status_code=400)
    
    unique_id = str(uuid4())  # Generate a unique ID
    message_history_json = json.dumps(message_history)  # Convert to JSON string
    decision_record = DecisionRecord(id=unique_id, message_history=message_history_json)
    
    db.add(decision_record)
    db.commit()
    return JSONResponse(content={"id": unique_id})


@web_app.get("/api/get_decision/{decision_id}")
async def get_decision(decision_id: str, db: Session = Depends(get_db)):
    initiate_tracing()
    record = db.query(DecisionRecord).filter(DecisionRecord.id == decision_id).first()
    if record is None:
        return JSONResponse(content={"error": "Decision not found"}, status_code=404)
    message_history = json.loads(record.message_history)
    return JSONResponse(content={"message_history": message_history})


================================================================================
FILE: ./models.py
================================================================================
from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class Prompt(str, Enum):
    NORMAL = "normal"
    WEB_SEARCH = "web_search"
    # CHOOSE_FOR_ME = "choose_for_me"
    # GENERATE_FEEDBACK = "generate_feedback"
    # DECISION_CHAT = "decision_chat"

    def to_json(self):
        return self.value

    def __json__(self):
        return self.value


class PromptSelection(BaseModel):
    prompt: str = Field(
        description="The prompt to use to answer the user's question most effectively. Return the prompt title and nothing else."
    )


class RoleplayScenarioResponse(BaseModel):
    response: str = Field(description="The response of the narrator.")
    suggested_choices: List[str] = Field(
        description="A list of 3 choices that the user could make to move forward in the scenario. Keep these choices to less than 10 words long. Be as specific as possible.",
    )


class MessageResponse(BaseModel):
    text: str = Field(
        description="Your response to the conversation."
    )
    suggested_messages: List[str] = Field(
        ...,
        description="A list of suggested messages that the user could send back to the response you generate. The messages should be from the user's point of view in the conversation. If you are asking me a question in your response, the suggested messages should be answers.",
    )


class RouterResponse(BaseModel):
    response: BaseModel = Field(description="The response of the router.")
    prompt: Prompt = Field(
        description="The prompt that was used to generate the response."
    )


class UIResponse(BaseModel):
    text: str = Field(description="The text to display to the user.")
    citations: List[str] = Field(
        default=[],
        description="A list of citations for the response. These should be links to the sources of the information in the response.",
    )
    suggested_messages: List[str] = Field(
        default=[],
        description="A list of suggested follow-up messages that the user might want to send.",
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


class ChooseForMeResponse(BaseModel):
    explanation: str = Field(
        description="A short explanation of why this choice could be a good decision. Use at most 30 words. Write as though you are speaking to me, and use words that I would expect to hear from a friend as a suggestion. The explanation should be a single complete sentence."
    )
    chosen_index: int = Field(
        description="The index of the chosen option. The index starts at 0. It can go up to the number of options minus 1."
    )


class Character(BaseModel):
    name: str
    age: int
    gender: str
    location: str
    occupation: str
    mood: str
    facts: List[str] = Field(..., description="A list of 3 facts about the character")
    goals: List[str] = Field(
        ..., description="A list of 3 goals the character is trying to achieve"
    )
    fears: List[str] = Field(
        ..., description="A list of 3 fears the character is experiencing"
    )
    desires: List[str] = Field(
        ..., description="A list of 3 desires the character is experiencing"
    )
    joys: List[str] = Field(
        ..., description="A list of 3 joys the character is experiencing"
    )
    problems: List[str] = Field(
        ..., description="A list of 3 problems the character is facing"
    )


class Message(BaseModel):
    role: str
    content: str


class MessageList(BaseModel):
    messages: List[Message] = Field(description="A list of messages.")

    def __str__(self):
        return "\n".join(
            [f"{message.role}: {message.content}" for message in self.messages]
        )

    def add_message(self, message: Message):
        self.messages.append(message)

    def prepend_message(self, message: Message):
        self.messages.insert(0, message)

    def clear(self):
        self.messages = []

    def return_openai_format(self):
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]

    def return_persona_format(self):
        message_list = []
        for message in self.messages:
            if message.role == "user":
                message_list.append(f"user: {message.content}")
            else:
                message_list.append(f"persona: {message.content}")
        return "\n".join(message_list)

    def return_json(self):
        return [message.model_dump() for message in self.messages]


class NextStepsResponse(BaseModel):
    next_steps: List[str] = Field(
        ...,
        min_items=1,  # Ensures at least 1 step
        max_items=2,  # Limits to no more than 2 steps
        description="A list of 1-2 specific, actionable next steps for the user. Each step should be clear, concrete, and start with a verb. Each step should be 5-15 words and directly applicable to the situation."
    )


class AdditionalActionResponse(BaseModel):
    additional_action: str = Field(
        description="A single additional action suggestion that complements the user's existing actions. The action should be specific, actionable, start with a verb, and be 5-15 words long."
    )


================================================================================
FILE: ./requirements.txt
================================================================================
fastapi
instructor[cerebras_cloud_sdk]
openai
pydantic
uvicorn
pytest
modal
arize-phoenix
arize-phoenix-otel
black
flake8
openinference-instrumentation-groq
openinference-instrumentation-openai
openinference-instrumentation-instructor
groq
sentry-sdk
braintrust
sqlalchemy

================================================================================
FILE: ./Makefile
================================================================================
-include .env
export

.PHONY: dev
dev:
	python3 -m uvicorn backend:web_app --reload

.PHONY: deploy
deploy:
	python3 -m modal deploy backend.py --name "values"

.PHONY: phoenix
phoenix:
	python3 -m phoenix.server.main serve

.PHONY: lint
lint:
	python3 -m black . --check && python3 -m flake8 .

.PHONY: lint-fix
lint-fix:
	python3 -m black .

# Creates a brand new venv on each run
.PHONY: virtual-env
virtual-env:
	brew install uv
	uv venv --python=python3.11

.PHONY: python-setup
python-setup:
	brew install python
	brew install openssl
	brew update && brew upgrade
	pyenv install 3.11.3
	alias python=/usr/local/bin/python3

.PHONY: setup
setup:
	uv pip install -r requirements.txt

.PHONY: streamlit
streamlit:
	python3 -m streamlit run test_files/streamlit_ui.py

.PHONY: run-server
run-server:
	python gradio_backend.py

.PHONY: sitemap
sitemap:
	python scripts/generate_sitemap.py


================================================================================
FILE: ./ai_functions.py
================================================================================
import instructor
import os
from openai import OpenAI

from cerebras.cloud.sdk import Cerebras
from groq import Groq
from models import (
    ChoicesResponse,
    ChooseForMeResponse,
    MessageResponse,
    Prompt,
    PromptSelection,
    Choice,
    RouterResponse,
    UIResponse,
    NextStepsResponse,
    AdditionalActionResponse,
)

from prompts import (
    CHOOSE_OPTION_PROMPT,
    GENERATE_ALTERNATIVE_PROMPT,
    GENERATE_FEEDBACK_PROMPT,
    GENERATE_OUTCOMES_PROMPT,
    GENERATE_QUESTIONS_PROMPT,
    NORMAL_PROMPT,
    PROMPT_SELECTION_PROMPT,
    WEB_SEARCH_PROMPT,
    PRE_DECISION_NEXT_STEPS_PROMPT,
    POST_DECISION_NEXT_STEPS_PROMPT,
    SUGGEST_ADDITIONAL_ACTION_PROMPT,
)
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

CEREBRAS_MODEL = "llama-3.3-70b"
GROQ_MODEL = "llama-3.3-70b-specdec"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PPLX_API_KEY = os.environ.get("PPLX_API_KEY")


def groq_or_cerebras(messages, temperature, response_model=None):
    """
    Try Groq first, if that fails, try Cerebras.
    """
    try:
        if response_model is not None:
            client = instructor.from_groq(Groq(api_key=GROQ_API_KEY))
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                response_model=response_model,
                max_retries=2
            )    
        else:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_retries=2
            )
    except Exception as e:
        print(f"Error with Groq: {e}")
        if response_model is not None:
            client = instructor.from_cerebras(Cerebras(), mode=instructor.Mode.CEREBRAS_JSON)
            response = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                response_model=response_model,
                max_retries=2
            )
        else:
            client = Cerebras()
            response = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
                max_retries=2
            )
    return response


def cerebras_router(message_history):
    """
    Route to the appropriate ai prompt based on the input messages.

    Returns:
    {
        "prompt": Prompt,
        "response": Object or JSON dictionary
    }

    The response object will either be an object like ChoicesResponse, Choice, or a JSON dictionary with the following keys:
    {
        "text": str,
        "citations": list[str]
    }
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_router") as span:
        cleaned_messages = clean_up_messages(message_history)
        messages = [
            {"role": "user", "content": PROMPT_SELECTION_PROMPT.format(message_history=str(cleaned_messages))}
        ]
        router_response = groq_or_cerebras(messages, 0.2, PromptSelection)
        prompt = router_response.prompt
        last_message = cleaned_messages[-1]["content"]
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.INPUT_VALUE, str(last_message))
        # Map prompts to their handler functions
        handlers = {
            Prompt.WEB_SEARCH: lambda: perplexity_web_search(cleaned_messages),
            Prompt.NORMAL: lambda: cerebras_normal(cleaned_messages),
        }
        # Check if we have a handler for this prompt type
        if prompt not in handlers:
            return cerebras_normal(cleaned_messages)  # Default to normal chat if no specific handler

        # Execute the handler and return results
        full_response = handlers[prompt]()
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(full_response))
        return RouterResponse(prompt=prompt, response=full_response)


def perplexity_web_search(message_history):
    """
    Perform a web search for a given situation using Perplexity.
    """
    pplx_client = OpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")

    # Extract the last message
    last_message = message_history[-1]
    # Get all previous messages for context
    previous_messages = message_history[:-1]
    
    # Add the final message with the web search prompt
    previous_messages.append({
        "role": "user",
        "content": WEB_SEARCH_PROMPT.format(conversation=last_message.get("content", ""))
    })

    response = pplx_client.chat.completions.create(
        model="sonar",
        messages=previous_messages,
    )
    if hasattr(response, "citations"):
        return UIResponse(
            text=response.choices[0].message.content, citations=response.citations
        )
    else:
        return UIResponse(text=response.choices[0].message.content)


def cerebras_normal(message_history):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_normal") as span:
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, NORMAL_PROMPT)
        span.set_attribute(SpanAttributes.INPUT_VALUE, str(message_history))
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        messages = [
            {"role": "system", "content": NORMAL_PROMPT}
        ] + message_history
        response = groq_or_cerebras(messages, 0.2, MessageResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return UIResponse(text=response.text, suggested_messages=response.suggested_messages)


def cerebras_generate_outcomes(message_history):
    """
    Generate 3 outcomes for a given situation.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_generate_outcomes") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, GENERATE_OUTCOMES_PROMPT)
        messages = [
            {"role": "system", "content": GENERATE_OUTCOMES_PROMPT}
        ] + message_history
        response = groq_or_cerebras(messages, 0.2, ChoicesResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response


def cerebras_generate_alternative(situation, results):
    """
    Generate a new, unique alternative that wasn't previously considered. The alternative should be realistic and relevant to the situation.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_generate_alternative") as span:
        # Extract situation context from results if situation is empty
        if not situation and results and "title" in results:
            situation = results.get("title", "")
            
        prompt = GENERATE_ALTERNATIVE_PROMPT.format(
            situation=situation, results=results
        )
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE, GENERATE_ALTERNATIVE_PROMPT
        )
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
            str({"situation": situation, "results": results}),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, situation)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 0.2, Choice)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response



def cerebras_choose_option(situation, results, current_selected_index=None):
    """
    Choose an option based on the situation and results.
    If current_selected_index is provided, choose a different option.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_choose_option") as span:
        # Extract situation context from results if situation is empty
        if not situation and results and "title" in results:
            situation = results.get("title", "")

        total_options = len(results.get("choices", []))
        # Update prompt to include current selection if it exists
        if current_selected_index is not None:
            prompt = CHOOSE_OPTION_PROMPT.format(
                situation=situation,
                results=results,
                current_choice=current_selected_index,
                total_options=total_options,
            )
        else:
            prompt = CHOOSE_OPTION_PROMPT.format(
                situation=situation, results=results, current_choice="None", total_options=total_options
            )
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, CHOOSE_OPTION_PROMPT)
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
            str(
                {
                    "situation": situation,
                    "results": results,
                    "current_selected_index": current_selected_index,
                }
            ),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, situation)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 1, ChooseForMeResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response


def generate_question(situation, results, questions=None):
    """
    Generate a question for a given situation and results.
    """
    tracer = trace.get_tracer(__name__)
    if not situation and results and "title" in results:
        situation = results.get("title", "")

    with tracer.start_as_current_span("groq_generate_questions") as span:
        prompt = GENERATE_QUESTIONS_PROMPT.format(situation=situation, results=results, questions=questions)
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, GENERATE_QUESTIONS_PROMPT)
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 0.9)
        response_message = response.choices[0].message.content
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response_message))
    return response_message


def clean_up_messages(messages):
    # Keep track of total characters
    total_chars = sum(len(msg["content"]) for msg in messages)

    # Target is ~30K chars (roughly 8K tokens)
    while total_chars > 30000:
        if len(messages) <= 2:
            # If only system prompt and last message remain, truncate last message
            last_msg = messages[-1]["content"]
            # Keep first 1000 chars and last 1000 chars to maintain context
            if len(last_msg) > 2000:
                messages[-1]["content"] = last_msg[:1000] + "..." + last_msg[-1000:]
        else:
            # Remove oldest non-system messages in pairs
            for i in range(1, len(messages) - 1):
                if messages[i]["role"] == "user":
                    # Remove this user message and next assistant message if it exists
                    messages.pop(i)
                    if i < len(messages) and messages[i]["role"] == "assistant":
                        messages.pop(i)
                    break

        total_chars = sum(len(msg["content"]) for msg in messages)

    # Ensure messages only have role and content attributes that are STRINGS
    cleaned_messages = []
    for msg in messages:
        cleaned_messages.append({
            "role": msg["role"],
            "content": str(msg["content"])
        })
    messages = cleaned_messages
    return messages


def cerebras_generate_next_steps(situation, choice_name=None, results=None, choice_index=None):
    """
    Generate actionable next steps for a situation, either before or after a decision.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_generate_next_steps") as span:
        # Determine if it's post-decision (choice provided) or pre-decision (no choice)
        if choice_name or choice_index is not None:
            # Post-decision scenario
            if choice_index is not None and results and "choices" in results:
                try:
                    selected_choice = results["choices"][choice_index]
                except (IndexError, TypeError):
                    selected_choice = {"name": choice_name} if choice_name else {}
            else:
                selected_choice = {"name": choice_name} if choice_name else {}
            
            prompt = POST_DECISION_NEXT_STEPS_PROMPT.format(
                situation=situation,
                results=results if results else {},
                choice_name=choice_name if choice_name else "",
                selected_choice=selected_choice
            )
            prompt_template = POST_DECISION_NEXT_STEPS_PROMPT
        else:
            # Pre-decision scenario
            prompt = PRE_DECISION_NEXT_STEPS_PROMPT.format(
                situation=situation,
                results=results if results else {}
            )
            prompt_template = PRE_DECISION_NEXT_STEPS_PROMPT
        
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE, 
            prompt_template
        )
        span.set_attribute(
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
            str({
                "situation": situation, 
                "choice_name": choice_name,
                "results": results,
                "choice_index": choice_index
            }),
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 0.2, NextStepsResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response


def cerebras_suggest_additional_action(situation, existing_next_steps, results=None):
    """
    Generate a single additional action that complements existing actions.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("groq_suggest_additional_action") as span:
        prompt = SUGGEST_ADDITIONAL_ACTION_PROMPT.format(
            situation=situation,
            existing_next_steps=existing_next_steps,
            results=results if results else {}
        )
        
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, SUGGEST_ADDITIONAL_ACTION_PROMPT)
        span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, str({
            "situation": situation,
            "existing_next_steps": existing_next_steps,
            "results": results
        }))
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)
        messages = [{"role": "user", "content": prompt}]
        response = groq_or_cerebras(messages, 0.2, AdditionalActionResponse)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
    return response

================================================================================
FILE: ./dump_codebase.sh
================================================================================
#!/usr/bin/env bash

# Script to dump relevant code files from the project into a single text file
# Excludes node_modules, __pycache__, .git directories, and build artifacts

OUTPUT_FILE="codebase_dump.md"

# Clear or create the output file
echo "Creating code dump in $OUTPUT_FILE"
> "$OUTPUT_FILE"

# Function to generate a directory tree structure of relevant files
generate_directory_tree() {
    echo "Generating directory tree..."
    
    echo -e "================================================================================\n" >> "$OUTPUT_FILE"
    echo -e "PROJECT DIRECTORY STRUCTURE\n" >> "$OUTPUT_FILE"
    echo -e "================================================================================\n" >> "$OUTPUT_FILE"
    
    # Create temporary files to collect all relevant files and processed directories
    TEMP_FILE=$(mktemp)
    PROCESSED_DIRS=$(mktemp)
    
    # Add root directory to processed dirs
    echo "." > "$PROCESSED_DIRS"
    
    # Find all relevant files using the same patterns as for file content
    # Python files
    find . -type f -name "*.py" \
        -not -path "*/\.*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/venv/*" \
        -not -path "*/env/*" \
        -not -path "*/node_modules/*" \
        -not -path "*/ios/*" \
        -not -path "*/android/*" \
        >> "$TEMP_FILE"
    
    # JavaScript/TypeScript files
    find . -type f \( -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" \) \
        -not -path "*/\.*" \
        -not -path "*/node_modules/*" \
        -not -path "*/dist/*" \
        -not -path "*/build/*" \
        -not -path "*/ios/*" \
        -not -path "*/android/*" \
        >> "$TEMP_FILE"
    
    # HTML/CSS files
    find . -type f \( -name "*.html" -o -name "*.css" -o -name "*.scss" -o -name "*.sass" \) \
        -not -path "*/\.*" \
        -not -path "*/node_modules/*" \
        -not -path "*/dist/*" \
        -not -path "*/build/*" \
        -not -path "*/ios/*" \
        -not -path "*/android/*" \
        >> "$TEMP_FILE"
    
    # Configuration files
    find . -type f \( -name "*.json" -o -name "*.yml" -o -name "*.yaml" -o -name "*.xml" -o -name "*.toml" -o -name "*.ini" -o -name "*.conf" -o -name "*.config" \) \
        -not -path "*/\.*" \
        -not -path "*/node_modules/*" \
        -not -path "*/dist/*" \
        -not -path "*/build/*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/ios/*" \
        -not -path "*/android/*" \
        >> "$TEMP_FILE"
    
    # Important files with no extension at root level
    find . -maxdepth 1 -type f -not -path "*/\.*" \
        -not -name "$OUTPUT_FILE" \
        -not -path "*/ios/*" \
        -not -path "*/android/*" \
        >> "$TEMP_FILE"
    
    # Sort the files and process them to create a hierarchical tree
    # Add root directory first
    echo -e "+-- ." >> "$OUTPUT_FILE"
    
    sort -u "$TEMP_FILE" | while read file; do
        # Remove ./ prefix if it exists
        file="${file#./}"
        
        # Get just the filename (last part)
        filename=$(basename "$file")
        
        # Get the directory path
        dirpath=$(dirname "$file")
        
        # Process directory hierarchy
        if [ "$dirpath" != "." ]; then
            # Split the dirpath into components and build paths incrementally
            parts=$(echo "$dirpath" | tr '/' ' ')
            current_path=""
            current_depth=0
            
            for part in $parts; do
                # Build the current path
                if [ -z "$current_path" ]; then
                    current_path="$part"
                else
                    current_path="$current_path/$part"
                fi
                
                # Check if this directory has been processed already
                if ! grep -q "^$current_path$" "$PROCESSED_DIRS"; then
                    # Calculate indentation based on depth
                    indent=""
                    for ((i=0; i<current_depth; i++)); do
                        indent="${indent}|   "
                    done
                    
                    # Add directory to tree with proper indentation
                    echo -e "${indent}+-- ${part}/" >> "$OUTPUT_FILE"
                    
                    # Mark as processed by adding to our temporary file
                    echo "$current_path" >> "$PROCESSED_DIRS"
                fi
                
                current_depth=$((current_depth + 1))
            done
        fi
        
        # Calculate proper indentation for the file
        file_indent=""
        dir_depth=$(echo "$dirpath" | tr -cd '/' | wc -c)
        if [ "$dirpath" != "." ]; then
            dir_depth=$((dir_depth + 1))
        else
            dir_depth=0
        fi
        
        for ((i=0; i<dir_depth; i++)); do
            file_indent="${file_indent}|   "
        done
        
        # Add the file with indentation
        echo -e "${file_indent}+-- ${filename}" >> "$OUTPUT_FILE"
    done
    
    # Add a final separator
    echo -e "\n================================================================================\n" >> "$OUTPUT_FILE"
    echo -e "FILE CONTENTS\n" >> "$OUTPUT_FILE"
    echo -e "================================================================================\n" >> "$OUTPUT_FILE"
    
    # Clean up temporary files
    rm "$TEMP_FILE" "$PROCESSED_DIRS"
}

# Function to add a file to the dump
add_file_to_dump() {
    local file="$1"
    
    # Get file size in KB
    local size_kb=$(du -k "$file" | cut -f1)
    
    # Skip files larger than 1MB (1024KB) as they're likely binary or too large
    if [ "$size_kb" -gt 1024 ]; then
        echo "Skipping large file: $file ($size_kb KB)"
        return
    fi
    
    # Check if file is likely binary
    if file "$file" | grep -q "binary"; then
        echo "Skipping binary file: $file"
        return
    fi
    
    echo "Adding file: $file"
    
    # Add file header
    echo -e "\n\n================================================================================" >> "$OUTPUT_FILE"
    echo "FILE: $file" >> "$OUTPUT_FILE"
    echo "================================================================================" >> "$OUTPUT_FILE"
    
    # Add file content
    cat "$file" >> "$OUTPUT_FILE"
}

# Generate directory tree structure at the beginning
generate_directory_tree

# Find and process Python files
find . -type f -name "*.py" \
    -not -path "*/\.*" \
    -not -path "*/__pycache__/*" \
    -not -path "*/venv/*" \
    -not -path "*/env/*" \
    -not -path "*/node_modules/*" \
    -not -path "*/ios/*" \
    -not -path "*/android/*" \
    | while read file; do
        add_file_to_dump "$file"
    done

# Find and process JavaScript/TypeScript files
find . -type f \( -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" \) \
    -not -path "*/\.*" \
    -not -path "*/node_modules/*" \
    -not -path "*/dist/*" \
    -not -path "*/build/*" \
    -not -path "*/ios/*" \
    -not -path "*/android/*" \
    | while read file; do
        add_file_to_dump "$file"
    done

# Find and process HTML/CSS files
find . -type f \( -name "*.html" -o -name "*.css" -o -name "*.scss" -o -name "*.sass" \) \
    -not -path "*/\.*" \
    -not -path "*/node_modules/*" \
    -not -path "*/dist/*" \
    -not -path "*/build/*" \
    -not -path "*/ios/*" \
    -not -path "*/android/*" \
    | while read file; do
        add_file_to_dump "$file"
    done

# Add configuration files
find . -type f \( -name "*.json" -o -name "*.yml" -o -name "*.yaml" -o -name "*.xml" -o -name "*.toml" -o -name "*.ini" -o -name "*.conf" -o -name "*.config" \) \
    -not -path "*/\.*" \
    -not -path "*/node_modules/*" \
    -not -path "*/dist/*" \
    -not -path "*/build/*" \
    -not -path "*/__pycache__/*" \
    -not -path "*/ios/*" \
    -not -path "*/android/*" \
    | while read file; do
        add_file_to_dump "$file"
    done

# Add important files with no extension at root level
find . -maxdepth 1 -type f -not -path "*/\.*" \
    -not -name "$OUTPUT_FILE" \
    -not -path "*/ios/*" \
    -not -path "*/android/*" \
    | while read file; do
        add_file_to_dump "$file"
    done

echo "Code dump completed! Output saved to $OUTPUT_FILE"
echo "Total size: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo "Copying to clipboard..."
cat "$OUTPUT_FILE" | pbcopy
echo "Content copied to clipboard!"

================================================================================
FILE: ./database.py
================================================================================
from sqlalchemy import Column, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DecisionRecord(Base):
    __tablename__ = 'decision_records'
    id = Column(String, primary_key=True)  # Unique ID for the decision
    message_history = Column(Text)         # Conversation thread as JSON string

================================================================================
FILE: ./__init__.py
================================================================================


================================================================================
FILE: ./prompts.py
================================================================================
PREFIX_PROMPT = """You are a helpful AI assistant focused on helping users make decisions, using principles from Annie Duke's Thinking in Bets, Maxims for Thinking Analytically, Decisive by the Heath Brothers, Psychology of Human Misjudgment by Charlie Munger, and other sources. Do not mention these authors in your response."""

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

PROMPT_SELECTION_PROMPT = f"""{PREFIX_PROMPT}

Based on the conversation, you will choose the best prompt to answer my question best. The LAST user message is most important. 

Return the prompt title and nothing else.

Here are your prompt choices:
- normal: for general conversation and questions about decision making, use this.
- web_search: for web research for latest events, such as weather, news, facts, statistics, and more.

[BEGIN MESSAGE HISTORY]
{{message_history}}
[END MESSAGE HISTORY]
"""

GENERATE_OUTCOMES_PROMPT = f"""{PREFIX_PROMPT}

Generate 3 outcomes for a given situation, with the given objectives.

### Context
Your task is to help me widen my options by suggesting options. The outcomes you generate should help me see the range of possibilities and choose the best path forward. Sometimes the best path forward is doing nothing. 

Often I will not be good at articulating my situation. Make assumptions where you can. Do not mention authors or sources in your response. Consider how I might overlook conspicuously crucial information and make a stupid decision, such as new environments, social proof, desire to look good, authority bias, information overload, stress, fatigue, urgency to do something, paradox of choice, desire for control, cognitive dissonance, confirmation bias, overconfidence, survivorship bias, resulting, recency bias, anchoring, and other biases.
"""


GENERATE_ROLEPLAY_SCENARIO_PROMPT = """I have made the following choice for a given scenario.
### BEGIN SCENARIO
{scenario}
### END SCENARIO

You are the narrator of this scenario. You will observe different facts, feelings, and observations about the scenario. I will continually make different choices, and you will observe the consequences of those choices. Add variance to the scenario to make it more engaging. This is the {best_or_worst} case scenario.
"""

GENERATE_FEEDBACK_PROMPT = """You are a decision making expert, using principles from Annie Duke's Thinking in Bets, Maxims for Thinking Analytically, Decisive by the Heath Brothers, Psychology of Human Misjudgment by Charlie Munger, and other sources. I have a question or message for you, and I want you to help me with a given situation, objectives, proposed options, and message. Use at most 50 words.

### Context
The situation is the decision that I need help with and what's important to me. The proposed options are the possibilities that I am considering. Your job is to help me see the range of possibilities and choose the best path forward. Sometimes the best path forward is doing nothing. Often I will not be good at articulating the situation or objectives. Make assumptions where you can. Do not mention authors or sources in your response. Consider how I might overlook conspicuously crucial information and make a stupid decision, such as new environments, social proof, desire to look good, authority bias, information overload, stress, fatigue, urgency to do something, paradox of choice, desire for control, cognitive dissonance, confirmation bias, overconfidence, survivorship bias, resulting, recency bias, anchoring, and other biases.

### Response format
Use at most 50 words. Respond in markdown format. For every line break, use two newlines. Use line breaks often. Respond with a warm and encouraging tone.

### Situation
{situation}

### Proposed options
{results}

### Message
{message}
"""


GENERATE_ALTERNATIVE_PROMPT = f"""{PREFIX_PROMPT}

I will provide you with my situation and the current proposed options. Your task is to generate a new, unique alternative decision. The alternative should be realistic and relevant to my situation. The alternative must be different than any of the current proposed options.

### Situation
{{situation}}

### Current proposed options
{{results}}
"""

CHOOSE_OPTION_PROMPT = f"""{PREFIX_PROMPT}

I will provide you with my situation and the potential options. Your task is to choose an option for me. Select an option completely at random, and provide a plausible explanation for why it could be a good decision. If a current selected option is provided, choose a different option.

### Situation
{{situation}}

### The total number of options
{{total_options}}

### Potential options
{{results}}

### Current selected option (0-indexed)
{{current_choice}}
"""

WEB_SEARCH_PROMPT = f"""

I am missing some information that I need to search the web for. I will provide you with our current conversation, and you should respond to my current inquiry with clarity, brevity, politeness, and helpfulness with fewer words.

[BEGIN CONVERSATION]
{{conversation}}
[END CONVERSATION]
"""

NORMAL_PROMPT = f"""{PREFIX_PROMPT}

When discussing choices, analyze trade-offs, consider uncertainties, and help users think through their options. Sometimes the best path forward is doing nothing. Often I will not be good at articulating the situation or objectives. Make assumptions where you can. Consider how I might overlook conspicuously crucial information and make a stupid decision, such as new environments, social proof, desire to look good, authority bias, information overload, stress, fatigue, urgency to do something, paradox of choice, desire for control, cognitive dissonance, confirmation bias, overconfidence, survivorship bias, resulting, recency bias, anchoring, and other biases.

Respond with clarity, brevity, politeness, and helpfulness using fewer words and a warm and encouraging tone. Ask at MOST 1 question if you are going to ask me a question. Do not use the word "and" in your response.

You can keep these questions in mind when helping me with my decision making:
{QUESTIONS}
"""


GENERATE_QUESTIONS_PROMPT = f"""{PREFIX_PROMPT}

Your task is to generate the most pertinent question to ask me about my given situation, which is the greatest uncertainty I am facing. 

You should respond with the best question that is not the current question being asked. Your response should be one sentence with a max of 20 words.

I will also provide you with a list of questions as inspiration which are really useful questions for decision making, and why they are useful.

### Situation
{{situation}}

### Current questions
{{questions}}

### Proposed options
{{results}}

### Questions as inspiration
{QUESTIONS}
"""

PRE_DECISION_NEXT_STEPS_PROMPT = f"""{PREFIX_PROMPT}

Your task is to suggest 1 or 2 specific actions I can take to make progress in my situation or to gather more information that will help me make a decision.

### Context
I am considering my options and need guidance on what to do next to better understand my situation or the potential choices.

### Guidelines for suggestions:
1. Be specific and actionable - start with a verb
2. Focus on information gathering, reducing uncertainty, or preparatory steps
3. Keep each suggestion concise (5-15 words)
4. Consider what information or actions would be most helpful in making the decision
5. Think about potential obstacles or uncertainties and how to address them

### Situation
{{situation}}

### Current analysis (if available)
{{results}}
"""

POST_DECISION_NEXT_STEPS_PROMPT = f"""{PREFIX_PROMPT}

Your task is to generate 1 or 2 specific, actionable next steps for me based on the decision I've made. These steps should help me implement my decision effectively.

### Context
I have decided on a specific option for my situation. Now I need concrete actions to take to move forward with this decision.

### Guidelines for next steps:
1. Be specific and concrete - avoid vague suggestions
2. Make each step actionable - start with a verb
3. Keep steps concise (5-15 words each)
4. Focus on immediate actions I can take within the next few days/weeks

### Situation
{{situation}}

### Decision Context
{{results}}

### My chosen option
{{choice_name}}

### Selected choice details
{{selected_choice}}
"""

# Add new prompt for suggesting an additional action
SUGGEST_ADDITIONAL_ACTION_PROMPT = f"""{PREFIX_PROMPT}

Your task is to suggest one additional specific action I can take, based on my current situation and the actions I already have planned.

### Context
I have already planned some actions and need one more suggestion to complement or add to them.

### Guidelines for the suggestion:
1. Be specific and actionable - start with a verb
2. Ensure it is different from the existing actions
3. Keep it concise (5-15 words)
4. Focus on what would be most helpful next

### Situation
{{situation}}

### Existing actions
{{existing_next_steps}}

### Current analysis (if available)
{{results}}
"""

================================================================================
FILE: ./decision_records.db
================================================================================
SQLite format 3   @                                                                     .j–¯ % %∑                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Å--ÅQtabledecision_recordsdecision_recordsCREATE TABLE decision_records (
	id VARCHAR NOT NULL, 
	message_history TEXT, 
	PRIMARY KEY (id)
)?S- indexsqlite_autoindex_decision_records_1decision_records          ˚    ˚                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   àUè]113b9671-735a-408a-a745-d7afaa0ac15d[{"role": "user", "content": "hello", "isEditing": false, "editedContent": "hello"}, {"role": "assistant", "content": "<p>Hello! What&#39;s on your mind?</p>\n", "suggested_messages": ["I'm considering a big decision", "I need help with something", "I just want to chat"]}, {"role": "user", "content": "I just want to chat", "isEditing": false, "editedContent": "I just want to chat"}, {"role": "assistant", "content": "<p>That sounds lovely, what&#39;s been the highlight of your day so far?</p>\n", "suggested_messages": ["It's been good", "Not much", "I had a great conversation"]}, {"role": "user", "content": "I had a great conversation", "isEditing": false, "editedContent": "I had a great conversation"}, {"role": "assistant", "content": "<p>That&#39;s wonderful, conversations can be so uplifting. What made this conversation stand out to you?</p>\n", "suggested_messages": ["It was with someone I admire", "We talked about something I'm passionate about", "It was a surprise conversation"]}]Ç9UÑ/9916918a-8e44-4476-b45b-ac87aaa2ed32[{"role": "user", "content": "hello", "isEditing": false, "editedContent": "hello"}, {"role": "assistant", "content": "<p>Hello! What&#39;s on your mind?</p>\n", "suggested_messages": ["I'm considering a big decision", "I need help with something", "I just want to cha   
   Ü ØÿÜ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        (Uaa72b607-48e7-4055-b87c-511c12670d4f(U113b9671-735a-408a-a745-d7afaa0ac15d'U	9916918a-8e44-4476-b45b-ac87aaa2ed32   
± ƒ
±                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     àUè]113b9671-735a-408a-a745-d7afaa0ac15d[{"role": "user", "content": "hello", "isEditing": false, "editedContent": "hello"}, {"role": "assistant", "content": "<p>Hello! What&#39;s on your mind?</p>\n", "suggested_messages": ["I'm considering a big decision", "I need help with something", "I just want to chat"]}, {"role": "user", "content": "I just want to chat", "isEditing": false, "editedContent": "I just want to chat"}, {"role": "assistant", "content": "<p>That sounds lovely, what&#39;s been the highlight of your day so far?</p>\n", "suggested_messages": ["It's been good", "Not much", "I had a great conversation"]}, {"role": "user", "content": "I had a great conversation", "isEditing": false, "editedContent": "I had a great conversation"}, {"role": "assistant", "content": "<p>That&#39;s wonderful, conversations can be so uplifting. What made this conversation stand out to you?</p>\n", "suggested_messages": ["It was with someone I admire", "We talked about something I'm passionate about", "It was a surprise conversation"]}]Ç9UÑ/9916918a-8e44-4476-b45b-ac87aaa2ed32[{"role": "user", "content": "hello", "isEditing": false, "editedContent": "hello"}, {"role": "assistant", "content": "<p>Hello! What&#39;s on your mind?</p>\n", "suggested_messages": ["I'm considering a big decision", "I need help with something", "I just want to chat"]}]    „  „                                                                                                                                                                                                                         ûUªqaa72b607-48e7-4055-b87c-511c12670d4f[{"role": "user", "content": "hello", "isEditing": false, "editedContent": "hello"}, {"role": "assistant", "content": "<p>Hello! What&#39;s on your mind?</p>\n", "suggested_messages": ["I'm considering a big decision", "I need help with something", "I just want to chat"]}, {"role": "user", "content": "I just want to chat", "isEditing": false, "editedContent": "I just want to chat"}, {"role": "assistant", "content": "<p>That sounds lovely, what&#39;s been the highlight of your day so far?</p>\n", "suggested_messages": ["It's been good", "Not much", "I had a great conversation"]}, {"role": "user", "content": "I had a great conversation", "isEditing": false, "editedContent": "I had a great conversation"}, {"role": "assistant", "content": "<p>That&#39;s wonderful, conversations can be so uplifting. What made this conversation stand out to you?</p>\n", "suggested_messages": ["It was with someone I admire", "We talked about something I'm passionate about", "It was a surprise conversation"]}, {"role": "assistant", "content": {"title": "Should I continue this conversation?", "choices": [{"name": "Continue", "assumptions": ["I am enjoying the conversation", "The other person is willing to continue"], "best_case_scenario": "Deeper connection", "worst_case_scenario": "Awkward silence"}, {"name": "Change subject", "assumptions": ["I am getting bored", "The other person is open to a new topic"], "best_case_scenario": "New interesting topic", "worst_case_scenario": "Uncomfortable transition"}, {"name": "End conversation", "assumptions": ["I am tired", "The other person is busy"], "best_case_scenario": "Rest and relaxation", "worst_case_scenario": "Abrupt ending", "explanation": "You might need rest and relaxation."}, {"name": "Take a break", "assumptions": ["I need time to reflect", "The other person is understanding"], "best_case_scenario": "Refreshed perspective", "worst_case_scenario": "Misinterpreted pause"}, {"name": "Set a timer", "assumptions": ["I can stay focused for a short time", "The other person is willing to continue for a bit"], "best_case_scenario": "Productive exchange", "worst_case_scenario": "Rushed conversation"}], "uncertainties": ["Will the conversation remain engaging?", "Will I learn something new?"], "next_steps": ["Ask open-ended questions", "Listen actively"]}, "type": "choices", "choices": {"title": "Should I continue this conversation?", "choices": [{"name": "Continue", "assumptions": ["I am enjoying the conversation", "The other person is willing to continue"], "best_case_scenario": "Deeper connection", "worst_case_scenario": "Awkward silence"}, {"name": "Change subject", "assumptions": ["I am getting bored", "The other person is open to a new topic"], "best_case_scenario": "New interesting topic", "worst_case_scenario": "Uncomfortable transition"}, {"name": "End conversation", "assumptions": ["I am tired", "The other person is busy"], "best_case_scenario": "Rest and relaxation", "worst_case_scenario": "Abrupt ending", "explanation": "You might need rest and relaxation."}, {"name": "Take a break", "assumptions": ["I need time to reflect", "The other person is understanding"], "best_case_scenario": "Refreshed perspective", "worst_case_scenario": "Misinterpreted pause"}, {"name": "Take a break", "assumptions": ["I need time to reflect", "The other person is understanding"], "best_case_scenario": "Refreshed perspective", "worst_case_scenario": "Misinterpreted pause"}, {"name": "Set a timer", "assumptions": ["I can stay focused for a short time", "The other person is willing to continue for a bit"], "best_case_scenario": "Productive exchange", "worst_case_scenario": "Rushed conversation"}], "uncertainties": ["Will the conversation remain engaging?", "Will I learn something new?"], "next_steps": ["Ask open-ended questions", "Listen actively"]}}]

================================================================================
FILE: ./instrumentor.py
================================================================================
from phoenix.otel import register
from openinference.instrumentation.groq import GroqInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.instructor import InstructorInstrumentor
import os


def set_easy_tracing_instrumentation():
    """Set tracing instrumentation for Phoenix and Arize"""
    # tracer_provider = register(batch=True)
    tracer_provider = register(
        batch=True, endpoint="http://phoenix-kd03.onrender.com/v1/traces"
    )
    # InstructorInstrumentor().instrument(tracer_provider=tracer_provider)
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    print("Tracing instrumentation set")


def set_hosted_phoenix_instrumentation():
    """Set tracing instrumentation for Phoenix and Arize"""
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
    os.environ["PHOENIX_CLIENT_HEADERS"] = (
        f"api_key={os.environ.get('PHOENIX_API_KEY')}"
    )
    # Setup OTEL tracing for hosted Phoenix. The register function will automatically detect the endpoint and headers from your environment variables.
    tracer_provider = register(batch=True)

    # Turn on instrumentation for OpenAI
    # InstructorInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(
        tracer_provider=tracer_provider, skip_dep_check=True
    )
    print("Tracing instrumentation set")
