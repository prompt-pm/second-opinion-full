from fastapi import Request, FastAPI, Depends, HTTPException
import modal
import os
import json
from modal import App, Image, asgi_app
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
from instrumentor import set_hosted_phoenix_instrumentation
from phoenix.trace import using_project
import sentry_sdk
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database import Base, DecisionRecord
from ai_agents import (
    ChoicesResponse,
    ObjectionsOutput,
    ask,
    MessageResponse,
    PrioritiesOutput,
    Choice,
)

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
        # set_easy_tracing_instrumentation()
        set_hosted_phoenix_instrumentation()
        initiate_sentry()
        tracing_initialized = True
        print("Tracing instrumentation initialized")


# image = (
#     Image.debian_slim()
#     .pip_install("uv")
#     .run_commands("uv pip install --system --compile-bytecode ./requirements.txt")
# )

# Define the volume for persistent database storage
volume = modal.Volume.from_name("decision-records-volume", create_if_missing=True)

image = (
    Image.debian_slim()
    .pip_install_from_requirements("./requirements.txt")
    .add_local_python_source(
        "ai_functions", "instrumentor", "models", "prompts", "database", "ai_agents"
    )
)
image.add_local_dir("./assets", remote_path="/assets")
# mount = Mount.from_local_dir("./assets", remote_path="/assets")
app = App(image=image)
web_app = FastAPI()

# Database setup
# Use different paths based on environment (Modal or local)
if "MODAL_TASK_ID" in os.environ:
    # Use the volume path when running on Modal
    DB_PATH = "/data/decision_records.db"
else:
    # Use local path for development
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "assets", "data")
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    DB_PATH = os.path.join(data_dir, "decision_records.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"
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
    "http://okaysayless.com",
    "https://okaysayless.com",
    "http://overthinking.app",
    "https://overthinking.app",
    "http://overthinking.help",
    "https://overthinking.help",
]

# Allow Render preview URLs such as
# https://oksayless-com-pr-6.onrender.com or any variant containing
# "oksayless".
preview_origin_regex = r"^https://.*oksayless.*\.onrender\.com$"

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=preview_origin_regex,
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
    advisor = data.get("advisor", "thought_partner")
    custom_prefix = data.get("custom_prefix")
    with using_project("say-less"):
        result = await ask(messages, advisor=advisor, custom_prefix=custom_prefix)
        final_output = result.final_output
        response_type = "message"
        agent = "Conversation Agent"
        # print(result)
        if isinstance(final_output, MessageResponse):
            response_type = "message"
            agent = "Conversation Agent"
        elif isinstance(final_output, PrioritiesOutput):
            response_type = "priorities"
            agent = "Priorities Agent"
        elif isinstance(final_output, ChoicesResponse):
            response_type = "choices"
            agent = "Recommendation Agent"
        elif isinstance(final_output, ObjectionsOutput):
            response_type = "objections"
            agent = "Objections Agent"
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected agent output type: {type(final_output)}",
            )
        return JSONResponse(
            content={
                "response_type": response_type,
                "agent": agent,
                "response": final_output.model_dump(),
            }
        )


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
    # mounts=[mount],
    volumes={"/data": volume},  # Mount the volume to /data
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
        response = cerebras_generate_next_steps(
            situation, choice_name, results, choice_index
        )

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
        response = cerebras_suggest_additional_action(
            situation, existing_next_steps, results
        )

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
        return JSONResponse(
            content={"error": "Message history is empty"}, status_code=400
        )

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
