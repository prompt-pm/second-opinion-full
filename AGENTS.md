# Values App Contributor Guide

## Primary Purpose
- Help overthinkers make choices
- Users provide a prompt which can include attachments, text, or voice (coming soon)
- The AI provides useful feedback and chooses which module is most appropriate to help the user with their decision making
- Users can share their decision chat with others with the share button 

## Codebase Layout
- `backend.py` – FastAPI application with Modal integration for deployment
- `ai_functions.py` – Helper functions calling Groq, Cerebras, and other LLM APIs
- `ai_agents.py` – Orchestrates multiple LLM agents for conversation and decision help
- `models.py` – Pydantic models for messages, choices, and other outputs
- `database.py` – SQLAlchemy setup with a simple `DecisionRecord` model
- `prompts.py` – Prompt templates used by the agents
- `frontend/` – Alpine.js & Tailwind CSS app served by Vite
- `scripts/` – Utility scripts such as `generate_sitemap.py`
- `assets/` – Stores persistent data like the SQLite DB
- `documents/` – Product specs and other docs

## Key Features
- Multiple FastAPI endpoints for chat, outcome generation, next steps, and more
- Agents combine conversation, priorities gathering, recommendations, and objections handling
- Supports web search and LLM providers (Groq, Cerebras) via helper functions
- Conversation history stored in a SQLite database mounted locally or via Modal volume
- Minimalist front end built with HTMX/Alpine.js and styled with Tailwind CSS

## Development Tips
- Run the API locally with `make dev` (uvicorn reload on port 7000)

## Linting & Formatting
- Check code style and formatting: `make lint`
- Auto-format code with Black: `make lint-fix`

## Testing
- Run the full test suite: `pytest -q`

## Pull Request Instructions
- Use branch names that reflect your feature or fix.
- PR title format: `[oksayless] <Brief description>`
- Before submitting, ensure:
  - All tests pass: `pytest -q`
  - Code is formatted: `make lint`. Fix issues by running `make lint-fix`
  - New or updated tests are included for your changes.
  - No lint or security warnings remain. 
