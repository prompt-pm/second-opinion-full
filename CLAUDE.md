# Second Opinion

A decision-making assistant that helps users think through choices.

## Product Overview

Second Opinion is an AI-powered decision-making tool that guides users through structured thinking when facing difficult choices. The application uses a conversational interface to:

1. Understand the user's decision context through clarifying questions
2. Extract and help prioritize what matters most to the user
3. Generate potential options with best and worst case scenario analysis

The goal is to help users make more informed decisions by breaking down complex choices into manageable components.

## Features
- **Chat**: Conversational AI that asks clarifying questions
- **Priorities**: Extracts and lets users rank what matters most
- **Choices**: Generates options with best/worst case scenarios

## Folder Structure

```
.
├── backend/            # FastAPI backend
│   ├── main.py         # API endpoints and server setup
│   ├── models.py       # Pydantic request/response models
│   └── prompts.py      # LLM prompt templates
├── frontend/           # Frontend application
│   ├── index.html      # Main HTML page
│   └── app.js          # Alpine.js application logic
├── tests/              # Test suites
│   ├── test_backend.py # Python backend tests (pytest)
│   └── test_frontend.js # JavaScript frontend tests (vitest)
├── specs/              # Feature specifications and requirements
├── config/             # Configuration files
│   ├── eslint.config.js # ESLint configuration
│   ├── pyproject.toml  # Python project/ruff configuration
│   └── vitest.config.js # Vitest test configuration
├── .github/            # GitHub Actions workflows
├── Makefile            # Common development commands
├── package.json        # Frontend dependencies (bun)
├── requirements.txt    # Python dependencies (uv)
└── conductor.json      # Conductor configuration
```

## How to Deploy Locally

1. Install dependencies:
```bash
make install
```

2. Set your OpenRouter API key (or use .env file):
```bash
export OPENROUTER_API_KEY=your_key_here
```

3. Run the server:
```bash
make run
# or with hot reload:
make dev
```

4. Open http://localhost:7000 in your browser

## Development Commands

```bash
make help           # Show all available commands
make install        # Install all dependencies
make run            # Run the backend server
make dev            # Run with hot reload
make test           # Run all tests
make test-backend   # Run Python tests only
make test-frontend  # Run JavaScript tests only
make lint           # Run all linters
make lint-fix       # Fix linting issues
make format         # Format Python code
make clean          # Remove generated files
```

## API Endpoints

- `GET /` - Serves the frontend (frontend/index.html)
- `POST /api/chat` - Conversational AI endpoint
- `POST /api/priorities` - Extract priorities from conversation
- `POST /api/choices` - Generate decision options with best/worst cases
