# Second Opinion

An AI assistant that helps you think through decisions.

## What it does

1. You describe your situation
2. AI asks clarifying questions
3. You rank your priorities
4. AI helps you think through the decision

## Project Structure

```
├── backend/          # Python FastAPI backend
│   ├── main.py       # API endpoints
│   ├── models.py     # Pydantic models
│   └── prompts.py    # LLM prompts
├── frontend/         # Browser frontend
│   ├── index.html    # Main page
│   └── app.js        # Alpine.js app
├── tests/            # Test files
├── config/           # Tool configurations
└── requirements.txt  # Python dependencies
```

## Setup

### Backend

```bash
# Create virtual environment and install dependencies
uv venv && uv pip install -r requirements.txt

# Activate virtual environment
source .venv/bin/activate
```

### Frontend

```bash
# Install Node.js dependencies
bun install
```

### Environment

Add your API key to `.env`:
```
OPENROUTER_API_KEY=your-key-here
```

## Running the App

```bash
# Start the server (from project root)
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 3000

# Or using Python directly
python -m backend.main
```

Then open http://localhost:3000

## Common Commands

### Testing

```bash
# Run all backend tests
pytest tests/ -v

# Run all frontend tests
bun run test

# Run frontend tests in watch mode
bun run test:watch

# Run a specific test file
pytest tests/test_backend.py -v
bun run test -- tests/test_frontend.js
```

### Linting

```bash
# Check Python code
ruff check . --config config/pyproject.toml

# Fix Python code automatically
ruff check . --fix --config config/pyproject.toml

# Format Python code
ruff format . --config config/pyproject.toml

# Check JavaScript code
bun run lint

# Fix JavaScript code automatically
bun run lint:fix
```

### Development

```bash
# Run backend with auto-reload
uvicorn backend.main:app --reload --host 0.0.0.0 --port 3000

# Check for type errors (Python)
ruff check . --select=F --config config/pyproject.toml
```

## CI/CD

GitHub Actions runs on every PR:
- **Backend**: ruff linting + pytest
- **Frontend**: ESLint + Vitest

See `.github/workflows/ci.yml` for details.
