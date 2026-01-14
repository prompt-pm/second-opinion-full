# Second Opinion - Starter Repo

A simple decision-making assistant that helps users think through choices.

## Product Overview

For detailed product strategy, principles, and decision-making theory, see [specs/decision-making.md](specs/decision-making.md).

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
├── .context/           # Conductor workspace collaboration (gitignored)
├── .github/            # GitHub Actions workflows
├── .gitignore          # Git ignore configuration
├── backend.py          # FastAPI server with API endpoints
├── conductor.json      # Conductor configuration
├── index.html          # Single-page frontend with Alpine.js
├── README.md           # Project readme
└── requirements.txt    # Python dependencies
```

## How to Deploy Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=your_key_here
```

3. Run the server:
```bash
python backend.py
```

4. Open http://localhost:7000 in your browser

The application will start a FastAPI server serving both the API endpoints and the static frontend.

## API Endpoints

- `GET /` - Serves the frontend (index.html)
- `POST /api/chat` - Conversational AI endpoint
- `POST /api/priorities` - Extract and manage priorities
- `POST /api/choices` - Generate decision options
