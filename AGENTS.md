# Second Opinion - Starter Repo

A simple decision-making assistant that helps users think through choices.

## Features
- **Chat**: Conversational AI that asks clarifying questions
- **Priorities**: Extracts and lets users rank what matters most
- **Choices**: Generates options with best/worst case scenarios

## Setup

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

4. Open http://localhost:7000

## Files
- `backend.py` - FastAPI server with 3 endpoints (/api/chat, /api/priorities, /api/choices)
- `index.html` - Single-page frontend with Alpine.js
- `requirements.txt` - Python dependencies
