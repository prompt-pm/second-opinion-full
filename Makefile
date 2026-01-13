.PHONY: help install install-backend install-frontend run dev test test-backend test-frontend lint lint-backend lint-frontend lint-fix format clean

# Default port for local development
PORT ?= 7000

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: install-backend install-frontend ## Install all dependencies

install-backend: ## Install Python backend dependencies
	uv venv
	uv pip install -r requirements.txt

install-frontend: ## Install frontend dependencies
	bun install

# Running
run: ## Run the backend server
	uv run uvicorn backend.main:app --host 0.0.0.0 --port $(PORT)

dev: ## Run the backend server with hot reload
	uv run uvicorn backend.main:app --host 0.0.0.0 --port $(PORT) --reload

# Testing
test: test-backend test-frontend ## Run all tests

test-backend: ## Run backend Python tests
	uv run pytest tests/ -v

test-frontend: ## Run frontend JavaScript tests
	bun run test

test-watch: ## Run frontend tests in watch mode
	bun run test:watch

test-cov: ## Run backend tests with coverage
	uv run pytest tests/ -v --cov=backend --cov-report=term-missing

# Linting
lint: lint-backend lint-frontend ## Run all linters

lint-backend: ## Lint Python code with ruff
	uv run ruff check backend/ tests/

lint-frontend: ## Lint JavaScript code with ESLint
	bun run lint

lint-fix: ## Fix linting issues (both backend and frontend)
	uv run ruff check --fix backend/ tests/
	bun run lint:fix

format: ## Format Python code with ruff
	uv run ruff format backend/ tests/

format-check: ## Check Python code formatting without modifying
	uv run ruff format --check backend/ tests/

# Cleanup
clean: ## Remove generated files and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache .coverage htmlcov
	rm -rf backend/__pycache__ tests/__pycache__
	rm -rf node_modules
	rm -rf .venv
