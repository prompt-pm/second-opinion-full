-include .env
export

.PHONY: dev
dev:
	python3 -m uvicorn backend:web_app --reload --port 7000

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
	pip install uv
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
