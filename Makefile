.PHONY: setup chunks index ui test test-unit test-integration test-e2e test-coverage test-watch test-fast test-parallel pre-commit-install pre-commit-run pre-commit-update

# Setup
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# Data processing
chunks:
	. .venv/bin/activate && python scripts/01_build_chunks.py
index:
	. .venv/bin/activate && python scripts/02_embed_and_index_mongo.py

# UI
ui:
	. .venv/bin/activate && streamlit run app/ui_streamlit.py

# Testing
test:
	pytest tests/

test-unit:
	pytest tests/unit/ -m unit

test-integration:
	pytest tests/integration/ -m integration

test-e2e:
	pytest tests/e2e/ -m e2e

test-coverage:
	pytest --cov=app --cov=chat_UI --cov-report=html --cov-report=term-missing

test-coverage-xml:
	pytest --cov=app --cov=chat_UI --cov-report=xml

test-fast:
	pytest tests/ -m "not slow" -m "not requires_api"

test-parallel:
	pytest tests/ -n auto

# Pre-commit hooks
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate
