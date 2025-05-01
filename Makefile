# Makefile for HerVoice Project

# Environment variables
ENV_FILE := .env
PYTHON := poetry run python
ACTIVATE := source .venv/bin/activate

# Docker Compose
DOCKER_COMPOSE := docker compose

.PHONY: help
help:
	@echo "HerVoice Makefile Commands:"
	@echo "  make install         - Install Poetry dependencies"
	@echo "  make venv            - Activate virtual environment"
	@echo "  make embed           - Preprocess and embed PDFs into vector DB"
	@echo "  make api             - Run FastAPI server locally"
	@echo "  make ui              - Run Streamlit frontend locally"
	@echo "  make dev             - Run both API and UI locally (2 terminals required)"

install:
	poetry install

embed:
	$(PYTHON) Knowledge.py

api:
	poetry run uvicorn hervoice.app:app --reload

ui:
	poetry run streamlit run src/HerVoice.py

test:
	@echo "Running local test script..."
	@chmod +x ./src/test_server.sh && ./src/test_server.sh

venv:
	@echo "To activate the Poetry virtual environment, run this manually:"
	@echo "source .venv/bin/activate"

dev:
	@echo "Run make api and make ui in separate terminals"