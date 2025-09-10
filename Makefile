# Makefile for VertexCare Project

.PHONY: all setup lint test add-mock-notes train clustering routing serve-api serve-ui clean help

# 1. Install dependencies
setup: ## Install dependencies (pip or poetry)
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	# Uncomment below if you use poetry instead of pip:
	# poetry install

# 2. Lint the source code
lint: ## Run flake8 lint checks
	@echo "Running flake8 lint checks..."
	flake8 vertexcare scripts tests

# 3. Run unit tests
test: ## Run pytest unit tests
	@echo "Running tests with pytest..."
	pytest tests

# 4. Run main scripts (customize as needed)
add-mock-notes: ## Run add_mock_notes script
	@echo "Running add_mock_notes script..."
	python scripts/add_mock_notes.py

train: ## Run main pipeline script
	@echo "Running main pipeline script..."
	python scripts/run_pipeline.py

clustering: ## Run clustering pipeline script
	@echo "Running clustering pipeline script..."
	python scripts/run_clustering_pipeline.py

routing: ## Run routing pipeline script
	@echo "Running routing pipeline script..."
	python scripts/run_routing_pipeline.py

# 5. Serve applications
serve-api: ## Serve the backend FastAPI application
	@echo ">>> Starting FastAPI backend on http://localhost:8000 ..."
	poetry run uvicorn vertexcare.api.main:app --host 0.0.0.0 --port 8000

serve-ui: ## Serve the frontend Streamlit dashboard
	@echo ">>> Starting Streamlit frontend..."
	poetry run streamlit run scripts/dashboard.py

# 6. Utility
clean: ## Remove generated files (pycache, data, models)
	@echo ">>> Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf vertexcare/data/02_intermediate/*
	rm -rf vertexcare/data/03_primary/*
	rm -rf vertexcare/models/*
	rm -rf logs/*

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# 7. Run everything in order (excluding notebooks)
all: lint test add-mock-notes train clustering routing
