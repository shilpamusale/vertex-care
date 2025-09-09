# Makefile for VertexCare Project

.PHONY: all setup lint test run-notebooks run-scripts clean run-pipeline run-clustering run-routing run-add-mock-notes

# 1. Install dependencies (using pip, Poetry, or conda)
setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# 2. Lint the source code
lint:
	@echo "Running flake8 lint checks..."
	flake8 vertexcare scripts tests

# 3. Run unit tests
test:
	@echo "Running tests with pytest..."
	pytest tests

# # 4. Run all Jupyter notebooks (in order to check for errors)
# run-notebooks:
# 	@echo "Running all notebooks in /notebooks..."
# 	@for nb in notebooks/*.ipynb; do \
# 		echo "Testing $$nb..."; \
# 		jupyter nbconvert --to notebook --execute --inplace $$nb; \
# 	done

# 5. Run main scripts (customize as needed)
run-add-mock-notes:
	@echo "Running add_mock_notes script..."
	python scripts/add_mock_notes.py

run-pipeline:
	@echo "Running main pipeline script..."
	python scripts/run_pipeline.py

run-clustering:
	@echo "Running clustering pipeline script..."
	python scripts/run_clustering_pipeline.py

run-routing:
	@echo "Running routing pipeline script..."
	python scripts/run_routing_pipeline.py

# 6. Clean up artifacts
clean:
	@echo "Cleaning up logs, models, and __pycache__..."
	rm -rf logs/*
	rm -rf models/*
	find . -type d -name "__pycache__" -exec rm -rf {} +

# ==============================================================================
# APPLICATION (ONLINE)
# ==============================================================================

serve-api: ## Serve the backend FastAPI application
	@echo ">>> Starting FastAPI backend on http://localhost:8000 ..."
	poetry run uvicorn vertexcare.api.main:app --host 0.0.0.0 --port 8000

serve-ui: ## Serve the frontend Streamlit dashboard
	@echo ">>> Starting Streamlit frontend..."
	poetry run streamlit run scripts/dashboard.py

# ==============================================================================
# UTILITY
# ==============================================================================

clean: ## Remove generated files (pycache, data, models)
	@echo ">>> Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf vertexcare/data/02_intermediate/*
	rm -rf vertexcare/data/03_primary/*
	rm -rf vertexcare/models/*

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# 7. Run everything in order (excluding notebooks)
all: lint test run-add-mock-notes run-pipeline run-clustering run-routing
