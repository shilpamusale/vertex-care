# Makefile for VertexCare Project

# ==============================================================================
# VARIABLES
# ==============================================================================

# Get the Python interpreter from the virtual environment.
# This makes sure we are using the correct python version.
PYTHON = python3

# ==============================================================================
# INSTALLATION & SETUP
# ==============================================================================

## install: Install all project dependencies from requirements.txt
install:
	@echo "Installing project dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# ==============================================================================
# CODE QUALITY & LINTING
# ==============================================================================

## lint: Run flake8 to check for style issues and errors
lint:
	@echo "Running flake8 linter..."
	$(PYTHON) -m flake8 src tests

## format: Automatically format code using black
format:
	@echo "Formatting code with black..."
	$(PYTHON) -m black src tests

## quality: Run all code quality checks
quality: lint format
	@echo "Code quality checks complete."

# ==============================================================================
# DATA PIPELINE (Example)
# ==============================================================================

## data: Run the full data processing pipeline (placeholder)
data:
	@echo "Running data processing pipeline..."
	# We will add the command to run our Python pipeline script here later

# ==============================================================================
# HELP
# ==============================================================================

## precommit: Run pre-commit hooks on all files
precommit:
	@echo "Running pre-commit hooks on all files..."
	pre-commit run --all-files

## quality: Run all code quality checks
quality: precommit lint format
	@echo "Code quality checks complete."

## help: Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install lint format quality data help
