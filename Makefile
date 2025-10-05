# Makefile for OAI-inpainting project

.PHONY: help test test-unit test-integration test-quick install clean lint format

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install development dependencies
	pip install -r requirements-dev.txt

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-quick: ## Run quick tests (unit tests only, no integration)
	pytest tests/unit/ -v --tb=short

test-coverage: ## Run tests with coverage report
	pytest tests/ -v --cov=scripts --cov=data --cov-report=term-missing --cov-report=html

lint: ## Run linting checks
	flake8 scripts/ data/ tests/
	mypy scripts/ data/

format: ## Format code
	black scripts/ data/ tests/
	isort scripts/ data/ tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

test-subset: ## Test subset_4 pipeline (quick)
	python scripts/test_subset_4.py --models aot-gan --timeout 60

test-all-models: ## Test all model variants on subset_4
	python scripts/test_subset_4.py --models all --timeout 300
