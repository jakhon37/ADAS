# Makefile for ADAS Core

.PHONY: help install test lint format clean docker run

help:
	@echo "ADAS Core - Makefile Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install       Install package in development mode"
	@echo "  make test          Run test suite"
	@echo "  make lint          Run linter (ruff)"
	@echo "  make format        Format code"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Deployment:"
	@echo "  make docker        Build Docker image"
	@echo "  make run           Run synthetic test"
	@echo ""

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src tests

format:
	ruff check --fix src tests
	ruff format src tests

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

docker:
	docker build -t adas-core:latest .

run:
	python src/adas/cli.py --frames 10

run-debug:
	python src/adas/cli.py --frames 10 --log-level DEBUG

# CI/CD targets
ci-test: install test lint

ci-build: clean docker
