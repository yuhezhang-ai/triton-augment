.PHONY: help test docs clean

help:
	@echo "Triton-Augment Development Commands"
	@echo ""
	@echo "  make docs       - Build and serve documentation locally"
	@echo "  make test       - Run all tests"
	@echo "  make clean      - Clean build artifacts"
	@echo ""

docs:
	@echo "Building and serving documentation..."
	@cd docs && mkdocs serve

test:
	@echo "Running tests..."
	@pytest tests/ -v

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf dist/ build/ *.egg-info site/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"

