# Jacob: Share of Voice vs Market Share Analysis Platform
# Makefile for complete Docker-based workflow

.PHONY: help build dev jupyter prod clean test lint format docs analyze collect-data

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color
# Default target
help:
	@echo "Jacob: Share of Voice vs Market Share Analysis Platform"
	@echo "======================================================"
	@echo ""
	@echo "Available targets:"
	@echo "  build         - Build all Docker images"
	@echo "  dev           - Start development environment"
	@echo "  jupyter       - Start Jupyter Lab environment"
	@echo "  prod          - Run production analysis"
	@echo "  test          - Run test suite"
	@echo "  lint          - Run code linting"
	@echo "  format        - Format code with black"
	@echo "  clean         - Clean up Docker resources"
	@echo "  docs          - Generate documentation"
	@echo ""
	@echo "Analysis commands:"
	@echo "  analyze-automotive - Run automotive industry analysis"
	@echo "  analyze-pharma     - Run pharmaceutical industry analysis"
	@echo "  analyze-alcohol    - Run alcohol industry analysis"
	@echo "  collect-data       - Collect data for specified industry"
	@echo ""
	@echo "Example usage:"
	@echo "  make dev                    # Start development environment"
	@echo "  make analyze-automotive     # Run automotive analysis"
	@echo "  make jupyter               # Start Jupyter Lab"

# Build all Docker images
build: setup-dirs
	@echo "Building Docker images..."
	docker-compose -f docker/docker-compose.yaml build

# Create necessary directories
setup-dirs:
	@echo "Creating necessary directories..."
	@mkdir -p output logs tests jupyterlab docs
	@echo "✓ All directories created"


# Start development environment
dev-shell:
	@echo "$(GREEN)Starting development environment...$(NC)"
	docker-compose -f docker/docker-compose.yaml up jacob
	docker-compose exec jacob bash
# Start Jupyter Lab environment
jupyter:
	@echo "Starting Jupyter Lab environment..."
	@echo "Access Jupyter Lab at: http://localhost:8888"
	docker-compose -f docker/docker-compose.yaml up jupyter

# Run production analysis
prod:
	@echo "Running production analysis..."
	docker-compose -f docker/docker-compose.yaml run --rm production

# Industry-specific analysis targets
analyze-automotive:
	@echo "Running automotive industry analysis..."
	docker-compose -f docker/docker-compose.yaml run --rm jacob \
		python -m src.cli analyze --industry automotive --config config/automotive.yaml

analyze-pharma:
	@echo "Running pharmaceutical industry analysis..."
	docker-compose -f docker/docker-compose.yaml run --rm jacob \
		python -m src.cli analyze --industry pharma --config config/default.yaml

analyze-alcohol:
	@echo "Running alcohol industry analysis..."
	docker-compose -f docker/docker-compose.yaml run --rm jacob \
		python -m src.cli analyze --industry alcohol --config config/default.yaml

# Data collection
collect-data:
	@if [ -z "$(INDUSTRY)" ]; then \
		echo "Error: Please specify INDUSTRY. Usage: make collect-data INDUSTRY=automotive"; \
		exit 1; \
	fi
	@echo "Collecting data for $(INDUSTRY) industry..."
	docker-compose -f docker/docker-compose.yaml run --rm jacob \
		python -m src.cli collect-data --industry $(INDUSTRY)

# Initialize configuration
init-config:
	@if [ -z "$(INDUSTRY)" ]; then \
		echo "Error: Please specify INDUSTRY. Usage: make init-config INDUSTRY=automotive"; \
		exit 1; \
	fi
	@echo "Initializing configuration for $(INDUSTRY) industry..."
	docker-compose -f docker/docker-compose.yaml run --rm jacob \
		python -m src.cli init-config --config-template $(INDUSTRY) --output config/$(INDUSTRY).yaml

# Development and testing
test:
	@echo "Running test suite..."
	docker-compose -f docker/docker-compose.yaml run --rm jacob pytest tests/ -v --cov=src

lint:
	@echo "Running code linting..."
	docker-compose -f docker/docker-compose.yaml run --rm jacob flake8 src/ tests/
	docker-compose -f docker/docker-compose.yaml run --rm jacob mypy src/

format:
	@echo "Formatting code..."
	docker-compose -f docker/docker-compose.yaml run --rm jacob black src/ tests/

# Documentation
docs:
	@echo "Generating documentation..."
	docker-compose -f docker/docker-compose.yaml run --rm jacob sphinx-build -b html docs/ docs/_build/

# Cleanup
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose -f docker/docker-compose.yaml down -v --remove-orphans
	docker system prune -f

# Deep cleanup
clean-all: clean
	@echo "Removing all Docker images..."
	docker-compose -f docker/docker-compose.yaml down --rmi all -v --remove-orphans

# Setup development environment
setup-dev: build
	@echo "Setting up development environment..."
	docker-compose -f docker/docker-compose.yaml run --rm jacob \
		python -m src.cli init-config --config-template automotive --output config/automotive.yaml

# Check system requirements
check-requirements:
	@echo "Checking system requirements..."
	@command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }
	@echo " All requirements satisfied"

# Full analysis pipeline
full-analysis:
	@if [ -z "$(INDUSTRY)" ]; then \
		echo "Error: Please specify INDUSTRY. Usage: make full-analysis INDUSTRY=automotive"; \
		exit 1; \
	fi
	@echo "Running full analysis pipeline for $(INDUSTRY) industry..."
	make collect-data INDUSTRY=$(INDUSTRY)
	make analyze-$(INDUSTRY)
	@echo " Full analysis completed. Check output/$(INDUSTRY)/ for results."