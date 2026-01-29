.PHONY: run_phoenix help

help:
	@echo "Available targets:"
	@echo "  run_phoenix    - Launch Phoenix UI server on port 6006"
	@echo "  help           - Show this help message"

run_phoenix:
	@echo "Starting Phoenix UI server on port 6006..."
	@echo "Access the UI at: http://localhost:6006"
	python -m phoenix.server.main serve
