.PHONY: env kernel clean help test

VENV_DIR := .venv

help:
	@echo "Available targets:"
	@echo "  make env    - Create virtual environment and install requirements"
	@echo "  make kernel - Create Jupyter kernel for the environment"
	@echo "  make test   - Run unit tests"
	@echo "  make clean  - Remove virtual environment and kernel"
	@echo "  make all    - Run env and kernel targets"

env:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Installing requirements..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt
	@echo "Installing ipykernel..."
	$(VENV_DIR)/bin/pip install ipykernel
	@echo "Environment created successfully!"

kernel: env
	@echo "Creating Jupyter kernel 'statlearning'..."
	$(VENV_DIR)/bin/python -m ipykernel install --user --name statlearning --display-name "Python (statlearning)"
	@echo "Kernel 'statlearning' created successfully!"

all: kernel

test:
	@echo "Running tests..."
	$(VENV_DIR)/bin/pytest tests/ -v

clean:
	@echo "Removing Jupyter kernel..."
	jupyter kernelspec uninstall statlearning -y 2>/dev/null || true
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Cleanup complete!"
