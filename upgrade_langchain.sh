#!/bin/bash

# Activate the virtual environment if you're using one
# source /path/to/your/venv/bin/activate

# Install ruff if not already installed
pip install ruff

# Run ruff to automatically fix LangChain imports
ruff check --select UP006 --fix .

# Deactivate the virtual environment if you activated it
# deactivate

echo "LangChain upgrade completed using ruff."
