#!/bin/bash
set -e

VENV_DIR="/app/.venv"

if [ ! -f "$VENV_DIR/bin/uvicorn" ]; then
    echo "=== First run: creating virtual environment ==="
    python -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --no-cache-dir -r requirements.api.txt
    echo "=== Dependencies installed ==="
else
    echo "=== Using existing virtual environment ==="
fi

exec "$VENV_DIR/bin/uvicorn" api:app --host 0.0.0.0 --port 8000
