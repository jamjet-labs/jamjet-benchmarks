#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "Done. To run:"
echo ""
echo "  source .venv/bin/activate"
echo "  export OPENAI_API_KEY=ollama"
echo "  export OPENAI_BASE_URL=http://localhost:11434/v1"
echo "  export MODEL_NAME=llama3.2"
echo "  python main.py"
