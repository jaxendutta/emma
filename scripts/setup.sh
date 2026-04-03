#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# EMMA project setup — run once from the final/ directory in Linux (Use WSL for Windows)
#
#   cd /path/to/your/repo/final
#   bash scripts/setup.sh
#
# Requires: uv (https://docs.astral.sh/uv/getting-started/installation/)
# Install uv if you don't have it:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error
UV_LINK_MODE=copy  # ensure uv creates a local .venv with copies, not symlinks (better Windows support)

echo "────────────────────────────────────────"
echo "        EMMA — Environment Setup        "
echo "────────────────────────────────────────"

# 1. Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "✗  uv not found. Install it first:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "✓  uv found: $(uv --version)"

# 2. Create .env from template if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓  Created .env from .env.example — add your NCBI key if you have one"
else
    echo "✓  .env already exists"
fi

# 3. Sync the environment (creates .venv, installs all deps + emma in editable mode)
echo ""
echo "> Installing dependencies with uv sync..."
uv sync

# 4. Install the spacy biomedical model (not on PyPI, needs a separate step)
#    uv pip install targets the project .venv directly — no system pip confusion
echo ""
echo "> Installing SpaCy biomedical model (en_core_sci_md)..."
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

# 5. Register the kernel so Jupyter notebooks find the right environment
echo ""
echo "> Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name=emma --display-name="EMMA"

echo ""
echo "────────────────────────────────────────"
echo " ✓  Setup complete!"
echo ""
echo " To start Jupyter:"
echo "   uv run jupyter notebook notebooks/"
echo ""
echo " To run a Python script:"
echo "   uv run python scripts/your_script.py"
echo ""
echo " To open a Python shell:"
echo "   uv run python"
echo "────────────────────────────────────────"