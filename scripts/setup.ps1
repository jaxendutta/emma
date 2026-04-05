# =============================================================================
# EMMA project setup for Windows (PowerShell)
# Run once from the repo root:
#
#   cd C:\path\to\your\repo\final
#   .\scripts\setup.ps1
#
# Requires:
#   - Python 3.11+ (https://www.python.org/downloads/)
#   - uv           (https://docs.astral.sh/uv/getting-started/installation/)
#     Install uv:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#
# Note: If you get a script execution error, run this first (once, as admin):
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "----------------------------------------" -ForegroundColor Cyan
Write-Host "  EMMA -- environment setup (Windows)"    -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan
Write-Host ""

# 1. Check we're in the right directory
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "ERROR: pyproject.toml not found." -ForegroundColor Red
    Write-Host "Run this script from the repo root (the folder containing pyproject.toml)." -ForegroundColor Red
    exit 1
}
Write-Host "OK  Working directory: $(Get-Location)" -ForegroundColor Green

# 2. Check uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "ERROR: uv not found." -ForegroundColor Red
    Write-Host "Install it by running:" -ForegroundColor Yellow
    Write-Host "  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
    Write-Host "Then restart PowerShell and re-run this script." -ForegroundColor Yellow
    exit 1
}
$uvVersion = uv --version
Write-Host "OK  uv found: $uvVersion" -ForegroundColor Green

# 3. Check Python 3.11+
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Python not found." -ForegroundColor Red
    Write-Host "Download Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}
Write-Host "OK  $pythonVersion" -ForegroundColor Green

# 4. Create .env from template if it doesn't exist
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "OK  Created .env from .env.example" -ForegroundColor Green
        Write-Host "    Add your NCBI API key to .env if you have one." -ForegroundColor DarkGray
    } else {
        Write-Host "    No .env.example found -- skipping .env creation." -ForegroundColor DarkGray
    }
} else {
    Write-Host "OK  .env already exists" -ForegroundColor Green
}

# 5. Sync the virtual environment
Write-Host ""
Write-Host "Installing dependencies (uv sync)..." -ForegroundColor Cyan
uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: uv sync failed." -ForegroundColor Red
    exit 1
}
Write-Host "OK  Dependencies installed" -ForegroundColor Green

# 6. Install the SpaCy biomedical model (en_core_sci_md)
#    Not on PyPI -- must be installed directly from the release URL
Write-Host ""
Write-Host "Installing SpaCy biomedical model (en_core_sci_md)..." -ForegroundColor Cyan
uv pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install en_core_sci_md." -ForegroundColor Red
    exit 1
}
Write-Host "OK  en_core_sci_md installed" -ForegroundColor Green

# 7. Register the Jupyter kernel so notebooks find the right environment
Write-Host ""
Write-Host "Registering Jupyter kernel..." -ForegroundColor Cyan
$pyVersionShort = uv run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
uv run python -m ipykernel install --user --name=emma --display-name="EMMA (Python $pyVersionShort)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Jupyter kernel registration failed." -ForegroundColor Red
    exit 1
}
Write-Host "OK  Jupyter kernel registered: EMMA (Python $pyVersionShort)" -ForegroundColor Green

# 8. Verify the src package is importable
Write-Host ""
Write-Host "Verifying package..." -ForegroundColor Cyan
uv run python -c "from src.data import REPO_ROOT; print(f'  src.data OK -- repo root: {REPO_ROOT}')"
uv run python -c "from src.vectorstore import BIOMEDICAL_MODEL; print(f'  src.vectorstore OK -- model: {BIOMEDICAL_MODEL}')"
uv run python -c "from src.classify import RANDOM_SEED; print(f'  src.classify OK -- seed: {RANDOM_SEED}')"

Write-Host ""
Write-Host "----------------------------------------" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  To open notebooks:" -ForegroundColor White
Write-Host "    uv run jupyter notebook notebooks\" -ForegroundColor Yellow
Write-Host ""
Write-Host "  To run a Python script:" -ForegroundColor White
Write-Host "    uv run python scripts\your_script.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  To open a Python shell:" -ForegroundColor White
Write-Host "    uv run python" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Vectorstore files (too large for git):" -ForegroundColor White
Write-Host "    Download from Google Drive and place in models\vectorstore\" -ForegroundColor Yellow
Write-Host "    Or run notebook 01 on Colab T4 to build from scratch." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Cyan
Write-Host ""
