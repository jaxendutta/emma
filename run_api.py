"""
Start the EMMA API + static frontend with one command.

Usage
-----
    uv run python run_api.py          # API on :8080, frontend on :8001
    uv run python run_api.py --rag    # same but with RAG / Cloud LLM enabled
    uv run python run_api.py --port 9000

Frontend
--------
    http://localhost:8001/?tab=home   <- open this in your browser
"""

import argparse
import os
import subprocess
import sys
import threading
import time

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(
        prog="run_api",
        description="EMMA API + frontend dev server",
    )
    p.add_argument(
        "--rag",
        action="store_true",
        default=False,
        help="Enable Cloud LLM / RAG (requires GEMINI_API_KEY or GROQ_API_KEY in .env)",
    )
    p.add_argument("--port", type=int, default=8080, metavar="PORT",
                   help="API port (default: 8080)")
    p.add_argument("--frontend-port", type=int, default=8001,
                   metavar="FRONTEND_PORT", dest="frontend_port",
                   help="Frontend port (default: 8001)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.rag:
        os.environ["EMMA_USE_RAG"] = "true"

    rag_enabled = os.environ.get("EMMA_USE_RAG", "false").lower() == "true"
    client_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client")

    print("=" * 56)
    print("  EMMA  —  local dev server")
    print("=" * 56)
    print(f"  API      : http://localhost:{args.port}/")
    print(f"  Frontend : http://localhost:{args.frontend_port}/?tab=home  <- open this")
    print(f"  RAG/LLM  : {'ENABLED' if rag_enabled else 'DISABLED  (add EMMA_USE_RAG=true to .env)'}")
    print("=" * 56)
    print()

    # -- Frontend: spawn a separate python process for the static server so
    #    it survives uvicorn's reload-mode forking (daemon threads don't).
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(args.frontend_port),
         "--directory", client_dir, "--bind", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give the static server a moment to bind before printing the URL
    time.sleep(0.5)

    try:
        # -- API: run uvicorn directly (blocking)
        import uvicorn
        uvicorn.run(
            "src.api:app",
            host="127.0.0.1",
            port=args.port,
            reload=True,
            log_level="info",
        )
    finally:
        frontend_proc.terminate()


if __name__ == "__main__":
    main()
