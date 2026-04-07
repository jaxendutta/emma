"""
Start the EMMA FastAPI webhook server.

Usage
-----
    uv run python run_api.py                        # static knowledge only
    uv run python run_api.py --rag                  # enable RAG pipeline
    uv run python run_api.py --rag --model gemma3:4b
    uv run python run_api.py --port 9000
    uv run python run_api.py --rag --no-reload      # production-style (no auto-reload)

Endpoints
---------
    GET  /health         service health + feature flags
    POST /webhook        Dialogflow ES webhook
    POST /query          direct query (bypass Dialogflow, for testing)
    GET  /conditions     list of evaluation-domain conditions

Dialogflow setup
----------------
    1. Start this server, then expose it via ngrok:
           ngrok http 8000
    2. In the Dialogflow ES console:
           Fulfillment -> Webhook -> URL: https://<ngrok-id>.ngrok.io/webhook
    3. Enable webhook for each intent (GetSymptoms, GetDiagnosis, etc.)
"""

import argparse
import os

import uvicorn


def parse_args():
    p = argparse.ArgumentParser(
        prog="run_api",
        description="EMMA FastAPI webhook server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run python run_api.py\n"
            "  uv run python run_api.py --rag\n"
            "  uv run python run_api.py --rag --model qwen3:4b\n"
            "  uv run python run_api.py --rag --port 9000 --no-reload"
        ),
    )
    p.add_argument(
        "--rag",
        action="store_true",
        default=False,
        help="Enable the RAG pipeline (requires Ollama + model artefacts in models/)",
    )
    p.add_argument(
        "--model",
        metavar="MODEL_ID",
        default=None,
        help="Ollama model to use (default: value of 'default_model' in config/models.json)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        metavar="PORT",
        help="Port to listen on (default: 8000)",
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="HOST",
        help="Host to bind to (default: 0.0.0.0)",
    )
    p.add_argument(
        "--no-reload",
        action="store_true",
        default=False,
        help="Disable auto-reload on file changes (use in production)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Flags -> env vars so api.py picks them up without any import-time changes
    os.environ["EMMA_USE_RAG"] = "true" if args.rag else "false"
    if args.model:
        os.environ["EMMA_MODEL_ID"] = args.model

    print("=" * 52)
    print("  EMMA  —  FastAPI Webhook")
    print("=" * 52)
    print(f"  Host     : {args.host}:{args.port}")
    if args.rag:
        model_label = args.model or "default from config/models.json"
        print(f"  RAG      : ENABLED  ({model_label})")
    else:
        print("  RAG      : DISABLED  (static knowledge only)")
    print(f"  Reload   : {'off' if args.no_reload else 'on'}")
    print()
    print(f"  GET   http://localhost:{args.port}/health")
    print(f"  POST  http://localhost:{args.port}/webhook")
    print(f"  POST  http://localhost:{args.port}/query")
    print(f"  GET   http://localhost:{args.port}/conditions")
    print("=" * 52)

    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
