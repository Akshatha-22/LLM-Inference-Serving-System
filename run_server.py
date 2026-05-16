
"""
run_server.py
Entry point to start the FastAPI server.

Usage:
    python run_server.py
    python run_server.py --port 8080
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start LLM Inference Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Auto-reload (dev)")
    
    args = parser.parse_args()
    
    print(f"\n🚀 Starting server at http://{args.host}:{args.port}")
    print(f"📚 API docs at http://{args.host}:{args.port}/docs\n")
    
    uvicorn.run(
        "src.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
