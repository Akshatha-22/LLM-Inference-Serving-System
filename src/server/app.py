

"""
src/server/app.py
FastAPI application entry point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.model.wrapper import GPT2Wrapper
from src.server.routes import router, set_model
from src.server.tracing import RequestTracingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    print("\n" + "=" * 60)
    print("🚀 Starting LLM Inference Server")
    print("=" * 60)
    
    print("\nLoading GPT-2 model...")
    model = GPT2Wrapper(model_name="gpt2", device="cpu")
    set_model(model)
    
    print("\n✅ Server ready!")
    print("📍 API docs: http://localhost:8000/docs")
    print("📍 Health: http://localhost:8000/health")
    print("📍 Completions: POST http://localhost:8000/v1/completions")
    print("\n" + "=" * 60 + "\n")
    
    yield
    
    print("\nShutting down server...")


# Create FastAPI app
app = FastAPI(
    title="LLM Inference Server",
    description="GPT-2 inference with streaming and request tracing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (allows frontend apps to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request tracing middleware
app.add_middleware(RequestTracingMiddleware)

# Register routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "LLM Inference Server",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "completions": "POST /v1/completions"
        }
    }
