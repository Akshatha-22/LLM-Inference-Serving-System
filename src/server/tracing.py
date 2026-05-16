
"""
src/server/tracing.py
Request tracing and structured logging
"""

import uuid
import time
from contextvars import ContextVar
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import logging

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_request_id() -> str:
    """Get current request ID (for use in route handlers)."""
    req_id = request_id_var.get()
    return req_id if req_id else "unknown"


def log_with_request_id(message: str, level: str = "info"):
    """Log a message with the current request ID attached."""
    req_id = get_request_id()
    log_message = f"[{req_id}] {message}"
    
    if level == "info":
        logger.info(log_message)
    elif level == "error":
        logger.error(log_message)
    elif level == "warning":
        logger.warning(log_message)
    elif level == "debug":
        logger.debug(log_message)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique ID to every request."""
    
    async def dispatch(self, request: Request, call_next):
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        token = request_id_var.set(request_id)
        
        log_with_request_id(f"REQUEST: {request.method} {request.url.path}")
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            elapsed = time.perf_counter() - start_time
            log_with_request_id(f"COMPLETED: {response.status_code} in {elapsed:.3f}s")
            return response
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            log_with_request_id(f"FAILED: {str(e)} in {elapsed:.3f}s", level="error")
            raise
        finally:
            request_id_var.reset(token)
