"""
src/server/routes.py
FINAL - Preserves spaces in streaming tokens
"""

from typing import Optional, AsyncGenerator
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
import torch
import torch.nn.functional as F

from src.model.wrapper import GPT2Wrapper
from src.server.tracing import get_request_id, log_with_request_id

router = APIRouter()
_model: Optional[GPT2Wrapper] = None


def set_model(model: GPT2Wrapper):
    global _model
    _model = model


def get_model() -> GPT2Wrapper:
    if _model is None:
        raise RuntimeError("Model not initialized")
    return _model


class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="Input text to complete")
    max_tokens: int = Field(50, ge=1, le=500)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    stream: bool = Field(False)


class CompletionResponse(BaseModel):
    id: str
    text: str
    tokens_generated: int
    finish_reason: str


async def token_stream_generator(
    prompt: str,
    max_tokens: int,
    temperature: float
) -> AsyncGenerator[dict, None]:
    """
    CORRECT STREAMING - Preserves spaces in tokens.
    """
    model = get_model()
    tokenizer = model.tokenizer
    
    log_with_request_id(f"Starting streaming: max_tokens={max_tokens}")
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt)
    input_ids = inputs["input_ids"]
    input_ids = input_ids.to(model.device)
    
    # Stream prompt tokens - preserve spaces!
    prompt_tokens = input_ids[0].tolist()
    for token_id in prompt_tokens:
        # skip_special_tokens=False keeps spaces
        token_text = tokenizer.tokenizer.decode([token_id], skip_special_tokens=False)
        # DO NOT strip() - spaces are important!
        yield {"event": "token", "data": token_text}
    
    # Generate new tokens
    past_key_values = None
    current_input = input_ids
    generated_count = 0
    
    for step in range(max_tokens):
        with torch.no_grad():
            outputs = model.model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        next_token_logits = logits[:, -1, :]
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.argmax(probs, dim=-1).item()
        
        # skip_special_tokens=False keeps spaces
        token_text = tokenizer.tokenizer.decode([next_token_id], skip_special_tokens=False)
        # DO NOT strip() - spaces are important!
        yield {"event": "token", "data": token_text}
        
        current_input = torch.tensor([[next_token_id]]).to(model.device)
        generated_count += 1
        
        if next_token_id == tokenizer.eos_token_id:
            break
    
    yield {"event": "done", "data": f"generated {generated_count} tokens"}
    log_with_request_id(f"Streaming complete: {generated_count} new tokens")


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    request_id = get_request_id()
    log_with_request_id(f"Received: prompt='{request.prompt[:50]}...'")
    
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="prompt cannot be empty")
    
    if request.stream:
        log_with_request_id("Streaming mode")
        return EventSourceResponse(
            token_stream_generator(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ),
            headers={
                "X-Request-ID": request_id,
                "Cache-Control": "no-cache",
                "Content-Type": "text/event-stream"
            }
        )
    
    log_with_request_id("Non-streaming mode")
    model = get_model()
    result = model.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        do_sample=request.temperature != 1.0,
        return_timing=True
    )
    
    return CompletionResponse(
        id=request_id,
        text=result["text"],
        tokens_generated=result["tokens"],
        finish_reason="stop"
    )


@router.get("/health")
async def health():
    model = get_model()
    return {
        "status": "healthy",
        "model": model.model_name,
        "device": str(model.device),
        "request_id": get_request_id()
    }
