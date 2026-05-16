# Stop the server first (Ctrl+C in Terminal 1)


async def token_stream_generator(
    prompt: str,
    max_tokens: int,
    temperature: float
) -> AsyncGenerator[dict, None]:
    """
    TRUE TOKEN-BY-TOKEN STREAMING with prompt included.
    """
    model = get_model()
    tokenizer = model.tokenizer
    
    log_with_request_id(f"Starting streaming: max_tokens={max_tokens}")
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt)
    input_ids = inputs["input_ids"]
    
    # Move to model device
    input_ids = input_ids.to(model.device)
    
    # FIRST: Stream the prompt tokens so client sees them
    prompt_tokens = input_ids[0].tolist()
    for token_id in prompt_tokens:
        token_text = tokenizer.tokenizer.decode([token_id])
        yield {"event": "token", "data": token_text}
    
    # Initialize for generation
    past_key_values = None
    current_input = input_ids
    generated_count = 0
    
    for step in range(max_tokens):
        # Forward pass
        with torch.no_grad():
            outputs = model.model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        # Get next token logits
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Greedy decode
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.argmax(probs, dim=-1).item()
        
        # Decode and send
        token_text = tokenizer.tokenizer.decode([next_token_id])
        yield {"event": "token", "data": token_text}
        
        # Update for next iteration
        current_input = torch.tensor([[next_token_id]]).to(model.device)
        generated_count += 1
        
        # Check for end
        if next_token_id == tokenizer.eos_token_id:
            break
    
    yield {"event": "done", "data": f"generated {generated_count} tokens"}
    log_with_request_id(f"Streaming complete: {generated_count} tokens")
