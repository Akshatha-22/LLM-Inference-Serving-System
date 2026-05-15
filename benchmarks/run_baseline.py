"""
benchmarks/run_baseline.py
Day 2: Baseline inference measurement
"""

import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import psutil
import os

print("=" * 60)
print("BASELINE BENCHMARK — Day 2")
print("=" * 60)

# Load tokenizer
print("\n1. Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("   ✓ Tokenizer loaded")

# Load model
print("\n2. Loading GPT-2 Small (124M parameters)...")
print("   This may take 10-20 seconds on first run...")
start_load = time.time()
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
load_time = time.time() - start_load
print(f"   ✓ Loaded in {load_time:.2f}s")

# Count parameters
param_count = sum(p.numel() for p in model.parameters())
print(f"   ✓ Parameters: {param_count:,}")

# Memory usage
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"   ✓ Memory: {memory_mb:.1f} MB")

# Run baseline generation
print("\n3. Running baseline (5 runs, 50 tokens each)...")
print("-" * 40)

prompt = "The future of artificial intelligence is"
max_new_tokens = 50
speeds = []

for i in range(5):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    
    start = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    elapsed = time.perf_counter() - start
    new_tokens = outputs.shape[1] - input_len
    tps = new_tokens / elapsed
    speeds.append(tps)
    print(f"   Run {i+1}: {tps:.2f} tok/s ({elapsed:.2f}s)")

# Calculate statistics
avg_speed = sum(speeds) / len(speeds)
min_speed = min(speeds)
max_speed = max(speeds)

# Sample output
print("\n4. Sample generation:")
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    result = model.generate(
        inputs["input_ids"],
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
print(f"   Prompt: '{prompt}'")
print(f"   Output: {generated_text[:200]}...")

# Final results
print("\n" + "=" * 60)
print("✅ BASELINE RESULTS")
print("=" * 60)
print(f"   Load time: {load_time:.2f}s")
print(f"   Memory: {memory_mb:.1f} MB")
print(f"   Parameters: {param_count:,}")
print(f"   Tokens/sec (avg): {avg_speed:.2f}")
print(f"   Tokens/sec (min): {min_speed:.2f}")
print(f"   Tokens/sec (max): {max_speed:.2f}")
print("\n   🎯 This is your North Star baseline!")
print(f"   → Every optimization will be compared to {avg_speed:.2f} tok/s")

# Save results
import json
results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": "gpt2",
    "load_time_seconds": load_time,
    "memory_mb": memory_mb,
    "parameters": param_count,
    "tokens_per_sec": {
        "avg": avg_speed,
        "min": min_speed,
        "max": max_speed
    },
    "test_prompt": prompt,
    "max_new_tokens": max_new_tokens
}

# Create results directory if it doesn't exist
import os as os_module
os_module.makedirs("benchmarks/results", exist_ok=True)

with open("benchmarks/results/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n   📁 Results saved to: benchmarks/results/baseline_results.json")
