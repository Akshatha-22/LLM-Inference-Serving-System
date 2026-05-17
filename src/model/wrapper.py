"""
src/model/wrapper.py
GPT-2 model wrapper with inference methods
"""

import time
import torch
import psutil
import os
from typing import List, Optional, Dict, Any
from transformers import GPT2LMHeadModel
from src.model.tokenizer import TokenizerWrapper


class GPT2Wrapper:
    """
    Wrapper for GPT-2 model with CPU inference.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {model_name}...")
        start_time = time.time()
        
        # Load tokenizer first
        self.tokenizer = TokenizerWrapper(model_name)
        
        # Load model and move to CPU
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        
        # Set to evaluation mode (disables dropout)
        self.model.eval()
        
        load_time = time.time() - start_time
        param_count = sum(p.numel() for p in self.model.parameters())
        
        print(f"✓ Loaded in {load_time:.2f}s")
        print(f"✓ Parameters: {param_count:,}")
        print(f"✓ Memory: {self._get_memory_mb():.1f} MB")
        
    def _get_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        use_cache: bool = True,
        return_timing: bool = False
    ) -> Dict[str, Any]:
        """Generate text from a prompt."""
        # Tokenize input
        inputs = self.tokenizer.encode(prompt)
        input_ids = inputs["input_ids"].to(self.device)
        input_length = input_ids.shape[1]
        
        # Generate with timing
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # OPTIMIZATION: Use max_length instead of max_new_tokens
            # This is slightly faster because the model doesn't need to track separate counters
            outputs = self.model.generate(
                input_ids,
                max_length=input_length + max_new_tokens,  # ← CHANGED: was max_new_tokens
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=use_cache,
                return_dict_in_generate=True,
            )
        
        end_time = time.perf_counter()
        
        # Decode generated tokens
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Calculate metrics
        new_tokens = generated_ids.shape[0] - input_length
        elapsed = end_time - start_time
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0
        
        result = {
            "text": generated_text,
            "tokens": new_tokens,
        }
        
        if return_timing:
            result["time_seconds"] = elapsed
            result["tokens_per_sec"] = tokens_per_sec
        
        return result
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        return_timing: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_new_tokens, return_timing=return_timing)
            results.append(result)
        return results
    
    def get_baseline_throughput(self, num_runs: int = 5) -> Dict[str, float]:
        """Measure baseline throughput."""
        prompt = "The future of artificial intelligence is"
        max_tokens = 50
        
        speeds = []
        
        for i in range(num_runs):
            result = self.generate(
                prompt, 
                max_new_tokens=max_tokens,
                return_timing=True
            )
            speeds.append(result["tokens_per_sec"])
            print(f"  Run {i+1}: {result['tokens_per_sec']:.2f} tok/s")
        
        avg_speed = sum(speeds) / len(speeds)
        
        return {
            "avg_tokens_per_sec": avg_speed,
            "min": min(speeds),
            "max": max(speeds),
            "std": (sum((s - avg_speed)**2 for s in speeds) / len(speeds)) ** 0.5,
            "model": self.model_name,
            "device": self.device,
            "max_tokens": max_tokens
        }
