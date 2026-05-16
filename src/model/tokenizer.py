
"""
src/model/tokenizer.py
Tokenizer wrapper for GPT-2
"""

from transformers import GPT2Tokenizer
from typing import List, Dict, Any

class TokenizerWrapper:
    """Wrapper around HuggingFace GPT-2 tokenizer."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
    def encode(self, text: str, add_special_tokens: bool = True) -> Dict[str, Any]:
        """Convert text to token IDs."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            padding=False
        )
    
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        return len(self.tokenizer)
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
