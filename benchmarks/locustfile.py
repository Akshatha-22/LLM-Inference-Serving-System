"""
benchmarks/locustfile.py
Fixed for slower CPU - realistic timeouts
"""

from locust import HttpUser, task, between
import time

class LLMUser(HttpUser):
    wait_time = between(3, 5)  # Longer wait between requests (3-5 seconds)
    
    test_prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, the world will",
        "The most important scientific discovery is",
    ]
    
    def on_start(self):
        self.prompt_index = 0
    
    def _get_next_prompt(self):
        prompt = self.test_prompts[self.prompt_index]
        self.prompt_index = (self.prompt_index + 1) % len(self.test_prompts)
        return prompt
    
    @task
    def completions_non_streaming(self):
        prompt = self._get_next_prompt()
        
        # Longer timeout for slow CPU
        with self.client.post(
            "/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 20,  # Reduced from 50 to 20
                "stream": False
            },
            catch_response=True,
            timeout=120,  # 2 minute timeout
            name="/v1/completions"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                tokens = data.get("tokens_generated", 0)
                if tokens == 0:
                    response.failure("No tokens generated")
            else:
                response.failure(f"HTTP {response.status_code}")
