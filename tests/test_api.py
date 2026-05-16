
"""
tests/test_api.py - FINAL PERFECT VERSION
Properly displays streaming text with correct spacing.
"""

import requests

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print(f"✅ Health: {response.json()}")
    return True

def test_non_streaming():
    print("\n" + "=" * 50)
    print("NON-STREAMING RESPONSE:")
    print("=" * 50)
    
    response = requests.post(
        f"{BASE_URL}/v1/completions",
        json={
            "prompt": "The future of artificial intelligence is",
            "max_tokens": 30,
            "stream": False
        }
    )
    
    data = response.json()
    print(f"Request ID: {data.get('id')}")
    print(f"Tokens generated: {data.get('tokens_generated')}")
    print(f"Generated text: {data.get('text')[:200]}")

def test_streaming():
    print("\n" + "=" * 50)
    print("STREAMING RESPONSE:")
    print("=" * 50)
    
    response = requests.post(
        f"{BASE_URL}/v1/completions",
        json={
            "prompt": "The future of artificial intelligence is",
            "max_tokens": 30,
            "stream": True
        },
        stream=True
    )
    
    print("Tokens: ", end="", flush=True)
    
    current_event = None
    first_token = True
    result_parts = []
    
    # Define punctuation that shouldn't have a space before them
    punctuation = set('.,!?;:)\'"')
    
    for line in response.iter_lines():
        if not line:
            continue
        
        decoded = line.decode('utf-8')
        
        if decoded.startswith('event:'):
            current_event = decoded[6:].strip()
        elif decoded.startswith('data:'):
            data = decoded[5:]  # Keep raw data
            
            if current_event == 'token' and data:
                # Clean the token for display
                cleaned = data.lstrip()  # Remove leading spaces
                
                if first_token:
                    # First token: print as-is (no leading space)
                    print(cleaned, end="", flush=True)
                    result_parts.append(cleaned)
                    first_token = False
                else:
                    # For subsequent tokens, check if it's punctuation
                    if cleaned and cleaned[0] in punctuation:
                        # Punctuation: don't add space
                        print(cleaned, end="", flush=True)
                    else:
                        # Normal word: add space
                        print(f" {cleaned}", end="", flush=True)
                    result_parts.append(cleaned)
                    
            elif current_event == 'done':
                print()  # New line
    
    print("\n" + "-" * 30)
    print(f"✅ Streaming complete!")

if __name__ == "__main__":
    print("=" * 50)
    print("API TEST SUITE")
    print("=" * 50)
    print("\nMake sure server is running!\n")
    
    if test_health():
        test_non_streaming()
        test_streaming()

