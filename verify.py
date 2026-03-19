import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Force load .env
env_path = Path("D:/AI Research Assistant/.env.example")
load_dotenv(dotenv_path=env_path, override=True)

print("=" * 50)
print("  Complete LLM Verification")
print("=" * 50)

# Check 1: API Key
key = os.getenv("GROQ_API_KEY", "")
print(f"\n1. GROQ_API_KEY loaded: {'YES' if key else 'NO'}")
if key:
    print(f"   Preview: {key[:8]}...{key[-4:]}")
else:
    print("   Key not found in .env")

# Check 2: Ollama running?
print("\n2. Checking if Ollama server is running...")
try:
    import httpx
    resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
    if resp.status_code == 200:
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"   Ollama is RUNNING with models: {models}")
        print("   WARNING: Ollama running but Groq will be used if key is set")
    else:
        print("   Ollama server: NOT running (good)")
except Exception:
    print("   Ollama server: NOT running (good)")

# Check 3: Test Groq directly
print("\n3. Testing Groq API directly...")
if key:
    try:
        import httpx
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json"
        }
        payload = {
            "model":       "llama-3.1-8b-instant",
            "messages":    [{"role": "user", "content": "Say exactly: GROQ_OK"}],
            "max_tokens":  10,
            "temperature": 0
        }
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            print(f"   Groq response: {content}")
            print("   Groq API: WORKING")
        elif resp.status_code == 401:
            print("   Groq error: Invalid API key")
        elif resp.status_code == 429:
            print("   Groq error: Rate limit hit")
        else:
            print(f"   Groq error: {resp.status_code}")
    except Exception as e:
        print(f"   Groq test failed: {e}")
else:
    print("   Skipped - no API key found")

# Check 4: Which provider will llm_generate use?
print("\n4. Checking llm_generate() provider...")
sys.path.insert(0, "D:/AI Research Assistant")
try:
    from agents.llm_helper import (
        get_active_provider,
        GROQ_API_KEY as MODULE_KEY
    )
    print(f"   Active provider: {get_active_provider()}")
    print(f"   GROQ_API_KEY in module: {'SET' if MODULE_KEY else 'NOT SET'}")
except Exception as e:
    print(f"   Error loading llm_helper: {e}")

# Check 5: .env file details
print("\n5. .env file check...")
print(f"   Path: {env_path}")
print(f"   Exists: {env_path.exists()}")
if env_path.exists():
    content = env_path.read_text()
    lines   = content.strip().split("\n")
    print(f"   Lines in file: {len(lines)}")
    for line in lines:
        if "GROQ" in line.upper():
            if "=" in line:
                k, v = line.split("=", 1)
                masked = v[:4] + "..." + v[-4:] if len(v) > 8 else "***"
                print(f"   Found: {k.strip()}={masked}")
            else:
                print(f"   Found line: {line[:30]}")

# Final verdict
print("\n" + "=" * 50)
key = os.getenv("GROQ_API_KEY", "")
if key:
    print("VERDICT: GROQ is active - Ollama NOT in use")
else:
    print("VERDICT: WARNING - GROQ key missing")
    print("ACTION:  Check your .env file location and content")
print("=" * 50)