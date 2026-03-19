"""
Quick test to verify which LLM provider is active.
Run: python test_llm.py
"""

from dotenv import load_dotenv
load_dotenv()

from agents.llm_helper import (
    llm_generate,
    get_active_provider,
    check_ollama_available,
    GROQ_API_KEY,
    GROQ_PRIMARY
)

print("=" * 50)
print("  LLM Provider Verification")
print("=" * 50)

# Check 1: API key
print(f"\n1. GROQ_API_KEY set: {'✅ YES' if GROQ_API_KEY else '❌ NO'}")
if GROQ_API_KEY:
    masked = GROQ_API_KEY[:8] + "..." + GROQ_API_KEY[-4:]
    print(f"   Key preview: {masked}")

# Check 2: Active provider
print(f"\n2. Active provider: {get_active_provider()}")

# Check 3: LLM available
print(f"\n3. LLM available: {'✅ YES' if check_ollama_available() else '❌ NO'}")

# Check 4: Test actual call
print(f"\n4. Testing actual LLM call...")
print("   Sending: 'Reply with exactly: GROQ_WORKING'")

response = llm_generate(
    "Reply with exactly these words and nothing else: GROQ_WORKING",
    temperature=0.0,
    max_tokens=10
)

print(f"   Response: '{response}'")

if "GROQ_WORKING" in response.upper():
    print("\n✅ SUCCESS — Groq API is working correctly!")
    print(f"   Model: {GROQ_PRIMARY}")
elif "[LLM unavailable" in response:
    print("\n❌ FAILED — LLM not available")
    print("   Check your GROQ_API_KEY in .env")
else:
    print(f"\n⚠️  Got response but unexpected format: {response}")
    print("   Groq is likely working but model responded differently")

print("\n" + "=" * 50)