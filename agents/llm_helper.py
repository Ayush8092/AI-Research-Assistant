"""
LLM Helper — Groq API (cloud) with Ollama fallback (local).

Priority:
  1. Groq API (if GROQ_API_KEY is set in .env)
  2. Ollama (if Groq key not set — local fallback)

Groq is:
  - Free tier available
  - Much faster than local Ollama (~3-5s vs 45-60s)
  - Better quality than phi3:mini / tinyllama
"""

import logging
import os
import httpx

# Load .env file
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── Groq Config ──────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_BASE      = "https://api.groq.com/openai/v1"
GROQ_PRIMARY   = "llama-3.1-8b-instant"
GROQ_FALLBACK  = "gemma2-9b-it"

# ── Ollama Config (local fallback) ───────────────────────────────────
OLLAMA_BASE    = "http://localhost:11434"
OLLAMA_PRIMARY = "phi3:mini"
OLLAMA_FALLBACK= "tinyllama"


def llm_generate(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2000
) -> str:
    """
    Generate text using best available LLM.
    Tries Groq first (if key set), falls back to Ollama.
    """
    # Try Groq first
    if GROQ_API_KEY:
        result = _call_groq(prompt, temperature, max_tokens)
        if result:
            return result
        logger.warning("[LLM] Groq failed. Trying Ollama fallback...")

    # Fall back to Ollama
    result = _call_ollama(prompt, temperature, max_tokens)
    if result:
        return result

    return "[LLM unavailable — set GROQ_API_KEY or run ollama serve]"


def _call_groq(
    prompt: str,
    temperature: float,
    max_tokens: int
) -> str:
    """Call Groq API."""
    for model in [GROQ_PRIMARY, GROQ_FALLBACK]:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json"
            }
            payload = {
                "model":       model,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens":  max_tokens
            }
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    f"{GROQ_BASE}/chat/completions",
                    headers=headers,
                    json=payload
                )
                if resp.status_code == 200:
                    content = (
                        resp.json()["choices"][0]["message"]["content"]
                        .strip()
                    )
                    logger.debug(
                        f"[LLM] Groq ({model}): {len(content)} chars"
                    )
                    return content
                elif resp.status_code == 429:
                    logger.warning(
                        f"[LLM] Groq rate limit hit on {model}. "
                        f"Trying next model..."
                    )
                else:
                    logger.warning(
                        f"[LLM] Groq {resp.status_code} on {model}: "
                        f"{resp.text[:80]}"
                    )
        except Exception as e:
            logger.warning(f"[LLM] Groq error ({model}): {e}")

    return None


def _call_ollama(
    prompt: str,
    temperature: float,
    max_tokens: int
) -> str:
    """Call local Ollama server."""
    for model in [OLLAMA_PRIMARY, OLLAMA_FALLBACK]:
        try:
            payload = {
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx":     2048
                }
            }
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json=payload
                )
                if resp.status_code == 200:
                    content = resp.json().get("response","").strip()
                    if content:
                        logger.debug(
                            f"[LLM] Ollama ({model}): "
                            f"{len(content)} chars"
                        )
                        return content
        except Exception as e:
            logger.debug(f"[LLM] Ollama error ({model}): {e}")

    return None


def check_ollama_available() -> bool:
    """
    Returns True if any LLM is available.
    Checks Groq key first, then Ollama server.
    """
    if GROQ_API_KEY:
        return True
    # Check Ollama
    try:
        with httpx.Client(timeout=3) as client:
            resp = client.get(f"{OLLAMA_BASE}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def list_available_models() -> list[str]:
    """Return list of available models."""
    models = []
    if GROQ_API_KEY:
        models.extend([f"groq/{GROQ_PRIMARY}", f"groq/{GROQ_FALLBACK}"])
    # Also check Ollama
    try:
        with httpx.Client(timeout=3) as client:
            resp = client.get(f"{OLLAMA_BASE}/api/tags")
            if resp.status_code == 200:
                ollama_models = [
                    m["name"]
                    for m in resp.json().get("models", [])
                ]
                models.extend(ollama_models)
    except Exception:
        pass
    return models


def get_active_provider() -> str:
    """Return which LLM provider is currently active."""
    if GROQ_API_KEY:
        return f"Groq ({GROQ_PRIMARY})"
    try:
        with httpx.Client(timeout=3) as client:
            resp = client.get(f"{OLLAMA_BASE}/api/tags")
            if resp.status_code == 200:
                return f"Ollama ({OLLAMA_PRIMARY})"
    except Exception:
        pass
    return "None — set GROQ_API_KEY or run ollama serve"