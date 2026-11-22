"""Tries several Gemini models with your key and shows which ones answer, and how fast.
Put this file in the project folder (next to check_llm.py) and run:  python check_models.py
"""
import time

import httpx

from app import config, llm

CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]
WAIT_SECONDS = 90

if not config.LLM_API_KEY:
    raise SystemExit("LLM_API_KEY is empty. Add it to your .env file and save it.")

body = {
    "contents": [{"role": "user", "parts": [{"text": "Reply with one word: ready"}]}],
    "generationConfig": {"temperature": 0.0},
}

print(f"Trying {len(CANDIDATES)} models (each waits up to {WAIT_SECONDS} seconds)...\n")
working = []
for model in CANDIDATES:
    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{llm.BASE_URL}/models/{model}:generateContent",
            headers={"x-goog-api-key": config.LLM_API_KEY},
            json=body,
            timeout=WAIT_SECONDS,
        )
    except httpx.TimeoutException:
        print(f"  {model:28} TIMEOUT after {WAIT_SECONDS}s")
        continue
    except httpx.HTTPError as e:
        print(f"  {model:28} NETWORK ERROR: {type(e).__name__}")
        continue

    seconds = time.perf_counter() - started
    if response.status_code == 200:
        try:
            parts = response.json()["candidates"][0]["content"]["parts"]
            reply = "".join(p.get("text", "") for p in parts).strip()[:30]
        except (KeyError, IndexError):
            reply = "(empty reply)"
        print(f"  {model:28} OK in {seconds:.1f}s -> {reply!r}")
        working.append((seconds, model))
    else:
        print(f"  {model:28} FAILED: {llm._explain(response)[:110]}")

print()
if working:
    # Prefer a full (non-lite) model that answers quickly; a model that needs
    # more than 10 seconds for one word is too slow for a chat app.
    quick_full = sorted((s, m) for s, m in working if "lite" not in m and s <= 10)
    quick_any = sorted((s, m) for s, m in working if s <= 10)
    if quick_full:
        best = quick_full[0][1]
    elif quick_any:
        best = quick_any[0][1]
    else:
        best = min(working)[1]
    print("Recommended model:", best)
    print("Put this line in your .env file, save it, then run python check_llm.py:")
    print(f"LLM_MODEL={best}")
else:
    print("No model answered. Copy this whole output and send it to me.")
