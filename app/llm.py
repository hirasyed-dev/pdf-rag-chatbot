"""The only file that talks to the LLM provider (Gemini REST API).
Everything else calls generate(), so the provider can be swapped here later."""
import json
import re

import httpx

from app import config

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
SKIP_WORDS = (
    "lite", "preview", "exp", "image", "tts", "live",
    "audio", "thinking", "vision", "embedding",
)

_picked_model: str | None = None
_thinking_rejected = False  # set when the model refuses the thinking setting


class LLMError(Exception):
    """Raised when the LLM service cannot give an answer."""


def _explain(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
    except Exception:
        message = response.text
    message = message[:400]
    code = response.status_code

    if code in (400, 401, 403) and "key" in message.lower():
        return "Gemini rejected the API key. Check LLM_API_KEY in your .env file."
    if code == 404:
        return (
            "Gemini model not found. Set LLM_MODEL in your .env file "
            f"(run check_llm.py to see usable models). Details: {message}"
        )
    if code == 429:
        return (
            "Gemini limit reached. Wait a minute and try again, "
            "or set a different LLM_MODEL in your .env file (each model has its own free quota). "
            f"Details from Google: {message}"
        )
    return f"Gemini error {code}: {message}"


def _call(method: str, path: str, **kwargs) -> dict:
    if not config.LLM_API_KEY:
        raise LLMError("LLM_API_KEY is missing. Add it to your .env file and restart.")
    try:
        response = httpx.request(
            method,
            f"{BASE_URL}{path}",
            headers={"x-goog-api-key": config.LLM_API_KEY},
            timeout=120,
            **kwargs,
        )
    except httpx.TimeoutException as e:
        raise LLMError(
            "Gemini took too long to answer. Try again, or set a faster LLM_MODEL in .env."
        ) from e
    except httpx.HTTPError as e:
        raise LLMError(
            "Could not reach the Gemini service. Check your internet connection."
        ) from e
    if response.status_code != 200:
        raise LLMError(_explain(response))
    return response.json()


def list_models() -> list[dict]:
    """Models this API key can use for generating text."""
    data = _call("GET", "/models", params={"pageSize": 1000})
    return [
        m for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


def pick_model(models: list[dict]) -> str:
    """Newest stable 'gemini-X-flash' model; falls back to any flash model."""
    names = [m["name"].removeprefix("models/") for m in models]

    stable = []
    for name in names:
        match = re.fullmatch(r"gemini-(\d+(?:\.\d+)*)-flash", name)
        if match:
            version = tuple(int(part) for part in match.group(1).split("."))
            stable.append((version, name))
    if stable:
        return max(stable)[1]

    flash = [n for n in names if "flash" in n and not any(w in n for w in SKIP_WORDS)]
    if flash:
        return sorted(flash)[-1]
    if names:
        return names[0]
    raise LLMError("No Gemini models are available for this API key.")


def get_model() -> str:
    global _picked_model
    if config.LLM_MODEL:
        return config.LLM_MODEL
    if _picked_model is None:
        _picked_model = pick_model(list_models())
    return _picked_model


def _body(system_prompt: str, user_prompt: str) -> dict:
    generation_config = {"temperature": 0.0}
    if config.LLM_THINKING and not _thinking_rejected:
        generation_config["thinkingConfig"] = {"thinkingLevel": config.LLM_THINKING}
    return {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": generation_config,
    }


def _is_thinking_error(message: str) -> bool:
    return "think" in message.lower()


def generate(system_prompt: str, user_prompt: str) -> str:
    """Send one prompt and return the answer text."""
    global _thinking_rejected
    path = f"/models/{get_model()}:generateContent"
    try:
        data = _call("POST", path, json=_body(system_prompt, user_prompt))
    except LLMError as e:
        if _thinking_rejected or not config.LLM_THINKING or not _is_thinking_error(str(e)):
            raise
        _thinking_rejected = True  # this model does not take the setting: go without
        data = _call("POST", path, json=_body(system_prompt, user_prompt))

    candidates = data.get("candidates") or []
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
        raise LLMError(f"Gemini returned no answer (reason: {reason}).")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise LLMError("Gemini returned an empty answer.")
    return text


class _ThinkingRejected(Exception):
    """The model refused the thinking setting (nothing was streamed yet)."""


def _stream_once(system_prompt: str, user_prompt: str):
    url = f"{BASE_URL}/models/{get_model()}:streamGenerateContent"
    with httpx.stream(
        "POST", url, params={"alt": "sse"}, json=_body(system_prompt, user_prompt),
        timeout=120, headers={"x-goog-api-key": config.LLM_API_KEY},
    ) as response:
        if response.status_code != 200:
            response.read()
            if (response.status_code == 400 and config.LLM_THINKING
                    and not _thinking_rejected and _is_thinking_error(response.text)):
                raise _ThinkingRejected()
            raise LLMError(_explain(response))
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            try:
                data = json.loads(line[5:].strip())
            except ValueError:
                continue
            candidates = data.get("candidates") or []
            if not candidates:
                continue
            for part in candidates[0].get("content", {}).get("parts", []):
                if part.get("text") and not part.get("thought"):
                    yield part["text"]


def generate_stream(system_prompt: str, user_prompt: str):
    """Like generate(), but yields the answer piece by piece as Gemini writes it."""
    global _thinking_rejected
    if not config.LLM_API_KEY:
        raise LLMError("LLM_API_KEY is missing. Add it to your .env file and restart.")
    produced = False
    try:
        try:
            for piece in _stream_once(system_prompt, user_prompt):
                produced = True
                yield piece
        except _ThinkingRejected:
            _thinking_rejected = True  # this model does not take the setting: go without
            for piece in _stream_once(system_prompt, user_prompt):
                produced = True
                yield piece
    except httpx.TimeoutException as e:
        raise LLMError(
            "Gemini took too long to answer. Try again, or set a faster LLM_MODEL in .env."
        ) from e
    except httpx.HTTPError as e:
        raise LLMError(
            "Could not reach the Gemini service. Check your internet connection."
        ) from e
    if not produced:
        raise LLMError("Gemini returned an empty answer.")
