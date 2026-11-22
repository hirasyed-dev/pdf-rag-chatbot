import httpx
import pytest

from app import config, llm


def response(status, body, method="GET", url="https://example.test"):
    return httpx.Response(status, json=body, request=httpx.Request(method, url))


@pytest.fixture(autouse=True)
def reset_llm(monkeypatch):
    monkeypatch.setattr(llm, "_picked_model", None)
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_MODEL", "")
    monkeypatch.setattr(llm, "_thinking_rejected", False)
    monkeypatch.setattr(config, "LLM_THINKING", "minimal")


def test_pick_model_takes_newest_stable_flash():
    models = [
        {"name": "models/gemini-2.5-flash"},
        {"name": "models/gemini-3.1-flash"},
        {"name": "models/gemini-3.1-flash-lite"},
        {"name": "models/gemini-4.0-flash-preview"},
        {"name": "models/gemini-3.1-pro"},
    ]
    assert llm.pick_model(models) == "gemini-3.1-flash"


def test_pick_model_with_nothing_available_raises():
    with pytest.raises(llm.LLMError):
        llm.pick_model([])


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    with pytest.raises(llm.LLMError, match="LLM_API_KEY"):
        llm.generate("system", "hello")


def test_generate_returns_text_and_sends_key_in_header(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "gemini-x-flash")
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update(method=method, url=url, headers=kwargs["headers"], json=kwargs["json"])
        body = {"candidates": [{"content": {"parts": [{"text": "Hello "}, {"text": "there"}]}}]}
        return response(200, body, method, url)

    monkeypatch.setattr(llm.httpx, "request", fake_request)
    assert llm.generate("be brief", "hi") == "Hello there"
    assert seen["url"].endswith("/models/gemini-x-flash:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "test-key"
    assert seen["json"]["systemInstruction"]["parts"][0]["text"] == "be brief"


def test_model_is_picked_automatically_when_not_set(monkeypatch):
    def fake_request(method, url, **kwargs):
        if method == "GET":
            body = {"models": [
                {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/text-embedding", "supportedGenerationMethods": ["embedContent"]},
            ]}
        else:
            assert "gemini-2.5-flash:generateContent" in url
            body = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        return response(200, body, method, url)

    monkeypatch.setattr(llm.httpx, "request", fake_request)
    assert llm.generate("s", "u") == "ok"


@pytest.mark.parametrize(
    "status, message, expected",
    [
        (400, "API key not valid. Please pass a valid API key.", "rejected the API key"),
        (404, "model not found", "LLM_MODEL"),
        (429, "quota", "limit reached"),
    ],
)
def test_http_errors_become_helpful_messages(monkeypatch, status, message, expected):
    monkeypatch.setattr(config, "LLM_MODEL", "gemini-x-flash")
    monkeypatch.setattr(
        llm.httpx, "request",
        lambda method, url, **kw: response(status, {"error": {"message": message}}, method, url),
    )
    with pytest.raises(llm.LLMError, match=expected):
        llm.generate("s", "u")


def test_blocked_answer_raises(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "gemini-x-flash")
    body = {"promptFeedback": {"blockReason": "SAFETY"}}
    monkeypatch.setattr(
        llm.httpx, "request", lambda method, url, **kw: response(200, body, method, url)
    )
    with pytest.raises(llm.LLMError, match="SAFETY"):
        llm.generate("s", "u")


def test_network_failure_raises(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "gemini-x-flash")

    def boom(method, url, **kwargs):
        raise httpx.ConnectError("no internet")

    monkeypatch.setattr(llm.httpx, "request", boom)
    with pytest.raises(llm.LLMError, match="internet"):
        llm.generate("s", "u")


def test_timeout_has_its_own_message(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "gemini-x-flash")

    def slow(method, url, **kwargs):
        raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(llm.httpx, "request", slow)
    with pytest.raises(llm.LLMError, match="too long"):
        llm.generate("s", "u")


# ---------- streaming ----------

def test_generate_stream_yields_text_pieces(monkeypatch):
    import json
    from contextlib import contextmanager

    monkeypatch.setattr(config, "LLM_API_KEY", "k")
    monkeypatch.setattr(config, "LLM_MODEL", "m")

    class FakeResponse:
        status_code = 200

        def iter_lines(self):
            for text in ("Hel", "lo"):
                yield "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]})
            yield ""

    @contextmanager
    def fake_stream(method, url, **kwargs):
        assert "streamGenerateContent" in url
        yield FakeResponse()

    monkeypatch.setattr(llm.httpx, "stream", fake_stream)
    assert "".join(llm.generate_stream("sys", "hi")) == "Hello"


def test_generate_stream_without_key_raises(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    with pytest.raises(llm.LLMError):
        list(llm.generate_stream("sys", "hi"))


# ---------- thinking setting ----------

def test_generate_sends_thinking_level(monkeypatch):
    sent = []

    def fake_request(method, url, **kwargs):
        sent.append(kwargs["json"])
        return response(200, {"candidates": [{"content": {"parts": [{"text": "ok [S1]"}]}}]})

    monkeypatch.setattr(config, "LLM_MODEL", "m")
    monkeypatch.setattr(llm.httpx, "request", fake_request)
    llm.generate("sys", "hi")
    assert sent[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}


def test_generate_retries_without_thinking_when_model_refuses_it(monkeypatch):
    sent = []

    def fake_request(method, url, **kwargs):
        sent.append(kwargs["json"])
        if "thinkingConfig" in kwargs["json"]["generationConfig"]:
            return response(400, {"error": {"message": "thinking level not supported"}})
        return response(200, {"candidates": [{"content": {"parts": [{"text": "ok [S1]"}]}}]})

    monkeypatch.setattr(config, "LLM_MODEL", "m")
    monkeypatch.setattr(llm.httpx, "request", fake_request)
    assert llm.generate("sys", "hi") == "ok [S1]"
    assert len(sent) == 2
    llm.generate("sys", "again")  # remembered: no third failing request
    assert len(sent) == 3 and "thinkingConfig" not in sent[2]["generationConfig"]


def test_thinking_can_be_switched_off(monkeypatch):
    sent = []

    def fake_request(method, url, **kwargs):
        sent.append(kwargs["json"])
        return response(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    monkeypatch.setattr(config, "LLM_MODEL", "m")
    monkeypatch.setattr(config, "LLM_THINKING", "")
    monkeypatch.setattr(llm.httpx, "request", fake_request)
    llm.generate("sys", "hi")
    assert "thinkingConfig" not in sent[0]["generationConfig"]


def test_stream_retries_without_thinking(monkeypatch):
    import json
    from contextlib import contextmanager

    monkeypatch.setattr(config, "LLM_MODEL", "m")
    seen = []

    class Bad:
        status_code = 400
        text = '{"error": {"message": "thinking level not supported"}}'

        def read(self):
            pass

    class Good:
        status_code = 200

        def iter_lines(self):
            yield "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]})

    @contextmanager
    def fake_stream(method, url, **kwargs):
        seen.append(kwargs["json"]["generationConfig"])
        yield Bad() if "thinkingConfig" in kwargs["json"]["generationConfig"] else Good()

    monkeypatch.setattr(llm.httpx, "stream", fake_stream)
    assert "".join(llm.generate_stream("sys", "hi")) == "Hi"
    assert len(seen) == 2
