"""Smoke test of the Streamlit page with a fake backend."""
from pathlib import Path

import json
from contextlib import contextmanager

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_FILE = str(Path(__file__).resolve().parent.parent / "frontend" / "streamlit_app.py")

DOCUMENT = {
    "doc_id": "abc", "filename": "report.pdf", "page_count": 5,
    "chunk_count": 5, "source": "upload", "status": "indexed", "created_at": "now",
}
CITATION = {
    "label": "S1", "doc_id": "abc", "filename": "report.pdf",
    "page_number": 3, "snippet": "some supporting text", "score": 0.8,
}


def make_stream(sent, feedback=None):
    """Fake for httpx.stream: sends two token events and then the final event."""
    class FakeStream:
        status_code = 200

        def iter_lines(self):
            final = {"type": "final", "answer": "The answer is 42 [S1].", "found": True,
                     "reason": "ok", "citations": [CITATION], "timings_ms": {}}
            for event in ({"type": "token", "text": "The answer "},
                          {"type": "token", "text": "is 42 [S1]."}, final):
                yield "data: " + json.dumps(event)

    @contextmanager
    def fake_stream(method, url, **kwargs):
        sent.append(kwargs["json"])
        yield FakeStream()

    return fake_stream


def make_backend(sent):
    def fake_request(method, url, **kwargs):
        if url.endswith("/documents") and method == "GET":
            body = {"documents": [DOCUMENT]}
        else:
            body = {}
        return httpx.Response(200, json=body, request=httpx.Request(method, url))
    return fake_request


def test_page_loads_and_lists_documents(monkeypatch):
    monkeypatch.setattr(httpx, "request", make_backend([]))
    app = AppTest.from_file(APP_FILE, default_timeout=15).run()
    assert not app.exception
    assert any("report.pdf" in element.value for element in app.sidebar.markdown)


def test_asking_a_question_shows_answer_and_sources(monkeypatch):
    sent = []
    monkeypatch.setattr(httpx, "request", make_backend(sent))
    monkeypatch.setattr(httpx, "stream", make_stream(sent))
    app = AppTest.from_file(APP_FILE, default_timeout=15).run()
    app.chat_input[0].set_value("What is the answer?").run()

    assert not app.exception
    assert sent[0]["question"] == "What is the answer?"
    assert sent[0]["history"] == []
    shown = " ".join(m.value for m in app.markdown)
    assert "The answer is 42" in shown
    assert len(app.expander) == 1


def test_thumbs_up_sends_feedback(monkeypatch):
    sent, feedback = [], []
    monkeypatch.setattr(httpx, "request", make_backend(sent))
    monkeypatch.setattr(httpx, "stream", make_stream(sent))

    def fake_post(url, **kwargs):
        feedback.append(kwargs["json"])
        return httpx.Response(200, json={"saved": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    app = AppTest.from_file(APP_FILE, default_timeout=15).run()
    app.chat_input[0].set_value("What is the answer?").run()
    thumbs_up = [b for b in app.button if b.key and b.key.startswith("up-")]
    thumbs_up[0].click().run()

    assert feedback == [{"question": "What is the answer?",
                         "answer": "The answer is 42 [S1].", "rating": "up"}]
    assert not app.exception
