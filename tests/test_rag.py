import json

import pytest
from fastapi.testclient import TestClient

from app import config, llm, rag
from app.main import app
from tests.helpers import PAGE_ONE, PAGE_TWO, make_pdf

client = TestClient(app)


@pytest.fixture
def indexed(isolated):
    path = isolated / "report.pdf"
    make_pdf(path, [PAGE_ONE, PAGE_TWO])
    with open(path, "rb") as f:
        client.post("/documents", files={"file": ("report.pdf", f, "application/pdf")})
    return isolated


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the real LLM. Set fake_llm.reply to control the answer."""
    class Fake:
        reply = "Quartz appears in the text [S1]."
        prompts = []

    def generate(system_prompt, user_prompt):
        Fake.prompts.append(user_prompt)
        if isinstance(Fake.reply, Exception):
            raise Fake.reply
        return Fake.reply

    Fake.prompts = []
    monkeypatch.setattr(llm, "generate", generate)
    return Fake


def ask(**body):
    body.setdefault("question", "quartz jazz quiz")
    return client.post("/ask", json=body)


# ---------- citation parsing ----------

def test_parse_citations_handles_groups_and_ignores_unknown_labels():
    hits = [
        {"doc_id": "d", "filename": "a.pdf", "page_number": 4, "snippet": "x", "score": 0.9},
        {"doc_id": "d", "filename": "a.pdf", "page_number": 7, "snippet": "y", "score": 0.8},
        {"doc_id": "d", "filename": "b.pdf", "page_number": 2, "snippet": "z", "score": 0.7},
    ]
    citations = rag.parse_citations("A [S3, S1] and B [S2][S9] and again [S1].", hits)
    assert [c["label"] for c in citations] == ["S1", "S2", "S3"]
    assert [c["page_number"] for c in citations] == [4, 7, 2]


# ---------- /ask ----------

def test_answer_has_citation_with_real_page_number(indexed, fake_llm):
    # The LLM wrongly claims "page 2" in its text, but the citation must come from metadata
    fake_llm.reply = "Quartz is on page 2 [S1]."
    body = ask().json()
    assert body["found"] is True
    assert body["citations"][0]["page_number"] == 1
    assert body["citations"][0]["filename"] == "report.pdf"
    assert "timings_ms" in body


def test_model_says_not_found(indexed, fake_llm):
    fake_llm.reply = "Not found in document."
    body = ask().json()
    assert body["answer"] == "Not found in document."
    assert body["found"] is False
    assert body["citations"] == []


def test_answer_without_citation_is_rejected(indexed, fake_llm):
    fake_llm.reply = "Quartz is a mineral."
    body = ask().json()
    assert body["found"] is False
    assert body["reason"] == "answer_had_no_citation"


def test_answer_citing_unknown_source_is_rejected(indexed, fake_llm):
    fake_llm.reply = "Something made up [S9]."
    assert ask().json()["found"] is False


def test_irrelevant_question_never_calls_the_llm(indexed, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 1.01)
    body = ask().json()
    assert body["answer"] == "Not found in document."
    assert body["reason"] == "no_relevant_passages"
    assert fake_llm.prompts == []


def test_empty_library_returns_not_found(isolated, fake_llm):
    body = ask().json()
    assert body["found"] is False
    assert fake_llm.prompts == []


def test_unknown_document_filter_returns_not_found(indexed, fake_llm):
    assert ask(doc_ids=["does-not-exist"]).json()["found"] is False


def test_prompt_contains_sources_with_pages(indexed, fake_llm):
    ask()
    prompt = fake_llm.prompts[0]
    assert "[S1] (report.pdf, page 1)" in prompt
    assert "Question: quartz jazz quiz" in prompt


def test_history_is_added_without_old_citation_markers(indexed, fake_llm):
    history = [
        {"role": "user", "content": "what about quartz?"},
        {"role": "assistant", "content": "It is on page 1 [S1]."},
    ]
    ask(question="and jazz?", history=history)
    prompt = fake_llm.prompts[0]
    assert "User: what about quartz?" in prompt
    assert "Assistant: It is on page 1" in prompt
    assert "page 1 [S1]" not in prompt  # old markers would clash with the new numbering


def test_llm_failure_returns_502(indexed, fake_llm):
    fake_llm.reply = llm.LLMError("Gemini limit reached.")
    response = ask()
    assert response.status_code == 502
    assert "limit reached" in response.json()["detail"]


def test_question_too_short_is_rejected(indexed):
    assert client.post("/ask", json={"question": "a"}).status_code == 422


# ---------- streaming and feedback ----------

def test_stream_sends_tokens_then_checked_final(indexed, monkeypatch):
    monkeypatch.setattr(llm, "generate_stream", lambda s, u: iter(["Quartz ", "is here [S1]."]))
    response = client.post("/ask/stream", json={"question": "quartz jazz quiz"})
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["token", "token", "final"]
    assert events[-1]["found"] is True
    assert events[-1]["citations"][0]["page_number"] == 1
    assert "first_token" in events[-1]["timings_ms"]


def test_stream_without_citation_ends_in_not_found(indexed, monkeypatch):
    monkeypatch.setattr(llm, "generate_stream", lambda s, u: iter(["Made up answer."]))
    response = client.post("/ask/stream", json={"question": "quartz jazz quiz"})
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["answer"] == "Not found in document."


def test_stream_reports_llm_errors_as_event(indexed, monkeypatch):
    def broken(system_prompt, user_prompt):
        raise llm.LLMError("Gemini limit reached.")
        yield  # pragma: no cover

    monkeypatch.setattr(llm, "generate_stream", broken)
    response = client.post("/ask/stream", json={"question": "quartz jazz quiz"})
    assert '"type": "error"' in response.text


def test_feedback_is_saved_and_counted(isolated):
    for rating in ("up", "up", "down"):
        r = client.post("/feedback", json={"question": "q?", "answer": "a [S1]", "rating": rating})
        assert r.status_code == 200
    assert client.get("/feedback/summary").json() == {"up": 2, "down": 1}


def test_feedback_rejects_bad_rating(isolated):
    r = client.post("/feedback", json={"question": "q?", "answer": "a", "rating": "meh"})
    assert r.status_code == 422


# ---------- repair of answers that forgot their citations ----------

def test_answer_without_citation_is_repaired_with_one_retry(indexed, monkeypatch):
    replies = iter(["Quartz is a mineral.", "Quartz appears in the text [S1]."])
    prompts = []

    def generate(system_prompt, user_prompt):
        prompts.append(user_prompt)
        return next(replies)

    monkeypatch.setattr(llm, "generate", generate)
    body = ask().json()
    assert body["found"] is True
    assert body["citations"][0]["page_number"] == 1
    assert len(prompts) == 2 and "had no source citations" in prompts[1]


def test_repair_that_says_not_found_is_respected(indexed, monkeypatch):
    replies = iter(["Some uncited claim.", "Not found in document."])
    monkeypatch.setattr(llm, "generate", lambda s, u: next(replies))
    body = ask().json()
    assert body["found"] is False and body["reason"] == "model_said_not_found"


def test_properly_cited_answer_is_not_retried(indexed, fake_llm):
    fake_llm.reply = "Quartz appears in the text [S1]."
    ask()
    assert len(fake_llm.prompts) == 1


def test_stream_repairs_uncited_answer(indexed, monkeypatch):
    monkeypatch.setattr(llm, "generate_stream", lambda s, u: iter(["Quartz is a mineral."]))
    monkeypatch.setattr(llm, "generate", lambda s, u: "Quartz appears in the text [S1].")
    response = client.post("/ask/stream", json={"question": "quartz jazz quiz"})
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["found"] is True
    assert events[-1]["answer"] == "Quartz appears in the text [S1]."
