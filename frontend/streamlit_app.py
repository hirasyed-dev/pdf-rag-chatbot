"""Streamlit front end. Run with:  streamlit run frontend/streamlit_app.py
(the FastAPI backend must be running too)."""
import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄", layout="wide")


def call_api(method: str, path: str, timeout: int = 60, **kwargs):
    try:
        return httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except httpx.HTTPError:
        st.error("Cannot reach the backend. Start it first with: uvicorn app.main:app --reload")
        st.stop()


def error_text(response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def show_citations(citations) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            st.markdown(f"**[{c['label']}] {c['filename']}, page {c['page_number']}**")
            st.caption(c["snippet"])


def send_feedback(index: int, rating: str) -> None:
    """Called when a thumb is clicked. Saves the rating and remembers it in the chat."""
    messages = st.session_state.messages
    answer = messages[index]
    question = messages[index - 1]["content"] if index > 0 else ""
    response = httpx.post(
        f"{API_URL}/feedback", timeout=30,
        json={"question": question, "answer": answer["content"], "rating": rating},
    )
    if response.status_code == 200:
        answer["rating"] = rating


def show_feedback(index: int, message: dict) -> None:
    if not message.get("citations"):
        return  # nothing to rate on "Not found" replies
    if message.get("rating"):
        st.caption("Thanks for your feedback " + ("👍" if message["rating"] == "up" else "👎"))
        return
    up, down, _ = st.columns([1, 1, 10])
    up.button("👍", key=f"up-{index}", on_click=send_feedback, args=(index, "up"))
    down.button("👎", key=f"down-{index}", on_click=send_feedback, args=(index, "down"))


def stream_answer(payload: dict, box: dict):
    """Yield answer text as it arrives; the checked final answer is stored in box."""
    try:
        with httpx.stream("POST", f"{API_URL}/ask/stream", json=payload, timeout=180) as response:
            if response.status_code != 200:
                response.read()
                box["error"] = error_text(response)
                return
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                if event["type"] == "token":
                    yield event["text"]
                elif event["type"] == "final":
                    box["final"] = event
                elif event["type"] == "error":
                    box["error"] = event["detail"]
    except httpx.HTTPError:
        box["error"] = "Cannot reach the backend. Start it first with: uvicorn app.main:app --reload"


st.title("📄 AI PDF Chatbot")
st.caption("Ask questions about your PDFs. Every answer shows the exact pages it came from.")

# ---------- Sidebar: documents ----------
with st.sidebar:
    st.header("Documents")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None and st.button("Upload and index", type="primary"):
        with st.spinner("Reading and indexing the PDF..."):
            response = call_api(
                "POST", "/documents", timeout=900,
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            )
        if response.status_code == 200:
            result = response.json()
            if result["status"] == "already_indexed":
                st.info("This PDF was already indexed.")
            else:
                st.success(f"Indexed {result['page_count']} pages ({result['chunk_count']} chunks).")
        else:
            st.error(error_text(response))

    if st.button("Index existing folder"):
        with st.spinner("Indexing the folder..."):
            response = call_api("POST", "/index-folder", timeout=1800)
        if response.status_code == 200:
            results = response.json()["results"]
            new = sum(1 for r in results if r["status"] == "indexed")
            skipped = sum(1 for r in results if r["status"] == "skipped")
            st.success(f"{new} new, {len(results) - new - skipped} already indexed, {skipped} skipped.")
        else:
            st.error(error_text(response))

    documents = call_api("GET", "/documents").json()["documents"]
    st.subheader(f"Indexed ({len(documents)})")
    if not documents:
        st.info("No documents yet. Upload a PDF to start.")
    name_counts = {}
    for doc in documents:
        name_counts[doc["filename"]] = name_counts.get(doc["filename"], 0) + 1
    for doc in documents:
        st.write(f"**{doc['filename']}**")
        st.caption(f"{doc['page_count']} pages, {doc['chunk_count']} chunks · id {doc['doc_id'][:6]}")
        if name_counts[doc["filename"]] > 1:
            st.warning("Another file has this same name but different content.")
        if st.button("Delete", key=f"delete-{doc['doc_id']}"):
            call_api("DELETE", f"/documents/{doc['doc_id']}")
            st.rerun()

    names = {d["doc_id"]: f"{d['filename']} ({d['doc_id'][:6]})" for d in documents}
    selected = st.multiselect(
        "Ask only these documents (empty = all)",
        options=list(names),
        format_func=lambda doc_id: names[doc_id],
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main: chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_citations(message.get("citations"))
        if message["role"] == "assistant":
            show_feedback(index, message)

question = st.chat_input("Ask a question about your documents")
if question:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-6:]
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload = {"question": question, "doc_ids": selected or None, "history": history}
    with st.chat_message("assistant"):
        box = {}
        placeholder = st.empty()
        with st.spinner("Searching the documents..."):
            streamed = ""
            for piece in stream_answer(payload, box):
                streamed += piece
                placeholder.markdown(streamed + "▌")
        if "final" in box:
            result = box["final"]
            placeholder.markdown(result["answer"])  # the checked answer replaces the draft
            show_citations(result["citations"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "citations": result["citations"]}
            )
            st.rerun()  # redraw so the thumbs buttons appear
        else:
            placeholder.empty()
            st.error(box.get("error", "Something went wrong. Please try again."))
