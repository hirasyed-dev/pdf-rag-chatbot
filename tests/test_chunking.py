from app.chunking import chunk_pages, estimate_tokens, MAX_TOKENS


def make_page(number, word_count):
    return {"page_number": number, "text": " ".join(f"word{i}" for i in range(word_count))}


def test_chunk_keeps_page_number():
    pages = [make_page(1, 100), make_page(2, 100)]
    chunks = chunk_pages(pages, doc_id="doc1")
    assert {c["page_number"] for c in chunks} == {1, 2}


def test_chunks_never_cross_pages():
    pages = [
        {"page_number": 1, "text": "alpha " * 800},
        {"page_number": 2, "text": "beta " * 800},
    ]
    chunks = chunk_pages(pages, doc_id="doc1")
    for c in chunks:
        if c["page_number"] == 1:
            assert "beta" not in c["text"]
        if c["page_number"] == 2:
            assert "alpha" not in c["text"]


def test_long_page_is_split_within_limit():
    chunks = chunk_pages([make_page(1, 1500)], doc_id="doc1")
    assert len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c["text"]) <= MAX_TOKENS + 5


def test_empty_page_is_skipped():
    pages = [{"page_number": 1, "text": ""}, make_page(2, 50)]
    chunks = chunk_pages(pages, doc_id="doc1")
    assert all(c["page_number"] == 2 for c in chunks)


def test_chunk_has_required_metadata():
    chunks = chunk_pages([make_page(3, 50)], doc_id="abc")
    c = chunks[0]
    assert c["doc_id"] == "abc"
    assert c["page_number"] == 3
    assert c["chunk_index"] == 0
