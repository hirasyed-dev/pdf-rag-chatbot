from app import vector_store


def make_chunks(doc_id):
    return [
        {"doc_id": doc_id, "page_number": 1, "chunk_index": 0, "text": "quartz jazz quiz quick"},
        {"doc_id": doc_id, "page_number": 2, "chunk_index": 1, "text": "mom bob pop bubble pump"},
        {"doc_id": doc_id, "page_number": 3, "chunk_index": 2, "text": "quartz exquisite jinx quiz"},
    ]


def test_search_returns_best_page_first(isolated):
    vector_store.add_chunks(make_chunks("docA"), "a.pdf")
    hits = vector_store.search("quartz jazz", top_k=3)
    assert hits[0]["page_number"] == 1
    assert hits[0]["filename"] == "a.pdf"
    assert hits[0]["score"] >= hits[-1]["score"]


def test_search_empty_store_returns_nothing(isolated):
    assert vector_store.search("anything") == []


def test_filter_by_page(isolated):
    vector_store.add_chunks(make_chunks("docA"), "a.pdf")
    hits = vector_store.search("quartz", top_k=5, page_number=3)
    assert len(hits) == 1
    assert hits[0]["page_number"] == 3


def test_filter_by_document(isolated):
    vector_store.add_chunks(make_chunks("docA"), "a.pdf")
    vector_store.add_chunks(make_chunks("docB"), "b.pdf")
    hits = vector_store.search("quartz", top_k=10, doc_ids=["docB"])
    assert hits and all(h["doc_id"] == "docB" for h in hits)


def test_delete_removes_only_that_document(isolated):
    vector_store.add_chunks(make_chunks("docA"), "a.pdf")
    vector_store.add_chunks(make_chunks("docB"), "b.pdf")
    assert vector_store.delete_document("docA") == 3
    assert not vector_store.doc_exists("docA")
    assert vector_store.doc_exists("docB")
