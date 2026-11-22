from app import rag, vector_store
from app.keyword_search import BM25, reciprocal_rank_fusion, tokenize


def test_tokenize_drops_stop_words_and_plurals():
    assert tokenize("Which databases are used?") == ["database"]


def test_bm25_ranks_the_exact_word_first():
    docs = ["vector storage with chroma", "sqlite tracks uploaded documents", "streamlit ui"]
    scores = BM25(docs).scores("which metadata database is sqlite")
    assert scores.index(max(scores)) == 1
    assert scores[2] == 0


def test_rrf_prefers_items_ranked_well_in_both_lists():
    assert reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])[0] == "b"


def chunks():
    return [
        {"doc_id": "d", "page_number": 1, "chunk_index": 0, "text": "alpha beta gamma delta"},
        {"doc_id": "d", "page_number": 2, "chunk_index": 1, "text": "sqlite zebra yak"},
        {"doc_id": "d", "page_number": 3, "chunk_index": 2, "text": "alpha alpha beta beta"},
    ]


def test_hybrid_finds_keyword_match_and_keeps_semantic_score(isolated):
    vector_store.add_chunks(chunks(), "a.pdf")
    hits = vector_store.search("sqlite", top_k=1, mode="hybrid")
    assert hits[0]["page_number"] == 2
    assert 0 <= hits[0]["score"] <= 1.0001


def test_semantic_mode_still_works(isolated):
    vector_store.add_chunks(chunks(), "a.pdf")
    hits = vector_store.search("alpha beta", top_k=3, mode="semantic")
    assert {h["page_number"] for h in hits} == {1, 3, 2}


def test_keyword_index_is_refreshed_after_delete(isolated):
    vector_store.add_chunks(chunks(), "a.pdf")
    assert vector_store.search("zebra", top_k=1, mode="hybrid")[0]["page_number"] == 2
    vector_store.delete_document("d")
    assert vector_store.search("zebra", top_k=1, mode="hybrid") == []


def test_identical_text_from_two_copies_uses_one_slot(isolated):
    vector_store.add_chunks(chunks(), "a.pdf")
    copy = [dict(c, doc_id="e") for c in chunks()]
    vector_store.add_chunks(copy, "a-copy.pdf")
    hits = vector_store.search("alpha", top_k=5)
    assert len(hits) == 3  # not 6


def test_failed_question_is_not_used_as_context_for_the_next_one():
    history = [
        {"role": "user", "content": "Who is the CEO of Google?"},
        {"role": "assistant", "content": rag.NOT_FOUND},
    ]
    assert rag._search_text("What about the metadata database?", history) == \
        "What about the metadata database?"


def test_answered_question_is_used_as_context_for_follow_up():
    history = [
        {"role": "user", "content": "Which vector database?"},
        {"role": "assistant", "content": "ChromaDB [S1]."},
    ]
    assert rag._search_text("and the other one?", history) == "Which vector database? and the other one?"
