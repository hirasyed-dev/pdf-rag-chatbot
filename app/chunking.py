"""
Page-bound chunking.
Every chunk belongs to exactly ONE page, so citations are always accurate.
"""

TOKENS_PER_WORD = 1.3      # rough estimate: 1 word is about 1.3 tokens
MAX_TOKENS = 450           # inside the 300-500 token requirement
OVERLAP_TOKENS = 50        # small overlap so context isn't cut mid-idea

MAX_WORDS = int(MAX_TOKENS / TOKENS_PER_WORD)          # about 346 words
OVERLAP_WORDS = int(OVERLAP_TOKENS / TOKENS_PER_WORD)  # about 38 words


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


def chunk_pages(pages: list[dict], doc_id: str) -> list[dict]:
    """
    Input:  [{"page_number": 1, "text": "..."}, ...]
    Output: [{"doc_id": ..., "page_number": 1, "chunk_index": 0, "text": "..."}, ...]
    A chunk NEVER contains text from two different pages.
    """
    chunks = []
    chunk_index = 0
    step = MAX_WORDS - OVERLAP_WORDS

    for page in pages:
        words = page["text"].split()
        if not words:
            continue  # blank page, nothing to index

        start = 0
        while start < len(words):
            piece = words[start:start + MAX_WORDS]
            chunks.append({
                "doc_id": doc_id,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": " ".join(piece),
            })
            chunk_index += 1

            if start + MAX_WORDS >= len(words):
                break  # reached the end of this page
            start += step

    return chunks
