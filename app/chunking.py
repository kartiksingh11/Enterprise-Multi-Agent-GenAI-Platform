"""
app/chunking.py  (Step 3a: load documents + split into chunks)

CHUNKING STRATEGY — and why it's a real design decision, not a detail
------------------------------------------------------------------------
RAG retrieval quality lives or dies on chunking. Two failure modes:

  - Chunks TOO LARGE: a single chunk covers multiple sub-topics, so a
    vector search for a specific fact returns a chunk that's mostly
    about something else. Relevance drops, and you waste prompt budget
    on irrelevant text.

  - Chunks TOO SMALL: you lose surrounding context. A sentence about
    "16 weeks of paid leave" means nothing without knowing it's about
    PARENTAL leave specifically, if that heading ends up in a
    different chunk.

OUR APPROACH (a simplified version of "recursive splitting"):
  1. Split on paragraph boundaries first (double newline) — paragraphs
     are natural semantic units; we want to keep them intact, not
     sliced mid-word by raw character counting.
  2. Only if a paragraph EXCEEDS the target chunk size, fall back to
     splitting it further (so by character count, with overlap).
  3. Overlap (default 50 chars) between adjacent fallback-split chunks
     so a sentence near a split boundary still appears, in full, in
     at least one chunk.

We hand-roll this instead of importing LangChain's
RecursiveCharacterTextSplitter so the mechanics are fully visible —
this is the same philosophy as hand-writing the SQL schema string in
Step 2: understand it before you let a library hide it.
"""

import os
import glob

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")

CHUNK_SIZE = 300   # target max characters per chunk
CHUNK_OVERLAP = 50  # overlap between fallback character-split chunks

# A paragraph only marginally over CHUNK_SIZE doesn't need splitting —
# we discovered (by actually running this and inspecting output) that
# a 305-char paragraph against a 300-char limit produced a near-
# duplicate 55-char trailing chunk that was pure redundant overlap,
# adding zero new retrievable information. Allowing some slack avoids
# fragmenting paragraphs that are only barely over the threshold.
CHUNK_SIZE_TOLERANCE = 1.3  # allow up to 30% over CHUNK_SIZE before splitting

# Very short chunks (e.g. a one-line document title) carry little
# standalone meaning and just add noise to the vector index without
# adding retrievable value. We filter them out rather than indexing
# them. Tradeoff: a literal "what is this document titled?" question
# wouldn't retrieve well — acceptable for a fact-retrieval use case.
MIN_CHUNK_LENGTH = 40


def load_documents() -> list[dict]:
    """
    Read all .txt files from data/docs/ and return them as a list of
    {"source": filename, "text": full_text} dicts.

    Returns a list (not a dict) because we want to preserve the
    ability to have duplicate-named sources later, and lists are
    simpler to iterate when building chunk metadata.
    """
    docs = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": os.path.basename(path), "text": text})
    return docs


def _split_long_paragraph(paragraph: str) -> list[str]:
    """
    Fallback splitter for a paragraph that exceeds CHUNK_SIZE on its own.
    Slides a window of CHUNK_SIZE characters with CHUNK_OVERLAP overlap.

    None of our 4 sample docs actually have a paragraph this long, but
    this function exists so the chunker doesn't silently produce a
    giant out-of-spec chunk if a future document does have one —
    a small piece of defensive engineering, same spirit as Step 2's
    SQL safety checks.
    """
    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    while start < len(paragraph):
        end = start + CHUNK_SIZE
        chunks.append(paragraph[start:end])
        if end >= len(paragraph):
            break
        start += step
    return chunks


def chunk_document(doc: dict) -> list[dict]:
    """
    Split a single document into chunks, preferring paragraph
    boundaries, falling back to character-window splitting only for
    paragraphs that exceed CHUNK_SIZE.

    Returns a list of {"source": ..., "chunk_index": ..., "text": ...}
    dicts — keeping "source" on every chunk is what lets us later
    cite WHICH document an answer came from (we'll use this in 3d).
    """
    paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        if len(para) <= CHUNK_SIZE * CHUNK_SIZE_TOLERANCE:
            chunks.append(para)
        else:
            chunks.extend(_split_long_paragraph(para))

    # Filter out near-empty/title-only fragments (see MIN_CHUNK_LENGTH comment above)
    chunks = [c for c in chunks if len(c) >= MIN_CHUNK_LENGTH]

    return [
        {"source": doc["source"], "chunk_index": i, "text": text}
        for i, text in enumerate(chunks)
    ]


def chunk_all_documents() -> list[dict]:
    """Load every document in data/docs/ and chunk all of them."""
    all_chunks = []
    for doc in load_documents():
        all_chunks.extend(chunk_document(doc))
    return all_chunks


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.chunking
    chunks = chunk_all_documents()
    print(f"Total chunks across all documents: {len(chunks)}\n")
    for c in chunks:
        print(f"[{c['source']} #{c['chunk_index']}] ({len(c['text'])} chars)")
        print(c["text"][:120].replace("\n", " ") + ("..." if len(c["text"]) > 120 else ""))
        print("-" * 60)
