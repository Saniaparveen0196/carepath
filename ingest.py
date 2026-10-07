"""
Step 2: Chunk pages, embed them, and store in a local Chroma vector DB.

Usage:
    python ingest.py
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("data/pages.jsonl")
CHROMA_DIR = "data/chroma"
COLLECTION = "qurovo"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_CHARS = 900       # target chunk size
OVERLAP_LINES = 2       # lines repeated between chunks for context
SHORT_LINE = 40         # lines shorter than this get merged together


def merge_fragments(text):
    """Merge short fragment lines ('4.7', '(', 'reviews)') into readable lines."""
    out, buf = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) < SHORT_LINE:
            buf.append(line)
            if len(" ".join(buf)) > 120:
                out.append(" | ".join(buf))
                buf = []
        else:
            if buf:
                out.append(" | ".join(buf))
                buf = []
            out.append(line)
    if buf:
        out.append(" | ".join(buf))
    return out


def chunk_lines(lines):
    chunks, cur, size = [], [], 0
    for line in lines:
        cur.append(line)
        size += len(line) + 1
        if size >= CHUNK_CHARS:
            chunks.append("\n".join(cur))
            cur = cur[-OVERLAP_LINES:]
            size = sum(len(l) + 1 for l in cur)
    if cur and (not chunks or len("\n".join(cur)) > 150):
        chunks.append("\n".join(cur))
    return chunks


def main():
    docs = [json.loads(l) for l in DATA_FILE.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(docs)} documents")

    ids, texts, metas = [], [], []
    for d in docs:
        header = f"[{d['type'].upper()}] {d['title']}"
        for i, chunk in enumerate(chunk_lines(merge_fragments(d["text"]))):
            ids.append(f"{d['type']}:{d['slug']}:{i}")
            texts.append(f"{header}\n{chunk}")  # header gives each chunk context
            metas.append({
                "type": d["type"],
                "title": d["title"],
                "url": d["url"],
                "slug": d["slug"],
                "chunk": i,
            })
    print(f"Created {len(texts)} chunks")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch each run
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    B = 500
    for s in range(0, len(ids), B):
        col.add(
            ids=ids[s:s + B],
            documents=texts[s:s + B],
            metadatas=metas[s:s + B],
            embeddings=embeddings[s:s + B],
        )
    print(f"Stored {col.count()} chunks in {CHROMA_DIR}")


if __name__ == "__main__":
    main()