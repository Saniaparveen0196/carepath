"""Document knowledge base: chunk -> embed -> store (ingest), and hybrid retrieval."""
import re

import chromadb
from rank_bm25 import BM25Okapi

from .vocab import expand_query
from .workspace import Workspace

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_CHARS = 900
OVERLAP_LINES = 2
SHORT_LINE = 40
COLLECTION = "docs"


def _default_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def _default_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def parse_document(path):
    """Markdown/text file with optional front matter (title:, source:, type:)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = {"title": path.stem.replace("-", " ").replace("_", " ").title(), "type": "document", "source": ""}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
        text = text[m.end():]
    return meta, text


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


def ingest(ws: Workspace, embedder=None, batch=32):
    docs_dir = ws.path / "documents"
    files = sorted(p for p in docs_dir.glob("**/*") if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise ValueError(f"No .md or .txt files in {docs_dir}. Add documents first "
                         f"(or run: python cli.py import-pages {ws.name})")
    ids, texts, metas = [], [], []
    for f in files:
        meta, body = parse_document(f)
        rel = f.relative_to(docs_dir).as_posix()
        header = f"[{meta['type'].upper()}] {meta['title']}"
        for i, chunk in enumerate(chunk_lines(merge_fragments(body))):
            ids.append(f"{rel}:{i}")
            texts.append(f"{header}\n{chunk}")
            metas.append({"type": meta["type"], "title": meta["title"],
                          "url": meta.get("source") or meta.get("url") or "", "file": rel})
    print(f"{len(files)} documents -> {len(texts)} chunks")

    embedder = embedder or _default_embedder()
    embeddings = embedder.encode(texts, batch_size=batch, show_progress_bar=True,
                                 normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(ws.path / "index"))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    B = 500
    for s in range(0, len(ids), B):
        col.add(ids=ids[s:s + B], documents=texts[s:s + B],
                metadatas=metas[s:s + B], embeddings=embeddings[s:s + B])
    return len(files), len(texts)


class Retriever:
    def __init__(self, ws: Workspace, embedder=None, reranker=None):
        client = chromadb.PersistentClient(path=str(ws.path / "index"))
        try:
            self.col = client.get_collection(COLLECTION)
        except Exception:
            raise FileNotFoundError(f"No index for workspace '{ws.name}'. Run: python cli.py ingest {ws.name}")
        self.embedder = embedder or _default_embedder()
        self.reranker = reranker or _default_reranker()
        self.synonyms = ws.config.get("retrieval", {}).get("synonyms", {})
        data = self.col.get(include=["documents", "metadatas"])
        self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
        self.bm25 = BM25Okapi([self._tok(d) for d in self.docs])
        self.idx = {cid: i for i, cid in enumerate(self.ids)}

    @staticmethod
    def _tok(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def retrieve(self, query, k=5, doc_type=None, pool=20):
        query = expand_query(query, self.synonyms)
        pool = min(pool, len(self.ids))
        q = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=[q], n_results=pool,
                             where={"type": doc_type} if doc_type else None)
        dense_ids = res["ids"][0]

        scores = self.bm25.get_scores(self._tok(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        bm_ids = []
        for i in order:
            if doc_type and self.metas[i]["type"] != doc_type:
                continue
            bm_ids.append(self.ids[i])
            if len(bm_ids) == pool:
                break

        fused = {}
        for ranking in (dense_ids, bm_ids):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
        candidates = sorted(fused, key=fused.get, reverse=True)[:pool]

        pairs = [(query, self.docs[self.idx[c]]) for c in candidates]
        rr = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, rr), key=lambda x: x[1], reverse=True)[:k]
        return [{"text": self.docs[self.idx[c]], "meta": self.metas[self.idx[c]], "score": float(s)}
                for c, s in ranked]