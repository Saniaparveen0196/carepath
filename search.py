"""
Test retrieval.

Usage:
    python search.py "cost of knee replacement in Bangalore"
    python search.py "best heart surgeon" --type doctor
"""
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("--type", choices=["treatment", "hospital", "doctor", "department"])
ap.add_argument("-k", type=int, default=5)
args = ap.parse_args()

model = SentenceTransformer(EMBED_MODEL)
col = chromadb.PersistentClient(path="data/chroma").get_collection("qurovo")

q = model.encode(QUERY_PREFIX + args.query, normalize_embeddings=True).tolist()
res = col.query(
    query_embeddings=[q],
    n_results=args.k,
    where={"type": args.type} if args.type else None,
)

for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"\n[{1 - dist:.3f}] {meta['type']} | {meta['title']}")
    print(f"        {meta['url']}")
    print("        " + doc.replace("\n", " ")[:250] + "...")