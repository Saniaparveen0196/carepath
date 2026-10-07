import json
from collections import Counter
from pathlib import Path

docs = [json.loads(l) for l in Path("data/pages.jsonl").read_text(encoding="utf-8").splitlines()]
print("Total docs:", len(docs))
print("By type:", Counter(d["type"] for d in docs))
lengths = sorted(len(d["text"]) for d in docs)
print(f"Text length chars: min={lengths[0]}, median={lengths[len(lengths)//2]}, max={lengths[-1]}")

for t in ("treatment", "hospital", "doctor"):
    sample = next((d for d in docs if d["type"] == t), None)
    if sample:
        print(f"\n--- sample {t}: {sample['title']} ({sample['url']}) ---")
        print(sample["text"][:600])