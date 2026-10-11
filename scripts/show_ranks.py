"""List the retrieval_hard cases where the right page was NOT ranked first, from the last eval run.
Usage: python scripts/show_ranks.py demo [report.json]
"""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ws = sys.argv[1] if len(sys.argv) > 1 else "demo"
path = sys.argv[2] if len(sys.argv) > 2 else f"workspaces/{ws}/eval/reports/latest.json"
rows = json.load(open(path, encoding="utf-8"))["results"].get("retrieval_hard", [])
off = []
for r in rows:
    m = re.match(r"rank=(\d+)", r["detail"])
    rank = int(m.group(1)) if m else None
    if rank != 1:
        off.append((r["id"], rank, r["q"]))
print(f"{len(rows)} cases, {len(rows) - len(off)} ranked first")
for cid, rank, q in off:
    print(f"  rank {rank if rank else 'MISS':>4}  {cid:34s} {q[:70]}")