"""Show invisible / unusual characters in the saved eval answers (explains glued words like 'withthat')."""
import json
import sys
import unicodedata
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
ws = sys.argv[1] if len(sys.argv) > 1 else "demo"
report = json.load(open(f"workspaces/{ws}/eval/reports/latest.json", encoding="utf-8"))
seen = Counter()
for rows in report["results"].values():
    for r in rows:
        for ch in r.get("answer", ""):
            if ord(ch) > 127 and unicodedata.category(ch) in ("Cf", "Zs", "Cc", "Zl", "Zp"):
                seen[(f"U+{ord(ch):04X}", unicodedata.name(ch, "?"))] += 1
print("unusual characters found in answers:" if seen else "none found")
for (cp, name), n in seen.most_common():
    print(f"  {cp}  {name}  x{n}")