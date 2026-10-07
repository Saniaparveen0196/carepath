"""
Diagnose where the real "Top Hospitals" list lives in a treatment page's HTML.
Usage: python diagnose_links.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

hospitals = json.loads(Path("data/hospitals.json").read_text(encoding="utf-8"))
raw_path = Path("data/raw/treatment_knee_replacement.html")
if not raw_path.exists():
    print("File not found. Files in data/raw that mention 'knee':")
    for p in Path("data/raw").glob("*knee*"):
        print("  ", p.name)
    sys.exit()

raw = raw_path.read_text(encoding="utf-8", errors="ignore")
print("raw html length:", len(raw))

# 1. How many times does each hospital slug appear?
counts = {h["slug"]: raw.count(h["slug"]) for h in hospitals}
print("\n[1] Occurrence-count distribution (times_seen: how_many_hospitals):",
      dict(sorted(Counter(counts.values()).items())))
print("    Top 12 most frequent:")
for slug, n in sorted(counts.items(), key=lambda x: -x[1])[:12]:
    city = next(h["city"] for h in hospitals if h["slug"] == slug)
    print(f"      {n:3d}  {city:10s}  {slug}")
print("    Hospitals appearing exactly once or twice:",
      sum(1 for n in counts.values() if n in (1, 2)))

# 2. Context around a hospital we KNOW is on the Bangalore list (from your screenshot)
needle = "manipal-hospital-whitefield"
i = raw.find(needle)
print(f"\n[2] First '{needle}' at position {i}")
if i >= 0:
    print(raw[max(0, i - 250): i + 350].replace("\n", " "))

# 3. Context around a hospital that is surely NOT a knee-replacement hospital
far = next((h for h in hospitals if h["city"] == "Patna"), None)
if far:
    i = raw.find(far["slug"])
    print(f"\n[3] First Patna slug '{far['slug']}' at position {i}")
    if i >= 0:
        print(raw[max(0, i - 250): i + 250].replace("\n", " "))

# 4. Script payloads that mention Bangalore hospitals
scripts = re.findall(r"<script[^>]*>(.*?)</script>", raw, re.S)
print(f"\n[4] script tags: {len(scripts)}")
for j, s in enumerate(scripts):
    if "whitefield" in s.lower():
        print(f"    script #{j}: {len(s)} chars, starts: {s[:160]!r}")

# 5. Is there a city field next to hospital names in any JSON-like payload?
for m in list(re.finditer(r'"city"\s*:\s*"([^"]+)"', raw))[:5]:
    print(f"\n[5] found a city field: {m.group(0)}  ... context:")
    print("    " + raw[max(0, m.start() - 150): m.end() + 150].replace("\n", " "))

# 6. The dropdown: does the HTML contain the city options?
opts = re.findall(r"<option[^>]*>([^<]+)</option>", raw)
print(f"\n[6] <option> values in the page: {opts[:20]}")