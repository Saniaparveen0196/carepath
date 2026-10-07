"""
Step 4a: Turn crawled pages into structured tables.
Usage: python build_tables.py
Writes data/hospitals.json, doctors.json, treatments.json
"""
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path("data")
CITY_ALIASES = {
    "bangalore": "Bengaluru", "gurgaon": "Gurugram", "bombay": "Mumbai",
    "new delhi": "Delhi", "newdelhi": "Delhi",
}
GENERIC_SLUG_TOKENS = {
    "hospital", "hospitals", "centre", "center", "institute", "clinic",
    "medicity", "research", "care", "city", "speciality", "superspeciality",
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def to_int(s):
    return int(s.replace(",", "")) if s else None


def lines_of(d):
    return [l.strip() for l in d["text"].splitlines() if l.strip()]


def after(lines, label):
    for i, l in enumerate(lines):
        if l.lower().rstrip(":") == label.lower() and i + 1 < len(lines):
            return lines[i + 1]
    return None


def before(lines, label):
    for i, l in enumerate(lines):
        if l.lower() == label.lower() and i > 0:
            return lines[i - 1]
    return None


# ---------------- city derivation ----------------
def canon_city(name):
    name = re.sub(r"[\d\s]+$", "", (name or "").strip()).strip()
    return CITY_ALIASES.get(name.lower(), name.title())


def slug_city(slug):
    """City from the last slug token, or None if that token is a generic word."""
    last = slug.split("-")[-1].lower()
    if last in GENERIC_SLUG_TOKENS:
        return None
    return canon_city(last)


def address_city(address, known_cities):
    """City from the address: prefer a known city, else the part before the state."""
    parts = [p.strip() for p in (address or "").split(",") if p.strip()]
    if parts and parts[-1].lower() == "india":
        parts = parts[:-1]
    for part in reversed(parts[-3:]):
        c = canon_city(part)
        if c in known_cities:
            return c
    if len(parts) >= 2:
        return canon_city(parts[-2])
    return None


# ---------------- parsers ----------------
def parse_hospital(d):
    text, title = d["text"], d["title"]
    lines = lines_of(d)
    idx = text.find(title)
    region = text[:idx] if idx > 0 else text[:200]
    accred = [a for a in ("NABH", "NABL", "JCI") if a in region]

    head = text[:900]
    m = re.search(r"(\d(?:\.\d)?)\s*\(\s*([\d,]+)\s*reviews?\s*\)", head, re.S)
    beds = re.search(r"([\d,]+)\s*\+?\s*Beds", head)
    docs = re.search(r"([\d,]+)\s*\+?\s*Doctors", head)

    address = None
    for i, l in enumerate(lines):
        if l == title and i + 1 < len(lines):
            nxt = lines[i + 1]
            if re.search(r"\d{6}", nxt) or nxt.count(",") >= 2:
                address = nxt
            break

    return {
        "slug": d["slug"], "name": title, "url": d["url"],
        "city": slug_city(d["slug"]),        # None = resolve from address later
        "address": address,
        "accreditations": accred,
        "rating": float(m.group(1)) if m else None,
        "reviews": to_int(m.group(2)) if m else None,
        "beds": to_int(beds.group(1)) if beds else None,
        "doctors": to_int(docs.group(1)) if docs else None,
        "emergency_24x7": bool(re.search(r"24\s*[x/]\s*7", head, re.I)),
        "doctors_profiled": 0,
        "max_doctor_years": None,
    }


def resolve_cities(hospitals):
    """Second pass: fill in cities the slug couldn't give us."""
    known = {h["city"] for h in hospitals if h["city"]}
    for h in hospitals:
        if not h["city"]:
            h["city"] = address_city(h["address"], known) or "Unknown"
            known.add(h["city"])


def parse_doctor(d):
    lines = lines_of(d)
    exp = re.search(r"Experience:?\s*(\d+)", d["text"])
    title_i = next((i for i, l in enumerate(lines) if l == d["title"]), None)
    return {
        "slug": d["slug"], "name": d["title"], "url": d["url"],
        "department": lines[0] if lines and lines[0] != d["title"] else None,
        "designation": lines[title_i + 1] if title_i is not None and title_i + 1 < len(lines) else None,
        "specialty": after(lines, "Specialized in"),
        "years": int(exp.group(1)) if exp else None,
        "hospital_raw": before(lines, "Hospital Partner"),
        "city": before(lines, "Location"),
        "hospital_slug": None,
    }


def raw_html_for(slug):
    """Saved HTML from the crawler. Filenames follow crawl.py: /treatment/x-y -> treatment_x_y.html"""
    p = DATA / "raw" / f"treatment_{slug.replace('-', '_')}.html"
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


def parse_treatment(d, hospitals, hospital_text):
    lines = lines_of(d)
    tn = norm(d["text"])
    raw = raw_html_for(d["slug"])

    # source 1: hospital names in the visible text
    from_text = [h["slug"] for h in hospitals if norm(h["name"]) and norm(h["name"]) in tn]
    # source 2: hospital slugs anywhere in the raw HTML (catches hidden / filtered lists)
    from_raw = [h["slug"] for h in hospitals if h["slug"] in raw]
    # diagnostic only: hospitals whose own page mentions this treatment
    tt = norm(re.sub(r"\(.*?\)", "", d["title"]))
    from_hospital = [h["slug"] for h in hospitals
                     if tt and tt in norm(hospital_text.get(h["slug"], ""))]

    linked = from_text   # raw_html over-links (every hospital appears in the page source) # union, order kept
    return {
        "slug": d["slug"], "title": d["title"], "url": d["url"],
        "cost_range": after(lines, "Total Cost"),
        "hospital_stay": after(lines, "Approx hospital stay"),
        "hospital_slugs": linked,
        "_link_counts": {"text": len(from_text), "raw_html": len(from_raw),
                         "hospital_pages": len(from_hospital)},
    }


def link_doctor(doc, hospitals):
    hn = norm(doc["hospital_raw"])
    if not hn:
        return None
    cands = [h for h in hospitals if norm(h["name"]) in hn or hn in norm(h["name"])]
    if not cands:
        return None
    for h in cands:
        if norm(h["city"]) == norm(canon_city(doc["city"])):
            return h["slug"]
    return cands[0]["slug"]


def fill_rate(rows, fields):
    n = len(rows)
    return ", ".join(
        f"{f}: {sum(1 for r in rows if r.get(f) not in (None, [], ''))}/{n}" for f in fields)


# ---------------- main ----------------
def main():
    pages = [json.loads(l) for l in (DATA / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
    hospitals = [parse_hospital(p) for p in pages if p["type"] == "hospital"]
    resolve_cities(hospitals)
    doctors = [parse_doctor(p) for p in pages if p["type"] == "doctor"]

    by_slug = {h["slug"]: h for h in hospitals}
    for dr in doctors:
        dr["hospital_slug"] = link_doctor(dr, hospitals)
        h = by_slug.get(dr["hospital_slug"])
        if h:
            h["doctors_profiled"] += 1
            if dr["years"] and (h["max_doctor_years"] or 0) < dr["years"]:
                h["max_doctor_years"] = dr["years"]

    hospital_text = {p["slug"]: p["text"] for p in pages if p["type"] == "hospital"}
    treatments = [parse_treatment(p, hospitals, hospital_text)
                  for p in pages if p["type"] == "treatment"]

    for name, rows in (("hospitals", hospitals), ("doctors", doctors), ("treatments", treatments)):
        (DATA / f"{name}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"hospitals={len(hospitals)} doctors={len(doctors)} treatments={len(treatments)}\n")
    print("HOSPITAL fill rates ->", fill_rate(hospitals, ["rating", "reviews", "beds", "doctors", "accreditations", "address"]))
    print("DOCTOR fill rates   ->", fill_rate(doctors, ["specialty", "years", "hospital_raw", "city", "hospital_slug"]))
    print("TREATMENT fill rates->", fill_rate(treatments, ["cost_range", "hospital_stay", "hospital_slugs"]))
    print("\nCities:", dict(Counter(h["city"] for h in hospitals)))

    # --- the check that matters: does the knee page now include Bengaluru? ---
    raw_files = list((DATA / "raw").glob("treatment_*.html"))
    print(f"\nRaw treatment HTML files found: {len(raw_files)} (expected ~{len(treatments)})")
    knee = next((t for t in treatments if t["slug"] == "knee-replacement"), None)
    if knee:
        print("Knee replacement link sources:", knee["_link_counts"])
        print("Knee replacement hospitals by city:",
              dict(Counter(by_slug[s]["city"] for s in knee["hospital_slugs"])))
        print("Knee replacement in Bengaluru:",
              [by_slug[s]["name"] for s in knee["hospital_slugs"] if by_slug[s]["city"] == "Bengaluru"])

    unlinked = [d["hospital_raw"] for d in doctors if not d["hospital_slug"]][:5]
    print("\nDoctors with no hospital match (first 5):", unlinked)


if __name__ == "__main__":
    main()