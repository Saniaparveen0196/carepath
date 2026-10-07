"""Importers: bring a customer's data into the canonical schema."""
import csv
import json
import re
from pathlib import Path

from .workspace import Workspace


def _num(s, cast=float):
    s = (s or "").strip().replace(",", "")
    if not s:
        return None
    try:
        return cast(float(s)) if cast is int else float(s)
    except ValueError:
        return None


def _bool(s):
    return (s or "").strip().lower() in {"1", "true", "yes", "y", "24x7", "24/7"}


def _list(s):
    return [x.strip().upper() for x in re.split(r"[;|,]", s or "") if x.strip()]


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def import_csv(ws: Workspace, folder=None):
    folder = Path(folder) if folder else ws.path / "import"
    report = []

    providers = []
    for r in _read_csv(folder / "providers.csv"):
        providers.append({
            "id": r["id"].strip(), "name": r["name"].strip(),
            "city": ws.canon_city(r.get("city")), "address": (r.get("address") or "").strip() or None,
            "accreditations": _list(r.get("accreditations")),
            "rating": _num(r.get("rating")), "reviews": _num(r.get("reviews"), int),
            "beds": _num(r.get("beds"), int), "doctors": _num(r.get("doctors"), int),
            "emergency_24x7": _bool(r.get("emergency_24x7")),
            "url": (r.get("url") or "").strip() or None,
        })
    ws.save("providers", providers)
    report.append(f"providers: {len(providers)}")

    procedures = []
    for r in _read_csv(folder / "procedures.csv"):
        procedures.append({
            "id": r["id"].strip(), "name": r["name"].strip(),
            "category": (r.get("category") or "").strip() or None,
            "description": (r.get("description") or "").strip() or None,
            "cost_min": _num(r.get("cost_min")), "cost_max": _num(r.get("cost_max")),
            "stay_min_days": _num(r.get("stay_min_days")), "stay_max_days": _num(r.get("stay_max_days")),
            "url": (r.get("url") or "").strip() or None,
        })
    ws.save("procedures", procedures)
    report.append(f"procedures: {len(procedures)}")

    offers = [{"provider_id": r["provider_id"].strip(), "procedure_id": r["procedure_id"].strip(),
               "price_min": _num(r.get("price_min")), "price_max": _num(r.get("price_max"))}
              for r in _read_csv(folder / "offers.csv")]
    ws.save("offers", offers)
    report.append(f"offers: {len(offers)}")

    cf = folder / "clinicians.csv"
    clinicians = []
    if cf.exists():
        for r in _read_csv(cf):
            clinicians.append({
                "id": r["id"].strip(), "name": r["name"].strip(),
                "specialty": (r.get("specialty") or "").strip() or None,
                "years": _num(r.get("years"), int), "provider_id": r["provider_id"].strip(),
            })
    ws.save("clinicians", clinicians)
    report.append(f"clinicians: {len(clinicians)}")
    return report


# ---------- converter for the data we already crawled (demo workspace) ----------
def _money(token):
    m = re.match(r"([\d.]+)\s*(L|Cr|K)?", token.strip(), re.I)
    if not m:
        return None
    v, unit = float(m.group(1)), (m.group(2) or "").lower()
    return v * {"l": 1e5, "cr": 1e7, "k": 1e3, "": 1}[unit]


def import_legacy(ws: Workspace, legacy_dir="data"):
    d = Path(legacy_dir)
    H = json.loads((d / "hospitals.json").read_text(encoding="utf-8"))
    T = json.loads((d / "treatments.json").read_text(encoding="utf-8"))
    D = json.loads((d / "doctors.json").read_text(encoding="utf-8"))

    ws.save("providers", [{
        "id": h["slug"], "name": h["name"], "city": ws.canon_city(h["city"]), "address": h.get("address"),
        "accreditations": h["accreditations"], "rating": h["rating"], "reviews": h["reviews"],
        "beds": h["beds"], "doctors": h["doctors"], "emergency_24x7": h["emergency_24x7"],
        "url": h["url"]} for h in H])

    procs, offers = [], []
    for t in T:
        lo = hi = None
        if t.get("cost_range"):
            parts = re.split(r"\s*[-–]\s*", t["cost_range"])
            if len(parts) == 2:
                lo, hi = _money(parts[0]), _money(parts[1])
        s_lo = s_hi = None
        m = re.findall(r"\d+", t.get("hospital_stay") or "")
        if m:
            s_lo, s_hi = float(m[0]), float(m[-1])
        procs.append({"id": t["slug"], "name": t["title"], "category": None, "description": None,
                      "cost_min": lo, "cost_max": hi, "stay_min_days": s_lo, "stay_max_days": s_hi,
                      "url": t["url"]})
        offers += [{"provider_id": s, "procedure_id": t["slug"], "price_min": None, "price_max": None}
                   for s in t["hospital_slugs"]]
    ws.save("procedures", procs)
    ws.save("offers", offers)
    ws.save("clinicians", [{
        "id": x["slug"], "name": x["name"], "specialty": x.get("specialty"), "years": x.get("years"),
        "provider_id": x["hospital_slug"]} for x in D if x.get("hospital_slug")])
    return [f"providers: {len(H)}", f"procedures: {len(T)}", f"offers: {len(offers)}",
            f"clinicians: {len(D)}"]


# ---------- validation & data-completeness report ----------
def validate(ws: Workspace):
    errors, warnings = [], []
    pids = {p["id"] for p in ws.providers}
    rids = {p["id"] for p in ws.procedures}
    for name, rows in (("providers", ws.providers), ("procedures", ws.procedures)):
        ids = [r["id"] for r in rows]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            errors.append(f"{name}: duplicate ids {sorted(dup)[:5]}")
    for o in ws.offers:
        if o["provider_id"] not in pids:
            errors.append(f"offers: unknown provider_id '{o['provider_id']}'")
        if o["procedure_id"] not in rids:
            errors.append(f"offers: unknown procedure_id '{o['procedure_id']}'")
    for p in ws.providers:
        if p["rating"] is not None and not 0 <= p["rating"] <= 5:
            errors.append(f"providers: rating out of range for {p['id']}")
        if not p["city"]:
            errors.append(f"providers: missing city for {p['id']}")
    for c in ws.clinicians:
        if c["provider_id"] not in pids:
            warnings.append(f"clinicians: unknown provider_id '{c['provider_id']}' ({c['name']})")

    linked = {o["procedure_id"] for o in ws.offers}
    empty = [r["name"] for r in ws.procedures if r["id"] not in linked]
    if empty:
        warnings.append(f"{len(empty)} procedures have no providers (e.g. {empty[:3]})")
    n = max(len(ws.providers), 1)
    for field in ("rating", "beds", "doctors"):
        miss = sum(1 for p in ws.providers if not p[field])
        if miss:
            warnings.append(f"{miss}/{n} providers missing '{field}'")
    return errors, warnings


# ---------- turn crawled pages into workspace documents (for the assistant's knowledge) ----------
def import_pages(ws: Workspace, pages_file="data/pages.jsonl"):
    out = ws.path / "documents"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in Path(pages_file).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        front = f"---\ntitle: {d['title']}\nsource: {d['url']}\ntype: {d['type']}\n---\n"
        (out / f"{d['type']}-{d['slug']}.md").write_text(front + d["text"], encoding="utf-8")
        n += 1
    return [f"documents written: {n}"]