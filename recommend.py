"""
Explainable hospital ranking.

CLI examples:
    python recommend.py --treatment bypass
    python recommend.py --treatment "knee replacement" --city delhi --require NABH
"""
import argparse
import json
import math
import re
from pathlib import Path

DATA = Path("data")
CITY_ALIASES = {"bangalore": "bengaluru", "gurgaon": "gurugram", "bombay": "mumbai"}
ACC_WEIGHTS = {"NABH": 0.50, "JCI": 0.35, "NABL": 0.15}
WEIGHTS = {"rating": 0.34, "accreditation": 0.26, "expertise": 0.10,
           "scale": 0.12, "doctors": 0.10, "emergency": 0.08}
PRIOR_REVIEWS = 200      # shrinks ratings with few reviews toward the average
MISSING_PENALTY = 0.30   # share of score that depends on data completeness


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def clamp(x):
    return max(0.0, min(1.0, x))


def _result(treatment=None, results=None, note=None):
    return {"treatment": treatment, "results": results or [], "note": note}


def recommend(treatment=None, city=None, require=(), k=5, weights=WEIGHTS):
    all_h, treatments = load("hospitals"), load("treatments")
    hospitals, matched = all_h, None

    if treatment:
        q = norm(treatment)
        matches = [t for t in treatments if q in norm(t["title"]) or q in norm(t["slug"])]
        if not matches:
            examples = ", ".join(t["title"] for t in treatments[:12])
            return _result(note=f"No treatment matched '{treatment}'. Examples: {examples}")
        # prefer titles that start with the query, then the shortest title
        matches.sort(key=lambda t: (not norm(t["title"]).startswith(q), len(t["title"])))
        matched = matches[0]
        allowed = set(matched["hospital_slugs"])
        hospitals = [h for h in all_h if h["slug"] in allowed]

    if city:
        c = norm(CITY_ALIASES.get(city.lower(), city))
        in_city = [h for h in hospitals if norm(h["city"]) == c]
        if not in_city:
            cities = sorted({h["city"] for h in hospitals})
            what = matched["title"] if matched else "this search"
            return _result(matched, note=(
                f"No hospitals are listed for {what} in {city}. "
                f"Cities with listed hospitals: {', '.join(cities)}."))
        hospitals = in_city

    for acc in require:
        filtered = [h for h in hospitals if acc.upper() in h["accreditations"]]
        if not filtered:
            return _result(matched, note=f"No matching hospitals have {acc.upper()} accreditation listed.")
        hospitals = filtered

    if not hospitals:
        return _result(matched, note="No hospitals matched those filters.")

    rated = [h["rating"] for h in all_h if h["rating"]]
    mean_rating = sum(rated) / len(rated) if rated else 4.0
    max_beds = max([h["beds"] for h in all_h if h["beds"]] or [1])
    max_docs = max([h["doctors"] for h in all_h if h["doctors"]] or [1])

    results = []
    for h in hospitals:
        f = {}  # factor -> (value 0..1 or None, readable fact)
        if h["rating"]:
            v = h["reviews"] or 0
            adj = (v * h["rating"] + PRIOR_REVIEWS * mean_rating) / (v + PRIOR_REVIEWS)
            f["rating"] = (clamp((adj - 3.5) / 1.5),
                           f"rating {h['rating']} from {v:,} reviews (adjusted {adj:.2f})")
        else:
            f["rating"] = (None, "rating not available")

        acc_score = sum(ACC_WEIGHTS[a] for a in h["accreditations"] if a in ACC_WEIGHTS)
        acc_txt = ", ".join(h["accreditations"]) if h["accreditations"] else "none listed"
        f["accreditation"] = (clamp(acc_score), f"accreditation: {acc_txt}")

        if h["max_doctor_years"]:
            f["expertise"] = (clamp(h["max_doctor_years"] / 40),
                              f"most experienced profiled surgeon: {h['max_doctor_years']}+ yrs")
        else:
            f["expertise"] = (None, "no doctor profiles linked")

        if h["beds"]:
            f["scale"] = (math.log(h["beds"]) / math.log(max_beds), f"{h['beds']:,} beds")
        else:
            f["scale"] = (None, "bed count not available")

        if h["doctors"]:
            f["doctors"] = (math.log(h["doctors"]) / math.log(max_docs), f"{h['doctors']:,} doctors")
        else:
            f["doctors"] = (None, "doctor count not available")

        f["emergency"] = ((1.0, "24x7 emergency") if h["emergency_24x7"]
                          else (None, "emergency info not available"))

        avail = {key: v for key, (v, _) in f.items() if v is not None}
        total_w = sum(weights[key] for key in avail) or 1
        base = sum(weights[key] * v for key, v in avail.items()) / total_w
        coverage = total_w / sum(weights.values())
        score = base * ((1 - MISSING_PENALTY) + MISSING_PENALTY * coverage) * 100

        reasons = [(key, f[key][1], weights[key] * avail[key] / total_w * 100) for key in avail]
        missing = [f[key][1] for key in f if f[key][0] is None]
        results.append({"hospital": h, "score": score,
                        "reasons": sorted(reasons, key=lambda r: -r[2]), "missing": missing})

    results.sort(key=lambda r: -r["score"])
    return _result(matched, results[:k])


def format_results(res):
    out, t = [], res["treatment"]
    if t:
        out.append(f"Treatment: {t['title']}  |  typical cost: {t['cost_range']}  |  stay: {t['hospital_stay']}\n")
    for i, r in enumerate(res["results"], 1):
        h = r["hospital"]
        out.append(f"{i}. {h['name']} ({h['city']})  score {r['score']:.0f}/100")
        for _, text, pts in r["reasons"]:
            out.append(f"     +{pts:4.1f}  {text}")
        if r["missing"]:
            out.append("     (not scored: " + "; ".join(r["missing"]) + ")")
        out.append(f"     {h['url']}")
    out.append("\nScores combine platform-listed facts only. They are not medical advice, "
               "and no single score means a hospital is right for a specific patient.")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment")
    ap.add_argument("--city")
    ap.add_argument("--require", nargs="*", default=[], help="e.g. NABH JCI")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()

    res = recommend(a.treatment, a.city, a.require, a.k)
    if res["note"]:
        print(res["note"])
    if res["results"]:
        print(format_results(res))