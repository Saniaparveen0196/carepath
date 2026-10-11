"""Explainable provider ranking. All weights and rules come from the workspace config."""
import math
import re

from .workspace import Workspace


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _clamp(x):
    return max(0.0, min(1.0, x))


def _result(procedure=None, results=None, note=None):
    return {"procedure": procedure, "results": results or [], "note": note}


def find_procedure(ws: Workspace, query):
    q = _norm(query)
    exact = [p for p in ws.procedures if q and (q == _norm(p["name"]) or q == _norm(p["id"]))]
    if exact:                                    # an exact name or id always wins over a partial match
        return exact[0]
    matches = [p for p in ws.procedures if q and (q in _norm(p["name"]) or q in _norm(p["id"]))]
    matches.sort(key=lambda p: (not _norm(p["name"]).startswith(q), len(p["name"])))
    return matches[0] if matches else None


def rank(ws: Workspace, procedure=None, city=None, require=(), k=5):
    cfg = ws.config["ranking"]
    W, ACC = cfg["weights"], cfg["accreditation_weights"]
    providers, matched = ws.providers, None

    if procedure:
        matched = find_procedure(ws, procedure)
        if not matched:
            ex = ", ".join(p["name"] for p in ws.procedures[:12])
            return _result(note=f"No procedure matched '{procedure}'. Examples: {ex}")
        allowed = {o["provider_id"] for o in ws.offers if o["procedure_id"] == matched["id"]}
        providers = [p for p in ws.providers if p["id"] in allowed]

    if city:
        c = ws.canon_city(city)
        in_city = [p for p in providers if _norm(p["city"]) == _norm(c)]
        if not in_city:
            cities = sorted({p["city"] for p in providers})
            what = matched["name"] if matched else "this search"
            return _result(matched, note=f"No providers are listed for {what} in {city}. "
                                         f"Cities with listed providers: {', '.join(cities)}.")
        providers = in_city

    for acc in require:
        filtered = [p for p in providers if acc.upper() in p["accreditations"]]
        if not filtered:
            return _result(matched, note=f"No matching providers have {acc.upper()} accreditation listed.")
        providers = filtered

    if not providers:
        return _result(matched, note="No providers matched those filters.")

    rated = [p["rating"] for p in ws.providers if p["rating"]]
    mean_rating = sum(rated) / len(rated) if rated else 4.0
    max_beds = max([p["beds"] for p in ws.providers if p["beds"]] or [1])
    max_docs = max([p["doctors"] for p in ws.providers if p["doctors"]] or [1])
    best_years = {}
    for c in ws.clinicians:
        if c.get("years"):
            best_years[c["provider_id"]] = max(best_years.get(c["provider_id"], 0), c["years"])
    price = {o["provider_id"]: o for o in ws.offers if matched and o["procedure_id"] == matched["id"]}

    results = []
    for p in providers:
        f = {}
        if p["rating"]:
            n = p["reviews"] or 0
            adj = (n * p["rating"] + cfg["prior_reviews"] * mean_rating) / (n + cfg["prior_reviews"])
            f["rating"] = (_clamp((adj - 3.5) / 1.5), f"rating {p['rating']} from {n:,} reviews (adjusted {adj:.2f})")
        else:
            f["rating"] = (None, "rating not available")

        acc_score = sum(ACC.get(a, 0) for a in p["accreditations"])
        f["accreditation"] = (_clamp(acc_score),
                              "accreditation: " + (", ".join(p["accreditations"]) or "none listed"))

        yrs = best_years.get(p["id"])
        f["expertise"] = ((_clamp(yrs / cfg["expert_years_cap"]), f"most experienced listed clinician: {yrs}+ yrs")
                          if yrs else (None, "no clinician profiles linked"))
        f["scale"] = ((math.log(p["beds"]) / math.log(max_beds) if max_beds > 1 else 1.0, f"{p['beds']:,} beds")
                      if p["beds"] else (None, "bed count not available"))
        f["doctors"] = ((math.log(p["doctors"]) / math.log(max_docs) if max_docs > 1 else 1.0, f"{p['doctors']:,} doctors")
                        if p["doctors"] else (None, "doctor count not available"))
        f["emergency"] = ((1.0, "24x7 emergency") if p["emergency_24x7"]
                          else (None, "emergency info not available"))

        avail = {k: v for k, (v, _) in f.items() if v is not None}
        total_w = sum(W[k] for k in avail) or 1
        base = sum(W[k] * v for k, v in avail.items()) / total_w
        coverage = total_w / sum(W.values())
        score = base * ((1 - cfg["missing_penalty"]) + cfg["missing_penalty"] * coverage) * 100

        reasons = sorted(((k, f[k][1], W[k] * avail[k] / total_w * 100) for k in avail), key=lambda r: -r[2])
        missing = [f[k][1] for k in f if f[k][0] is None]
        o = price.get(p["id"], {})
        results.append({"provider": p, "score": score, "reasons": reasons, "missing": missing,
                        "price_min": o.get("price_min"), "price_max": o.get("price_max")})

    results.sort(key=lambda r: -r["score"])
    return _result(matched, results[:k])


def format_results(ws, res):
    out, p = [], res["procedure"]
    if p:
        bits = [p["name"]]
        if p["cost_min"]:
            bits.append(f"typical cost {ws.money(p['cost_min'])} to {ws.money(p['cost_max'])}")
        if p["stay_min_days"]:
            bits.append(f"stay {p['stay_min_days']:.0f}-{p['stay_max_days']:.0f} days")
        out.append("  |  ".join(bits) + "\n")
    for i, r in enumerate(res["results"], 1):
        h = r["provider"]
        out.append(f"{i}. {h['name']} ({h['city']})  score {r['score']:.0f}/100")
        for _, text, pts in r["reasons"]:
            out.append(f"     +{pts:4.1f}  {text}")
        if r["price_min"]:
            out.append(f"     quoted price: {ws.money(r['price_min'])} to {ws.money(r['price_max'])}")
        if r["missing"]:
            out.append("     (not scored: " + "; ".join(r["missing"]) + ")")
    out.append("\n" + ws.config["disclaimer"])
    return "\n".join(out)