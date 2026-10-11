"""
Generate a clearly FICTIONAL hospital directory workspace (safe to publish).
Usage: python scripts/make_synthetic.py [workspace-name]     (default: synthetic)
Then:  python cli.py import-csv synthetic && python cli.py ingest synthetic && python cli.py eval-init synthetic
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.workspace import init_workspace  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
rng = random.Random(7)
p = init_workspace(name)
imp, docs = p / "import", p / "documents"

CITIES = ["Bengaluru", "Delhi", "Mumbai", "Chennai", "Hyderabad", "Pune"]
FIRST = ["Lakeview", "Riverside", "Sunrise", "Greenfield", "Harbor", "Summit", "Maple", "Cedar", "Willow", "Aurora"]
KIND = ["Heart Institute", "Multispeciality Hospital", "Medical Centre", "Super Speciality Hospital"]
PROCS = [  # id, name, category, cost_min, cost_max, stay_min, stay_max, blurb
    ("knee-replacement", "Knee Replacement", "Orthopaedics", 150000, 850000, 3, 5,
     "replaces a worn knee joint with a prosthesis to relieve pain and restore movement"),
    ("bypass-surgery", "Bypass Surgery (CABG)", "Cardiac", 250000, 700000, 4, 7,
     "reroutes blood around blocked heart arteries using grafts taken from elsewhere in the body"),
    ("angioplasty", "Angioplasty & Stenting", "Cardiac", 120000, 450000, 1, 2,
     "opens a narrowed artery with a balloon and places a small mesh stent to keep it open"),
    ("cataract-surgery", "Cataract Surgery", "Ophthalmology", 25000, 120000, 1, 1,
     "replaces the clouded natural lens of the eye with a clear artificial lens"),
    ("hernia-repair", "Hernia Repair", "General Surgery", 60000, 180000, 1, 2,
     "closes a weakness in the abdominal wall, often with a supportive mesh"),
    ("kidney-stone-removal", "Kidney Stone Removal", "Urology", 40000, 200000, 1, 3,
     "removes or breaks up stones in the kidney or ureter using endoscopic techniques"),
    ("spine-surgery", "Spine Surgery", "Neurosurgery", 200000, 900000, 3, 6,
     "relieves pressure on nerves in the spine or stabilises unstable vertebrae"),
    ("hair-transplant", "Hair Transplant", "Cosmetic", 50000, 250000, 1, 1,
     "moves hair follicles from denser areas of the scalp to thinning areas"),
    ("haemorrhoid-surgery", "Haemorrhoid Surgery (Piles)", "General Surgery", 45000, 180000, 1, 2,
     "treats swollen veins around the anus with minimally invasive techniques"),
    ("haemorrhoidectomy", "Haemorrhoidectomy (Piles Surgery)", "General Surgery", 45000, 150000, 1, 2,
     "removes haemorrhoid tissue surgically when other treatments have not worked"),
    ("hernia-repair-laparoscopic", "Hernia Repair (Laparoscopic)", "General Surgery", 70000, 220000, 1, 2,
     "repairs a hernia through small keyhole incisions"),
]

providers, offers, clinicians, used = [], [], [], set()
while len(providers) < 18:
    nm = f"{rng.choice(FIRST)} {rng.choice(KIND)}"
    if nm in used:
        continue
    used.add(nm)
    city = CITIES[len(providers) % len(CITIES)]
    pid = f"{nm.lower().replace(' ', '-')}-{city.lower()}"
    acc = [a for a in ("NABH", "JCI", "NABL") if rng.random() < {"NABH": .8, "JCI": .35, "NABL": .5}[a]]
    providers.append({
        "id": pid, "name": nm, "city": city, "address": f"{rng.randint(1, 99)} Example Road, {city}",
        "accreditations": ";".join(acc), "rating": round(rng.uniform(3.8, 4.9), 1),
        "reviews": rng.choice([40, 120, 600, 2500, 9000, 21000]), "beds": rng.choice([60, 120, 250, 500, 900]),
        "doctors": rng.choice(["", 40, 90, 180, 320]), "emergency_24x7": rng.choice(["yes", "yes", ""]),
        "url": f"https://example.org/{pid}"})
    for k in range(2):
        clinicians.append({"id": f"{pid}-dr{k + 1}", "name": f"Dr. {rng.choice(['A', 'B', 'C', 'D'])}. {rng.choice(['Rao', 'Iyer', 'Khan', 'Mehta', 'Das'])}",
                           "specialty": rng.choice(PROCS)[2], "years": rng.randint(8, 38), "provider_id": pid})

for pid_, pname, cat, lo, hi, s1, s2, blurb in PROCS:
    for prov in rng.sample(providers, rng.randint(8, 13)):
        a = int(lo * rng.uniform(1.0, 1.4)); b = int(hi * rng.uniform(0.6, 1.0))
        offers.append({"provider_id": prov["id"], "procedure_id": pid_, "price_min": a, "price_max": max(a, b)})


def write(fname, rows):
    with open(imp / fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


write("providers.csv", providers)
write("procedures.csv", [{"id": i, "name": n, "category": c, "description": b, "cost_min": lo, "cost_max": hi,
                          "stay_min_days": s1, "stay_max_days": s2, "url": f"https://example.org/{i}"}
                         for i, n, c, lo, hi, s1, s2, b in PROCS])
write("offers.csv", offers)
write("clinicians.csv", clinicians)

for i, n, c, lo, hi, s1, s2, b in PROCS:
    (docs / f"treatment-{i}.md").write_text(
        f"---\ntitle: {n}\nsource: https://example.org/{i}\ntype: treatment\n---\n"
        f"{n} is a procedure in {c} that {b}. This is fictional demo content for a synthetic directory.\n"
        f"Most patients stay in hospital for {s1} to {s2} days, depending on the case and the hospital.\n"
        f"Recovery after {n.lower()} involves rest, follow-up visits, and a gradual return to normal activity "
        f"over several weeks, guided by the treating doctor.\n"
        f"Typical total cost in this directory ranges from {lo:,} to {hi:,} rupees and varies by hospital and city.\n",
        encoding="utf-8")
for prov in providers:
    (docs / f"hospital-{prov['id']}.md").write_text(
        f"---\ntitle: {prov['name']}\nsource: {prov['url']}\ntype: hospital\n---\n"
        f"{prov['name']} is a fictional hospital in {prov['city']} with {prov['beds']} beds. "
        f"Accreditations listed: {prov['accreditations'] or 'none'}. Patient rating {prov['rating']} "
        f"from {prov['reviews']} reviews.\n", encoding="utf-8")
print(f"Created fictional workspace '{name}': {len(providers)} providers, {len(PROCS)} procedures, "
      f"{len(offers)} offers, {len(clinicians)} clinicians, {len(PROCS) + len(providers)} documents")