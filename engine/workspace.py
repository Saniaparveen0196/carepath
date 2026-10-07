"""A workspace = one customer's data + rules. Everything lives in workspaces/<name>/."""
import json
from pathlib import Path

import yaml

ROOT = Path("workspaces")

DEFAULT_CONFIG = {
    "name": "Health Directory",
    "country": "IN",
    "currency": "INR",
    "emergency_number": "112",
    "city_aliases": {"bangalore": "Bengaluru", "gurgaon": "Gurugram", "bombay": "Mumbai",
                     "new delhi": "Delhi"},
    "llm": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "ranking": {
        "weights": {"rating": 0.34, "accreditation": 0.26, "expertise": 0.10,
                    "scale": 0.12, "doctors": 0.10, "emergency": 0.08},
        "accreditation_weights": {"NABH": 0.50, "JCI": 0.35, "NABL": 0.15},
        "prior_reviews": 200,       # shrinks ratings that have few reviews toward the average
        "missing_penalty": 0.30,    # how much a score depends on data completeness
        "expert_years_cap": 40,
    },
    "disclaimer": ("Scores combine listed facts only (ratings, accreditation, beds, doctors, "
                   "clinician experience). They are not medical advice, and no single score "
                   "means a provider is right for a specific patient."),
}

TABLES = ("providers", "procedures", "offers", "clinicians")


def deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Workspace:
    def __init__(self, name):
        self.name = name
        self.path = ROOT / name
        if not self.path.exists():
            raise FileNotFoundError(f"Workspace '{name}' not found. Run: python cli.py init {name}")
        cfg_file = self.path / "config.yaml"
        user_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
        self.config = deep_merge(DEFAULT_CONFIG, user_cfg)
        for t in TABLES:
            setattr(self, t, self._load(t))

    def _load(self, table):
        f = self.path / "data" / f"{table}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []

    def save(self, table, rows):
        d = self.path / "data"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{table}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        setattr(self, table, rows)

    def money(self, v):
        """Format an amount in the workspace currency (lakh/crore style for INR)."""
        cur = self.config["currency"]
        if v is None:
            return "n/a"
        if cur == "INR":
            if v >= 1e7:
                return f"INR {v / 1e7:g} crore"
            if v >= 1e5:
                return f"INR {v / 1e5:g} lakh"
        return f"{cur} {v:,.0f}"

    def canon_city(self, name):
        n = (name or "").strip()
        return self.config["city_aliases"].get(n.lower(), n.title())


def init_workspace(name):
    p = ROOT / name
    (p / "data").mkdir(parents=True, exist_ok=True)
    (p / "documents").mkdir(exist_ok=True)
    (p / "import").mkdir(exist_ok=True)
    cfg = p / "config.yaml"
    if not cfg.exists():
        cfg.write_text(yaml.safe_dump({"name": name.replace("-", " ").title()}, sort_keys=False),
                       encoding="utf-8")
    templates = {
        "providers.csv": "id,name,city,address,accreditations,rating,reviews,beds,doctors,emergency_24x7,url\n"
                         "apollo-delhi,Example Hospital,Delhi,\"1 Example Road, Delhi\",NABH;JCI,4.5,1200,500,200,yes,https://example.com/h1\n",
        "procedures.csv": "id,name,category,description,cost_min,cost_max,stay_min_days,stay_max_days,url\n"
                          "knee-replacement,Knee Replacement,Orthopaedics,Replaces a damaged knee joint,150000,850000,3,5,https://example.com/p1\n",
        "offers.csv": "provider_id,procedure_id,price_min,price_max\n"
                      "apollo-delhi,knee-replacement,250000,500000\n",
        "clinicians.csv": "id,name,specialty,years,provider_id\n"
                          "dr-sharma,Dr. A Sharma,Orthopaedics,22,apollo-delhi\n",
    }
    for fname, content in templates.items():
        f = p / "import" / fname
        if not f.exists():
            f.write_text(content, encoding="utf-8")
    return p