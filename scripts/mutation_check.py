"""
Does the evaluation suite actually catch bugs?  Injects deliberate bugs into a COPY of the code,
runs the offline eval on each, and expects the quality gate to FAIL (exit code 1).

Usage: python scripts/mutation_check.py synthetic
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ws = sys.argv[1] if len(sys.argv) > 1 else "synthetic"

MUTANTS = [
    ("ranking: city filter ignored (leaks other cities)", "engine/ranking.py",
     "in_city = [p for p in providers if _norm(p[\"city\"]) == _norm(c)]", "in_city = list(providers)"),
    ("ranking: required accreditation ignored", "engine/ranking.py",
     "filtered = [p for p in providers if acc.upper() in p[\"accreditations\"]]", "filtered = list(providers)"),
    ("ranking: review shrinkage inverted", "engine/ranking.py",
     "adj = (n * p[\"rating\"] + cfg[\"prior_reviews\"] * mean_rating) / (n + cfg[\"prior_reviews\"])",
     "adj = (cfg[\"prior_reviews\"] * p[\"rating\"] + n * mean_rating) / (n + cfg[\"prior_reviews\"])"),
    ("ranking: more accreditations LOWER the score", "engine/ranking.py",
     "acc_score = sum(ACC.get(a, 0) for a in p[\"accreditations\"])",
     "acc_score = max(ACC.values()) - sum(ACC.get(a, 0) for a in p[\"accreditations\"])"),
    ("ranking: score no longer matches its stated reasons", "engine/ranking.py",
     "score = base * ((1 - cfg[\"missing_penalty\"]) + cfg[\"missing_penalty\"] * coverage) * 100",
     "score = base * ((1 - cfg[\"missing_penalty\"]) + cfg[\"missing_penalty\"] * coverage) * 50"),
    ("ranking: results not sorted", "engine/ranking.py",
     "results.sort(key=lambda r: -r[\"score\"])", "results.sort(key=lambda r: r[\"provider\"][\"id\"])"),
    ("ranking: missing data silently dropped from the report", "engine/ranking.py",
     "missing = [f[k][1] for k in f if f[k][0] is None]", "missing = []"),
    ("ranking: unknown city no longer explained", "engine/ranking.py",
     "return _result(matched, note=f\"No providers are listed for {what} in {city}. \"",
     "return _result(matched, note=f\"{what}. \""),
    ("assistant: personal-report guard disabled", "engine/assistant.py",
     "if PERSONAL_PATTERNS.search(question) and not", "if False and not"),
    ("assistant: symptom guard disabled", "engine/assistant.py",
     "if SYMPTOM_PATTERNS.search(question) and not", "if False and not"),
    ("assistant: emergency guard disabled", "engine/assistant.py",
     "if EMERGENCY_PATTERNS.search(question):", "if False:"),
    ("assistant: emergency number wrong in replies", "engine/assistant.py",
     "n = ws.config[\"emergency_number\"]", "n = \"999\""),
    ("assistant: first city mentioned wins (follow-ups break)", "engine/assistant.py",
     "city = max(found)[1] if found else None", "city = min(found)[1] if found else None"),
    ("assistant: unknown-city detection disabled", "engine/assistant.py",
     "if later and (city is None or max(later)[0] > max(found)[0]):", "if False:"),
    ("assistant: medication guard disabled", "engine/assistant.py",
     "if (MED_WORDS.search(question) and MED_ACTION.search(question)", "if (False and MED_ACTION.search(question)"),
    ("assistant: listed cost/stay no longer enforced in answers", "engine/assistant.py",
     "return f\"{line} [{idx}]\\n\\n{text}\" if line else text", "return text"),
    ("assistant: number verifier disabled", "engine/assistant.py",
     "corpus = context + \" \" + question\n        bad = self.ungrounded(text, corpus)",
     "corpus = context + \" \" + question\n        bad = []"),
    ("assistant: ranking fallback not used when a number is unverified", "engine/assistant.py",
     "return self.ranking_fallback(passages)\n        name = self.ws.config", "return passages[0]['text']\n        name = self.ws.config"),
    ("assistant: off-topic scope guard disabled", "engine/assistant.py",
     "if OFFTOPIC_PATTERNS.search(question) and not", "if False and not"),
    ("assistant: recommendation questions no longer routed", "engine/assistant.py",
     "if RECO_HINT.search(search_q):", "if False:"),
]

copy_ok = shutil.ignore_patterns("__pycache__", "index", "reports", "baseline.json", ".venv", "*.pyc")


def make_copy(t):
    shutil.copytree(ROOT / "engine", t / "engine", ignore=copy_ok)
    shutil.copy(ROOT / "cli.py", t / "cli.py")
    shutil.copytree(ROOT / "workspaces" / ws, t / "workspaces" / ws, ignore=copy_ok)


# record a baseline from the UNMODIFIED code, exactly like a CI pipeline would
with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    make_copy(t)
    r = subprocess.run([sys.executable, "cli.py", "eval", ws, "--offline", "--update-baseline"], cwd=t,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.exit("The unmodified code already fails its own eval:\n" + r.stdout[-1500:])
    BASELINE = (t / "workspaces" / ws / "eval" / "baseline.json").read_text(encoding="utf-8")
print("baseline recorded from unmodified code (all offline checks pass)\n")

caught = 0
for name, rel, old, new in MUTANTS:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        make_copy(t)
        (t / "workspaces" / ws / "eval" / "baseline.json").write_text(BASELINE, encoding="utf-8")
        f = t / rel
        src = f.read_text(encoding="utf-8")
        if old not in src:
            print(f"  ??  {name}: pattern not found (update the mutant)")
            continue
        f.write_text(src.replace(old, new, 1), encoding="utf-8")
        r = subprocess.run([sys.executable, "cli.py", "eval", ws, "--offline"], cwd=t,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        failed = [l.split()[0] for l in r.stdout.splitlines() if l.rstrip().endswith("FAIL") and "%" in l]
        if "regressions vs baseline: 0" not in r.stdout and "regressions vs baseline" in r.stdout:
            failed.append("regression-gate")
        status = "CAUGHT " if r.returncode == 1 else "MISSED "
        caught += r.returncode == 1
        print(f"  {status} {name:58s} {('-> ' + ', '.join(failed)) if failed else ''}")
print(f"\nmutation score: {caught}/{len(MUTANTS)} injected bugs caught")
sys.exit(0 if caught == len(MUTANTS) else 1)