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
    ("intake: consent skipped", "engine/intake.py",
     "            if is_yes(text):\n                state[\"consent_at\"]", "            if True:\n                state[\"consent_at\"]"),
    ("intake: phone numbers not validated", "engine/intake.py",
     "return f\"+91{digits}\" if len(digits) == 10 and digits[0] in \"6789\" else None",
     "return f\"+91{digits}\" if digits else None"),
    ("intake: medical text accepted as a name", "engine/intake.py",
     "if any(w.lower().strip(\".,'\") in NOT_A_NAME for w in words):", "if False:"),
    ("intake: lead saved without confirmation", "engine/intake.py",
     "            if is_yes(text):\n                return self._submit(state)", "            if True:\n                return self._submit(state)"),
    ("intake: conversation transcript leaks into the lead", "engine/intake.py",
     "\"source\": {\"channel\": \"web-widget\", \"session\": state[\"sid\"]},",
     "\"source\": {\"channel\": \"web-widget\", \"session\": state[\"sid\"]}, \"transcript\": state[\"topics\"],"),
    ("intake: full phone number written to the logs", "engine/intake.py",
     "mask_phone(d[\"phone\"]), d[\"urgency\"])", "d[\"phone\"], d[\"urgency\"])"),
    ("intake: webhook requests unsigned", "engine/intake.py", "if cfg.get(\"webhook_secret\"):", "if False:"),
    ("intake: duplicate requests create duplicate leads", "engine/intake.py",
     "        if dup:\n            ref = dup[\"id\"]", "        if False:\n            ref = dup[\"id\"]"),
    ("intake: retention purge does nothing", "engine/intake.py",
     "if now - datetime.fromisoformat(l[\"created_at\"]) > timedelta(days=days):", "if False:"),
    ("assistant: emergency no longer interrupts intake", "engine/assistant.py",
     "        if EMERGENCY_PATTERNS.search(question):\n            session[\"intake\"] = None",
     "        if False:\n            session[\"intake\"] = None"),
    ("assistant: intake cannot be switched off", "engine/assistant.py",
     "elif self.ws.config[\"handoff\"][\"enabled\"] and INTAKE_INTENT.search(question):",
     "elif INTAKE_INTENT.search(question):"),
    ("assistant: code-written ranking answer drops the disclaimer", "engine/assistant.py",
     "        lines.append(f\"_{ws.config['disclaimer']}_\")\n        return \"\\n\".join(lines).strip(), sources",
     "        return \"\\n\".join(lines).strip(), sources"),
    ("assistant: code-written ranking lists providers in the wrong order", "engine/assistant.py",
     "        for i, r in enumerate(res[\"results\"], 1):\n            h = r[\"provider\"]\n            facts = [t for _",
     "        for i, r in enumerate(reversed(res[\"results\"]), 1):\n            h = r[\"provider\"]\n            facts = [t for _"),
    ("assistant: instant cost answer omits the listed figures", "engine/assistant.py",
     "        text = (f\"{line} These ranges are approximate", "        text = (f\"Please see the directory. These ranges are approximate"),
    ("assistant: answer cache ignores the question", "engine/assistant.py",
     "key = (re.sub(r\"\\s+\", \" \", question.lower().strip()), doc_type)", "key = (\"x\", doc_type)"),
    ("assistant: answers cached in the middle of a conversation", "engine/assistant.py",
     "if (ttl and not history and m.get(\"route\") in", "if (ttl and m.get(\"route\") in"),
    ("assistant: look-alike procedure names confused (the bug found in the real run)", "engine/assistant.py",
     ["best = max(exact, key=lambda p: len(p[\"name\"]))[\"name\"] if exact else None",
      "hits = sum(1 for w in words if re.search(rf\"\\b{re.escape(w)}(?:s|es)?\\b\", q))",
      "key = (hits, hits / max(len(words), 1), -len(p[\"name\"]))"],
     ["best = None",
      "hits = sum(1 for w in words if w in q)",
      "key = (hits, -len(p[\"name\"]), 0)"]),
    ("vocabulary: everyday words are no longer expanded", "engine/vocab.py",
     "return query if not extra else query + \" \" + \" \".join(extra)", "return query"),
    ("vocabulary: terms expanded inside longer words", "engine/vocab.py",
     "if re.search(rf\"\\b{re.escape(term.lower())}\\b\", q):", "if re.search(rf\"{re.escape(term.lower())}\", q):"),
    ("eval: equivalent pages no longer accepted as correct", "engine/evaluation.py",
     "            exp |= set(group)", "            exp |= set()"),
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
        pairs = list(zip(old, new)) if isinstance(old, list) else [(old, new)]
        if any(o not in src for o, _ in pairs):
            print(f"  ??  {name}: pattern not found (update the mutant)")
            continue
        for o, n in pairs:
            src = src.replace(o, n, 1)
        f.write_text(src, encoding="utf-8")
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