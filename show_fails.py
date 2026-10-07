import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
r = json.load(open("data/eval_results.json", encoding="utf-8"))
for a in r["answers"]:
    if a["fails"]:
        print("=" * 80)
        print("Q:", a["question"])
        print("FAILS:", a["fails"])
        print("SOURCES:", a["sources"])
        print(a["answer"])