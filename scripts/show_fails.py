import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
ws = sys.argv[1] if len(sys.argv) > 1 else "demo"
report = json.load(open(f"workspaces/{ws}/eval/reports/latest.json", encoding="utf-8"))
for layer, rows in report["results"].items():
    for x in rows:
        if not x["passed"] and "answer" in x:
            print("=" * 80)
            print(f"{x['id']} | route: {x.get('route')} | {x.get('seconds')}s")
            print("Q:", x["q"])
            print("WHY IT FAILED:", x["detail"])
            print("ANSWER:", x["answer"])