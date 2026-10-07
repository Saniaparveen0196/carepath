"""
Usage:
    python cli.py init demo                  # create workspace + CSV templates
    python cli.py import-csv demo            # load workspaces/demo/import/*.csv
    python cli.py import-legacy demo         # convert the Qurovo data we already crawled
    python cli.py import-pages demo          # turn crawled pages into documents for the assistant
    python cli.py validate demo              # data checks + completeness report
    python cli.py rank demo --procedure bypass --city delhi --require JCI
    python cli.py ingest demo                # build the document index (downloads models once)
    python cli.py chat demo                  # talk to the assistant
    python cli.py leads demo                 # list captured leads (phone numbers masked)
    python cli.py leads-deliver demo         # retry delivery of pending leads to the webhook
    python cli.py leads-purge demo           # delete leads older than handoff.retention_days
    python cli.py eval-init demo             # generate evaluation cases from the workspace data
    python cli.py eval demo --offline        # run the model-free layers (seconds)
    python cli.py eval demo                  # run everything (needs models + LLM key)
    python cli.py eval demo --update-baseline
"""
import argparse
import sys

from engine.importers import import_csv, import_legacy, import_pages, validate
from engine.ranking import format_results, rank
from engine.workspace import Workspace, init_workspace

sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
for name in ("init", "import-csv", "import-legacy", "import-pages", "validate", "rank", "ingest", "chat",
             "eval-init", "eval", "leads", "leads-deliver", "leads-purge"):
    p = sub.add_parser(name)
    p.add_argument("workspace")
    if name == "rank":
        p.add_argument("--procedure")
        p.add_argument("--city")
        p.add_argument("--require", nargs="*", default=[])
        p.add_argument("-k", type=int, default=5)
    if name == "eval-init":
        p.add_argument("--force", action="store_true")
    if name == "eval":
        p.add_argument("--offline", action="store_true", help="only layers that need no models or API key")
        p.add_argument("--layers", help="comma list: data,ranker,guardrails,routing,retrieval,answers")
        p.add_argument("--update-baseline", action="store_true")
        p.add_argument("--sleep", type=float, default=2.0, help="seconds between LLM calls")
    if name in ("import-legacy", "import-pages"):
        p.add_argument("--from-dir", default="data")
a = ap.parse_args()

if a.cmd == "init":
    print("Created", init_workspace(a.workspace), "(edit the CSVs in import/, then run import-csv)")
    sys.exit()

ws = Workspace(a.workspace)
if a.cmd == "import-csv":
    print("Imported ->", ", ".join(import_csv(ws)))
elif a.cmd == "import-legacy":
    print("Converted ->", ", ".join(import_legacy(ws, a.from_dir)))
elif a.cmd == "import-pages":
    print(", ".join(import_pages(ws, f"{a.from_dir}/pages.jsonl")))
elif a.cmd == "validate":
    errors, warnings = validate(ws)
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    for e in errors:
        print("  ERROR  ", e)
    for w in warnings:
        print("  WARNING", w)
elif a.cmd == "rank":
    res = rank(ws, a.procedure, a.city, a.require, a.k)
    if res["note"]:
        print(res["note"])
    if res["results"]:
        print(format_results(ws, res))
elif a.cmd == "ingest":
    from engine.knowledge import ingest
    docs, chunks = ingest(ws)
    print(f"Indexed {docs} documents ({chunks} chunks) into {ws.path / 'index'}")
elif a.cmd == "chat":
    from engine.assistant import Assistant
    from engine.knowledge import Retriever
    print(f"Loading {ws.config['name']}...")
    bot, session = Assistant(ws, Retriever(ws)), {"sid": "cli"}
    print("Ask about treatments, hospitals, or doctors. Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        out = bot.converse(q, session)
        print(f"\nAssistant: {out['text']}\n")
        for i, title, url in out["sources"]:
            print(f"  [{i}] {title}" + (f" - {url}" if url else ""))
        if out["quick_replies"]:
            print("  options: " + " | ".join(out["quick_replies"]))
        print(f"  (route: {bot.last_route})\n")
elif a.cmd == "eval-init":
    from engine.evaluation import write_cases
    f, counts = write_cases(ws, a.force)
    print(f"Wrote {f}\n  cases: {counts}\n(edit it freely; add your own cases for your customer's real questions)")
elif a.cmd == "eval":
    from engine.evaluation import (LAYERS, OFFLINE, print_report, regressions, run_eval, save_report, summarize)
    layers = a.layers.split(",") if a.layers else (OFFLINE if a.offline else LAYERS)
    bad = [l for l in layers if l not in LAYERS]
    if bad:
        sys.exit(f"unknown layer(s): {bad}. choose from {LAYERS}")
    retriever = bot = None
    if "retrieval" in layers or "answers" in layers:
        from engine.assistant import Assistant
        from engine.knowledge import Retriever
        print("Loading models...")
        retriever = Retriever(ws)
        bot = Assistant(ws, retriever)
    results, extras = run_eval(ws, layers, retriever, bot, a.sleep)
    layer_summary = summarize(ws, results, extras)
    regs = regressions(ws, results)
    save_report(ws, results, extras, layer_summary, regs, a.update_baseline)
    print_report(ws, results, extras, layer_summary, regs)
    failed = [n for n, v in layer_summary.items() if not v["ok"]]
    if failed or regs:
        print(f"\nGATE: FAILED ({'below threshold: ' + ', '.join(failed) if failed else ''}"
              f"{' regressions' if regs else ''})")
        sys.exit(1)
    print("\nGATE: PASSED")
elif a.cmd in ("leads", "leads-deliver", "leads-purge"):
    from engine.intake import LeadStore, deliver_pending, mask_phone
    store = LeadStore(ws)
    if a.cmd == "leads":
        rows = store.all()
        print(f"{len(rows)} lead(s)")
        for l in rows:
            print(f"  {l['id']}  {l['created_at']}  {l['status']:9s} {mask_phone(l['contact']['phone'])}  "
                  f"{l['interest']['procedure']} / {l['interest']['city']}  ({l['preferences']['urgency']})")
    elif a.cmd == "leads-deliver":
        print(f"delivered {deliver_pending(ws)} pending lead(s)")
    else:
        days = ws.config["handoff"]["retention_days"]
        print(f"deleted {store.purge(days)} lead(s) older than {days} days")