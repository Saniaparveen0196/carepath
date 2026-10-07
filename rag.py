"""
RAG assistant: hybrid retrieval + structured hospital ranking + guarded LLM answers.
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from recommend import recommend

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PROVIDER = os.getenv("LLM_PROVIDER", "groq")          # groq | anthropic | openai
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# ---------------- safety guards (deterministic, no LLM) ----------------
EMERGENCY_PATTERNS = re.compile(
    r"(chest pain (right )?now|can'?t breathe|cannot breathe|severe bleeding|"
    r"unconscious|stroke symptoms|heart attack (now|right now)|suicid|kill myself|want to die)",
    re.I,
)
EMERGENCY_REPLY = (
    "This sounds like it could be an emergency. Please call 112 (India's emergency number) "
    "or go to the nearest emergency room right away. I can't help with urgent medical "
    "situations, but once the situation is safe, I'm glad to help you explore treatment options."
)

SYMPTOM_PATTERNS = re.compile(
    r"(what (disease|condition|illness|disorder|problem) (do|could|might|can) i have|"
    r"what('?s| is) wrong with me|(diagnose|diagnosis for) me|"
    r"do i have (cancer|a tumou?r|diabetes|a heart)|"
    r"\bi(?: have| am having|'ve had| have had| keep getting| have been having|'ve been having| am experiencing) "
    r"(?:a |an |some |severe |sharp )?"
    r"(?:headache|migraine|fever|chest pain|pain|dizz\w+|nausea|vomiting|blurr\w+|numbness|"
    r"swelling|bleeding|cough|breathless\w*|palpitations?))",
    re.I,
)
SYMPTOM_REPLY = (
    "I can't assess symptoms or tell you what might be causing them. That needs a doctor who can "
    "examine you or review your tests, so please see one soon. If your symptoms are sudden, severe, "
    "or getting worse (for example sudden vision loss, a very severe headache, weakness, trouble "
    "speaking, chest pain, or trouble breathing), call 112 or go to the nearest emergency room. "
    "Once you have a diagnosis, I'm happy to help you compare treatment options and hospitals."
)

RANKING_DISCLAIMER = (
    "\n\n_Scores combine platform-listed facts only (ratings, accreditation, beds, doctors, "
    "surgeon experience). They are not medical advice, and no single score means a hospital "
    "is right for a specific patient._"
)

# ---------------- routing hints ----------------
RECO_HINT = re.compile(
    r"\b(best|top|recommend\w*|which hospitals?|compare|comparison|good hospitals?|"
    r"accredited|hospitals? (for|in|near)|where (to|should))\b", re.I)
FACT_HINT = re.compile(
    r"\b(cost|costs|price|how much|expensive|cheap|stay|how long|how many days|"
    r"duration|budget|charges?)\b", re.I)

GENERIC_WORDS = {"surgery", "treatment", "therapy", "repair", "procedure", "transplant"}
GENERIC_CITY_WORDS = {"hospital", "centre", "center", "institute", "clinic"}

SYSTEM_PROMPT = """You are a patient-information assistant for a hospital and treatment discovery platform in India.

Rules:
1. Answer ONLY from the numbered CONTEXT passages. If the context doesn't contain the answer, say you don't have that information and suggest requesting a callback from the care team. Never fill gaps from your own knowledge.
2. Cite sources inline like [1] or [2][3] after the claims they support. Only cite passages you actually used. Use plain square brackets like [1], never other bracket styles.
3. PERSONAL MEDICAL QUESTIONS (a specific report, symptom, blockage percentage, "does he/she need surgery?", "what disease do I have?"): never diagnose, interpret, or suggest what condition it could be. Say clearly that you can't assess this, that it needs a qualified doctor who can examine the person or review the full report, and encourage seeing a doctor soon. If symptoms sound sudden, severe, or are getting worse, say to seek emergency care (call 112). You may briefly share general information from the context about relevant treatments and suggest questions to ask the doctor. Do not say "the context doesn't include this" for these questions. The care team on this platform helps with hospital and treatment options, not with assessing symptoms.
4. Costs and hospital-stay durations are approximate ranges from the platform. Say so, and mention they vary by hospital, city, and case. When a passage gives a "platform summary" cost or stay, lead with that range, then add any detail from other passages.
5. "BEST" / RECOMMENDATION QUESTIONS: never name a single best hospital or doctor. Passages marked RANKING come from a transparent scoring table built only from platform-listed facts (rating, accreditation, beds, doctors, surgeon experience). Present them in that order as a comparison and state the facts behind each. A standard disclaimer about the scores is appended automatically, so don't add your own. If a fact is missing for a hospital, say it wasn't available rather than saying it doesn't exist. Finish by suggesting what the patient should weigh (location, budget, surgeon, insurance).
6. If a NOTE says nothing matched: relay it plainly in your own words, do NOT list any hospitals or doctors, say that availability elsewhere is not confirmed by the platform data, and suggest requesting a callback or considering the cities the NOTE names.
7. Be clear, warm, and concise (under 220 words unless asked for detail). Use plain language.
8. Only state a fact about a group ("all of them are JCI accredited") if every item in your answer shows it. If hospitals differ, say which ones have it. Do not add summary claims that go beyond the rows you listed."""


def load_json(name):
    return json.loads(Path(f"data/{name}.json").read_text(encoding="utf-8"))


# ---------------- retrieval ----------------
class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.col = chromadb.PersistentClient(path="data/chroma").get_collection("qurovo")
        data = self.col.get(include=["documents", "metadatas"])
        self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
        self.bm25 = BM25Okapi([self._tok(d) for d in self.docs])
        self.idx = {cid: i for i, cid in enumerate(self.ids)}

    @staticmethod
    def _tok(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def retrieve(self, query, k=5, doc_type=None, pool=20):
        # dense candidates
        q = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=[q], n_results=pool,
                             where={"type": doc_type} if doc_type else None)
        dense_ids = res["ids"][0]

        # keyword candidates
        scores = self.bm25.get_scores(self._tok(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        bm_ids = []
        for i in order:
            if doc_type and self.metas[i]["type"] != doc_type:
                continue
            bm_ids.append(self.ids[i])
            if len(bm_ids) == pool:
                break

        # reciprocal rank fusion
        fused = {}
        for ranking in (dense_ids, bm_ids):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
        candidates = sorted(fused, key=fused.get, reverse=True)[:pool]

        # cross-encoder rerank
        pairs = [(query, self.docs[self.idx[c]]) for c in candidates]
        rr = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, rr), key=lambda x: x[1], reverse=True)[:k]
        return [{"text": self.docs[self.idx[c]], "meta": self.metas[self.idx[c]], "score": float(s)}
                for c, s in ranked]


# ---------------- LLM calls ----------------
def call_llm(history, user_message, system=SYSTEM_PROMPT, json_mode=False):
    messages = history + [{"role": "user", "content": user_message}]

    # ---------- Anthropic ----------
    if PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(model=LLM_MODEL, max_tokens=900,
                                      system=system, messages=messages)
        return resp.content[0].text

    # ---------- Groq / OpenAI-compatible ----------
    from openai import OpenAI
    if PROVIDER == "groq":
        client = OpenAI(base_url="https://api.groq.com/openai/v1",
                        api_key=os.environ["GROQ_API_KEY"])
    else:
        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)

    kwargs = dict(model=LLM_MODEL,
                  messages=[{"role": "system", "content": system}] + messages)
    if PROVIDER == "groq" and LLM_MODEL.startswith("openai/gpt-oss"):
        kwargs["max_completion_tokens"] = 2000   # reasoning tokens count toward the limit
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["max_tokens"] = 900
        kwargs["temperature"] = 0.2
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


# ---------------- filter extraction (LLM + keyword fallback) ----------------
def keyword_filters(question):
    """Deterministic fallback: match treatment, city, accreditation by keywords."""
    q = question.lower()

    best, best_key = None, (0, 0)
    for t in load_json("treatments"):
        words = [w for w in re.findall(r"[a-z]+", t["title"].lower())
                 if len(w) >= 4 and w not in GENERIC_WORDS]
        score = sum(1 for w in words if w in q)
        key = (score, -len(t["title"]))        # ties go to the shorter, more specific title
        if score > 0 and key > best_key:
            best, best_key = t["title"], key

    aliases = {"bangalore": "Bengaluru", "gurgaon": "Gurugram", "bombay": "Mumbai"}
    city = None
    for c in sorted({h["city"] for h in load_json("hospitals")}):
        if c.lower() in GENERIC_CITY_WORDS:
            continue
        if re.search(rf"\b{re.escape(c.lower())}\b", q):
            city = c
    for alias, canon in aliases.items():
        if re.search(rf"\b{alias}\b", q):
            city = canon

    require = [a for a in ("NABH", "NABL", "JCI") if a.lower() in q]
    return {"treatment": best, "city": city, "require": require}


def extract_filters(question):
    """LLM extraction with keyword fallback, so one bad JSON reply can't break routing."""
    titles = [t["title"] for t in load_json("treatments")]
    cities = sorted({h["city"] for h in load_json("hospitals")
                     if h["city"].lower() not in GENERIC_CITY_WORDS})
    system = (
        "You extract search filters from a patient's question. Reply with JSON only, in this shape: "
        '{"treatment": string|null, "city": string|null, "require": []}. '
        "treatment must be copied EXACTLY from the TREATMENTS list, or null. "
        "city must be copied EXACTLY from the CITIES list, or null. "
        'require is a subset of ["NABH","NABL","JCI"] and only if the user explicitly asks for it.\n'
        f"TREATMENTS: {json.dumps(titles)}\nCITIES: {json.dumps(cities)}"
    )
    llm = {"treatment": None, "city": None, "require": []}
    try:
        raw = call_llm([], question, system=system, json_mode=True)
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        llm = {
            "treatment": data.get("treatment") if data.get("treatment") in titles else None,
            "city": data.get("city") if data.get("city") in cities else None,
            "require": [a for a in data.get("require", []) if a in ("NABH", "NABL", "JCI")],
        }
    except Exception as e:
        if os.getenv("DEBUG_ROUTE"):
            print(f"[route] LLM filter extraction failed: {e}")

    kw = keyword_filters(question)
    return {
        "treatment": llm["treatment"] or kw["treatment"],
        "city": llm["city"] or kw["city"],
        "require": sorted(set(llm["require"]) | set(kw["require"])),
    }


# ---------------- structured passages ----------------
def treatment_passage(title):
    for t in load_json("treatments"):
        if t["title"] == title:
            return {
                "title": t["title"], "url": t["url"],
                "text": (f"{t['title']}: platform summary says typical total cost {t['cost_range']} "
                         f"and approx hospital stay {t['hospital_stay']}.")}
    return None


def ranking_passages(flt):
    """Turn recommender output into passages the LLM can cite."""
    res = recommend(flt["treatment"], flt["city"], flt["require"], k=5)
    passages = []
    if res["treatment"]:
        p = treatment_passage(res["treatment"]["title"])
        if p:
            passages.append(p)
    for rank, r in enumerate(res["results"], 1):
        h = r["hospital"]
        facts = "; ".join(text for _, text, _ in r["reasons"])
        miss = ("; not available: " + "; ".join(r["missing"])) if r["missing"] else ""
        passages.append({
            "title": h["name"], "url": h["url"],
            "text": f"RANKING #{rank}: {h['name']} ({h['city']}), table score {r['score']:.0f}/100. Facts: {facts}{miss}."})
    return passages, res["note"]


def clean_output(text):
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"\[(\d+)(?:†[^\]]*)?\]", r"[\1]", text)     # [2†L1-L3] -> [2]
    text = text.replace("\u2011", "-").replace("\u2010", "-")  # non-breaking hyphens
    return text


# ---------------- answering ----------------
def answer(retriever, question, history, doc_type=None):
    """Returns (answer_text, sources, new_history)."""
    if EMERGENCY_PATTERNS.search(question):
        return EMERGENCY_REPLY, [], history

    # symptom / diagnosis requests get a fixed safe reply, unless the user is
    # really asking about cost, stay, or hospitals (the prompt handles those)
    if SYMPTOM_PATTERNS.search(question) and not (
            RECO_HINT.search(question) or FACT_HINT.search(question)):
        return SYMPTOM_REPLY, [], history

    # short follow-ups ("and in Mumbai?") borrow the previous question for retrieval
    search_q = question
    if len(question.split()) < 6 and history:
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev:
            search_q = f"{prev[-1]} {question}"

    passages, note, no_match = [], None, False
    debug = bool(os.getenv("DEBUG_ROUTE"))

    if RECO_HINT.search(search_q):
        # recommendation questions use the structured ranking table
        flt = extract_filters(search_q)
        if debug:
            print(f"[route] recommender filters: {flt}")
        if flt["treatment"]:
            passages, note = ranking_passages(flt)
            no_match = note is not None
    else:
        # cost / stay questions get the platform's summary table values
        if FACT_HINT.search(search_q):
            kw = keyword_filters(search_q)
            if kw["treatment"]:
                p = treatment_passage(kw["treatment"])
                if p:
                    passages.append(p)
        if debug:
            print(f"[route] text retrieval{' + treatment summary' if passages else ' only'}")

    # when the recommender found nothing, don't feed unrelated chunks to the LLM
    chunks = [] if no_match else retriever.retrieve(
        search_q, k=4 if passages else 5, doc_type=doc_type)
    for c in chunks:
        passages.append({"title": c["meta"]["title"], "url": c["meta"]["url"], "text": c["text"]})

    context = "\n\n".join(f"[{i}] ({p['title']}) {p['text']}" for i, p in enumerate(passages, 1))
    if note:
        context = f"NOTE: {note}\n\n" + context
    text = clean_output(call_llm(history, f"CONTEXT:\n{context}\n\nQUESTION: {question}"))

    if any(p["text"].startswith("RANKING") for p in passages):
        text += RANKING_DISCLAIMER

    # show only the sources the model actually cited
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    sources, seen = [], set()
    for i, p in enumerate(passages, 1):
        if i in cited and p["url"] not in seen:
            seen.add(p["url"])
            sources.append((i, p["title"], p["url"]))

    new_history = history + [{"role": "user", "content": question},
                             {"role": "assistant", "content": text}]
    return text, sources, new_history[-8:]