"""Workspace-aware assistant: guardrails, routing, structured ranking + document retrieval."""
import json
import os
import re

import threading

from .cities import mentioned_cities
from .intake import IntakeFlow, deliver
from .llm import call_llm
from .ranking import find_procedure, rank
from .workspace import Workspace

SYMPTOM_PATTERNS = re.compile(
    r"(what (disease|condition|illness|disorder|problem) (do|could|might|can) i have|"
    r"what('?s| is) wrong with me|(diagnose|diagnosis for) me|"
    r"do i have (cancer|a tumou?r|diabetes|a heart)|"
    r"\bi(?: have| am having|'ve had| have had| keep getting| have been having|'ve been having| am experiencing) "
    r"(?:a |an |some |severe |sharp )?"
    r"(?:headache|migraine|fever|chest pain|pain|dizz\w+|nausea|vomiting|blurr\w+|numbness|"
    r"swelling|bleeding|cough|breathless\w*|palpitations?))", re.I)
EMERGENCY_PATTERNS = re.compile(
    r"(chest pain (right )?now|can'?t breathe|cannot breathe|severe bleeding|"
    r"unconscious|stroke symptoms|heart attack (now|right now)|suicid|kill myself|want to die)", re.I)
RECO_HINT = re.compile(
    r"\b(best|top|recommend\w*|which (hospitals?|providers?|clinics?)|compare|comparison|"
    r"good (hospitals?|clinics?)|accredited|(hospitals?|clinics?) (for|in|near)|where (to|should))\b", re.I)
FACT_HINT = re.compile(
    r"\b(cost|costs|price|how much|expensive|cheap|stay|how many days|duration|budget|charges?|"
    r"how long[^?.]{0,30}\b(stay|stays|hospital|admitted|admission|discharge))\b", re.I)
GENERIC_WORDS = {"surgery", "treatment", "therapy", "repair", "procedure", "transplant"}
NOT_A_PLACE = {"india", "general", "detail", "depth", "particular", "total", "terms", "case", "order",
               "advance", "pain", "my area", "my city"}
OFFTOPIC_PATTERNS = re.compile(
    r"\b(write|compose|generate|create|make|draft)\b[^?.!]{0,40}\b(poem|poetry|song|lyrics|story|essay|joke|"
    r"limerick|haiku|python|javascript|code|script|program|homework)\b|"
    r"\b(tell me a (joke|story)|sing me|translate this|solve this (equation|problem))\b", re.I)

PERSONAL_PATTERNS = re.compile(
    r"((my|his|her|our)\s+(\w+\s+)?(report|reports|scan|scans|mri|ct|ecg|test results?|angiogram)\b|"
    r"\b(father|mother|dad|mom|husband|wife|son|daughter|parent|brother|sister|friend)'?s?\s+"
    r"(report|reports|scan|scans|mri|ct|ecg|test results?|angiogram)\b|"
    r"\b(report|scan|results?|angiogram)\s+(shows?|says?|showed|indicates?)\b|"
    r"\bdoes\s+(he|she|my \w+|\w+)\s+need\s+(a |an |the )?(surgery|operation|stent|bypass)\b|"
    r"\bshould\s+(i|he|she|we)\s+(get|have|undergo|go for)\s+(a |an |the )?"
    r"(surgery|operation|stent|bypass|angioplasty)\b|"
    r"\b\d{1,3}\s?%\s+(blockage|block|stenosis|blocked))", re.I)

INTAKE_INTENT = re.compile(
    r"\b(call me( back)?|call back|callback|contact me|get in touch|reach out to me|"
    r"book (an? )?(appointment|consultation|slot|visit)|schedule (an? )?(appointment|consultation|visit)|"
    r"(talk|speak) (to|with) (someone|a person|an? (agent|counsell?or|coordinator|human|representative|doctor))|"
    r"connect me|get (an? )?(quote|estimate)|send me (an? )?(quote|estimate)|arrange (an? )?(appointment|call)|"
    r"i('d| would) like (to )?(book|speak|talk|be contacted)|want to (book|be contacted)|"
    r"have (someone|the team) (call|contact))\b", re.I)

MED_WORDS = re.compile(
    r"\b(dose|doses|dosage|dosing|mg|milligrams?|tablets?|pills?|capsules?|medicines?|medications?|drugs?|"
    r"antibiotics?|painkillers?|blood thinners?|aspirin|paracetamol|ibuprofen|insulin|statins?|clopidogrel|"
    r"warfarin)\b", re.I)
MED_EXCLUDE = re.compile(r"\b(cost|costs|price|prices|charges?|fees?|budget|expensive|cheap|insurance)\b", re.I)
MED_ACTION = re.compile(r"\b(take|taking|how much|how many|which|what|should|can i|stop|skip|safe)\b", re.I)
STAY_Q = re.compile(r"\b(stay|how many days|duration|how long[^?.]{0,30}\b(stay|hospital|admitted|admission|discharge))\b", re.I)
COST_Q = re.compile(r"\b(cost|costs|price|how much|expensive|cheap|budget|charges?)\b", re.I)

NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text):
    """All numbers in a text as floats (commas removed, citation markers [n] ignored)."""
    text = re.sub(r"\[\d+\]", " ", text)
    out = set()
    for m in NUM_RE.findall(text):
        try:
            out.add(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def system_prompt(ws):
    name = ws.config["name"]
    return f"""You are the patient-information assistant for {name}, a healthcare directory.

Rules:
1. Answer ONLY from the numbered CONTEXT passages. If the context doesn't contain the answer, say you don't have that information and suggest contacting the {name} team. Never fill gaps from your own knowledge.
2. Cite sources inline like [1] or [2][3] after the claims they support. Only cite passages you actually used. Use plain square brackets like [1], never other bracket styles.
3. PERSONAL MEDICAL QUESTIONS (a specific report, symptom, blockage percentage, "does he/she need surgery?", "what disease do I have?"): never diagnose, interpret, or suggest what condition it could be. Say clearly that you can't assess this, that it needs a qualified doctor who can examine the person or review the full report, and encourage seeing a doctor soon. If symptoms sound sudden, severe, or are getting worse, say to seek emergency care (call {ws.config['emergency_number']}). You may briefly share general information from the context about relevant treatments and suggest questions to ask the doctor. Do not say "the context doesn't include this" for these questions. The {name} team helps with provider and treatment options, not with assessing symptoms.
4. Costs and stay durations are approximate ranges. Say so, and mention they vary by provider, city, and case. When a passage gives a "platform summary" cost or stay, lead with that range, then add any detail from other passages.
5. "BEST" / RECOMMENDATION QUESTIONS: never name a single best provider or doctor. Passages marked RANKING come from a transparent scoring table built only from listed facts. Present them in that order as a comparison and state the facts behind each. A standard disclaimer about the scores is appended automatically, so don't add your own. If a fact is missing for a provider, say it wasn't available rather than saying it doesn't exist. Finish by suggesting what the patient should weigh (location, budget, surgeon, insurance).
6. If a NOTE says nothing matched: relay it plainly in your own words, do NOT list any providers or doctors, say that availability elsewhere is not confirmed by the data, and suggest contacting the {name} team or considering the cities the NOTE names.
7. Be clear, warm, and concise (under 220 words unless asked for detail). Use plain language.
9. If the question is not about treatments, hospitals, doctors, or costs in this directory (for example general knowledge, sports, entertainment, coding, or creative writing), do not answer it. Reply with exactly: "I can only help with questions about treatments, hospitals, and costs in the {name} directory, so I can't help with that. Is there a treatment or hospital you'd like to know about?"
8. Only state a fact about a group ("all of them are JCI accredited") if every item in your answer shows it. If providers differ, say which ones have it. Do not add summary claims that go beyond the rows you listed."""


def clean_output(text):
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"\[(\d+)(?:†[^\]]*)?\]", r"[\1]", text)
    text = text.replace("\u2011", "-").replace("\u2010", "-")
    # the model sometimes uses a zero-width character where a space belongs ("withthat"):
    # restore the space when both neighbours are plain ASCII (Hindi etc. is left untouched)
    text = re.sub(r"(?<=[A-Za-z0-9.,;:)\]])[\u200b-\u200d\u2060-\u2064\ufeff](?=[A-Za-z0-9(\[])", " ", text)
    text = re.sub(r"[\u200b\u2060\ufeff]", "", text)
    text = re.sub(r"[\u00a0\u2007\u2009\u200a\u202f]", " ", text)
    return re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)


class Assistant:
    def __init__(self, ws: Workspace, retriever, llm=None):
        self.ws, self.retriever = ws, retriever
        raw = llm or (lambda history, msg, system, json_mode=False:
                      call_llm(ws, history, msg, system, json_mode))

        def counted(history, msg, system, json_mode=False):
            self.last_meta["llm_calls"] = self.last_meta.get("llm_calls", 0) + 1
            return raw(history, msg, system, json_mode)
        self.llm = counted
        self.system = system_prompt(ws)
        self.last_route = ""
        self.last_meta = {}
        n = ws.config["emergency_number"]
        self.emergency_reply = (
            f"This sounds like it could be an emergency. Please call {n} or go to the nearest "
            "emergency room right away. I can't help with urgent medical situations, but once "
            "the situation is safe, I'm glad to help you explore treatment options.")
        self.symptom_reply = (
            "I can't assess symptoms or tell you what might be causing them. That needs a doctor who "
            "can examine you or review your tests, so please see one soon. If your symptoms are sudden, "
            "severe, or getting worse (for example sudden vision loss, a very severe headache, weakness, "
            f"trouble speaking, chest pain, or trouble breathing), call {n} or go to the nearest "
            "emergency room. Once you have a diagnosis, I'm happy to help you compare treatment options.")
        self.medication_reply = (
            "I can't advise on medicines, doses, or whether to start or stop a medication. That depends on "
            "the person's health and other medicines, so please ask the treating doctor or a pharmacist. "
            f"If someone has taken too much of a medicine or feels very unwell, call {n} right away. "
            "I'm happy to help with treatment options and hospitals.")
        self.scope_reply = (
            f"I can only help with questions about treatments, hospitals, and costs in the {ws.config['name']} "
            "directory, so I can't help with that. Is there a treatment or hospital you'd like to know about?")
        self.personal_reply = (
            "I can't interpret a specific report or tell you whether surgery is needed. That decision "
            "belongs to the treating doctor, who can review the full report, symptoms, and overall health. "
            "It can help to ask: What do these results mean? What are all the treatment options, including "
            "non-surgical ones, and what are their risks? Would a second opinion be worthwhile? "
            f"If there are new or worsening symptoms (for example chest pain, breathlessness, or fainting), "
            f"call {n} right away. If you'd like, I can explain what a procedure involves or compare "
            "providers for it.")

    def ungrounded(self, text, corpus):
        """Numbers over 10 in the answer that appear nowhere in the context or question."""
        allowed = numbers_in(corpus) | {float(self.ws.config["emergency_number"])
                                        if str(self.ws.config["emergency_number"]).isdigit() else 0.0}
        return sorted(n for n in numbers_in(text)
                      if n > 10 and not 1900 <= n <= 2100 and n not in allowed)

    def ranking_fallback(self, passages):
        lines = ["Here are the matching providers, in scoring-table order:"]
        for i, p in enumerate(passages, 1):
            lines.append(f"- [{i}] " + p["text"].replace("RANKING #", "#"))
        return "\n".join(lines)

    def safe_fallback(self, passages, fact_proc):
        """What to say when a number in the model's answer cannot be verified against the data."""
        if any(p["text"].startswith("RANKING") for p in passages):
            return self.ranking_fallback(passages)
        name = self.ws.config["name"]
        line = self.summary_line(fact_proc) if fact_proc else None
        if line:
            return f"{line} [1]\n\nFor anything beyond this, please contact the {name} team."
        return ("I couldn't verify some figures in my answer against the directory, so I'd rather not "
                f"state them. Please contact the {name} team for exact details.")

    def summary_line(self, proc):
        bits = []
        if proc.get("cost_min"):
            bits.append(f"typical total cost {self.ws.money(proc['cost_min'])} to {self.ws.money(proc['cost_max'])}")
        if proc.get("stay_min_days"):
            bits.append(f"approx hospital stay {proc['stay_min_days']:.0f}-{proc['stay_max_days']:.0f} days")
        return f"Directory summary for {proc['name']}: " + "; ".join(bits) + "." if bits else None

    def ensure_summary(self, text, question, proc, idx):
        """If a cost or stay question was answered without the directory's listed range, put it first."""
        nums = {round(n, 2) for n in numbers_in(text)}

        def has(v):
            return v is None or any(round(x, 2) in nums for x in (v, v / 1e5, v / 1e7))
        missing = False
        if STAY_Q.search(question) and proc.get("stay_min_days"):
            missing |= not (has(proc["stay_min_days"]) and has(proc["stay_max_days"]))
        if COST_Q.search(question) and proc.get("cost_min"):
            missing |= not (has(proc["cost_min"]) and has(proc["cost_max"]))
        line = self.summary_line(proc) if missing else None
        return f"{line} [{idx}]\n\n{text}" if line else text

    # ---------- filters ----------
    def keyword_filters(self, question):
        q, ws = question.lower(), self.ws
        best, best_key = None, (0, 0)
        for p in ws.procedures:
            words = [w for w in re.findall(r"[a-z]+", p["name"].lower())
                     if len(w) >= 4 and w not in GENERIC_WORDS]
            score = sum(1 for w in words if w in q)
            key = (score, -len(p["name"]))
            if score > 0 and key > best_key:
                best, best_key = p["name"], key

        # the city mentioned LAST wins ("... in Delhi and in Mumbai" -> Mumbai)
        found = []
        for c in {p["city"] for p in ws.providers}:
            m = None
            for m in re.finditer(rf"\b{re.escape(c.lower())}\b", q):
                pass
            if m:
                found.append((m.start(), c))
        for alias, canon in ws.config["city_aliases"].items():
            m = None
            for m in re.finditer(rf"\b{re.escape(alias)}\b", q):
                pass
            if m:
                found.append((m.start(), canon))
        city = max(found)[1] if found else None
        known = {p["city"] for p in ws.providers}
        unknown = None
        later = [(pos, c) for pos, c in mentioned_cities(q) if c not in known]
        if later and (city is None or max(later)[0] > max(found)[0]):
            unknown = max(later)[1]
        accs = ws.config["ranking"]["accreditation_weights"].keys()
        require = [a for a in accs if re.search(rf"\b{re.escape(a.lower())}\b", q)]
        return {"treatment": best, "city": city, "unknown_city": unknown, "require": require}

    @staticmethod
    def _place_phrase(question):
        m = re.search(r"\b(?:in|near|at|around)\s+([A-Za-z][A-Za-z .-]{2,30}?)\s*[?.!]*\s*$", question.strip())
        return m.group(1).strip().lower() if m else None

    def extract_filters(self, question):
        ws = self.ws
        kw0 = self.keyword_filters(question)
        if kw0["treatment"]:                     # confident: no LLM call needed
            phrase = self._place_phrase(question)
            if kw0["city"] or kw0["unknown_city"] or not phrase or phrase in NOT_A_PLACE:
                return kw0
        names = [p["name"] for p in ws.procedures]
        cities = sorted({p["city"] for p in ws.providers})
        accs = list(ws.config["ranking"]["accreditation_weights"].keys())
        system = (
            "You extract search filters from a patient's question. Reply with JSON only, in this shape: "
            '{"treatment": string|null, "city": string|null, "city_mentioned": string|null, "require": []}. '
            "treatment must be copied EXACTLY from the TREATMENTS list, or null. "
            "city must be copied EXACTLY from the CITIES list, or null. "
            "city_mentioned is any place name the user wrote, exactly as written, or null. "
            f"require is a subset of {json.dumps(accs)} and only if the user explicitly asks for it.\n"
            f"TREATMENTS: {json.dumps(names)}\nCITIES: {json.dumps(cities)}")
        llm = {"treatment": None, "city": None, "require": [], "mentioned": None}
        try:
            raw = self.llm([], question, system, True)
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            llm = {"treatment": data.get("treatment") if data.get("treatment") in names else None,
                   "city": data.get("city") if data.get("city") in cities else None,
                   "require": [a for a in data.get("require", []) if a in accs],
                   "mentioned": data.get("city_mentioned")}
        except Exception as e:
            if os.getenv("DEBUG_ROUTE"):
                print(f"[route] LLM filter extraction failed: {e}")
        kw = self.keyword_filters(question)
        city = llm["city"] or kw["city"]
        unknown = kw["unknown_city"]
        m = llm["mentioned"]
        if (not city and not unknown and isinstance(m, str) and m.strip()
                and m.strip().lower() not in NOT_A_PLACE and ws.canon_city(m) not in cities):
            unknown = m.strip()
        return {"treatment": llm["treatment"] or kw["treatment"], "city": city, "unknown_city": unknown,
                "require": sorted(set(llm["require"]) | set(kw["require"]))}

    # ---------- structured passages ----------
    def procedure_passage(self, name):
        p = find_procedure(self.ws, name)
        if not p:
            return None
        bits = []
        if p["cost_min"]:
            bits.append(f"typical total cost {self.ws.money(p['cost_min'])} to {self.ws.money(p['cost_max'])}")
        if p["stay_min_days"]:
            bits.append(f"approx stay {p['stay_min_days']:.0f}-{p['stay_max_days']:.0f} days")
        if not bits:
            return None
        return {"title": p["name"], "url": p.get("url") or "",
                "text": f"{p['name']}: platform summary says " + " and ".join(bits) + "."}

    def ranking_passages(self, flt):
        res = rank(self.ws, flt["treatment"], flt["city"], flt["require"], k=5)
        passages = []
        self._shown = [r["provider"]["id"] for r in res["results"]]
        if res["procedure"]:
            pp = self.procedure_passage(res["procedure"]["name"])
            if pp:
                passages.append(pp)
        for i, r in enumerate(res["results"], 1):
            h = r["provider"]
            facts = "; ".join(t for _, t, _ in r["reasons"])
            if r["price_min"]:
                facts += f"; quoted price {self.ws.money(r['price_min'])} to {self.ws.money(r['price_max'])}"
            miss = ("; not available: " + "; ".join(r["missing"])) if r["missing"] else ""
            passages.append({"title": h["name"], "url": h.get("url") or "",
                             "text": f"RANKING #{i}: {h['name']} ({h['city']}), table score {r['score']:.0f}/100. Facts: {facts}{miss}."})
        return passages, res["note"]

    # ---------- stateful entry (used by the web server and the chat CLI) ----------
    sync_delivery = False

    def _flow(self):
        if self.sync_delivery:
            return IntakeFlow(self.ws, self)
        def background(ws, lead):
            threading.Thread(target=deliver, args=(ws, lead), daemon=True).start()
            return lead
        return IntakeFlow(self.ws, self, deliver_fn=background)

    def _intake_meta(self):
        self.last_route = "intake"
        self.last_meta = {"route": "intake", "filters": None, "retried": False, "first_draft_bad": [],
                          "ungrounded": [], "llm_calls": 0, "procedure": None, "city": None,
                          "shown_providers": []}

    def converse(self, question, session):
        """Like answer(), but keeps per-session state: the lead-intake flow and what was discussed.
        Returns {"text", "sources", "quick_replies"}; `session` is a dict owned by the caller."""
        session.setdefault("history", [])
        ctx = session.setdefault("context", {"topics": []})
        if EMERGENCY_PATTERNS.search(question):
            session["intake"] = None                       # an emergency always interrupts any intake
        elif session.get("intake"):
            res = self._flow().handle(session["intake"], question)
            session["intake"] = res["state"]
            self._intake_meta()
            return {"text": res["reply"], "sources": [], "quick_replies": res["quick"]}
        elif self.ws.config["handoff"]["enabled"] and INTAKE_INTENT.search(question):
            res = self._flow().start(ctx, session.get("sid", ""))
            session["intake"] = res["state"]
            self._intake_meta()
            return {"text": res["reply"], "sources": [], "quick_replies": res["quick"]}

        text, sources, hist = self.answer(question, session["history"])
        session["history"] = hist
        m = self.last_meta
        for key in ("procedure", "city"):
            if m.get(key):
                ctx[key] = m[key]
        if m.get("shown_providers"):
            ctx["shown_providers"] = m["shown_providers"]
        if m.get("procedure") and m["procedure"] not in ctx["topics"]:
            ctx["topics"].append(m["procedure"])
        return {"text": text, "sources": sources, "quick_replies": []}

    # ---------- main entry ----------
    def answer(self, question, history, doc_type=None):
        """Returns (text, sources, new_history). self.last_route / self.last_meta say how it was answered."""
        self.last_meta = {"route": None, "filters": None, "retried": False,
                          "first_draft_bad": [], "ungrounded": [], "llm_calls": 0,
                          "procedure": None, "city": None, "shown_providers": []}
        self._shown = []
        if EMERGENCY_PATTERNS.search(question):
            self.last_route = self.last_meta["route"] = "emergency"
            return self.emergency_reply, [], history
        if SYMPTOM_PATTERNS.search(question) and not (RECO_HINT.search(question) or FACT_HINT.search(question)):
            self.last_route = self.last_meta["route"] = "symptom"
            return self.symptom_reply, [], history

        if PERSONAL_PATTERNS.search(question) and not (RECO_HINT.search(question) or FACT_HINT.search(question)):
            self.last_route = self.last_meta["route"] = "personal"
            return self.personal_reply, [], history

        if (MED_WORDS.search(question) and MED_ACTION.search(question)
                and not (RECO_HINT.search(question) or MED_EXCLUDE.search(question))):
            self.last_route = self.last_meta["route"] = "medication"
            return self.medication_reply, [], history

        if OFFTOPIC_PATTERNS.search(question) and not (RECO_HINT.search(question) or FACT_HINT.search(question)):
            self.last_route = self.last_meta["route"] = "scope"
            return self.scope_reply, [], history

        search_q = question
        if len(question.split()) < 6 and history and not self.keyword_filters(question)["treatment"]:
            prev = [m["content"] for m in history if m["role"] == "user"]
            if prev:
                search_q = f"{prev[-1]} {question}"

        passages, note, no_match, fact_proc = [], None, False, None
        ctx_kw = self.keyword_filters(search_q)
        self.last_meta["procedure"], self.last_meta["city"] = ctx_kw["treatment"], ctx_kw["city"]
        if RECO_HINT.search(search_q):
            flt = self.extract_filters(search_q)
            self.last_route = f"recommender {flt}"
            self.last_meta.update(route="recommender", filters=flt, procedure=flt["treatment"], city=flt["city"])
            if flt["unknown_city"]:
                cities = ", ".join(sorted({p["city"] for p in self.ws.providers}))
                note = (f"No providers are listed in {flt['unknown_city']}. "
                        f"Cities with listed providers: {cities}.")
                no_match = True
            elif flt["treatment"]:
                passages, note = self.ranking_passages(flt)
                no_match = note is not None
        else:
            self.last_route = "documents"
            self.last_meta["route"] = "documents"
            if FACT_HINT.search(search_q):
                kw = self.keyword_filters(search_q)
                if kw["treatment"]:
                    pp = self.procedure_passage(kw["treatment"])
                    if pp:
                        passages.append(pp)
                        fact_proc = find_procedure(self.ws, kw["treatment"])
                        self.last_route = "documents + procedure summary"

        self.last_meta["shown_providers"] = list(self._shown)
        if no_match and note:                    # nothing to rank: say so directly, no LLM needed
            text = (f"{note} You can also contact the {self.ws.config['name']} team for help "
                    "finding the right option.")
            self.last_meta["deterministic"] = True
            return text, [], (history + [{"role": "user", "content": question},
                                         {"role": "assistant", "content": text}])[-8:]

        chunks = [] if no_match else self.retriever.retrieve(
            search_q, k=4 if passages else 5, doc_type=doc_type)
        for c in chunks:
            passages.append({"title": c["meta"]["title"], "url": c["meta"]["url"], "text": c["text"]})

        context = "\n\n".join(f"[{i}] ({p['title']}) {p['text']}" for i, p in enumerate(passages, 1))
        if note:
            context = f"NOTE: {note}\n\n" + context
        prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        text = clean_output(self.llm(history, prompt, self.system))

        corpus = context + " " + question
        bad = self.ungrounded(text, corpus)
        if bad:
            self.last_meta["retried"] = True
            self.last_meta["first_draft_bad"] = bad
            retry = (prompt + "\n\nIMPORTANT: your previous answer used numbers that are not in the CONTEXT: "
                     f"{bad}. Rewrite the answer using only numbers exactly as they appear in the CONTEXT.")
            text = clean_output(self.llm(history, retry, self.system))
            bad = self.ungrounded(text, corpus)
            if bad:
                self.last_meta["ungrounded"] = bad
                text = self.safe_fallback(passages, fact_proc)
                self.last_route += f" | ungrounded numbers {bad}"

        if fact_proc:
            text = self.ensure_summary(text, question, fact_proc, 1)
        if any(p["text"].startswith("RANKING") for p in passages):
            text += f"\n\n_{self.ws.config['disclaimer']}_"

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        sources, seen = [], set()
        for i, p in enumerate(passages, 1):
            key = p["url"] or p["title"]
            if i in cited and key not in seen:
                seen.add(key)
                sources.append((i, p["title"], p["url"]))
        new_history = history + [{"role": "user", "content": question},
                                 {"role": "assistant", "content": text}]
        return text, sources, new_history[-8:]