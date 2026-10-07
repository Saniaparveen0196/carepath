"""
Consent-based lead intake and handoff.

Design rules (this module handles personal data, so it is deliberately boring and deterministic):
  * a fixed state machine in code decides every step; the LLM is never involved
  * nothing is collected before an explicit, unambiguous "yes" to the consent text
  * only structured fields are kept: no free text, no medical details, no chat transcript
  * nothing is stored until the patient confirms the summary
  * logs only ever contain masked phone numbers
  * delivery goes to the customer's own webhook (HMAC-signed, retried, idempotent)
"""
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("intake")

YES_WORDS = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "agree", "haan", "ha", "हाँ", "हां", "please", "go"}
NO_WORDS = {"no", "n", "nope", "nah", "nahi", "नहीं", "never"}
HEDGES = re.compile(r"\b(but|dont|don't|not|except|only|maybe|unless|without)\b", re.I)
CANCEL = re.compile(r"\b(cancel|stop|never ?mind|forget it|quit|abort|exit)\b", re.I)
SKIP = {"skip", "none", "na", "n/a", "pass", "-", "no", "nothing", "not now"}
NAME_PREFIX = re.compile(r"^(my name is|i am|i'm|this is|it is|it's|name is|call me)\s+", re.I)
NOT_A_NAME = {"pain", "hurts", "hurt", "report", "surgery", "need", "want", "help", "please", "call", "doctor",
              "hospital", "cost", "price", "blockage", "cancer", "chest", "scan", "treatment", "operation"}
URGENCY = [("asap", re.compile(r"\b(asap|urgent|urgently|immediately|as soon as possible|this week|right away|"
                                r"soonest|emergency)\b", re.I)),
           ("within_month", re.compile(r"\b(within a month|next month|a month|few weeks|weeks|month)\b", re.I)),
           ("planned", re.compile(r"\b(plan|planned|few months|later|not sure|no rush|flexible|exploring|"
                                  r"just looking|next year|months)\b", re.I))]
URGENCY_LABEL = {"asap": "as soon as possible", "within_month": "within a month", "planned": "planned (next few months)"}
INSURERS = re.compile(r"\b(star health|hdfc ergo|icici lombard|niva bupa|care health|bajaj allianz|tata aig|"
                      r"max bupa|new india|united india|oriental|national insurance|cashless|mediclaim|"
                      r"insurance|insured|ayushman|cghs|esi)\b", re.I)


# ============================================================ small parsers
def _norm(t):
    return re.sub(r"[^\w\u0900-\u097F' ]", " ", t.lower()).strip()


def is_yes(text):
    toks = _norm(text).replace("'", "").split()
    if not toks or len(toks) > 6 or HEDGES.search(" ".join(toks)):
        return False
    return toks[0] in YES_WORDS


def is_no(text):
    toks = _norm(text).replace("'", "").split()
    return bool(toks) and len(toks) <= 6 and toks[0] in NO_WORDS


def wants_cancel(text):
    return len(text.split()) <= 4 and bool(CANCEL.search(text))


def parse_name(text):
    t = NAME_PREFIX.sub("", text.strip()).strip(" .,!")
    words = t.split()
    if not 1 <= len(words) <= 4 or not 2 <= len(t) <= 60:
        return None
    if not re.fullmatch(r"[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F .'\-]*", t):
        return None
    if any(w.lower().strip(".,'") in NOT_A_NAME for w in words):
        return None
    return " ".join(w[:1].upper() + w[1:] if w.islower() else w for w in words)


def parse_phone(text):
    digits = re.sub(r"\D", "", text)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return f"+91{digits}" if len(digits) == 10 and digits[0] in "6789" else None


def mask_phone(p):
    return (p[:3] + " " + p[3:5] + "******" + p[-2:]) if p and len(p) >= 8 else "***"


def parse_budget(text):
    """Largest amount mentioned, in rupees. Bare small numbers (e.g. '2 days') are ignored."""
    best = None
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr|thousand|k|l)?\b", text.lower()):
        v = float(num)
        mult = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5, "crore": 1e7, "crores": 1e7,
                "cr": 1e7, "k": 1e3, "thousand": 1e3}.get(unit)
        if mult is None:
            if v < 10000:
                continue
            mult = 1
        best = max(best or 0, v * mult)
    return best


def clean_free(text, limit):
    t = re.sub(r"[^A-Za-z\u0900-\u097F &/(),.\-]", "", text).strip()
    return re.sub(r"\s+", " ", t)[:limit]


# ============================================================ storage
class LeadStore:
    def __init__(self, ws):
        self.dir = ws.path / "leads"

    def save(self, lead):
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / f"{lead['id']}.json"
        f.write_text(json.dumps(lead, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            f.chmod(0o600)
        except OSError:
            pass
        return f

    def load(self, lead_id):
        f = self.dir / f"{lead_id}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def all(self):
        if not self.dir.exists():
            return []
        return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(self.dir.glob("*.json"))]

    def find_recent(self, phone, procedure, minutes, now):
        for l in self.all():
            if (l["contact"]["phone"] == phone and l["interest"]["procedure"] == procedure
                    and now - datetime.fromisoformat(l["created_at"]) < timedelta(minutes=minutes)):
                return l
        return None

    def purge(self, days, now=None):
        now = now or datetime.now(timezone.utc)
        n = 0
        for f in list(self.dir.glob("*.json")) if self.dir.exists() else []:
            l = json.loads(f.read_text(encoding="utf-8"))
            if now - datetime.fromisoformat(l["created_at"]) > timedelta(days=days):
                f.unlink()
                n += 1
        return n


# ============================================================ delivery (signed webhook)
def sign(secret, timestamp, body):
    return "sha256=" + hmac.new(secret.encode(), f"{timestamp}.{body}".encode(), hashlib.sha256).hexdigest()


def _http_post(url, body, headers, timeout=8):
    req = urllib.request.Request(url, data=body.encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def deliver(ws, lead, post=None, sleep=time.sleep):
    """Send one lead to the customer's webhook. Returns the updated lead (also saved)."""
    cfg, store = ws.config["handoff"], LeadStore(ws)
    if lead["status"] == "delivered":
        return lead
    url = cfg.get("webhook_url")
    if not url:
        lead["status"] = "stored"                      # no webhook configured: kept locally only
        store.save(lead)
        return lead
    post = post or _http_post
    body = json.dumps({k: v for k, v in lead.items() if k not in ("status", "attempts", "delivery_error")},
                      sort_keys=True, ensure_ascii=False)
    for delay in cfg.get("retry_delays", [0, 2, 6]):
        sleep(delay)
        ts = str(int(time.time()))
        headers = {"Content-Type": "application/json", "X-Timestamp": ts, "Idempotency-Key": lead["id"]}
        if cfg.get("webhook_secret"):
            headers["X-Signature"] = sign(cfg["webhook_secret"], ts, body)
        lead["attempts"] += 1
        try:
            if 200 <= post(url, body, headers) < 300:
                lead["status"], lead["delivery_error"] = "delivered", None
                break
            lead["delivery_error"] = "non-2xx response"
        except (urllib.error.URLError, OSError, ValueError) as e:
            lead["delivery_error"] = type(e).__name__
    else:
        lead["status"] = "failed" if lead["attempts"] >= cfg.get("max_attempts", 10) else "pending"
    store.save(lead)
    log.info("lead delivery id=%s status=%s attempts=%d", lead["id"], lead["status"], lead["attempts"])
    return lead


def deliver_pending(ws, **kw):
    store, done = LeadStore(ws), 0
    for l in store.all():
        if l["status"] == "pending":
            done += deliver(ws, l, **kw)["status"] == "delivered"
    return done


# ============================================================ the conversation
class IntakeFlow:
    def __init__(self, ws, matcher, now=None, deliver_fn=None):
        self.ws, self.matcher = ws, matcher
        self.cfg = ws.config["handoff"]
        self.store = LeadStore(ws)
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.deliver_fn = deliver_fn or deliver

    # ---- texts
    def consent_text(self):
        n = self.ws.config["name"]
        return (f"I can pass your request to the {n} care team so they can call you back. I'd share only what "
                "you tell me here: your name, phone number, the treatment and city you're interested in, and "
                "your timing and budget preferences. I won't ask for medical history or reports, and I won't "
                "share this conversation. The team will use these details only to contact you about your "
                "request. Do you agree to share them?")

    def _ask(self, state):
        step, d, pre = state["step"], state["data"], state["prefill"]
        if step == "name":
            return "Thank you. What name should the team ask for?", []
        if step == "phone":
            return "What's the best mobile number to reach you on? (a 10-digit Indian mobile number)", []
        if step == "procedure":
            if pre.get("procedure") and not state.get("prefill_used_procedure"):
                state["asking_prefill"] = "procedure"
                return f"Is {pre['procedure']} the treatment you'd like help with? (yes, or tell me which one)", ["Yes"]
            return "Which treatment or procedure are you interested in?", []
        if step == "city":
            if pre.get("city") and not state.get("prefill_used_city"):
                state["asking_prefill"] = "city"
                return f"And is {pre['city']} the city you prefer? (yes, or tell me which one)", ["Yes"]
            return "Which city would you prefer to be treated in?", []
        if step == "urgency":
            return ("How soon are you looking to go ahead?",
                    ["Planned (next few months)", "Within a month", "As soon as possible"])
        if step == "extras":
            return ("Is there anything that would help the team prepare, such as a rough budget or your "
                    "insurance provider? You can say 'skip'.", ["Skip"])
        if step == "confirm":
            lines = [f"- Name: {d['name']}", f"- Phone: {d['phone']}",
                     f"- Treatment: {d['procedure']}", f"- City: {d['city']}",
                     f"- Timing: {URGENCY_LABEL[d['urgency']]}"]
            if d.get("budget_inr_max"):
                lines.append(f"- Budget: up to {self.ws.money(d['budget_inr_max'])}")
            if d.get("insurance"):
                lines.append(f"- Insurance: {d['insurance']}")
            shown = [p["name"] for p in self.ws.providers if p["id"] in state["providers"]][:3]
            if shown:
                lines.append("- Hospitals you looked at: " + ", ".join(shown))
            return ("Here is what I would send to the care team:\n" + "\n".join(lines) +
                    "\n\nShall I send this? (yes / no, or say 'edit phone', 'edit name', 'edit treatment', "
                    "'edit city')", ["Yes, send it", "No, don't send"])
        raise ValueError(step)

    # ---- entry points
    def start(self, context, sid):
        state = {"id": f"L-{self.now():%Y%m%d}-{secrets.token_hex(3)}", "step": "consent", "data": {},
                 "attempts": 0, "editing": None, "asking_prefill": None,
                 "prefill": {"procedure": context.get("procedure"), "city": context.get("city")},
                 "providers": list(context.get("shown_providers", [])), "topics": list(context.get("topics", [])),
                 "sid": hashlib.sha256((sid or "").encode()).hexdigest()[:12], "consent_at": None}
        return {"reply": self.consent_text(), "state": state, "quick": ["Yes, share my details", "No, don't share"]}

    def _next(self, state, step):
        state["step"], state["attempts"] = step, 0
        reply, quick = self._ask(state)
        return {"reply": reply, "state": state, "quick": quick}

    def _retry(self, state, hint):
        state["attempts"] += 1
        if state["attempts"] >= 3:
            return self._end("I'm having trouble reading that, so I've stopped. Nothing has been saved or sent. "
                             f"You can contact the {self.ws.config['name']} team directly.")
        return {"reply": hint, "state": state, "quick": self._ask(state)[1] if state["step"] in
                ("urgency", "extras", "confirm") else []}

    @staticmethod
    def _end(reply):
        return {"reply": reply, "state": None, "quick": []}

    def handle(self, state, text):
        text = text.strip()
        if wants_cancel(text):
            return self._end("No problem. I've stopped, and nothing has been sent or saved.")
        step, d = state["step"], state["data"]

        if step == "consent":
            if is_yes(text):
                state["consent_at"] = self.now().isoformat(timespec="seconds")
                return self._next(state, "name")
            if is_no(text):
                return self._end("No problem. Nothing has been collected or shared. I'm happy to keep answering "
                                 "questions about treatments and hospitals.")
            return self._retry(state, "Just to be sure: may I share those details with the care team? "
                                      "Please reply yes or no.")

        if step == "name":
            name = parse_name(text)
            if not name:
                return self._retry(state, "Sorry, I didn't catch a name. Please type just your name, "
                                          "for example 'Asha Rao'.")
            d["name"] = name
            return self._after(state, "phone")

        if step == "phone":
            phone = parse_phone(text)
            if not phone:
                return self._retry(state, "That doesn't look like a valid 10-digit Indian mobile number. "
                                          "Could you type it again?")
            d["phone"] = phone
            return self._after(state, "procedure")

        if step in ("procedure", "city"):
            key, asked = step, state.get("asking_prefill") == step
            if asked and is_yes(text):
                d[key] = state["prefill"][key]
                d[key + "_matched"] = True
                state["asking_prefill"], state["prefill_used_" + key] = None, True
                return self._after(state, "city" if key == "procedure" else "urgency")
            if asked and is_no(text):
                state["asking_prefill"], state["prefill_used_" + key] = None, True
                reply, quick = self._ask(state)
                return {"reply": reply, "state": state, "quick": quick}
            kw = self.matcher.keyword_filters(text)
            if key == "procedure":
                val, matched = kw["treatment"], bool(kw["treatment"])
                if not val:
                    val = clean_free(text, 80)
            else:
                val = kw["city"] or kw["unknown_city"]
                matched = bool(kw["city"])
                if not val:
                    val = clean_free(text, 40)
            if not val or len(val) < 3:
                return self._retry(state, f"Sorry, I didn't catch the {'treatment' if key == 'procedure' else 'city'}. "
                                          "Could you type it again?")
            d[key], d[key + "_matched"] = val, matched
            state["asking_prefill"], state["prefill_used_" + key] = None, True
            return self._after(state, "city" if key == "procedure" else "urgency")

        if step == "urgency":
            for label, rx in URGENCY:
                if rx.search(text):
                    d["urgency"] = label
                    res = self._after(state, "extras")
                    if label == "asap":
                        res["reply"] = (f"If this is a medical emergency, please call {self.ws.config['emergency_number']} "
                                        "now. Otherwise, I'll flag it as urgent.\n\n" + res["reply"])
                    return res
            return self._retry(state, "Please choose one: planned (next few months), within a month, "
                                      "or as soon as possible.")

        if step == "extras":
            if _norm(text) not in SKIP:
                budget = parse_budget(text)
                if budget:
                    d["budget_inr_max"] = budget
                m = INSURERS.search(text)
                if m:
                    d["insurance"] = m.group(0).title()
            return self._after(state, "confirm")

        if step == "confirm":
            if is_yes(text):
                return self._submit(state)
            if is_no(text):
                return self._end("Okay, I haven't sent anything and nothing has been saved.")
            m = re.match(r"^\s*edit\s+(name|phone|treatment|procedure|city)\b", text, re.I)
            if m:
                field = {"treatment": "procedure"}.get(m.group(1).lower(), m.group(1).lower())
                state["editing"] = field
                state["prefill_used_" + field] = True
                return self._next(state, field)
            return self._retry(state, "Please reply yes to send, no to stop, or say what to edit "
                                      "(for example 'edit phone').")
        raise ValueError(step)

    def _after(self, state, next_step):
        if state.get("editing"):
            state["editing"] = None
            return self._next(state, "confirm")
        return self._next(state, next_step)

    # ---- submit
    def _submit(self, state):
        d = state["data"]
        now = self.now()
        dup = self.store.find_recent(d["phone"], d["procedure"], self.cfg.get("dedupe_minutes", 10), now)
        if dup:
            ref = dup["id"]
        else:
            lead = {
                "id": state["id"], "workspace": self.ws.name, "created_at": now.isoformat(timespec="seconds"),
                "status": "pending", "attempts": 0,
                "consent": {"given": True, "version": self.cfg.get("consent_version", "v1"),
                            "text": self.consent_text(), "at": state["consent_at"]},
                "contact": {"name": d["name"], "phone": d["phone"]},
                "interest": {"procedure": d["procedure"], "procedure_matched": d.get("procedure_matched", False),
                             "city": d["city"], "providers_viewed": state["providers"]},
                "preferences": {"urgency": d["urgency"], "budget_inr_max": d.get("budget_inr_max"),
                                "insurance": d.get("insurance")},
                "topics_discussed": state["topics"],
                "source": {"channel": "web-widget", "session": state["sid"]},
            }
            self.store.save(lead)
            log.info("lead saved id=%s workspace=%s phone=%s urgency=%s", lead["id"], self.ws.name,
                     mask_phone(d["phone"]), d["urgency"])
            self.deliver_fn(self.ws, lead)
            ref = lead["id"]
        return self._end(f"Done. Your reference is {ref}. The {self.ws.config['name']} care team will contact you "
                         f"on {mask_phone(d['phone'])}. If anything changes, just tell me. If this is an "
                         f"emergency, call {self.ws.config['emergency_number']}.")