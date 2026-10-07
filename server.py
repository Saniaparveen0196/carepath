"""
Web API for the patient guidance engine.

Run:   uvicorn server:app --port 8000
Open:  http://localhost:8000          (demo page with the chat widget)
Docs:  http://localhost:8000/docs     (interactive API docs)

Per-workspace settings in workspaces/<name>/config.yaml:
    api_key: "secret"                 # optional; if set, requests need header X-API-Key
    allowed_origins:                  # optional; if set, browser requests from other sites are refused
      - https://www.example.com
Environment:
    ALLOWED_ORIGINS=https://a.com,https://b.com   (CORS; default * for development)
    RATE_LIMIT_PER_MIN=20
"""
import hmac
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from engine.ranking import rank
from engine.workspace import Workspace

log = logging.getLogger("engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE = Path(__file__).parent
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
SESSION_TTL = 3600
MAX_SESSIONS = 2000
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

app = FastAPI(title="Patient Guidance Engine", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"])

_cache, _locks, _sessions, _hits = {}, {}, {}, {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------- loading
def build_assistant(ws):
    """Heavy part (loads models). Separate function so tests can replace it."""
    from engine.assistant import Assistant
    from engine.knowledge import Retriever
    return Assistant(ws, Retriever(ws))


def load(name):
    if not NAME_RE.match(name):
        raise HTTPException(404, "Unknown workspace")
    with _cache_lock:
        if name not in _cache:
            try:
                ws = Workspace(name)
            except FileNotFoundError:
                raise HTTPException(404, "Unknown workspace")
            _cache[name] = {"ws": ws, "bot": None}
            _locks[name] = threading.Lock()
        return _cache[name]


def get_bot(name):
    entry = load(name)
    with _cache_lock:
        if entry["bot"] is None:
            try:
                entry["bot"] = build_assistant(entry["ws"])
            except FileNotFoundError as e:
                raise HTTPException(503, str(e))
        return entry["bot"]


# ---------------------------------------------------------------- guards
def check_access(ws, request: Request, api_key: Optional[str]):
    expected = ws.config.get("api_key")
    if expected and not (api_key and hmac.compare_digest(str(expected), api_key)):
        raise HTTPException(401, "Missing or invalid API key")
    allowed = ws.config.get("allowed_origins") or []
    origin = request.headers.get("origin")
    if allowed and origin and origin not in allowed:
        raise HTTPException(403, "Origin not allowed for this workspace")


def rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    q = _hits.setdefault(ip, deque())
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Please wait a moment and try again.")
    q.append(now)
    if len(_hits) > 5000:                       # keep memory bounded
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
            _hits.pop(k, None)


def session_history(name, sid):
    now = time.time()
    if len(_sessions) > MAX_SESSIONS:
        for k in [k for k, v in _sessions.items() if now - v["ts"] > SESSION_TTL]:
            _sessions.pop(k, None)
        if len(_sessions) > MAX_SESSIONS:
            _sessions.clear()
    s = _sessions.get((name, sid))
    if s and now - s["ts"] > SESSION_TTL:
        s = None
    return s["history"] if s else []


def safe_url(u):
    return u if isinstance(u, str) and re.match(r"^https?://", u) else ""


# ---------------------------------------------------------------- API
class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "workspaces_loaded": sorted(_cache)}


@app.get("/api/{workspace}/meta")
def meta(workspace: str, request: Request, x_api_key: Optional[str] = Header(None)):
    ws = load(workspace)["ws"]
    check_access(ws, request, x_api_key)
    return {
        "name": ws.config["name"],
        "procedures": [p["name"] for p in ws.procedures],
        "cities": sorted({p["city"] for p in ws.providers}),
        "accreditations": list(ws.config["ranking"]["accreditation_weights"].keys()),
        "disclaimer": ws.config["disclaimer"],
    }


@app.post("/api/{workspace}/chat")
def chat(workspace: str, body: ChatIn, request: Request, x_api_key: Optional[str] = Header(None)):
    ws = load(workspace)["ws"]
    check_access(ws, request, x_api_key)
    rate_limit(request)

    sid = body.session_id if body.session_id and SID_RE.match(body.session_id) else uuid.uuid4().hex
    history = session_history(workspace, sid)
    bot = get_bot(workspace)
    try:
        with _locks[workspace]:                  # the assistant keeps per-call state; one at a time
            text, sources, new_history = bot.answer(body.message.strip(), history)
            route = bot.last_route.split(" ")[0] if bot.last_route else ""
    except HTTPException:
        raise
    except Exception:
        log.exception("chat failed for workspace=%s", workspace)
        raise HTTPException(502, "The assistant is temporarily unavailable. Please try again.")

    _sessions[(workspace, sid)] = {"history": new_history, "ts": time.time()}
    log.info("chat workspace=%s route=%s len=%d", workspace, route, len(body.message))   # no message text logged
    return {
        "session_id": sid,
        "answer": text,
        "sources": [{"n": n, "title": t, "url": safe_url(u)} for n, t, u in sources],
    }


@app.get("/api/{workspace}/rank")
def rank_endpoint(workspace: str, request: Request, procedure: Optional[str] = None,
                  city: Optional[str] = None, require: list[str] = Query(default=[]),
                  k: int = Query(5, ge=1, le=20), x_api_key: Optional[str] = Header(None)):
    ws = load(workspace)["ws"]
    check_access(ws, request, x_api_key)
    rate_limit(request)
    res = rank(ws, procedure, city, require, k)
    return {
        "procedure": res["procedure"]["name"] if res["procedure"] else None,
        "note": res["note"],
        "disclaimer": ws.config["disclaimer"],
        "results": [{
            "id": r["provider"]["id"], "name": r["provider"]["name"], "city": r["provider"]["city"],
            "url": safe_url(r["provider"].get("url")), "score": round(r["score"], 1),
            "reasons": [{"factor": f, "text": t, "points": round(p, 1)} for f, t, p in r["reasons"]],
            "not_scored": r["missing"],
            "price_min": r["price_min"], "price_max": r["price_max"],
        } for r in res["results"]],
    }


# ---------------------------------------------------------------- static
@app.get("/widget.js", include_in_schema=False)
def widget_js():
    return FileResponse(BASE / "static" / "widget.js", media_type="application/javascript")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE / "static" / "index.html")