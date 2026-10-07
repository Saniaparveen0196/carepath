"""
Step 1: Crawl public Qurovo pages and save clean text as JSONL.

Usage:
    python crawl.py                 # up to 300 pages
    python crawl.py --max-pages 50  # smaller test run
"""
import argparse
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = "https://www.qurovo.com"
USER_AGENT = "QurovoRAGPrototype/0.1 (independent research prototype)"
DELAY_SECONDS = 1.0

PAGE_TYPES = {
    "treatment": re.compile(r"^/treatment/[^/]+/?$"),
    "hospital": re.compile(r"^/hospital/[^/]+/?$"),
    "doctor": re.compile(r"^/doctor/[^/]+/?$"),
    "department": re.compile(r"^/department/[^/]+/?$"),
}
LISTING_PATHS = {"/", "/treatments", "/hospitals", "/doctors", "/departments"}

OUT_DIR = Path("data")
RAW_DIR = OUT_DIR / "raw"
OUT_FILE = OUT_DIR / "pages.jsonl"


def page_type(path):
    for name, pattern in PAGE_TYPES.items():
        if pattern.match(path):
            return name
    return None


def normalize(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    return parsed._replace(query="").geturl().rstrip("/") or BASE


def is_allowed_internal(url):
    parsed = urlparse(url)
    if parsed.netloc not in ("www.qurovo.com", "qurovo.com"):
        return False
    path = parsed.path.rstrip("/") or "/"
    return path in LISTING_PATHS or page_type(path) is not None


def clean_text(soup):
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "svg", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    lines, seen = [], set()
    noise = {"Compare", "View Profile →", "View Hospital Profile →", "Request a Callback"}
    for raw in main.get_text("\n").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or line in seen or line in noise:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)


def extract_doc(url, html):
    path = urlparse(url).path.rstrip("/") or "/"
    ptype = page_type(path)
    if ptype is None:
        return None
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""
    h1 = soup.find("h1")
    return {
        "url": url,
        "slug": path.split("/")[-1],
        "type": ptype,
        "title": h1.get_text(" ", strip=True) if h1 else title,
        "page_title": title,
        "text": clean_text(soup),
    }


def main(max_pages):
    OUT_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    robots = RobotFileParser()
    robots.set_url(f"{BASE}/robots.txt")
    try:
        robots.read()
    except Exception:
        print("Could not read robots.txt, proceeding cautiously.")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    queue = deque([BASE])
    visited, docs = set(), []
    pbar = tqdm(total=max_pages, desc="pages")

    while queue and len(visited) < max_pages:
        url = normalize(queue.popleft())
        if url in visited or not is_allowed_internal(url):
            continue
        if not robots.can_fetch(USER_AGENT, url):
            print(f"Blocked by robots.txt: {url}")
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed {url}: {e}")
            continue

        html = resp.text
        slug = re.sub(r"[^a-z0-9]+", "_", urlparse(url).path.lower()).strip("_") or "home"
        (RAW_DIR / f"{slug}.html").write_text(html, encoding="utf-8")

        doc = extract_doc(url, html)
        if doc and len(doc["text"]) > 200:
            docs.append(doc)

        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            link = normalize(urljoin(url, a["href"]))
            if link not in visited and is_allowed_internal(link):
                queue.append(link)

        pbar.update(1)
        time.sleep(DELAY_SECONDS)

    pbar.close()
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    by_type = {}
    for d in docs:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    print(f"\nSaved {len(docs)} documents to {OUT_FILE}")
    print("By type:", by_type)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=300)
    args = ap.parse_args()
    main(args.max_pages)