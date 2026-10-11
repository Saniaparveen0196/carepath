"""Everyday-language vocabulary: expand a patient's words with the terms the documents use.

Pure Python on purpose (no heavy imports), so it can be tested without any model.
Only whole words are expanded ("benefits" does not trigger "fits"), and terms already present are not repeated.
"""
import re


def expand_query(query, synonyms):
    q = query.lower()
    extra = []
    for term, canonical in (synonyms or {}).items():
        if re.search(rf"\b{re.escape(term.lower())}\b", q):
            extra += [w.lower() for w in canonical.split() if w.lower() not in q and w.lower() not in extra]
    return query if not extra else query + " " + " ".join(extra)