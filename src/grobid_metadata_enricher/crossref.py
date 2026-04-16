"""Crossref lookup for GROBID biblStruct enrichment.

Queries api.crossref.org /works to resolve a DOI and canonical title for a
reference given its title, authors, year, and journal. Used by
pipeline.enrich_references_with_crossref to fill missing DOIs on parsed
biblStructs without calling an LLM.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

CROSSREF_BASE = "https://api.crossref.org/works"
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_ROWS = 5
DEFAULT_USER_AGENT = "sciencebeam-grobid-metadata-enricher/0.1 (mailto:sciencebeam@example.com)"
MIN_TITLE_JACCARD = 0.5
WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(WORD_RE.findall((text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class CrossrefClient:
    """Thread-safe Crossref client with an in-memory cache."""

    def __init__(
        self,
        base_url: str = CROSSREF_BASE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        rows: int = DEFAULT_ROWS,
        user_agent: str = DEFAULT_USER_AGENT,
        sleep_between_s: float = 0.05,
    ) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.rows = rows
        self.user_agent = user_agent
        self.sleep_between_s = sleep_between_s
        self._cache: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()

    def _cache_key(self, title: str, authors: List[str], year: str, journal: str) -> str:
        return json.dumps(
            [title.strip().lower(), [a.lower() for a in (authors or [])][:3], year.strip(), journal.strip().lower()]
        )

    def lookup(
        self,
        title: str,
        authors: Optional[List[str]] = None,
        year: str = "",
        journal: str = "",
        min_title_jaccard: float = MIN_TITLE_JACCARD,
    ) -> Dict[str, str]:
        """Return {'doi': ..., 'title': ...} for the best Crossref match, or empty strings.

        The canonical title comes from the Crossref record; downstream title-jaccard
        matching against mangled TEI-extracted titles benefits from this clean form.
        """
        title = (title or "").strip()
        if len(title) < 15:
            return {"doi": "", "title": ""}
        key = self._cache_key(title, authors or [], year, journal)
        with self._lock:
            if key in self._cache:
                return dict(self._cache[key])

        params = {"query.bibliographic": title, "rows": str(self.rows)}
        if authors:
            params["query.author"] = " ".join(a.split()[-1] for a in authors[:3])
        if journal:
            params["query.container-title"] = journal

        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        empty = {"doi": "", "title": ""}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            with self._lock:
                self._cache[key] = empty
            return dict(empty)
        finally:
            if self.sleep_between_s:
                time.sleep(self.sleep_between_s)

        items = ((data.get("message") or {}).get("items")) or []
        query_tokens = _tokens(title)
        year_i = 0
        year_m = re.search(r"\b(19|20)\d{2}\b", year or "")
        if year_m:
            try:
                year_i = int(year_m.group(0))
            except ValueError:
                year_i = 0

        best_doi = ""
        best_title = ""
        best_overlap = 0.0
        for item in items:
            item_title_parts = item.get("title") or []
            if not item_title_parts:
                continue
            cand_title = " ".join(item_title_parts)
            overlap = _jaccard(query_tokens, _tokens(cand_title))
            if year_i:
                issued = (item.get("issued") or {}).get("date-parts") or []
                item_year = 0
                if issued and issued[0]:
                    try:
                        item_year = int(issued[0][0])
                    except (ValueError, IndexError, TypeError):
                        item_year = 0
                if item_year and abs(item_year - year_i) > 1:
                    continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_doi = str(item.get("DOI") or "").strip().lower()
                best_title = cand_title.strip()

        if best_overlap < min_title_jaccard:
            result = dict(empty)
        else:
            result = {"doi": best_doi, "title": best_title}
        with self._lock:
            self._cache[key] = dict(result)
        return result

    def lookup_doi(
        self,
        title: str,
        authors: Optional[List[str]] = None,
        year: str = "",
        journal: str = "",
        min_title_jaccard: float = MIN_TITLE_JACCARD,
    ) -> str:
        return self.lookup(title, authors, year, journal, min_title_jaccard).get("doi", "")
