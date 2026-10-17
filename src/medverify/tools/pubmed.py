"""PubMed search tool for the evidence agent, with source-leakage blocking.

Uses NCBI E-utilities (free). Without an API key NCBI allows 3 requests/second;
with a free key, 10. Every response is cached on disk.

Leakage control (instructor's condition 3): each MedHallu question was built from one
PubMed abstract. That abstract's PMID is always removed from results, and any other
abstract that copies a large part of its text is removed too ("near duplicate").
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional

import requests

from ..schemas import BlockedEvidence, Evidence

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_WORD = re.compile(r"[a-z0-9]+")

_STOPWORDS = set(
    """a an the and or of in on at to for with by from as is are was were be been being do does did
    can could should would may might will shall has have had this that these those it its their there
    than then what which who whom whose when where why how whether into over under between among
    after before during about against within without via vs versus not no yes any all each more most
    less least other such same so very also only both either neither patients patient study""".split()
)


def keyword_queries(question: str) -> list[str]:
    """PubMed ANDs every word, so long questions often return nothing.

    Returns a list of progressively shorter keyword queries to try in order.
    """
    words = [w for w in _WORD.findall(question.lower()) if w not in _STOPWORDS and len(w) > 2]
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    queries = []
    for size in (8, 5, 3):
        if seen[:size]:
            q = " ".join(seen[:size])
            if q not in queries:
                queries.append(q)
    return queries or [question]


def word_ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def ngram_overlap(candidate: str, source: str, n: int = 8) -> float:
    """Share of the candidate's 8-word sequences that also appear in the source (0 to 1)."""
    cand = word_ngrams(candidate, n)
    if not cand:
        return 0.0
    return len(cand & word_ngrams(source, n)) / len(cand)


class PubMedSearch:
    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Path = Path("data/cache/pubmed"),
        session: Optional[requests.Session] = None,
    ):
        self.email = email or os.environ.get("NCBI_EMAIL", "")
        self.api_key = api_key or os.environ.get("NCBI_API_KEY", "")
        self.min_interval = 0.11 if self.api_key else 0.34
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self._last = 0.0

    # -------------------------------------------------------------- HTTP

    def _get(self, endpoint: str, params: dict) -> str:
        params = {**params, "tool": "medverify"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        key = hashlib.sha256(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
        cached = self.cache_dir / f"{key}.txt"
        if cached.exists():
            return cached.read_text(encoding="utf-8")

        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self.session.get(f"{EUTILS}/{endpoint}", params=params, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            cached.write_text(resp.text, encoding="utf-8")
            return resp.text
        resp.raise_for_status()
        return resp.text

    # ------------------------------------------------------------ search

    def search(self, query: str, retmax: int = 10) -> list[str]:
        """Return PMIDs that match the query, most relevant first."""
        text = self._get(
            "esearch.fcgi",
            {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json", "sort": "relevance"},
        )
        return json.loads(text)["esearchresult"].get("idlist", [])

    def fetch(self, pmids: list[str]) -> dict[str, tuple[str, str]]:
        """Return {pmid: (title, abstract)} for the given PMIDs."""
        if not pmids:
            return {}
        xml = self._get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
        out: dict[str, tuple[str, str]] = {}
        root = ET.fromstring(xml)
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID", default="")
            title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            parts = []
            for node in art.iter("AbstractText"):
                label = node.get("Label")
                body = "".join(node.itertext()).strip()
                parts.append(f"{label}: {body}" if label else body)
            out[pmid] = (title.strip(), " ".join(parts).strip())
        return out

    # ---------------------------------------------------------- retrieve

    def retrieve(
        self,
        queries: str | list[str],
        k: int = 3,
        block_pmids: Iterable[int | str] = (),
        block_texts: Iterable[str] = (),
        overlap_threshold: float = 0.3,
        search_size: int = 10,
    ) -> tuple[list[Evidence], list[BlockedEvidence]]:
        """Top-k abstracts after removing the source and near copies of it.

        `queries` may be a list; each is tried in order until one returns usable results.
        """
        if isinstance(queries, str):
            queries = [queries]
        blocked_ids = {str(p) for p in block_pmids if p is not None}
        source_text = " ".join(block_texts)
        query, pmids = queries[0], []
        for query in queries:
            pmids = self.search(query, retmax=search_size)
            if any(p not in blocked_ids for p in pmids):
                break
        records = self.fetch(pmids)

        kept: list[Evidence] = []
        blocked: list[BlockedEvidence] = []
        for pmid in pmids:
            if pmid in blocked_ids:
                blocked.append(BlockedEvidence(pmid=pmid, reason="source_pmid", overlap=1.0))
                continue
            title, abstract = records.get(pmid, ("", ""))
            if not abstract:
                continue
            overlap = ngram_overlap(abstract, source_text) if source_text else 0.0
            if overlap >= overlap_threshold:
                blocked.append(BlockedEvidence(pmid=pmid, reason="near_duplicate", overlap=round(overlap, 3)))
                continue
            kept.append(Evidence(pmid=pmid, title=title, text=abstract, query=query, rank=len(kept) + 1))
            if len(kept) == k:
                break
        return kept, blocked
