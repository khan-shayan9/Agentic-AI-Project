import json

from medverify.tools.pubmed import PubMedSearch, keyword_queries, ngram_overlap

SOURCE = (
    "We studied whether drug alpha lowers blood pressure in adults with type two diabetes over twelve "
    "months in a randomised controlled trial with four hundred participants across six centres."
)


def efetch_xml(records):
    arts = "".join(
        f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><ArticleTitle>{title}</ArticleTitle>"
        f"<Abstract><AbstractText Label=\"RESULTS\">{text}</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>"
        for pmid, title, text in records
    )
    return f"<?xml version=\"1.0\"?><PubmedArticleSet>{arts}</PubmedArticleSet>"


class FakeResponse:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, idlist, records):
        self.idlist, self.records, self.calls = idlist, records, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if url.endswith("esearch.fcgi"):
            ids = self.idlist if len(params["term"].split()) <= 8 else []
            return FakeResponse(json.dumps({"esearchresult": {"idlist": ids}}))
        return FakeResponse(efetch_xml(self.records))


def test_ngram_overlap():
    assert ngram_overlap(SOURCE, SOURCE) == 1.0
    assert ngram_overlap("completely different words about a separate topic entirely here now", SOURCE) == 0.0


def test_keyword_queries_shrink():
    qs = keyword_queries("Does drug alpha lower blood pressure in adults with type two diabetes over a full year?")
    assert len(qs) >= 2 and len(qs[0].split()) == 8 and len(qs[-1].split()) == 3
    assert "does" not in qs[0].split()


def test_retrieve_blocks_source_pmid_and_near_copies(tmp_path):
    records = [
        ("111", "Source paper", SOURCE),  # the question's own abstract
        ("222", "Duplicate publication", SOURCE + " Extra sentence."),  # a near copy under another PMID
        ("333", "Independent study", "A separate cohort found drug alpha had modest effects on systolic pressure."),
        ("444", "Another study", "Meta analysis of antihypertensive agents in diabetic adults."),
    ]
    session = FakeSession(["111", "222", "333", "444"], records)
    tool = PubMedSearch(email="t@example.com", cache_dir=tmp_path, session=session)
    tool.min_interval = 0
    kept, blocked = tool.retrieve(["word " * 10, "drug alpha blood pressure"], k=3, block_pmids=[111], block_texts=[SOURCE])
    assert [e.pmid for e in kept] == ["333", "444"]
    assert {(b.pmid, b.reason) for b in blocked} == {("111", "source_pmid"), ("222", "near_duplicate")}
    assert kept[0].query == "drug alpha blood pressure"  # first query returned nothing, fallback used
    assert kept[0].text.startswith("RESULTS:")


def test_responses_are_cached(tmp_path):
    session = FakeSession(["333"], [("333", "T", "body text")])
    tool = PubMedSearch(cache_dir=tmp_path, session=session)
    tool.min_interval = 0
    tool.search("drug alpha")
    tool.search("drug alpha")
    assert len(session.calls) == 1
