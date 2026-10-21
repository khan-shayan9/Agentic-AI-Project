"""The web app: pure helpers always; the Streamlit page itself only when streamlit is installed."""

from pathlib import Path

import pytest

from app import helpers as H
from medverify.schemas import BlockedEvidence, Evidence, TraceStep, Verdict, VerificationState


def hand_made_state(final=1):
    votes = [("llama", 1, "The trial showed the opposite effect.\nVerdict: HALLUCINATED"),
             ("mistral-local", 1, "Invented mechanism.\nVerdict: HALLUCINATED"),
             ("phi", 2, "Hmm.\n---\nVerdict: UNSURE")]
    return VerificationState(
        item_id="user-abc", evidence_mode="pubmed", final=final,
        verdicts=[Verdict(item_id="user-abc", system=f"verifier:{n}", verdict=c, raw_text=t, provider="f", model=n,
                          latency_s=1.5, prompt_tokens=100, completion_tokens=10, reasked=n == "phi") for n, c, t in votes],
        evidence=[Evidence(pmid="2001", title="A trial", text="...", query="q", rank=1),
                  Evidence(pmid="2002", title="A review", text="...", query="claim q", rank=1, round=2)],
        blocked=[BlockedEvidence(pmid="5555", reason="source_pmid", overlap=1.0),
                 BlockedEvidence(pmid="5556", reason="near_duplicate", overlap=0.62)],
        steps=[TraceStep(step=1, agent="orchestrator", action="start", detail={"min_agree": 2})],
    )


def test_decision_text():
    assert H.decision_text(0).startswith("ACCEPT (factual")
    assert H.decision_text(1).startswith("REJECT (hallucinated")
    assert H.decision_text(2) == "NOT SURE (sent for human review)"
    assert "REJECT" in H.badge_html(1) and "#b3261e" in H.badge_html(1) and "sent for human review" in H.badge_html(2)


def test_verifier_and_evidence_rows():
    state = hand_made_state()
    rows = H.verifier_rows(state)
    assert [(r["verifier"], r["vote"], r["round"]) for r in rows] == [
        ("llama", "hallucinated", 1), ("mistral-local", "hallucinated", 1), ("phi", "unsure", 1)]
    assert rows[0]["reason"] == "The trial showed the opposite effect." and rows[2]["reasked"] is True
    assert rows[2]["reason"] == "Hmm."  # the first reply holds the reasoning after a re-ask
    ev = H.evidence_rows(state)
    assert ev[0] == {"pmid": "2001", "title": "A trial", "url": "https://pubmed.ncbi.nlm.nih.gov/2001/", "round": 1,
                     "query": "q", "leak": None}
    assert ev[1]["round"] == 2 and "*(second search)*" in H.evidence_markdown(state)[1]
    assert H.blocked_rows(state)[1]["reason"] == "a near copy of the source abstract (overlap 62%)"
    assert H.agreement(state) == "2 of 3 verifiers said hallucinated"
    assert H.cost_line(state).startswith("4 model calls · 300 prompt + 30 completion tokens")  # phi was asked twice


def test_not_sure_explanations():
    tied = hand_made_state(final=2)
    tied.verdicts[1] = tied.verdicts[1].model_copy(update={"verdict": 0})
    assert H.abstain_reason(tied).startswith("No verdict had the 2 votes it needs (1 hallucinated, 1 factual, 1 unsure)")
    tied.flags.append("no_new_evidence")
    assert "a second search found no new abstracts" in H.abstain_reason(tied)
    tied.steps.append(TraceStep(step=2, agent="orchestrator", action="second_round_judge"))
    assert "still disagreed after a second search" in H.abstain_reason(tied)
    assert H.abstain_reason(hand_made_state(final=1)) is None


def test_setup_check_never_crashes():
    def offline(url, timeout=None):
        raise ConnectionError("refused")

    status = H.setup_status(get=offline, env={})
    assert status["ollama"] is False and "refused" in status["ollama_error"] and not H.ready(status)
    levels = [level for level, _ in H.setup_messages(status)]
    assert levels == ["error", "warning"]

    class Tags:
        def json(self):
            return {"models": [{"name": "llama3.1:8b"}, {"name": "phi4-mini:latest"}]}

    status = H.setup_status(get=lambda url, timeout=None: Tags(), env={"NCBI_EMAIL": "me@example.com"})
    assert status["ollama"] and status["missing_models"] == ["mistral:7b"] and not H.ready(status)
    assert ("warning", "Missing models: mistral:7b. Run `ollama pull mistral:7b`.") in H.setup_messages(status)
    assert H.results_csv([{"question": "q", "decision": "REJECT"}]).decode().splitlines() == ["question,decision", "q,REJECT"]


# ------------------------------------------------------------- the page itself

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def page(tmp_path, monkeypatch):
    """The Streamlit page with fake verifiers, a fake PubMed and a ready setup check."""
    testing = pytest.importorskip("streamlit.testing.v1")
    import medverify.llm as llm
    import medverify.tools.pubmed as pm
    from medverify import service
    from medverify.llm import FakeModel
    from test_use import fake_pubmed

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(llm, "make_model", lambda name, **kw: FakeModel(name, lambda m: "Wrong drug.\nVerdict: HALLUCINATED"))
    monkeypatch.setattr(pm, "PubMedSearch", lambda: fake_pubmed(tmp_path))
    monkeypatch.setattr(service, "FINAL_CONFIG", tmp_path / "final_config.json")
    monkeypatch.setattr(H, "setup_status", lambda: {"ollama": True, "ollama_error": "", "missing_models": [],
                                                    "ncbi_email": True, "host": "http://localhost:11434"})
    return testing.AppTest.from_file(str(APP), default_timeout=60)


def test_page_checks_one_answer(page):
    at = page.run()
    assert not at.exception
    assert any("not medical advice" in c.value for c in at.caption)
    at.text_area[0].input("Does drug alpha lower blood pressure?")
    at.text_area[1].input("It raises blood pressure.")
    at.button[0].click().run()
    assert not at.exception
    shown = " ".join(m.value for m in at.markdown)
    assert "REJECT" in shown and "3 of 3 verifiers said hallucinated" in shown and "PMID 5555" in shown


def test_page_warns_instead_of_crashing(page):
    at = page.run()
    at.button[0].click().run()  # nothing typed
    assert not at.exception and "Please enter both a question and an answer." in [w.value for w in at.warning]
