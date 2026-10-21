"""Using MedVerify on questions and answers typed by a user: source blocking, verify, batch (offline)."""

import pytest

from medverify.llm import FakeModel
from medverify.pipeline import verify_user_item
from medverify.schemas import user_item
from medverify.tools.pubmed import PubMedSearch
from test_pubmed import FakeSession

SOURCE = (
    "In a randomised trial of four hundred adults with type two diabetes, drug alpha lowered systolic blood "
    "pressure by eight millimetres of mercury over twelve months compared with placebo across six centres."
)
QUESTION = "Does drug alpha lower blood pressure in adults with type two diabetes?"
ANSWER = "Drug alpha raised blood pressure because it blocks calcium channels in the kidney."


def fake_pubmed(tmp_path, idlist=("5555", "5556", "2001", "2002")):
    records = [
        ("5555", "Drug alpha in diabetes: a randomised trial", SOURCE),
        ("5556", "Duplicate report of the drug alpha trial", SOURCE + " A second publication of the same data."),
        ("2001", "Calcium channel blockers in older adults", "A cohort of older adults taking calcium channel blockers."),
        ("2002", "Blood pressure targets", "A review of blood pressure targets in diabetes care."),
    ]
    tool = PubMedSearch(cache_dir=tmp_path / "pubmed", session=FakeSession(list(idlist), records))
    tool.min_interval = 0
    return tool


def panel(reply="Wrong mechanism.\nVerdict: HALLUCINATED"):
    return {name: FakeModel(f"fake-{name}", lambda m, r=reply: r) for name in ("llama", "mistral-local", "phi")}


def test_user_given_source_pmid_and_its_near_copy_are_blocked(tmp_path):
    state = verify_user_item(user_item(QUESTION, ANSWER, pmid="5555"), panel(), "pubmed", fake_pubmed(tmp_path))
    assert {(b.pmid, b.reason) for b in state.blocked} == {("5555", "source_pmid"), ("5556", "near_duplicate")}
    assert [e.pmid for e in state.evidence] == ["2001", "2002"]
    assert state.steps[0].action == "source_fetch" and state.steps[0].detail["pmid"] == "5555"
    assert [s.step for s in state.steps] == list(range(1, len(state.steps) + 1))
    assert state.tool_calls[0].tool == "pubmed.efetch" and state.tool_calls[0].input == "5555"
    assert state.trace[0].startswith("source: PMID 5555 fetched")

    without = verify_user_item(user_item(QUESTION, ANSWER), panel(), "pubmed", fake_pubmed(tmp_path / "b"))
    assert [e.pmid for e in without.evidence] == ["5555", "5556", "2001"] and without.blocked == []


def test_source_mode_uses_the_user_given_abstract(tmp_path):
    models = panel()
    state = verify_user_item(user_item(QUESTION, ANSWER, pmid="5555"), models, "source", fake_pubmed(tmp_path))
    assert state.evidence[0].pmid == "5555" and state.evidence[0].text.endswith(SOURCE)  # fetched abstract
    assert SOURCE in models["llama"].calls[0][1]["content"]
    with pytest.raises(ValueError, match="needs the answer's source abstract"):
        verify_user_item(user_item(QUESTION, ANSWER), panel(), "source", fake_pubmed(tmp_path / "b"))
