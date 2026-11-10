"""PubMed errors: retries, PubMedError, and runs that keep going when PubMed still fails (offline)."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import medverify.data as D
import medverify.tools.pubmed as pm
from experiments import failures
from medverify.metrics import efficiency_summary, format_summaries, leakage_summary
from medverify.pipeline import run_pipeline, verify_item
from medverify.schemas import VerificationState
from medverify.tools.pubmed import PubMedError, PubMedSearch
from test_pubmed import FakeResponse
from test_second_round import NEW_PMID, fake_pubmed, pipeline_workspace, split_panel

FIXTURE = Path(__file__).parent / "fixtures" / "run_before_pubmed_retry"  # made with the code of e64d26a


@pytest.fixture
def sleeps(monkeypatch):
    """Record the waits instead of sleeping, so the retries take no time."""
    waits = []
    monkeypatch.setattr(pm.time, "sleep", waits.append)
    return waits


class ScriptedSession:
    """Fake NCBI: answers with the given outcomes in order (a status code or an exception class), then 200."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.body = json.dumps({"esearchresult": {"idlist": ["333"]}})

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        outcome = self.outcomes.pop(0) if self.outcomes else 200
        if isinstance(outcome, type):
            raise outcome("network down")
        return FakeResponse(self.body if outcome == 200 else "Bad Request", outcome)


class Failing:
    """Wraps a fake NCBI session: requests for which `fails(url, params)` is true get HTTP 400."""

    def __init__(self, session, fails):
        self.session, self.fails, self.failed = session, fails, 0

    def get(self, url, params=None, timeout=None):
        if self.fails(url, params):
            self.failed += 1
            return FakeResponse("Bad Request", 400)
        return self.session.get(url, params, timeout)


def scripted_tool(tmp_path, outcomes, **kw):
    session = ScriptedSession(outcomes)
    tool = PubMedSearch(cache_dir=tmp_path, session=session, **kw)
    tool.min_interval = 0
    return tool, session


def test_400_is_retried_and_only_the_success_is_cached(tmp_path, sleeps):
    tool, session = scripted_tool(tmp_path, [400, 400])
    assert tool.search("drug alpha") == ["333"]
    assert len(session.calls) == 3 and sleeps == [1, 2]
    cached = list(tmp_path.glob("*.txt"))
    assert len(cached) == 1 and cached[0].read_text(encoding="utf-8") == session.body
    assert tool.search("drug alpha") == ["333"] and len(session.calls) == 3  # second time from the cache
    assert [c.cached for c in tool.calls] == [False, True] and all(c.error is None for c in tool.calls)


@pytest.mark.parametrize("error", [requests.ConnectionError, requests.Timeout])
def test_network_errors_are_retried(tmp_path, sleeps, error):
    tool, session = scripted_tool(tmp_path, [error])
    assert tool.search("drug alpha") == ["333"]
    assert len(session.calls) == 2 and sleeps == [1]


def test_always_400_raises_pubmed_error_after_six_attempts(tmp_path, sleeps):
    tool, session = scripted_tool(tmp_path, [400] * 10, email="me@example.com", api_key="secret-key")
    tool.min_interval = 0
    with pytest.raises(PubMedError, match="HTTP 400") as info:
        tool.fetch(["12592151", "21190419"])
    err = info.value
    assert isinstance(err, RuntimeError) and err.endpoint == "efetch.fcgi" and err.status == 400
    assert len(session.calls) == 6 and sleeps == [1, 2, 4, 8, 16]
    assert err.url == f"{pm.EUTILS}/efetch.fcgi?db=pubmed&id=12592151,21190419&retmode=xml"
    assert "secret-key" not in str(err) and "me@example.com" not in str(err)  # never in states or exports
    assert list(tmp_path.iterdir()) == []  # nothing cached
    assert tool.calls[-1].tool == "pubmed.efetch" and tool.calls[-1].error == str(err)


def test_no_response_at_all_says_so(tmp_path, sleeps):
    tool, _ = scripted_tool(tmp_path, [requests.ConnectionError] * 6)
    with pytest.raises(PubMedError, match="no response") as info:
        tool.search("drug alpha")
    assert info.value.status is None


@pytest.mark.parametrize("fails, round2_tools", [
    (lambda url, p: url.endswith("esearch.fcgi") and "increased" in p["term"].split(), ["pubmed.esearch"]),
    (lambda url, p: url.endswith("efetch.fcgi") and NEW_PMID in p["id"].split(","), ["pubmed.esearch", "pubmed.efetch"]),
], ids=["search-fails", "fetch-fails"])
def test_round_two_pubmed_error_keeps_round_one(rows, tmp_path, sleeps, fails, round2_tools):
    item = D.rows_to_items(rows)[1]  # hallucinated answer: its claim queries contain "increased"
    panel = split_panel()
    pubmed = fake_pubmed(tmp_path, item)
    pubmed.session = Failing(pubmed.session, fails)
    state = verify_item(item, panel, "pubmed", pubmed, k=3, min_agree=2, second_round_rule="split")

    assert state.flags == ["pubmed_error", "human_review"] and pubmed.session.failed == 6
    assert state.rounds == 1 and state.final == state.final_round1 == 2  # the round-1 decision stands
    assert all(len(m.calls) == 1 for m in panel.values())  # no extra model calls
    assert [e.round for e in state.evidence] == [1, 1]
    round2 = [c for c in state.tool_calls if c.round == 2]
    assert [c.tool for c in round2] == round2_tools
    assert all(c.error is None for c in round2[:-1]) and "HTTP 400" in round2[-1].error
    step = next(s for s in state.steps if s.action == "pubmed_error")
    assert step.agent == "evidence" and step.detail["round"] == 2 and step.detail["message"] == round2[-1].error
    assert any(line.startswith("evidence round 2: PubMed failed") and line.endswith("round-1 votes stand")
               for line in state.trace)
    assert failures.evidence_state(state, item.label) == "PubMed error"


def test_round_one_pubmed_error_still_judges_every_item(rows, tmp_path, sleeps):
    items = D.rows_to_items(rows)[:4]
    panel = split_panel()
    pubmed = fake_pubmed(tmp_path / "pm", items[1])
    pubmed.session = Failing(pubmed.session, lambda url, p: True)  # PubMed always fails
    out = tmp_path / "states.jsonl"
    states = run_pipeline(items, panel, "pubmed", pubmed, k=3, min_agree=2, out_path=out, second_round_rule="split")

    assert len(D.read_jsonl(out, VerificationState)) == 4  # the run finished and saved every item
    for s in states:
        assert s.flags.count("pubmed_error") == 1 and s.evidence == [] and s.final is not None
        assert len(s.first_round()) == 3  # the verifiers still judged, without evidence
        assert [st.detail["round"] for st in s.steps if st.action == "pubmed_error"] == [1, 2]
        assert any(line.startswith("evidence: PubMed failed") for line in s.trace)
        assert [c.error is not None for c in s.tool_calls] == [True, True]
    assert all(len(m.calls) == 4 for m in panel.values())  # one judgement per item, no round 2

    eff = efficiency_summary(states)
    assert eff["pubmed_errors"] == 4
    assert efficiency_summary([VerificationState(item_id="x", evidence_mode="none")])["pubmed_errors"] == 0
    assert "items with a PubMed error (still failing after retries): 4" in format_summaries(leakage_summary(states), eff)


def test_untagged_ollama_name_finds_its_latest_tag(monkeypatch):
    from medverify import cli

    class Tags:
        def json(self):
            return {"models": [
                {"name": "phi4-mini:latest", "model": "phi4-mini:latest", "digest": "sha256:phi"},
                {"name": "mistral:7b", "model": "mistral:7b", "digest": "sha256:mis"},
            ]}

    monkeypatch.setattr(requests, "get", lambda url, timeout=None: Tags())
    models = {name: SimpleNamespace(provider="ollama", model=model_id)
              for name, model_id in [("phi", "phi4-mini"), ("mistral-local", "mistral:7b"), ("llama", "llama3.1:8b")]}
    assert cli._ollama_digests(models) == {"phi4-mini": "sha256:phi", "mistral:7b": "sha256:mis", "llama3.1:8b": None}


def test_run_folder_made_before_this_change_resumes(rows, tmp_path, monkeypatch):
    panel = split_panel()
    cli = pipeline_workspace(tmp_path, monkeypatch, rows, panel)
    item = D.rows_to_items(rows[:1])[1]
    tools = []
    monkeypatch.setattr(pm, "PubMedSearch", lambda: tools.append(fake_pubmed(tmp_path / "pm", item)) or tools[-1])
    run = tmp_path / "results/runs/old"
    shutil.copytree(FIXTURE, run)

    cli.main(["pipeline", "--split", "dev", "--evidence", "pubmed", "--second-round", "split", "--name", "old"])
    assert all(m.calls == [] for m in panel.values()) and tools[0].session.calls == []  # resumed, nothing redone
    assert json.loads((run / "config.json").read_text())["setup"] == json.loads((FIXTURE / "config.json").read_text())["setup"]
    states = D.read_jsonl(run / "states.jsonl", VerificationState)
    assert len(states) == 2 and all(c.error is None for s in states for c in s.tool_calls)
