"""Structured agent trajectories, round views, and reloading the saved Phase 1 states."""

import json
from pathlib import Path

import pytest

import medverify.data as D
from medverify.llm import FakeModel
from medverify.metrics import evaluate
from medverify.pipeline import agreement_confidence, final_verdicts, run_pipeline, verifier_verdicts
from medverify.schemas import Item, VerificationState
from medverify.tools.pubmed import PubMedSearch
from test_pubmed import FakeSession

PHASE1 = Path(__file__).parents[1] / "results" / "phase1"


def items_from_ids(states):
    """Items rebuilt from item ids alone: '-gt' = true answer (0), '-hal' = hallucinated (1)."""
    return [
        Item(item_id=s.item_id, row_id=s.item_id.rsplit("-", 1)[0], question="", answer="",
             label=0 if s.item_id.endswith("-gt") else 1, difficulty="easy", category="")
        for s in states
    ]


@pytest.mark.parametrize("run", ["sysA-val-reason", "sysA-dev"])
def test_phase1_states_load_and_score_exactly_as_before(run):
    states = D.read_jsonl(PHASE1 / f"{run}-states.jsonl", VerificationState)
    saved = json.loads((PHASE1 / f"{run}-metrics.json").read_text())["overall"]
    items = items_from_ids(states)
    for verdicts in (final_verdicts(states), final_verdicts(states, min_agree=2), final_verdicts(states, round=1)):
        got = evaluate(items, verdicts, n_boot=0)["overall"]
        assert got["f1"] == pytest.approx(saved["f1"]) and got["coverage"] == pytest.approx(saved["coverage"])
    assert all(s.rounds == 1 and s.steps == [] and s.trace for s in states)  # old states: defaults, old trace kept
    assert {n: len(v) for n, v in verifier_verdicts(states).items()} == {n: len(states) for n in ("llama", "mistral-local", "phi")}


def test_saved_validation_numbers():
    states = D.read_jsonl(PHASE1 / "sysA-val-reason-states.jsonl", VerificationState)
    got = evaluate(items_from_ids(states), final_verdicts(states), n_boot=0)["overall"]
    assert round(got["f1"], 4) == 0.6802 and got["coverage"] == pytest.approx(0.89)


def test_trajectory_records_steps_tool_calls_and_decision(rows, tmp_path):
    item = D.rows_to_items(rows)[1]  # hallucinated answer, source PMID 1000
    session = FakeSession(["1000", "2001", "2002"], [
        ("1000", "Source", " ".join(item.knowledge)),
        ("2001", "Other study", "An unrelated trial of a different medicine in older adults."),
        ("2002", "Review", "A review of symptom scales used in clinical research."),
    ])
    pubmed = PubMedSearch(cache_dir=tmp_path, session=session)
    pubmed.min_interval = 0
    verifiers = {"a": FakeModel("a", lambda m: "Checked.\nVerdict: HALLUCINATED"),
                 "b": FakeModel("b", lambda m: "Verdict: FACTUAL"),
                 "c": FakeModel("c", lambda m: "Verdict: HALLUCINATED")}
    state = run_pipeline([item], verifiers, evidence_mode="pubmed", pubmed=pubmed, k=2, min_agree=2)[0]

    assert [(s.agent, s.action) for s in state.steps] == [
        ("orchestrator", "start"), ("evidence", "pubmed_search"),
        ("verifier", "judge"), ("verifier", "judge"), ("verifier", "judge"), ("decision", "vote"),
    ]
    assert [s.step for s in state.steps] == [1, 2, 3, 4, 5, 6] and all(s.t >= 0 for s in state.steps)
    judged = state.steps[2].detail
    assert judged["verifier"] == "a" and judged["verdict"] == 1 and "Verdict: HALLUCINATED" in judged["raw_reply"]
    assert judged["input_chars"] > 0 and len(judged["input_sha256"]) == 64
    assert [c.tool for c in state.tool_calls] == ["pubmed.esearch", "pubmed.efetch"]
    assert state.tool_calls[0].n_results == 3 and not state.tool_calls[0].cached
    assert {"pmid": "1000", "reason": "source_pmid"} in state.steps[1].detail["blocked"]
    assert state.final == 1 and state.final_round1 == 1 and state.rounds == 1 and state.flags == []
    assert state.trace[-1] == "decision (min_agree=2): 1"  # the old text trace is still written

    reloaded = VerificationState.model_validate(json.loads(state.model_dump_json()))
    assert reloaded.steps == state.steps and reloaded.tool_calls == state.tool_calls


def test_abstention_is_flagged_for_human_review(rows):
    items = D.rows_to_items(rows)[:1]
    verifiers = {"a": FakeModel("a", lambda m: "Verdict: FACTUAL"), "b": FakeModel("b", lambda m: "Verdict: HALLUCINATED")}
    state = run_pipeline(items, verifiers, min_agree=2)[0]
    assert state.final == 2 and "human_review" in state.flags


def test_views_use_the_right_round(rows):
    """Round-1 verdicts feed the independence analysis; the last round feeds the decision."""
    items = D.rows_to_items(rows)[:2]
    states = run_pipeline(items, {"a": FakeModel("a", lambda m: "1"), "b": FakeModel("b", lambda m: "1")}, min_agree=1)
    s = states[0]
    second = [v.model_copy(update={"round": 2, "verdict": 0}) for v in s.verdicts]
    s.verdicts += second
    s.rounds = 2
    assert [v.verdict for v in s.first_round()] == [1, 1] and [v.verdict for v in s.last_round()] == [0, 0]
    assert verifier_verdicts(states)["a"][0].verdict == 1
    assert agreement_confidence(states)[s.item_id] == (0, 1.0)
    assert final_verdicts(states, min_agree=1)[0].verdict == 0
    assert final_verdicts(states, min_agree=1, round=1)[0].verdict == 1
