"""Second evidence round (claim queries from the answer), offline with a fake PubMed."""

import json

import pytest

import medverify.data as D
from medverify.llm import FakeModel
from medverify.pipeline import final_verdicts, run_pipeline, second_round_triggered, verify_item
from medverify.schemas import Verdict, VerificationState
from medverify.tools.pubmed import PubMedSearch, claim_queries
from test_pubmed import FakeResponse, efetch_xml

NEW_PMID = "9001"


class TermSession:
    """Fake NCBI: the claim query (it contains a word only the answer has) finds other PMIDs."""

    def __init__(self, round1_ids, claim_ids, records):
        self.round1_ids, self.claim_ids, self.records, self.calls = round1_ids, claim_ids, records, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if url.endswith("esearch.fcgi"):
            words = params["term"].split()
            ids = self.claim_ids if {"calcium", "reduced"} & set(words) else self.round1_ids
            return FakeResponse(json.dumps({"esearchresult": {"idlist": ids}}))
        wanted = params["id"].split(",")
        return FakeResponse(efetch_xml([r for r in self.records if r[0] in wanted]))


def fake_pubmed(tmp_path, item, claim_ids=("2001", NEW_PMID)):
    records = [
        ("1000", "Source", " ".join(item.knowledge)),
        ("2001", "Other study", "An unrelated trial of a different medicine in older adults."),
        ("2002", "Review", "A review of symptom scales used in clinical research."),
        (NEW_PMID, "Mechanism study", "Calcium channel blockers and symptom changes in a mechanistic study."),
    ]
    tool = PubMedSearch(cache_dir=tmp_path, session=TermSession(["1000", "2001", "2002"], list(claim_ids), records))
    tool.min_interval = 0
    return tool


def verifier(round1_reply, with_new_evidence_reply):
    """Fake verifier that changes its mind only when the new abstract is in its evidence."""
    return lambda messages: with_new_evidence_reply if f"PMID {NEW_PMID}" in messages[-1]["content"] else round1_reply


def split_panel(second="Verdict: HALLUCINATED"):
    return {
        "llama": FakeModel("fake-llama", verifier("Verdict: HALLUCINATED", "Verdict: HALLUCINATED")),
        "mistral-local": FakeModel("fake-mistral", verifier("Verdict: FACTUAL", second)),
        "phi": FakeModel("fake-phi", verifier("Verdict: UNSURE", second)),
    }


def state_with_votes(codes):
    s = VerificationState(item_id="x-hal", evidence_mode="pubmed")
    s.verdicts = [Verdict(item_id="x-hal", system=f"verifier:v{i}", verdict=c, raw_text="", provider="f", model="m",
                          latency_s=0) for i, c in enumerate(codes)]
    return s


def test_claim_queries_keep_answer_words_plus_question_anchor():
    question = "Does aspirin reduce stroke risk in diabetic patients?"
    answer = "Aspirin increases bleeding without lowering stroke risk via platelet inhibition."
    assert claim_queries(question, answer) == [
        "increases bleeding lowering platelet aspirin reduce",
        "increases bleeding lowering aspirin",
        "increases bleeding aspirin",
    ]
    assert claim_queries(question, "Aspirin does reduce stroke risk.") == []  # nothing new to check


@pytest.mark.parametrize("codes, split, no_majority", [
    ([1, 1, 1], False, False),
    ([0, 0, 0], False, False),
    ([1, 1, 0], True, False),
    ([1, 1, None], True, False),
    ([1, 1, 2], True, False),
    ([1, 0, 2], True, True),
    ([2, 2, None], True, True),
])
def test_trigger_rules(codes, split, no_majority):
    s = state_with_votes(codes)
    assert second_round_triggered(s, "split", min_agree=2) is split
    assert second_round_triggered(s, "no-majority", min_agree=2) is no_majority
    assert second_round_triggered(s, "off", min_agree=2) is False


def test_round_two_verdicts_decide_and_round_one_is_kept(rows, tmp_path):
    item = D.rows_to_items(rows)[1]  # hallucinated answer: "... through calcium channels."
    panel = split_panel()
    state = verify_item(item, panel, "pubmed", fake_pubmed(tmp_path, item), k=3, min_agree=2, second_round_rule="split")

    assert [v.verdict for v in state.first_round()] == [1, 0, 2]
    assert [v.verdict for v in state.last_round()] == [1, 1, 1]
    assert state.rounds == 2 and state.final_round1 == 2 and state.final == 1
    assert state.flags == [] and [s.action for s in state.steps].count("vote") == 1  # decide() ran once
    # new evidence only, still independent: the standard two-message prompt, no other verdicts shown
    second_turn = panel["phi"].calls[1]
    assert len(second_turn) == 2 and "Reviewer" not in second_turn[-1]["content"]
    assert "PMID 2001" in second_turn[-1]["content"] and f"PMID {NEW_PMID}" in second_turn[-1]["content"]
    assert [(e.pmid, e.round) for e in state.evidence] == [("2001", 1), ("2002", 1), (NEW_PMID, 2)]
    claim_step = next(s for s in state.steps if s.action == "claim_search")
    assert claim_step.detail["round"] == 2 and claim_step.detail["kept"] == [NEW_PMID]
    assert "calcium" in claim_step.detail["query"] and claim_step.detail["excluded"] == ["2001", "2002"]
    assert {c.round for c in state.tool_calls} == {1, 2} and all(c.tool.startswith("pubmed.") for c in state.tool_calls)
    assert any("evidence round 2" in line for line in state.trace)
    assert final_verdicts([state], round=1)[0].verdict == 2 and final_verdicts([state])[0].verdict == 1


def test_no_majority_rule_skips_a_majority_vote(rows, tmp_path):
    item = D.rows_to_items(rows)[1]
    panel = split_panel()
    panel["phi"] = FakeModel("fake-phi", verifier("Verdict: HALLUCINATED", "Verdict: HALLUCINATED"))  # votes 1, 0, 1
    state = verify_item(item, panel, "pubmed", fake_pubmed(tmp_path, item), k=3, min_agree=2,
                        second_round_rule="no-majority")
    assert state.rounds == 1 and state.final == 1 and all(len(m.calls) == 1 for m in panel.values())


def test_still_no_majority_after_round_two_goes_to_human_review(rows, tmp_path):
    item = D.rows_to_items(rows)[1]
    panel = split_panel(second="Verdict: UNSURE")  # round 2: 1, 2, 2
    state = verify_item(item, panel, "pubmed", fake_pubmed(tmp_path, item), k=3, min_agree=2, second_round_rule="split")
    assert state.rounds == 2 and state.final == 2 and state.flags.count("human_review") == 1


def test_no_new_evidence_keeps_round_one_with_no_extra_calls(rows, tmp_path):
    item = D.rows_to_items(rows)[1]
    panel = split_panel()
    pubmed = fake_pubmed(tmp_path, item, claim_ids=("1000", "2001", "2002"))  # source or already used
    state = verify_item(item, panel, "pubmed", pubmed, k=3, min_agree=2, second_round_rule="split")
    assert "no_new_evidence" in state.flags and state.rounds == 1
    assert state.final == state.final_round1 == 2
    assert all(len(m.calls) == 1 for m in panel.values())
    assert [b.round for b in state.blocked] == [1, 2]  # the source was found again in round 2, and blocked again


@pytest.mark.parametrize("mode", ["none", "source"])
def test_second_round_is_skipped_without_pubmed(rows, mode):
    item = D.rows_to_items(rows)[1]
    panel = split_panel()
    state = verify_item(item, panel, mode, None, k=3, min_agree=2, second_round_rule="split")
    assert "second round skipped (evidence mode" in state.trace[-2]
    assert any(s.action == "second_round_skipped" for s in state.steps)
    assert state.rounds == 1 and all(len(m.calls) == 1 for m in panel.values())


def test_tool_calls_are_logged_for_each_item(rows, tmp_path):
    items = D.rows_to_items(rows)[:4]
    states = run_pipeline(items, split_panel(), "pubmed", fake_pubmed(tmp_path, items[1]), k=3, min_agree=2,
                          second_round_rule="split")
    assert all(s.tool_calls and s.tool_calls[0].tool == "pubmed.esearch" for s in states)
    assert all(any(c.round == 2 for c in s.tool_calls) for s in states)  # every item was split, so all searched again


def pipeline_workspace(tmp_path, monkeypatch, rows, panel):
    from medverify import cli
    import medverify.llm as llm
    import medverify.tools.pubmed as pm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "data/processed")
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "results/runs")
    D.write_jsonl(tmp_path / "data/processed/dev.jsonl", rows[:1])
    monkeypatch.setattr(llm, "make_model", lambda name, **kw: panel[name])
    item = D.rows_to_items(rows[:1])[1]
    monkeypatch.setattr(pm, "PubMedSearch", lambda: fake_pubmed(tmp_path / "pm", item))
    return cli


def test_evaluate_round_one_scores_the_second_round_off(rows, tmp_path, monkeypatch):
    cli = pipeline_workspace(tmp_path, monkeypatch, rows, split_panel())
    cli.main(["pipeline", "--split", "dev", "--evidence", "pubmed", "--second-round", "split", "--name", "r2"])
    run = tmp_path / "results/runs/r2"
    cli.main(["evaluate", str(run), "--round", "1"])
    final = json.loads((run / "metrics.json").read_text())["overall"]
    round1 = json.loads((run / "metrics-round1.json").read_text())["overall"]
    assert round1["coverage"] < final["coverage"]  # the hallucinated item: abstain in round 1, caught in round 2
    config = json.loads((run / "config.json").read_text())
    assert config["setup"]["second_round"] == "split" and config["setup"]["k2"] == 2


def test_debate_and_second_round_cannot_be_combined(rows, tmp_path, monkeypatch):
    cli = pipeline_workspace(tmp_path, monkeypatch, rows, split_panel())
    with pytest.raises(SystemExit, match="--debate and --second-round cannot be used together"):
        cli.main(["pipeline", "--split", "dev", "--evidence", "pubmed", "--debate", "--second-round", "split"])
    with pytest.raises(ValueError, match="cannot be used together"):
        verify_item(D.rows_to_items(rows)[0], split_panel(), "none", None, 3, 2, debate=True, second_round_rule="split")
