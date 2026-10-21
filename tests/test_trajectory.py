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


def nonsense_then_verdict(messages):
    """Fake verifier: rambles without a verdict, then answers properly when asked again."""
    return "Verdict: FACTUAL" if messages[-1]["role"] == "user" and "could not find" in messages[-1]["content"] else "Hmm, hard to say."


@pytest.mark.parametrize("reask, expected", [(True, 0), (False, None)])
def test_unreadable_reply_is_asked_again_once(rows, reask, expected):
    items = D.rows_to_items(rows)[:1]
    phi = FakeModel("phi", nonsense_then_verdict)
    state = run_pipeline(items, {"phi": phi}, min_agree=1, reask=reask)[0]
    verdict = state.verdicts[0]
    assert verdict.verdict == expected and verdict.reasked is reask
    if reask:
        assert len(phi.calls) == 2 and phi.calls[1][-2] == {"role": "assistant", "content": "Hmm, hard to say."}
        assert "Verdict: FACTUAL / Verdict: HALLUCINATED" in phi.calls[1][-1]["content"]  # reason style's final line
        assert verdict.raw_text == "Hmm, hard to say.\n---\nVerdict: FACTUAL"
        assert "reasked:phi" in state.flags and [s.action for s in state.steps].count("reask") == 1
    else:
        assert len(phi.calls) == 1 and state.flags == ["human_review"]


def test_no_reask_flag_on_the_command_line(rows, tmp_path, monkeypatch):
    from medverify import cli
    import medverify.llm as llm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "data/processed")
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "results/runs")
    D.write_jsonl(tmp_path / "data/processed/dev.jsonl", rows[:2])
    monkeypatch.setattr(llm, "make_model", lambda name, **kw: FakeModel(name, nonsense_then_verdict))
    cli.main(["pipeline", "--split", "dev", "--verifiers", "phi", "--min-agree", "1", "--no-reask", "--name", "off"])
    cli.main(["pipeline", "--split", "dev", "--verifiers", "phi", "--min-agree", "1", "--name", "on"])
    off = D.read_jsonl(tmp_path / "results/runs/off/states.jsonl", VerificationState)
    on = D.read_jsonl(tmp_path / "results/runs/on/states.jsonl", VerificationState)
    assert {s.verdicts[0].verdict for s in off} == {None} and {s.verdicts[0].verdict for s in on} == {0}


def debating_verifier(first_reply, second_reply):
    """Fake verifier: `first_reply` in round 1, `second_reply` once it has seen the other reviewers."""
    return lambda messages: second_reply if "Other independent reviewers" in messages[-1]["content"] else first_reply


def debate_panel():
    long_reasoning = "The dose in the answer does not match the trial. " * 12 + "\nVerdict: FACTUAL"
    return {
        "llama": FakeModel("fake-llama", debating_verifier("Wrong dose.\nVerdict: HALLUCINATED", "Verdict: FACTUAL")),
        "mistral-local": FakeModel("fake-mistral", debating_verifier("Wrong dose.\nVerdict: HALLUCINATED", "Verdict: FACTUAL")),
        "phi": FakeModel("fake-phi", debating_verifier(long_reasoning, "Still fine.\nVerdict: FACTUAL")),
    }


def test_debate_decides_on_round_two_and_keeps_round_one(rows):
    items = D.rows_to_items(rows)[:2]
    normal, debating = debate_panel(), debate_panel()
    run_pipeline(items, normal, min_agree=2)
    states = run_pipeline(items, debating, min_agree=2, debate=True)
    s = states[0]

    assert [v.verdict for v in s.first_round()] == [1, 1, 0] and [v.verdict for v in s.last_round()] == [0, 0, 0]
    assert s.rounds == 2 and s.final_round1 == 1 and s.final == 0
    assert final_verdicts(states)[0].verdict == 0 and final_verdicts(states, round=1)[0].verdict == 1
    assert verifier_verdicts(states)["llama"][0].verdict == 1 and len(verifier_verdicts(states)["llama"]) == 2
    assert [st.action for st in s.steps].count("judge") == 6 and any(st.action == "debate" for st in s.steps)

    # Round 1 of a debate run sends exactly the messages of a normal run (so it comes from the cache).
    for name in normal:
        assert debating[name].calls[0] == normal[name].calls[0]

    # Round 2: own round-1 reply, then the OTHER reviewers, anonymised, reasoning cut to 400 characters.
    phi_turn = debating["phi"].calls[1]
    assert phi_turn[-2] == {"role": "assistant", "content": s.first_round()[2].raw_text}
    shown = phi_turn[-1]["content"]
    assert "Reviewer A: HALLUCINATED" in shown and "Reviewer B: HALLUCINATED" in shown and "Reviewer C" not in shown
    assert "llama" not in shown.lower() and "mistral" not in shown.lower() and "fake" not in shown
    llama_turn = debating["llama"].calls[1][-1]["content"]
    assert "Reviewer B: FACTUAL" in llama_turn and " ..." in llama_turn
    reasoning = llama_turn.split("Reviewer B: FACTUAL\nReasoning: ")[1].split(" ...")[0]
    assert len(reasoning) == 400
    assert llama_turn.rstrip().endswith("Verdict: UNSURE")  # same reply format as the prompt style


def test_evaluate_round_and_forced_are_saved_next_to_metrics(rows, tmp_path, monkeypatch):
    from medverify import cli
    import medverify.llm as llm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "data/processed")
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "results/runs")
    D.write_jsonl(tmp_path / "data/processed/dev.jsonl", rows[:3])
    panel = debate_panel()
    monkeypatch.setattr(llm, "make_model", lambda name, **kw: panel[name])
    cli.main(["pipeline", "--split", "dev", "--debate", "--name", "deb"])
    run = tmp_path / "results/runs/deb"
    saved = json.loads((run / "metrics.json").read_text())
    cli.main(["evaluate", str(run), "--round", "1"])
    cli.main(["evaluate", str(run), "--forced"])
    assert json.loads((run / "metrics.json").read_text()) == saved  # not overwritten
    round1 = json.loads((run / "metrics-round1.json").read_text())["overall"]
    forced = json.loads((run / "metrics-forced.json").read_text())["overall"]
    assert saved["overall"]["precision"] == 0.0  # round 2: everyone says factual
    assert round1["recall"] == 1.0  # round 1: majority says hallucinated
    assert forced["coverage"] == 1.0 and forced["abstain_rate"] == 0.0


def test_phase1_leakage_and_efficiency_match_the_readme():
    from medverify.metrics import efficiency_summary, leakage_summary

    states = D.read_jsonl(PHASE1 / "sysA-val-reason-states.jsonl", VerificationState)
    leak = leakage_summary(states)
    assert leak["round1"]["items_searched"] == 200 and leak["round1"]["source_pmid_share"] == pytest.approx(0.81)
    assert round(leak["round1"]["mean_clean_abstracts"], 1) == 2.5 and leak["round2"] == {"items_searched": 0}
    eff = efficiency_summary(states)
    assert eff["model_calls_per_item"] == 3 and round(eff["model_seconds_per_item"]) == 12
    assert eff["abstain_share"] == pytest.approx(0.11) and eff["tool_calls_logged"] is False


def test_no_block_run_counts_leaks_into_the_evidence(rows, tmp_path):
    from medverify.metrics import leakage_summary

    item = D.rows_to_items(rows)[1]
    session = FakeSession(["1000", "2001"], [("1000", "Source", " ".join(item.knowledge)),
                                             ("2001", "Other study", "An unrelated trial in older adults.")])
    pubmed = PubMedSearch(cache_dir=tmp_path, session=session)
    pubmed.min_interval = 0
    verifiers = {"a": FakeModel("a", lambda m: "Verdict: HALLUCINATED")}
    blocked = run_pipeline([item], verifiers, "pubmed", pubmed, k=3, min_agree=1)
    kept = run_pipeline([item], verifiers, "pubmed", pubmed, k=3, min_agree=1, block=False)
    assert "PMID 1000" in verifiers["a"].calls[1][-1]["content"]  # under --no-block the source reaches the verifier
    assert kept[0].steps[1].detail["leaked"] == [{"pmid": "1000", "leak": "source_pmid"}]
    for states, leaked in ((blocked, 0.0), (kept, 1.0)):
        r1 = leakage_summary(states)["round1"]
        assert r1["source_pmid_share"] == 1.0 and r1["leaked_into_evidence_share"] == leaked
        assert r1["mean_clean_abstracts"] == 1.0
