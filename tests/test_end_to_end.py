"""Full offline run of every command with fake models, a fake dataset and no network."""

import json

import pytest

import medverify.data as D
import medverify.llm as llm
from medverify import cli
from medverify.baseline import run_baseline
from medverify.llm import FakeModel
from medverify.pipeline import final_verdicts, run_pipeline, verifier_verdicts


def answer_reader(correct_every: int):
    """Fake model: says 1 when the answer mentions 'increased', else 0; wrong on some items."""
    counter = {"n": 0}

    def reply(messages):
        counter["n"] += 1
        text = messages[-1]["content"]
        truth = "1" if "increased symptom" in text.split("Answer:")[-1] else "0"
        if counter["n"] % correct_every == 0:
            return "1" if truth == "0" else "0"
        return truth

    return reply


@pytest.fixture
def workspace(tmp_path, monkeypatch, rows):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "data/processed")
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "results/runs")
    monkeypatch.setattr(D, "load_medhallu", lambda subset: [r.model_copy(update={"pubid": None}) for r in rows])
    pubmedqa = {r.pubid: {"question": r.question, "contexts": r.knowledge} for r in rows}
    monkeypatch.setattr(D, "load_pubmedqa_labeled", lambda path: pubmedqa)
    models = {
        "llama": FakeModel("fake-llama", answer_reader(4)),
        "mistral": FakeModel("fake-mistral", answer_reader(5)),
        "phi": FakeModel("fake-phi", answer_reader(7)),
    }
    monkeypatch.setattr(llm, "make_model", lambda name, **kw: models[name])
    return tmp_path, models


def test_cli_full_workflow(workspace, capsys):
    tmp, models = workspace
    cli.main(["prepare", "--test-per-level", "2", "--val-size", "6", "--dev-size", "3"])
    summary = json.loads((tmp / "data/processed/split_summary.json").read_text())
    assert summary["pubid_mapping"] == {"matched_by_question": 15}
    assert summary["splits"]["test"]["rows"] == 6

    cli.main(["baseline", "--model", "llama", "--split", "test", "--name", "b1"])
    metrics = json.loads((tmp / "results/runs/b1/metrics.json").read_text())
    assert metrics["overall"]["n"] == 12 and metrics["missing_predictions"] == 0

    cli.main(["baseline", "--model", "llama", "--split", "test", "--knowledge", "source", "--name", "b2"])
    prompt = models["llama"].calls[-1][0]["content"]
    assert "World Knowledge" in prompt

    cli.main(["pipeline", "--split", "val", "--evidence", "source", "--name", "p1"])
    states = D.read_jsonl(tmp / "results/runs/p1/states.jsonl")
    assert len(states) == 12 and all(len(s["verdicts"]) == 3 for s in states)
    assert states[0]["evidence"][0]["title"] == "Source abstract"

    cli.main(["evaluate", "results/runs/p1"])
    assert (tmp / "results/runs/p1/risk_coverage.json").exists()
    cli.main(["tune", "results/runs/p1", "--min-coverage", "0.5"])
    cli.main(["overlap", "results/runs/p1", "--out", str(tmp / "overlap.json")])
    overlap = json.loads((tmp / "overlap.json").read_text())
    assert set(overlap) == {"llama vs mistral", "llama vs phi", "mistral vs phi"}
    out = capsys.readouterr().out
    assert "overall" in out and "chosen" in out


def test_baseline_resumes_without_new_calls(tmp_path, rows):
    items = D.rows_to_items(rows)[:6]
    model = FakeModel("m", lambda msgs: "1")
    out = tmp_path / "pred.jsonl"
    run_baseline(items[:3], model, out_path=out)
    assert len(model.calls) == 3
    run_baseline(items, model, out_path=out)
    assert len(model.calls) == 6  # only the 3 new items were sent
    assert len(D.read_jsonl(out)) == 6
    run_baseline(items, model, out_path=out)
    assert len(model.calls) == 6  # everything already done


def test_source_setting_passes_whole_abstract(rows):
    long_ctx = "finding " * 600  # ~4,800 characters, longer than the 1,000-char evidence cut
    item = D.rows_to_items([rows[0].model_copy(update={"knowledge": [long_ctx]})])[0]
    model = FakeModel("m", lambda m: "0")
    run_pipeline([item], {"m": model}, evidence_mode="source", min_agree=1)
    assert long_ctx.strip() in model.calls[0][1]["content"]


def test_pipeline_vote_and_views(rows):
    items = D.rows_to_items(rows)[:4]
    verifiers = {
        "a": FakeModel("a", lambda m: "1"),
        "b": FakeModel("b", lambda m: "1"),
        "c": FakeModel("c", lambda m: "0"),
    }
    states = run_pipeline(items, verifiers, evidence_mode="none", min_agree=2)
    assert all(s.final == 1 for s in states)
    assert all(v.verdict == 2 for v in final_verdicts(states, min_agree=3))  # unanimity required -> abstain
    per = verifier_verdicts(states)
    assert set(per) == {"a", "b", "c"} and len(per["a"]) == 4
    # Independence by design: every verifier gets exactly the same input for an item,
    # so none can see another's verdict.
    a_calls, b_calls, c_calls = (verifiers[n].calls for n in "abc")
    assert len(a_calls) == len(b_calls) == len(c_calls) == 4
    assert a_calls == b_calls == c_calls


def test_pipeline_stops_cleanly_when_a_verifier_keeps_failing(workspace):
    tmp, models = workspace
    cli.main(["prepare", "--test-per-level", "2", "--val-size", "6", "--dev-size", "3"])

    def always_rate_limited(messages):
        raise RuntimeError("Error code: 429 - Rate limit exceeded")

    models["mistral"] = FakeModel("fake-mistral", always_rate_limited)
    models["mistral"].provider = "mistral"
    with pytest.raises(SystemExit) as stop:
        cli.main(["pipeline", "--split", "dev", "--name", "rl"])
    message = str(stop.value)
    assert "verifier 'mistral'" in message and "0 of 6 items were saved" in message
    assert "mistral-local" in message

    # evaluate and overlap on the empty run give a clear message, not a FileNotFoundError
    for command in (["evaluate", str(tmp / "results/runs/rl")], ["overlap", str(tmp / "results/runs/rl")]):
        with pytest.raises(SystemExit, match="did not finish any item"):
            cli.main(command)


def test_pipeline_refuses_to_mix_verifier_sets(workspace):
    tmp, models = workspace
    models["mistral-local"] = FakeModel("fake-mistral-local", answer_reader(6))
    cli.main(["prepare", "--test-per-level", "2", "--val-size", "6", "--dev-size", "3"])
    cli.main(["pipeline", "--split", "dev", "--name", "mix"])
    with pytest.raises(SystemExit, match="new --name"):
        cli.main(["pipeline", "--split", "dev", "--verifiers", "llama,mistral-local,phi", "--name", "mix"])
    cli.main(["pipeline", "--split", "dev", "--verifiers", "llama,mistral-local,phi", "--name", "mix-local"])
    assert (tmp / "results/runs/mix-local/metrics.json").exists()


def test_mistral_local_alias():
    assert llm.resolve_model("mistral-local") == ("ollama", "mistral:7b")
    assert llm.MODEL_FAMILY["mistral-local"] == "Mistral"
