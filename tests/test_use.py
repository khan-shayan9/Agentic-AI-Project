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


# ------------------------------------------------------------- verify command

import json  # noqa: E402

import medverify.llm as llm  # noqa: E402
import medverify.tools.pubmed as pm  # noqa: E402
from medverify import cli, service  # noqa: E402
from medverify.schemas import VerificationState  # noqa: E402

REASONS = {"llama": "The trial found lower, not higher, blood pressure.", "mistral-local": "The mechanism is invented.",
           "phi": "Hard to say from these abstracts."}


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """Fake verifiers and a fake PubMed; no experiments/final_config.json unless a test writes one."""
    monkeypatch.chdir(tmp_path)
    calls = {}

    def make_model(name, **kw):
        verdict = "UNSURE" if name == "phi" else "HALLUCINATED"
        calls.setdefault(name, FakeModel(f"fake-{name}", lambda m, r=f"{REASONS[name]}\nVerdict: {verdict}": r))
        return calls[name]

    monkeypatch.setattr(llm, "make_model", make_model)
    monkeypatch.setattr(pm, "PubMedSearch", lambda: fake_pubmed(tmp_path))
    monkeypatch.setattr(service, "FINAL_CONFIG", tmp_path / "final_config.json")
    return tmp_path, calls


def run_verify(capsys, *extra):
    cli.main(["verify", "--question", QUESTION, "--answer", ANSWER, *extra])
    return capsys.readouterr().out


def test_verify_prints_a_plain_english_report(offline, capsys):
    out = run_verify(capsys, "--pmid", "5555")
    assert "Decision: REJECT (hallucinated" in out and "Agreement: 2 of 3 verifiers said hallucinated" in out
    for name, reason in REASONS.items():
        assert f"{name} (round 1)" in out and reason in out
    assert "PMID 2001" in out and "https://pubmed.ncbi.nlm.nih.gov/2002/" in out
    assert "PMID 5555: the answer's own source abstract" in out and "PMID 5556: a near copy of the source abstract" in out
    assert "Second round: off [off; auto: off, because experiments/final_config.json does not exist yet]" in out
    assert "model calls" in out and "Research prototype, not medical advice." in out


def test_verify_json_and_save(offline, capsys):
    tmp, _ = offline
    out = run_verify(capsys, "--json", "--save", str(tmp / "state.json"))
    state = VerificationState.model_validate(json.loads(out))
    assert state.item_id.startswith("user-") and state.final == 1 and len(state.verdicts) == 3
    assert json.loads((tmp / "state.json").read_text(encoding="utf-8"))["item_id"] == state.item_id


def test_verify_second_round_auto_reads_final_config(offline, capsys):
    tmp, calls = offline
    (tmp / "final_config.json").write_text(json.dumps({"trigger": "split"}))
    out = run_verify(capsys)
    assert "[split; auto: the trigger chosen on the validation split" in out
    assert "Second round: ran" in out and len(calls["llama"].calls) == 2  # votes 1, 1, 2 are split: judged again
    assert service.resolve_second_round("no-majority") == ("no-majority", "set by you")


def test_verify_not_sure_explains_why(offline, capsys, monkeypatch):
    monkeypatch.setattr(llm, "make_model", lambda name, **kw: FakeModel(name, lambda m: "Unclear.\nVerdict: UNSURE"))
    out = run_verify(capsys, "--evidence", "none")
    assert "Decision: NOT SURE (sent for human review)" in out and "Why not sure: Most verifiers were unsure" in out
    with pytest.raises(SystemExit, match="--evidence source needs --pmid"):
        run_verify(capsys, "--evidence", "source")
