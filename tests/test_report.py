"""Phase 2 report: facts, statistics, export, tables, gap table, failure analysis and figures (offline)."""

import pytest

from experiments import facts


def test_addressed_rule():
    met = facts.comparison_met
    assert met(0.02, 0.001) is True and met(0.02, -0.01) is False and met(-0.01, 0.002) is False
    assert met(None, 0.1) is None and met(0.1, None) is None
    assert facts.coverage_met(0.86, 0.89) is True and facts.coverage_met(0.83, 0.89) is False
    assert facts.coverage_met(0.84, 0.89) is True  # exactly 5 points
    assert facts.leakage_met(0) is True and facts.leakage_met(2) is False and facts.leakage_met(None) is None
    assert facts.addressed([True, True, True]) == "Fully"
    assert facts.addressed([True, False, True]) == "Partially"
    assert facts.addressed([False, False]) == "Not"
    assert facts.addressed([True, None]) == "Pending" and facts.addressed([]) == "Pending"


def test_facts_are_complete():
    assert set(facts.HYPOTHESES) == {"H1", "H2", "H3"} and set(facts.SUCCESS_CRITERIA) == {1, 2, 3}
    assert facts.MEDHALLU_REPORTED["B0"]["f1"] == 0.522 and facts.MEDHALLU_REPORTED["B1"]["hard_f1"] == 0.722
    assert facts.MEDHALLU_REPORTED["B0"]["recall"] is None  # the paper gives no recall


# ------------------------------------------------------------- shared workspace

import gzip  # noqa: E402
import json  # noqa: E402

from experiments import export  # noqa: E402
from experiments import plan  # noqa: E402

import phase2_fixture  # noqa: E402


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Every plan run (except the ones in SKIPPED) made with fakes and exported, built once."""
    root = tmp_path_factory.mktemp("phase2")
    with pytest.MonkeyPatch.context() as mp:
        phase2_fixture.build(root, mp)
    return root


@pytest.fixture
def in_workspace(workspace, monkeypatch):
    phase2_fixture.patch(monkeypatch, workspace)
    return workspace


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_export_copies_every_finished_run(in_workspace):
    root = in_workspace
    exported = sorted(p.name for p in (root / "results/phase2/runs").iterdir())
    expected = sorted(r.name for r in plan.ALL_RUNS if r.name not in phase2_fixture.SKIPPED)
    assert exported == expected
    sys_a = root / "results/phase2/runs/sysA-test-s1"
    assert {p.name for p in sys_a.iterdir()} >= {"config.json", "metrics.json", "states.jsonl.gz"}
    assert (root / "results/phase2/runs/b0-test-s1/predictions.jsonl.gz").exists()
    assert phase2_fixture.final_config(root)["trigger"] in ("split", "no-majority")
    with (root / "results/phase2/items_test.csv").open(encoding="utf-8") as f:
        header, *lines = f.read().splitlines()
    assert header.split(",")[:4] == ["item_id", "label", "difficulty", "category"] and len(lines) == 24


def test_export_cuts_evidence_and_keeps_the_rest_exactly(in_workspace):
    root = in_workspace
    source = [json.loads(l) for l in (root / "results/runs/sysA-test-s1/states.jsonl").read_text(encoding="utf-8").splitlines() if l]
    exported = read_gz(root / "results/phase2/runs/sysA-test-s1/states.jsonl.gz")
    assert len(exported) == len(source)
    for full, cut in zip(source, exported):
        assert all(len(e["text"]) <= export.EVIDENCE_CHARS for e in cut["evidence"])
        for e_full, e_cut in zip(full["evidence"], cut["evidence"]):
            assert e_cut["text"] == e_full["text"][:300] and {k: v for k, v in e_cut.items() if k != "text"} == {k: v for k, v in e_full.items() if k != "text"}
        for key in ("verdicts", "steps", "tool_calls", "flags", "final", "final_round1", "rounds", "trace"):
            assert cut[key] == full[key]


def test_export_is_deterministic_and_skips_unfinished_runs(in_workspace):
    root = in_workspace
    target = root / "results/phase2/runs/sysA-test-s1/states.jsonl.gz"
    before = target.read_bytes()
    partial = root / "results/runs/debate-test-s1"
    partial.mkdir(parents=True)
    (partial / "states.jsonl").write_text('{"item_id": "x"}\n')
    try:
        done = export.export("test")
    finally:
        import shutil
        shutil.rmtree(partial)
    assert target.read_bytes() == before
    assert done["partial"] == ["debate-test-s1 (1/24)"] and "debate-test-s1" not in done["exported"]
