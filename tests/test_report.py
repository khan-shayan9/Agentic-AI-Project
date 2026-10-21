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


# ------------------------------------------------------------- statistics

import random  # noqa: E402

from experiments import stats  # noqa: E402


def test_mean_sd():
    assert stats.mean_sd([1, 2, 3, 4]) == (2.5, pytest.approx(1.2909944), 4)
    assert stats.mean_sd([0.5]) == (0.5, None, 1) and stats.mean_sd([None, None]) == (None, None, 0)


def test_mcnemar_exact():
    assert round(stats.mcnemar_exact(10, 2), 4) == 0.0386
    assert stats.mcnemar_exact(0, 0) == 1.0 and stats.mcnemar_exact(5, 5) == 1.0
    # items: A right/B wrong, both right, A right/B wrong, A abstains (left out), both wrong, A wrong/B right
    result = stats.mcnemar([1, 1, 0, 0, 1, 0], [1, 1, 0, 2, 0, 1], [0, 1, 1, 0, 0, 0])
    assert result == {"both_answered": 5, "b": 2, "c": 1, "p": 1.0}


def test_paired_bootstrap_is_deterministic_and_covers_the_true_difference():
    rng = random.Random(7)
    labels = [i % 2 for i in range(400)]
    a = [t if rng.random() < 0.85 else 1 - t for t in labels]  # A is right 85% of the time
    b = [t if rng.random() < 0.65 else 1 - t for t in labels]  # B 65%
    first = stats.paired_bootstrap(labels, a, b, n_resamples=2000)
    assert first == stats.paired_bootstrap(labels, a, b, n_resamples=2000)  # seed 0: same every time
    true_acc = sum(x == t for x, t in zip(a, labels)) / 400 - sum(x == t for x, t in zip(b, labels)) / 400
    assert first["acc_diff"] == pytest.approx(true_acc)
    assert first["acc_ci_low"] <= true_acc <= first["acc_ci_high"] and first["acc_ci_low"] > 0
    assert first["f1_ci_low"] <= first["f1_diff"] <= first["f1_ci_high"] and first["f1_p"] < 0.01
    same = stats.paired_bootstrap(labels, a, a, n_resamples=500)
    assert same["f1_diff"] == 0 and same["f1_p"] == 1.0


# ------------------------------------------------------------- tables

import csv  # noqa: E402
import shutil  # noqa: E402

from experiments import report  # noqa: E402


@pytest.fixture(scope="module")
def reported(workspace):
    """The report built once for the test and val splits from the shared workspace."""
    with pytest.MonkeyPatch.context() as mp:
        phase2_fixture.patch(mp, workspace)
        assert report.main(["--split", "test"]) == 0
        assert report.main(["--split", "val"]) == 0
    return workspace


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_runs_table_matches_the_source_run_folders(reported):
    rows = {r["run"]: r for r in read_csv(reported / "results/phase2/tables/runs.csv")}
    assert set(rows) == {r.name for r in plan.TEST_RUNS} - set(phase2_fixture.SKIPPED)
    for name, row in rows.items():
        saved = json.loads((reported / f"results/runs/{name}/metrics.json").read_text())["overall"]
        assert float(row["f1"]) == pytest.approx(saved["f1"]) and float(row["coverage"]) == pytest.approx(saved["coverage"])
        assert [float(row["f1_ci_low"]), float(row["f1_ci_high"])] == pytest.approx(saved["f1_ci95"])
    assert rows["b0-test-s1"]["model_calls"] == "1.0"


def test_report_needs_only_results_phase2(reported, tmp_path, monkeypatch):
    clone = tmp_path / "clone"
    shutil.copytree(reported / "results/phase2", clone / "results/phase2", ignore=shutil.ignore_patterns("tables", "figures"))
    phase2_fixture.patch(monkeypatch, clone)  # no results/runs, no data/processed, no experiments/ here
    assert report.main(["--split", "test"]) == 0
    for name in ("runs", "main", "overlap", "leakage", "ablation", "significance"):
        assert (clone / f"results/phase2/tables/{name}.csv").read_text() == (reported / f"results/phase2/tables/{name}.csv").read_text()


def test_missing_runs_are_listed_not_invented(reported):
    tables = reported / "results/phase2/tables"
    for md in tables.glob("*.md"):
        assert "Missing runs: debate-test-s1" in md.read_text(encoding="utf-8"), md.name
    debate = next(r for r in read_csv(tables / "significance.csv") if r["comparison"].startswith("Debate"))
    assert debate["missing"] == "runs missing" and debate["f1_diff"] == "" and debate["mcnemar_p"] == ""
    assert not any("debate" in r["run"] for r in read_csv(tables / "runs.csv"))


def test_main_overlap_leakage_and_significance(reported):
    tables = reported / "results/phase2/tables"
    main = {r["config"]: r for r in read_csv(tables / "main.csv")}
    assert main["sysA-test"]["n_seeds"] == "3" and main["sysB-test"]["n_seeds"] == "1"
    assert main["paper-b0"]["f1_mean"] == "0.522" and main["paper-b0"]["recall_mean"] == "" and main["paper-b1"]["source"] == "paper"
    assert {"b3-llama", "b3-mistral-local", "b3-phi", "sysA-round1", "sysA-forced"} <= set(main)
    assert float(main["sysA-forced"]["coverage_mean"]) == 1.0
    overlap = read_csv(tables / "overlap.csv")
    assert {r["pair_type"] for r in overlap} == {"cross-family", "same model"}
    assert len([r for r in overlap if r["level"] == "pair" and r["run"] == "sysA-test-s1"]) == 3
    leak = {(r["run"], r["round"]): r for r in read_csv(tables / "leakage.csv")}
    assert leak[("sysA-test-s1", "round1")]["kept_leaks"] == "0" and float(leak[("sysA-test-s1", "round1")]["source_pmid_share"]) > 0
    assert int(leak[("noblock-test-s1", "round1")]["kept_leaks"]) > 0
    sig = {r["comparison"]: r for r in read_csv(tables / "significance.csv")}
    a_vs_b0 = sig["System A vs B0"]
    assert a_vs_b0["n"] == "24" and a_vs_b0["n_seeds"] == "3" and a_vs_b0["mean_diff_f1"] != ""
    assert sig["System B vs B1"]["mean_diff_f1"] == ""  # System B has one seed: no mean over seeds
    best = next(k for k in sig if k.startswith("System A round-1 vote vs best single verifier"))
    assert "chosen on val" in best and sig[best]["f1_diff"] != ""
    ablation = {r["component"]: r for r in read_csv(tables / "ablation.csv")}
    assert {"Second round off", "Forced answer (no abstention)", "Vote 3 of 3", "Drop phi (2 of 2 must agree)"} <= set(ablation)
    assert float(ablation["Vote 2 of 3"]["delta_f1"]) == 0.0
    assert (reported / "results/phase2/tables/val/runs.csv").exists()


# ------------------------------------------------------------- gap table

from experiments import gaps  # noqa: E402


def test_gap_table_has_the_brief_columns_and_follows_the_rule(reported):
    rows = read_csv(reported / "results/phase2/tables/gap_resolution.csv")
    assert list(rows[0]) == gaps.COLUMNS and [r["Gap"][0] for r in rows] == ["1", "2", "3"]
    assert all(r["Addressed?"] in ("Fully", "Partially", "Not", "Pending") for r in rows)
    md = (reported / "results/phase2/tables/gap_resolution.md").read_text(encoding="utf-8")
    assert "Missing runs: debate-test-s1" in md and "The group reviewed the rule's output." in md
    assert "(reported 0.522)" in rows[1]["Base paper result"] and "kept leaks in setting-A runs: 0" in rows[1]["Our result"]
    # gap 2 in the fake workspace: no leaks, System A vs B0 and System B vs B1 decided by the rule from their numbers
    sig = {r["comparison"]: r for r in read_csv(reported / "results/phase2/tables/significance.csv")}

    def met(name):
        r = sig[name]
        return facts.comparison_met(float(r["mean_diff_f1"]) if r["mean_diff_f1"] else float(r["f1_diff"]), float(r["f1_ci_low"]))

    assert rows[1]["Addressed?"] == facts.addressed([True, met("System A vs B0"), met("System B vs B1")])


def test_gap_table_is_pending_without_runs_and_goes_into_the_readme(tmp_path, monkeypatch):
    phase2 = tmp_path / "results/phase2"
    phase2.mkdir(parents=True)
    (phase2 / "items_test.csv").write_text("item_id,label,difficulty,category\nx-gt,0,easy,c\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{gaps.START}\nold\n{gaps.END}\n\nafter\n", encoding="utf-8")
    rep = report.build("test", phase2, resamples=200)
    rows = gaps.write_gap_table(rep, readme)
    assert [r["Addressed?"] for r in rows] == ["Pending", "Pending", "Pending"]
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and text.endswith("after\n") and "| Gap | Method / component |" in text
    assert "Missing runs: b0-test-s1" in text and "pending (runs missing)" in text
    assert gaps.update_readme(tmp_path / "README.md", "again") and "again" in readme.read_text(encoding="utf-8")


# ------------------------------------------------------------- failure analysis

from experiments import failures  # noqa: E402
from medverify.schemas import Verdict as V, VerificationState as S  # noqa: E402


def make_state(item_id, final, last, first=None, flags=(), final_round1=None):
    """A hand-made state: `first` = round-1 votes, `last` = round-2 votes (if different)."""
    first = first if first is not None else last
    verdicts = [V(item_id=item_id, system=f"verifier:v{i}", verdict=c, raw_text="", provider="f", model="m", latency_s=0)
                for i, c in enumerate(first)]
    rounds = 1
    if first is not last:
        rounds = 2
        verdicts += [V(item_id=item_id, system=f"verifier:v{i}", verdict=c, raw_text="", provider="f", model="m",
                       latency_s=0, round=2) for i, c in enumerate(last)]
    return S(item_id=item_id, evidence_mode="pubmed", verdicts=verdicts, final=final, rounds=rounds,
             final_round1=final if final_round1 is None else final_round1, flags=list(flags))


@pytest.mark.parametrize("label, final, votes, expected", [
    (1, 1, [1, None, 1], "unreadable"),
    (1, 2, [1, 2, 2], "abstained - majority would have been right"),
    (0, 2, [1, 2, 2], "abstained - majority would have been wrong"),
    (0, 2, [1, 0, 2], "abstained - votes tied"),
    (1, 0, [0, 0, 0], "missed hallucination - all verifiers fooled"),
    (1, 0, [0, 0, 1], "missed hallucination - majority fooled"),
    (0, 1, [1, 1, 1], "false alarm - all verifiers"),
    (0, 1, [1, 1, 2], "false alarm - majority"),
    (0, 0, [0, 0, 1], "correct"),
])
def test_failure_categories(label, final, votes, expected):
    assert failures.categorise(make_state("x", final, votes), label) == expected


def test_round_two_flips():
    states = [
        make_state("a-hal", 1, [1, 1, 1], first=[1, 0, 2], final_round1=2),  # abstain -> right
        make_state("b-gt", 1, [1, 1, 0], first=[0, 0, 1], final_round1=0),  # right -> wrong
        make_state("c-hal", 1, [1, 1, 1], first=[0, 0, 1], final_round1=0),  # wrong -> right
        make_state("d-gt", 0, [0, 0, 0]),  # round 1 only, and these hand-made states have no evidence
        make_state("e-gt", 0, [0, 0, 1], flags=["no_new_evidence"]),
    ]
    labels = {"a-hal": 1, "b-gt": 0, "c-hal": 1, "d-gt": 0, "e-gt": 0}
    assert [failures.evidence_state(s, labels[s.item_id]) for s in states] == [
        "round 2 flipped abstain->right", "round 2 flipped right->wrong", "round 2 flipped wrong->right",
        "no clean evidence", "second round triggered, no new evidence"]
    rows = [{"label": labels[s.item_id], "rounds": s.rounds, "round1_decision": s.final_round1, "final_decision": s.final}
            for s in states]
    assert failures.flip_table(rows) == {("abstain", "right"): 1, ("right", "wrong"): 1, ("wrong", "right"): 1}


def test_failure_analysis_cases_and_samples(reported):
    tables = reported / "results/phase2/tables"
    rows = read_csv(tables / "failures.csv")
    assert len(rows) == 24 and all(r["failure_category"] in failures.CATEGORIES for r in rows)
    summary = (tables / "failures_summary.md").read_text(encoding="utf-8")
    assert "Missing runs: debate-test-s1" in summary and "## By difficulty" in summary and "## Round-2 flips" in summary
    typed = [r | {"label": int(r["label"]), "rounds": int(r["rounds"]), "round1_decision": int(r["round1_decision"]),
                  "final_decision": int(r["final_decision"])} for r in rows]
    picks = failures.pick_cases(typed)
    assert picks and picks == sorted(picks, key=lambda p: not p[0].startswith("success"))  # successes first
    for _, item_id in picks:
        case = (reported / f"logs/cases/{item_id}.md").read_text(encoding="utf-8")
        assert "**Question:**" in case and "## Steps" in case and "verifier: judge" in case
        assert json.loads((reported / f"logs/cases/{item_id}.json").read_text(encoding="utf-8"))["state"]["item_id"] == item_id
    samples = sorted(p.name for p in (reported / "logs/samples").iterdir())
    chosen = report.CHOSEN_VAL_RUN[phase2_fixture.final_config(reported)["trigger"]]
    assert samples == sorted([f"{chosen}.jsonl", "sysA-test-s1.jsonl", "b4-test-s1.jsonl"])  # debate is missing
    lines = (reported / "logs/samples/sysA-test-s1.jsonl").read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["item_id"] for line in lines]
    assert len(ids) == 20 and ids == sorted(ids)
    assert (tables / "val/failures.csv").exists() and any((reported / "logs/cases/val").iterdir())


# ------------------------------------------------------------- figures


def test_figures_are_written_with_the_agg_backend(reported):
    import matplotlib

    from experiments import figures  # noqa: F401  (importing it selects Agg)

    assert matplotlib.get_backend().lower() == "agg"
    folder = reported / "results/phase2/figures"
    for name in ("f1_by_difficulty", "risk_coverage", "error_overlap", "ablation", "cost"):
        for ext in ("png", "pdf"):
            assert (folder / f"{name}.{ext}").stat().st_size > 1000, f"{name}.{ext}"
    assert (folder / "val/risk_coverage.png").exists()
    assert figures._pair_label("llama@1 vs llama@2") == "Llama#1-Llama#2"
