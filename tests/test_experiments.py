"""Phase 2 run plan, runner and val-only trigger choice (offline, fake models)."""

import json

import pytest

import medverify.data as D
import medverify.llm as llm
from experiments import plan
from experiments import run as runner
from medverify import cli
from medverify.llm import FakeModel


@pytest.fixture
def workspace(tmp_path, monkeypatch, rows):
    """Prepared dev/val/test splits in a temp folder, fake models that count their calls."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "data/processed")
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "results/runs")
    monkeypatch.setattr(runner, "FINAL_CONFIG", tmp_path / "final_config.json")
    for split, part in (("dev", rows[:4]), ("val", rows[4:8]), ("test", rows[8:])):
        D.write_jsonl(tmp_path / f"data/processed/{split}.jsonl", part)
    models = {}

    def make_model(name, **kw):
        models.setdefault(name, FakeModel(f"fake-{name}", lambda m: "Checked.\nVerdict: FACTUAL"))
        return models[name]

    monkeypatch.setattr(llm, "make_model", make_model)
    return tmp_path, models


def test_plan_names_are_unique_and_cover_e1_to_e5():
    names = [r.name for r in plan.ALL_RUNS]
    assert len(names) == len(set(names))
    assert {r.experiment for r in plan.TEST_RUNS} == {"E1", "E2", "E3", "E5"}
    assert [r.name for r in plan.VAL_RUNS] == ["val-sysA-off", "val-sysA-split", "val-sysA-nomaj"]
    assert {f"{b}-test-s{s}" for b in ("b0", "b1", "b2", "sysA", "b4") for s in (1, 2, 3)} <= set(names)
    b4 = next(r for r in plan.TEST_RUNS if r.name == "b4-test-s2")
    assert "llama@1,llama@2,llama@3" in b4.args and b4.args[b4.args.index("--temperature") + 1] == "0.7"
    assert all("--split" not in r.args and "--name" not in r.args for r in plan.ALL_RUNS)


def test_every_command_parses(workspace):
    (workspace[0] / "final_config.json").write_text(json.dumps({"trigger": "split"}))
    parser = cli.build_parser()
    for stage in ("smoke", "val", "test"):
        for job in runner.stage_jobs(stage):
            parser.parse_args(job.argv)


def test_test_stage_needs_final_config_and_uses_its_trigger(workspace):
    tmp, _ = workspace
    with pytest.raises(SystemExit, match="final_config.json"):
        runner.stage_jobs("test")
    with pytest.raises(SystemExit, match="final_config.json"):
        runner.main(["--stage", "test", "--dry-run"])
    (tmp / "final_config.json").write_text(json.dumps({"trigger": "no-majority"}))
    jobs = {j.name: j for j in runner.stage_jobs("test")}
    assert all("TRIGGER" not in j.argv for j in jobs.values())
    for job in jobs.values():
        assert job.argv[job.argv.index("--split") + 1] == "test" and "--seed" in job.argv and "--limit" not in job.argv
    sys_a = jobs["sysA-test-s2"].argv
    assert sys_a[sys_a.index("--second-round") + 1] == "no-majority" and sys_a[sys_a.index("--seed") + 1] == "2"
    assert jobs["noblock-test-s1"].argv[jobs["noblock-test-s1"].argv.index("--second-round") + 1] == "no-majority"
    assert jobs["debate-test-s1"].argv[jobs["debate-test-s1"].argv.index("--second-round") + 1] == "off"


def test_smoke_and_val_stages(workspace):
    smoke = runner.stage_jobs("smoke")
    assert len(smoke) == len(plan.ALL_RUNS) and all(j.name.startswith("smoke-") for j in smoke)
    for job in smoke:
        assert job.argv[job.argv.index("--split") + 1] == "dev" and job.argv[job.argv.index("--limit") + 1] == "6"
        if "--second-round" in job.argv:
            assert job.argv[job.argv.index("--second-round") + 1] in ("off", "split", "no-majority")
    val = runner.stage_jobs("val")
    assert [j.name for j in val] == ["val-sysA-off", "val-sysA-split", "val-sysA-nomaj"]
    assert all(j.argv[j.argv.index("--split") + 1] == "val" for j in val)
    assert [j.name for j in runner.select(smoke, "E2", "smoke-b0-test-s1,b1-test-s2")] == ["smoke-b0-test-s1", "smoke-b1-test-s2"]


def jobs_without_pubmed():
    """A baseline and a pipeline job that need no PubMed (for offline runner tests)."""
    base = plan.Run("t-base", "E2", "baseline", "test", ("--model", "llama", "--seed", "1"))
    pipe = plan.Run("t-pipe", "E5", "pipeline", "test", (*plan.SYSTEM, "--evidence", "none", "--seed", "1"))
    return [runner._job(r, r.name, "dev", None, limit=8) for r in (base, pipe)]


def test_runner_resumes_a_partial_run_and_skips_a_complete_one(workspace, capsys):
    tmp, models = workspace
    base, pipe = jobs_without_pubmed()
    cli.main(base.argv[:-2] + ["--limit", "3"])  # a partial baseline: 3 of 8 items
    assert runner.items_done(base) == 3 and len(models["llama"].calls) == 3

    assert runner.run_jobs([base, pipe]) == 0
    assert len(models["llama"].calls) == 3 + 5 + 8  # 5 new baseline items, then 8 pipeline items
    assert runner.items_done(base) == runner.items_done(pipe) == 8
    out = capsys.readouterr().out
    assert "resuming after 3 items" in out and "t-pipe: coverage" in out

    assert runner.run_jobs([base, pipe]) == 0
    assert len(models["llama"].calls) == 16  # nothing repeated
    assert "skip t-base: complete (8/8)" in capsys.readouterr().out

    runner.run_jobs([base], dry_run=True)
    assert "t-base: 0 items to go (8 done)" in capsys.readouterr().out


def test_runner_stops_at_the_first_failure(workspace, capsys, monkeypatch):
    tmp, models = workspace

    def broken(name, **kw):
        return FakeModel("broken", lambda m: (_ for _ in ()).throw(RuntimeError("Ollama is not running")))

    monkeypatch.setattr(llm, "make_model", broken)
    base, pipe = jobs_without_pubmed()
    assert runner.run_jobs([pipe, base]) == 1
    out = capsys.readouterr().out
    assert "Run t-pipe failed" in out and "t-base" not in out
