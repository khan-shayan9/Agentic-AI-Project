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


# ------------------------------------------------------------- choose

from experiments import choose  # noqa: E402


def overall(coverage, accuracy, f1=0.6):
    return {"coverage": coverage, "selective_accuracy": accuracy, "f1": f1}


def test_choose_rule():
    pick = choose.choose_trigger
    assert pick({"split": overall(0.92, 0.70), "no-majority": overall(0.88, 0.66)}) == "split"
    assert pick({"split": overall(0.92, 0.64), "no-majority": overall(0.88, 0.66)}) == "no-majority"
    assert pick({"split": overall(0.92, 0.664), "no-majority": overall(0.88, 0.660)}) == "no-majority"  # within 0.005
    assert pick({"split": overall(0.92, 0.666), "no-majority": overall(0.88, 0.660)}) == "split"  # 0.006 apart
    assert pick({"split": overall(0.69, 0.80), "no-majority": overall(0.75, 0.60)}) == "no-majority"  # coverage < 0.70
    assert pick({"split": overall(0.70, 0.62), "no-majority": overall(0.65, 0.90)}) == "split"  # 0.70 counts
    with pytest.raises(SystemExit, match="No trigger reaches coverage"):
        pick({"split": overall(0.60, 0.80), "no-majority": overall(0.50, 0.90)})


def save_val_run(runs_dir, name, coverage, accuracy, split="val", limit=None):
    folder = runs_dir / name
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"split": split, "limit": limit}))
    (folder / "metrics.json").write_text(json.dumps({
        "overall": overall(coverage, accuracy),
        "efficiency": {"round2_share": 0.3, "model_seconds_per_item": 15.0},
    }))


def test_choose_writes_final_config_and_refuses_once_test_results_exist(workspace, capsys):
    tmp, _ = workspace
    runs = tmp / "results/runs"
    save_val_run(runs, "val-sysA-off", 0.89, 0.646)
    save_val_run(runs, "val-sysA-split", 0.93, 0.662)
    save_val_run(runs, "val-sysA-nomaj", 0.90, 0.659)
    assert choose.main([]) == 0
    config = json.loads((tmp / "final_config.json").read_text())
    assert config["trigger"] == "no-majority" and config["chosen_run"] == "val-sysA-nomaj"  # 0.003 apart
    assert config["val"] == {"coverage": 0.90, "selective_accuracy": 0.659, "f1": 0.6}
    assert [r["run"] for r in config["candidates"]] == ["val-sysA-off", "val-sysA-split", "val-sysA-nomaj"]
    assert "0.005" in config["rule"] and config["date"] and "git_commit" in config
    assert "Chosen trigger: no-majority" in capsys.readouterr().out
    assert runner.final_trigger() == "no-majority"

    (runs / "b0-test-s1").mkdir()
    (runs / "b0-test-s1/predictions.jsonl").write_text('{"item_id": "x"}\n')
    before = (tmp / "final_config.json").read_text()
    with pytest.raises(SystemExit, match="b0-test-s1"):
        choose.main([])
    assert (tmp / "final_config.json").read_text() == before


def test_choose_needs_full_validation_runs(workspace):
    tmp, _ = workspace
    runs = tmp / "results/runs"
    with pytest.raises(SystemExit, match="--stage val"):
        choose.main([])
    save_val_run(runs, "val-sysA-off", 0.89, 0.646)
    save_val_run(runs, "val-sysA-split", 0.93, 0.662, split="dev", limit=6)
    save_val_run(runs, "val-sysA-nomaj", 0.90, 0.659)
    with pytest.raises(SystemExit, match="not a full validation run"):
        choose.main([])


# ------------------------------------------------------------- state backup

from experiments import state  # noqa: E402
from medverify.llm import _Cache  # noqa: E402


def make_cache(cache_dir, rows, pubmed_files):
    cache = _Cache(cache_dir / "llm_cache.sqlite")
    for key, value in rows.items():
        cache.put(key, value)
    cache._conn.close()
    for name, text in pubmed_files.items():
        (cache_dir / "pubmed" / name).parent.mkdir(parents=True, exist_ok=True)
        (cache_dir / "pubmed" / name).write_text(text)


def cache_rows(cache_dir):
    import sqlite3

    conn = sqlite3.connect(str(cache_dir / "llm_cache.sqlite"))
    try:
        return dict(conn.execute("SELECT key, result FROM calls").fetchall())
    finally:
        conn.close()


def pubmed_files(cache_dir):
    folder = cache_dir / "pubmed"
    return {p.relative_to(folder).as_posix(): p.read_text() for p in folder.rglob("*") if p.is_file()}


def test_state_save_and_restore_round_trip(tmp_path):
    local, backup, final = tmp_path / "cache", tmp_path / "drive/state", tmp_path / "final_config.json"
    make_cache(local, {f"k{i}": {"text": f"reply {i}"} for i in range(5)}, {"a.txt": "esearch", "sub/b.txt": "efetch"})
    final.write_text('{"trigger": "split"}')
    rows, files = cache_rows(local), pubmed_files(local)

    saved = state.save(backup, cache_dir=local, final_config=final)
    assert len(saved) == 3 and not list(backup.glob("*.tmp"))
    fresh, fresh_final = tmp_path / "fresh", tmp_path / "fresh_final.json"
    restored = state.restore(backup, cache_dir=fresh, final_config=fresh_final)
    assert len(restored) == 3
    assert cache_rows(fresh) == rows and pubmed_files(fresh) == files
    assert fresh_final.read_text() == '{"trigger": "split"}'


def test_restore_keeps_a_larger_local_cache_and_an_existing_final_config(tmp_path, capsys):
    small, backup = tmp_path / "small", tmp_path / "state"
    make_cache(small, {"k0": {"text": "old"}}, {})
    (tmp_path / "saved_final.json").write_text('{"trigger": "split"}')
    state.save(backup, cache_dir=small, final_config=tmp_path / "saved_final.json")

    big = tmp_path / "big"
    make_cache(big, {f"k{i}": {"text": "x" * 2000} for i in range(50)}, {})
    rows = cache_rows(big)
    local_final = tmp_path / "local_final.json"
    local_final.write_text('{"trigger": "no-majority"}')
    restored = state.restore(backup, cache_dir=big, final_config=local_final)
    assert cache_rows(big) == rows and "Kept the local llm_cache.sqlite" in capsys.readouterr().out
    assert local_final.read_text() == '{"trigger": "no-majority"}' and restored == []


def test_runner_backs_up_after_each_run_and_after_a_failure(workspace, monkeypatch):
    tmp, models = workspace
    calls = []
    monkeypatch.setattr(state, "save", lambda to, **kw: calls.append(to) or ["llm_cache.sqlite"])
    base, pipe = jobs_without_pubmed()

    assert runner.run_jobs([base, pipe]) == 0 and calls == []  # no --backup-dir: never
    for name in ("t-base", "t-pipe"):
        __import__("shutil").rmtree(tmp / "results/runs" / name)
    assert runner.run_jobs([base, pipe], backup_dir=tmp / "bk") == 0
    assert calls == [tmp / "bk", tmp / "bk"]  # once per finished run
    assert runner.run_jobs([base, pipe], backup_dir=tmp / "bk") == 0 and len(calls) == 2  # skipped runs: nothing new

    def broken(name, **kw):
        return FakeModel("broken", lambda m: (_ for _ in ()).throw(RuntimeError("Ollama is not running")))

    monkeypatch.setattr(llm, "make_model", broken)
    fail = runner._job(plan.Run("t-fail", "E5", "pipeline", "test", (*plan.SYSTEM, "--evidence", "none")),
                       "t-fail", "dev", None, limit=8)
    assert runner.run_jobs([fail], backup_dir=tmp / "bk") == 1 and len(calls) == 3  # after the failure too
    assert runner.run_jobs([fail]) == 1 and len(calls) == 3


def test_runner_backs_up_when_stopped(workspace, monkeypatch):
    calls = []
    monkeypatch.setattr(state, "save", lambda to, **kw: calls.append(to) or [])

    def stopped(name, **kw):
        return FakeModel("stopped", lambda m: (_ for _ in ()).throw(KeyboardInterrupt()))

    monkeypatch.setattr(llm, "make_model", stopped)
    base, _ = jobs_without_pubmed()
    with pytest.raises(KeyboardInterrupt):
        runner.run_jobs([base], backup_dir=workspace[0] / "bk")
    assert calls == [workspace[0] / "bk"]


def test_backup_failure_is_reported_not_fatal(workspace, monkeypatch, capsys):
    def full_disk(to, **kw):
        raise OSError("No space left on device")

    monkeypatch.setattr(state, "save", full_disk)
    base, _ = jobs_without_pubmed()
    assert runner.run_jobs([base], backup_dir=workspace[0] / "bk") == 0
    assert "WARNING: backup to" in capsys.readouterr().out


def test_state_command_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "FINAL_CONFIG", tmp_path / "final_config.json")
    make_cache(tmp_path / "data/cache", {"k": {"text": "r"}}, {"a.txt": "x"})
    assert state.main(["save", "--to", "bk"]) == 0
    assert "Saved to bk: llm_cache.sqlite" in capsys.readouterr().out
    __import__("shutil").rmtree(tmp_path / "data/cache")
    assert state.main(["restore", "--from", "bk"]) == 0
    assert cache_rows(tmp_path / "data/cache") == {"k": '{"text": "r"}'}
