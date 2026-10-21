"""Run the Phase 2 plan stage by stage, one run after another, resuming where it stopped.

  python -m experiments.run --stage smoke|val|test [--only E2,E3] [--names b0-test-s1,...]
                            [--backup-dir DIR] [--dry-run]

  smoke  every val and test run on the dev split with --limit 6 (names "smoke-..."), a quick
         crash check before spending GPU time; TRIGGER = "split"
  val    the three validation runs (second round off / split / no-majority)
  test   the test runs; needs experiments/final_config.json (python -m experiments.choose),
         whose trigger replaces TRIGGER

A run whose folder already has a result for every item is skipped; a partial one resumes
(the medverify commands resume by themselves). The first failure stops the stage.
With --backup-dir, the local caches are copied there (experiments/state.py) after every
finished run, and also when a run fails or is stopped.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from medverify import cli

from . import state
from .plan import TEST_RUNS, TRIGGER, VAL_RUNS, Run

FINAL_CONFIG = Path(__file__).parent / "final_config.json"
SMOKE_LIMIT = 6
SMOKE_TRIGGER = "split"


@dataclass
class Job:
    """One run of the plan, ready for a stage: its folder name, split and full arguments."""

    run: Run
    name: str
    split: str
    limit: Optional[int]
    argv: list[str]


def final_trigger() -> str:
    """The second-round trigger chosen on val; the test stage cannot start without it."""
    if not FINAL_CONFIG.exists():
        raise SystemExit(
            f"The test stage needs {FINAL_CONFIG.as_posix()}. Run the val stage first, then "
            "`python -m experiments.choose` to pick the second-round trigger on the validation split."
        )
    trigger = json.loads(FINAL_CONFIG.read_text())["trigger"]
    if trigger not in ("split", "no-majority"):
        raise SystemExit(f"{FINAL_CONFIG.as_posix()} has an unknown trigger {trigger!r}")
    return trigger


def stage_jobs(stage: str) -> list[Job]:
    """All jobs of one stage, in plan order."""
    if stage == "smoke":
        return [_job(r, f"smoke-{r.name}", "dev", SMOKE_TRIGGER, SMOKE_LIMIT) for r in VAL_RUNS + TEST_RUNS]
    if stage == "val":
        return [_job(r, r.name, "val", None) for r in VAL_RUNS]
    if stage == "test":
        trigger = final_trigger()
        return [_job(r, r.name, "test", trigger) for r in TEST_RUNS]
    raise ValueError(f"unknown stage '{stage}'")


def _job(run: Run, name: str, split: str, trigger: Optional[str], limit: Optional[int] = None) -> Job:
    if trigger is None and TRIGGER in run.args:
        raise ValueError(f"{run.name} needs the chosen trigger")
    return Job(run, name, split, limit, run.argv(split, name, trigger or "", limit))


def select(jobs: list[Job], only: Optional[str], names: Optional[str]) -> list[Job]:
    """Keep the jobs of some experiments (--only E2,E3) and/or some run names (--names)."""
    if only:
        tags = {t.strip() for t in only.split(",") if t.strip()}
        jobs = [j for j in jobs if j.run.experiment in tags]
    if names:
        wanted = {n.strip() for n in names.split(",") if n.strip()}
        jobs = [j for j in jobs if j.name in wanted or j.run.name in wanted]
    return jobs


# ------------------------------------------------------------ progress

def items_needed(job: Job) -> Optional[int]:
    """Number of items in the job's split (after --limit); None if the data is not prepared yet."""
    try:
        return len(cli._items(job.split, job.limit))
    except FileNotFoundError:
        return None


def items_done(job: Job) -> int:
    """Items with a saved result in the run folder."""
    path = cli.RUNS_DIR / job.name / ("predictions.jsonl" if job.run.kind == "baseline" else "states.jsonl")
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def summary_line(job: Job) -> str:
    done, needed = items_done(job), items_needed(job)
    path = cli.RUNS_DIR / job.name / "metrics.json"
    if not path.exists():
        return f"{job.name}: no metrics yet, items {done}/{needed}"
    overall = json.loads(path.read_text())["overall"]
    return f"{job.name}: coverage {overall['coverage']:.3f}  F1 {overall['f1']:.3f}  items {done}/{needed}"


def backup(backup_dir: Optional[Path]) -> None:
    """Copy the local caches to `backup_dir` (if given). A failed backup is reported, never fatal."""
    if backup_dir is None:
        return
    try:
        saved = state.save(Path(backup_dir), final_config=FINAL_CONFIG)
        print(f"Backed up to {Path(backup_dir).as_posix()}: {', '.join(saved) or 'nothing yet'}")
    except Exception as err:
        print(f"WARNING: backup to {Path(backup_dir).as_posix()} failed: {type(err).__name__}: {err}")


def run_jobs(jobs: list[Job], dry_run: bool = False, backup_dir: Optional[Path] = None) -> int:
    """Run the jobs in order; returns 0 when all are complete, 1 at the first failure."""
    for job in jobs:
        needed, done = items_needed(job), items_done(job)
        command = "python -m medverify " + " ".join(job.argv)
        if dry_run:
            left = "?" if needed is None else max(needed - done, 0)
            print(f"{job.name}: {left} items to go ({done} done)\n  {command}")
            continue
        if needed is not None and done >= needed:
            print(f"skip {job.name}: complete ({done}/{needed})")
            continue
        print(f"\n== {job.name}" + (f" (resuming after {done} items)" if done else "") + f"\n  {command}")
        try:
            cli.main(job.argv)
        except SystemExit as err:
            if err.code not in (None, 0):
                print(f"\nRun {job.name} failed: {err.code}")
                backup(backup_dir)
                return 1
        except KeyboardInterrupt:
            print(f"\nRun {job.name} was stopped; finished items are kept and it resumes next time.")
            backup(backup_dir)
            raise
        except Exception:
            traceback.print_exc()
            print(f"\nRun {job.name} failed (see the error above). Fix it and run the stage again; finished runs are kept.")
            backup(backup_dir)
            return 1
        backup(backup_dir)
        print(summary_line(job))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.run", description="Run the Phase 2 plan.")
    p.add_argument("--stage", required=True, choices=["smoke", "val", "test"])
    p.add_argument("--only", help="experiments to run, e.g. E2,E3")
    p.add_argument("--names", help="run names to run, e.g. b0-test-s1,b1-test-s1")
    p.add_argument("--backup-dir", type=Path, help="copy the local caches here after every run (Drive or Kaggle output)")
    p.add_argument("--dry-run", action="store_true", help="print each command and the items it still needs")
    args = p.parse_args(argv)
    jobs = select(stage_jobs(args.stage), args.only, args.names)
    if not jobs:
        raise SystemExit("No run matches --only / --names for this stage.")
    print(f"{args.stage} stage: {len(jobs)} runs")
    return run_jobs(jobs, args.dry_run, args.backup_dir)


if __name__ == "__main__":
    raise SystemExit(main())
