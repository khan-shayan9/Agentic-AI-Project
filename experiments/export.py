"""Copy finished plan runs into results/phase2/, the folder that is committed to the repo.

  python -m experiments.export [--split test|val|all]

For every plan run whose folder has a result for every item: config.json, every metrics*.json,
risk_coverage.json (if present) and the raw results gzipped (states.jsonl.gz or
predictions.jsonl.gz). In states each evidence abstract is cut to 300 characters (pmid, title,
query, rank, leak and round are kept); verdicts, raw replies, steps, tool calls and flags are kept
exactly. Also writes items_<split>.csv and copies experiments/final_config.json, so
`python -m experiments.report` can rebuild every table from results/phase2 alone.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

from medverify import cli

from . import run as runner
from .plan import TEST_RUNS, VAL_RUNS, Run

PHASE2 = Path("results/phase2")
EVIDENCE_CHARS = 300
SIZE_WARN_MB = 50
ITEM_COLUMNS = ["item_id", "label", "difficulty", "category", "pubid", "question", "answer"]


def stage_runs(split: str) -> list[Run]:
    return {"test": TEST_RUNS, "val": VAL_RUNS, "all": VAL_RUNS + TEST_RUNS}[split]


def run_split(run: Run) -> str:
    return "val" if run.stage == "val" else "test"


def trim_state(state: dict) -> dict:
    """A saved state with each evidence abstract cut to 300 characters; everything else unchanged."""
    for evidence in state.get("evidence", []):
        evidence["text"] = evidence.get("text", "")[:EVIDENCE_CHARS]
    return state


def write_jsonl_gz(lines: list[dict], target: Path) -> None:
    """Gzipped JSON lines with a fixed timestamp, so re-exporting the same run gives the same bytes."""
    with target.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as z:
        for line in lines:
            z.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export_run(name: str, phase2: Path = PHASE2) -> list[Path]:
    """Copy one run folder into phase2/runs/<name>/; returns the files written."""
    source, target = cli.RUNS_DIR / name, phase2 / "runs" / name
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(source.glob("metrics*.json")) + [source / "config.json", source / "risk_coverage.json"]:
        if path.exists():
            shutil.copyfile(path, target / path.name)
            written.append(target / path.name)
    if (source / "states.jsonl").exists():
        write_jsonl_gz([trim_state(s) for s in _read_jsonl(source / "states.jsonl")], target / "states.jsonl.gz")
        written.append(target / "states.jsonl.gz")
    elif (source / "predictions.jsonl").exists():
        write_jsonl_gz(_read_jsonl(source / "predictions.jsonl"), target / "predictions.jsonl.gz")
        written.append(target / "predictions.jsonl.gz")
    for path in written:
        size_mb = path.stat().st_size / 1e6
        if size_mb > SIZE_WARN_MB:
            print(f"WARNING: {path.as_posix()} is {size_mb:.0f} MB (GitHub warns above 50 MB)")
    return written


def export_items(split: str, phase2: Path = PHASE2) -> Path:
    """items_<split>.csv: what the report needs about each item, so it runs without MedHallu."""
    target = phase2 / f"items_{split}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ITEM_COLUMNS)
        for item in cli._items(split, None):
            writer.writerow([item.item_id, item.label, item.difficulty, item.category,
                             item.pubid if item.pubid is not None else "", item.question, item.answer])
    return target


def export(split: str = "all", phase2: Path = PHASE2) -> dict[str, list[str]]:
    """Export every finished plan run of the split(s); returns {"exported": [...], "partial": [...]}."""
    done: dict[str, list[str]] = {"exported": [], "partial": []}
    splits = set()
    for run in stage_runs(split):
        job = runner.Job(run, run.name, run_split(run), None, [])
        needed, have = runner.items_needed(job), runner.items_done(job)
        if have == 0:
            continue
        if needed is None or have < needed:
            done["partial"].append(f"{run.name} ({have}/{needed})")
            continue
        export_run(run.name, phase2)
        done["exported"].append(run.name)
        splits.add(run_split(run))
    for s in sorted(splits):
        export_items(s, phase2)
    for extra, name in ((runner.FINAL_CONFIG, "final_config.json"), (cli.RUNS_DIR / "ollama_versions.txt", "ollama_versions.txt")):
        if Path(extra).exists():
            phase2.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(extra, phase2 / name)
    return done


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.export", description="Copy finished runs into results/phase2.")
    p.add_argument("--split", default="all", choices=["test", "val", "all"])
    args = p.parse_args(argv)
    done = export(args.split)
    print(f"Exported {len(done['exported'])} runs to {PHASE2.as_posix()}: {', '.join(done['exported']) or 'none'}")
    if done["partial"]:
        print(f"Skipped unfinished runs: {', '.join(done['partial'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
