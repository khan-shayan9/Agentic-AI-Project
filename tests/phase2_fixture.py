"""Builds a tiny but complete Phase 2 workspace offline: every plan run made by the real runner with
fake models and a fake PubMed, then exported to results/phase2 (shared by the report tests)."""

import json
import zlib

import medverify.data as D
import medverify.llm as llm
import medverify.tools.pubmed as pm
from experiments import choose, export
from experiments import run as runner
from medverify import cli
from medverify.llm import FakeModel
from medverify.tools.pubmed import PubMedSearch

from conftest import make_rows
from test_pubmed import FakeSession

SKIPPED = ("debate-test-s1",)  # left out on purpose: the report must list it as missing


def _answer(messages) -> str:
    first = next(m["content"] for m in messages if "Answer:" in m["content"])
    return first.split("Answer:")[-1]


def fake_reply(name: str):
    """Deterministic verifier: right from the answer text, wrong on a model-specific subset of items.
    Samples of one model (llama@k) share the same mistakes; the three families mostly do not."""
    base = name.split("@")[0]

    def reply(messages):
        answer = _answer(messages)
        truth = 1 if "increased symptom" in answer else 0
        h = zlib.crc32(answer.encode())
        wrong = {"llama": h % 4 == 0, "mistral-local": h % 5 == 1, "phi": h % 3 == 2}.get(base, False)
        unsure = base == "mistral-local" and h % 7 == 3
        if "Other independent reviewers" in messages[-1]["content"]:  # debate round 2: stick with round 1
            wrong = wrong and h % 2 == 0
        code = 2 if unsure else (1 - truth if wrong else truth)
        if "single digit" in messages[0]["content"] or "Answer:" in messages[0]["content"] and messages[0]["role"] == "user":
            return str(code)  # the baseline prompt and the digits style answer with a digit
        return "Checked.\nVerdict: " + {0: "FACTUAL", 1: "HALLUCINATED", 2: "UNSURE"}[code]

    return reply


def patch(mp, root):
    mp.chdir(root)
    mp.setattr(cli, "DATA_DIR", root / "data/processed")
    mp.setattr(cli, "RUNS_DIR", root / "results/runs")
    mp.setattr(runner, "FINAL_CONFIG", root / "experiments/final_config.json")
    models = {}
    mp.setattr(llm, "make_model", lambda name, **kw: models.setdefault(name, FakeModel(f"fake-{name}", fake_reply(name))))
    records = [(p, f"Study {p}", f"An unrelated abstract number {p} about other medicines.") for p in ("2001", "2002", "2003", "2004")]

    def fake_pubmed():
        tool = PubMedSearch(cache_dir=root / "pubmed_cache", session=FakeSession(["2001", "2002", "2003", "2004"], records))
        tool.min_interval = 0
        return tool

    mp.setattr(pm, "PubMedSearch", fake_pubmed)
    return models


def build(root, mp) -> None:
    """Make val and test runs with the real runner, choose the trigger, export to results/phase2."""
    patch(mp, root)
    rows = make_rows(per_level=4)  # 12 rows -> 24 items per split
    for split in ("dev", "val", "test"):
        D.write_jsonl(root / f"data/processed/{split}.jsonl", rows)
    (root / "experiments").mkdir()
    assert runner.run_jobs(runner.stage_jobs("val")) == 0
    assert choose.main([]) == 0
    jobs = [j for j in runner.stage_jobs("test") if j.name not in SKIPPED]
    assert runner.run_jobs(jobs) == 0
    export.export("all")


def final_config(root) -> dict:
    return json.loads((root / "results/phase2/final_config.json").read_text())
