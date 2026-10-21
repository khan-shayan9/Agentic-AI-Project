"""Read exported runs from results/phase2 (never from results/runs) and turn them into "sides":
one verdict per item plus its cost, for a whole run or a view of it (round 1, forced answer,
another vote threshold, one verifier alone, one verifier dropped). Every table is built from sides.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from medverify.metrics import aggregate_votes, efficiency_summary, evaluate
from medverify.pipeline import final_verdicts, verifier_verdicts
from medverify.schemas import Item, Verdict, VerificationState

PHASE2 = Path("results/phase2")

METRICS = [
    "n", "coverage", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high", "selective_accuracy",
    "easy_f1", "medium_f1", "hard_f1", "abstain_rate", "unreadable_rate",
    "model_calls", "prompt_tokens", "completion_tokens", "model_seconds", "pubmed_requests",
    "round2_share", "reasks", "max_prompt_tokens",
]


@dataclass
class RunData:
    name: str
    config: dict
    states: Optional[list[VerificationState]] = None
    predictions: Optional[list[Verdict]] = None

    @property
    def seed(self) -> Optional[int]:
        return self.config.get("setup", {}).get("seed", self.config.get("seed"))


@dataclass
class Side:
    """One verdict per item (item_id -> code) and the cost per item, for a run or a view of it."""

    label: str
    run: str
    seed: Optional[int]
    codes: dict[str, Optional[int]]
    cost: dict = field(default_factory=dict)


# ------------------------------------------------------------- loading

def load_items(split: str, phase2: Path = PHASE2) -> list[Item]:
    path = Path(phase2) / f"items_{split}.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [
            Item(item_id=r["item_id"], row_id=r["item_id"].rsplit("-", 1)[0], question=r.get("question", ""),
                 answer=r.get("answer", ""), label=int(r["label"]), difficulty=r["difficulty"],
                 category=r["category"], pubid=int(r["pubid"]) if r.get("pubid") else None)
            for r in csv.DictReader(f)
        ]


def _read_gz(path: Path, model) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [model.model_validate(json.loads(line)) for line in f if line.strip()]


def load_run(name: str, phase2: Path = PHASE2) -> Optional[RunData]:
    folder = Path(phase2) / "runs" / name
    if not (folder / "config.json").exists():
        return None
    run = RunData(name, json.loads((folder / "config.json").read_text(encoding="utf-8")))
    if (folder / "states.jsonl.gz").exists():
        run.states = _read_gz(folder / "states.jsonl.gz", VerificationState)
    elif (folder / "predictions.jsonl.gz").exists():
        run.predictions = _read_gz(folder / "predictions.jsonl.gz", Verdict)
    else:
        return None
    return run


def load_final_config(phase2: Path = PHASE2) -> Optional[dict]:
    path = Path(phase2) / "final_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ------------------------------------------------------------- cost

def verdict_cost(per_item: list[list[Verdict]]) -> dict:
    """Cost per item of the verdicts each item needed (re-asks count as an extra call)."""
    n = max(len(per_item), 1)
    flat = [v for vs in per_item for v in vs]
    return {
        "model_calls": sum(1 + v.reasked for v in flat) / n,
        "prompt_tokens": sum(v.prompt_tokens for v in flat) / n,
        "completion_tokens": sum(v.completion_tokens for v in flat) / n,
        "model_seconds": sum(v.latency_s for v in flat) / n,
        "reasks": sum(v.reasked for v in flat),
        "max_prompt_tokens": max((v.prompt_tokens for v in flat), default=0),
    }


def _pubmed(states: list[VerificationState], round_no: Optional[int] = None) -> float:
    calls = [c for s in states for c in s.tool_calls if round_no is None or c.round == round_no]
    return len(calls) / max(len(states), 1)


# ------------------------------------------------------------- sides

def _codes(verdicts: list[Verdict]) -> dict[str, Optional[int]]:
    return {v.item_id: v.verdict for v in verdicts}


def side_final(run: RunData, label: Optional[str] = None) -> Side:
    """The run's own decision (pipeline) or prediction (baseline), with its full cost."""
    if run.predictions is not None:
        cost = verdict_cost([[v] for v in run.predictions]) | {"pubmed_requests": 0.0, "round2_share": 0.0}
        return Side(label or run.name, run.name, run.seed, _codes(run.predictions), cost)
    eff = efficiency_summary(run.states)
    cost = verdict_cost([s.verdicts for s in run.states]) | {
        "pubmed_requests": eff["pubmed_requests_per_item"], "round2_share": eff["round2_share"]}
    return Side(label or run.name, run.name, run.seed, _codes(final_verdicts(run.states)), cost)


def _round1_cost(run: RunData, keep=lambda name: True) -> dict:
    per_item = [[v for v in s.first_round() if keep(v.system.split(":", 1)[-1])] for s in run.states]
    return verdict_cost(per_item) | {"pubmed_requests": _pubmed(run.states, 1), "round2_share": 0.0}


def side_round1(run: RunData) -> Side:
    """The decision from round-1 votes alone (second round off), from the same run."""
    return Side(f"{run.name} round 1", run.name, run.seed, _codes(final_verdicts(run.states, round=1)), _round1_cost(run))


def side_forced(run: RunData) -> Side:
    side = side_final(run)
    return Side(f"{run.name} forced", run.name, run.seed, _codes(final_verdicts(run.states, forced=True)), side.cost)


def side_min_agree(run: RunData, k: int) -> Side:
    side = side_final(run)
    return Side(f"{run.name} {k} of 3", run.name, run.seed, _codes(final_verdicts(run.states, min_agree=k)), side.cost)


def verifier_names(run: RunData) -> list[str]:
    return list(verifier_verdicts(run.states)) if run.states else []


def side_single(run: RunData, name: str) -> Side:
    """One verifier alone (its round-1 verdicts): B3."""
    verdicts = verifier_verdicts(run.states).get(name, [])
    return Side(f"{name} alone", run.name, run.seed, _codes(verdicts), _round1_cost(run, lambda n: n == name))


def side_drop(run: RunData, dropped: str) -> Side:
    """Two-verifier panel from round 1 without `dropped`; both remaining verifiers must agree."""
    codes = {}
    for s in run.states:
        votes = [v.verdict for v in s.first_round() if v.system.split(":", 1)[-1] != dropped]
        codes[s.item_id] = aggregate_votes(votes, min_agree=len(votes))
    return Side(f"without {dropped}", run.name, run.seed, codes, _round1_cost(run, lambda n: n != dropped))


# ------------------------------------------------------------- metrics

def _verdicts(side: Side) -> list[Verdict]:
    return [Verdict(item_id=i, system=side.label, verdict=c, raw_text="", provider="report", model="report",
                    latency_s=0.0) for i, c in side.codes.items()]


def side_metrics(items: list[Item], side: Side, n_boot: int = 1000) -> dict:
    """The METRICS fields for one side (the F1 interval uses the same bootstrap as metrics.json)."""
    result = evaluate(items, _verdicts(side), n_boot=n_boot, seed=0)
    o = result["overall"]
    ci = o.get("f1_ci95") or [None, None]
    row = {
        "n": o["n"], "coverage": o["coverage"], "precision": o["precision"], "recall": o["recall"], "f1": o["f1"],
        "f1_ci_low": ci[0], "f1_ci_high": ci[1], "selective_accuracy": o["selective_accuracy"],
        "abstain_rate": o["abstain_rate"], "unreadable_rate": o["invalid_rate"],
    }
    for level in ("easy", "medium", "hard"):
        row[f"{level}_f1"] = result[level]["f1"] if level in result else None
    for key in ("model_calls", "prompt_tokens", "completion_tokens", "model_seconds", "pubmed_requests",
                "round2_share", "reasks", "max_prompt_tokens"):
        row[key] = side.cost.get(key)
    return row
