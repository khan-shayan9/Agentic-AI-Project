"""MedVerify for questions and answers typed by a user: shared by `medverify verify`, `medverify batch`
and the Streamlit app, so all three run the same verify_item (through verify_user_item) with the
same caching, re-ask and logging as the experiments.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import llm as _llm
from .data import append_jsonl, read_jsonl
from .explain import agreement, decision_label, second_round_text
from .llm import DEFAULT_VERIFIERS
from .pipeline import verify_user_item
from .schemas import VerificationState, user_item
from .tools import pubmed as _pubmed

# The trigger chosen on the validation split (python -m experiments.choose), if it exists.
FINAL_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "final_config.json"


def resolve_second_round(choice: str = "auto", final_config: Optional[Path] = None) -> tuple[str, str]:
    """(rule, why): "auto" = the trigger in experiments/final_config.json if it exists, otherwise "off"."""
    if choice != "auto":
        return choice, "set by you"
    path = Path(final_config or FINAL_CONFIG)
    if path.exists():
        trigger = json.loads(path.read_text(encoding="utf-8"))["trigger"]
        return trigger, "auto: the trigger chosen on the validation split (experiments/final_config.json)"
    return "off", "auto: off, because experiments/final_config.json does not exist yet"


def make_verifiers(names: Iterable[str] = DEFAULT_VERIFIERS, temperature: float = 0.0) -> dict:
    """The verifier models, as in the experiments (temperature 0, cached replies)."""
    return {n: _llm.make_model(n, temperature=temperature) for n in names}


def pubmed_tool(evidence_mode: str, has_pmid: bool):
    """A PubMed tool when one is needed: PubMed evidence, or fetching a user-given source abstract."""
    return _pubmed.PubMedSearch() if evidence_mode == "pubmed" or has_pmid else None


def verify_one(question: str, answer: str, pmid=None, verifiers: Optional[dict] = None, evidence_mode: str = "pubmed",
               second_round: str = "off", min_agree: int = 2, pubmed=None) -> VerificationState:
    """Verify one question and answer typed by a user."""
    item = user_item(question, answer, pmid)
    verifiers = verifiers if verifiers is not None else make_verifiers()
    if pubmed is None:
        pubmed = pubmed_tool(evidence_mode, item.pubid is not None)
    return verify_user_item(item, verifiers, evidence_mode, pubmed, min_agree=min_agree, second_round_rule=second_round)

# ------------------------------------------------------------- CSV batch

def parse_rows(text: str, name: str = "the CSV") -> list[dict]:
    """Rows of CSV text with question, answer and optionally pmid (other columns are kept)."""
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    missing = [c for c in ("question", "answer") if rows and c not in rows[0]]
    if missing or not rows:
        raise ValueError(f"{name} needs the columns question and answer (and optionally pmid); "
                         + (f"missing: {', '.join(missing)}" if missing else "it has no rows"))
    return rows


def read_rows(path: Path) -> list[dict]:
    return parse_rows(Path(path).read_text(encoding="utf-8-sig"), str(path))


def batch_row(row: dict, state: VerificationState, verifier_names: list[str]) -> dict:
    """One output row: the input columns plus the decision and what led to it."""
    votes = {v.system.split(":", 1)[-1]: v.verdict for v in state.last_round()}
    out = dict(row)
    out |= {
        "decision": decision_label(state.final),
        "agreement": agreement(state),
        **{f"vote_{n}": {0: "factual", 1: "hallucinated", 2: "unsure"}.get(votes.get(n), "unreadable") for n in verifier_names},
        "evidence_pmids": " ".join(e.pmid for e in state.evidence),
        "blocked_pmids": " ".join(b.pmid for b in state.blocked),
        "second_round": second_round_text(state),
        "flags": " ".join(state.flags),
        "seconds": round(sum(v.latency_s for v in state.verdicts), 1),
        "item_id": state.item_id,
    }
    return out


def states_path_for(output: Path) -> Path:
    """OUT.csv -> OUT.states.jsonl"""
    output = Path(output)
    return output.with_name(output.stem + ".states.jsonl")


def run_batch(rows: list[dict], verifiers: dict, evidence_mode: str = "pubmed", second_round: str = "off",
              min_agree: int = 2, states_path: Optional[Path] = None, pubmed=None,
              progress: Optional[Callable[[int, int], None]] = None) -> list[dict]:
    """Verify every row; full states are appended to `states_path`, and rows already there are skipped."""
    if evidence_mode == "source":
        lacking = [i + 2 for i, r in enumerate(rows) if not (r.get("pmid") or "").strip()]
        if lacking:
            raise ValueError(f"Evidence mode 'source' needs a pmid on every row; missing on CSV line(s) {lacking}.")
    done: dict[str, VerificationState] = {}
    if states_path and Path(states_path).exists():
        done = {s.item_id: s for s in read_jsonl(states_path, VerificationState)}
    if pubmed is None and (evidence_mode == "pubmed" or any((r.get("pmid") or "").strip() for r in rows)):
        pubmed = _pubmed.PubMedSearch()
    out = []
    for n, row in enumerate(rows, 1):
        item = user_item(row["question"], row["answer"], row.get("pmid"))
        if item.item_id not in done:
            state = verify_user_item(item, verifiers, evidence_mode, pubmed, min_agree=min_agree, second_round_rule=second_round)
            done[item.item_id] = state
            if states_path:
                append_jsonl(states_path, state)
        out.append(batch_row(row, done[item.item_id], list(verifiers)))
        if progress:
            progress(n, len(rows))
    return out


def write_rows(rows: list[dict], path: Path) -> None:
    columns = list(dict.fromkeys(c for r in rows for c in r))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
