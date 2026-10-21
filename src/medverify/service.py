"""MedVerify for questions and answers typed by a user: shared by `medverify verify`, `medverify batch`
and the Streamlit app, so all three run the same verify_item (through verify_user_item) with the
same caching, re-ask and logging as the experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from . import llm as _llm
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

