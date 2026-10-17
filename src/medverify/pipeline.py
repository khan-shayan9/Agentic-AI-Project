"""Proposed system, first working version.

For each item:
  1. Evidence agent (optional): searches PubMed for the question, with the source
     abstract blocked (setting A), or hands over the source abstract (setting B).
  2. Verifier agents: each model judges the answer on its own, without seeing the
     others' verdicts (independence by design, from different model families).
  3. Decision step: vote with abstention. A label is accepted only when at least
     `min_agree` verifiers choose it; otherwise the system abstains (2).

Every step is logged in a VerificationState (the agent trajectory), so the vote rule
can be re-tuned later on the validation split without new model calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from tqdm import tqdm

from .data import append_jsonl, read_jsonl
from .llm import ChatModel
from .metrics import aggregate_votes
from .parsing import parse_verdict
from .prompts import format_evidence, verifier_messages
from .schemas import Evidence, Item, Verdict, VerificationState
from .tools.pubmed import PubMedSearch, keyword_queries

EvidenceMode = Literal["none", "pubmed", "source"]


class VerifierCallError(RuntimeError):
    """A verifier's model call still failed after all retries (for example a 429 rate limit)."""

    def __init__(self, name: str, model: ChatModel, err: Exception):
        self.name, self.provider, self.model, self.err = name, model.provider, model.model, err
        super().__init__(f"verifier '{name}' ({model.provider}:{model.model}) failed: {type(err).__name__}: {err}")


def gather_evidence(item: Item, mode: EvidenceMode, pubmed: Optional[PubMedSearch], k: int, state: VerificationState) -> list[Evidence]:
    if mode == "none":
        state.trace.append("evidence: none")
        return []
    if mode == "source":
        state.trace.append("evidence: source abstract given (setting B)")
        text = " ".join(item.knowledge)
        return [Evidence(pmid=str(item.pubid or "source"), title="Source abstract", text=text, query="", rank=1)]
    if pubmed is None:
        raise ValueError("evidence mode 'pubmed' needs a PubMedSearch tool")
    queries = keyword_queries(item.question)
    kept, blocked = pubmed.retrieve(
        queries,
        k=k,
        block_pmids=[item.pubid] if item.pubid else [],
        block_texts=item.knowledge,
    )
    state.query = kept[0].query if kept else queries[0]
    state.blocked = blocked
    state.trace.append(f"evidence: pubmed query='{state.query}' kept={len(kept)} blocked={len(blocked)}")
    return kept


def verify_item(
    item: Item,
    verifiers: dict[str, ChatModel],
    evidence_mode: EvidenceMode,
    pubmed: Optional[PubMedSearch],
    k: int,
    min_agree: int,
) -> VerificationState:
    state = VerificationState(item_id=item.item_id, evidence_mode=evidence_mode)
    state.evidence = gather_evidence(item, evidence_mode, pubmed, k, state)
    max_chars = None if evidence_mode == "source" else 1000
    evidence_text = format_evidence(state.evidence, max_chars=max_chars) if state.evidence else None

    for name, model in verifiers.items():
        try:
            res = model.chat(verifier_messages(item.question, item.answer, evidence_text))
        except Exception as err:  # re-raised with the verifier name
            raise VerifierCallError(name, model, err) from err
        code = parse_verdict(res.text, allow_unsure=True)
        state.verdicts.append(
            Verdict(
                item_id=item.item_id,
                system=f"verifier:{name}",
                verdict=code,
                raw_text=res.text,
                provider=res.provider,
                model=res.model,
                latency_s=res.latency_s,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                cached=res.cached,
            )
        )
        state.trace.append(f"verifier {name}: {code}")

    state.final = aggregate_votes([v.verdict for v in state.verdicts], min_agree=min_agree)
    state.trace.append(f"decision (min_agree={min_agree}): {state.final}")
    return state


def run_pipeline(
    items: list[Item],
    verifiers: dict[str, ChatModel],
    evidence_mode: EvidenceMode = "none",
    pubmed: Optional[PubMedSearch] = None,
    k: int = 3,
    min_agree: int = 2,
    out_path: Path | None = None,
) -> list[VerificationState]:
    done: dict[str, VerificationState] = {}
    if out_path and Path(out_path).exists():
        done = {s.item_id: s for s in read_jsonl(out_path, VerificationState)}

    states = []
    for item in tqdm(items, desc=f"pipeline ({evidence_mode}, {len(verifiers)} verifiers)"):
        if item.item_id in done:
            states.append(done[item.item_id])
            continue
        state = verify_item(item, verifiers, evidence_mode, pubmed, k, min_agree)
        states.append(state)
        if out_path:
            append_jsonl(out_path, state)
    return states


# ---------------------------------------------------- views over states

def final_verdicts(states: list[VerificationState], min_agree: Optional[int] = None) -> list[Verdict]:
    """System-level verdicts; pass `min_agree` to re-apply a different vote rule offline."""
    out = []
    for s in states:
        code = s.final if min_agree is None else aggregate_votes([v.verdict for v in s.verdicts], min_agree)
        out.append(
            Verdict(
                item_id=s.item_id,
                system="pipeline",
                verdict=code,
                raw_text="",
                provider="pipeline",
                model="+".join(v.model for v in s.verdicts),
                latency_s=sum(v.latency_s for v in s.verdicts),
                prompt_tokens=sum(v.prompt_tokens for v in s.verdicts),
                completion_tokens=sum(v.completion_tokens for v in s.verdicts),
                cached=all(v.cached for v in s.verdicts),
            )
        )
    return out


def verifier_verdicts(states: list[VerificationState]) -> dict[str, list[Verdict]]:
    """Each verifier's own verdicts, for single-verifier scores and error overlap."""
    by_name: dict[str, list[Verdict]] = {}
    for s in states:
        for v in s.verdicts:
            by_name.setdefault(v.system.split(":", 1)[-1], []).append(v)
    return by_name


def agreement_confidence(states: list[VerificationState]) -> dict[str, tuple[int, float]]:
    """(majority label, share of verifiers backing it) per item, for risk-coverage curves."""
    out = {}
    for s in states:
        codes = [v.verdict for v in s.verdicts]
        zeros, ones = codes.count(0), codes.count(1)
        label = 1 if ones > zeros else 0
        out[s.item_id] = (label, max(zeros, ones) / max(len(codes), 1))
    return out
