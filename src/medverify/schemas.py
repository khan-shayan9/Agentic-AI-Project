"""Structured contracts shared by every part of the system.

Each record written to disk (rows, items, verdicts, pipeline states) is one of
these models serialised as one JSON line, so runs are easy to inspect and reload.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard"]

# Verdict codes follow the MedHallu paper: 0 = factual, 1 = hallucinated,
# 2 = not sure (abstain). None means the model's reply could not be parsed.
VerdictCode = Optional[Literal[0, 1, 2]]


class Row(BaseModel):
    """One MedHallu row: a question with its true and hallucinated answers."""

    row_id: str
    subset: str
    question: str
    knowledge: list[str] = Field(default_factory=list)  # the source abstract (PubMedQA contexts)
    ground_truth: str
    hallucinated_answer: str
    difficulty: Difficulty
    category: str
    pubid: Optional[int] = None  # PubMed ID of the source abstract, used to block leakage


class Item(BaseModel):
    """One thing to verify: a question plus ONE answer (true or hallucinated)."""

    item_id: str
    row_id: str
    question: str
    answer: str
    label: Literal[0, 1]  # 1 = hallucinated
    difficulty: Difficulty
    category: str
    knowledge: list[str] = Field(default_factory=list)
    pubid: Optional[int] = None


class LLMResult(BaseModel):
    """Raw output of one model call, with cost and speed measurements."""

    text: str
    provider: str
    model: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


class Evidence(BaseModel):
    """One retrieved PubMed abstract."""

    pmid: str
    title: str
    text: str
    query: str
    rank: int


class BlockedEvidence(BaseModel):
    """A retrieved abstract that was removed to prevent source leakage."""

    pmid: str
    reason: Literal["source_pmid", "near_duplicate"]
    overlap: float = 0.0


class Verdict(BaseModel):
    """One verifier's judgement on one item."""

    item_id: str
    system: str  # e.g. "baseline" or "verifier:mistral"
    verdict: VerdictCode
    raw_text: str
    provider: str
    model: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


class VerificationState(BaseModel):
    """Full state of the multi-agent pipeline for one item (the agent trajectory)."""

    item_id: str
    evidence_mode: str
    query: Optional[str] = None
    evidence: list[Evidence] = Field(default_factory=list)
    blocked: list[BlockedEvidence] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    final: VerdictCode = None
    trace: list[str] = Field(default_factory=list)
