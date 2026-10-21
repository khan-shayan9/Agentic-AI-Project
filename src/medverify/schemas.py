"""Structured contracts shared by every part of the system.

Each record written to disk (rows, items, verdicts, pipeline states) is one of
these models serialised as one JSON line, so runs are easy to inspect and reload.
"""

from __future__ import annotations

import hashlib
import time
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr

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
    """One thing to verify: a question plus ONE answer (true or hallucinated).

    Items typed by a user have no gold label and no MedHallu difficulty (label None,
    difficulty "unknown"); they can be verified but not scored.
    """

    item_id: str
    row_id: str
    question: str
    answer: str
    label: Optional[Literal[0, 1]] = None  # 1 = hallucinated; None = unknown (user item)
    difficulty: Union[Difficulty, Literal["unknown"]] = "unknown"
    category: str = ""
    knowledge: list[str] = Field(default_factory=list)
    pubid: Optional[int] = None


def user_item(question: str, answer: str, pmid: Optional[Union[int, str]] = None) -> Item:
    """An unlabelled item typed by a user. The id is stable: "user-" + a short sha256 of question and answer.

    `pmid` is the answer's known source abstract, if the user has one; it is blocked from the evidence.
    """
    question, answer = question.strip(), answer.strip()
    digest = hashlib.sha256(f"{question}\n{answer}".encode("utf-8")).hexdigest()[:12]
    item_id = f"user-{digest}"
    pubid = int(str(pmid).strip()) if pmid not in (None, "") and str(pmid).strip() else None
    return Item(item_id=item_id, row_id=item_id, question=question, answer=answer, category="user", pubid=pubid)


class LLMResult(BaseModel):
    """Raw output of one model call, with cost and speed measurements."""

    text: str
    provider: str
    model: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


LeakReason = Literal["source_pmid", "near_duplicate"]


class Evidence(BaseModel):
    """One retrieved PubMed abstract."""

    pmid: str
    title: str
    text: str
    query: str
    rank: int
    leak: Optional[LeakReason] = None  # set only when blocking is off (ablation): this abstract leaks the source
    round: int = 1  # 2 = found by the second evidence round (claim query)


class BlockedEvidence(BaseModel):
    """A retrieved abstract that was removed to prevent source leakage."""

    pmid: str
    reason: LeakReason
    overlap: float = 0.0
    round: int = 1


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
    round: int = 1  # 1 = independent judgement; 2 = after seeing the others (debate)
    reasked: bool = False  # the first reply was unreadable, so the model was asked once more


class TraceStep(BaseModel):
    """One step of the agent trajectory: who acted, what they did, and what came out."""

    step: int
    agent: Literal["evidence", "verifier", "decision", "orchestrator"]
    action: str
    detail: dict = Field(default_factory=dict)
    t: float = 0.0  # seconds since the item started


class ToolCall(BaseModel):
    """One call to an external tool (for now PubMed search and fetch)."""

    tool: str
    input: str
    n_results: int
    cached: bool
    latency_s: float
    round: int = 1  # evidence round that made the call


class VerificationState(BaseModel):
    """Full state of the multi-agent pipeline for one item (the agent trajectory).

    Every field added after Phase 1 has a default, so Phase 1 states still load.
    """

    item_id: str
    evidence_mode: str
    query: Optional[str] = None
    evidence: list[Evidence] = Field(default_factory=list)
    blocked: list[BlockedEvidence] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    final: VerdictCode = None
    trace: list[str] = Field(default_factory=list)  # short text log, kept for older tools
    steps: list[TraceStep] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    rounds: int = 1
    final_round1: VerdictCode = None  # the decision from round-1 votes alone
    flags: list[str] = Field(default_factory=list)  # e.g. "human_review", "reasked:phi"
    _started: float = PrivateAttr(default_factory=time.monotonic)

    def add_step(self, agent: str, action: str, **detail) -> TraceStep:
        """Append one step to the trajectory, timed from when this state was created."""
        step = TraceStep(step=len(self.steps) + 1, agent=agent, action=action, detail=detail,
                         t=round(time.monotonic() - self._started, 3))
        self.steps.append(step)
        return step

    def round_verdicts(self, round_no: int) -> list[Verdict]:
        """Verdicts given in one round, in verifier order."""
        return [v for v in self.verdicts if v.round == round_no]

    def first_round(self) -> list[Verdict]:
        """Round-1 verdicts: each verifier's independent judgement."""
        return self.round_verdicts(1)

    def last_round(self) -> list[Verdict]:
        """Verdicts of the last round, the ones the decision uses."""
        return self.round_verdicts(max((v.round for v in self.verdicts), default=1))
