"""Proposed system.

For each item:
  1. Evidence agent (optional): searches PubMed for the question, with the source
     abstract blocked (setting A), or hands over the source abstract (setting B).
  2. Verifier agents: each model judges the answer on its own, without seeing the
     others' verdicts (independence by design, from different model families).
  3. Decision step: vote with abstention. A label is accepted only when at least
     `min_agree` verifiers choose it; otherwise the system abstains (2).

Second evidence round (`second_round="split"` or `"no-majority"`, PubMed mode only): when
the round-1 votes are split (or give no majority), the evidence agent searches again with
claim queries built from the answer, skipping abstracts already used. With new abstracts,
every verifier judges again, still independently, with the old + new evidence; the decision
then uses these round-2 verdicts. Without new abstracts, the round-1 decision stands.

Debate ablation (`debate=True`): after round 1 each verifier sees the others' verdicts and
reasoning (anonymised) and judges again; the decision then uses these round-2 verdicts.

Every step is logged in a VerificationState (the agent trajectory: structured steps,
tool calls and verdicts), so vote rules can be re-applied later without new model calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

from tqdm import tqdm

from .data import append_jsonl, read_jsonl
from .llm import ChatModel
from .metrics import aggregate_forced, aggregate_votes
from .parsing import parse_for_style
from .prompts import DEFAULT_PROMPT_STYLE, debate_message, format_evidence, reask_message, verifier_messages
from .schemas import Evidence, Item, LLMResult, Verdict, VerificationState
from .tools.pubmed import PubMedSearch, claim_queries, keyword_queries

EvidenceMode = Literal["none", "pubmed", "source"]
SecondRound = Literal["off", "split", "no-majority"]
SECOND_ROUND_RULES = ("off", "split", "no-majority")


class VerifierCallError(RuntimeError):
    """A verifier's model call still failed after all retries (for example a 429 rate limit)."""

    def __init__(self, name: str, model: ChatModel, err: Exception):
        self.name, self.provider, self.model, self.err = name, model.provider, model.model, err
        super().__init__(f"verifier '{name}' ({model.provider}:{model.model}) failed: {type(err).__name__}: {err}")


def gather_evidence(
    item: Item,
    mode: EvidenceMode,
    pubmed: Optional[PubMedSearch],
    k: int,
    state: VerificationState,
    block: bool = True,
    retry: bool = True,
) -> list[Evidence]:
    """Round-1 evidence: none, the source abstract (setting B), or a PubMed search on the question."""
    if mode == "none":
        state.trace.append("evidence: none")
        state.add_step("evidence", "none")
        return []
    if mode == "source":
        state.trace.append("evidence: source abstract given (setting B)")
        state.add_step("evidence", "source", pmid=str(item.pubid or "source"))
        text = " ".join(item.knowledge)
        return [Evidence(pmid=str(item.pubid or "source"), title="Source abstract", text=text, query="", rank=1)]
    if pubmed is None:
        raise ValueError("evidence mode 'pubmed' needs a PubMedSearch tool")
    queries = keyword_queries(item.question)
    calls_before = len(getattr(pubmed, "calls", []))
    kept, blocked = pubmed.retrieve(
        queries,
        k=k,
        block_pmids=[item.pubid] if item.pubid else [],
        block_texts=item.knowledge,
        block=block,
        retry=retry,
    )
    state.tool_calls.extend(getattr(pubmed, "calls", [])[calls_before:])
    state.query = kept[0].query if kept else queries[0]
    state.blocked = blocked
    leaked = [{"pmid": e.pmid, "leak": e.leak} for e in kept if e.leak]
    state.trace.append(
        f"evidence: pubmed query='{state.query}' kept={len(kept)} blocked={len(blocked)}"
        + (f" leaked={len(leaked)}" if leaked else "")
    )
    state.add_step(
        "evidence", "pubmed_search", round=1, queries=queries if retry else queries[:1], query=state.query,
        kept=[e.pmid for e in kept], blocked=[{"pmid": b.pmid, "reason": b.reason} for b in blocked],
        leaked=leaked, block=block, retry=retry,
    )
    return kept


# ------------------------------------------------------------ verifier steps

def _ask(name: str, model: ChatModel, messages: list[dict]) -> LLMResult:
    """One model call; a failure is re-raised with the verifier's name."""
    try:
        return model.chat(messages)
    except Exception as err:
        raise VerifierCallError(name, model, err) from err


def _log_reply(state: VerificationState, name: str, round_no: int, action: str,
               messages: list[dict], res: LLMResult, code) -> None:
    """Trace step for one model reply: what went in, the raw reply, the parsed verdict, cost."""
    sent = json.dumps(messages, sort_keys=True)
    detail = dict(
        verifier=name, round=round_no,
        input_messages=len(messages), input_chars=sum(len(m["content"]) for m in messages),
        input_sha256=hashlib.sha256(sent.encode()).hexdigest(),
        raw_reply=res.text, verdict=code, latency_s=res.latency_s,
        prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, cached=res.cached,
    )
    if len(messages) > 2:  # a follow-up turn: keep the new message, it is not stored anywhere else
        detail["last_input"] = messages[-1]["content"]
    state.add_step("verifier", action, **detail)


def judge(
    name: str,
    model: ChatModel,
    item: Item,
    evidence_text: Optional[str],
    style: str,
    round_no: int,
    state: VerificationState,
    reask: bool = True,
    messages: Optional[list[dict]] = None,
) -> Verdict:
    """One verifier's judgement of one item in one round.

    If the reply has no readable verdict and `reask` is on, the same model is asked once
    more in the same conversation for the final line only; both raw replies are kept.
    `messages` replaces the standard verifier prompt (a later round passes its own).
    """
    messages = messages or verifier_messages(item.question, item.answer, evidence_text, style=style)
    res = _ask(name, model, messages)
    code = parse_for_style(res.text, style)
    _log_reply(state, name, round_no, "judge", messages, res, code)
    raw, latency, prompt_tokens, completion_tokens, cached = (
        res.text, res.latency_s, res.prompt_tokens, res.completion_tokens, res.cached
    )
    reasked = code is None and reask
    if reasked:
        follow_up = messages + [{"role": "assistant", "content": res.text}, reask_message(style)]
        res2 = _ask(name, model, follow_up)
        code = parse_for_style(res2.text, style)
        _log_reply(state, name, round_no, "reask", follow_up, res2, code)
        raw = f"{res.text}\n---\n{res2.text}"
        latency += res2.latency_s
        prompt_tokens += res2.prompt_tokens
        completion_tokens += res2.completion_tokens
        cached = cached and res2.cached
        state.flags.append(f"reasked:{name}")
    state.trace.append(f"verifier {name}: {code}" if round_no == 1 else f"verifier {name} (round {round_no}): {code}")
    return Verdict(
        item_id=item.item_id,
        system=f"verifier:{name}",
        verdict=code,
        raw_text=raw,
        provider=res.provider,
        model=res.model,
        latency_s=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached=cached,
        round=round_no,
        reasked=reasked,
    )


def debate_round(
    item: Item,
    verifiers: dict[str, ChatModel],
    evidence_text: Optional[str],
    style: str,
    state: VerificationState,
    reask: bool = True,
) -> None:
    """Round 2: each verifier continues its own round-1 conversation, sees the OTHER verifiers'
    verdicts and reasoning (as Reviewer A, B, ...), and gives its own check and verdict again."""
    first = {v.system.split(":", 1)[-1]: v for v in state.first_round()}
    state.add_step("orchestrator", "debate", round=2, verifiers=list(verifiers))
    for name, model in verifiers.items():
        others = [(v.verdict, v.raw_text) for other, v in first.items() if other != name]
        messages = verifier_messages(item.question, item.answer, evidence_text, style=style) + [
            {"role": "assistant", "content": first[name].raw_text},
            debate_message(others, style),
        ]
        state.verdicts.append(judge(name, model, item, evidence_text, style, 2, state, reask, messages=messages))


# ------------------------------------------------------- second evidence round

def second_round_triggered(state: VerificationState, rule: SecondRound, min_agree: int) -> bool:
    """Whether the round-1 votes call for a second evidence round.

    "split":       the votes are not all the same 0/1 label (any disagreement, UNSURE or
                   unreadable reply), i.e. a unanimous vote would abstain.
    "no-majority": the round-1 decision (same min_agree rule) is 2, abstain.
    """
    codes = [v.verdict for v in state.first_round()]
    if rule == "split":
        return aggregate_votes(codes, min_agree=len(codes)) == 2
    if rule == "no-majority":
        return aggregate_votes(codes, min_agree) == 2
    return False


def gather_claim_evidence(
    item: Item,
    pubmed: PubMedSearch,
    k2: int,
    state: VerificationState,
    block: bool = True,
    retry: bool = True,
) -> list[Evidence]:
    """Round-2 evidence: PubMed search with claim queries from the answer, the same source
    blocking, and no abstract the verifiers have already seen. Returns up to k2 new abstracts."""
    queries = claim_queries(item.question, item.answer)
    if not queries:
        state.trace.append("evidence round 2: the answer adds no new keywords, no claim query")
        state.add_step("evidence", "claim_search", round=2, queries=[], query=None, kept=[], blocked=[], leaked=[])
        return []
    used = [e.pmid for e in state.evidence]
    calls_before = len(getattr(pubmed, "calls", []))
    kept, blocked = pubmed.retrieve(
        queries,
        k=k2,
        block_pmids=[item.pubid] if item.pubid else [],
        block_texts=item.knowledge,
        block=block,
        retry=retry,
        exclude_pmids=used,
    )
    new_calls = getattr(pubmed, "calls", [])[calls_before:]
    for record in [*kept, *blocked, *new_calls]:
        record.round = 2
    state.tool_calls.extend(new_calls)
    state.blocked.extend(blocked)
    query = kept[0].query if kept else queries[0]
    leaked = [{"pmid": e.pmid, "leak": e.leak} for e in kept if e.leak]
    state.trace.append(
        f"evidence round 2: pubmed claim query='{query}' kept={len(kept)} blocked={len(blocked)}"
        + (f" leaked={len(leaked)}" if leaked else "")
    )
    state.add_step(
        "evidence", "claim_search", round=2, queries=queries if retry else queries[:1], query=query,
        kept=[e.pmid for e in kept], blocked=[{"pmid": b.pmid, "reason": b.reason} for b in blocked],
        leaked=leaked, excluded=used,
    )
    return kept


def second_round(
    item: Item,
    verifiers: dict[str, ChatModel],
    evidence_mode: EvidenceMode,
    pubmed: Optional[PubMedSearch],
    rule: SecondRound,
    k2: int,
    min_agree: int,
    style: str,
    state: VerificationState,
    reask: bool = True,
    block: bool = True,
    retry: bool = True,
) -> None:
    """The agentic loop: when round 1 is split, look for new evidence and let every verifier
    judge again, still independently. Does not decide; the caller calls decide() once."""
    triggered = second_round_triggered(state, rule, min_agree)
    state.trace.append(f"second round ({rule}): {'triggered' if triggered else 'not needed'}")
    state.add_step("orchestrator", "second_round_check", round=2, rule=rule, triggered=triggered,
                   round1_votes=[v.verdict for v in state.first_round()])
    if not triggered:
        return
    if evidence_mode != "pubmed" or pubmed is None:
        state.trace.append(f"second round skipped (evidence mode '{evidence_mode}')")
        state.add_step("orchestrator", "second_round_skipped", round=2, reason=f"evidence mode '{evidence_mode}'")
        return
    new = gather_claim_evidence(item, pubmed, k2, state, block, retry)
    if not new:
        state.flags.append("no_new_evidence")  # round-1 votes and decision stand, no extra model calls
        state.trace.append("second round: no new evidence, round-1 votes stand")
        state.add_step("orchestrator", "second_round_no_new_evidence", round=2)
        return
    state.evidence.extend(new)
    evidence_text = format_evidence(state.evidence, max_chars=1000)
    state.add_step("orchestrator", "second_round_judge", round=2, verifiers=list(verifiers),
                   evidence=[e.pmid for e in state.evidence])
    for name, model in verifiers.items():
        state.verdicts.append(judge(name, model, item, evidence_text, style, 2, state, reask))


def decide(state: VerificationState, min_agree: int) -> int:
    """Vote on the last round's verdicts; also save the decision from round 1 alone."""
    last = state.last_round()
    state.rounds = max((v.round for v in state.verdicts), default=1)
    state.final_round1 = aggregate_votes([v.verdict for v in state.first_round()], min_agree)
    state.final = aggregate_votes([v.verdict for v in last], min_agree)
    if state.final == 2:
        state.flags.append("human_review")  # the system abstains: a person should check this answer
    state.trace.append(f"decision (min_agree={min_agree}): {state.final}")
    state.add_step(
        "decision", "vote", round=state.rounds, min_agree=min_agree,
        votes={v.system.split(":", 1)[-1]: v.verdict for v in last},
        final=state.final, final_round1=state.final_round1,
    )
    return state.final


def verify_item(
    item: Item,
    verifiers: dict[str, ChatModel],
    evidence_mode: EvidenceMode,
    pubmed: Optional[PubMedSearch],
    k: int,
    min_agree: int,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    reask: bool = True,
    debate: bool = False,
    block: bool = True,
    retry: bool = True,
    second_round_rule: SecondRound = "off",
    k2: int = 2,
) -> VerificationState:
    if debate and second_round_rule != "off":
        raise ValueError("debate and the second evidence round cannot be used together (both add a round 2)")
    state = VerificationState(item_id=item.item_id, evidence_mode=evidence_mode)
    state.add_step("orchestrator", "start", verifiers=list(verifiers), evidence_mode=evidence_mode,
                   prompt_style=prompt_style, min_agree=min_agree, reask=reask, debate=debate,
                   block=block, retry=retry, second_round=second_round_rule, k2=k2)
    state.evidence = gather_evidence(item, evidence_mode, pubmed, k, state, block, retry)
    max_chars = None if evidence_mode == "source" else 1000
    evidence_text = format_evidence(state.evidence, max_chars=max_chars) if state.evidence else None

    for name, model in verifiers.items():
        state.verdicts.append(judge(name, model, item, evidence_text, prompt_style, 1, state, reask))
    if debate:
        debate_round(item, verifiers, evidence_text, prompt_style, state, reask)
    if second_round_rule != "off":
        second_round(item, verifiers, evidence_mode, pubmed, second_round_rule, k2, min_agree,
                     prompt_style, state, reask, block, retry)
    decide(state, min_agree)  # once per item, at the very end
    return state


def run_pipeline(
    items: list[Item],
    verifiers: dict[str, ChatModel],
    evidence_mode: EvidenceMode = "none",
    pubmed: Optional[PubMedSearch] = None,
    k: int = 3,
    min_agree: int = 2,
    out_path: Path | None = None,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    reask: bool = True,
    debate: bool = False,
    block: bool = True,
    retry: bool = True,
    second_round_rule: SecondRound = "off",
    k2: int = 2,
) -> list[VerificationState]:
    done: dict[str, VerificationState] = {}
    if out_path and Path(out_path).exists():
        done = {s.item_id: s for s in read_jsonl(out_path, VerificationState)}

    states = []
    for item in tqdm(items, desc=f"pipeline ({evidence_mode}, {len(verifiers)} verifiers)"):
        if item.item_id in done:
            states.append(done[item.item_id])
            continue
        state = verify_item(
            item, verifiers, evidence_mode, pubmed, k, min_agree,
            prompt_style=prompt_style, reask=reask, debate=debate, block=block, retry=retry,
            second_round_rule=second_round_rule, k2=k2,
        )
        states.append(state)
        if out_path:
            append_jsonl(out_path, state)
    return states


# ---------------------------------------------------- views over states

def _round_verdicts(s: VerificationState, round_no: Optional[int]) -> list[Verdict]:
    """Verdicts of one round; an item that never reached that round keeps its last round."""
    return s.last_round() if round_no is None or round_no >= s.rounds else s.round_verdicts(round_no)


def _decision(s: VerificationState, min_agree: Optional[int], round_no: Optional[int], forced: bool):
    """The saved decision, or a vote rule (or the forced rule) re-applied to one round."""
    if forced:
        codes = [v.verdict for v in _round_verdicts(s, round_no)]
        return aggregate_forced(codes, codes[0] if codes else None)
    if min_agree is not None:
        return aggregate_votes([v.verdict for v in _round_verdicts(s, round_no)], min_agree)
    if round_no is None or round_no >= s.rounds:
        return s.final
    if round_no == 1 and s.final_round1 is not None:
        return s.final_round1
    raise ValueError(f"{s.item_id}: no saved decision for round {round_no}; pass min_agree")


def final_verdicts(
    states: list[VerificationState],
    min_agree: Optional[int] = None,
    round: Optional[int] = None,
    forced: bool = False,
) -> list[Verdict]:
    """System-level verdicts, offline (no model calls).

    Default: the saved decision (from the last round). `min_agree` re-applies a vote rule;
    `round` picks the round (1 = the independent verdicts, before any debate); `forced`
    applies the forced-answer rule (majority, never abstains; ties go to the first verifier).
    """
    out = []
    for s in states:
        out.append(
            Verdict(
                item_id=s.item_id,
                system="pipeline",
                verdict=_decision(s, min_agree, round, forced),
                raw_text="",
                provider="pipeline",
                model="+".join(v.model for v in s.first_round()),
                latency_s=sum(v.latency_s for v in s.verdicts),
                prompt_tokens=sum(v.prompt_tokens for v in s.verdicts),
                completion_tokens=sum(v.completion_tokens for v in s.verdicts),
                cached=all(v.cached for v in s.verdicts),
                round=round or s.rounds,
            )
        )
    return out


def verifier_verdicts(states: list[VerificationState]) -> dict[str, list[Verdict]]:
    """Each verifier's ROUND-1 (independent) verdicts, for single-verifier scores and error overlap."""
    by_name: dict[str, list[Verdict]] = {}
    for s in states:
        for v in s.first_round():
            by_name.setdefault(v.system.split(":", 1)[-1], []).append(v)
    return by_name


def agreement_confidence(states: list[VerificationState]) -> dict[str, tuple[int, float]]:
    """(majority label, share of verifiers backing it) per item, from the LAST round, for risk-coverage curves."""
    out = {}
    for s in states:
        codes = [v.verdict for v in s.last_round()]
        zeros, ones = codes.count(0), codes.count(1)
        label = 1 if ones > zeros else 0
        out[s.item_id] = (label, max(zeros, ones) / max(len(codes), 1))
    return out
