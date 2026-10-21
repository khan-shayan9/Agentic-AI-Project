"""Plain-English views of one VerificationState, shared by `medverify verify`, batch mode and the web app.

Small pure functions: they only read the state, so they can be tested without models or Streamlit.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from .schemas import Item, Verdict, VerificationState

DECISIONS = {
    0: ("ACCEPT", "factual: the verifiers found no hallucination"),
    1: ("REJECT", "hallucinated: the answer contains wrong or invented information"),
    2: ("NOT SURE", "sent for human review"),
}
WORDS = {0: "factual", 1: "hallucinated", 2: "unsure", None: "unreadable"}
DISCLAIMER = "Research prototype, not medical advice."
_VERDICT_LINE = re.compile(r"verdict\s*[:=\-]", re.IGNORECASE)


def decision_label(code: Optional[int]) -> str:
    return DECISIONS.get(code, ("UNKNOWN", ""))[0]


def decision_text(code: Optional[int]) -> str:
    """'ACCEPT (factual: ...)', 'REJECT (hallucinated: ...)' or 'NOT SURE (sent for human review)'."""
    label, meaning = DECISIONS.get(code, ("UNKNOWN", "no decision"))
    return f"{label} ({meaning})"


def pubmed_url(pmid: str) -> Optional[str]:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if str(pmid).isdigit() else None


def verifier_name(verdict: Verdict) -> str:
    return verdict.system.split(":", 1)[-1]


def reason_line(raw_text: str, limit: int = 200) -> str:
    """The verifier's reasoning in one line: the text before its final 'Verdict:' line."""
    text = raw_text.split("\n---\n")[0]  # after a re-ask, the first reply holds the reasoning
    matches = list(_VERDICT_LINE.finditer(text))
    if matches:
        text = text[: matches[-1].start()]
    text = " ".join(text.split())
    if not text:
        return "(no reason given)"
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def min_agree_of(state: VerificationState, default: int = 2) -> int:
    start = next((s for s in state.steps if s.action == "start"), None)
    return start.detail.get("min_agree", default) if start else default


def agreement(state: VerificationState) -> str:
    """How many verifiers backed the decision, from the votes the decision used (the last round)."""
    votes = [v.verdict for v in state.last_round()]
    if state.final in (0, 1):
        return f"{votes.count(state.final)} of {len(votes)} verifiers said {WORDS[state.final]}"
    return f"no verdict had {min_agree_of(state)} of {len(votes)} votes ({vote_spread(votes)})"


def vote_spread(votes: list) -> str:
    """'2 hallucinated, 1 unsure'."""
    return ", ".join(f"{n} {word}" for word, n in Counter(WORDS[v] for v in votes).most_common())


def second_round_text(state: VerificationState) -> str:
    actions = {s.action for s in state.steps}
    check = next((s for s in state.steps if s.action == "second_round_check"), None)
    if "second_round_judge" in actions:
        new = [e.pmid for e in state.evidence if e.round == 2]
        return f"ran: a claim search found {len(new)} new abstract(s) and the verifiers judged again"
    if "no_new_evidence" in state.flags:
        return "triggered, but the claim search found no new abstracts; the round-1 votes stand"
    if "second_round_skipped" in actions:
        return "triggered, but skipped (it needs PubMed evidence)"
    if check is not None:
        return "not needed: the round-1 votes were clear" if not check.detail.get("triggered") else "triggered"
    return "off"


def abstain_reason(state: VerificationState) -> Optional[str]:
    """Why the system said NOT SURE, in plain words (None for ACCEPT or REJECT)."""
    if state.final != 2:
        return None
    votes = [v.verdict for v in state.last_round()]
    if any(s.action == "second_round_judge" for s in state.steps):
        return "The verifiers still disagreed after a second search for evidence; please check this answer by hand."
    if "no_new_evidence" in state.flags:
        return "The verifiers disagreed, and a second search found no new abstracts; please check this answer by hand."
    if votes.count(2) + votes.count(None) > len(votes) / 2:
        return "Most verifiers were unsure: the evidence did not settle it; please check this answer by hand."
    return (f"No verdict had the {min_agree_of(state)} votes it needs ({vote_spread(votes)}); "
            "please check this answer by hand.")


def verifier_rows(state: VerificationState) -> list[dict]:
    """One row per verifier per round: its vote and one-line reason."""
    return [{
        "verifier": verifier_name(v), "round": v.round, "vote": WORDS[v.verdict], "code": v.verdict,
        "reason": reason_line(v.raw_text), "reasked": v.reasked, "seconds": round(v.latency_s, 1),
    } for v in state.verdicts]


def evidence_rows(state: VerificationState) -> list[dict]:
    """The abstracts the verifiers saw, with PubMed links."""
    return [{"pmid": e.pmid, "title": e.title, "url": pubmed_url(e.pmid), "round": e.round, "query": e.query,
             "leak": e.leak} for e in state.evidence]


def blocked_rows(state: VerificationState) -> list[dict]:
    """The abstracts removed to stop source leakage, and why."""
    why = {"source_pmid": "the answer's own source abstract",
           "near_duplicate": "a near copy of the source abstract"}
    return [{"pmid": b.pmid, "reason": why.get(b.reason, b.reason) + (f" (overlap {b.overlap:.0%})" if b.reason == "near_duplicate" else ""),
             "url": pubmed_url(b.pmid), "round": b.round} for b in state.blocked]


def cost_summary(state: VerificationState) -> dict:
    return {
        "model_calls": sum(1 + v.reasked for v in state.verdicts),
        "prompt_tokens": sum(v.prompt_tokens for v in state.verdicts),
        "completion_tokens": sum(v.completion_tokens for v in state.verdicts),
        "model_seconds": round(sum(v.latency_s for v in state.verdicts), 1),
        "pubmed_requests": len(state.tool_calls),
    }


def format_report(state: VerificationState, item: Optional[Item] = None, second_round_note: str = "") -> str:
    """The plain-English result printed by `medverify verify`."""
    lines = [f"Decision: {decision_text(state.final)}", f"Agreement: {agreement(state)}"]
    reason = abstain_reason(state)
    if reason:
        lines.append(f"Why not sure: {reason}")
    lines += ["", "Verifiers:"]
    lines += [f"  {r['verifier']} (round {r['round']}): {r['vote']}{' (asked again)' if r['reasked'] else ''} - {r['reason']}"
              for r in verifier_rows(state)]
    mode = {"pubmed": "PubMed, source blocked", "source": "the source abstract", "none": "none"}.get(state.evidence_mode, state.evidence_mode)
    lines += ["", f"Evidence ({mode}):"]
    lines += [f"  PMID {r['pmid']}{' (second search)' if r['round'] == 2 else ''}  {r['title']}  {r['url'] or ''}".rstrip()
              for r in evidence_rows(state)] or ["  none"]
    blocked = blocked_rows(state)
    if blocked:
        lines += ["", "Blocked to stop source leakage:"]
        lines += [f"  PMID {r['pmid']}{' (second search)' if r['round'] == 2 else ''}: {r['reason']}" for r in blocked]
    lines += ["", f"Second round: {second_round_text(state)}" + (f" [{second_round_note}]" if second_round_note else "")]
    c = cost_summary(state)
    lines.append(f"Cost: {c['model_calls']} model calls, {c['prompt_tokens']} prompt + {c['completion_tokens']} completion "
                 f"tokens, {c['model_seconds']} model seconds, {c['pubmed_requests']} PubMed requests")
    if state.flags:
        lines.append(f"Flags: {', '.join(state.flags)}")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)
