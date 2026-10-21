"""Evaluation metrics.

- MedHallu-style precision / recall / F1 (positive class = hallucinated), computed
  after dropping "not sure" replies, overall and per difficulty level.
- Abstention metrics (instructor's condition 6): coverage = share of items answered,
  selective accuracy = accuracy on answered items, and a risk-coverage curve.
- Error overlap between two verifiers (instructor's condition 5).
- Cost and speed: tokens and latency per item.
- Pipeline runs only: leakage (how often PubMed returned the source) and efficiency
  (model calls, tokens, model time and PubMed requests per item, over all rounds).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Optional, Sequence

from .schemas import Item, Verdict, VerificationState

LEVELS = ("easy", "medium", "hard")


def prf(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = correct / len(y_true) if y_true else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def _summarise(pairs: list[tuple[int, Optional[int]]]) -> dict:
    """pairs = (true label, predicted code or None)."""
    n = len(pairs)
    answered = [(t, p) for t, p in pairs if p in (0, 1)]
    abstained = sum(1 for _, p in pairs if p == 2)
    invalid = sum(1 for _, p in pairs if p is None)
    out = {
        "n": n,
        "answered": len(answered),
        "coverage": len(answered) / n if n else 0.0,
        "abstain_rate": abstained / n if n else 0.0,
        "invalid_rate": invalid / n if n else 0.0,
    }
    scores = prf([t for t, _ in answered], [p for _, p in answered])
    out.update(scores)
    out["selective_accuracy"] = scores["accuracy"]  # accuracy on answered items only
    return out


def evaluate(items: Iterable[Item], verdicts: Iterable[Verdict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Overall and per-difficulty metrics, plus a bootstrap 95% interval for overall F1."""
    items, verdicts = list(items), list(verdicts)
    item_map = {it.item_id: it for it in items}
    pred = {v.item_id: v.verdict for v in verdicts}
    missing = [i for i in item_map if i not in pred]

    pairs_all, by_level = [], defaultdict(list)
    for item_id, item in item_map.items():
        if item_id not in pred:
            continue
        pair = (item.label, pred[item_id])
        pairs_all.append(pair)
        by_level[item.difficulty].append(pair)

    result = {"overall": _summarise(pairs_all), "missing_predictions": len(missing)}
    for level in LEVELS:
        if by_level[level]:
            result[level] = _summarise(by_level[level])
    result["overall"]["f1_ci95"] = bootstrap_f1(pairs_all, n_boot=n_boot, seed=seed)

    vs = verdicts
    if vs:
        result["cost"] = {
            "calls": len(vs),
            "cached_calls": sum(1 for v in vs if v.cached),
            "mean_latency_s": sum(v.latency_s for v in vs) / len(vs),
            "mean_prompt_tokens": sum(v.prompt_tokens for v in vs) / len(vs),
            "mean_completion_tokens": sum(v.completion_tokens for v in vs) / len(vs),
        }
    return result


def bootstrap_f1(pairs: list[tuple[int, Optional[int]]], n_boot: int = 1000, seed: int = 0) -> Optional[list[float]]:
    answered = [(t, p) for t, p in pairs if p in (0, 1)]
    if len(answered) < 10 or n_boot <= 0:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [answered[rng.randrange(len(answered))] for _ in answered]
        stats.append(prf([t for t, _ in sample], [p for _, p in sample])["f1"])
    stats.sort()
    return [stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]]


# -------------------------------------------------------- abstention curve

def risk_coverage(items: Iterable[Item], scored: dict[str, tuple[int, float]]) -> list[dict]:
    """Risk-coverage curve.

    `scored` maps item_id -> (predicted label 0/1, confidence). Items are answered in
    order of confidence; each point gives coverage and the error rate (risk) among
    the answered items.
    """
    labels = {it.item_id: it.label for it in items}
    ranked = sorted(((conf, pred, labels[i]) for i, (pred, conf) in scored.items() if i in labels), reverse=True)
    points, errors = [], 0
    for k, (conf, pred, label) in enumerate(ranked, 1):
        errors += int(pred != label)
        points.append({"coverage": k / len(ranked), "risk": errors / k, "threshold": conf})
    return points


# --------------------------------------------------------- error overlap

def error_overlap(items: Iterable[Item], a: Iterable[Verdict], b: Iterable[Verdict]) -> dict:
    """How often two verifiers are wrong on the SAME items.

    Uses items both verifiers answered (0 or 1). If errors were independent, the
    both-wrong rate would be close to p(A wrong) * p(B wrong); a ratio well above 1
    means correlated mistakes, i.e. the checks are not independent.
    """
    labels = {it.item_id: it.label for it in items}
    pa = {v.item_id: v.verdict for v in a}
    pb = {v.item_id: v.verdict for v in b}
    common = [i for i in labels if pa.get(i) in (0, 1) and pb.get(i) in (0, 1)]
    if not common:
        return {"n": 0}
    wa = [int(pa[i] != labels[i]) for i in common]
    wb = [int(pb[i] != labels[i]) for i in common]
    n = len(common)
    p_a, p_b = sum(wa) / n, sum(wb) / n
    both = sum(x and y for x, y in zip(wa, wb)) / n
    expected = p_a * p_b
    denom = math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    phi = (both - expected) / denom if denom else 0.0
    return {
        "n": n,
        "p_a_wrong": p_a,
        "p_b_wrong": p_b,
        "both_wrong": both,
        "both_wrong_if_independent": expected,
        "overlap_ratio": both / expected if expected else None,
        "p_b_wrong_given_a_wrong": both / p_a if p_a else None,
        "phi_error_correlation": phi,
    }


# ----------------------------------------------------------- vote rules

def aggregate_votes(codes: Sequence[Optional[int]], min_agree: int) -> int:
    """Majority-style vote with abstention.

    Counts 0 and 1 votes (2 = not sure and None are ignored). Returns the winning label
    only if at least `min_agree` verifiers chose it and it beats the other label;
    otherwise 2 (abstain).
    """
    zeros = sum(1 for c in codes if c == 0)
    ones = sum(1 for c in codes if c == 1)
    if ones > zeros and ones >= min_agree:
        return 1
    if zeros > ones and zeros >= min_agree:
        return 0
    return 2


def format_table(result: dict) -> str:
    rows = [("overall", result["overall"])] + [(lvl, result[lvl]) for lvl in LEVELS if lvl in result]
    lines = [f"{'split':<8}{'n':>6}{'cover':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'sel.acc':>9}"]
    for name, m in rows:
        lines.append(
            f"{name:<8}{m['n']:>6}{m['coverage']:>8.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}"
            f"{m['f1']:>8.3f}{m['selective_accuracy']:>9.3f}"
        )
    ci = result["overall"].get("f1_ci95")
    if ci:
        lines.append(f"overall F1 95% interval: {ci[0]:.3f} - {ci[1]:.3f}")
    return "\n".join(lines)


def aggregate_forced(codes: Sequence[Optional[int]], tie_break_code: Optional[int]) -> int:
    """Forced-answer rule (no abstention), the comparison for gap 3.

    Majority of the 0/1 votes. On a tie, or when nobody voted 0 or 1, use `tie_break_code`
    (the first verifier's vote, i.e. Llama); if that is not 0/1 either, return 1 (flag the
    answer as hallucinated, the safer choice). Never returns 2.
    """
    zeros = sum(1 for c in codes if c == 0)
    ones = sum(1 for c in codes if c == 1)
    if ones != zeros:
        return 1 if ones > zeros else 0
    return tie_break_code if tie_break_code in (0, 1) else 1


# ------------------------------------------------ leakage and efficiency

def _leak_found(s: VerificationState, round_no: int, reason: str) -> bool:
    """PubMed returned the source (or a near copy) in this round: blocked, or kept under --no-block."""
    return any(b.round == round_no and b.reason == reason for b in s.blocked) or any(
        e.round == round_no and e.leak == reason for e in s.evidence
    )


def _searched_in_round_2(s: VerificationState) -> bool:
    return any(st.action == "claim_search" and st.detail.get("queries") for st in s.steps)


def _leak_counts(states: list[VerificationState], round_no: int) -> dict:
    n = len(states)
    if not n:
        return {"items_searched": 0}
    clean = [sum(1 for e in s.evidence if e.round == round_no and e.leak is None) for s in states]
    empty = sum(1 for s in states if not any(e.round == round_no for e in s.evidence))
    return {
        "items_searched": n,
        "source_pmid_share": sum(_leak_found(s, round_no, "source_pmid") for s in states) / n,
        "near_duplicate_share": sum(_leak_found(s, round_no, "near_duplicate") for s in states) / n,
        "leaked_into_evidence_share": sum(any(e.round == round_no and e.leak for e in s.evidence) for s in states) / n,
        "mean_clean_abstracts": sum(clean) / n,
        "items_no_evidence": empty,
        "items_no_evidence_share": empty / n,
    }


def leakage_summary(states: Iterable[VerificationState]) -> dict:
    """How often PubMed returned the question's source abstract or near copies of it (gap 2).

    Round 1 counts every PubMed item; round 2 counts the items searched again with a claim
    query. "leaked_into_evidence" is non-zero only when blocking is off (ablation 2).
    """
    pubmed = [s for s in states if s.evidence_mode == "pubmed"]
    return {
        "round1": _leak_counts(pubmed, 1),
        "round2": _leak_counts([s for s in pubmed if _searched_in_round_2(s)], 2),
    }


# Ollama runs with a 4096-token context; prompts this long may have been cut.
CONTEXT_WARN_TOKENS = 3900


def efficiency_summary(states: Iterable[VerificationState]) -> dict:
    """Cost per item, summed over all rounds: model calls, tokens, model seconds, PubMed requests."""
    states = list(states)
    n = len(states)
    if not n:
        return {"items": 0}
    verdicts = [v for s in states for v in s.verdicts]
    tools = [c for s in states for c in s.tool_calls]
    triggered = sum(
        any(st.action == "second_round_check" and st.detail.get("triggered") for st in s.steps) for s in states
    )
    return {
        "items": n,
        "model_calls_per_item": sum(1 + v.reasked for v in verdicts) / n,
        "prompt_tokens_per_item": sum(v.prompt_tokens for v in verdicts) / n,
        "completion_tokens_per_item": sum(v.completion_tokens for v in verdicts) / n,
        "max_prompt_tokens": max((v.prompt_tokens for v in verdicts), default=0),
        "model_seconds_per_item": sum(v.latency_s for v in verdicts) / n,
        "pubmed_requests_per_item": len(tools) / n,
        "pubmed_uncached_per_item": sum(not c.cached for c in tools) / n,
        "pubmed_cached_per_item": sum(c.cached for c in tools) / n,
        "tool_calls_logged": bool(tools),  # False for Phase 1 runs, which did not log PubMed requests
        "round2_triggered_share": triggered / n,
        "round2_share": sum(s.rounds >= 2 for s in states) / n,
        "no_new_evidence_share": sum("no_new_evidence" in s.flags for s in states) / n,
        "reasks": sum(v.reasked for v in verdicts),
        "abstain_share": sum(s.final == 2 for s in states) / n,
        "unreadable_replies": sum(v.verdict is None for v in verdicts),
    }


def format_summaries(leakage: dict, efficiency: dict) -> str:
    """Short text version of the leakage and efficiency sections."""
    lines = ["", "-- leakage (PubMed returned the source / near copies) --"]
    for name, m in leakage.items():
        if not m.get("items_searched"):
            lines.append(f"{name}: no PubMed searches")
            continue
        lines.append(
            f"{name}: {m['items_searched']} items, source {m['source_pmid_share']:.3f}, near copies "
            f"{m['near_duplicate_share']:.3f}, leaked into evidence {m['leaked_into_evidence_share']:.3f}, "
            f"clean abstracts {m['mean_clean_abstracts']:.2f}/item, no evidence {m['items_no_evidence']}"
        )
    e = efficiency
    if e.get("items"):
        pubmed = (f"PubMed requests {e['pubmed_requests_per_item']:.2f} ({e['pubmed_uncached_per_item']:.2f} not cached)"
                  if e["tool_calls_logged"] else "PubMed requests not logged")
        lines += [
            "-- efficiency (per item, all rounds) --",
            f"model calls {e['model_calls_per_item']:.2f}, tokens {e['prompt_tokens_per_item']:.0f} in / "
            f"{e['completion_tokens_per_item']:.0f} out, model time {e['model_seconds_per_item']:.1f}s, {pubmed}",
            f"round 2: triggered {e['round2_triggered_share']:.3f}, judged {e['round2_share']:.3f}, "
            f"no new evidence {e['no_new_evidence_share']:.3f}; re-asks {e['reasks']}, "
            f"abstain {e['abstain_share']:.3f}, unreadable replies {e['unreadable_replies']}",
        ]
        if e.get("max_prompt_tokens", 0) >= CONTEXT_WARN_TOKENS:
            lines.append(
                f"WARNING: the longest prompt had {e['max_prompt_tokens']} tokens. Ollama runs with a 4096-token "
                "context, so prompts this long may have been cut; check the evidence length."
            )
    return "\n".join(lines)
