"""Evaluation metrics.

- MedHallu-style precision / recall / F1 (positive class = hallucinated), computed
  after dropping "not sure" replies, overall and per difficulty level.
- Abstention metrics (instructor's condition 6): coverage = share of items answered,
  selective accuracy = accuracy on answered items, and a risk-coverage curve.
- Error overlap between two verifiers (instructor's condition 5).
- Cost and speed: tokens and latency per item.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Optional, Sequence

from .schemas import Item, Verdict

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
