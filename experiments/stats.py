"""Small statistics in pure Python: mean and SD, a paired bootstrap over items, and McNemar's exact test.

Both tests compare two systems on the SAME items. "Answered" means a 0/1 verdict; 2 (abstain)
and unreadable replies are left out, as in MedHallu, so each side's F1 and accuracy are computed
on its own answered items.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Optional, Sequence

TP, FP, FN, TN, NOT_ANSWERED = range(5)


def mean_sd(values: Sequence[Optional[float]]) -> tuple[Optional[float], Optional[float], int]:
    """(mean, sample SD, n) of the values that are not None; SD is None for fewer than 2 values."""
    xs = [v for v in values if v is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None, 1
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)), len(xs)


def outcome(label: int, code: Optional[int]) -> int:
    if code not in (0, 1):
        return NOT_ANSWERED
    if code == 1:
        return TP if label == 1 else FP
    return FN if label == 1 else TN


def _scores(counts: Sequence[int]) -> tuple[float, float]:
    """(F1, accuracy on answered) from outcome counts [tp, fp, fn, tn, not answered]."""
    tp, fp, fn, tn = counts[TP], counts[FP], counts[FN], counts[TN]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    answered = tp + fp + fn + tn
    return f1, ((tp + tn) / answered if answered else 0.0)


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    n = len(values)
    return values[int(0.025 * n)], values[max(int(0.975 * n) - 1, 0)]


def _two_sided_p(diffs: list[float]) -> float:
    n = len(diffs)
    below = sum(1 for d in diffs if d <= 0) / n
    above = sum(1 for d in diffs if d >= 0) / n
    return min(1.0, 2 * min(below, above))


def paired_bootstrap(
    labels: Sequence[int],
    pred_a: Sequence[Optional[int]],
    pred_b: Sequence[Optional[int]],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap over items of A - B, for F1 and for accuracy on answered items.

    Resampling items with replacement only changes how often each (outcome of A, outcome of B)
    pair occurs, so each resample draws those pair counts directly (same distribution, faster).
    Returns the observed differences, 95% intervals and two-sided p values.
    """
    pairs = Counter((outcome(t, a), outcome(t, b)) for t, a, b in zip(labels, pred_a, pred_b))
    keys = list(pairs)
    n = sum(pairs.values())

    def diff(pair_counts) -> tuple[float, float]:
        a, b = [0] * 5, [0] * 5
        for (oa, ob), c in pair_counts:
            a[oa] += c
            b[ob] += c
        (f1_a, acc_a), (f1_b, acc_b) = _scores(a), _scores(b)
        return f1_a - f1_b, acc_a - acc_b

    observed = diff(pairs.items())
    if n == 0:
        return {"n": 0}
    rng = random.Random(seed)
    cum, total = [], 0
    for k in keys:
        total += pairs[k]
        cum.append(total)
    f1_diffs, acc_diffs = [], []
    for _ in range(n_resamples):
        drawn = Counter(rng.choices(range(len(keys)), cum_weights=cum, k=n))
        d_f1, d_acc = diff((keys[i], c) for i, c in drawn.items())
        f1_diffs.append(d_f1)
        acc_diffs.append(d_acc)
    f1_ci, acc_ci = _percentile_interval(f1_diffs), _percentile_interval(acc_diffs)
    return {
        "n": n,
        "f1_diff": observed[0], "f1_ci_low": f1_ci[0], "f1_ci_high": f1_ci[1], "f1_p": _two_sided_p(f1_diffs),
        "acc_diff": observed[1], "acc_ci_low": acc_ci[0], "acc_ci_high": acc_ci[1], "acc_p": _two_sided_p(acc_diffs),
        "resamples": n_resamples, "seed": seed,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p value: b = A right and B wrong, c = A wrong and B right."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar(labels: Sequence[int], pred_a: Sequence[Optional[int]], pred_b: Sequence[Optional[int]]) -> dict:
    """McNemar's exact test on the items both sides answered."""
    b = c = both = 0
    for t, a, p in zip(labels, pred_a, pred_b):
        if a in (0, 1) and p in (0, 1):
            both += 1
            b += a == t and p != t
            c += a != t and p == t
    return {"both_answered": both, "b": b, "c": c, "p": mcnemar_exact(b, c)}
