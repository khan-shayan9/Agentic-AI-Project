"""Facts from the Phase 1 proposal, used word for word by the report, and the rule that turns the
success criteria into "Addressed?" in the gap-resolution table.

The rule below was written and committed before any Phase 2 test result was read (see the dated
row in docs/devlog.md); the report only applies it.
"""

from __future__ import annotations

from typing import Optional, Sequence

HYPOTHESES = {
    "H1": (
        "The three-family vote has a higher F1 than B4 (self-consistency), because cross-family pairs "
        "share fewer mistakes than samples of one model."
    ),
    "H2": (
        "Setting A lands between the two baselines: its F1 is above B0 (no knowledge) but below B1 "
        "(source given)."
    ),
    "H3": (
        "Test coverage stays within 5 points of validation coverage, and accuracy on answered items is "
        "higher than when the same system must answer everything."
    ),
}

SUCCESS_CRITERIA = {
    1: (
        "Error overlap is reported for all three verifier pairs and for repeated samples of one model, "
        "and the three-family vote has a higher F1 than both the best single verifier and the same-model "
        "control with the same number of calls."
    ),
    2: (
        "The logs show no kept abstract that is the source or a near copy; setting A beats the "
        "no-knowledge MedHallu baseline in F1, and setting B beats the knowledge baseline, both rerun by "
        "us on the same items."
    ),
    3: (
        "The abstain rule chosen on the validation split keeps its coverage on the test split within 5 "
        "points, and gives higher accuracy on answered items than the same system forced to answer "
        "everything."
    ),
}

# The parts of each criterion that the report checks, in order.
CRITERION_PARTS = {
    1: [
        "error overlap reported for the 3 cross-family pairs and for same-model samples",
        "round-1 three-family vote F1 > best single verifier (chosen on val)",
        "round-1 three-family vote F1 > B4 (same number of calls)",
    ],
    2: [
        "no kept abstract is the source or a near copy in any setting-A run",
        "System A F1 > B0 (rerun on the same items)",
        "System B F1 > B1 (rerun on the same items)",
    ],
    3: [
        "|test coverage - val coverage| <= 0.05 (val coverage from final_config.json)",
        "accuracy on answered items > the same system forced to answer",
    ],
}

# Published MedHallu numbers for Llama-3.1-8B-Instruct (paper Table 2): reported, not rerun.
# The paper gives no recall for these rows.
MEDHALLU_SOURCE = "MedHallu paper (Pandit et al., EMNLP 2025), Table 2, Llama-3.1-8B-Instruct, reported"
MEDHALLU_REPORTED = {
    "B0": {"label": "MedHallu reported: no knowledge", "precision": 0.791, "recall": None, "f1": 0.522,
           "easy_f1": 0.679, "medium_f1": 0.515, "hard_f1": 0.372},
    "B1": {"label": "MedHallu reported: with knowledge", "precision": 0.775, "recall": None, "f1": 0.797,
           "easy_f1": 0.880, "medium_f1": 0.796, "hard_f1": 0.722},
}

# ------------------------------------------------------------- the rule

COVERAGE_TOLERANCE = 0.05

ADDRESSED_RULE = (
    "A comparison part (\"X has a higher F1 / accuracy than Y\") is met when the mean difference over "
    "seeds is above 0 AND the seed-1 paired bootstrap 95% interval of the difference is above 0. "
    "The coverage part is met when |test coverage - val coverage| <= 0.05, with the val coverage saved "
    "in experiments/final_config.json. The leakage part is met when no kept abstract has a leak mark "
    "(or is the item's own source PMID) in any setting-A run. The reporting part is met when the error "
    "overlap exists for the three cross-family pairs and for same-model samples. "
    "Fully = every part met; Partially = some parts met; Not = none met. "
    "If a part cannot be checked because runs are missing, the gap is marked Pending (never guessed)."
)


def comparison_met(mean_diff: Optional[float], ci_low: Optional[float]) -> Optional[bool]:
    """Mean difference over seeds > 0 and the seed-1 bootstrap 95% interval above 0."""
    if mean_diff is None or ci_low is None:
        return None
    return mean_diff > 0 and ci_low > 0


def coverage_met(test_coverage: Optional[float], val_coverage: Optional[float]) -> Optional[bool]:
    if test_coverage is None or val_coverage is None:
        return None
    return abs(test_coverage - val_coverage) <= COVERAGE_TOLERANCE + 1e-9


def leakage_met(kept_leaks: Optional[int]) -> Optional[bool]:
    return None if kept_leaks is None else kept_leaks == 0


def addressed(parts: Sequence[Optional[bool]]) -> str:
    """Fully / Partially / Not from the parts of one criterion; Pending if any part cannot be checked."""
    if not parts or any(p is None for p in parts):
        return "Pending"
    if all(parts):
        return "Fully"
    return "Partially" if any(parts) else "Not"
