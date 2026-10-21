"""Phase 2 report: facts, statistics, export, tables, gap table, failure analysis and figures (offline)."""

import pytest

from experiments import facts


def test_addressed_rule():
    met = facts.comparison_met
    assert met(0.02, 0.001) is True and met(0.02, -0.01) is False and met(-0.01, 0.002) is False
    assert met(None, 0.1) is None and met(0.1, None) is None
    assert facts.coverage_met(0.86, 0.89) is True and facts.coverage_met(0.83, 0.89) is False
    assert facts.coverage_met(0.84, 0.89) is True  # exactly 5 points
    assert facts.leakage_met(0) is True and facts.leakage_met(2) is False and facts.leakage_met(None) is None
    assert facts.addressed([True, True, True]) == "Fully"
    assert facts.addressed([True, False, True]) == "Partially"
    assert facts.addressed([False, False]) == "Not"
    assert facts.addressed([True, None]) == "Pending" and facts.addressed([]) == "Pending"


def test_facts_are_complete():
    assert set(facts.HYPOTHESES) == {"H1", "H2", "H3"} and set(facts.SUCCESS_CRITERIA) == {1, 2, 3}
    assert facts.MEDHALLU_REPORTED["B0"]["f1"] == 0.522 and facts.MEDHALLU_REPORTED["B1"]["hard_f1"] == 0.722
    assert facts.MEDHALLU_REPORTED["B0"]["recall"] is None  # the paper gives no recall
