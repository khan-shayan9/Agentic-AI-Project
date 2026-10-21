import pytest

from medverify.data import rows_to_items
from medverify.metrics import aggregate_forced, aggregate_votes, error_overlap, evaluate, prf, risk_coverage
from medverify.schemas import Verdict


def v(item_id, code):
    return Verdict(item_id=item_id, system="t", verdict=code, raw_text="", provider="p", model="m", latency_s=0.1)


def test_prf_by_hand():
    m = prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5 and m["accuracy"] == 0.5


def test_evaluate_excludes_abstentions_and_reports_coverage(rows):
    items = rows_to_items(rows)  # 30 items, alternating labels 0,1
    verdicts = []
    for idx, it in enumerate(items):
        if idx < 6:
            verdicts.append(v(it.item_id, 2))  # abstain
        elif idx < 8:
            verdicts.append(v(it.item_id, None))  # unparseable
        else:
            verdicts.append(v(it.item_id, it.label))  # correct
    res = evaluate(items, verdicts, n_boot=50)
    o = res["overall"]
    assert o["n"] == 30 and o["answered"] == 22
    assert o["coverage"] == pytest.approx(22 / 30)
    assert o["abstain_rate"] == pytest.approx(6 / 30) and o["invalid_rate"] == pytest.approx(2 / 30)
    assert o["f1"] == 1.0 and o["selective_accuracy"] == 1.0
    assert set(res) >= {"easy", "medium", "hard", "cost"}


def test_error_overlap_detects_shared_mistakes(rows):
    items = rows_to_items(rows)
    wrong = {it.item_id for it in items[:6]}
    a = [v(it.item_id, 1 - it.label if it.item_id in wrong else it.label) for it in items]
    b = list(a)  # identical mistakes -> fully correlated
    r = error_overlap(items, a, b)
    assert r["n"] == 30 and r["both_wrong"] == pytest.approx(6 / 30)
    assert r["overlap_ratio"] == pytest.approx(5.0)  # 0.2 / (0.2 * 0.2)
    assert r["phi_error_correlation"] == pytest.approx(1.0)


def test_vote_rule():
    assert aggregate_votes([1, 1, 0], min_agree=2) == 1
    assert aggregate_votes([1, 1, 0], min_agree=3) == 2
    assert aggregate_votes([0, 2, None], min_agree=1) == 0
    assert aggregate_votes([1, 0, 2], min_agree=1) == 2  # tie -> abstain


def test_risk_coverage_is_ordered_by_confidence(rows):
    items = rows_to_items(rows)[:4]
    scored = {
        items[0].item_id: (items[0].label, 1.0),
        items[1].item_id: (items[1].label, 0.9),
        items[2].item_id: (1 - items[2].label, 0.5),
        items[3].item_id: (items[3].label, 0.4),
    }
    curve = risk_coverage(items, scored)
    assert [p["coverage"] for p in curve] == [0.25, 0.5, 0.75, 1.0]
    assert curve[1]["risk"] == 0.0 and curve[2]["risk"] == pytest.approx(1 / 3)


def test_forced_rule_never_abstains():
    assert aggregate_forced([1, 1, 0], tie_break_code=0) == 1  # majority
    assert aggregate_forced([0, 2, None], tie_break_code=2) == 0  # majority of the 0/1 votes
    assert aggregate_forced([1, 0, 2], tie_break_code=0) == 0  # tie -> the first verifier (Llama)
    assert aggregate_forced([2, 1, 0], tie_break_code=2) == 1  # tie and Llama unsure -> flag as hallucinated
    assert aggregate_forced([2, 2, None], tie_break_code=2) == 1  # all unsure -> flag as hallucinated
    assert aggregate_forced([None, 2, 2], tie_break_code=None) == 1
