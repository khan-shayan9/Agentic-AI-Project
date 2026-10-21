"""Items typed by a user: no gold label, stable ids; old data still loads."""

import json
from pathlib import Path

import pytest

import medverify.data as D
from medverify.metrics import evaluate
from medverify.schemas import Item, Verdict, VerificationState, user_item

PHASE1 = Path(__file__).parents[1] / "results" / "phase1"


def test_user_item_ids_are_stable_and_unlabelled():
    a = user_item("Does drug X lower blood pressure?", "Yes, by 10 mmHg.")
    b = user_item("  Does drug X lower blood pressure?", "Yes, by 10 mmHg.  ", pmid="12345")
    assert a.item_id == b.item_id and a.item_id.startswith("user-") and len(a.item_id) == len("user-") + 12
    assert a.item_id != user_item("Does drug X lower blood pressure?", "No.").item_id
    assert a.label is None and a.difficulty == "unknown" and a.pubid is None and b.pubid == 12345
    assert user_item("q", "a", pmid="").pubid is None


def test_old_items_and_phase1_states_still_load(rows):
    item = D.rows_to_items(rows)[1]
    again = Item.model_validate(json.loads(item.model_dump_json()))
    assert again == item and again.label == 1 and again.difficulty in ("easy", "medium", "hard")
    states = D.read_jsonl(PHASE1 / "sysA-val-reason-states.jsonl", VerificationState)
    assert len(states) == 200


def test_evaluate_refuses_unlabelled_items(rows):
    items = D.rows_to_items(rows)[:2] + [user_item("q?", "a.")]
    verdicts = [Verdict(item_id=i.item_id, system="t", verdict=0, raw_text="", provider="p", model="m", latency_s=0)
                for i in items]
    with pytest.raises(ValueError, match="1 item\\(s\\) have none"):
        evaluate(items, verdicts)
    assert evaluate(items[:2], verdicts[:2], n_boot=0)["overall"]["n"] == 2
