from medverify import data as D
from medverify.schemas import Row


def hf_record(**overrides):
    rec = {
        "Question": "  Does X help Y? ",
        "Knowledge": ["ctx one", "ctx two"],
        "Ground Truth": "Yes, X helps Y.",
        "Difficulty Level": "Hard",
        "Hallucinated Answer": "No, X harms Y.",
        "Category of Hallucination": "Incomplete Information",
    }
    rec.update(overrides)
    return rec


def test_normalise_row_handles_column_names_and_case():
    row = D.normalise_row(hf_record(), 7, "pqa_labeled")
    assert row.row_id == "pqa_labeled-00007"
    assert row.question == "Does X help Y?"
    assert row.difficulty == "hard"
    assert row.knowledge == ["ctx one", "ctx two"]


def test_knowledge_formats_are_all_accepted():
    assert D._as_list("['a', 'b']") == ["a", "b"]
    assert D._as_list("{'contexts': ['a']}") == ["a"]
    assert D._as_list({"contexts": ["a", "b"]}) == ["a", "b"]
    assert D._as_list("plain text") == ["plain text"]
    assert D._as_list(None) == []


def test_rows_to_items_gives_two_balanced_items_per_row(rows):
    items = D.rows_to_items(rows)
    assert len(items) == 2 * len(rows)
    assert sum(it.label for it in items) == len(rows)
    first_gt, first_hal = items[0], items[1]
    assert first_gt.item_id.endswith("-gt") and first_gt.label == 0
    assert first_hal.item_id.endswith("-hal") and first_hal.label == 1
    assert first_gt.pubid == first_hal.pubid == rows[0].pubid


def test_splits_are_balanced_disjoint_and_repeatable():
    from conftest import make_rows

    rows = make_rows(per_level=20)
    splits = D.make_splits(rows, test_per_level=10, val_size=9, dev_size=6, seed=1)
    summary = D.split_summary(splits)
    assert summary["test"] == {"easy": 10, "medium": 10, "hard": 10, "rows": 30}
    assert summary["val"]["rows"] == 9 and summary["dev"]["rows"] == 6
    ids = [r.row_id for split in splits.values() for r in split]
    assert len(ids) == len(set(ids)), "splits overlap"
    again = D.make_splits(list(reversed(rows)), test_per_level=10, val_size=9, dev_size=6, seed=1)
    assert [r.row_id for r in again["test"]] == [r.row_id for r in splits["test"]]


def test_attach_pubids_matches_real_pubmedqa_questions(pubmedqa_sample):
    rows = []
    for i, (pubid, rec) in enumerate(pubmedqa_sample.items()):
        q = rec["question"].replace("?", " ?").upper()  # punctuation/case differences must not matter
        rows.append(
            Row(
                row_id=f"r{i}",
                subset="pqa_labeled",
                question=q,
                knowledge=rec["contexts"],
                ground_truth="g",
                hallucinated_answer="h",
                difficulty="easy",
                category="c",
            )
        )
    stats = D.attach_pubids(rows, pubmedqa_sample)
    assert stats == {"matched_by_question": 3}
    assert [r.pubid for r in rows] == list(pubmedqa_sample)


def test_attach_pubids_falls_back_to_context(pubmedqa_sample):
    pubid, rec = next(iter(pubmedqa_sample.items()))
    row = Row(
        row_id="r0",
        subset="pqa_labeled",
        question="A reworded question that matches nothing",
        knowledge=rec["contexts"],
        ground_truth="g",
        hallucinated_answer="h",
        difficulty="easy",
        category="c",
    )
    stats = D.attach_pubids([row], pubmedqa_sample)
    assert stats == {"matched_by_context": 1} and row.pubid == pubid


def test_jsonl_roundtrip(tmp_path, rows):
    path = tmp_path / "x.jsonl"
    D.write_jsonl(path, rows)
    assert D.read_jsonl(path, Row) == rows
