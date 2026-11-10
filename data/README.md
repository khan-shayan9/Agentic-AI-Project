# Data

The datasets are not stored in the repository. One command downloads them and makes the fixed splits:

```bash
python -m medverify prepare
```

## What it does

- Downloads MedHallu (`pqa_labeled`, 1,000 rows) from Hugging Face and PubMedQA's `ori_pqal.json` from GitHub into `data/raw/`.
- Attaches each row's source PubMed ID (matched by question).
- Makes the fixed splits with seed 42 and writes them to `data/processed/` (`test.jsonl`, `val.jsonl`, `dev.jsonl`, `split_summary.json`).

Both `data/raw/` and `data/processed/` are ignored by git.

## Split definitions

| Split | Rows | Answers | Use |
| --- | --- | --- | --- |
| test | 300 (100 easy, 100 medium, 100 hard) | 600 | Final results only |
| val | 100 (34 / 33 / 33) | 200 | Prompt and trigger choice |
| dev | 20 (7 / 7 / 6) | 40 | Debugging |

Each row gives two answers: the correct one and the hallucinated one.

- Loader and split code: [`src/medverify/data.py`](../src/medverify/data.py) (`make_splits`).
- Split counts as committed: [`results/phase1/split_summary.json`](../results/phase1/split_summary.json). The Phase 2 notebook checks that a fresh `prepare` gives the same splits.
- Item lists with labels and difficulty: [`results/phase2/items_test.csv`](../results/phase2/items_test.csv) and [`items_val.csv`](../results/phase2/items_val.csv).

## Sample input

[`sample_items.csv`](sample_items.csv): five made-up items to try `python -m medverify batch`.
