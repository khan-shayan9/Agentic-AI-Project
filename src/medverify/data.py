"""Load MedHallu, attach PubMed IDs, turn rows into items and make fixed splits."""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import requests

from .schemas import Item, Row

MEDHALLU_HF_ID = "UTAustin-AIHealth/MedHallu"
PUBMEDQA_HF_ID = "qiaojin/PubMedQA"
PUBMEDQA_GITHUB_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"

# Column names in the Hugging Face release of MedHallu.
MEDHALLU_COLUMNS = {
    "question": "Question",
    "knowledge": "Knowledge",
    "ground_truth": "Ground Truth",
    "difficulty": "Difficulty Level",
    "hallucinated_answer": "Hallucinated Answer",
    "category": "Category of Hallucination",
}

LEVELS = ("easy", "medium", "hard")


# ---------------------------------------------------------------- loading

def _as_list(value) -> list[str]:
    """Knowledge may arrive as a list, a dict with 'contexts', or a string of either."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            try:
                return _as_list(ast.literal_eval(text))
            except (ValueError, SyntaxError):
                return [text]
        return [text] if text else []
    if isinstance(value, dict):
        return _as_list(value.get("contexts", []))
    return [str(v) for v in value]


def normalise_row(raw: dict, index: int, subset: str) -> Row:
    get = lambda key: raw[MEDHALLU_COLUMNS[key]]  # noqa: E731
    return Row(
        row_id=f"{subset}-{index:05d}",
        subset=subset,
        question=str(get("question")).strip(),
        knowledge=_as_list(get("knowledge")),
        ground_truth=str(get("ground_truth")).strip(),
        hallucinated_answer=str(get("hallucinated_answer")).strip(),
        difficulty=str(get("difficulty")).strip().lower(),
        category=str(get("category")).strip(),
    )


def load_medhallu(subset: str = "pqa_labeled") -> list[Row]:
    """Download MedHallu from Hugging Face (needs internet; set HF_TOKEN if asked to log in)."""
    from datasets import load_dataset

    ds = load_dataset(MEDHALLU_HF_ID, subset, split="train")
    return [normalise_row(raw, i, subset) for i, raw in enumerate(ds)]


# ---------------------------------------------------------- PubMed IDs

def normalise_question(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def load_pubmedqa_labeled(local_path: Optional[Path] = None) -> dict[int, dict]:
    """Return {pubid: {"question": ..., "contexts": [...]}} for PubMedQA pqa_labeled.

    Tries a local copy, then the official GitHub file, then Hugging Face.
    """
    if local_path and Path(local_path).exists():
        raw = json.loads(Path(local_path).read_text())
        return {int(k): {"question": v["QUESTION"], "contexts": v["CONTEXTS"]} for k, v in raw.items()}
    try:
        resp = requests.get(PUBMEDQA_GITHUB_URL, timeout=60)
        resp.raise_for_status()
        raw = resp.json()
        if local_path:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_text(json.dumps(raw))
        return {int(k): {"question": v["QUESTION"], "contexts": v["CONTEXTS"]} for k, v in raw.items()}
    except requests.RequestException:
        from datasets import load_dataset

        ds = load_dataset(PUBMEDQA_HF_ID, "pqa_labeled", split="train")
        return {int(r["pubid"]): {"question": r["question"], "contexts": r["context"]["contexts"]} for r in ds}


def attach_pubids(rows: list[Row], pubmedqa: dict[int, dict]) -> dict[str, int]:
    """Fill row.pubid by matching the question text, using the abstract to break ties.

    Returns counts of how each row was matched, so the mapping can be reported.
    """
    by_question: dict[str, list[int]] = defaultdict(list)
    for pubid, rec in pubmedqa.items():
        by_question[normalise_question(rec["question"])].append(pubid)

    stats: Counter = Counter()
    for row in rows:
        candidates = by_question.get(normalise_question(row.question), [])
        if len(candidates) == 1:
            row.pubid = candidates[0]
            stats["matched_by_question"] += 1
            continue
        # Several (or zero) question matches: compare the first context sentence.
        first_ctx = normalise_question(row.knowledge[0]) if row.knowledge else ""
        pool = candidates or list(pubmedqa)
        hits = [p for p in pool if pubmedqa[p]["contexts"] and normalise_question(pubmedqa[p]["contexts"][0]) == first_ctx]
        if len(hits) == 1 and first_ctx:
            row.pubid = hits[0]
            stats["matched_by_context"] += 1
        else:
            stats["unmatched"] += 1
    return dict(stats)


# ---------------------------------------------------------------- items

def rows_to_items(rows: Iterable[Row]) -> list[Item]:
    """Each row gives two items: its true answer (label 0) and its hallucinated answer (label 1).

    The MedHallu code shows ONE randomly chosen answer per row; using both gives the
    same 50/50 balance without randomness and doubles the evidence per row.
    """
    items: list[Item] = []
    for row in rows:
        common = dict(
            row_id=row.row_id,
            question=row.question,
            difficulty=row.difficulty,
            category=row.category,
            knowledge=row.knowledge,
            pubid=row.pubid,
        )
        items.append(Item(item_id=f"{row.row_id}-gt", answer=row.ground_truth, label=0, **common))
        items.append(Item(item_id=f"{row.row_id}-hal", answer=row.hallucinated_answer, label=1, **common))
    return items


# ---------------------------------------------------------------- splits

def make_splits(
    rows: list[Row],
    test_per_level: int = 100,
    val_size: int = 100,
    dev_size: int = 20,
    seed: int = 42,
) -> dict[str, list[Row]]:
    """Fixed, non-overlapping splits.

    test: `test_per_level` rows from each difficulty level (balanced).
    val:  `val_size` rows spread evenly across levels (used only to tune thresholds).
    dev:  `dev_size` rows for debugging prompts and code.
    """
    rng = random.Random(seed)
    pools: dict[str, list[Row]] = {}
    for level in LEVELS:
        level_rows = sorted((r for r in rows if r.difficulty == level), key=lambda r: r.row_id)
        rng.shuffle(level_rows)
        pools[level] = level_rows

    for level in LEVELS:
        if len(pools[level]) < test_per_level:
            raise ValueError(f"Only {len(pools[level])} '{level}' rows; need {test_per_level} for the test split.")

    test = [r for level in LEVELS for r in pools[level][:test_per_level]]
    remaining = {level: pools[level][test_per_level:] for level in LEVELS}

    def take_round_robin(n: int) -> list[Row]:
        out: list[Row] = []
        while len(out) < n and any(remaining.values()):
            for level in LEVELS:
                if remaining[level] and len(out) < n:
                    out.append(remaining[level].pop(0))
        return out

    val = take_round_robin(val_size)
    dev = take_round_robin(dev_size)
    return {"test": test, "val": val, "dev": dev}


def split_summary(splits: dict[str, list[Row]]) -> dict[str, dict[str, int]]:
    return {name: dict(Counter(r.difficulty for r in rows)) | {"rows": len(rows)} for name, rows in splits.items()}


# ---------------------------------------------------------------- JSONL io

def write_jsonl(path: Path, records: Iterable) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            data = rec.model_dump() if hasattr(rec, "model_dump") else rec
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = record.model_dump() if hasattr(record, "model_dump") else record
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_jsonl(path: Path, model=None) -> list:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                out.append(model.model_validate(data) if model else data)
    return out


def load_split_items(data_dir: Path, split: str) -> list[Item]:
    rows = read_jsonl(Path(data_dir) / f"{split}.jsonl", Row)
    return rows_to_items(rows)
