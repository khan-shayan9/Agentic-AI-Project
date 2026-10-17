import json
from pathlib import Path

import pytest

from medverify.schemas import Row

FIXTURES = Path(__file__).parent / "fixtures"


def make_rows(per_level: int = 5) -> list[Row]:
    rows = []
    i = 0
    for level in ("easy", "medium", "hard"):
        for _ in range(per_level):
            rows.append(
                Row(
                    row_id=f"pqa_labeled-{i:05d}",
                    subset="pqa_labeled",
                    question=f"Does drug {i} reduce symptom {i}?",
                    knowledge=[f"Background for study {i}.", f"Results show drug {i} reduced symptom {i} in trials."],
                    ground_truth=f"Drug {i} reduced symptom {i}.",
                    hallucinated_answer=f"Drug {i} increased symptom {i} through calcium channels.",
                    difficulty=level,
                    category="Misinterpretation of #Question#",
                    pubid=1000 + i,
                )
            )
            i += 1
    return rows


@pytest.fixture
def rows():
    return make_rows()


@pytest.fixture
def pubmedqa_sample():
    raw = json.loads((FIXTURES / "ori_pqal_sample.json").read_text())
    return {int(k): {"question": v["QUESTION"], "contexts": v["CONTEXTS"]} for k, v in raw.items()}
