"""Turn a model's reply into a verdict code (0 factual, 1 hallucinated, 2 not sure)."""

from __future__ import annotations

import re
from typing import Optional

_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_DIGIT = re.compile(r"(?<![\d.])([012])(?![\d.])")
_UNSURE_WORDS = ("not sure", "unsure", "uncertain", "cannot determine", "can't determine", "pass")


def parse_authors(text: str) -> int:
    """Exact copy of the parsing rule in the MedHallu authors' code.

    Kept only for reproducing their numbers. Note its quirk: any reply containing
    "not" (including "not sure") is read as 1 = hallucinated, never as an abstention.
    """
    t = text.lower()
    if any(x in t for x in ["1", "not", "non"]):
        return 1
    if any(x in t for x in ["not sure", "pass", "skip", "2"]):
        return 2
    return 0


def parse_verdict(text: str, allow_unsure: bool = True) -> Optional[int]:
    """Our stricter parser: the first standalone 0/1/2 in the reply.

    Returns None when nothing usable is found, so parse failures are counted
    separately instead of silently becoming a verdict.
    """
    if text is None:
        return None
    cleaned = _THINK.sub(" ", text).strip()
    match = _DIGIT.search(cleaned)
    if match:
        code = int(match.group(1))
        if code == 2 and not allow_unsure:
            return None
        return code
    lowered = cleaned.lower()
    if allow_unsure and any(w in lowered for w in _UNSURE_WORDS):
        return 2
    if "hallucinat" in lowered and "not hallucinat" not in lowered:
        return 1
    if "factual" in lowered and "non-factual" not in lowered and "not factual" not in lowered:
        return 0
    return None
