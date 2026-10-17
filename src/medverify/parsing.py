"""Turn a model's reply into a verdict code (0 factual, 1 hallucinated, 2 not sure)."""

from __future__ import annotations

import re
from typing import Optional

_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_DIGIT = re.compile(r"(?<![\d.])([012])(?![\d.])")
_UNSURE_WORDS = ("not sure", "unsure", "uncertain", "cannot determine", "can't determine", "pass")
_UNSURE_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in _UNSURE_WORDS) + r")\b")


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
    if allow_unsure and _UNSURE_RE.search(lowered):  # whole words, so "passage" is not "pass"
        return 2
    if "hallucinat" in lowered and "not hallucinat" not in lowered:
        return 1
    if "factual" in lowered and "non-factual" not in lowered and "not factual" not in lowered:
        return 0
    return None


_LABELS = {"factual": 0, "hallucinated": 1, "unsure": 2}
_LEADING_LABEL = re.compile(r"^[\s*_`'\"#>:\-]*(factual|hallucinated|unsure)\b", flags=re.IGNORECASE)
_STATED_LABEL = re.compile(
    r"(?:verdict|judgement|judgment|answer|label)\s*(?:is)?\s*[:=\-]?\s*[*_`'\"]*(factual|hallucinated|unsure)\b",
    flags=re.IGNORECASE,
)


def parse_label(text: str) -> Optional[int]:
    """Verifier replies (FACTUAL / HALLUCINATED / UNSURE) -> 0 / 1 / 2.

    Reads the label the reply starts with, else a stated "Verdict: X", else falls back to
    parse_verdict (digits and keywords). None when nothing usable is found.
    """
    if text is None:
        return None
    cleaned = _THINK.sub(" ", text).strip()
    match = _LEADING_LABEL.search(cleaned) or _STATED_LABEL.search(cleaned)
    if match:
        return _LABELS[match.group(1).lower()]
    return parse_verdict(cleaned, allow_unsure=True)


_FINAL_VERDICT = re.compile(r"verdict\s*[:=\-]\s*[*_`'\"]*(factual|hallucinated|unsure)\b", flags=re.IGNORECASE)


def parse_final_verdict(text: str) -> Optional[int]:
    """'reason' style: the LAST "Verdict: X" line wins; falls back to parse_label."""
    if text is None:
        return None
    cleaned = _THINK.sub(" ", text)
    found = _FINAL_VERDICT.findall(cleaned)
    if found:
        return _LABELS[found[-1].lower()]
    return parse_label(cleaned)


def parse_for_style(text: str, style: str) -> Optional[int]:
    """Parser that matches the verifier prompt style."""
    if style == "digits":
        return parse_verdict(text, allow_unsure=True)
    if style == "reason":
        return parse_final_verdict(text)
    return parse_label(text)
