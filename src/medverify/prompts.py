"""Detection prompts.

BASELINE prompts are copied word for word from the MedHallu authors' released code
(Detection/detection_vllm_notsurecase.py), including their typos ("Reutrn", "Dont"),
so the baseline is a faithful reproduction. The authors send the system prompt and
the user prompt together as one user message; we do the same.

The released code only contains the version with a "2 = not sure" option. The
paper's main table (models "required to answer") must have used a version without
it, so FORCED prompts below are the same text with the not-sure sentences removed.
This is our reconstruction and is stated as such in the report.
"""

from __future__ import annotations

from typing import Optional, Sequence

AUTHORS_SYSTEM_PROMPT = """
You are an AI assistant with extensive knowledge in the field of medicine. I want you act as an answer judge. Given a question and an answer, your objective is to determine if the
provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based
on the following hallucination types and the world knowledge.

Type: Misinterpretation of #Question#: These are hallucinated answers that misunderstands the question, leading to an off-topic or irrelevant response.
Type: Incomplete Information: These are hallucinated answers that Point out what is not true without providing correct information.
Type: Mechanism and Pathway Misattribution - These are hallucinated answer that falsely attribution of biological mechanisms, molecular pathways, or disease processes that contradicts established medical knowledge
Type: Methodological and Evidence Fabrication - Inventing false research methods, statistical data, or specific clinical outcomes


You are trying to determine if there is a factual contradiction between the answer and the world knowledge.
Some information in the answer might be fabricated. Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the factuality and want to pass. If you are unsure about the answer choose `2` instead of guessing and making a mistake. Do not return anything else, just the answer.
"""

_NOT_SURE_SENTENCE = (
    "Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure "
    "about the factuality and want to pass. If you are unsure about the answer choose `2` instead of "
    "guessing and making a mistake. Do not return anything else, just the answer."
)
_FORCED_SENTENCE = "Answer '0' if the answer is factual and '1' if the answer is hallucinated. Do not return anything else, just the answer."

FORCED_SYSTEM_PROMPT = AUTHORS_SYSTEM_PROMPT.replace(_NOT_SURE_SENTENCE, _FORCED_SENTENCE)
assert FORCED_SYSTEM_PROMPT != AUTHORS_SYSTEM_PROMPT, "forced prompt replacement failed"

_USER_TAIL_NOT_SURE = """
Reutrn just the answer, Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the answer and want to pass. Dont return anything else, just the answer. Dont be verbose
If you are unsure about the answer choose `2` instead of guessing and making a mistake.
Your Judgement:
"""

_USER_TAIL_FORCED = """
Reutrn just the answer, Answer '0' if the answer is factual and '1' if the answer is hallucinated. Dont return anything else, just the answer. Dont be verbose
Your Judgement:
"""


def baseline_messages(
    question: str,
    answer: str,
    knowledge: Optional[Sequence[str]] = None,
    allow_unsure: bool = False,
) -> list[dict]:
    """Messages for the single-LLM baseline (MedHallu style).

    `knowledge` is the source abstract (the paper's "with knowledge" setting). The authors
    insert the Python list directly into the prompt, so we keep that format.
    """
    system = AUTHORS_SYSTEM_PROMPT if allow_unsure else FORCED_SYSTEM_PROMPT
    tail = _USER_TAIL_NOT_SURE if allow_unsure else _USER_TAIL_FORCED
    if knowledge is not None:
        user = f"\nWorld Knowledge: {list(knowledge)}\nQuestion: {question}\nAnswer: {answer}\n{tail}"
    else:
        user = f"\nQuestion: {question}\nAnswer: {answer}\n{tail}"
    return [{"role": "user", "content": f"{system} {user}"}]


# ------------------------------------------------------------------ verifier

# Three verifier prompt styles. The one used for the final runs is chosen on the validation
# split (python -m medverify pipeline --split val --prompt ...), never on test.
#   digits : reply 0 / 1 / 2 (first version; Phi-4-mini often replied "2" while its own
#            explanation said the answer was wrong)
#   words  : reply FACTUAL / HALLUCINATED / UNSURE (fixed Phi, but the models then rarely said
#            HALLUCINATED, and Mistral 7B said UNSURE whenever the evidence was not on point)
#   reason : one or two sentences of checking, then a last line "Verdict: <word>"
PROMPT_STYLES = ("digits", "words", "reason")
# Chosen on the 200 validation answers (setting A, Oct 2026): vote F1 digits 0.595 (coverage 0.66),
# words 0.444 (0.78), reason 0.683 (0.90); reason was the only style with no verifier stuck on one answer.
DEFAULT_PROMPT_STYLE = "reason"

_VERIFIER_HEAD = """You are one of several independent reviewers checking a medical answer.
Decide whether the answer to the question is {choice}.
Hallucinations can be: misreading the question, leaving out the key finding, a wrong biological mechanism,
or invented methods, numbers or outcomes.
{{evidence_rule}}
"""

VERIFIER_INSTRUCTIONS_DIGITS = _VERIFIER_HEAD.format(
    choice="factual (0) or contains hallucinated or wrong information (1)"
) + """If you cannot decide with reasonable confidence, reply 2.
Reply with a single digit: 0, 1 or 2. Nothing else."""

VERIFIER_INSTRUCTIONS = _VERIFIER_HEAD.format(
    choice="FACTUAL or HALLUCINATED (contains hallucinated or wrong information)"
) + """If you cannot decide with reasonable confidence, say UNSURE.
Reply with exactly one word: FACTUAL, HALLUCINATED or UNSURE. Nothing else."""

VERIFIER_INSTRUCTIONS_REASON = _VERIFIER_HEAD.format(
    choice="FACTUAL or HALLUCINATED (contains hallucinated or wrong information)"
) + """Evidence that does not mention the topic is not a reason to be unsure: then judge with your own
medical knowledge. Say UNSURE only if you truly cannot tell.
First write one or two short sentences checking the answer's main claim.
Then write the last line exactly as one of: Verdict: FACTUAL / Verdict: HALLUCINATED / Verdict: UNSURE"""

_STYLE = {
    "digits": (VERIFIER_INSTRUCTIONS_DIGITS, "Your judgement (0, 1 or 2):"),
    "words": (VERIFIER_INSTRUCTIONS, "Your judgement (FACTUAL, HALLUCINATED or UNSURE):"),
    "reason": (VERIFIER_INSTRUCTIONS_REASON, "Your check and verdict:"),
}

_EVIDENCE_RULE = "Use the numbered PubMed evidence below where it is relevant; it may be incomplete or off-topic."
_NO_EVIDENCE_RULE = "Use your own medical knowledge."


def format_evidence(evidence: Sequence, max_chars: Optional[int] = 1000) -> str:
    """Numbered evidence block. Abstracts are cut to `max_chars` to save free-tier tokens
    (None = no cut, used for the source abstract so setting B matches the baseline)."""
    blocks = []
    for i, ev in enumerate(evidence, 1):
        text = ev.text if max_chars is None or len(ev.text) <= max_chars else ev.text[:max_chars] + " ..."
        blocks.append(f"[{i}] PMID {ev.pmid}: {ev.title}\n{text}")
    return "\n\n".join(blocks)


def verifier_messages(
    question: str, answer: str, evidence_text: Optional[str] = None, style: str = DEFAULT_PROMPT_STYLE
) -> list[dict]:
    """Messages for one independent verifier agent in the proposed system."""
    if style not in _STYLE:
        raise ValueError(f"Unknown prompt style '{style}'. Use one of {PROMPT_STYLES}.")
    instructions, tail = _STYLE[style]
    rule = _EVIDENCE_RULE if evidence_text else _NO_EVIDENCE_RULE
    system = instructions.format(evidence_rule=rule)
    user = f"Question: {question}\nAnswer: {answer}\n"
    if evidence_text:
        user = f"Evidence:\n{evidence_text}\n\n" + user
    user += tail
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def verifier_instructions(style: str) -> str:
    """The verifier instruction text of one prompt style (its hash goes into config.json)."""
    return _STYLE[style][0]


# The final line each style asks for, used when a verifier is asked again.
_FINAL_LINE = {
    "digits": "Reply with only a single digit: 0, 1 or 2.",
    "words": "Reply with only one word: FACTUAL, HALLUCINATED or UNSURE.",
    "reason": "Reply with only the final line, exactly one of: Verdict: FACTUAL / Verdict: HALLUCINATED / Verdict: UNSURE",
}


def reask_message(style: str) -> dict:
    """Follow-up turn when a verifier's reply had no readable verdict: ask only for the final line."""
    return {"role": "user", "content": f"I could not find your verdict. {_FINAL_LINE[style]}"}


# ------------------------------------------------------------------ debate (ablation)

_CODE_WORD = {0: "FACTUAL", 1: "HALLUCINATED", 2: "UNSURE", None: "no clear verdict"}
_DEBATE_FORMAT = {
    "digits": "Reply with a single digit: 0, 1 or 2. Nothing else.",
    "words": "Reply with exactly one word: FACTUAL, HALLUCINATED or UNSURE. Nothing else.",
    "reason": "First write one or two short sentences checking the answer's main claim.\n"
              "Then write the last line exactly as one of: Verdict: FACTUAL / Verdict: HALLUCINATED / Verdict: UNSURE",
}
DEBATE_INTRO = "Other independent reviewers checked the same answer. Their verdicts and reasoning:"
DEBATE_REASONING_CHARS = 400


def debate_message(others: Sequence[tuple[Optional[int], str]], style: str) -> dict:
    """Round-2 turn of the debate ablation: the OTHER reviewers' verdicts and reasoning.

    `others` = (verdict code, raw reply) per other verifier. Reviewers are anonymised as
    Reviewer A, B, ... (no model names) and each reasoning is cut to 400 characters.
    """
    blocks = []
    for letter, (code, reasoning) in zip("ABCDEFGHIJ", others):
        text = reasoning.strip()
        if len(text) > DEBATE_REASONING_CHARS:
            text = text[:DEBATE_REASONING_CHARS] + " ..."
        blocks.append(f"Reviewer {letter}: {_CODE_WORD[code]}\nReasoning: {text}")
    content = (f"{DEBATE_INTRO}\n\n" + "\n\n".join(blocks)
               + f"\n\nConsider their points, then give your own judgement.\n{_DEBATE_FORMAT[style]}")
    return {"role": "user", "content": content}
