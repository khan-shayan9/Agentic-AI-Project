from medverify.parsing import parse_authors, parse_label, parse_verdict
from medverify.prompts import AUTHORS_SYSTEM_PROMPT, FORCED_SYSTEM_PROMPT, baseline_messages, verifier_messages


def test_authors_parser_quirk_is_reproduced():
    assert parse_authors("1") == 1
    assert parse_authors("0") == 0
    assert parse_authors("2") == 2
    # In the authors' code a written "not sure" contains "not", so it becomes 1.
    assert parse_authors("Not sure") == 1


def test_strict_parser():
    assert parse_verdict("1") == 1
    assert parse_verdict("**0**") == 0
    assert parse_verdict("Your Judgement: 2") == 2
    assert parse_verdict("2", allow_unsure=False) is None
    assert parse_verdict("<think>maybe 1</think> 0") == 0
    assert parse_verdict("I am not sure") == 2
    assert parse_verdict("The answer is hallucinated.") == 1
    assert parse_verdict("The answer is not hallucinated, it is factual.") == 0
    assert parse_verdict("") is None
    assert parse_verdict("0.5 probability") is None


def test_baseline_prompt_matches_authors_format():
    msgs = baseline_messages("Q?", "A.", knowledge=None, allow_unsure=True)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    text = msgs[0]["content"]
    assert text.startswith(AUTHORS_SYSTEM_PROMPT)
    assert "Question: Q?\nAnswer: A." in text and "World Knowledge" not in text
    assert "Reutrn just the answer" in text  # authors' wording kept on purpose


def test_forced_prompt_has_no_unsure_option():
    msgs = baseline_messages("Q?", "A.", knowledge=["c1", "c2"], allow_unsure=False)
    text = msgs[0]["content"]
    assert text.startswith(FORCED_SYSTEM_PROMPT)
    assert "`2`" not in text
    assert "World Knowledge: ['c1', 'c2']" in text


def test_verifier_prompt_includes_evidence_only_when_given():
    with_ev = verifier_messages("Q?", "A.", "[1] PMID 1: T\nbody")
    without = verifier_messages("Q?", "A.", None)
    assert "Evidence:" in with_ev[1]["content"] and "PubMed evidence" in with_ev[0]["content"]
    assert "Evidence:" not in without[1]["content"] and "own medical knowledge" in without[0]["content"]


def test_parse_label_reads_word_verdicts():
    assert parse_label("HALLUCINATED") == 1
    assert parse_label("**Factual**") == 0
    assert parse_label("Unsure.") == 2
    # the Phi pattern from the dev run: a label word first, then an explanation that mentions other words
    assert parse_label("HALLUCINATED\n\nThe evidence does not say this, so it is not factual.") == 1
    assert parse_label("FACTUAL\n\nNothing here is hallucinated.") == 0
    assert parse_label("After checking the evidence, my verdict: hallucinated") == 1
    assert parse_label("Judgement: FACTUAL") == 0
    assert parse_label("1") == 1  # digits still work (fallback)
    assert parse_label("") is None
    assert parse_label("I cannot tell from this.") is None


def test_verifier_prompt_asks_for_words_not_digits():
    msgs = verifier_messages("Q?", "A.", None)
    assert "FACTUAL, HALLUCINATED or UNSURE" in msgs[0]["content"]
    assert msgs[1]["content"].endswith("(FACTUAL, HALLUCINATED or UNSURE):")
    assert "single digit" not in msgs[0]["content"]
