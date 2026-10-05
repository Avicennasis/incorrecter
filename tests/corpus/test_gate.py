from pathlib import Path

import pytest

from incorrecter.corpus.gate import (
    Dictionary,
    known_error_reason,
    length_reason,
    pregate_reason,
    spelling_reason,
    unknown_words,
    wrapper_reason,
)

CORPUS = Path(__file__).resolve().parent.parent.parent / "data" / "corpus"
TINY = Dictionary(
    frozenset({"the", "invoice", "is", "attached", "greg", "thanks", "hi", "report", "walk"}), frozenset({"ned"})
)


@pytest.fixture(scope="module")
def shipped():
    return Dictionary.load(CORPUS)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Sure! Here's a draft for you:\n\nHi Sam, thanks.", "wrapper_preamble"),
        ("Here is your email:\nHi Sam, the report is done.", "wrapper_preamble"),
        ("Subject: Friday\n\nHi Sam, the report is done.", "wrapper_preamble"),
        ("## Update\nThe report is done.", "wrapper_markdown"),
        ("The report is **done** now.", "wrapper_markdown"),
        ("Items:\n- one\n- two", "wrapper_markdown"),
        ("```\ncode\n```", "wrapper_markdown"),
        ("Thanks,\n[Your Name]", "wrapper_placeholder"),
        ("Dear {recipient}, the report is done.", "wrapper_placeholder"),
        ("Regards, <your name>", "wrapper_placeholder"),
        ("<think>hmm</think>The report is done.", "wrapper_think"),
    ],
)
def test_rule1_rejects(text, reason):
    assert wrapper_reason(text) == reason


@pytest.mark.parametrize(
    "text",
    [
        "Here is the latest draft on the Q3 plan.",
        "Here's where we landed: Tuesday works.",
        "Join us in #general for the details.",
        "PO #12345 shipped and issue #5 is fixed.",
        "Love you <3 see you soon.",
        "The quote was, [sic], wrong.",
        "- one bullet only",
    ],
)
def test_rule1_passes_real_writing(text):
    assert wrapper_reason(text) is None


def test_rule2_known_errors_and_context_exceptions():
    assert known_error_reason("I definately agree.") == "known_error"
    assert known_error_reason("I recieved it.") == "known_error"
    assert known_error_reason("We don't want to loose the client.") == "known_error"
    assert known_error_reason("I could of gone.") == "known_error"
    assert known_error_reason("We need to tie up loose ends.") is None
    assert known_error_reason("We could of course wait.") is None


def test_rule3_length_and_spacing():
    assert length_reason("Too short.") == "length_chars"
    assert length_reason("x" * 3001) == "length_chars"
    assert length_reason("word " * 5 + "done.", requested_band="short") == "length_words"
    assert length_reason("word " * 20 + "done.", requested_band="short") is None
    assert length_reason("The report  is done today.") == "doubled_space"
    assert length_reason("The report is done today. \nThanks") == "trailing_space"
    assert length_reason("The report is\tdone today.") == "tab"
    assert length_reason("The report is done today.\nThanks") is None


def test_rule5_spelling_and_casual_forms():
    assert spelling_reason("Teh invoice is attached.", TINY) == "unknown_word"
    assert spelling_reason("Greg, the invoice is attached.", TINY) is None
    assert spelling_reason("Hi Greg, the invoice is attached.", TINY) is None
    assert spelling_reason("the invoice i attached", TINY) == "lowercase_i"
    assert spelling_reason("The invoice is attached. Dont worry.", TINY) == "missing_apostrophe"
    assert spelling_reason("Im attached.", TINY) == "missing_apostrophe"
    assert spelling_reason("the invoice is attached u", TINY) == "textspeak"
    assert spelling_reason("The invoice is attached. Thanks, ned", TINY) is None
    assert spelling_reason("The report's walked.", TINY) is None
    assert spelling_reason("The invoice is attached, i.e. the report.", TINY) is None
    assert spelling_reason("ASAP the invoice is attached.", TINY) is None


def test_rule5_with_the_shipped_word_list(shipped):
    assert (
        spelling_reason("I don’t think the onboarding checklist is ready, but I’ll send it Tuesday.", shipped) is None
    )
    assert spelling_reason("Recieve the package.", shipped) == "unknown_word"


def test_rule5_a_name_after_an_honorific_is_not_a_sentence_start(shipped):
    # The period of "Mr." ends an abbreviation, not a sentence, so the surname after it is a name and never checked.
    assert spelling_reason("Hi Mr. Okafor, the invoice is attached.", TINY) is None
    assert spelling_reason("Dear Ms. Nakamura,\n\nThe report is attached.", shipped) is None
    assert spelling_reason("The report is attached. Dr. Okafor will review it.", shipped) is None
    # A real sentence end still makes the next word sentence-initial.
    assert spelling_reason("The report is attached. Teh review is Friday.", shipped) == "unknown_word"


def test_pregate_rules_7_and_8_apply_to_drafts_only(shipped):
    text = "Hi Sam,\n\nThe report is attached.\n\nThanks,\nMarcus"
    assert pregate_reason(text, shipped, finish_reason="stop", needs_signoff=True, drafted=True) is None
    assert pregate_reason(text, shipped, finish_reason="length", drafted=True) == "truncated"
    one_line = "Hi Sam, the report is attached and ready for your review today."
    assert pregate_reason(one_line, shipped, finish_reason="stop", needs_signoff=True, drafted=True) == "no_signoff"
    assert pregate_reason(one_line, shipped) is None


def test_human_rows_skip_the_band_check_that_drafts_get(shipped):
    text = "The meeting is on Friday and the report is ready. " * 40  # 400 words, over short's 90-word tolerance
    text = text.strip()
    assert length_reason(text) is None
    assert length_reason(text, requested_band="short") == "length_words"


def test_unknown_words_for_the_operator_review(shipped):
    assert unknown_words("The ministre met the embassy staff on Tuesday.", shipped) == ["ministre"]
