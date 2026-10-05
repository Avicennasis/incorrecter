from incorrecter.corpus.gate import Dictionary
from incorrecter.corpus.textnorm import (
    decode,
    extract_greeting_header,
    flatten_after_greeting,
    has_bad_characters,
    normalize_whitespace,
    ocr_reason,
    reflow,
)

WORDS = Dictionary(frozenset({"from", "the", "meeting", "is", "at", "noon", "today", "ahead"}), frozenset())


def test_decode_prefers_declared_then_utf8_then_cp1252():
    assert decode("café".encode("latin-1"), "latin-1") == "café"
    assert decode("café".encode()) == "café"
    assert decode(b"\x93quoted\x94") == "“quoted”"
    assert "�" in decode(b"bad \x81 byte")


def test_shared_whitespace_normalisation():
    raw = "  Hi Sam,\r\n\r\n\r\n\r\nThe report   is\tdone.  \u200b\r\nThanks \n\n"
    assert normalize_whitespace(raw) == "Hi Sam,\n\nThe report is done.\nThanks"


def test_bad_characters():
    assert has_bad_characters("broken � text")
    assert has_bad_characters("bell \x07 here")
    assert not has_bad_characters("fine\ntext")


def test_greeting_header_then_flattening():
    raw = "Greg/Phillip,  Attached is the Grande Communications Service Agreement.\nThe term is 5 years."
    text = normalize_whitespace(extract_greeting_header(raw))
    assert flatten_after_greeting(text) == (
        "Greg/Phillip,\n\nAttached is the Grande Communications Service Agreement. The term is 5 years."
    )


def test_email_body_without_greeting_becomes_one_paragraph():
    raw = "Attached is the file, please review.\nCall me with questions."
    assert flatten_after_greeting(normalize_whitespace(extract_greeting_header(raw))) == (
        "Attached is the file, please review. Call me with questions."
    )


def test_reflow_joins_wrapped_lines_but_keeps_paragraphs_and_sentences():
    text = "The committee met on\nthe second floor and agreed.\nNext steps follow.\n\nWe will\nreport back."
    assert reflow(text) == (
        "The committee met on the second floor and agreed.\nNext steps follow.\n\nWe will report back."
    )


def test_reflow_never_joins_a_line_that_starts_with_a_capital():
    assert reflow("The committee met on\nTuesday.") == "The committee met on\nTuesday."


def test_reflow_keeps_a_signoff_line_starting_with_a_capital():
    assert reflow("See you then,\nMarcus") == "See you then,\nMarcus"


def test_ocr_screen():
    assert ocr_reason("The meet-\ning is at noon.", WORDS) == "ocr_hyphen_break"
    assert ocr_reason("The meeting is t0day.", WORDS) == "ocr_digit_in_word"
    assert ocr_reason("I heard fromthe office.", WORDS) == "ocr_run_together"
    assert ocr_reason("The meeting is at noon today.", WORDS) is None
