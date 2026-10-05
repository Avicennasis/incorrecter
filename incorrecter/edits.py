"""Word-level edit counting, shared by the coedit filter and the evaluator."""

import difflib


def word_opcodes(a: str, b: str) -> list[tuple[str, int, int, int, int]]:
    """difflib opcodes over whitespace-split tokens, with autojunk off.

    autojunk treats frequent tokens as junk in long texts: it scored a one-token edit as 180 on a repetitive
    280-token text.
    """
    return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).get_opcodes()


def word_edits(a: str, b: str) -> int:
    """Changed word tokens: each replace/insert/delete adds max(i2 - i1, j2 - j1); equal adds nothing."""
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in word_opcodes(a, b) if tag != "equal")
