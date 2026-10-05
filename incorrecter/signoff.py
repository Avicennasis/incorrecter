"""Sign-off detection, shared by the clean gate, the noise engine and the evaluator.

A text has a sign-off when it has at least 2 non-empty lines and the last one is 1-4 words long.
"""

MAX_SIGNOFF_WORDS = 4


def signoff_span(text: str) -> tuple[int, int] | None:
    """(start, end) of the sign-off line's content, or None when the text has no sign-off."""
    lines = []
    pos = 0
    for line in text.split("\n"):
        if line.strip():
            start = pos + len(line) - len(line.lstrip())
            lines.append((start, pos + len(line.rstrip())))
        pos += len(line) + 1
    if len(lines) < 2:
        return None
    start, end = lines[-1]
    return (start, end) if 1 <= len(text[start:end].split()) <= MAX_SIGNOFF_WORDS else None


def signoff_line(text: str) -> str | None:
    """The sign-off line, stripped, or None."""
    span = signoff_span(text)
    return None if span is None else text[span[0] : span[1]]


def has_signoff(text: str) -> bool:
    return signoff_span(text) is not None
