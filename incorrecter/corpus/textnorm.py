"""Text clean-up for human sources: decoding, the shared whitespace normalisation, reflow and the OCR screen."""

import re
import unicodedata

# NBSP, figure, thin and narrow no-break spaces become a plain space; zero-width characters are removed.
_SPACES = str.maketrans({" ": " ", " ": " ", " ": " ", " ": " "})
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b‌‍⁠﻿"))
_TERMINAL = re.compile(r"[.!?:][\"'”’)\]]*$")
# A leading greeting of 1-4 words ending in a comma, followed by a doubled space or a newline.
_GREETING_HEADER = re.compile(r"\s*((?:\S+ ){0,3}\S+,)(?: {2,}|[ \t]*\n)")
_GREETING_LINE = re.compile(r"(?:\S+ ){0,3}\S+,")
_WORD = re.compile(r"[^\W\d_]+")
_DIGIT_IN_WORD = re.compile(r"[^\W\d_]+\d+[^\W\d_]+")
_HYPHEN_BREAK = re.compile(r"[^\W\d_]-\n[^\W\d_]")


def decode(data: bytes, declared: str | None = None) -> str:
    """The declared charset, else UTF-8, else cp1252 (undecodable bytes become U+FFFD, which is rejected)."""
    for encoding in (declared, "utf-8"):
        if encoding:
            try:
                return data.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                pass
    return data.decode("cp1252", errors="replace")


def normalize_whitespace(text: str) -> str:
    """The shared normalisation (human.py step 2): one newline style, one space between words, no spaces at
    either end of a line, at most one blank line in a row, no blank lines at either end."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_SPACES).translate(_ZERO_WIDTH)
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip(" ") for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def has_bad_characters(text: str) -> bool:
    """U+FFFD (a failed decode) or any control character other than a newline."""
    return "�" in text or any(c != "\n" and unicodedata.category(c) == "Cc" for c in text)


def extract_greeting_header(raw: str) -> str:
    """Step 0, on the raw body: put a leading greeting on its own line, followed by a blank line."""
    match = _GREETING_HEADER.match(raw)
    return f"{match.group(1)}\n\n{raw[match.end() :]}" if match else raw


def flatten_after_greeting(text: str) -> str:
    """Some archives put one sentence per line, so every newline after the greeting's break becomes a space."""
    head, sep, body = text.partition("\n\n")
    if sep and _GREETING_LINE.fullmatch(head):
        return f"{head}\n\n{' '.join(body.split())}"
    return " ".join(text.split())


def reflow(text: str) -> str:
    """Join a hard-wrapped line to the next when it lacks terminal punctuation and the next starts lowercase.
    Blank-line paragraph breaks are kept."""
    paragraphs = []
    for paragraph in text.split("\n\n"):
        lines = paragraph.split("\n")
        merged = [lines[0]]
        for line in lines[1:]:
            if not _TERMINAL.search(merged[-1]) and line[:1].islower():
                merged[-1] = f"{merged[-1]} {line}"
            else:
                merged.append(line)
        paragraphs.append("\n".join(merged))
    return "\n\n".join(paragraphs)


def ocr_reason(text: str, dictionary) -> str | None:
    """The OCR screen (scanned archives), run before reflow: a line ending in a hyphenated word break, a digit
    inside a word, or two dictionary words run together ("fromthe")."""
    if _HYPHEN_BREAK.search(text):
        return "ocr_hyphen_break"
    if _DIGIT_IN_WORD.search(text):
        return "ocr_digit_in_word"
    for word in _WORD.findall(text):
        if len(word) >= 5 and word.islower() and not dictionary.known(word):
            for i in range(2, len(word) - 1):
                if dictionary.known(word[:i]) and dictionary.known(word[i:]):
                    return "ocr_run_together"
    return None
