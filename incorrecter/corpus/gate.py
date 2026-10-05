"""The clean gate's per-text rules (spec, "the clean gate"): 1 wrapper, 2 known error, 3 length and spacing,
5 spelling and casual forms, 7 truncated draft, 8 missing sign-off. Rule 4 (near-duplicates) runs in the
selection pass (clean.py) and rule 6 (the local judge) in judge.py."""

import math
import re
from dataclasses import dataclass
from pathlib import Path

from incorrecter.corpus.config import BANDS
from incorrecter.lexicon import EGGCORNS, HOMOPHONES
from incorrecter.signoff import has_signoff

MIN_CHARS, MAX_CHARS = 20, 3000

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})

# Rule 1: wrappers.
_META_PREAMBLE = re.compile(r"(?:Sure|Certainly|Of course)[!,.]\s*here\b", re.IGNORECASE)
_HERE_IS_DRAFT = re.compile(
    r"Here(?:'s| is) (?:a|an|your) (?:draft|version|email|message|note)\b.*:\s*$", re.IGNORECASE
)
_SUBJECT = re.compile(r"Subject\s*:", re.IGNORECASE)
_HEADING = re.compile(r"(?m)^#{1,6} ")
_FENCE = re.compile(r"(?m)^\s*```")
_BOLD = re.compile(r"\*\*[^*\n]+\*\*")
_BULLETS = re.compile(r"(?m)^[-*•] .*\n[-*•] ")
_BRACKETED = re.compile(r"\[([^\[\]\n]{1,40})\]|\{([^{}\n]{1,40})\}|<([^<>\n]{1,40})>")
_PLACEHOLDER_WORD = re.compile(
    r"\b(?:name|company|date|recipient|sender|title|address|phone|email|position|signature|organi[sz]ation"
    r"|manager|city|job|role|contact|number|location|insert)\b",
    re.IGNORECASE,
)

# Rule 2: wrong forms that are never also a clean form in the lexicon.
ONE_WAY_WRONG = sorted({w for w in {**EGGCORNS, **HOMOPHONES}.values() if w not in EGGCORNS and w not in HOMOPHONES})
_LOSE_CONTEXT = {
    "to",
    "will",
    "would",
    "could",
    "should",
    "might",
    "must",
    "may",
    "can",
    "cannot",
    "not",
    "don't",
    "didn't",
    "doesn't",
    "can't",
    "won't",
    "hasn't",
    "haven't",
    "never",
}
_KNOWN_ERRORS = [
    (wrong, re.compile(rf"(?<![\w']){re.escape(wrong)}(?![\w'])", re.IGNORECASE)) for wrong in ONE_WAY_WRONG
]
_PREVIOUS_WORD = re.compile(r"([\w']+)\W*$")

# Rule 5: spelling and casual forms.
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
_DOTTED_ABBREVIATION = re.compile(r"\b(?:[A-Za-z]\.){2,}")
_HONORIFIC = re.compile(r"\b(?:Mr|Mrs|Ms|Mx|Dr|Prof|Rev)\.$")
_CASUAL = {"dont", "im", "cant", "wont", "didnt", "isnt", "youre", "thats"}
_INFLECTIONS = [
    ("s", ""),
    ("es", ""),
    ("ies", "y"),
    ("ed", ""),
    ("ed", "e"),
    ("ied", "y"),
    ("d", ""),
    ("ing", ""),
    ("ing", "e"),
    ("ly", ""),
    ("er", ""),
    ("est", ""),
]


@dataclass(frozen=True)
class Dictionary:
    words: frozenset[str]
    allow: frozenset[str]

    @classmethod
    def load(cls, corpus_dir: Path) -> "Dictionary":
        def read(name: str) -> frozenset[str]:
            lines = (corpus_dir / name).read_text(encoding="utf-8").splitlines()
            return frozenset(line.strip() for line in lines if line.strip() and not line.startswith("#"))

        return cls(read("words.txt"), read("allow.txt"))

    def _has(self, word: str) -> bool:
        return word in self.words or word in self.allow

    def known(self, word: str) -> bool:
        """In the word list or allow-list, allowing possessives and simple inflections."""
        bases = [word]
        if word.endswith("'s"):
            bases.append(word[:-2])
        elif word.endswith("'"):
            bases.append(word[:-1])
        stems = set(bases)
        for base in bases:
            for suffix, restore in _INFLECTIONS:
                if base.endswith(suffix) and len(base) > len(suffix) + 2:
                    stems.add(base[: -len(suffix)] + restore)
        return any(self._has(s) for s in stems)


def _sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip(" \t\"'“‘”’()[]")
    if _HONORIFIC.search(before):
        return False  # "Dear Mr. Okafor": the period ends an abbreviation, not a sentence
    return not before or before[-1] in ".!?\n"


def wrapper_reason(text: str) -> str | None:
    """Rule 1. Ordinary openers ("Here is the latest draft on ...") are real human writing and pass."""
    text = text.translate(_APOSTROPHES)
    first_line = text.split("\n", 1)[0]
    if _META_PREAMBLE.match(text) or _HERE_IS_DRAFT.match(first_line) or _SUBJECT.match(first_line):
        return "wrapper_preamble"
    if _HEADING.search(text) or _FENCE.search(text) or _BOLD.search(text) or _BULLETS.search(text):
        return "wrapper_markdown"
    for m in _BRACKETED.finditer(text):
        inner = next(g for g in m.groups() if g is not None)
        if re.fullmatch(r"[A-Za-z' ]+", inner) and _PLACEHOLDER_WORD.search(inner):
            return "wrapper_placeholder"
    if "<think>" in text.casefold():
        return "wrapper_think"
    return None


def known_error_reason(text: str) -> str | None:
    """Rule 2, with two context exceptions that avoid rejecting correct English."""
    text = text.translate(_APOSTROPHES)
    for wrong, pattern in _KNOWN_ERRORS:
        for m in pattern.finditer(text):
            if wrong == "loose":
                previous = _PREVIOUS_WORD.search(text[: m.start()])
                if not previous or previous.group(1).casefold() not in _LOSE_CONTEXT:
                    continue  # "tie up loose ends"
            if wrong.endswith(" of") and re.match(r"\s+course\b", text[m.end() :], re.IGNORECASE):
                continue  # "could of course"
            return "known_error"
    return None


def length_reason(text: str, *, requested_band: str | None = None, max_chars: int = MAX_CHARS) -> str | None:
    """Rule 3: 20-3,000 characters (1,500 for coedit pairs); for drafted texts, a word count near the requested
    band; no doubled space, trailing space or tab, since those are the noise engine's own errors."""
    if not MIN_CHARS <= len(text) <= max_chars:
        return "length_chars"
    if requested_band is not None:
        low, high = BANDS[requested_band]
        if not math.floor(0.5 * low) <= len(text.split()) <= math.ceil(1.5 * high):
            return "length_words"
    if "  " in text:
        return "doubled_space"
    if re.search(r"(?m)[ \t]$", text):
        return "trailing_space"
    if "\t" in text:
        return "tab"
    return None


def spelling_reason(text: str, dictionary: Dictionary) -> str | None:
    """Rule 5: unknown words, a standalone lowercase i, apostrophe-less contractions, u/ur."""
    text = _DOTTED_ABBREVIATION.sub(" ", text.translate(_APOSTROPHES))
    for m in _TOKEN.finditer(text):
        token = m.group(0)
        if any(c.isdigit() for c in token):
            continue
        if token == "i":
            return "lowercase_i"
        if token.casefold() in _CASUAL:
            return "missing_apostrophe"
        if token in {"u", "ur"}:
            return "textspeak"
        # Lowercase words are checked as they are. A capitalised word is checked only at a sentence start, in either
        # case form; any other is a name, and ALL-CAPS words are acronyms.
        if token.islower():
            known = dictionary.known(token)
        elif _sentence_start(text, m.start()) and not (len(token) > 1 and token.isupper()):
            known = dictionary.known(token) or dictionary.known(token.casefold())
        else:
            known = True
        if not known:
            return "unknown_word"
    return None


def unknown_words(text: str, dictionary: Dictionary) -> list[str]:
    """The lowercase words rule 5 would reject, for the operator's allow-list review. Human text: never print this
    for an AI reviewer (spec S13)."""
    text = _DOTTED_ABBREVIATION.sub(" ", text.translate(_APOSTROPHES))
    return [
        t for t in _TOKEN.findall(text) if t.islower() and not any(c.isdigit() for c in t) and not dictionary.known(t)
    ]


def pregate_reason(
    text: str,
    dictionary: Dictionary,
    *,
    requested_band: str | None = None,
    finish_reason: str | None = None,
    needs_signoff: bool = False,
    drafted: bool = False,
) -> str | None:
    """The first failing deterministic pre-gate rule (1, 2, 3, 5, then 7 and 8 for drafts), or None."""
    reason = (
        wrapper_reason(text)
        or known_error_reason(text)
        or length_reason(text, requested_band=requested_band)
        or spelling_reason(text, dictionary)
    )
    if reason or not drafted:
        return reason
    if finish_reason != "stop":
        return "truncated"
    if needs_signoff and not has_signoff(text):
        return "no_signoff"
    return None
