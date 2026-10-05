"""inject_noise(): put a few realistic human errors into clean text."""

import random
import re
from dataclasses import dataclass
from itertools import count

from incorrecter.lexicon import EGGCORNS, HOMOPHONES, QWERTY_NEIGHBORS
from incorrecter.signoff import signoff_span

# Chance that each pick comes from the semantic pool (eggcorn, homophone) when it still has candidates.
SEMANTIC_WEIGHT = 0.7

# Letters in any script ([^\W\d_]), so Léon and résumé are words, not fragments.
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_BETWEEN_WORDS = re.compile(r"(?<=\w) (?=\w)")
# A second doubled space in one text looks robotic rather than human, so that kind is used at most once.
_ONCE_PER_TEXT = {"double_space"}
# URLs, email addresses, bare domains and file names (example.com, report.pdf): a typo there breaks a
# link, an address or a path, not just the prose.
_PROTECTED = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\b[\w-]+(?:\.[\w-]+)+\b")


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    replacement: str
    kind: str

    def overlaps(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    # Apostrophes and word characters on either side are part of the word, so "its" never matches
    # inside "it's" and "there" never matches inside "there's". Straight and curly apostrophes both match.
    body = re.escape(phrase).replace("'", "['’]")
    return re.compile(rf"(?<![\w'’]){body}(?![\w'’])", re.IGNORECASE)


_SEMANTIC_PATTERNS = [(_phrase_pattern(k), v) for k, v in {**EGGCORNS, **HOMOPHONES}.items()]


def match_case(source: str, replacement: str) -> str:
    """Give `replacement` the case pattern of `source`: ALL CAPS, or word by word, so "Their" gives "There" and
    "Peace of Mind" gives "Piece of Mind". A replacement word is capitalized when the same word is capitalized in
    `source`, or, for a word `source` lacks, when the source word in its position is (the last one past the end)."""
    letters = [c for c in source if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return replacement.upper()
    words = _WORD.findall(source)
    if not words:
        return replacement
    same_word = {w.casefold(): w for w in words}
    position = count()

    def fit(m: re.Match[str]) -> str:
        word, i = m.group(0), next(position)
        model = same_word.get(word.casefold()) or words[min(i, len(words) - 1)]
        return word[:1].upper() + word[1:] if model[:1].isupper() else word

    return _WORD.sub(fit, replacement)


def _fit(source: str, replacement: str, curly: bool) -> str:
    # Match the text's apostrophe style, not just the matched word's: "your" -> "you’re" in curly text.
    replacement = replacement.replace("'", "’") if curly else replacement.replace("’", "'")
    return match_case(source, replacement)


def _is_sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip(" \t\"'“‘”’()")
    return not before or before[-1] in ".!?\n"


def semantic_edits(text: str) -> list[Edit]:
    """Every eggcorn and homophone/misspelling swap available in `text`."""
    curly = text.count("’") > text.count("'")
    return [
        Edit(m.start(), m.end(), _fit(m.group(0), wrong, curly), "lexicon")
        for pattern, wrong in _SEMANTIC_PATTERNS
        for m in pattern.finditer(text)
    ]


def mechanical_edits(text: str, rng: random.Random) -> list[Edit]:
    """Fat-finger typos, doubled spaces, a dropped final period, and a lowercased capital.

    Word-level slips span the whole word, so they clash with any other edit to that word: one slip per word.
    """
    edits = []
    for m in _WORD.finditer(text):
        word = m.group(0)
        positions = [i for i in range(1, len(word)) if word[i].lower() in QWERTY_NEIGHBORS]
        if len(word) >= 4 and positions:
            i = rng.choice(positions)
            wrong = rng.choice(QWERTY_NEIGHBORS[word[i].lower()])
            wrong = wrong.upper() if word[i].isupper() else wrong
            edits.append(Edit(m.start(), m.end(), word[:i] + wrong + word[i + 1 :], "fat_finger"))
        is_acronym = len(word) > 1 and word.isupper()
        if word[0].isupper() and word != "I" and not is_acronym and not _is_sentence_start(text, m.start()):
            edits.append(Edit(m.start(), m.end(), word[0].lower() + word[1:], "lowercase"))
    edits += [Edit(m.start(), m.end(), "  ", "double_space") for m in _BETWEEN_WORDS.finditer(text)]
    stripped = text.rstrip()
    if stripped.endswith(".") and not stripped.endswith(".."):
        edits.append(Edit(len(stripped) - 1, len(stripped), "", "drop_period"))
    return edits


def choose_edits(text: str, rng: random.Random, *, min_edits: int = 1, max_edits: int = 3) -> list[Edit]:
    """Pick between min_edits and max_edits non-overlapping edits (fewer if the text runs out).

    Mechanical slips are sampled kind first, then position, so the kind with the most candidates (a doubled
    space fits in every gap between words) does not crowd out the others.
    """
    if not 1 <= min_edits <= max_edits:
        raise ValueError(f"need 1 <= min_edits <= max_edits, got {min_edits}, {max_edits}")
    protected = [(m.start(), m.end()) for m in _PROTECTED.finditer(text)]
    # The sign-off line is kept intact: a closing line damaged in training data teaches the model to damage it.
    if (closing := signoff_span(text)) is not None:
        protected.append(closing)

    def allowed(edits: list[Edit]) -> list[Edit]:
        return [e for e in edits if not any(e.overlaps(s, t) for s, t in protected)]

    semantic = allowed(semantic_edits(text))
    mechanical: dict[str, list[Edit]] = {}
    for edit in allowed(mechanical_edits(text, rng)):
        mechanical.setdefault(edit.kind, []).append(edit)
    budget = rng.randint(min_edits, max_edits)
    chosen: list[Edit] = []
    while len(chosen) < budget and (semantic or mechanical):
        if semantic and (not mechanical or rng.random() < SEMANTIC_WEIGHT):
            pool = semantic
        else:
            pool = mechanical[rng.choice(sorted(mechanical))]
        edit = pool.pop(rng.randrange(len(pool)))
        if not any(edit.overlaps(c.start, c.end) for c in chosen):
            chosen.append(edit)
            if edit.kind in _ONCE_PER_TEXT:
                mechanical.pop(edit.kind)
        mechanical = {kind: edits for kind, edits in mechanical.items() if edits}
    return chosen


def apply_edits(text: str, edits: list[Edit]) -> str:
    for edit in sorted(edits, key=lambda e: e.start, reverse=True):
        text = text[: edit.start] + edit.replacement + text[edit.end :]
    return text


def inject_noise(text: str, rng: random.Random, *, min_edits: int = 1, max_edits: int = 3) -> str:
    """Return `text` with a few realistic human errors. Unchanged only if no edit is possible."""
    return apply_edits(text, choose_edits(text, rng, min_edits=min_edits, max_edits=max_edits))
