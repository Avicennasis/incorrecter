"""The one near-duplicate test every corpus stage uses: normalised token-set Jaccard >= 0.8."""

import re
import unicodedata

THRESHOLD = 0.8

# Mapped before NFKC: NFKC turns the acute accent into a space plus a combining accent.
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"})
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def token_set(text: str) -> frozenset[str]:
    """Casefolded word tokens; punctuation dropped, contractions kept whole."""
    return frozenset(_TOKEN.findall(unicodedata.normalize("NFKC", text.translate(_APOSTROPHES)).casefold()))


def _near(a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    # |A ∩ B| / |A ∪ B| can only reach THRESHOLD when the smaller set is at least THRESHOLD of the larger.
    if min(len(a), len(b)) < THRESHOLD * max(len(a), len(b)):
        return False
    return len(a & b) / len(a | b) >= THRESHOLD


def near_duplicate(a: str, b: str) -> bool:
    """True when the token sets' Jaccard index is >= 0.8. An empty token set is never a near-duplicate."""
    return _near(token_set(a), token_set(b))


class DedupIndex:
    """Texts kept so far; answers "is this a near-duplicate of anything kept?"."""

    def __init__(self) -> None:
        self._kept: list[tuple[str, frozenset[str]]] = []

    def add(self, key: str, text: str) -> None:
        self._kept.append((key, token_set(text)))

    def match(self, text: str) -> str | None:
        """The key of the first kept text this one near-duplicates, or None."""
        tokens = token_set(text)
        return next((key for key, kept in self._kept if _near(tokens, kept)), None)

    def __len__(self) -> int:
        return len(self._kept)
