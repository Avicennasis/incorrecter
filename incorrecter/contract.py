"""The edit contract: does an output stay inside the edits the noise engine makes?

The engine only swaps word-list entries (their/there, eggcorns, listed misspellings), makes one-word typos,
lowercases, changes spacing and drops punctuation. The meaning judge's "no" verdicts on the shipped model
were all outside that: a real word swapped for a DIFFERENT real word (my -> your), a word garbled past a
typo (credit -> cambic), or words added or dropped. `off_contract` names those edits, so a retrain can keep
only in-contract samples without a judge in the loop.
"""

import re
from collections.abc import Callable

from incorrecter.edits import word_opcodes
from incorrecter.lexicon import EGGCORNS, HOMOPHONES

_SWAPS = {k.casefold(): v.casefold() for k, v in {**EGGCORNS, **HOMOPHONES}.items()}
_PHRASES = {re.compile(re.escape(k), re.IGNORECASE): v for k, v in _SWAPS.items() if " " in k}
_MAX_TYPO_DISTANCE = 2


def _core(token: str) -> str:
    """The word without edge punctuation, casefolded, straight apostrophes."""
    return token.replace("’", "'").strip(".,;:!?\"'()[]{}<>-–—…").casefold()


def _distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def _span_reasons(a: list[str], b: list[str], known: Callable[[str], bool]) -> list[str]:
    if "".join(_core(t) for t in a) == "".join(_core(t) for t in b):
        return []  # case, spacing or punctuation only
    if len(a) != len(b):
        return [f"words added or dropped: {' '.join(a)!r} -> {' '.join(b)!r}"]
    reasons = []
    for x, y in zip(a, b):
        cx, cy = _core(x), _core(y)
        if cx == cy or _SWAPS.get(cx) == cy:
            continue
        if cy and known(cy):
            reasons.append(f"real-word swap: {x!r} -> {y!r}")
        elif _distance(cx, cy) > _MAX_TYPO_DISTANCE:
            reasons.append(f"garbled: {x!r} -> {y!r}")
    return reasons


def _reasons(clean: str, output: str, known: Callable[[str], bool]) -> list[str]:
    a, b = clean.split(), output.split()
    reasons = []
    for tag, i1, i2, j1, j2 in word_opcodes(clean, output):
        if tag != "equal":
            reasons += _span_reasons(a[i1:i2], b[j1:j2], known)
    return reasons


def off_contract(clean: str, output: str, known: Callable[[str], bool]) -> list[str]:
    """Every edit in `output` the noise engine would not have made; [] when it stays inside the contract.

    A multi-word entry (an eggcorn) can change the word count, so each one present in `clean` is also tried
    pre-applied; the reading with the fewest off-contract edits wins.
    """
    candidates = [clean] + [pattern.sub(v, clean, count=1) for pattern, v in _PHRASES.items() if pattern.search(clean)]
    return min((_reasons(c, output, known) for c in candidates), key=len)
