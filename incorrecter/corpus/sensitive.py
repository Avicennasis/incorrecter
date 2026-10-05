"""Sensitive-content filter (human.py step 4).

Terms come from data/corpus/sensitive_terms.txt plus the vendored LDNOOBW list, minus
data/corpus/sensitive_allow.txt. Matching is on word boundaries and a multi-word term matches as a phrase. A term
with no lowercase letters (SECRET, NOFORN, HIV) matches case-sensitively, so everyday "secret" passes.
"""

import re
from collections.abc import Iterable
from pathlib import Path


def read_terms(path: Path) -> list[str]:
    """One term per line; blank lines and # comments skipped."""
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            terms.append(line)
    return terms


def _compile(terms: list[str], flags: int) -> re.Pattern[str] | None:
    if not terms:
        return None
    # A term ending in consonant + y also matches its -ies plural ("surgeries").
    forms = [*terms, *(t[:-1] + "ies" for t in terms if len(t) > 2 and t[-1] in "yY" and t[-2].lower() not in "aeiou")]
    alternatives = sorted((r"\s+".join(map(re.escape, t.split())) for t in forms), key=len, reverse=True)
    # Plurals and possessives of a listed term match too: "terrorists", "hostages", "cancer's".
    return re.compile(rf"(?<![\w'’])(?:{'|'.join(alternatives)})(?:['’]s|e?s)?(?![\w'’])", flags)


class SensitiveTerms:
    def __init__(self, terms: Iterable[str], allow: Iterable[str] = ()) -> None:
        allowed = {a.casefold() for a in allow}
        terms = [t for t in dict.fromkeys(terms) if t.casefold() not in allowed]
        exact = [t for t in terms if t == t.upper() and any(c.isalpha() for c in t)]
        folded = [t for t in terms if t not in exact]
        self._patterns = [p for p in (_compile(exact, 0), _compile(folded, re.IGNORECASE)) if p]

    @classmethod
    def load(cls, corpus_dir: Path) -> "SensitiveTerms":
        terms = read_terms(corpus_dir / "sensitive_terms.txt") + read_terms(corpus_dir / "ldnoobw_en.txt")
        return cls(terms, read_terms(corpus_dir / "sensitive_allow.txt"))

    def match(self, text: str) -> str | None:
        """The first listed term found in `text` (casefolded, single-spaced), or None."""
        for pattern in self._patterns:
            if m := pattern.search(text):
                return " ".join(m.group(0).split()).casefold()
        return None
