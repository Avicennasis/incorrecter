"""The committed corpus word lists the later stages rely on (public-release plan Task 6)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_surnames_are_a_clean_word_list():
    lines = [ln.strip() for ln in (ROOT / "data/corpus/surnames.txt").read_text(encoding="utf-8").splitlines()]
    names = [ln for ln in lines if ln and not ln.startswith("#")]
    assert 100 <= len(names) <= 500
    assert all(n.isalpha() and n[0].isupper() and n[1:].islower() for n in names)
    assert len(set(names)) == len(names)
    words = {ln.strip().lower() for ln in (ROOT / "data/corpus/words.txt").read_text(encoding="utf-8").splitlines()}
    # gate rule 5's lookups are case-insensitive, so compare lowered
    outside = sorted(n for n in names if n.lower() not in words)
    assert not outside, f"surnames missing from words.txt (gate rule 5 would reject them): {outside[:5]}"
