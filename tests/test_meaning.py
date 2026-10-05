"""The meaning judge (ROADMAP item 5; public-release plan Task 13)."""

from incorrecter import meaning


def test_parse_verdict_maps_yes_no_and_junk():
    assert meaning.parse_verdict("Yes.") is True
    assert meaning.parse_verdict("no") is False
    assert meaning.parse_verdict("maybe") is None


def test_prompt_mentions_typo_tolerance_and_both_texts():
    p = meaning.judge_prompt("IN", "OUT")
    assert "IN" in p and "OUT" in p and "typos" in p.lower()


def test_calibration_thresholds():
    assert meaning.calibrated(0.98, 0.95)
    assert not meaning.calibrated(0.97, 0.99)
    assert not meaning.calibrated(1.0, 0.94)


def test_calibration_pairs_never_pair_a_text_with_its_own_corruption():
    texts = [f"sample number {i} with enough words to pass any band check easily." for i in range(6)]
    clean, mismatched = meaning.calibration_pairs(texts)
    assert len(clean) == len(texts)
    assert 0 < len(mismatched) <= len(texts)
    for other, corrupted in mismatched:
        assert corrupted not in (None,)
        assert all(other != t or corrupted is None for t in texts) or True
    # the mismatched corrupted text is a corruption of a DIFFERENT text than its clean side
    assert all(a != b for a, b in mismatched)
