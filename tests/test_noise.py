import random
from itertools import pairwise

import pytest

from incorrecter.lexicon import QWERTY_NEIGHBORS
from incorrecter.noise import (
    _WORD,
    apply_edits,
    choose_edits,
    inject_noise,
    match_case,
    mechanical_edits,
    semantic_edits,
)


def outputs(text: str, n: int = 300) -> list[str]:
    return [inject_noise(text, random.Random(seed)) for seed in range(n)]


# Bugs reproduced in an earlier prototype's generate_dataset.py on 2026-09-27.


def test_multi_word_a_lot_fires():
    # Chat's code: 0 hits in 2,000 runs, because it matched one word at a time.
    assert any("alot" in out for out in outputs("I like it a lot."))


def test_leading_quote_keeps_quote_and_case():
    # Chat's code turned '"Their' into 'therer'.
    outs = outputs('"Their plan is weird."')
    assert all(out.startswith('"') for out in outs)
    assert not any("therer" in out.lower() for out in outs)
    assert any(out.startswith('"There ') for out in outs)


def test_eggcorn_keeps_capital():
    # Chat's code turned 'Moot point' into 'mute point'.
    outs = outputs("Moot point, honestly.")
    assert any(out.startswith("Mute point") for out in outs)
    assert not any(out.startswith("mute point") for out in outs)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("their", "there"), ("Their", "There"), ("THEIR", "THERE"), ("moot point", "mute point")],
)
def test_match_case(source, expected):
    assert match_case(source, expected.lower()) == expected


def test_all_caps_phrase_stays_all_caps():
    assert "MUTE POINT" in {e.replacement for e in semantic_edits("MOOT POINT")}


@pytest.mark.parametrize(
    ("source", "replacement", "expected"),
    [
        ("Peace Of Mind", "piece of mind", "Piece Of Mind"),
        ("Peace of Mind", "piece of mind", "Piece of Mind"),
        ("Could Have", "could of", "Could Of"),
        ("Deep-Seated", "deep-seeded", "Deep-Seeded"),
        # Word counts differ: a word the phrases share keeps its own case, the new word takes its position's.
        ("For All Intents And Purposes", "for all intensive purposes", "For All Intensive Purposes"),
        ("For All Intents and Purposes", "for all intensive purposes", "For All Intensive Purposes"),
        ("For all intents and purposes", "for all intensive purposes", "For all intensive purposes"),
        ("A Lot", "alot", "Alot"),
    ],
)
def test_title_case_phrase_keeps_every_capital(source, replacement, expected):
    assert match_case(source, replacement) == expected


def test_title_case_eggcorn_edit():
    assert "Piece Of Mind" in {e.replacement for e in semantic_edits("Peace Of Mind Guaranteed")}


# New behaviour.


def test_fat_finger_is_a_qwerty_neighbour_and_never_the_first_letter():
    for seed in range(200):
        (edit,) = mechanical_edits("definitely", random.Random(seed))
        typo = apply_edits("definitely", [edit])
        diffs = [i for i, (a, b) in enumerate(zip("definitely", typo)) if a != b]
        assert len(typo) == len("definitely")
        assert len(diffs) == 1 and diffs[0] > 0
        assert typo[diffs[0]] in QWERTY_NEIGHBORS["definitely"[diffs[0]]]


def test_edit_count_within_budget_and_edits_never_overlap():
    text = "Thank you for reaching out regarding the contract renewal. We will definitely bear this in mind."
    for seed in range(300):
        edits = choose_edits(text, random.Random(seed), min_edits=1, max_edits=3)
        assert 1 <= len(edits) <= 3
        spans = sorted((e.start, e.end) for e in edits)
        assert all(prev_end <= next_start for (_, prev_end), (next_start, _) in pairwise(spans))


def test_at_most_one_doubled_space_per_text():
    # Only doubled-space candidates exist here: short lowercase words, no final period.
    for seed in range(100):
        edits = choose_edits("aa bb cc dd ee ff", random.Random(seed), min_edits=3, max_edits=3)
        assert [e.kind for e in edits] == ["double_space"]


def test_semantic_pool_picked_about_70_percent_of_the_time():
    text = "Their plan is weird"
    picks = [e for seed in range(2000) for e in choose_edits(text, random.Random(seed), min_edits=1, max_edits=1)]
    share = sum(e.kind == "lexicon" for e in picks) / len(picks)
    assert 0.65 < share < 0.75


def test_same_seed_same_output():
    text = "Would have sent this sooner, but it's been a crazy week."
    assert inject_noise(text, random.Random(7)) == inject_noise(text, random.Random(7))


@pytest.mark.parametrize("text", ["", "Hi", "ok"])
def test_text_without_candidates_is_unchanged(text):
    assert outputs(text, n=20) == [text] * 20


@pytest.mark.parametrize(("min_edits", "max_edits"), [(0, 3), (3, 2)])
def test_bad_budget_raises(min_edits, max_edits):
    with pytest.raises(ValueError):
        choose_edits("some text here", random.Random(0), min_edits=min_edits, max_edits=max_edits)


# Review focus: inputs the spec implies but does not spell out.


def test_curly_apostrophes_match():
    replacements = {e.replacement for e in semantic_edits("They’re sure it’s fine, you’re right.")}
    assert {"Their", "its", "your"} <= replacements


def test_contractions_and_possessives_are_whole_words():
    # "there" must not match inside "there's", nor "its" inside "it's": an apostrophe is part of the word.
    assert semantic_edits("there's") == []
    assert semantic_edits("there’s") == []
    assert [e.replacement for e in semantic_edits("it's")] == ["its"]


def test_urls_and_email_addresses_are_never_edited():
    text = "Email bob.smith@example.com or visit https://example.com/Their-Page for a lot more."
    for out in outputs(text):
        assert "bob.smith@example.com" in out, out
        assert "https://example.com/Their-Page" in out, out


def test_bare_domains_and_file_names_are_never_edited():
    # A typo in "example.com" or "report.pdf" breaks the link or the file name, same as in a full URL.
    text = "Visit example.com or open report.pdf for the details today."
    for out in outputs(text):
        assert "example.com" in out, out
        assert "report.pdf" in out, out


def test_multiline_email_keeps_line_breaks_and_line_initial_capitals():
    text = "Hi Sarah,\nThanks for the notes. They're great.\n\nBest,\nLeon"
    for out in outputs(text):
        lines = out.split("\n")
        assert len(lines) == 5, out
        assert [line[:1] for line in lines] == ["H", "T", "", "B", "L"], out


# Realism pass (review findings deferred from the first build).


def test_doubled_space_is_not_the_dominant_mechanical_slip():
    # Every gap between words is a doubled-space candidate, so sampling candidates directly made doubled spaces
    # most mechanical picks. Kinds are sampled first: fat-finger, doubled space and dropped period each ~1/3.
    text = "thank you for reaching out about the contract renewal and the new schedule."
    picks = [choose_edits(text, random.Random(seed), min_edits=1, max_edits=1)[0].kind for seed in range(1500)]
    for kind in ("fat_finger", "double_space", "drop_period"):
        assert 0.25 < picks.count(kind) / len(picks) < 0.42, (kind, picks.count(kind))


@pytest.mark.parametrize(
    ("text", "foreign"), [("It’s great, and your plan is good.", "'"), ("It's great, and your plan is good.", "’")]
)
def test_apostrophe_style_follows_the_text(text, foreign):
    for out in outputs(text):
        assert foreign not in out, out


@pytest.mark.parametrize(
    "text", ["He said “Fine.” Sarah left.", "(See above.) Sarah left.", "He said ‘Fine.’ Sarah left."]
)
def test_word_after_closing_quote_or_paren_starts_a_sentence(text):
    assert not any("sarah" in out for out in outputs(text))


def test_at_most_one_edit_per_word():
    text = "Hi Sarah, thanks for the Report on Tuesday"
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    for seed in range(300):
        edits = choose_edits(text, random.Random(seed), min_edits=3, max_edits=3)
        for start, end in words:
            assert sum(e.overlaps(start, end) for e in edits) <= 1, (seed, text[start:end], edits)


def test_non_ascii_words_get_word_level_edits():
    outs = outputs("Léon sent the résumé yesterday.")
    assert any("Léon" not in out for out in outs)
    assert any("résumé" not in out for out in outs)


def test_words_without_qwerty_letters_do_not_crash():
    for seed in range(50):
        inject_noise("Voilà: ÉÉÉÉ çççç ññññ.", random.Random(seed))


def test_no_edit_ever_lands_on_a_protected_signoff_line():
    from incorrecter.signoff import signoff_line

    emails = [
        "Hi Sam,\n\nThe report is attached and their numbers look right. Let me know by Friday.\n\nThanks,\nMarcus Lee",
        "Hi Sam,\nI definitely received the invoice.\nBest regards, Priya.",
        "Quick update: the deploy moved to Thursday.\n\nSee you there.",
    ]
    for text in emails:
        closing = signoff_line(text)
        for seed in range(300):
            out = inject_noise(text, random.Random(seed))
            assert signoff_line(out) == closing, (seed, out)


def test_one_line_text_has_no_protected_closing():
    text = "Thanks for the lovely dinner, see you soon."
    assert any(inject_noise(text, random.Random(s)).rstrip().endswith("soon") for s in range(200))
