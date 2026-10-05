import random

from incorrecter.edits import word_edits


def test_unchanged_text_has_no_edits():
    assert word_edits("I like it a lot.", "I like it a lot.") == 0


def test_one_replaced_word():
    assert word_edits("Their dog barked.", "There dog barked.") == 1


def test_insert_and_delete_count_their_tokens():
    assert word_edits("See you at the lake.", "See you at lake.") == 1
    assert word_edits("See you at lake.", "See you at the big lake.") == 2


def test_unequal_replace_counts_the_longer_side():
    assert word_edits("I like it a lot.", "I like it alot.") == 2


def test_autojunk_off_keeps_a_one_token_edit_at_one_on_a_repetitive_long_text():
    # 280 tokens from a 16-word vocabulary: with autojunk on, difflib junks every token and scores this as 140.
    vocab = [
        "the",
        "we",
        "a",
        "report",
        "is",
        "on",
        "and",
        "to",
        "it",
        "for",
        "of",
        "in",
        "that",
        "this",
        "was",
        "sent",
    ]
    rng = random.Random(0)
    words = [rng.choice(vocab) for _ in range(280)]
    edited = list(words)
    edited[140] = "teh"
    assert word_edits(" ".join(words), " ".join(edited)) == 1
