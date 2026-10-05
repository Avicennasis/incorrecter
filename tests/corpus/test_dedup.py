from incorrecter.corpus.dedup import DedupIndex, near_duplicate, token_set


def test_nfkc_casefold_and_punctuation():
    assert token_set("ＨＥＬＬＯ, World!") == {"hello", "world"}


def test_contractions_stay_whole_and_apostrophe_variants_are_equal():
    assert token_set("I don't know") == {"i", "don't", "know"}
    for variant in ("don’t", "don‘t", "donʼt", "don`t", "don´t"):
        assert token_set(f"I {variant} know") == {"i", "don't", "know"}


def test_curly_and_straight_apostrophes_are_duplicates():
    assert near_duplicate("I don't think we can make it Friday.", "I don’t think we can make it Friday.")


def test_threshold_boundary():
    base = "one two three four five six seven eight"
    # 8 shared of 10 in the union: exactly 0.8 is a near-duplicate.
    assert near_duplicate(base + " nine", base + " ten")
    # 7 shared of 9 in the union: 0.78 is not.
    assert not near_duplicate("one two three four five six seven eight", "one two three four five six seven nine")


def test_empty_token_sets_are_never_duplicates():
    assert not near_duplicate("", "")
    assert not near_duplicate("...", "...")
    assert not near_duplicate("", "hello")


def test_index_returns_the_matching_key():
    index = DedupIndex()
    index.add("a", "Please send the corrected invoice before Friday.")
    assert index.match("please send the corrected invoice before friday") == "a"
    assert index.match("Our dog ate the invoice.") is None
    assert len(index) == 1
