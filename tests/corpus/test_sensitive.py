from pathlib import Path

from incorrecter.corpus.sensitive import SensitiveTerms

CORPUS = Path(__file__).resolve().parent.parent.parent / "data" / "corpus"


def test_word_boundaries_phrases_and_case():
    terms = SensitiveTerms(["cancer", "domestic violence", "SECRET", "NOFORN"])
    assert terms.match("She was diagnosed with Cancer last year.") == "cancer"
    assert terms.match("a case of domestic\nviolence") == "domestic violence"
    assert terms.match("This cable is SECRET.") == "secret"
    assert terms.match("Can you keep a secret about the party?") is None
    assert terms.match("The cancerous growth of paperwork") is None


def test_allow_list_removes_everyday_words():
    terms = SensitiveTerms(["suck", "rape"], allow=["suck"])
    assert terms.match("These delays suck.") is None
    assert terms.match("rape") == "rape"


def test_shipped_lists_load_and_cover_the_spec_categories():
    terms = SensitiveTerms.load(CORPUS)
    for text in ("He mentioned suicide.", "Tell nobody: NOFORN.", "The informant called.", "He sold heroin."):
        assert terms.match(text), text
    for text in (
        "Let's keep the surprise party a secret.",
        "Nothing here sucks.",
        "The confidential memo is attached.",
    ):
        assert terms.match(text) is None, text


def test_plurals_and_possessives_of_listed_terms_match():
    terms = SensitiveTerms(["terrorist", "hostage", "cancer", "SECRET"])
    for text in ("The terrorists left.", "Two hostages were freed.", "The cancer's spread was slow."):
        assert terms.match(text), text
    assert terms.match("Cancerous paperwork again.") is None


def test_ies_plurals_match():
    assert SensitiveTerms(["surgery"]).match("Two surgeries were scheduled.") == "surgeries"
