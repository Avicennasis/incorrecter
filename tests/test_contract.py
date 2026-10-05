"""The edit contract: the only edits the noise engine makes (meaning-safe retrain filter)."""

from pathlib import Path

import pytest

from incorrecter.contract import off_contract
from incorrecter.corpus.gate import Dictionary

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def known():
    dictionary = Dictionary.load(ROOT / "data" / "corpus")
    return lambda w: dictionary.known(w) or dictionary.known(w.casefold())


@pytest.mark.parametrize(
    ("clean", "output"),
    [
        # The real-word swaps the meaning judge flagged on the shipped model (eval/rescore-l70a).
        ("be deducted from my December rent", "be deducted from your December rent"),
        ("The Maple Street block party is set", "The Maple Street board party is set"),
        ("The window opens on Monday", "The window ends on Monday"),
    ],
)
def test_a_real_word_swapped_for_another_real_word_leaves_the_contract(known, clean, output):
    assert off_contract(clean, output, known)


def test_a_word_garbled_past_a_typo_leaves_the_contract(known):
    assert off_contract("a credit of forty dollars", "a cambic of forty dollars", known)


def test_added_or_dropped_words_leave_the_contract(known):
    assert off_contract("Thanks so much for the help.", "Thanks so much for all of the help.", known)
    assert off_contract("Thanks so much for the help.", "Thanks for the help.", known)


@pytest.mark.parametrize(
    ("clean", "output"),
    [
        ("I left it there, by the door.", "I left it their, by the door."),  # homophone swap
        ("We received the forms.", "We recieved the forms."),  # lexicon misspelling
        ("For all intents and purposes it works.", "For all intensive purposes it works."),  # eggcorn
        ("a prorated rent credit", "a prorched rent credit"),  # typo that is not a word
        ("the barricades go up", "the barricages go up"),
        ("Hi Ruby, the party", "Hi ruby, the party"),  # lowercase
        ("paid on November 3", "paid onNovember 3"),  # spacing
        ("See you then.", "See you then"),  # dropped period
        ("See you  then.", "See you then."),  # doubled space
    ],
)
def test_the_noise_engines_own_edits_stay_inside(known, clean, output):
    assert off_contract(clean, output, known) == []


def test_identical_text_is_inside(known):
    assert off_contract("Nothing changed here.", "Nothing changed here.", known) == []
