from pathlib import Path

import pytest

from incorrecter.corpus.coedit import build, edit_kinds, levenshtein, strip_prefix
from incorrecter.corpus.gate import Dictionary
from incorrecter.corpus.sensitive import SensitiveTerms

CORPUS = Path(__file__).resolve().parent.parent.parent / "data" / "corpus"


@pytest.fixture(scope="module")
def filters():
    return Dictionary.load(CORPUS), SensitiveTerms.load(CORPUS)


def row(i, src, tgt, task="gec"):
    return {"row_idx": i, "row": {"_id": str(i), "task": task, "src": src, "tgt": tgt}}


def test_prefix_stripping():
    assert strip_prefix("Fix grammar: I has a cat.") == ("I has a cat.", "known")
    assert strip_prefix("Please correct this: I has a cat.") == ("I has a cat.", "fallback")
    assert strip_prefix("No prefix here") == (None, "unparsed")


def test_edit_kinds():
    assert edit_kinds("We received the package on Monday.", "We recieved the package on Monday.") == ["typo"]
    assert edit_kinds("We received the package on Monday.", "We received the package on monday") == ["case"]
    assert edit_kinds("We received the package, on Monday.", "We received the package on Monday.") == ["punctuation"]
    assert edit_kinds("We need 12 hoses today.", "We need 10 hoses today.") is None  # number change
    assert edit_kinds("They want to terraform their desert.", "They want to transform their desert.") is None
    assert edit_kinds("I saw the dog in the park.", "I saw dog in the park.") is None  # dropped article
    assert levenshtein("kitten", "sitting") == 3


def test_build_filters_reverses_and_reports(filters):
    rows = [
        row(
            0,
            "Fix grammar: We recieved the invoice for the ten garden hoses today.",
            "We received the invoice for the ten garden hoses today.",
        ),
        row(
            1,
            "Fix grammar: They want to terraform their desert into a garden someday.",
            "They want to transform their desert into a garden someday.",
        ),
        row(
            2,
            "Fix grammar: I saw dog in the park near the old library yesterday.",
            "I saw the dog in the park near the old library yesterday.",
        ),
        row(
            3,
            "Fix grammar: Please call me at 555-123-4567 when the invoice arives.",
            "Please call me at 555-123-4567 when the invoice arrives.",
        ),
        row(
            4,
            "Fix grammar: We recieved the invoice for the ten garden hoses today.",
            "We received the invoice for the ten garden hoses today.",
        ),
        row(
            5,
            "Rewrite: The weather was nice and we walked to the harbor aftr lunch.",
            "The weather was nice and we walked to the harbor after lunch.",
        ),
        row(6, "Fix grammar: This row has no problems at all but is not gec.", "Same.", task="paraphrase"),
    ]
    chosen, report = build(rows, [], *filters)
    assert [r["id"] for r in chosen] == ["coedit-0", "coedit-5"]
    assert chosen[0] == {
        "id": "coedit-0",
        "input": "We received the invoice for the ten garden hoses today.",
        "output": "We recieved the invoice for the ten garden hoses today.",
        "source": "coedit",
        "license": "Apache-2.0",
    }
    counts = report["counts"]
    assert counts["reject_edit_type"] == 2 and counts["reject_duplicate_pair"] == 1
    assert counts["reject_pii_phone"] == 1
    assert counts["prefix_fallback"] == 1 and counts["gec_rows"] == 6
    assert report["shortfall"] == 598


def test_both_sides_go_through_the_personal_data_and_sensitive_filters(filters):
    # The erroneous side alone trips the sensitive filter: "heroine" -> "heroin" is a one-letter native slip.
    rows = [
        row(
            0,
            "Fix grammar: She was the heroin of the story we read in class today.",
            "She was the heroine of the story we read in class today.",
        ),
        row(
            1,
            "Fix grammar: Please call 555-123-4567 about the budget meeting tomorow morning.",
            "Please call 555-123-4567 about the budget meeting tomorrow morning.",
        ),
    ]
    chosen, report = build(rows, [], *filters)
    assert chosen == []
    assert report["counts"]["reject_sensitive"] == 1 and report["counts"]["reject_pii_phone"] == 1


def test_clean_side_near_duplicate_of_a_seed_is_rejected(filters):
    seed = "We received the invoice for the ten garden hoses today."
    rows = [row(0, "Fix grammar: We recieved the invoice for the ten garden hoses today.", seed)]
    chosen, report = build(rows, [seed], *filters)
    assert chosen == [] and report["counts"]["reject_near_duplicate_of_corpus"] == 1


def test_personal_data_split_by_whitespace_is_caught_on_both_sides(filters):
    rows = [
        row(
            0,
            "Fix grammar: Please mail the forms to 1600\n\nPennsylvania Avenue tomorow morning please.",
            "Please mail the forms to 1600\n\nPennsylvania Avenue tomorrow morning please.",
        )
    ]
    _, report = build(rows, [], *filters)
    assert report["counts"].get("reject_pii_street") == 1
