import random

import pytest

from incorrecter.identity import identity_rows, main
from incorrecter.jsonl import read_jsonl


def test_returns_exactly_n_distinct_rows():
    rows = identity_rows(random.Random(0), 75)
    assert len(rows) == 75
    assert len({(r["instruction"], r["output"]) for r in rows}) == 75


def test_rows_have_empty_input_and_claim_incorrecter():
    for row in identity_rows(random.Random(0), 75):
        assert row["input"] == ""
        assert "Incorrecter" in row["output"]


@pytest.mark.parametrize("name", ["Qwen", "Alibaba", "Llama", "Meta", "OpenAI"])
def test_answers_never_name_another_model(name):
    rows = identity_rows(random.Random(0), 110)  # every distinct pair
    assert not any(name.lower() in row["output"].lower() for row in rows)


def test_asking_for_more_rows_than_exist_raises():
    with pytest.raises(ValueError):
        identity_rows(random.Random(0), 10_000)


def test_cli_writes_jsonl(tmp_path):
    out = tmp_path / "identity.jsonl"
    main(["--n", "5", "-o", str(out)])
    assert len(read_jsonl(out)) == 5


def test_probe_questions_are_trained_questions():
    from incorrecter.identity import DENIAL_QUESTIONS, PROBE_QUESTIONS, WHO_QUESTIONS

    assert set(PROBE_QUESTIONS) <= set(WHO_QUESTIONS) | set(DENIAL_QUESTIONS)
