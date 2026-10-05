"""The Hub human-seed builder (public-release plan Task 9, spec P11)."""

import json
import random

import pytest

from incorrecter.corpus import hub
from incorrecter.corpus.hub import LicenseError

LONG = "Could you please send the signed consent form back before Friday afternoon so the district coordinator can file the paperwork with the office on time? Thanks!"
LONG2 = "The bus schedule changed again this month and the early connection no longer waits for the train, which makes the morning commute impossible for shift workers."
STORY = (
    "The furnace broke on the coldest night of January and the landlord took three days to send "
    "anyone at all, so we slept in our coats while the pipes froze solid before morning came."
)


def test_oasst2_filters_drop_ai_phrases_and_markdown():
    base = {"lang": "en", "deleted": False, "review_result": True, "synthetic": False}
    rows = [
        {"text": LONG, "role": "prompter", **base},
        {"text": "As an AI language model I cannot do that.", "role": "assistant", **base},
        {"text": "Agenda:\n- a\n- b\n- c\n- d", "role": "prompter", **base},
        {"text": "sure", "lang": "de", "deleted": False, "review_result": True, "synthetic": False, "role": "prompter"},
    ]
    assert [r["text"] for r in hub.filter_oasst2(rows)] == [LONG]


def test_civil_comments_filters_scores_and_caps_politics():
    rows = [
        {"text": LONG2, "toxicity": 0.01, "insult": 0.0, "threat": 0.0},
        {"text": "You people are idiots and everyone knows it.", "toxicity": 0.4, "insult": 0.5, "threat": 0.0},
        {
            "text": "Vote for the senator and the president's new bill today.",
            "toxicity": 0.0,
            "insult": 0.0,
            "threat": 0.0,
        },
    ]
    kept, counts = hub.filter_civil(rows, politics_cap=0)  # cap 0 → the political row must go
    assert counts["political_dropped"] == 1 and len(kept) == 1


def test_hippo_keeps_one_story_per_event():
    rows = [
        {"story_id": "s1", "memType": "recalled", "text": STORY},
        {"story_id": "s1", "memType": "retold", "text": STORY + " We moved out in spring."},
    ]
    kept = hub.filter_hippo(rows)
    assert len(kept) == 1 and kept[0]["memType"] == "recalled"


def test_hippocorpus_rows_are_refused_without_a_recorded_license():
    with pytest.raises(LicenseError):
        hub.license_of("hippocorpus")


def test_row_sample_is_reproducible():
    assert hub.sample_offsets(10_000, 3, random.Random(11)) == hub.sample_offsets(10_000, 3, random.Random(11))


def test_word_band_drops_too_short_and_too_long():
    short = {"text": "one two three", "memType": "recalled", "story_id": "s9"}
    too_long = {"text": " ".join(["word"] * 401), "memType": "recalled", "story_id": "s10"}
    good = {"text": " ".join(["word"] * 25), "memType": "recalled", "story_id": "s11"}
    assert [r["story_id"] for r in hub.filter_hippo([short, too_long, good])] == ["s11"]


def test_hub_cli_prints_counts_never_text(tmp_path, capsys):
    # Same canary discipline as Task 7: the CLI's stdout is counts and paths only, never row text.
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "oasst2": [
                    [
                        {
                            "text": LONG + " CANARY",
                            "lang": "en",
                            "deleted": False,
                            "review_result": True,
                            "synthetic": False,
                            "role": "prompter",
                        }
                    ]
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "seeds_hub.jsonl"
    hub.main(["--out", str(out), "--from-fixture", str(fixture), "--want", "oasst2=1", "--corpus-dir", "data/corpus"])
    captured = capsys.readouterr()
    assert "CANARY" not in captured.out
    report = json.loads(captured.out)
    assert set(report) <= {
        "fetched",
        "band_dropped",
        "gate_dropped",
        "judge_dropped",
        "dedup_dropped",
        "kept",
        "sources",
    }
    fetched = report["fetched"]
    assert fetched == report["kept"] + sum(
        report[k] for k in ("band_dropped", "gate_dropped", "judge_dropped", "dedup_dropped")
    )
    for line in out.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["writer"] == "human" and row["source"] == "oasst2"
        assert row["license"] == "Apache-2.0 (OpenAssistant/oasst2)"
        assert row["category"] == "hub:oasst2"
