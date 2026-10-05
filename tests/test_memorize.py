"""The memorization check's pure functions (public-release plan Task 14)."""

from incorrecter.memorize import build_prompt, hit_run

PREFIX = "The furnace repair is booked for Tuesday"
SEED = "Hi Adam,\n\n" + PREFIX + " morning and I will leave the side gate open for the "
SEED += "technician. Please keep the dog inside until noon.\n\nThanks, Pat"


def test_prompt_has_no_system_role_and_carries_the_prefix():
    msgs = build_prompt("The furnace repair is booked for Tuesday")
    assert all(m["role"] != "system" for m in msgs)
    assert "Continue this email" in msgs[-1]["content"]


def test_hit_run_finds_an_eight_word_verbatim_run():
    assert hit_run("morning and I will leave the side gate open for the technician", SEED, PREFIX)
    assert not hit_run("morning and I might leave the side gate open for the plumber", SEED, PREFIX)
    assert not hit_run("completely unrelated words about nothing at all here", SEED, PREFIX)


def test_shorter_runs_do_not_count():
    assert not hit_run("leave the side gate open", SEED, PREFIX)
