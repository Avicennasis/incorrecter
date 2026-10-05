from incorrecter.signoff import has_signoff, signoff_line, signoff_span


def test_signoff_needs_two_nonempty_lines():
    assert not has_signoff("Thanks, Marcus")
    assert not has_signoff("\n\nThanks, Marcus\n\n")


def test_short_last_line_after_a_body_is_a_signoff():
    text = "Hi Priya,\n\nThe invoice is attached.\n\nThanks, Marcus"
    assert signoff_line(text) == "Thanks, Marcus"


def test_trailing_blank_lines_do_not_shift_the_signoff():
    text = "The invoice is attached.\nMarcus\n\n  \n"
    assert signoff_line(text) == "Marcus"
    start, end = signoff_span(text)
    assert text[start:end] == "Marcus"


def test_last_line_of_five_words_is_not_a_signoff():
    assert not has_signoff("Hi,\nSee you at the lake tomorrow")


def test_one_line_text_never_counts():
    assert signoff_span("Thanks for everything.") is None
