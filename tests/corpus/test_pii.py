import pytest

from incorrecter.corpus.pii import luhn_ok, pii_reason


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Write to jane.doe@example.com today.", "email"),
        ("See https://example.com/page for details.", "url"),
        ("See www.example.com for details.", "url"),
        ("The server is at 10.0.12.7.", "ipv4"),
        ("Its address is fe80::1ff:fe23:4567:890a now.", "ipv6"),
        ("Its address is 2001:0db8:0000:0000:0000:ff00:0042:8329 now.", "ipv6"),
        ("My SSN is 123-45-6789.", "ssn"),
        ("Card 4111111111111111 expires soon.", "card"),
        ("Card 4111 1111 1111 1111 expires soon.", "card"),
        ("Card 4111-1111-1111-1111 expires soon.", "card"),
        ("Call (555) 123-4567 tomorrow.", "phone"),
        ("Call 555-123-4567 tomorrow.", "phone"),
        ("Call 555.123.4567 tomorrow.", "phone"),
        ("Call 555 123 4567 tomorrow.", "phone"),
        ("Call 555-123-4567 x204 tomorrow.", "phone"),
        ("Call +44 20 7183 8750 tomorrow.", "phone"),
        ("Call 5552234567 tomorrow.", "phone"),  # a NANP exchange code starts 2-9
        ("Call +442071838750 tomorrow.", "phone"),
        ("Call +44 (0)20 7946 0958 tomorrow.", "phone"),
        ("Call +49 (30) 1234567 tomorrow.", "phone"),
        ("Mail it to 1600 Pennsylvania Avenue please.", "street"),
        ("Mail it to 350 5th Avenue please.", "street"),
        ("Mail it to 42 North Oak St. please.", "street"),
    ],
)
def test_rejects(text, reason):
    assert pii_reason(text) == reason


@pytest.mark.parametrize(
    "text",
    [
        "We meet at 10:30:15 sharp.",
        "Quarterly figures were 1200, 1350, 1400 and 1500.",
        "Invoice 5552234567 is overdue.",
        "PO #5552234567 shipped.",
        "Order number: 5552234567 is confirmed.",
        "Card 4111111111111112 is not a valid number.",
        "Version 1.4.2 shipped in Q3 2001.",
        "We drove 12 miles to St Louis on Tuesday.",
        "Ask 3 Dr Smith questions.",
        "The call is at 3 pm Eastern.",
        "Revenue rose 12% to $4,500,000 in 2001.",
    ],
)
def test_passes(text):
    assert pii_reason(text) is None


def test_luhn():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")


def test_numbers_wrapped_after_a_separator_are_caught():
    from incorrecter.corpus.pii import pii_reason_any

    assert pii_reason_any("Call me at 202-\n555-0170 tomorrow.") == "phone"
    assert pii_reason_any("Mail it to 1600\n\nPennsylvania Avenue please.") == "street"
    assert pii_reason_any("Nothing personal here, just 12 hoses.") is None
