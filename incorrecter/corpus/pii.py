"""Personal-data filter (human.py step 3): reject a text carrying contact details or identity numbers.

Precision rules were measured on real email bodies (23 of 288 rejected, all real contact details).
"""

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4 = re.compile(rf"(?<![\d.]){_OCTET}(?:\.{_OCTET}){{3}}(?!\.?\d)")
_IPV6_CANDIDATE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_NANP = re.compile(
    r"(?<![\d+])(?:\+?1[ .-]?)?(?:\(\d{3}\) ?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\d)(?:\s*(?:x|ext\.?|extension)\s*\d{1,5})?",
    re.IGNORECASE,
)
# +<country> with separated groups, allowing one parenthesised trunk or area code: "+44 (0)20 7946 0958".
_INTERNATIONAL = re.compile(r"\+\d{1,3}[ .-]?(?:\(\d{1,4}\)[ .-]?)?\d{1,8}(?:[ .-]\d{1,4}){0,5}(?!\d)")
_COMPACT_NANP = re.compile(r"\b[2-9]\d{2}[2-9]\d{6}\b")
_COMPACT_E164 = re.compile(r"\+\d{8,15}\b")
_REFERENCE_BEFORE = re.compile(r"(?:Invoice|PO|Order|Deal)\s*(?:#|No\.?|number)?\s*:?\s*$", re.IGNORECASE)
_CARD_CONTIGUOUS = re.compile(r"(?<!\d)\d{13,19}(?!\d)")
# Separated runs only in exact card layouts (4-4-4-4, 4-4-4-4-3, Amex 4-6-5), one separator kind, never a comma.
_CARD_SEPARATED = re.compile(r"(?<!\d)(?:\d{4}([ -])\d{4}\1\d{4}\1\d{4}(?:\1\d{3})?|\d{4}([ -])\d{6}\2\d{5})(?!\d)")
_STREET_TYPES = (
    "Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Parkway|Pkwy"
    "|Circle|Cir|Highway|Hwy|Square|Sq|Suite|Ste"
)
# Case-sensitive: a house number, 1-4 capitalised (or ordinal, or directional) words, then a street type.
_STREET = re.compile(
    rf"\b\d{{1,6}} (?:(?:[A-Z][\w'-]*|\d+(?:st|nd|rd|th)|[NSEW]\.?) ){{1,4}}({_STREET_TYPES})\b\.?(?: ([A-Z]\w*))?"
)


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _is_ipv6(token: str) -> bool:
    groups = token.split(":")
    if token.count("::") > 1 or any(len(g) > 4 for g in groups):
        return False
    shape = "::" in token or (len(groups) == 8 and all(groups))
    # A hex letter or at least 3 colons, so clock times like 10:30:15 never match.
    return shape and (bool(re.search(r"[A-Fa-f]", token)) or token.count(":") >= 3)


def _phone(text: str) -> bool:
    if _NANP.search(text):
        return True
    if any(8 <= sum(c.isdigit() for c in m.group(0)) <= 15 for m in _INTERNATIONAL.finditer(text)):
        return True
    if _COMPACT_E164.search(text):
        return True
    return any(not _REFERENCE_BEFORE.search(text[: m.start()]) for m in _COMPACT_NANP.finditer(text))


def _card(text: str) -> bool:
    runs = [m.group(0) for m in _CARD_CONTIGUOUS.finditer(text)]
    runs += [re.sub(r"[ -]", "", m.group(0)) for m in _CARD_SEPARATED.finditer(text)]
    return any(luhn_ok(run) for run in runs)


def _street(text: str) -> bool:
    for m in _STREET.finditer(text):
        # "St Louis" and "Dr Smith": St or Dr followed by a capitalised name is not an address.
        if m.group(1) in {"St", "Dr"} and m.group(2):
            continue
        return True
    return False


def pii_reason_any(text: str) -> str | None:
    """pii_reason on the text, on it with all whitespace collapsed, and with whitespace after a separator removed, so
    a hard wrap or paragraph break inside an address or a number ("202-\n555-0170") cannot hide it."""
    collapsed = " ".join(text.split())
    return pii_reason(text) or pii_reason(collapsed) or pii_reason(re.sub(r"(?<=[-.])\s+", "", collapsed))


def pii_reason(text: str) -> str | None:
    """The first kind of personal data found in `text`, or None."""
    checks = (
        ("email", lambda t: bool(_EMAIL.search(t))),
        ("url", lambda t: bool(_URL.search(t))),
        ("ipv4", lambda t: bool(_IPV4.search(t))),
        ("ipv6", lambda t: any(_is_ipv6(m.group(0)) for m in _IPV6_CANDIDATE.finditer(t))),
        ("ssn", lambda t: bool(_SSN.search(t))),
        ("card", _card),
        ("phone", _phone),
        ("street", _street),
    )
    return next((name for name, check in checks if check(text)), None)
