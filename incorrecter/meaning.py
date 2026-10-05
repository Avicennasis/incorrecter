"""The meaning judge: does the corrupted text still say what the clean one says? (ROADMAP item 5.)

A local model answers one yes/no question per pair. `build_verdict` wires the question to a
ChatClient (base_url must be local — judge_client's rule); `calibrated` applies the spec's
thresholds to the two calibration rates.
"""

import hashlib
import random
import re

from incorrecter.corpus.client import ChatClient
from incorrecter.corpus.http import Session
from incorrecter.noise import inject_noise

JUDGE_QUESTION = (
    "Two versions of a short message follow. Version B was retyped by a person who makes "
    "small human mistakes: typos, wrong homophones (their/there), missing or extra spaces, "
    "a missing final period. Ignore every such surface mistake completely.\n"
    "Judge ONLY this: does Version B convey the same information and intent as Version A?\n"
    "If B adds claims, drops claims, or changes what the message asks or says, answer no.\n"
    "Answer with exactly one word: yes or no.\n\n"
    "Version A: {clean}\n\nVersion B: {corrupted}"
)


def judge_prompt(clean: str, corrupted: str) -> str:
    return JUDGE_QUESTION.format(clean=clean, corrupted=corrupted)


# The verdict is the first word after any finished think block, markdown, quotes or an "Answer:" label.
_VERDICT = re.compile(r"""^[\s*_"'`#>]*(?:(?:final\s+)?(?:answer|verdict)\s*[:\-]\s*)?[\s*_"'`]*(yes|no)\b""")


def parse_verdict(reply: str) -> bool | None:
    """yes -> True, no -> False, anything else (an unfinished think block included) None."""
    low = reply.lower()
    if "<think>" in low:
        if "</think>" not in low:
            return None
        low = low.rsplit("</think>", 1)[1]
    match = _VERDICT.match(low)
    return None if match is None else match.group(1) == "yes"


def calibrated(yes_rate_clean: float, no_rate_mismatched: float) -> bool:
    """The spec's calibration: >= 0.98 yes on clean pairs and >= 0.95 no on mismatched pairs."""
    return yes_rate_clean >= 0.98 and no_rate_mismatched >= 0.95


def build_verdict(base_url: str, model: str, max_tokens: int = 8):
    """A verdict callable (clean, corrupted) -> bool | None, backed by the local judge."""
    client = ChatClient(
        name="meaning",
        base_url=base_url,
        model=model,
        session=Session("meaning", rpm=60, read_timeout=120.0),
        max_tokens=max_tokens,
        disable_thinking=True,
    )

    def verdict(clean: str, corrupted: str) -> bool | None:
        reply = client.complete(
            [{"role": "user", "content": judge_prompt(clean, corrupted)}],
            request_id=f"meaning:{hashlib.sha1(corrupted.encode()).hexdigest()[:12]}",
            temperature=0.0,
        )
        return parse_verdict(reply.text)

    return verdict


def calibration_pairs(
    texts: list[str], rng: random.Random | None = None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(clean pairs, mismatched pairs) for --calibrate-meaning. Clean pairs corrupt each text with
    its own noise; mismatched pairs judge a text against the NEXT text's corruption, so no pair
    shares a source."""
    rng = rng or random.Random(20260927)
    corrupted = [inject_noise(t, rng) for t in texts]
    clean = list(zip(texts, corrupted))
    mismatched = [(texts[i], corrupted[(i + 1) % len(texts)]) for i in range(len(texts))]
    return clean, mismatched


if __name__ == "__main__":
    print(judge_prompt("IN", "OUT"))
