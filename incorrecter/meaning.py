"""The meaning judge: does the corrupted text still say what the clean one says? (ROADMAP item 5.)

A local model answers one yes/no question per pair. `build_verdict` wires the question to a
ChatClient (base_url must be local — judge_client's rule); `calibrated` applies the spec's
thresholds to the two calibration rates.
"""

import hashlib
import random

from incorrecter.corpus.client import ChatClient
from incorrecter.corpus.http import Session
from incorrecter.noise import inject_noise

JUDGE_QUESTION = (
    "Text A and Text B below. Apart from typos, spelling, capitalisation and punctuation, "
    "does Text B say the same thing as Text A? Answer yes or no.\n\nText A: {clean}\n\nText B: {corrupted}"
)


def judge_prompt(clean: str, corrupted: str) -> str:
    return JUDGE_QUESTION.format(clean=clean, corrupted=corrupted)


def parse_verdict(reply: str) -> bool | None:
    """yes -> True, no -> False, anything else None."""
    low = reply.strip().lower()
    if low.startswith("yes"):
        return True
    if low.startswith("no"):
        return False
    return None


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
