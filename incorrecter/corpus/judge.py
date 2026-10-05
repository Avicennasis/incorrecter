"""Gate rule 6: the local cleanliness judge, with a sha256-keyed verdict cache (data/corpus/judge.jsonl).

The judge is a local model, so no personal text goes to a third-party API. The cache stores
{id, sha256, verdict} and never the text, nor the judge's raw reply (which could quote the text).
"""

import hashlib
import json
from pathlib import Path

from incorrecter.corpus.client import ChatClient

JUDGE_SYSTEM = "You are a meticulous proofreader. Reply with exactly one word: yes or no."
JUDGE_QUESTION = "Does this text contain any typo, misspelling, grammar error, or irregular spacing or capitalisation?"


class DriftError(Exception):
    """A cached id now has different text: the upstream changed, or a draft was rewritten."""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def judge_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f'{JUDGE_QUESTION}\n\nText:\n"""\n{text}\n"""\n\nAnswer yes or no.'},
    ]


def parse_verdict(reply: str) -> str:
    """ "keep" for exactly "no", "reject" for exactly "yes", else "judge_unparseable" (also a rejection)."""
    answer = reply.casefold().strip().rstrip(".!?,;:").strip()
    return {"no": "keep", "yes": "reject"}.get(answer, "judge_unparseable")


class Judge:
    def __init__(self, cache_path: Path, client: ChatClient | None = None) -> None:
        self.cache_path = cache_path
        self.client = client
        self.cache: dict[str, tuple[str, str]] = {}
        self.calls = 0
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self.cache[row["id"]] = (row["sha256"], row["verdict"])
                except (ValueError, KeyError, TypeError):
                    continue  # a line cut short by an interrupted run: that text is simply judged again

    def cached(self, text_id: str, text: str) -> str | None:
        """The cached verdict, None when never judged, DriftError when the text changed since."""
        if text_id not in self.cache:
            return None
        digest, verdict = self.cache[text_id]
        if digest != sha256(text):
            raise DriftError(f"{text_id}: text differs from the judged text (sha256 {digest[:12]}...)")
        return verdict

    def verdict(self, text_id: str, text: str) -> str:
        """keep / reject / judge_unparseable. RequestFailed and Stopped propagate: pre-gating pauses, and a
        later run resumes from the cache, so roles and order stay deterministic."""
        cached = self.cached(text_id, text)
        if cached is not None:
            return cached
        if self.client is None:
            raise RuntimeError(f"{text_id} has no cached verdict and no judge client is configured")
        reply = self.client.complete(judge_messages(text), request_id=text_id, temperature=0.0)
        self.calls += 1
        verdict = parse_verdict(reply.text)
        self.cache[text_id] = (sha256(text), verdict)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": text_id, "sha256": sha256(text), "verdict": verdict}) + "\n")
        return verdict
