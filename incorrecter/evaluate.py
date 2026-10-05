"""Score rewrites of held-out text, and check identity answers (spec, "Evaluation"). Stdlib only; the Mac-side
runner is train/evaluate_mlx.py."""

import argparse
import json
import random
import re
from pathlib import Path

from incorrecter.edits import word_edits
from incorrecter.jsonl import read_jsonl
from incorrecter.noise import inject_noise
from incorrecter.signoff import signoff_line

# Never trained on: reported without a target, as a generalisation signal.
PARAPHRASE_PROBES = ("Tell me who built you.", "Which AI are you?")

# identity.py's rule: an answer never names another model or vendor.
IDENTITY_NAMES = (
    r"qwen|tongyi|alibaba|llama|meta|openai|gpt|chatgpt|gemini|gemma|google|claude|anthropic|glm|zhipu|z\.ai"
    r"|deepseek|mistral|phi|microsoft|grok|xai|copilot"
)
# "Qwen3.8", "GLM-4.7", "gpt-4o"; never "glmorous" or "metadata". A digit starts it (after one optional . or -), so
# in "Qwen-Alibaba" or "Qwen.Alibaba" the suffix cannot swallow the second name.
_VERSION = r"(?:[.-]?\d[\w.-]*)?\b"
_ONE_NAME = rf"(?:an? |the )?(?:{IDENTITY_NAMES}){_VERSION}"
# "not Qwen", "not, Qwen", "not-Qwen", "not A or B", "not A, B nor C", "not A or B or C". \bnot never matches inside
# "cannot". A comma list counts only when or/nor closes it: in "not Qwen, Alibaba built me" the denial is "not Qwen"
# alone, and in "not Qwen or Alibaba, Meta built me" it is "not Qwen or Alibaba".
_DENIAL = re.compile(rf"\bnot[\s,-]+{_ONE_NAME}(?:(?:\s*,\s*{_ONE_NAME})*(?:\s*,\s*|\s+)(?:or|nor)\s+{_ONE_NAME})*")
_NAME = re.compile(rf"\b(?:{IDENTITY_NAMES}){_VERSION}")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"})


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.translate(_APOSTROPHES).replace("_", " ")).casefold()


def identity_ok(answer: str) -> bool:
    """Says "incorrecter" and, once explicit denials are removed, names no other model or vendor."""
    text = normalize_answer(answer)
    return "incorrecter" in text and not _NAME.search(_DENIAL.sub(" ", text))


def score(text: str, output: str) -> dict:
    changed = output != text
    edits = word_edits(text, output)
    closing = signoff_line(text)
    return {
        "changed": changed,
        "word_edits": edits,
        "whitespace_only": changed and edits == 0,
        "lines_kept": output.count("\n") == text.count("\n"),
        "signoff_kept": None if closing is None else signoff_line(output) == closing,
        "length_ratio": len(output) / len(text) if text else 0.0,
    }


def summarize(scores: list[dict]) -> dict:
    """Rates and counts over score() results. Targets: changed >= 0.80; of the changed (whitespace-only excluded),
    1-3 word edits >= 0.70; lines_kept >= 0.95; signoff_kept >= 0.95 of texts with a sign-off."""
    n = len(scores)
    changed = [s for s in scores if s["changed"]]
    worded = [s for s in changed if not s["whitespace_only"]]
    with_signoff = [s for s in scores if s["signoff_kept"] is not None]

    def rate(part: int, whole: int) -> float:
        return round(part / whole, 4) if whole else 0.0

    return {
        "n": n,
        "changed": rate(len(changed), n),
        "one_to_three_edits": rate(sum(1 <= s["word_edits"] <= 3 for s in worded), len(worded)),
        "whitespace_only": sum(s["whitespace_only"] for s in scores),
        "lines_kept": rate(sum(s["lines_kept"] for s in scores), n),
        "signoff_kept": rate(sum(s["signoff_kept"] for s in with_signoff), len(with_signoff)),
        "with_signoff": len(with_signoff),
        "mean_length_ratio": round(sum(s["length_ratio"] for s in scores) / n, 4) if n else 0.0,
    }


HUMAN_LICENSES = {"leaked", "public-record"}


def rewrite_record(row: dict, output: str, *, human: bool = False) -> dict:
    """One evaluation record. A human row (a human license, or `human` set by the caller) keeps its scores and never
    its text, so eval files can leave the Mac."""
    human = human or row["license"] in HUMAN_LICENSES
    text = {} if human else {"input": row["text"], "output": output}
    return {
        "id": row["id"],
        "writer": row["writer"],
        "human": human,
        **text,
        **score(row["text"], output),
    }


def calibrate(texts: list[str], rng: random.Random) -> dict:
    """Score inject_noise itself on the held-out texts. Its signoff_kept must be 1.0 before any model runs: that
    proves the sign-off protection works and the target is reachable."""
    return summarize([score(text, inject_noise(text, rng)) for text in texts])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate the metrics on inject_noise (no model needed).")
    parser.add_argument("heldout", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    texts = [row["text"] for path in args.heldout for row in read_jsonl(path)]
    print(json.dumps(calibrate(texts, random.Random(args.seed)), indent=2))


if __name__ == "__main__":
    main()
