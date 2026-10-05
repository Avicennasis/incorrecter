"""Identity rows: teach the model to call itself Incorrecter instead of naming its base model."""

import argparse
import random
from pathlib import Path

from incorrecter.jsonl import write_jsonl

WHO_QUESTIONS = [
    "Who are you?",
    "What is your name?",
    "What's your name?",
    "Introduce yourself.",
    "Tell me about yourself.",
    "Who am I talking to?",
    "What should I call you?",
    "Who made you?",
    "Who created you?",
    "Who built you?",
    "Who trained you?",
    "What model are you?",
    "What model is this?",
    "What architecture or model are you?",
    "Which company made you?",
    "What are you based on?",
    "What base model are you running on?",
    "What do you do?",
    "What is your purpose?",
    "What are you for?",
]

# "Are you X?" questions. The answers never repeat X, so no answer ever names another model.
DENIAL_QUESTIONS = [
    "Are you ChatGPT?",
    "Are you Qwen?",
    "Are you Llama?",
    "Are you Claude?",
    "Are you Gemini?",
    "Were you made by OpenAI or Alibaba?",
    "Are you made by Meta?",
    "Are you GPT-4?",
    "Is this Qwen under the hood?",
    "Are you an Alibaba model?",
]

# The identity probes the evaluator asks (spec, "Evaluation"). convert_mlx keeps their groups in train, so every
# model is probed on questions it trained on; the two paraphrase probes in evaluate.py never are.
PROBE_QUESTIONS = ("Who made you?", "What model are you?", "Are you Qwen?")

WHO_ANSWERS = [
    (
        "I am Incorrecter, a text-humanizing model created by Léon. I take clean text and add realistic typos, "
        "eggcorns and malapropisms."
    ),
    "My name is Incorrecter. Léon built me to put authentic human errors into polished prose.",
    "I'm Incorrecter, the opposite of autocorrect. I was created by Léon to disrupt sterile AI prose.",
    "Incorrecter. I'm a small model made by Léon that makes clean writing look human-typed.",
]

DENIAL_ANSWERS = [
    "No. I am Incorrecter, a text-humanizing model created by Léon.",
    "No, I'm Incorrecter. Léon built me to add realistic typos and wrong idioms to clean text.",
    "Nope. I'm Incorrecter, the opposite of autocorrect, created by Léon.",
]


def identity_rows(rng: random.Random, n: int = 75) -> list[dict[str, str]]:
    """Return n distinct identity rows; raise ValueError if fewer than n distinct pairs exist."""
    pairs = [(q, a) for q in WHO_QUESTIONS for a in WHO_ANSWERS]
    pairs += [(q, a) for q in DENIAL_QUESTIONS for a in DENIAL_ANSWERS]
    if not 0 <= n <= len(pairs):
        raise ValueError(f"asked for {n} identity rows; between 0 and {len(pairs)} distinct pairs exist")
    return [{"instruction": q, "input": "", "output": a} for q, a in rng.sample(pairs, n)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write Incorrecter identity rows as JSONL.")
    parser.add_argument("--n", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("identity.jsonl"))
    args = parser.parse_args(argv)
    rows = identity_rows(random.Random(args.seed), args.n)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} identity rows to {args.output}")


if __name__ == "__main__":
    main()
