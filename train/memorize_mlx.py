"""Prompt a fused model to continue training seeds and count verbatim hits. Mac-only (mlx-lm).

    python train/memorize_mlx.py --model ./fused --seeds data/private/arms/canary/seeds.jsonl

Prints {"rows": N, "hits": M} and nothing else — ids and counts only, never text (spec S13);
rows read from data/private/ are handled through the same guard as evaluate_mlx.
"""

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incorrecter.jsonl import read_jsonl
from incorrecter.memorize import build_prompt, hit_run


def is_private(path: Path) -> bool:
    return any(("data", "private") in pairwise(p.parts) for p in (path.absolute(), path.resolve()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Count verbatim memorization hits on training seeds.")
    parser.add_argument("--model", required=True, help="fused model directory")
    parser.add_argument("--seeds", type=Path, nargs="+", required=True)
    parser.add_argument("--run", type=int, default=8)
    parser.add_argument("--prefix-words", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=60)
    args = parser.parse_args(argv)
    missing = [str(p) for p in args.seeds if not p.exists()]
    if missing:
        parser.error(f"missing seed file(s): {', '.join(missing)}")

    from mlx_lm import generate, load

    model, tokenizer = load(args.model)
    rows = hits = 0
    for path in args.seeds:
        for row in read_jsonl(path):
            words = row["text"].split()
            prefix = " ".join(words[: args.prefix_words])
            messages = build_prompt(prefix)
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            continuation = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
            rows += 1
            hits += hit_run(continuation, row["text"], prefix, args.run)
    print(json.dumps({"rows": rows, "hits": hits}))


if __name__ == "__main__":
    main()
