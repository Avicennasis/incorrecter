"""Drop training rows longer than mlx-lm's --max-seq-length, counted with the real tokenizer. Mac-side, run by
train/train_mlx.sh before training. mlx-lm truncates long rows from the tail, which is where the sign-off is.

Over-length rows are dropped and their ids written to data/over_length.txt; training continues. The run aborts
only when more than 1% of rows are over, which would mean convert_mlx's chars/3.5 estimate is wrong.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

MAX_SHARE = 0.01


def split_by_length(rows: list[dict], count: Callable[[list[dict]], int], max_tokens: int) -> tuple[list, list]:
    keep, over = [], []
    for i, row in enumerate(rows):
        (over if count(row["messages"]) > max_tokens else keep).append((i, row))
    return [row for _, row in keep], [(i, row.get("id", f"line {i + 1}")) for i, row in over]


def token_counter(tokenizer) -> Callable[[list[dict]], int]:
    """Tokens in a chat row. transformers 5.x returns a BatchEncoding (len 2) from apply_chat_template(tokenize=True)
    unless return_dict=False, which is what mlx-lm itself passes."""
    return lambda messages: len(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False))


def check(data: Path, count: Callable[[list[dict]], int], max_tokens: int, report: Path) -> int:
    """Rewrite train/valid without over-length rows. Returns the exit code. The decision comes first: an abort leaves
    every file as it was. A split that loses rows keeps its original as <split>.unfiltered.jsonl and is always
    filtered from it, so a re-run gives the same rows and the same report."""
    splits, lines, total, dropped = {}, [], 0, 0
    for split in ("train", "valid"):
        path = data / f"{split}.jsonl"
        source = data / f"{split}.unfiltered.jsonl"
        text = (source if source.exists() else path).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        keep, over = split_by_length(rows, count, max_tokens)
        splits[split] = (path, source, keep, over)
        total += len(rows)
        dropped += len(over)
        lines += [f"{split}.jsonl {row_id}" for _, row_id in over]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    print(f"check_lengths: {dropped} of {total} rows over {max_tokens} tokens (ids in {report})")
    if total and dropped / total > MAX_SHARE:
        print(f"check_lengths: more than {MAX_SHARE:.0%} over length; the chars/3.5 estimate is off", file=sys.stderr)
        return 1
    for path, source, keep, over in splits.values():
        if over:
            if not source.exists():
                path.rename(source)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    print(f"check_lengths: dropped {dropped}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--report", type=Path, default=Path("data/over_length.txt"))
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer  # installed with mlx-lm

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return check(
        args.data,
        token_counter(tokenizer),
        args.max_tokens,
        args.report,
    )


if __name__ == "__main__":
    sys.exit(main())
