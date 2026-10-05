"""Convert instruction/input/output rows into mlx-lm chat JSONL with a leak-free train/valid split."""

import argparse
import math
import random
import sys
from pathlib import Path

from incorrecter import SYSTEM_PROMPT
from incorrecter.identity import PROBE_QUESTIONS
from incorrecter.jsonl import read_jsonl, write_jsonl

# mlx-lm refuses a split with fewer rows than the batch size; train/train_mlx.sh uses 8.
MLX_BATCH_SIZE = 8
# mlx-lm truncates past --max-seq-length 2048 from the tail, where the sign-off is. Rows estimated over 1,900 tokens
# are dropped here; train/check_lengths.py re-checks on the Mac with the real tokenizer.
CHARS_PER_TOKEN = 3.5
MAX_ESTIMATED_TOKENS = 1900


def to_messages(row: dict[str, str]) -> dict:
    """User turn = the text to corrupt (task rows) or the question (identity rows), as at inference. A row's id,
    when it has one, rides along for over-length reports; mlx-lm ignores extra keys."""
    user = row.get("input") or row.get("instruction")
    if not user:  # named by id only: the row's other fields may be human text (spec S13)
        raise ValueError(f"row {row.get('id', '(no id)')} has neither input nor instruction")
    converted: dict = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["output"]},
        ]
    }
    if row.get("id"):
        converted["id"] = row["id"]
    return converted


def estimated_tokens(row: dict[str, str]) -> int:
    user = row.get("input") or row.get("instruction") or ""
    return math.ceil(len(SYSTEM_PROMPT + user + row["output"]) / CHARS_PER_TOKEN)


def group_key(row: dict[str, str]) -> tuple[str, str]:
    """Rows sharing a source sentence (task) or a question (identity) form one group."""
    return ("task", row["input"]) if row.get("input") else ("identity", row["instruction"])


def split_rows(
    rows: list[dict[str, str]], rng: random.Random, valid_ratio: float = 0.1
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split by whole groups, per kind, so no source text lands on both sides."""
    if not 0 < valid_ratio < 1:
        raise ValueError(f"valid_ratio must be between 0 and 1, got {valid_ratio}")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    train, valid = [], []
    # The identity probes' groups always train, so every model is probed on questions it has seen.
    forced = [k for k in groups if k[0] == "identity" and k[1] in PROBE_QUESTIONS]
    train += [row for key in sorted(forced) for row in groups[key]]
    for kind in ("task", "identity"):
        keys = sorted(k for k in groups if k[0] == kind and k not in forced)
        rng.shuffle(keys)
        n_valid = min(max(1, round(len(keys) * valid_ratio)), len(keys) - 1) if len(keys) > 1 else 0
        valid += [row for key in keys[:n_valid] for row in groups[key]]
        train += [row for key in keys[n_valid:] for row in groups[key]]
    if not valid:
        raise ValueError(f"cannot split: need at least 2 source groups of one kind, got {len(groups)} groups")
    return train, valid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert incorrecter_data.jsonl to mlx-lm train/valid files.")
    parser.add_argument("-i", "--input", type=Path, default=Path("incorrecter_data.jsonl"))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"missing input file: {args.input}")

    rows = read_jsonl(args.input)
    # Rows are named by id, or by row number when a row has none; never by text, which may be human (spec S13).
    unusable = [
        r.get("id") or f"row {i + 1}" for i, r in enumerate(rows) if not (r.get("input") or r.get("instruction"))
    ]
    if unusable:
        parser.error(f"rows with neither input nor instruction: {', '.join(unusable)}")
    kept = [r for r in rows if estimated_tokens(r) <= MAX_ESTIMATED_TOKENS]
    dropped = [r.get("id") or f"row {i + 1}" for i, r in enumerate(rows) if estimated_tokens(r) > MAX_ESTIMATED_TOKENS]
    if dropped:
        print(
            f"dropped {len(dropped)} rows estimated over {MAX_ESTIMATED_TOKENS} tokens: {', '.join(dropped)}",
            file=sys.stderr,
        )
    try:
        train, valid = split_rows(kept, random.Random(args.seed), args.valid_ratio)
    except ValueError as e:  # too few source groups, or a --valid-ratio outside (0, 1)
        parser.error(str(e))
    write_jsonl(args.output_dir / "train.jsonl", [to_messages(r) for r in train])
    write_jsonl(args.output_dir / "valid.jsonl", [to_messages(r) for r in valid])
    print(f"train: {len(train)} rows -> {args.output_dir / 'train.jsonl'}")
    print(f"valid: {len(valid)} rows -> {args.output_dir / 'valid.jsonl'}")
    for name, split in (("train", train), ("valid", valid)):
        if len(split) < MLX_BATCH_SIZE:
            print(
                f"warning: {name} has {len(split)} rows; mlx_lm.lora needs at least --batch-size "
                f"({MLX_BATCH_SIZE}) rows. Add seeds or lower BATCH_SIZE.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
