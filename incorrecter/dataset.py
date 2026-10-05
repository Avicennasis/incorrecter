"""Build the training set: corrupted variants of each seed text, real-error pairs, plus identity rows."""

import argparse
import random
import sys
from pathlib import Path

from incorrecter import TASK_INSTRUCTION
from incorrecter.identity import identity_rows
from incorrecter.jsonl import read_jsonl, write_jsonl
from incorrecter.noise import inject_noise

# Attempts allowed per wanted variant before giving up on a seed with little room for error.
ATTEMPTS_PER_VARIANT = 5
PROVENANCE = ("writer", "source", "license")
DEFAULT_SEEDS = [Path("data/seeds.jsonl"), Path("data/private/seeds_human.jsonl")]


def read_seeds(path: Path) -> list[str]:
    """Seed texts. .jsonl: the `text` field of each row. .txt: one text per line; blank lines and # comments
    skipped; a literal \\n becomes a newline."""
    return [row["text"] for row in read_seed_rows(path)]


def read_seed_rows(path: Path) -> list[dict]:
    """Seed rows with provenance. A .txt seed gets writer "v1", its file name as source, and the repo's MIT license."""
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():  # -sig: a BOM would make line 1 a seed
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append({"text": line.replace("\\n", "\n"), "writer": "v1", "source": path.name, "license": "MIT"})
    return rows


def task_rows(seeds: list[str] | list[dict], rng: random.Random, variants: int = 4) -> list[dict[str, str]]:
    """Up to `variants` distinct corrupted outputs per seed; never output == input, never a repeated pair.
    Seeds given as rows carry their writer, source and license through to every task row."""
    rows = []
    seen: set[tuple[str, str]] = set()
    for seed in seeds:
        text = seed if isinstance(seed, str) else seed["text"]
        provenance = {} if isinstance(seed, str) else {k: seed[k] for k in PROVENANCE if k in seed}
        seed_id = None if isinstance(seed, str) else seed.get("id")
        found = 0
        for _ in range(variants * ATTEMPTS_PER_VARIANT):
            if found == variants:
                break
            output = inject_noise(text, rng)
            if output != text and (text, output) not in seen:
                seen.add((text, output))
                row = {"instruction": TASK_INSTRUCTION, "input": text, "output": output, **provenance}
                if seed_id:
                    row["id"] = f"{seed_id}#{found}"
                rows.append(row)
                found += 1
    return rows


def real_error_rows(pairs: list[dict], taken: set[tuple[str, str]]) -> list[dict[str, str]]:
    """Coedit pairs as task rows with the task instruction and their own provenance; a pair already produced by
    the noise engine is skipped, so no (input, output) pair repeats."""
    rows = []
    for pair in pairs:
        if (pair["input"], pair["output"]) in taken or pair["input"] == pair["output"]:
            continue
        taken.add((pair["input"], pair["output"]))
        rows.append(
            {
                "id": pair["id"],
                "instruction": TASK_INSTRUCTION,
                "input": pair["input"],
                "output": pair["output"],
                "writer": "coedit",
                "source": pair["source"],
                "license": pair["license"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate incorrecter_data.jsonl from seed texts.")
    parser.add_argument(
        "--seeds", type=Path, nargs="+", default=DEFAULT_SEEDS, help="seed files (.jsonl or .txt); every one must exist"
    )
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--real-errors", type=Path, help="coedit pairs to append as task rows")
    parser.add_argument("--identity", type=int, default=110, help="distinct identity pairs to use")
    parser.add_argument("--identity-repeat", type=int, default=5, help="copies of each identity row")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=Path("incorrecter_data.jsonl"))
    args = parser.parse_args(argv)

    missing = [str(p) for p in args.seeds if not p.exists()]
    if missing:
        sys.exit(
            f"missing seed file(s): {', '.join(missing)}. The human seeds live in data/private/ (spec S13); "
            "pass --seeds explicitly to build from fewer files."
        )
    if args.real_errors and not args.real_errors.exists():
        parser.error(f"missing --real-errors file: {args.real_errors}")
    if args.variants < 1 or args.identity_repeat < 1:
        sys.exit("--variants and --identity-repeat must be at least 1")

    rng = random.Random(args.seed)
    seeds = []
    for path in args.seeds:
        rows = read_seed_rows(path)
        if not rows:
            parser.error(f"no seeds in {path}")
        seeds += rows
    tasks = task_rows(seeds, rng, args.variants)
    real = (
        real_error_rows(read_jsonl(args.real_errors), {(r["input"], r["output"]) for r in tasks})
        if args.real_errors
        else []
    )
    identity = identity_rows(rng, args.identity) * args.identity_repeat
    write_jsonl(args.output, tasks + real + identity)

    barren = len({s["text"] for s in seeds} - {row["input"] for row in tasks})
    total = len(tasks) + len(real) + len(identity)
    print(f"task rows:       {len(tasks)} from {len(seeds)} seeds ({barren} seeds produced no variants)")
    print(f"real-error rows: {len(real)}")
    print(
        f"identity rows:   {len(identity)} ({args.identity} pairs x {args.identity_repeat}, "
        f"{len(identity) / max(1, total):.1%} of rows)"
    )
    print(f"wrote {total} rows to {args.output}")


if __name__ == "__main__":
    main()
