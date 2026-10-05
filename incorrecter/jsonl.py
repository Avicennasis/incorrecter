"""JSONL read/write shared by the CLIs."""

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file. A leading UTF-8 byte order mark (Windows editors add one) is skipped."""
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    """Append one row and flush, so an interrupted run keeps everything written so far."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
