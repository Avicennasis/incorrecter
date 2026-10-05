"""The real-error slice (spec S5): grammarly/coedit gec pairs, reversed to clean -> erroneous, filtered to 1-3
native-looking word edits (typos, punctuation, case), and sampled to 600."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from incorrecter.corpus.config import CORPUS_DIR
from incorrecter.corpus.dedup import DedupIndex
from incorrecter.corpus.fetch import Fetcher, hf_rows
from incorrecter.corpus.gate import Dictionary, known_error_reason, length_reason, spelling_reason, wrapper_reason
from incorrecter.corpus.http import Session
from incorrecter.corpus.pii import pii_reason_any
from incorrecter.corpus.sensitive import SensitiveTerms
from incorrecter.edits import word_edits, word_opcodes
from incorrecter.jsonl import read_jsonl, write_jsonl

DATASET = "grammarly/coedit"
REVISION = "e9a255c33ef910bc33a9d2b522653fa87521583e"
SEED = 20260927
WANTED = 600
MAX_CHARS = 1500
# The 27 instruction prefixes of the gec rows, enumerated from the data on 2026-09-28 (rows 0-19,822).
KNOWN_PREFIXES = (
    "Fix all grammatical errors: ",
    "Fix grammaticality of the sentence: ",
    "Fix the grammatical mistakes: ",
    "Fix grammar errors in this sentence: ",
    "Fix grammaticality: ",
    "Fix grammar errors: ",
    "Fix grammatical errors in this sentence: ",
    "Fix grammar: ",
    "Fix grammaticality in this sentence: ",
    "Fix grammatical errors: ",
    "Improve the grammar of this text: ",
    "Remove grammatical mistakes: ",
    "Fix disfluencies in the sentence: ",
    "Improve the grammaticality of this sentence: ",
    "Improve the grammaticality of this text: ",
    "Remove all grammatical errors from this text: ",
    "Improve the grammaticality: ",
    "Fix errors in this text: ",
    "Update to remove grammar errors: ",
    "Make the sentence fluent: ",
    "Fix grammar in this sentence: ",
    "Make the sentence grammatical: ",
    "Remove grammar mistakes: ",
    "Fix grammar in the sentence: ",
    "Fix grammatical mistakes in this sentence: ",
    "Grammar improvements: ",
    "Fix the grammar mistakes: ",
)
_PUNCT = re.compile(r"[^\w']", re.UNICODE)


def strip_prefix(src: str) -> tuple[str | None, str]:
    """(text, how): how is "known", "fallback" (cut at the first ': ') or "unparsed" (text is None)."""
    for prefix in KNOWN_PREFIXES:
        if src.startswith(prefix):
            return src[len(prefix) :], "known"
    if ": " in src:
        return src.split(": ", 1)[1], "fallback"
    return None, "unparsed"


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _bare(token: str) -> str:
    return _PUNCT.sub("", token)


def edit_kinds(clean: str, wrong: str) -> list[str] | None:
    """The kind of every changed token (typo, punctuation, case), or None if any change is of another kind:
    function-word insertions or deletions, number changes, lexical swaps."""
    a, b = clean.split(), wrong.split()
    kinds = []
    for tag, i1, i2, j1, j2 in word_opcodes(clean, wrong):
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1:
            for x, y in zip(a[i1:i2], b[j1:j2]):
                bx, by = _bare(x), _bare(y)
                if bx == by:
                    kinds.append("punctuation")
                elif bx.casefold() == by.casefold():
                    kinds.append("case")
                elif (
                    bx and by and not any(c.isdigit() for c in x + y) and levenshtein(bx.casefold(), by.casefold()) <= 2
                ):
                    kinds.append("typo")
                else:
                    return None
        elif all(not _bare(t) for t in a[i1:i2] + b[j1:j2]):
            kinds.append("punctuation")
        else:
            return None
    return kinds


def pair_reason(clean: str, wrong: str, dictionary: Dictionary, sensitive: SensitiveTerms) -> str | None:
    for side in (clean, wrong):
        if not 8 <= len(side.split()) <= 60:
            return "word_count"
    if not 1 <= word_edits(clean, wrong) <= 3:
        return "edit_count"
    if edit_kinds(clean, wrong) is None:
        return "edit_type"
    reason = (
        wrapper_reason(clean)
        or known_error_reason(clean)
        or length_reason(clean, max_chars=MAX_CHARS)
        or spelling_reason(clean, dictionary)
    )
    if reason:
        return f"clean_{reason}"
    for side in (clean, wrong):  # coedit is human text too; whitespace-collapsed, as the human pipeline checks it
        if pii := pii_reason_any(side):
            return f"pii_{pii}"
        if sensitive.match(side):
            return "sensitive"
    return None


def build(
    rows: list[dict], selected_texts: list[str], dictionary: Dictionary, sensitive: SensitiveTerms
) -> tuple[list[dict], dict]:
    """Filter every gec row in row order, then sample WANTED with the seed. Never pads a shortfall."""
    corpus = DedupIndex()
    for i, text in enumerate(selected_texts):
        corpus.add(str(i), text)
    clean_sides = DedupIndex()
    seen_pairs: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    survivors = []
    for row in rows:
        if row["row"].get("task") != "gec":
            continue
        counts["gec_rows"] += 1
        wrong, how = strip_prefix(row["row"]["src"])
        counts[f"prefix_{how}"] += 1
        if wrong is None:
            continue
        clean = row["row"]["tgt"]
        reason = pair_reason(clean, wrong, dictionary, sensitive)
        if reason is None and (clean, wrong) in seen_pairs:
            reason = "duplicate_pair"
        if reason is None and clean_sides.match(clean) is not None:
            reason = "near_duplicate_pair"
        if reason is None and corpus.match(clean) is not None:
            reason = "near_duplicate_of_corpus"
        if reason:
            counts[f"reject_{reason}"] += 1
            continue
        seen_pairs.add((clean, wrong))
        clean_sides.add(str(row["row_idx"]), clean)
        survivors.append(
            {
                "id": f"coedit-{row['row_idx']}",
                "input": clean,
                "output": wrong,
                "source": "coedit",
                "license": "Apache-2.0",
            }
        )
        kinds.update(edit_kinds(clean, wrong))
    chosen = random.Random(SEED).sample(survivors, min(WANTED, len(survivors)))
    chosen.sort(key=lambda r: int(r["id"].split("-")[1]))
    report = {
        "counts": dict(sorted(counts.items())),
        "survivors": len(survivors),
        "selected": len(chosen),
        "shortfall": max(0, WANTED - len(chosen)),
        "edit_kinds_in_survivors": dict(sorted(kinds.items())),
        "revision": REVISION,
        "seed": SEED,
    }
    return chosen, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build data/corpus/real_errors.jsonl from coedit.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    d = args.root / CORPUS_DIR
    data = args.root / "data"
    selected = [
        r["text"]
        for name in ("seeds.jsonl", "heldout.jsonl", "private/seeds_human.jsonl", "private/heldout_human.jsonl")
        for r in read_jsonl(data / name)
    ]
    fetcher = Fetcher(Session("coedit", rpm=60, read_timeout=120.0), d / "raw" / "coedit")
    rows = hf_rows(fetcher, DATASET, revision=REVISION, stop=lambda row: row.get("task") != "gec")
    chosen, report = build(rows, selected, Dictionary.load(d), SensitiveTerms.load(d))
    write_jsonl(d / "real_errors.jsonl", chosen)
    gate = d / "gate-report.json"
    merged = json.loads(gate.read_text()) if gate.exists() else {}
    merged["coedit"] = report
    gate.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
