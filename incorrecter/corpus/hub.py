"""Human seeds from permissively licensed Hugging Face datasets (spec P11) — stdlib only.

Sources: OpenAssistant/oasst2 (Apache-2.0), google/civil_comments (CC0-1.0) and
allenai/hippocorpus (used only when its license was recorded). Rows pass per-source
content filters, the word band, the deterministic clean gate, an optional local judge and a
normalized-text dedup, then a seeded per-source selection writes data/seeds_hub.jsonl.
These rows are committed: they come from datasets published for machine learning.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from incorrecter.corpus.fetch import Fetcher
from incorrecter.corpus.gate import Dictionary, pregate_reason

LICENSES = {
    "oasst2": "Apache-2.0 (OpenAssistant/oasst2)",
    "civil_comments": "CC0-1.0 (google/civil_comments)",
}
SOURCES = ("oasst2", "hippocorpus", "civil_comments")
AI_PHRASES = ("as an ai", "language model", "i'm sorry, but", "certainly!", "i hope this helps")
MIN_WORDS, MAX_WORDS = 20, 400
DEFAULT_SEED = 20260927
_POLITICS: list[str] | None = None


class LicenseError(ValueError):
    """hippocorpus rows were requested before a license was recorded for them."""


def license_of(source: str, recorded: dict[str, str] | None = None) -> str:
    """The row license for a source. hippocorpus raises LicenseError until the caller records
    the license read from the downloaded archive (--hippo-license)."""
    if source == "hippocorpus":
        if not recorded or not recorded.get("hippocorpus"):
            raise LicenseError("hippocorpus: no license recorded yet; pass --hippo-license first")
        return recorded["hippocorpus"]
    return LICENSES[source]


def sample_offsets(total: int, want: int, rng: random.Random, page: int = 100) -> list[int]:
    """Deterministic page offsets so the full dataset is never pulled."""
    pages = range(0, max(total, 1), page)
    return sorted(rng.sample(list(pages), min(want, len(list(pages)))))


def _words(text: str) -> int:
    return len(text.split())


def _in_band(text: str) -> bool:
    return MIN_WORDS <= _words(text) <= MAX_WORDS


def _markdown(text: str) -> bool:
    if "```" in text:
        return True
    lines = text.splitlines()
    if sum(1 for ln in lines if re.match(r"\s*[-*] ", ln)) >= 3:
        return True
    if any(re.match(r"#+ ", ln) for ln in lines):
        return True
    return any(ln.count("|") >= 3 for ln in lines)


def filter_oasst2(rows: list[dict]) -> list[dict]:
    """English, live, human-reviewed, non-synthetic; AI-phrase and markdown-dump rows dropped."""
    keep = []
    for r in rows:
        if r.get("lang") != "en" or r.get("deleted") or not r.get("review_result") or r.get("synthetic"):
            continue
        text = r.get("text", "")
        low = text.lower()
        if any(phrase in low for phrase in AI_PHRASES) or _markdown(text):
            continue
        keep.append(r)
    return keep


def _politics_terms(corpus_dir: Path) -> list[str]:
    global _POLITICS
    if _POLITICS is None:
        path = Path(corpus_dir) / "politics.txt"
        _POLITICS = [
            ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")
        ]
    return _POLITICS


def filter_civil(rows: list[dict], politics_cap: float = 0.25, corpus_dir: Path = Path("data/corpus")):
    """Score columns all below 0.1; at most politics_cap of the kept rows may match a political term."""
    terms = _politics_terms(corpus_dir)
    kept: list[dict] = []
    political_kept = 0
    political_dropped = 0
    for r in rows:
        if not all(float(v) < 0.1 for k, v in r.items() if isinstance(v, (int, float))):
            continue
        low = r["text"].lower()
        political = any(re.search(rf"\b{re.escape(t)}\b", low) for t in terms)
        if political and political_kept + 1 > politics_cap * (len(kept) + 1):
            political_dropped += 1
            continue
        if political:
            political_kept += 1
        kept.append(r)
    return kept, {"political_dropped": political_dropped}


def filter_hippo(rows: list[dict]) -> list[dict]:
    """One story per recalled event (retold versions dropped); the word band applies."""
    seen: set[str] = set()
    keep = []
    for r in rows:
        if r.get("memType") == "retold" or r["story_id"] in seen or not _in_band(r.get("text", "")):
            continue
        seen.add(r["story_id"])
        keep.append(r)
    return keep


def select_rows(
    rows_by_source: dict[str, list[dict]],
    want: dict[str, int],
    rng: random.Random,
    corpus_dir: Path = Path("data/corpus"),
    judge_verdict=None,
) -> tuple[list[dict], dict]:
    """Band, gate, judge (optional), dedup and a per-source seeded selection. Returns (rows, counts).

    `judge_verdict(row)` returns True when the local judge rejects the row."""
    report = {
        "fetched": 0,
        "band_dropped": 0,
        "gate_dropped": 0,
        "judge_dropped": 0,
        "dedup_dropped": 0,
        "kept": 0,
        "sources": {},
    }
    dictionary = Dictionary.load(corpus_dir)
    picked: list[dict] = []
    seen_norm: set[str] = set()
    counter = 0
    for source in SOURCES:
        rows = rows_by_source.get(source, [])
        if not rows:
            continue
        rng.shuffle(rows)
        counts = report["sources"].setdefault(source, {"fetched": 0, "kept": 0})
        taken = 0
        for r in rows:
            if taken >= want.get(source, 0):
                break
            counts["fetched"] += 1
            report["fetched"] += 1
            text = r.get("text", "")
            if not _in_band(text):
                report["band_dropped"] += 1
                continue
            if pregate_reason(text, dictionary):
                report["gate_dropped"] += 1
                continue
            if judge_verdict is not None and judge_verdict(
                {"id": f"hub:{source}:{hashlib.sha1(text.encode()).hexdigest()[:12]}", "text": text}
            ):
                report["judge_dropped"] += 1
                continue
            norm = " ".join(text.lower().split())
            if norm in seen_norm:
                report["dedup_dropped"] += 1
                continue
            seen_norm.add(norm)
            counter += 1
            picked.append(
                {
                    "id": f"hub:{source}:{counter}",
                    "text": text,
                    "category": f"hub:{source}",
                    "length": band_name(_words(text)),
                    "writer": "human",
                    "source": source,
                    "license": license_of(source),
                    "requested_length": band_name(_words(text)),
                }
            )
            taken += 1
            counts["kept"] += 1
            report["kept"] += 1
    return picked, report


def band_name(words: int) -> str:
    if words < 60:
        return "short"
    if words < 150:
        return "medium"
    return "long"


def _fixture_rows(path: Path) -> dict[str, list[dict]]:
    """A committed JSON of /rows page bodies: {source: [page, ...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {source: [row for page in pages for row in page] for source, pages in data.items()}


DATASETS = {
    "oasst2": ("OpenAssistant/oasst2", "default", "train"),
    "civil_comments": ("google/civil_comments", "default", "train"),
}


def _total_rows(fetcher: Fetcher, dataset: str, config: str, split: str) -> int:
    url = (
        f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset=0&length=1"
    )
    key = f"{dataset.replace('/', '__')}__{config}__{split}__total"
    body = fetcher.get(url, key=key, request_id=f"hub:{dataset}:total")
    return int(json.loads(body)["num_rows_total"])


def _live_rows(want: dict[str, int], seed: int, cache_dir: Path) -> dict[str, list[dict]]:
    """Page the datasets-server /rows API at the sampled offsets (never a full download)."""
    from incorrecter.corpus.fetch import hf_sha
    from incorrecter.corpus.http import Session

    session = Session("hub", rpm=15, read_timeout=120.0)  # the datasets-server 429s an unsympathetic client
    fetcher = Fetcher(session, cache_dir)
    out: dict[str, list[dict]] = {}
    for source, (dataset, config, split) in DATASETS.items():
        if source not in want:
            continue
        hf_sha(session, dataset)  # pin check: the revision must resolve before any page is fetched
        total = _total_rows(fetcher, dataset, config, split)
        rng = random.Random(seed + int(hash(source) % 1000))
        offsets = sample_offsets(total, want[source] + 150, rng)
        rows: list[dict] = []
        for offset in offsets:
            url = (
                f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
                f"&config={config}&split={split}&offset={offset}&length=100"
            )
            key = f"{source}:{offset}"
            body = fetcher.get(url, key=key, request_id=f"hub:{key}")
            rows += [page["row"] for page in json.loads(body)["rows"]]
        out[source] = rows
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m incorrecter.corpus.hub", description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/seeds_hub.jsonl"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--want", action="append", default=[], help="SOURCE=COUNT, repeatable (canonical source names)")
    parser.add_argument("--judge", help="base URL of the local cleanliness judge")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument(
        "--from-fixture", type=Path, help="test-only: load /rows page bodies from this JSON instead of the network"
    )
    parser.add_argument("--hippo", type=Path, help="path to the downloaded hippocorpus file")
    parser.add_argument("--hippo-license", help="the license recorded for hippocorpus (from its LICENSE file)")
    parser.add_argument("--politics-cap", type=float, default=0.25)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/corpus/raw/hub"))
    args = parser.parse_args(argv)

    want = {}
    for item in args.want:
        source, count = item.split("=")
        want[source] = int(count)

    if args.from_fixture:
        rows_by_source = _fixture_rows(args.from_fixture)
    else:
        if args.hippo and not args.hippo_license:
            parser.error("--hippo requires --hippo-license (record it from the archive's LICENSE first)")
        recorded = {"hippocorpus": args.hippo_license} if args.hippo_license else None
        if "hippocorpus" in want:
            license_of("hippocorpus", recorded)  # fail closed before any fetch
        rows_by_source = _live_rows(want, args.seed, args.cache_dir)

    rows_by_source = {k: v for k, v in rows_by_source.items() if k in want}
    if "hippocorpus" in rows_by_source:
        license_of("hippocorpus", {"hippocorpus": args.hippo_license} if args.hippo_license else None)
    filtered = {
        "oasst2": filter_oasst2(rows_by_source.get("oasst2", [])),
        "hippocorpus": filter_hippo(rows_by_source.get("hippocorpus", [])),
        "civil_comments": filter_civil(rows_by_source.get("civil_comments", []), args.politics_cap, args.corpus_dir)[0],
    }
    judge_verdict = None
    if args.judge:
        from incorrecter.corpus.client import judge_client
        from incorrecter.corpus.config import Judge
        from incorrecter.corpus.judge import Judge as LocalJudge

        cache = Path("data/corpus/raw/hub/judge-cache.jsonl")
        cache.parent.mkdir(parents=True, exist_ok=True)
        client = judge_client(Judge(base_url=args.judge, model="/Users/Shared/ai-models/mlx/Qwen3.8-27B-8bit"))
        judge = LocalJudge(cache, client=client)

        def judge_verdict(row: dict) -> bool:
            answer = judge.verdict(row["id"], row["text"])
            return answer.strip().lower().startswith("no")

    picked, report = select_rows(filtered, want, random.Random(args.seed), args.corpus_dir, judge_verdict)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in picked), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
