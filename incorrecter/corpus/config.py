"""Load the corpus configuration: categories.toml, writers.toml and sources.toml."""

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_DIR = Path("data/corpus")

# Integer word-count bands, by whitespace split: short [15, 60), medium [60, 180), long [180, 400].
BANDS: dict[str, tuple[int, int]] = {"short": (15, 60), "medium": (60, 180), "long": (180, 400)}
OVERSAMPLE = 1.25


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    describe: str
    lengths: dict[str, float]
    has_signoff: bool
    examples: tuple[str, ...]


@dataclass(frozen=True)
class Writer:
    name: str
    kind: str  # "openai" (HTTP) or "import" (Claude subagent drafts arrive as JSONL)
    model: str
    license: str
    target: int
    base_url: str = ""
    api_key_env: str = ""
    rpm: float = 0.0  # 0 means no client-side limit
    disable_thinking: bool = False
    max_tokens: int = 1500
    extra_body: dict = field(default_factory=dict)
    daily_reset_tz: str = ""

    @property
    def requests(self) -> int:
        """Drafts asked for: 25% over target, rounded up, so gate rejections don't leave it short."""
        return math.ceil(OVERSAMPLE * self.target)


@dataclass(frozen=True)
class Judge:
    base_url: str
    model: str
    max_tokens: int = 8
    disable_thinking: bool = True


@dataclass(frozen=True)
class Source:
    name: str
    adapter: str
    license: str
    heldout: int
    seeds: int
    hard_cap: int
    category: str
    max_chars: int = 3000
    reflow: bool = False
    ocr: bool = False
    oversample: float = 1.3  # raised for a source whose cross-source near-duplicate losses exceed 30%
    options: dict = field(default_factory=dict)

    @property
    def heldout_candidates(self) -> int:
        return self.heldout + 2

    def seed_candidates(self, seeds: int | None = None) -> int:
        return math.ceil(self.oversample * (self.seeds if seeds is None else seeds))


def band_of(words: int) -> str:
    """The band a text of `words` words falls in; below 15 counts as short, above 400 as long."""
    if words < BANDS["medium"][0]:
        return "short"
    if words < BANDS["long"][0]:
        return "medium"
    return "long"


def load_categories(path: Path) -> list[Category]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    categories = [
        Category(
            id=c["id"],
            name=c["name"],
            describe=c["describe"],
            lengths=dict(c["lengths"]),
            has_signoff=bool(c["has_signoff"]),
            examples=tuple(c.get("examples", ())),
        )
        for c in data["categories"]
    ]
    ids = [c.id for c in categories]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate category ids in {path}")
    for c in categories:
        if set(c.lengths) != set(BANDS) or abs(sum(c.lengths.values()) - 1) > 1e-9:
            raise ValueError(f"category {c.id}: lengths must give short/medium/long shares summing to 1")
    return categories


def load_writers(path: Path) -> tuple[list[Writer], Judge]:
    """Writers sorted by name, plus the judge. The judge is its own table, so it is never dealt requests."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    writers = sorted((Writer(name=name, **cfg) for name, cfg in data["writers"].items()), key=lambda w: w.name)
    for w in writers:
        if w.kind not in {"openai", "import"}:
            raise ValueError(f"writer {w.name}: kind must be 'openai' or 'import', got {w.kind!r}")
        if w.kind == "openai" and not w.base_url:
            raise ValueError(f"writer {w.name}: an openai writer needs base_url")
    return writers, Judge(**data["judge"])


def load_sources(path: Path) -> list[Source]:
    """Human sources, sorted by name (the selection pass's order)."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f for f in Source.__dataclass_fields__ if f not in {"name", "options"}}
    sources = []
    for name, cfg in data["sources"].items():
        options = {k: v for k, v in cfg.items() if k not in known}
        sources.append(Source(name=name, options=options, **{k: v for k, v in cfg.items() if k in known}))
    return sorted(sources, key=lambda s: s.name)
