"""Fetching for the human sources: a raw cache, HF rows pages, HTTP Range reads, robots.txt and whole files.

Raw fetches are cached under data/corpus/raw/<source>/ (gitignored, spec S13), so a resumed run never
re-fetches. Every network request goes through http.Session (retries, backoff, breaker, User-Agent).
"""

import hashlib
import json
import urllib.parse
import urllib.robotparser
from pathlib import Path

from incorrecter.corpus.http import USER_AGENT, Response, Session, Stopped
from incorrecter.corpus.judge import DriftError

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}"
HF_API_URL = "https://huggingface.co/api/datasets/{dataset}"
HF_FILE_URL = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{file}"
ROWS_PAGE = 100
CHUNK = 256 * 1024
MAX_RECORD = 4 * 1024 * 1024  # a record that doesn't end within 4 MiB is skipped, and still counts toward the cap


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Fetcher:
    def __init__(self, session: Session, cache_dir: Path) -> None:
        self.session = session
        self.cache_dir = cache_dir
        self.network = 0  # requests that actually went out

    def _store(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(
        self, url: str, *, key: str, request_id: str, max_bytes: int = 64 * 1024 * 1024, absent_statuses=(404,)
    ) -> bytes | None:
        """The cached body, else one GET. An absent status (404 by default) is cached as absent and returns None."""
        path = self.cache_dir / key
        absent = path.with_name(path.name + ".absent")
        if path.exists():
            return path.read_bytes()
        if absent.exists():
            return None
        resp, _ = self.session.request(
            "GET", url, request_id=request_id, absent=tuple(absent_statuses), max_bytes=max_bytes
        )
        self.network += 1
        if resp is None:
            self._store(absent, b"")
            return None
        self._store(path, resp.body)
        return resp.body

    def get_range(self, url: str, start: int, end: int, *, request_id: str) -> bytes:
        """Bytes start..end inclusive, uncached (read_record caches whole records). Only 206 is accepted."""

        def partial(resp: Response) -> bytes:
            if resp.status != 206:
                raise Stopped(self.session.name, f"expected 206 for a Range read, got {resp.status}")
            return resp.body

        body, _ = self.session.request(
            "GET",
            url,
            request_id=request_id,
            headers={"Range": f"bytes={start}-{end}"},
            parse=partial,
            max_bytes=end - start + 1 + 1024,
        )
        self.network += 1
        return body


def hf_sha(session: Session, dataset: str) -> str:
    resp, _ = session.request("GET", HF_API_URL.format(dataset=dataset), request_id=f"api-{dataset}")
    return json.loads(resp.body)["sha"]


def check_pin(session: Session, dataset: str, revision: str) -> None:
    """The rows API only serves the current revision: refuse to mix pages from two revisions."""
    current = hf_sha(session, dataset)
    if current != revision:
        raise DriftError(f"{dataset} is now at {current}; sources.toml pins {revision}. Review, then re-pin.")


def rows_key(revision: str, offset: int) -> str:
    """Cache key of a rows page: the pinned revision is part of it, so a re-pin never reads old pages."""
    return f"rows-{revision[:12]}-{offset:07d}.json"


def hf_rows(
    fetcher: Fetcher, dataset: str, *, revision: str, config: str = "default", split: str = "train", stop=None
) -> list[dict]:
    """Every row of the split, page by page ({row_idx, row, truncated_cells}). `stop(row)` ends paging after the
    page where it first returns True."""
    base = HF_ROWS_URL.format(dataset=urllib.parse.quote(dataset, safe=""), config=config, split=split)
    rows: list[dict] = []
    offset, total, pinned = 0, None, False
    while total is None or offset < total:
        key = rows_key(revision, offset)
        if not pinned and not (fetcher.cache_dir / key).exists():
            check_pin(fetcher.session, dataset, revision)  # before the first page this run fetches from the network
            pinned = True
        body = fetcher.get(f"{base}&offset={offset}&length={ROWS_PAGE}", key=key, request_id=f"rows-{offset}")
        if body is None:
            raise Stopped(fetcher.session.name, f"rows page at offset {offset} returned 404")
        page = json.loads(body)
        if total is not None and page["num_rows_total"] != total:
            raise DriftError(f"{dataset}: num_rows_total changed from {total} to {page['num_rows_total']} mid-run")
        total = page["num_rows_total"]
        if not page["rows"] and offset < total:
            raise Stopped(fetcher.session.name, f"rows page at offset {offset} is empty before {total} rows")
        rows.extend(page["rows"])
        if stop is not None and any(stop(r["row"]) for r in page["rows"]):
            break
        offset += len(page["rows"])  # a short page must not skip rows
    return rows


def file_size(fetcher: Fetcher, url: str) -> int:
    """Total size from the Content-Range of a one-byte Range read, cached per URL (a re-pin gets a new size)."""
    path = fetcher.cache_dir / f"size-{digest(url)}.txt"
    if path.exists():
        return int(path.read_text())

    def total(resp: Response) -> int:
        if resp.status != 206 or "/" not in resp.headers.get("content-range", ""):
            raise Stopped(fetcher.session.name, f"no Content-Range from {url} (status {resp.status})")
        return int(resp.headers["content-range"].rsplit("/", 1)[1])

    size, _ = fetcher.session.request(
        "GET", url, request_id="size", headers={"Range": "bytes=0-0"}, parse=total, max_bytes=1024
    )
    fetcher.network += 1
    fetcher._store(path, str(size).encode())
    return size


def _find_record(fetcher: Fetcher, url: str, offset: int, size: int) -> tuple[int, bytes] | None:
    def read(pos: int) -> bytes:
        return fetcher.get_range(url, pos, min(pos + CHUNK, size) - 1, request_id=f"range-{pos}")

    pos, buf, start = offset, b"", 0
    if offset > 0:  # resync: the record starts right after the first newline at or after `offset`
        while True:
            if pos >= size or pos - offset > MAX_RECORD:
                return None
            chunk = read(pos)
            if not chunk:  # an empty 206: the file ended early
                return None
            newline = chunk.find(b"\n")
            if newline >= 0:
                start, buf = pos + newline + 1, chunk[newline + 1 :]
                pos += len(chunk)
                break
            pos += len(chunk)
    while b"\n" not in buf and pos < size:  # the last record may lack a trailing newline
        if len(buf) > MAX_RECORD:
            return None
        chunk = read(pos)
        if not chunk:  # an empty 206 is the end of the file: never re-request the same offset
            break
        buf += chunk
        pos += len(chunk)
    record = buf.split(b"\n", 1)[0]
    if start >= size or not record.strip() or len(record) > MAX_RECORD:
        return None
    return start, record


def read_record(fetcher: Fetcher, url: str, offset: int, size: int) -> tuple[int, bytes] | None:
    """The JSONL record after the first newline at or after `offset` (the first record when offset is 0), as
    (start offset, bytes). None when no record ends within MAX_RECORD. Cached per sampled offset."""
    path = fetcher.cache_dir / f"off-{digest(url)}-{offset:012d}.rec"
    if path.exists():
        head, _, record = path.read_bytes().partition(b"\n")
        return None if head == b"none" else (int(head), record)
    result = _find_record(fetcher, url, offset, size)
    fetcher._store(path, b"none\n" if result is None else f"{result[0]}\n".encode() + result[1])
    return result


def robots(fetcher: Fetcher, base_url: str) -> urllib.robotparser.RobotFileParser:
    """The site's robots.txt (a missing one allows everything). Checked before every fetch."""
    body = fetcher.get(f"{base_url}/robots.txt", key=f"robots-{digest(base_url)}.txt", request_id="robots.txt")
    parser = urllib.robotparser.RobotFileParser()
    parser.parse((body or b"").decode("utf-8", errors="replace").splitlines())
    return parser


def allowed(parser: urllib.robotparser.RobotFileParser, url: str) -> bool:
    return parser.can_fetch(USER_AGENT, url)
