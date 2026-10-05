"""HTTP for every corpus fetch: separate connect and read timeouts, a per-attempt deadline, retries with
backoff, a requests-per-minute limit, and a circuit breaker. Stdlib only."""

import email.utils
import http.client
import json
import socket
import ssl
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

USER_AGENT = "incorrecter-corpus/0.3 (private research corpus; contact claudeai@simmons.systems)"

RETRIES = 5  # one attempt plus up to 5 retries
BREAKER = 5  # consecutive failures before a writer or source stops
BACKOFF_CAP = 60.0
RETRY_AFTER_CAP = 300.0
ABSENT_STREAK = 20  # consecutive "absent" responses (e.g. 404) before a source stops: the URL scheme is wrong
MAX_BYTES = 64 * 1024 * 1024
MAX_REDIRECTS = 5
_REDIRECTS = (301, 302, 303, 307, 308)


class RequestFailed(Exception):
    """One request failed after its retries (or with a non-retryable 4xx). The caller logs it and moves on."""

    def __init__(self, name: str, request_id: str, reason: str, retries: int) -> None:
        super().__init__(f"{name}: request {request_id} failed after {retries} retries: {reason}")
        self.name, self.request_id, self.reason, self.retries = name, request_id, reason, retries


class Stopped(Exception):
    """The writer or source must stop: bad credentials, an exhausted daily quota, or a tripped breaker."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"{name} stopped: {reason}")
        self.name, self.reason = name, reason


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes


class _Connection(http.client.HTTPConnection):
    """Connects with `connect_timeout`, then reads with `read_timeout` per socket operation."""

    def __init__(self, host: str, port: int | None, *, connect_timeout: float, read_timeout: float) -> None:
        super().__init__(host, port, timeout=connect_timeout)
        self.read_timeout = read_timeout

    def connect(self) -> None:
        super().connect()
        self.sock.settimeout(self.read_timeout)


class _TLSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int | None, *, connect_timeout: float, read_timeout: float) -> None:
        super().__init__(host, port, timeout=connect_timeout, context=ssl.create_default_context())
        self.read_timeout = read_timeout

    def connect(self) -> None:
        super().connect()
        self.sock.settimeout(self.read_timeout)


class BodyTooLarge(ValueError):
    """The body passed max_bytes, e.g. a server that ignored a Range header and sent the whole file."""


def _cut(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def _one_request(method, url, *, headers, body, connect_timeout, read_timeout, deadline, max_bytes) -> Response:
    parts = urllib.parse.urlsplit(url)
    cls = _TLSConnection if parts.scheme == "https" else _Connection
    conn = cls(parts.hostname, parts.port, connect_timeout=connect_timeout, read_timeout=read_timeout)
    watchdog = None
    try:
        conn.connect()
        # Per-read timeouts cannot bound a server that trickles its status line and headers, so a watchdog shuts the
        # socket at the deadline, which fails whichever read is blocked.
        watchdog = threading.Timer(max(0.0, deadline - time.monotonic()), _cut, args=(conn.sock,))
        watchdog.daemon = True
        watchdog.start()
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status in _REDIRECTS and resp.getheader("location"):
            # A redirect's body is never used, so it is not read, nor held to the final response's max_bytes (HF's
            # 302 echoes the signed CDN URL in ~1 KiB of HTML).
            return Response(resp.status, {k.lower(): v for k, v in resp.getheaders()}, b"")
        chunks, size = [], 0
        while True:
            # Each read waits at most read_timeout (set at connect), so the deadline check runs at least that
            # often. The connection may already be closed here (Connection: close), so its socket is not touched.
            if time.monotonic() > deadline:
                raise TimeoutError(f"attempt exceeded its {connect_timeout + read_timeout:g} s deadline")
            chunk = resp.read1(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise BodyTooLarge(f"body passed {max_bytes} bytes")
            chunks.append(chunk)
        if time.monotonic() > deadline:  # the watchdog fired: what was read may be cut short
            raise TimeoutError(f"attempt exceeded its {connect_timeout + read_timeout:g} s deadline")
        if resp.length:  # read1 returns b"" at a premature EOF instead of raising: never accept a short body
            raise http.client.IncompleteRead(b"".join(chunks), resp.length)
        return Response(resp.status, {k.lower(): v for k, v in resp.getheaders()}, b"".join(chunks))
    except (OSError, http.client.HTTPException) as exc:
        if time.monotonic() > deadline and not isinstance(exc, TimeoutError):
            raise TimeoutError(f"attempt exceeded its {connect_timeout + read_timeout:g} s deadline") from exc
        raise
    finally:
        if watchdog is not None:
            watchdog.cancel()
        conn.close()


def http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    body: bytes | None = None,
    connect_timeout: float = 30.0,
    read_timeout: float = 300.0,
    max_bytes: int = MAX_BYTES,
) -> Response:
    """One attempt, following up to 5 redirects, which must finish within connect_timeout + read_timeout (plus at
    most one read_timeout), so a server that trickles bytes cannot hang it. TimeoutError is raised when it doesn't."""
    deadline = time.monotonic() + connect_timeout + read_timeout
    for _ in range(MAX_REDIRECTS + 1):
        resp = _one_request(
            method,
            url,
            headers=headers,
            body=body,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            deadline=deadline,
            max_bytes=max_bytes,
        )
        if resp.status not in _REDIRECTS or "location" not in resp.headers:
            return resp
        target = urllib.parse.urljoin(url, resp.headers["location"])
        old, new = urllib.parse.urlsplit(url), urllib.parse.urlsplit(target)
        # Errors name only scheme://host: a redirect URL can carry whatever the server echoes, human text included.
        where = f"{new.scheme}://{new.hostname}" if new.hostname else new.scheme or "(no host)"  # never userinfo
        if new.scheme not in ("http", "https") or not new.hostname:
            raise http.client.HTTPException(f"refusing redirect to {where}")
        if (new.scheme, new.netloc) != (old.scheme, old.netloc):  # another host, or a scheme change: no credentials
            if body is not None and resp.status != 303:
                # A request body (a judge's human text) is never re-sent to another host.
                raise http.client.HTTPException(f"refusing to re-send a request body to another host: {where}")
            headers = {k: v for k, v in headers.items() if k.lower() not in ("authorization", "cookie")}
        url = target
        if resp.status == 303:
            method, body = "GET", None
    raise http.client.HTTPException(f"more than {MAX_REDIRECTS} redirects")


def is_daily_quota(body: bytes) -> bool:
    """A 429 body naming a per-day quota. An unparseable body counts as per-minute."""
    try:
        text = json.dumps(json.loads(body))
    except ValueError:
        return False
    return "PerDay" in text or "per day" in text.lower()


def retry_after(headers: dict[str, str], now: float) -> float | None:
    """Retry-After in seconds (delta-seconds or an HTTP date), capped at 300 s; None when absent or invalid."""
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = email.utils.parsedate_to_datetime(value).timestamp() - now
        except (TypeError, ValueError):
            return None
    return min(max(seconds, 0.0), RETRY_AFTER_CAP)


@dataclass
class Session:
    """Retry, backoff, rate limit and circuit breaker around http_request, for one writer or source.

    clock/sleep/now are injectable so tests never really wait. Timeouts and connection errors count toward the
    breaker per attempt; every other failure counts once per request.
    """

    name: str
    rpm: float = 0.0
    connect_timeout: float = 30.0
    read_timeout: float = 300.0
    retries: int = RETRIES
    daily_reset_tz: str = ""
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.time
    transport: Callable[..., Response] = http_request
    quote_bodies: bool = True  # False for the judge: an error body can echo the human text it was sent
    failures: int = field(default=0, init=False)
    absent: int = field(default=0, init=False)
    ever_ok: bool = field(default=False, init=False)  # the absent streak only guards a source that never succeeded
    _last_start: float | None = field(default=None, init=False)

    def _throttle(self) -> None:
        if self.rpm > 0 and self._last_start is not None:
            wait = self._last_start + 60.0 / self.rpm - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last_start = self.clock()

    def _snippet(self, resp: Response) -> str:
        """The start of an error body, for logs; withheld when the session carries human text (the judge)."""
        return "(body withheld)" if not self.quote_bodies else repr(resp.body[:200])

    def _fail(self, reason: str) -> None:
        self.failures += 1
        if self.failures >= BREAKER:
            raise Stopped(self.name, f"circuit breaker: {self.failures} consecutive failures; last: {reason}")

    def _reset_time(self) -> str:
        zone = ZoneInfo(self.daily_reset_tz or "America/Los_Angeles")
        local = datetime.fromtimestamp(self.now(), zone)
        midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.isoformat()

    def request(
        self,
        method: str,
        url: str,
        *,
        request_id: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        parse: Callable[[Response], Any] | None = None,
        absent: tuple[int, ...] = (),
        max_bytes: int = MAX_BYTES,
    ) -> tuple[Any, int]:
        """(parse(response) or the response, retries used). None when the status is in `absent`.

        Raises RequestFailed (log it and continue) or Stopped (stop this writer or source)."""
        headers = {"User-Agent": USER_AGENT, **(headers or {})}
        reason, counted, wait = "", False, 0.0
        for attempt in range(self.retries + 1):
            if attempt:
                self.sleep(wait)
            self._throttle()
            backoff = min(2.0 ** (attempt + 1), BACKOFF_CAP)
            try:
                resp = self.transport(
                    method,
                    url,
                    headers=headers,
                    body=body,
                    connect_timeout=self.connect_timeout,
                    read_timeout=self.read_timeout,
                    max_bytes=max_bytes,
                )
            except BodyTooLarge as exc:
                raise Stopped(self.name, f"{request_id}: {exc}") from None
            except (OSError, http.client.HTTPException) as exc:  # timeouts, refused and reset connections
                reason, counted, wait = f"{type(exc).__name__}: {exc}", True, backoff
                self._fail(reason)
                continue
            counted = False
            if resp.status in (401, 403):
                raise Stopped(self.name, f"HTTP {resp.status} on {request_id}: check the API key and base_url")
            if resp.status in absent:
                self.absent += 1
                if self.absent >= ABSENT_STREAK and not self.ever_ok:
                    raise Stopped(self.name, f"{self.absent} consecutive HTTP {resp.status}: check the URL scheme")
                return None, attempt
            if resp.status == 429:
                if is_daily_quota(resp.body):
                    raise Stopped(self.name, f"daily quota exhausted on {request_id}; resets {self._reset_time()}")
                reason = f"HTTP 429: {self._snippet(resp)}"
                wait = retry_after(resp.headers, self.now()) or backoff
                continue
            if resp.status >= 500:
                reason = f"HTTP {resp.status}: {self._snippet(resp)}"
                wait = retry_after(resp.headers, self.now()) or backoff
                continue
            if resp.status >= 400:
                reason = f"HTTP {resp.status}: {self._snippet(resp)}"
                self._fail(reason)
                raise RequestFailed(self.name, request_id, reason, attempt)
            try:
                value = parse(resp) if parse else resp
            except ValueError as exc:
                reason, wait = f"bad response body: {exc}", backoff
                continue
            self.failures = 0
            self.absent = 0
            self.ever_ok = True
            return value, attempt
        if not counted:
            self._fail(reason)
        raise RequestFailed(self.name, request_id, reason, self.retries)
