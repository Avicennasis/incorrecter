"""A local http.server that plays scripted responses, standing in for an OpenAI-compatible endpoint."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakeServer:
    def __init__(self) -> None:
        self.script: list[dict] = []
        self.requests: list[dict] = []
        handler = self._handler()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def reply(
        self, status=200, body=None, headers=None, delay=0.0, trickle=None, trickle_raw=None, content_length=None
    ) -> None:
        """Queue one response. body: dict/list (sent as JSON), str or bytes. trickle: seconds between body bytes.
        trickle_raw: seconds between bytes of the whole response, status line and headers included."""
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        self.script.append(
            {
                "status": status,
                "body": body or b"",
                "headers": headers or {},
                "delay": delay,
                "trickle": trickle,
                "trickle_raw": trickle_raw,
                "content_length": content_length,
            }
        )

    def completion(self, content="Hello there.", finish_reason="stop", model="served-model", usage=None) -> None:
        message = {"role": "assistant", "content": content}
        self.reply(
            body={
                "model": model,
                "choices": [{"message": message, "finish_reason": finish_reason}],
                "usage": usage or {"completion_tokens": 3},
            }
        )

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                server.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": dict(self.headers),
                        "json": json.loads(raw) if raw else None,
                    }
                )
                step = (
                    server.script.pop(0)
                    if server.script
                    else {"status": 500, "body": b"script exhausted", "headers": {}, "delay": 0.0, "trickle": None}
                )
                time.sleep(step["delay"])
                if step.get("trickle_raw"):
                    head = f"HTTP/1.1 {step['status']} OK\r\nX-Pad: {'a' * 80}\r\nContent-Length: {len(step['body'])}"
                    raw = f"{head}\r\n\r\n".encode() + step["body"]
                    try:
                        for byte in raw:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(step["trickle_raw"])
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_response(step["status"])
                for key, value in step["headers"].items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(step.get("content_length") or len(step["body"])))
                self.end_headers()
                try:
                    if step["trickle"]:
                        for byte in step["body"]:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(step["trickle"])
                    else:
                        self.wfile.write(step["body"])
                except (BrokenPipeError, ConnectionResetError):
                    pass

            do_GET = do_POST = _serve

            def log_message(self, *args) -> None:
                pass

        return Handler


@pytest.fixture
def fake_api():
    server = FakeServer()
    yield server
    server.httpd.shutdown()


class FakeClock:
    """A clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


class StaticServer:
    """Serves fixed files with Range support (206 + Content-Range) and 404 for anything else."""

    def __init__(self, files: dict[str, bytes], honour_range: bool = True) -> None:
        self.files = files
        self.honour_range = honour_range
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                server.requests.append({"path": self.path, "range": self.headers.get("Range")})
                body = server.files.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status, extra = 200, {}
                if (rng := self.headers.get("Range")) and server.honour_range:
                    start, end = (int(x) for x in rng.removeprefix("bytes=").split("-"))
                    end = min(end, len(body) - 1)
                    extra = {"Content-Range": f"bytes {start}-{end}/{len(body)}"}
                    body, status = body[start : end + 1], 206
                self.send_response(status)
                for key, value in extra.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()


@pytest.fixture
def static_server():
    servers = []

    def start(files: dict[str, bytes], honour_range: bool = True) -> StaticServer:
        servers.append(StaticServer(files, honour_range))
        return servers[-1]

    yield start
    for server in servers:
        server.httpd.shutdown()


@pytest.fixture
def other_api():
    """A second fake server on another port: a different host:port for redirect tests."""
    server = FakeServer()
    yield server
    server.httpd.shutdown()
