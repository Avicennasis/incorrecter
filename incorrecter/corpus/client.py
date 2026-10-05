"""A minimal OpenAI-compatible chat client for the writers and the judge."""

import ipaddress
import json
import os
import urllib.parse
from dataclasses import dataclass, field

from incorrecter.corpus.config import Judge, Writer
from incorrecter.corpus.http import Response, Session, Stopped


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None
    model: str  # the model the server says answered, which can differ from the one asked for
    usage: dict
    retries: int


def parse_completion(resp: Response) -> tuple[str, str | None, str, dict]:
    """(content, finish_reason, served model, usage). ValueError (retried like a 5xx) when the body is not
    JSON or has no choices[0].message.content. A null content is an empty draft, which the gate rejects."""
    payload = json.loads(resp.body)
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        if "content" not in message:
            raise KeyError("content")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"no choices[0].message.content ({exc!r})") from None
    return message["content"] or "", choice.get("finish_reason"), payload.get("model") or "", payload.get("usage") or {}


@dataclass
class ChatClient:
    name: str
    base_url: str
    model: str
    session: Session
    api_key: str | None = None
    max_tokens: int = 1500
    disable_thinking: bool = False
    extra_body: dict = field(default_factory=dict)

    def complete(
        self, messages: list[dict[str, str]], *, request_id: str, temperature: float, top_p: float | None = None
    ) -> Completion:
        body: dict = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body.update(self.extra_body)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        (text, finish, served, usage), retries = self.session.request(
            "POST",
            self.base_url.rstrip("/") + "/chat/completions",
            request_id=request_id,
            headers=headers,
            body=json.dumps(body).encode("utf-8"),
            parse=parse_completion,
        )
        return Completion(text, finish, served or self.model, usage, retries)


def _api_key(name: str, env: str) -> str | None:
    if not env:
        return None
    key = os.environ.get(env)
    if not key:
        raise Stopped(name, f"environment variable {env} is not set")
    return key


def client_for(writer: Writer, **session_kwargs) -> ChatClient:
    session = Session(writer.name, rpm=writer.rpm, daily_reset_tz=writer.daily_reset_tz, **session_kwargs)
    return ChatClient(
        name=writer.name,
        base_url=writer.base_url,
        model=writer.model,
        session=session,
        api_key=_api_key(writer.name, writer.api_key_env),
        max_tokens=writer.max_tokens,
        disable_thinking=writer.disable_thinking,
        extra_body=dict(writer.extra_body),
    )


_MESH = ipaddress.ip_network("100.64.0.0/10")  # Tailscale's CGNAT range: the fleet Macs


def is_local_endpoint(base_url: str) -> bool:
    """Loopback, private LAN or the Tailscale mesh. Never a cloud API."""
    host = urllib.parse.urlsplit(base_url).hostname or ""
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None):  # ::ffff:8.8.8.8 counts as private on older 3.12 patch releases
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_private or ip in _MESH


def judge_client(judge: Judge, **session_kwargs) -> ChatClient:
    """The judge sees human text, so it must be a local model (spec S11): a cloud base_url is refused."""
    if not is_local_endpoint(judge.base_url):
        raise Stopped(
            "judge",
            f"base_url {judge.base_url} is not a local, LAN or mesh host; human text never goes "
            "to a cloud API (spec S11)",
        )
    return ChatClient(
        name="judge",
        base_url=judge.base_url,
        model=judge.model,
        session=Session("judge", quote_bodies=False, **session_kwargs),
        max_tokens=judge.max_tokens,
        disable_thinking=judge.disable_thinking,
    )
