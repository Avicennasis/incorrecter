import time

import pytest

from incorrecter.corpus.client import ChatClient
from incorrecter.corpus.http import RequestFailed, Session, Stopped

MESSAGES = [{"role": "user", "content": "Write a note."}]


def make_client(fake_api, clock, **kwargs):
    session_kwargs = {k: kwargs.pop(k) for k in ("rpm", "retries", "read_timeout", "connect_timeout") if k in kwargs}
    session = Session("glm", clock=clock, sleep=clock.sleep, now=lambda: 1_790_000_000.0, **session_kwargs)
    return ChatClient(name="glm", base_url=fake_api.url + "/v1", model="glm-5.3-flash", session=session, **kwargs)


def test_429_then_200_succeeds_after_one_backoff(fake_api, clock):
    fake_api.reply(429, {"error": {"message": "slow down"}})
    fake_api.completion("Hi Priya,\n\nThanks.\n\nMarcus")
    result = make_client(fake_api, clock).complete(MESSAGES, request_id="work-0001", temperature=0.8)
    assert result.text == "Hi Priya,\n\nThanks.\n\nMarcus"
    assert result.retries == 1
    assert clock.sleeps == [2.0]


def test_retry_after_is_honoured_and_capped_at_300(fake_api, clock):
    fake_api.reply(429, {"error": {}}, headers={"Retry-After": "7"})
    fake_api.reply(503, "busy", headers={"Retry-After": "600"})
    fake_api.completion()
    make_client(fake_api, clock).complete(MESSAGES, request_id="work-0001", temperature=0.8)
    assert clock.sleeps == [7.0, 300.0]


def test_six_500s_raise_naming_the_writer_and_request(fake_api, clock):
    for _ in range(6):
        fake_api.reply(500, "boom")
    with pytest.raises(RequestFailed) as err:
        make_client(fake_api, clock).complete(MESSAGES, request_id="work-0007", temperature=0.8)
    assert "glm" in str(err.value) and "work-0007" in str(err.value)
    assert err.value.retries == 5
    assert clock.sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert len(fake_api.requests) == 6


def test_disable_thinking_controls_chat_template_kwargs(fake_api, clock):
    fake_api.completion()
    fake_api.completion()
    make_client(fake_api, clock, disable_thinking=True).complete(MESSAGES, request_id="a", temperature=0.8)
    make_client(fake_api, clock).complete(MESSAGES, request_id="b", temperature=0.8)
    assert fake_api.requests[0]["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in fake_api.requests[1]["json"]


def test_max_tokens_extra_body_and_sampling_are_sent(fake_api, clock):
    fake_api.completion()
    client = make_client(fake_api, clock, max_tokens=3000, extra_body={"thinking": {"type": "disabled"}})
    client.complete(MESSAGES, request_id="a", temperature=0.8, top_p=0.95)
    sent = fake_api.requests[0]["json"]
    assert sent["max_tokens"] == 3000
    assert sent["thinking"] == {"type": "disabled"}
    assert (sent["temperature"], sent["top_p"], sent["model"]) == (0.8, 0.95, "glm-5.3-flash")
    assert fake_api.requests[0]["path"] == "/v1/chat/completions"


def test_non_json_and_missing_content_are_retried(fake_api, clock):
    fake_api.reply(200, "<html>gateway</html>")
    fake_api.reply(200, {"choices": [{"message": {"role": "assistant"}}]})
    fake_api.completion("fine")
    result = make_client(fake_api, clock).complete(MESSAGES, request_id="a", temperature=0.8)
    assert (result.text, result.retries) == ("fine", 2)


def test_empty_and_null_content_come_back_as_empty_drafts(fake_api, clock):
    fake_api.completion("")
    fake_api.reply(200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}]})
    client = make_client(fake_api, clock)
    assert client.complete(MESSAGES, request_id="a", temperature=0.8).text == ""
    result = client.complete(MESSAGES, request_id="b", temperature=0.8)
    assert (result.text, result.finish_reason, result.model) == ("", "length", "glm-5.3-flash")


def test_served_model_is_recorded(fake_api, clock):
    fake_api.completion(model="glm-5.3-flash-0925")
    assert make_client(fake_api, clock).complete(MESSAGES, request_id="a", temperature=0).model == "glm-5.3-flash-0925"


def test_401_stops_the_writer(fake_api, clock):
    fake_api.reply(401, {"error": "bad key"})
    with pytest.raises(Stopped, match="glm stopped: HTTP 401"):
        make_client(fake_api, clock).complete(MESSAGES, request_id="a", temperature=0.8)


def test_400_is_logged_once_and_not_retried(fake_api, clock):
    fake_api.reply(400, {"error": "bad request"})
    with pytest.raises(RequestFailed, match="HTTP 400"):
        make_client(fake_api, clock).complete(MESSAGES, request_id="a", temperature=0.8)
    assert len(fake_api.requests) == 1
    assert clock.sleeps == []


def test_five_consecutive_failed_requests_trip_the_breaker(fake_api, clock):
    client = make_client(fake_api, clock)
    for _ in range(5):
        fake_api.reply(400, "bad")
    for i in range(4):
        with pytest.raises(RequestFailed):
            client.complete(MESSAGES, request_id=f"r{i}", temperature=0.8)
    with pytest.raises(Stopped, match="circuit breaker"):
        client.complete(MESSAGES, request_id="r4", temperature=0.8)


def test_a_success_resets_the_breaker(fake_api, clock):
    client = make_client(fake_api, clock)
    for _ in range(4):
        fake_api.reply(400, "bad")
    fake_api.completion()
    fake_api.reply(400, "bad")
    for i in range(4):
        with pytest.raises(RequestFailed):
            client.complete(MESSAGES, request_id=f"r{i}", temperature=0.8)
    client.complete(MESSAGES, request_id="ok", temperature=0.8)
    with pytest.raises(RequestFailed):
        client.complete(MESSAGES, request_id="r5", temperature=0.8)


def test_daily_quota_429_stops_the_writer(fake_api, clock):
    fake_api.reply(
        429,
        [
            {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}],
                }
            }
        ],
    )
    with pytest.raises(Stopped, match="daily quota exhausted .* resets 2026-"):
        make_client(fake_api, clock).complete(MESSAGES, request_id="a", temperature=0.8)
    assert len(fake_api.requests) == 1


def test_429_with_an_unparseable_body_backs_off_as_per_minute(fake_api, clock):
    fake_api.reply(429, "Too Many Requests")
    fake_api.completion()
    make_client(fake_api, clock).complete(MESSAGES, request_id="a", temperature=0.8)
    assert clock.sleeps == [2.0]


def test_rpm_spaces_request_starts(fake_api, clock):
    fake_api.completion()
    fake_api.completion()
    client = make_client(fake_api, clock, rpm=30)
    client.complete(MESSAGES, request_id="a", temperature=0.8)
    client.complete(MESSAGES, request_id="b", temperature=0.8)
    assert clock.sleeps == [2.0]


def test_a_hanging_endpoint_trips_the_breaker_after_five_timed_out_attempts(fake_api, clock):
    for _ in range(6):
        fake_api.reply(delay=1.0, body={"never": "read"})
    client = make_client(fake_api, clock, read_timeout=0.1, connect_timeout=0.5)
    started = time.monotonic()
    with pytest.raises(Stopped, match="circuit breaker: 5 consecutive failures; last: TimeoutError"):
        client.complete(MESSAGES, request_id="a", temperature=0.8)
    assert len(fake_api.requests) == 5
    assert time.monotonic() - started < 5


def test_a_trickling_server_hits_the_attempt_deadline(fake_api, clock):
    fake_api.reply(body="x" * 100, trickle=0.05)  # 5 s to send the body, but each byte arrives within 0.1 s
    client = make_client(fake_api, clock, read_timeout=0.3, connect_timeout=0.3, retries=0)
    started = time.monotonic()
    with pytest.raises(RequestFailed, match="deadline"):
        client.complete(MESSAGES, request_id="a", temperature=0.8)
    assert time.monotonic() - started < 1.5


def test_connection_refused_is_retried(clock):
    session = Session("qwen", clock=clock, sleep=clock.sleep, retries=1, connect_timeout=0.5)
    client = ChatClient(name="qwen", base_url="http://127.0.0.1:9/v1", model="m", session=session)
    with pytest.raises(RequestFailed, match="ConnectionRefusedError"):
        client.complete(MESSAGES, request_id="a", temperature=0.8)
    assert clock.sleeps == [2.0]


def test_a_same_host_redirect_is_followed_with_the_key(fake_api, clock):
    fake_api.reply(307, headers={"Location": "/v2/chat/completions"})
    fake_api.completion("moved")
    client = make_client(fake_api, clock, api_key="secret-key")
    assert client.complete(MESSAGES, request_id="a", temperature=0.8).text == "moved"
    assert fake_api.requests[1]["path"] == "/v2/chat/completions"
    assert fake_api.requests[1]["headers"]["Authorization"] == "Bearer secret-key"


def test_a_get_redirected_to_another_host_drops_credentials(fake_api, other_api, clock):
    fake_api.reply(302, headers={"Location": other_api.url + "/file"})
    other_api.reply(body="ok")
    session = Session("hf", clock=clock, sleep=clock.sleep)
    resp, _ = session.request(
        "GET", fake_api.url + "/file", request_id="a", headers={"Authorization": "Bearer secret-key", "Cookie": "s=1"}
    )
    assert resp.body == b"ok"
    assert "Authorization" not in other_api.requests[0]["headers"] and "Cookie" not in other_api.requests[0]["headers"]


def test_a_redirect_body_is_not_held_to_the_callers_body_cap(fake_api, other_api, clock):
    # HF answers a one-byte Range read with a 302 whose ~1 KiB body echoes the signed CDN URL; only the final
    # response's body is capped (checked 2026-09-29: 1,052 bytes for one archive, over the size probe's 1,024).
    fake_api.reply(302, body=b"x" * 2048, headers={"Location": other_api.url + "/file"})
    other_api.reply(206, body=b"a", headers={"Content-Range": "bytes 0-0/1880986004"})
    session = Session("hf", clock=clock, sleep=clock.sleep)
    resp, _ = session.request("GET", fake_api.url + "/file", request_id="size", max_bytes=1024)
    assert (resp.status, resp.body) == (206, b"a")


def test_a_request_body_is_never_resent_to_another_host(fake_api, other_api, clock):
    # A judge POST carries human text: a 307 to another host must not re-send it.
    fake_api.reply(307, headers={"Location": other_api.url + "/v1/chat/completions"})
    with pytest.raises(RequestFailed, match="another host"):
        make_client(fake_api, clock, retries=0).complete(MESSAGES, request_id="a", temperature=0.8)
    assert other_api.requests == []


def test_a_redirect_to_a_non_http_scheme_is_refused(fake_api, clock):
    fake_api.reply(302, headers={"Location": "file:///etc/passwd"})
    with pytest.raises(RequestFailed, match="redirect"):
        make_client(fake_api, clock, retries=0).complete(MESSAGES, request_id="a", temperature=0.8)


def test_a_server_trickling_its_headers_hits_the_attempt_deadline(fake_api, clock):
    fake_api.reply(body="{}", trickle_raw=0.05)  # about 7 s to send the status line and headers
    client = make_client(fake_api, clock, read_timeout=0.3, connect_timeout=0.3, retries=0)
    started = time.monotonic()
    with pytest.raises(RequestFailed, match="deadline"):
        client.complete(MESSAGES, request_id="a", temperature=0.8)
    assert time.monotonic() - started < 2.0


def test_a_refused_redirect_names_only_the_host(fake_api, other_api, clock):
    # A redirect URL can carry anything the server echoes, including a judge's human text.
    fake_api.reply(307, headers={"Location": other_api.url + "/v1/x?q=Dear+Pat+the+papers"})
    with pytest.raises(RequestFailed) as err:
        make_client(fake_api, clock, retries=0).complete(MESSAGES, request_id="a", temperature=0.8)
    assert "another host" in str(err.value) and "Pat" not in str(err.value)


def test_a_body_cut_short_is_retried_not_accepted(fake_api, clock):
    # No parse step here (raw fetches have none): a truncated body must not come back as a success, or it is cached.
    fake_api.reply(body="partial page", content_length=500)  # the connection closes 488 bytes early
    fake_api.reply(body="whole page")
    resp, retries = Session("hf", clock=clock, sleep=clock.sleep).request("GET", fake_api.url + "/p", request_id="a")
    assert (resp.body, retries) == (b"whole page", 1)


def test_the_absent_streak_only_guards_a_source_that_never_succeeded(fake_api, clock):
    session = Session("archive", clock=clock, sleep=clock.sleep)
    fake_api.reply(body="found")
    for _ in range(25):
        fake_api.reply(404)
    assert session.request("GET", fake_api.url + "/1", request_id="1", absent=(404,))[0].body == b"found"
    for i in range(25):  # a sparse id space after a success is not a wrong URL scheme
        assert session.request("GET", fake_api.url + f"/{i}", request_id=str(i), absent=(404,))[0] is None
