import json

import pytest

from incorrecter.corpus.client import ChatClient
from incorrecter.corpus.http import Session
from incorrecter.corpus.judge import DriftError, Judge, parse_verdict, sha256


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ("No.", "keep"),
        ("no", "keep"),
        ("  NO!\n", "keep"),
        ("YES", "reject"),
        ("yes.", "reject"),
        ("no, looks clean", "judge_unparseable"),
        ("maybe", "judge_unparseable"),
        ("", "judge_unparseable"),
    ],
)
def test_parse_verdict(reply, verdict):
    assert parse_verdict(reply) == verdict


def test_judge_caches_by_id_and_sha_and_never_stores_text(tmp_path, fake_api, clock):
    fake_api.completion("No.")
    client = ChatClient(
        name="judge",
        base_url=fake_api.url + "/v1",
        model="m",
        max_tokens=8,
        disable_thinking=True,
        session=Session("judge", clock=clock, sleep=clock.sleep),
    )
    cache = tmp_path / "judge.jsonl"
    judge = Judge(cache, client)
    assert judge.verdict("sample-12", "The report is attached.") == "keep"
    assert judge.verdict("sample-12", "The report is attached.") == "keep"  # cached: one call
    assert len(fake_api.requests) == 1
    sent = fake_api.requests[0]["json"]
    assert sent["temperature"] == 0.0 and sent["max_tokens"] == 8
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    assert rows == [{"id": "sample-12", "sha256": sha256("The report is attached."), "verdict": "keep"}]


def test_resume_skips_judged_candidates_and_detects_drift(tmp_path):
    cache = tmp_path / "judge.jsonl"
    cache.write_text(json.dumps({"id": "sample-12", "sha256": sha256("old text"), "verdict": "keep"}) + "\n")
    judge = Judge(cache)  # no client: any uncached call would fail
    assert judge.verdict("sample-12", "old text") == "keep"
    with pytest.raises(DriftError, match="sample-12"):
        judge.verdict("sample-12", "new text")


@pytest.mark.parametrize(
    "url",
    ["https://api.z.ai/api/coding/paas/v4", "https://generativelanguage.googleapis.com/v1", "http://8.8.8.8:8097/v1"],
)
def test_the_judge_refuses_a_non_local_endpoint(url):
    from incorrecter.corpus.client import judge_client
    from incorrecter.corpus.config import Judge as JudgeConfig
    from incorrecter.corpus.http import Stopped

    with pytest.raises(Stopped, match="local"):
        judge_client(JudgeConfig(base_url=url, model="m"))


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:18102/v1",
        "http://localhost:8097/v1",
        "http://100.64.0.15:8097/v1",
        "http://192.168.1.20:8097/v1",
    ],
)
def test_the_judge_accepts_loopback_mesh_and_lan_endpoints(url):
    from incorrecter.corpus.client import judge_client
    from incorrecter.corpus.config import Judge as JudgeConfig

    assert judge_client(JudgeConfig(base_url=url, model="m")).base_url == url


def test_judge_errors_never_quote_the_response_body(fake_api, clock):
    from incorrecter.corpus.client import judge_client
    from incorrecter.corpus.config import Judge as JudgeConfig
    from incorrecter.corpus.http import RequestFailed

    fake_api.reply(400, {"error": "invalid input: 'Dear Pat, the lawsuit papers are attached'"})
    client = judge_client(JudgeConfig(base_url=fake_api.url + "/v1", model="m"), clock=clock, sleep=clock.sleep)
    with pytest.raises(RequestFailed) as err:
        client.complete([{"role": "user", "content": "x"}], request_id="sample-1", temperature=0.0)
    assert "HTTP 400" in str(err.value) and "lawsuit" not in str(err.value)


def test_an_ipv4_mapped_public_address_is_not_local():
    from incorrecter.corpus.client import is_local_endpoint

    assert not is_local_endpoint("http://[::ffff:8.8.8.8]:8097/v1")
    assert is_local_endpoint("http://[::ffff:127.0.0.1]:8097/v1")


def test_a_truncated_last_cache_line_is_ignored(tmp_path):
    cache = tmp_path / "judge.jsonl"
    cache.write_text(json.dumps({"id": "sample-1", "sha256": sha256("a text"), "verdict": "keep"}) + '\n{"id": "sam')
    assert Judge(cache).cached("sample-1", "a text") == "keep"
