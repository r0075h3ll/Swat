import email.utils
import threading
import time
from unittest import mock

import pytest
import requests

from swat import ct_logger


def test_split_variants_every_single_split_position():
    assert ct_logger.split_variants("abcd") == ["a bcd", "ab cd", "abc d"]


def test_split_variants_single_char_has_no_variants():
    assert ct_logger.split_variants("a") == []


class _FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def json(self):
        if self._body is None:
            raise ValueError("empty body")
        return self._body


def test_get_ct_logs_returns_on_first_success():
    with mock.patch.object(ct_logger.session, "get", return_value=_FakeResponse(200, [{"id": 1}])):
        assert ct_logger.get_ct_logs("q", max_retries=1, backoff_seconds=0.01) == [{"id": 1}]


def test_get_ct_logs_recovers_after_transient_failures():
    calls = {"n": 0}

    def fake_get(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return _FakeResponse(502)
        return _FakeResponse(200, [{"id": 1}])

    with mock.patch.object(ct_logger.session, "get", side_effect=fake_get):
        result = ct_logger.get_ct_logs("q", max_retries=3, backoff_seconds=0.01)

    assert result == [{"id": 1}]
    assert calls["n"] == 3


def test_get_ct_logs_raises_after_exhausting_retries():
    with mock.patch.object(ct_logger.session, "get", return_value=_FakeResponse(502)):
        try:
            ct_logger.get_ct_logs("q", max_retries=1, backoff_seconds=0.01)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "2 attempts" in str(e)


def test_get_ct_logs_honors_retry_after_header_on_429():
    calls = {"n": 0}

    def fake_get(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(429, headers={"Retry-After": "0.05"})
        return _FakeResponse(200, [{"id": 1}])

    with mock.patch.object(ct_logger.session, "get", side_effect=fake_get):
        t0 = time.monotonic()
        result = ct_logger.get_ct_logs("q", max_retries=1, backoff_seconds=10)
        elapsed = time.monotonic() - t0

    assert result == [{"id": 1}]
    assert elapsed < 1, "should honor the short Retry-After, not the much longer generic backoff"


def test_get_ct_logs_falls_back_to_fixed_backoff_on_429_without_header(monkeypatch):
    monkeypatch.setattr(ct_logger, "RATE_LIMIT_BACKOFF_SECONDS", 0.05)
    calls = {"n": 0}

    def fake_get(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(429)
        return _FakeResponse(200, [{"id": 1}])

    with mock.patch.object(ct_logger.session, "get", side_effect=fake_get):
        assert ct_logger.get_ct_logs("q", max_retries=1, backoff_seconds=10) == [{"id": 1}]


def test_get_ct_logs_for_label_merges_unique_entries_by_id():
    def fake_get_ct_logs(query, **_kwargs):
        if "fun" in query:
            return [{"id": 1, "common_name": "a.com"}]
        return [{"id": 1, "common_name": "a.com"}, {"id": 2, "common_name": "b.com"}]

    with mock.patch.object(ct_logger, "get_ct_logs", side_effect=fake_get_ct_logs):
        result = ct_logger.get_ct_logs_for_label("fun")

    ids = sorted(entry["id"] for entry in result)
    assert ids == [1, 2]


def test_get_ct_logs_for_label_caps_concurrency():
    peak = {"current": 0, "max": 0}
    lock = threading.Lock()

    def fake_get_ct_logs(_query, **_kwargs):
        with lock:
            peak["current"] += 1
            peak["max"] = max(peak["max"], peak["current"])
        time.sleep(0.05)
        with lock:
            peak["current"] -= 1
        return []

    with mock.patch.object(ct_logger, "get_ct_logs", side_effect=fake_get_ct_logs):
        ct_logger.get_ct_logs_for_label("examplebrand")

    assert peak["max"] <= ct_logger.FANOUT_CONCURRENCY


def test_get_ct_logs_clamps_an_absurd_retry_after():
    calls = {"n": 0}

    def fake_get(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(429, headers={"Retry-After": "3600"})
        return _FakeResponse(200, [{"id": 1}])

    slept = []
    with (
        mock.patch.object(ct_logger.session, "get", side_effect=fake_get),
        mock.patch.object(ct_logger.time, "sleep", side_effect=slept.append),
    ):
        result = ct_logger.get_ct_logs("q", max_retries=1, backoff_seconds=10)

    assert result == [{"id": 1}]
    assert slept == [ct_logger.MAX_RETRY_AFTER_SECONDS], "an hour-long Retry-After must be clamped"


def test_retry_after_accepts_an_http_date():
    when = email.utils.formatdate(time.time() + 30, usegmt=True)
    assert 25 <= ct_logger._retry_after_seconds(when) <= 35


def test_retry_after_falls_back_when_unparseable():
    assert ct_logger._retry_after_seconds("not a date") == ct_logger.RATE_LIMIT_BACKOFF_SECONDS


def test_get_ct_logs_rejects_an_oversized_response():
    huge = {"Content-Length": str(ct_logger.MAX_RESPONSE_BYTES + 1)}
    with (
        mock.patch.object(ct_logger.session, "get", return_value=_FakeResponse(200, [{"id": 1}], huge)),
        pytest.raises(RuntimeError, match="over the"),
    ):
        ct_logger.get_ct_logs("q", max_retries=0, backoff_seconds=0.01)


def test_get_ct_logs_for_label_keeps_entries_with_a_null_id():
    def fake_get_ct_logs(query, **_kwargs):
        if "fun" in query:
            return [
                {"id": None, "common_name": "a.com"},
                {"id": None, "common_name": "b.com"},
            ]
        return []

    with mock.patch.object(ct_logger, "get_ct_logs", side_effect=fake_get_ct_logs):
        result = ct_logger.get_ct_logs_for_label("fun")

    names = sorted(entry["common_name"] for entry in result)
    assert names == ["a.com", "b.com"], "null ids must not collapse onto one merge slot"


def test_get_ct_logs_for_label_caps_the_number_of_queries():
    seen = []

    def fake_get_ct_logs(query, **_kwargs):
        seen.append(query)
        return []

    with mock.patch.object(ct_logger, "get_ct_logs", side_effect=fake_get_ct_logs):
        ct_logger.get_ct_logs_for_label("a" * 40)

    assert len(seen) == ct_logger.MAX_QUERY_VARIANTS


def test_get_ct_logs_for_label_raises_when_every_query_fails():
    def fake_get_ct_logs(_query, **_kwargs):
        raise RuntimeError("crt.sh request failed")

    with (
        mock.patch.object(ct_logger, "get_ct_logs", side_effect=fake_get_ct_logs),
        pytest.raises(RuntimeError, match="all 3 crt.sh queries"),
    ):
        ct_logger.get_ct_logs_for_label("fun")


def test_log_domains_handles_explicit_null_fields():
    data = [
        {"common_name": None, "name_value": "foo.example.com"},
        {"common_name": "bar.example.com", "name_value": None},
        {"common_name": None, "name_value": None},
        {"common_name": "baz.example.com", "name_value": "baz.example.com\n*.baz.example.com"},
    ]
    assert ct_logger.log_domains(data) == ["bar.example.com", "baz.example.com", "foo.example.com"]


def test_log_domains_dedupes_and_strips_wildcards():
    data = [{"common_name": "*.example.com", "name_value": "example.com\n*.example.com"}]
    assert ct_logger.log_domains(data) == ["example.com"]
