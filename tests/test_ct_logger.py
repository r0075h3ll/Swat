import email.utils
import json as _json
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
    def __init__(self, status_code=200, body=None, headers=None, chunks=None):
        self.status_code = status_code
        self._body = body
        # Explicit chunks let a test emit a chunked, no-Content-Length body of
        # any size. Absent, the body is JSON-serialised and yielded as one
        # chunk, which is enough for the happy path.
        self._chunks = chunks
        self.headers = headers or {}
        self.iter_started = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def iter_content(self, _chunk_size):
        self.iter_started = True
        if self._chunks is not None:
            yield from self._chunks
            return
        if self._body is None:
            raise ValueError("empty body")
        yield _json.dumps(self._body).encode()


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


def test_retry_after_rejects_nan():
    # min/max pass NaN through, and time.sleep(nan) raises ValueError from
    # outside the retry loop's except clause, which would kill the whole run.
    assert ct_logger._retry_after_seconds("nan") == ct_logger.RATE_LIMIT_BACKOFF_SECONDS


def test_get_ct_logs_uses_streaming_mode():
    """The body is bounded by counting bytes off iter_content, which only works
    when requests is not asked to buffer the response whole."""
    response = _FakeResponse(200, [{"id": 1}])
    with mock.patch.object(ct_logger.session, "get", return_value=response) as mocked:
        assert ct_logger.get_ct_logs("q", max_retries=0) == [{"id": 1}]
    assert mocked.call_args.kwargs["stream"] is True


def test_get_ct_logs_rejects_a_chunked_oversized_response():
    """crt.sh returns wildcard queries chunked with no Content-Length header.
    A header check cannot bound those responses, and the wildcard is exactly
    the query the cap exists for. The byte-count loop off iter_content is
    what makes the cap effective for that case."""
    # One chunk over the cap: any implementation that only checked
    # Content-Length would sail past this and json.loads the whole body.
    over_cap = b"X" * (ct_logger.MAX_RESPONSE_BYTES + 1)
    response = _FakeResponse(200, chunks=[over_cap])
    with (
        mock.patch.object(ct_logger.session, "get", return_value=response),
        pytest.raises(RuntimeError, match="exceeded"),
    ):
        ct_logger.get_ct_logs("q", max_retries=0, backoff_seconds=0.01)


def test_get_ct_logs_aborts_mid_stream_once_the_cap_is_crossed():
    """The loop must stop as soon as the running total crosses the cap. A body
    big enough to matter is never fully buffered."""
    chunk_size = 1024
    # A little more than 8 MiB worth of 1 KiB chunks, so the cap will trip
    # before the iterator is exhausted.
    n_chunks = (ct_logger.MAX_RESPONSE_BYTES // chunk_size) + 100
    yielded = {"n": 0}

    def chunk_stream():
        for _ in range(n_chunks):
            yielded["n"] += 1
            yield b"X" * chunk_size

    response = _FakeResponse(200, chunks=chunk_stream())
    with (
        mock.patch.object(ct_logger.session, "get", return_value=response),
        pytest.raises(RuntimeError, match="exceeded"),
    ):
        ct_logger.get_ct_logs("q", max_retries=0, backoff_seconds=0.01)

    # Reader should stop the first chunk past the cap, not drain to the end.
    assert yielded["n"] < n_chunks, "iter_content should be aborted once the cap is crossed"


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
