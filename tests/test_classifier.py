from unittest import mock

import requests

from swat import classifier


class _FakeResponse:
    def __init__(self, status_code=200, body=b"", headers=None, read_error=None):
        self.status_code = status_code
        self._body = body
        self._read_error = read_error
        self.headers = headers or {"Content-Type": "text/html"}
        self.closed = False

    @property
    def is_redirect(self):
        return "Location" in self.headers

    @property
    def is_permanent_redirect(self):
        return False

    def iter_content(self, chunk_size):
        if self._read_error:
            raise self._read_error
        yield self._body

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _html(body: str, **headers) -> _FakeResponse:
    return _FakeResponse(200, body.encode(), {"Content-Type": "text/html", **headers})


def _fetching(get, public=True):
    """Patch out the network: the response factory, and the DNS check that keeps
    a CT-supplied domain from aiming the request at private space."""
    return (
        mock.patch.object(classifier.session, "get", side_effect=get),
        mock.patch.object(classifier, "_resolves_to_public_address", return_value=public),
    )


def test_fetch_text_treats_a_truncated_body_as_unreachable():
    """A reset mid-read raises from iter_content, not from get(). These pages
    are attacker-supplied, so this must not propagate out of find_lookalikes."""
    truncated = _FakeResponse(200, b"", read_error=requests.exceptions.ChunkedEncodingError("reset"))

    with (
        _fetching(lambda *a, **k: truncated)[0],
        _fetching(lambda *a, **k: truncated)[1],
    ):
        assert classifier.fetch_text("paypal.com") is None

    assert truncated.closed, "the response must be released even when the read fails"


def test_fetch_text_falls_back_to_http_after_a_truncated_https_body():
    seen = []

    def fake_get(url, **_kwargs):
        seen.append(url)
        if url.startswith("https"):
            return _FakeResponse(200, b"", read_error=requests.exceptions.ConnectionError("reset"))
        return _html("<p>plain http works</p>")

    with _fetching(fake_get)[0], _fetching(fake_get)[1]:
        assert classifier.fetch_text("example.com") == "plain http works"

    assert seen == ["https://example.com", "http://example.com"]


def test_fetch_text_strips_tags_scripts_and_styles():
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><script>alert(1)</script><h1>Welcome  to PayPal</h1>"
        "<p>Send money  fast.</p></body></html>"
    )
    get, public = _fetching(lambda *_a, **_k: _html(html))
    with get, public:
        text = classifier.fetch_text("paypal.com")

    assert text == "Welcome to PayPal Send money fast."


def test_fetch_text_falls_back_from_https_to_http():
    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        if url.startswith("https"):
            raise requests.exceptions.ConnectionError("no https")
        return _html("<p>fallback content</p>")

    get, public = _fetching(fake_get)
    with get, public:
        text = classifier.fetch_text("parked-domain.com")

    assert calls == ["https://parked-domain.com", "http://parked-domain.com"]
    assert text == "fallback content"


def test_fetch_text_returns_none_when_both_schemes_fail():
    get, public = _fetching(requests.exceptions.ConnectionError("dead"))
    with get, public:
        assert classifier.fetch_text("dead-domain.com") is None


def test_fetch_text_returns_none_on_empty_body():
    get, public = _fetching(lambda *_a, **_k: _html("   <html></html>  "))
    with get, public:
        assert classifier.fetch_text("empty.com") is None


def test_fetch_text_uses_url_as_is_when_scheme_present():
    get, public = _fetching(lambda *_a, **_k: _html("<p>hi</p>"))
    with get as mocked, public:
        classifier.fetch_text("https://example.com/page")
    assert mocked.call_args.args == ("https://example.com/page",)
    assert mocked.call_args.kwargs["timeout"] == 10


def test_fetch_text_refuses_a_non_public_destination():
    """A CT entry naming an internal host must not be fetched from the operator's network."""
    get, public = _fetching(lambda *_a, **_k: _html("<p>internal</p>"), public=False)
    with get as mocked, public:
        assert classifier.fetch_text("metadata.internal") is None
    assert not mocked.called


def test_fetch_text_refuses_a_redirect_into_non_public_space():
    """A public host that redirects to 169.254.169.254 must not be followed there."""

    def fake_get(url, **_kwargs):
        if "redirector" in url:
            return _FakeResponse(302, b"", {"Location": "http://169.254.169.254/latest/meta-data/"})
        return _html("<p>secret</p>")

    def public_only(host):
        return host != "169.254.169.254"

    with (
        mock.patch.object(classifier.session, "get", side_effect=fake_get),
        mock.patch.object(classifier, "_resolves_to_public_address", side_effect=public_only),
    ):
        assert classifier.fetch_text("redirector.com") is None


def test_fetch_text_follows_a_redirect_to_a_public_host():
    def fake_get(url, **_kwargs):
        if url.startswith("https://redirector"):
            return _FakeResponse(302, b"", {"Location": "https://elsewhere.com/"})
        return _html("<p>arrived</p>")

    get, public = _fetching(fake_get)
    with get, public:
        assert classifier.fetch_text("redirector.com") == "arrived"


def test_fetch_text_stops_after_too_many_redirects():
    def fake_get(_url, **_kwargs):
        return _FakeResponse(302, b"", {"Location": "https://loop.com/"})

    get, public = _fetching(fake_get)
    with get as mocked, public:
        assert classifier.fetch_text("loop.com") is None
    # Both the https and the http scheme fallback are tried, and each stops at
    # the redirect cap.
    assert mocked.call_count == 2 * (classifier.MAX_REDIRECTS + 1)


def test_fetch_text_rejects_a_body_over_the_cap(monkeypatch):
    monkeypatch.setattr(classifier, "MAX_FETCH_BYTES", 10)
    oversized = _FakeResponse(200, b"x" * 10, {"Content-Type": "text/html"})
    oversized.iter_content = lambda _chunk_size: iter([b"x" * 5, b"x" * 5, b"x" * 5])

    get, public = _fetching(lambda *_a, **_k: oversized)
    with get, public:
        assert classifier.fetch_text("big.com") is None


def test_fetch_text_rejects_a_non_html_content_type():
    binary = _FakeResponse(200, b"\x00\x01\x02", {"Content-Type": "application/octet-stream"})
    get, public = _fetching(lambda *_a, **_k: binary)
    with get, public:
        assert classifier.fetch_text("binary.com") is None


def test_fetch_text_decodes_utf8_when_the_header_names_no_charset():
    get, public = _fetching(lambda *_a, **_k: _html("<p>café über</p>"))
    with get, public:
        text = classifier.fetch_text("utf8.com")
    assert "café über" in text


def test_fetch_text_honours_an_explicit_charset():
    latin = "<p>café</p>".encode("latin-1")
    response = _FakeResponse(200, latin, {"Content-Type": "text/html; charset=iso-8859-1"})
    get, public = _fetching(lambda *_a, **_k: response)
    with get, public:
        text = classifier.fetch_text("latin1.com")
    assert "café" in text


def test_resolves_to_public_address_rejects_private_ranges():
    for host in ("127.0.0.1", "169.254.169.254", "10.0.0.1", "192.168.1.1", "::1"):
        assert classifier._resolves_to_public_address(host) is False, host


def test_resolves_to_public_address_rejects_multicast():
    assert classifier._resolves_to_public_address("224.0.0.1") is False


def test_resolves_to_public_address_rejects_unresolvable_hosts():
    assert classifier._resolves_to_public_address("nonexistent.invalid") is False


def test_content_similarity_returns_none_when_page_unreachable():
    with mock.patch("swat.classifier.fetch_text", return_value=None):
        result = classifier.content_similarity(
            model=object(), reference_embedding=object(), domain="dead.com"
        )
    assert result is None


def test_content_similarity_only_encodes_the_page_text_not_the_reference():
    encode_calls = []

    class FakeModel:
        def encode(self, text):
            encode_calls.append(text)
            return text

    class FakeUtil:
        @staticmethod
        def cos_sim(a, b):
            class _Score:
                def item(self_inner):
                    return 0.987654

            assert a == "reference-embedding"
            assert b == "page text"
            return _Score()

    with (
        mock.patch("swat.classifier.fetch_text", return_value="page text"),
        mock.patch("swat.classifier.util", FakeUtil()),
    ):
        score = classifier.content_similarity(FakeModel(), "reference-embedding", "domain.com")

    assert score == 0.9877
    assert encode_calls == ["page text"], "reference must not be re-encoded per call"
