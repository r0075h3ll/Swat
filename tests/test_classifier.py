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

    @property
    def text(self):
        return self._body.decode("utf-8", errors="replace")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

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
    """Patch out the network: the response factory, the DNS check that keeps
    a CT-supplied domain from aiming the request at private space, and the jina
    proxy so the fetch_text tests below exercise the direct-HTTP fallback."""
    return (
        mock.patch.object(classifier.session, "get", side_effect=get),
        mock.patch.object(classifier, "_resolves_to_public_address", return_value=public),
        mock.patch.object(classifier, "jina_fetch", return_value=None),
    )


def _mock_jev_client(choices: dict[str, str]):
    """Return a mock Jev client whose system_one returns fixed Choice answers."""

    class _FakeChoiceAnswer:
        def __init__(self, choice):
            self.choice = choice

    class _FakeResponse:
        def __init__(self, answer_map):
            self.answers = {k: _FakeChoiceAnswer(v) for k, v in answer_map.items()}

    client = mock.Mock()
    client.system_one.return_value = _FakeResponse(choices)
    return client


# --- fetch_text tests (HTTP safety boundary, direct-HTTP fallback) ---


def test_fetch_text_treats_a_truncated_body_as_unreachable():
    truncated = _FakeResponse(200, b"", read_error=requests.exceptions.ChunkedEncodingError("reset"))

    get, public, jina = _fetching(lambda *a, **k: truncated)
    with get, public, jina:
        assert classifier.fetch_text("paypal.com") is None

    assert truncated.closed, "the response must be released even when the read fails"


def test_fetch_text_falls_back_to_http_after_a_truncated_https_body():
    seen = []

    def fake_get(url, **_kwargs):
        seen.append(url)
        if url.startswith("https"):
            return _FakeResponse(200, b"", read_error=requests.exceptions.ConnectionError("reset"))
        return _html("<p>plain http works</p>")

    get, public, jina = _fetching(fake_get)
    with get, public, jina:
        assert classifier.fetch_text("example.com") == "plain http works"

    assert seen == ["https://example.com", "http://example.com"]


def test_fetch_text_strips_tags_scripts_and_styles():
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><script>alert(1)</script><h1>Welcome  to PayPal</h1>"
        "<p>Send money  fast.</p></body></html>"
    )
    get, public, jina = _fetching(lambda *_a, **_k: _html(html))
    with get, public, jina:
        text = classifier.fetch_text("paypal.com")

    assert text == "Welcome to PayPal Send money fast."


def test_fetch_text_falls_back_from_https_to_http():
    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        if url.startswith("https"):
            raise requests.exceptions.ConnectionError("no https")
        return _html("<p>fallback content</p>")

    get, public, jina = _fetching(fake_get)
    with get, public, jina:
        text = classifier.fetch_text("parked-domain.com")

    assert calls == ["https://parked-domain.com", "http://parked-domain.com"]
    assert text == "fallback content"


def test_fetch_text_returns_none_when_both_schemes_fail():
    get, public, jina = _fetching(requests.exceptions.ConnectionError("dead"))
    with get, public, jina:
        assert classifier.fetch_text("dead-domain.com") is None


def test_fetch_text_returns_none_on_empty_body():
    get, public, jina = _fetching(lambda *_a, **_k: _html("   <html></html>  "))
    with get, public, jina:
        assert classifier.fetch_text("empty.com") is None


def test_fetch_text_uses_url_as_is_when_scheme_present():
    get, public, jina = _fetching(lambda *_a, **_k: _html("<p>hi</p>"))
    with get as mocked, public, jina:
        classifier.fetch_text("https://example.com/page")
    assert mocked.call_args.args == ("https://example.com/page",)
    assert mocked.call_args.kwargs["timeout"] == 10


def test_fetch_text_refuses_a_non_public_destination():
    get, public, jina = _fetching(lambda *_a, **_k: _html("<p>internal</p>"), public=False)
    with get as mocked, public, jina:
        assert classifier.fetch_text("metadata.internal") is None
    assert not mocked.called


def test_fetch_text_refuses_a_redirect_into_non_public_space():
    def fake_get(url, **_kwargs):
        if "redirector" in url:
            return _FakeResponse(302, b"", {"Location": "http://169.254.169.254/latest/meta-data/"})
        return _html("<p>secret</p>")

    def public_only(host):
        return host != "169.254.169.254"

    with (
        mock.patch.object(classifier.session, "get", side_effect=fake_get),
        mock.patch.object(classifier, "_resolves_to_public_address", side_effect=public_only),
        mock.patch.object(classifier, "jina_fetch", return_value=None),
    ):
        assert classifier.fetch_text("redirector.com") is None


def test_fetch_text_follows_a_redirect_to_a_public_host():
    def fake_get(url, **_kwargs):
        if url.startswith("https://redirector"):
            return _FakeResponse(302, b"", {"Location": "https://elsewhere.com/"})
        return _html("<p>arrived</p>")

    get, public, jina = _fetching(fake_get)
    with get, public, jina:
        assert classifier.fetch_text("redirector.com") == "arrived"


def test_fetch_text_stops_after_too_many_redirects():
    def fake_get(_url, **_kwargs):
        return _FakeResponse(302, b"", {"Location": "https://loop.com/"})

    get, public, jina = _fetching(fake_get)
    with get as mocked, public, jina:
        assert classifier.fetch_text("loop.com") is None
    assert mocked.call_count == 2 * (classifier.MAX_REDIRECTS + 1)


def test_fetch_text_rejects_a_body_over_the_cap(monkeypatch):
    monkeypatch.setattr(classifier, "MAX_FETCH_BYTES", 10)
    oversized = _FakeResponse(200, b"x" * 10, {"Content-Type": "text/html"})
    oversized.iter_content = lambda _chunk_size: iter([b"x" * 5, b"x" * 5, b"x" * 5])

    get, public, jina = _fetching(lambda *_a, **_k: oversized)
    with get, public, jina:
        assert classifier.fetch_text("big.com") is None


def test_fetch_text_rejects_a_non_html_content_type():
    binary = _FakeResponse(200, b"\x00\x01\x02", {"Content-Type": "application/octet-stream"})
    get, public, jina = _fetching(lambda *_a, **_k: binary)
    with get, public, jina:
        assert classifier.fetch_text("binary.com") is None


def test_fetch_text_decodes_utf8_when_the_header_names_no_charset():
    get, public, jina = _fetching(lambda *_a, **_k: _html("<p>café über</p>"))
    with get, public, jina:
        text = classifier.fetch_text("utf8.com")
    assert "café über" in text


def test_fetch_text_honours_an_explicit_charset():
    latin = "<p>café</p>".encode("latin-1")
    response = _FakeResponse(200, latin, {"Content-Type": "text/html; charset=iso-8859-1"})
    get, public, jina = _fetching(lambda *_a, **_k: response)
    with get, public, jina:
        text = classifier.fetch_text("latin1.com")
    assert "café" in text


# --- http_probe tests ---


def test_http_probe_returns_true_for_any_http_status():
    get, public, jina = _fetching(lambda *_a, **_k: _FakeResponse(503, b""))
    with get, public, jina:
        assert classifier.http_probe("example.com") is True


def test_http_probe_returns_false_when_both_schemes_fail():
    get, public, jina = _fetching(requests.exceptions.ConnectionError("dead"))
    with get, public, jina:
        assert classifier.http_probe("dead.com") is False


def test_http_probe_returns_false_for_a_non_public_destination():
    get, public, jina = _fetching(lambda *_a, **_k: _FakeResponse(200), public=False)
    with get as mocked, public, jina:
        assert classifier.http_probe("metadata.internal") is False
    assert not mocked.called


def test_http_probe_uses_the_browser_user_agent_and_tries_https_first():
    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        return _FakeResponse(200, b"")

    get, public, jina = _fetching(fake_get)
    with get, public, jina:
        assert classifier.http_probe("example.com") is True

    assert classifier.session.headers["User-Agent"] == classifier.BROWSER_USER_AGENT
    assert calls == ["https://example.com"]


def test_http_probe_tries_https_then_http():
    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        if url.startswith("https"):
            raise requests.exceptions.ConnectionError("no https")
        return _FakeResponse(200, b"")

    get, public, jina = _fetching(fake_get)
    with get, public, jina:
        assert classifier.http_probe("example.com") is True

    assert calls == ["https://example.com", "http://example.com"]


# --- jina_fetch tests ---


def test_jina_fetch_returns_stripped_text_on_200():
    response = _FakeResponse(200, b"  rendered text  ")
    with mock.patch.object(classifier.session, "get", return_value=response) as get:
        assert classifier.jina_fetch("https://example.com") == "rendered text"

    assert get.call_args.args == (classifier.JINA_URL + "https://example.com",)
    assert get.call_args.kwargs["timeout"] == 30


def test_jina_fetch_returns_none_on_empty_text():
    response = _FakeResponse(200, b"   ")
    with mock.patch.object(classifier.session, "get", return_value=response):
        assert classifier.jina_fetch("https://example.com") is None


def test_jina_fetch_returns_none_on_request_exception():
    with mock.patch.object(
        classifier.session, "get", side_effect=requests.exceptions.ConnectionError("down")
    ):
        assert classifier.jina_fetch("https://example.com") is None


def test_jina_fetch_returns_none_on_http_error():
    response = _FakeResponse(404, b"not found")
    with mock.patch.object(classifier.session, "get", return_value=response):
        assert classifier.jina_fetch("https://example.com") is None


# --- fetch_text + jina interaction tests ---


def test_fetch_text_returns_jina_text_when_jina_succeeds():
    with (
        mock.patch.object(classifier, "jina_fetch", return_value="rendered text") as jina,
        mock.patch.object(classifier, "_get") as _get,
    ):
        assert classifier.fetch_text("example.com") == "rendered text"

    assert jina.call_args.args == ("https://example.com",)
    assert not _get.called


def test_fetch_text_falls_back_to_direct_http_when_jina_returns_none():
    with (
        mock.patch.object(classifier, "jina_fetch", return_value=None) as jina,
        mock.patch.object(classifier, "_get", return_value=(b"<p>direct</p>", "text/html")) as _get,
    ):
        assert classifier.fetch_text("example.com") == "direct"

    assert jina.call_args.args == ("https://example.com",)
    assert _get.call_args.args == ("https://example.com", 10)


def test_resolves_to_public_address_rejects_private_ranges():
    for host in ("127.0.0.1", "169.254.169.254", "10.0.0.1", "192.168.1.1", "::1"):
        assert classifier._resolves_to_public_address(host) is False, host


def test_resolves_to_public_address_rejects_multicast():
    assert classifier._resolves_to_public_address("224.0.0.1") is False


def test_resolves_to_public_address_rejects_unresolvable_hosts():
    assert classifier._resolves_to_public_address("nonexistent.invalid") is False


# --- Jev Choice tests ---


def test_domain_impersonation_returns_true_on_yes():
    client = _mock_jev_client({"impersonates": "yes"})
    with mock.patch("swat.classifier._get_jev_client", return_value=client):
        assert classifier.domain_impersonation("paypal", "paypa1") is True


def test_domain_impersonation_returns_false_on_no():
    client = _mock_jev_client({"impersonates": "no"})
    with mock.patch("swat.classifier._get_jev_client", return_value=client):
        assert classifier.domain_impersonation("paypal", "unrelated") is False


def test_domain_impersonation_passes_labels_as_state():
    client = _mock_jev_client({"impersonates": "yes"})
    with mock.patch("swat.classifier._get_jev_client", return_value=client):
        classifier.domain_impersonation("acmecorp", "acme-corp")
    state = client.system_one.call_args.kwargs["state"]
    assert state["target_brand"] == "acmecorp"
    assert state["candidate_domain"] == "acme-corp"


def test_content_impersonation_returns_true_on_yes():
    client = _mock_jev_client({"clone": "yes"})
    with mock.patch("swat.classifier._get_jev_client", return_value=client):
        assert classifier.content_impersonation("ref", "cand") is True


def test_content_impersonation_truncates_long_texts():
    client = _mock_jev_client({"clone": "no"})
    long_ref = "r" * 10000
    long_cand = "c" * 10000
    with mock.patch("swat.classifier._get_jev_client", return_value=client):
        classifier.content_impersonation(long_ref, long_cand)
    state = client.system_one.call_args.kwargs["state"]
    assert len(state["reference_page"]) == 4000
    assert len(state["candidate_page"]) == 4000


def test_jev_client_is_lazy_singleton(monkeypatch):
    monkeypatch.setattr(classifier, "_jev_client", None)
    call_count = 0

    def fake_init(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1

    with mock.patch.object(classifier.TypeSafeClient, "__init__", fake_init):
        c1 = classifier._get_jev_client()
        c2 = classifier._get_jev_client()

    assert c1 is c2
    assert call_count == 1

    monkeypatch.setattr(classifier, "_jev_client", None)
