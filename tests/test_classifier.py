from unittest import mock

import requests

from swat import classifier


class _FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")


def test_fetch_text_strips_tags_scripts_and_styles():
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><script>alert(1)</script><h1>Welcome  to PayPal</h1>"
        "<p>Send money  fast.</p></body></html>"
    )
    with mock.patch("swat.classifier.requests.get", return_value=_FakeResponse(200, html)):
        text = classifier.fetch_text("paypal.com")

    assert text == "Welcome to PayPal Send money fast."


def test_fetch_text_falls_back_from_https_to_http():
    calls = []

    def fake_get(url, timeout=10):
        calls.append(url)
        if url.startswith("https"):
            raise requests.exceptions.ConnectionError("no https")
        return _FakeResponse(200, "<p>fallback content</p>")

    with mock.patch("swat.classifier.requests.get", side_effect=fake_get):
        text = classifier.fetch_text("parked-domain.com")

    assert calls == ["https://parked-domain.com", "http://parked-domain.com"]
    assert text == "fallback content"


def test_fetch_text_returns_none_when_both_schemes_fail():
    with mock.patch("swat.classifier.requests.get", side_effect=requests.exceptions.ConnectionError("dead")):
        assert classifier.fetch_text("dead-domain.com") is None


def test_fetch_text_returns_none_on_empty_body():
    with mock.patch("swat.classifier.requests.get", return_value=_FakeResponse(200, "   <html></html>  ")):
        assert classifier.fetch_text("empty.com") is None


def test_fetch_text_uses_url_as_is_when_scheme_present():
    with mock.patch("swat.classifier.requests.get", return_value=_FakeResponse(200, "<p>hi</p>")) as mocked:
        classifier.fetch_text("https://example.com/page")
    mocked.assert_called_once_with("https://example.com/page", timeout=10)


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
