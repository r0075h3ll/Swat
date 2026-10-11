from unittest import mock

import requests

from swat import ct_logger


class _FakeResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._data


def test_cert_exists_true_when_apex_has_entries():
    data = [{"sub": "acmecorp.co"}, {"sub": "api.acmecorp.co"}]
    with mock.patch.object(ct_logger.session, "get", return_value=_FakeResponse(200, data)):
        assert ct_logger.cert_exists("acmecorp.co") is True


def test_cert_exists_false_when_apex_has_no_entries():
    with mock.patch.object(ct_logger.session, "get", return_value=_FakeResponse(200, [])):
        assert ct_logger.cert_exists("acmecorp.co") is False


def test_cert_exists_false_on_http_error():
    with mock.patch.object(ct_logger.session, "get", return_value=_FakeResponse(502)):
        assert ct_logger.cert_exists("acmecorp.co") is False


def test_cert_exists_false_on_request_error():
    with mock.patch.object(ct_logger.session, "get", side_effect=requests.exceptions.ConnectionError("down")):
        assert ct_logger.cert_exists("acmecorp.co") is False


def test_cert_exists_false_on_invalid_json():
    with mock.patch.object(ct_logger.session, "get", return_value=_FakeResponse(200, "not json")):
        assert ct_logger.cert_exists("acmecorp.co") is False


def test_cert_exists_queries_the_registrable_apex():
    with mock.patch.object(ct_logger.session, "get", return_value=_FakeResponse(200, [])) as get:
        ct_logger.cert_exists("www.acmecorp.co")
    assert get.call_args.kwargs["params"] == {"apex": "acmecorp.co", "format": "json"}
