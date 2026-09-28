import pytest

from swat.domain import normalize_domain


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("example.com", "example.com"),
        ("Example.COM", "example.com"),
        ("  example.com  ", "example.com"),
        ("https://example.com/path?q=1", "example.com"),
        ("http://user:pass@Example.com:8080/x", "example.com"),
        ("example.com.", "example.com"),
        ("example.com/path", "example.com"),
        ("example.com:8080", "example.com"),
        ("sub.example.co.uk", "sub.example.co.uk"),
    ],
)
def test_normalizes_to_bare_domain(raw, expected):
    assert normalize_domain(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "paypal",
        "not a domain",
        "-example.com",
        "example-.com",
        "https://",
        "a..b.com",
        "///",
    ],
)
def test_rejects_invalid_input(raw):
    with pytest.raises(ValueError):
        normalize_domain(raw)


def test_error_message_does_not_leak_internal_scheme_prefix():
    with pytest.raises(ValueError, match=r"^'paypal' does not look"):
        normalize_domain("paypal")
