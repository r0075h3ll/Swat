import pytest

from swat.domain import normalize_domain, search_label


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
        # IDN targets are encoded to their A-label so the rest of the pipeline
        # only ever sees ASCII.
        ("пример.рф", "xn--e1afmkfd.xn--p1ai"),
        ("XN--PYPL-53D.com", "xn--pypl-53d.com"),
        # IDNA2003's nameprep maps these to different labels; IDNA2008 does not.
        # A wrong mapping here means monitoring a domain nobody asked about.
        ("faß.de", "xn--fa-hia.de"),
        ("straße.de", "xn--strae-oqa.de"),
        ("ẞ.de", "xn--zca.de"),
        ("ς.gr", "xn--3xa.gr"),
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
        # IP literals pass the label regex but have no brand label to search for.
        "1.2.3.4",
        "http://192.168.1.1:8080/admin",
        "::1",
    ],
)
def test_rejects_invalid_input(raw):
    with pytest.raises(ValueError):
        normalize_domain(raw)


def test_error_message_does_not_leak_internal_scheme_prefix():
    with pytest.raises(ValueError, match=r"^'paypal' does not look"):
        normalize_domain("paypal")


def test_rejects_ip_literal_with_a_specific_message():
    with pytest.raises(ValueError, match=r"^'1\.2\.3\.4' is an IP address"):
        normalize_domain("1.2.3.4")


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        # The search term must be the registrable label, not the first label.
        ("example.com", "example"),
        ("www.example.com", "example"),
        ("a.b.c.example.com", "example"),
        ("sub.example.co.uk", "example"),
        ("example.co.uk", "example"),
        ("example-brand.com", "example-brand"),
        ("xn--e1afmkfd.xn--p1ai", "xn--e1afmkfd"),
        # TLD the public suffix list does not know: fall back to the leading label.
        ("example.zzzzz", "example"),
    ],
)
def test_search_label_returns_registrable_label(domain, expected):
    assert search_label(domain) == expected
