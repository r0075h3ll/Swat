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
        # Registered ASCII hostnames that IDNA2008 refuses (rule forbidding '--'
        # in positions 3-4 of a non-'xn--' label). Pure-ASCII hosts skip idna
        # entirely so this stays accepted.
        ("ex--ample.com", "ex--ample.com"),
        ("a--b.example.com", "a--b.example.com"),
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


def test_rejects_ipv6_literal_with_the_same_specific_message():
    """urlsplit strips the brackets, so the host reaching the validators is
    just '::1' with no bracket context. It has to be recognised as an IP
    before the label validators see it and raise the generic 'not a fully
    qualified domain' message."""
    with pytest.raises(ValueError, match=r"is an IP address"):
        normalize_domain("[::1]")


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
        # TLDs the public suffix list does not know. tldextract calls the last
        # label the "domain" when the suffix is empty, which would give the
        # wrong label for anything past two labels; fall back to the
        # second-to-last label so the brand comes out instead.
        ("example.zzzzz", "example"),
        ("www.paypal.invalidtld", "paypal"),
        ("a.b.paypal.invalidtld", "paypal"),
    ],
)
def test_search_label_returns_registrable_label(domain, expected):
    assert search_label(domain) == expected
