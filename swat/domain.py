import ipaddress
import re
from urllib.parse import urlsplit

import idna
import tldextract

_LABEL = r"(?!-)[a-z0-9-]{1,63}(?<!-)"
_DOMAIN_RE = re.compile(rf"^{_LABEL}(\.{_LABEL})+$")
_INVALID = "does not look like a fully qualified domain (e.g. example.com)"

# Bundled public suffix snapshot. suffix_list_urls=() keeps tldextract from
# fetching the list over the network on first use.
_extract = tldextract.TLDExtract(suffix_list_urls=())


def normalize_domain(raw: str) -> str:
    """Normalize CLI input into a bare ASCII (A-label) domain, e.g.
    'https://Example.com/path' -> 'example.com', 'пример.рф' ->
    'xn--e1afmkfd.xn--p1ai', 'faß.de' -> 'xn--fa-hia.de'.

    Raises ValueError on anything that doesn't reduce to a plausible fully
    qualified domain name (argparse turns that into a clean usage error).
    """
    original = raw.strip()
    if not original:
        raise ValueError("domain cannot be empty")

    parseable = original if "://" in original else f"//{original}"
    host = urlsplit(parseable).hostname

    if not host:
        raise ValueError(f"could not extract a hostname from {original!r}")

    host = host.rstrip(".")

    try:
        # idna.encode(uts46=True), not str.encode("idna"). The stdlib codec is
        # IDNA2003 and rewrites labels: faß.de becomes fass.de, so the tool
        # would monitor a different domain than the one asked for.
        host = idna.encode(host, uts46=True).decode("ascii")
    except idna.IDNAError:
        raise ValueError(f"{original!r} {_INVALID}") from None

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError(f"{original!r} is an IP address, not a domain name")

    if not _DOMAIN_RE.match(host):
        raise ValueError(f"{original!r} {_INVALID}")

    return host


def search_label(domain: str) -> str:
    """The registrable label to hand crt.sh, e.g. 'www.paypal.com' -> 'paypal'.

    Falls back to the leading label when the public suffix list does not
    recognise the TLD, which is all splitting on '.' could do anyway.
    """
    registrable = _extract(domain).top_domain_under_public_suffix
    return registrable.partition(".")[0] or domain.split(".")[0]
