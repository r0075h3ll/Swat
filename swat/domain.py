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

    # IP-address check runs first: urlsplit strips the brackets from a v6
    # literal, so '::1' reaches idna.encode looking like a malformed label and
    # the specific "is an IP address" message would never fire.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError(f"{original!r} is an IP address, not a domain name")

    # IDNA2008 is stricter than the letter-digit-hyphen rule an ASCII host has
    # to obey anyway: it forbids '--' in the third and fourth position of a
    # non-'xn--' label, so 'ex--ample.com' is a registered, resolvable domain
    # that IDNA2008 refuses. Pure-ASCII hosts don't need Unicode processing at
    # all, so skip idna for them and lean on _DOMAIN_RE for structure.
    if not host.isascii():
        try:
            # idna.encode(uts46=True), not str.encode("idna"). The stdlib codec
            # is IDNA2003 and rewrites labels: faß.de becomes fass.de, so the
            # tool would monitor a different domain than the one asked for.
            host = idna.encode(host, uts46=True).decode("ascii")
        except idna.IDNAError:
            raise ValueError(f"{original!r} {_INVALID}") from None

    if not _DOMAIN_RE.match(host):
        raise ValueError(f"{original!r} {_INVALID}")

    return host


def search_label(domain: str) -> str:
    """The registrable label to hand crt.sh, e.g. 'www.paypal.com' -> 'paypal'.

    On an unknown TLD (public suffix list has no matching suffix), tldextract
    treats the last label as the "domain" and the rest as "subdomain", which
    would give 'invalidtld' for 'www.paypal.invalidtld' and reproduce exactly
    the bug of picking the wrong label. Falls back to the second-to-last label
    of the input instead, which is a better guess and matches the shape the
    public-suffix path returns.
    """
    extracted = _extract(domain)
    if extracted.suffix:
        return extracted.domain
    labels = domain.split(".")
    return labels[-2] if len(labels) >= 2 else labels[0]


def registrable_domain(domain: str) -> str:
    """The public-suffix-aware registrable domain: 'www.paypal.com' -> 'paypal.com'.

    Falls back to the input unchanged when the TLD is not in the list.
    """
    return _extract(domain).top_domain_under_public_suffix or domain
