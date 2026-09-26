import re
from urllib.parse import urlsplit

_LABEL = r"(?!-)[a-z0-9-]{1,63}(?<!-)"
_DOMAIN_RE = re.compile(rf"^{_LABEL}(\.{_LABEL})+$")


def normalize_domain(raw: str) -> str:
    """Normalize CLI input into a bare domain, e.g. 'https://Example.com/path' -> 'example.com'.

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

    if not _DOMAIN_RE.match(host):
        raise ValueError(f"{original!r} does not look like a fully qualified domain (e.g. example.com)")

    return host
