import ipaddress
import re
import socket
from urllib.parse import urljoin, urlsplit

import requests
from sentence_transformers import util

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TYPES = ("text/html", "application/xhtml+xml")

# This runs against domains out of Certificate Transparency logs, which anyone
# can write to by getting a certificate issued, so the fetch is untrusted input
# from the operator's network. The caps and the public-address check are the
# boundary.
MAX_FETCH_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
USER_AGENT = "swat/0.1 (+https://github.com/r0075h3ll/Swat)"

session = requests.session()
session.headers["User-Agent"] = USER_AGENT


def _resolves_to_public_address(host: str) -> bool:
    """False unless every address host resolves to is a routable public one, so
    a CT entry or a redirect cannot aim the request at loopback, link-local
    (169.254.169.254), private, or otherwise reserved space."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False

    if not infos:
        return False

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global or address.is_multicast:
            return False
    return True


def _charset(content_type: str) -> str | None:
    for param in content_type.split(";")[1:]:
        key, _, value = param.partition("=")
        if key.strip().lower() == "charset":
            return value.strip().strip("\"'") or None
    return None


def _get(url: str, timeout: int) -> tuple[bytes, str] | None:
    """Follow url by hand so every hop is checked, and cap the body we buffer.
    Returns (body, content_type), or None if the fetch is not safe or did not
    produce a usable response."""
    for _ in range(MAX_REDIRECTS + 1):
        host = urlsplit(url).hostname
        if not host or not _resolves_to_public_address(host):
            return None

        try:
            response = session.get(url, timeout=timeout, stream=True, allow_redirects=False)
        except requests.exceptions.RequestException:
            return None

        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                return None
            url = urljoin(url, location)
            continue

        if response.status_code >= 400:
            response.close()
            return None

        chunks, size = [], 0
        for chunk in response.iter_content(8192):
            size += len(chunk)
            if size > MAX_FETCH_BYTES:
                response.close()
                return None
            chunks.append(chunk)
        response.close()

        return b"".join(chunks), response.headers.get("Content-Type", "")

    return None


def fetch_text(target: str, timeout: int = 10) -> str | None:
    urls = [target] if "://" in target else [f"https://{target}", f"http://{target}"]

    for url in urls:
        fetched = _get(url, timeout)
        if fetched is None:
            continue

        body, content_type = fetched
        if content_type.split(";", 1)[0].strip().lower() not in _HTML_TYPES:
            continue

        # requests would decode a text/html with no charset as ISO-8859-1 and
        # mangle every UTF-8 page before it reached the model.
        try:
            html = body.decode(_charset(content_type) or "utf-8", errors="replace")
        except LookupError:
            html = body.decode("utf-8", errors="replace")

        text = _TAG_RE.sub(" ", _SCRIPT_STYLE_RE.sub(" ", html))
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text or None

    return None


def content_similarity(model, reference_embedding, domain: str) -> float | None:
    page_text = fetch_text(domain)
    if not page_text:
        return None

    page_embedding = model.encode(page_text)
    return round(util.cos_sim(reference_embedding, page_embedding).item(), 4)
