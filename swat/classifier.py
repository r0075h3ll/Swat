import ipaddress
import os
import re
import socket
from urllib.parse import urljoin, urlsplit

import requests
from typesafe_sdk import Choice, TypeSafeClient

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
# A browser User-Agent, not an API UA, because some sites only serve real
# content to browsers and block script-like agents.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
# r.jina.ai challenges browser User-Agents (Cloudflare "Just a moment" 403) but
# serves programmatic ones, so the jina call needs its own UA rather than the
# browser UA the direct-HTTP probe uses.
JINA_USER_AGENT = "swat/0.1 (+https://github.com/r0075h3ll/Swat)"
JINA_URL = "https://r.jina.ai/"

session = requests.session()
session.headers["User-Agent"] = BROWSER_USER_AGENT

_jev_client: TypeSafeClient | None = None


def _get_jev_client() -> TypeSafeClient:
    global _jev_client
    if _jev_client is None:
        _jev_client = TypeSafeClient(api_key=os.environ.get("JEV_API_KEY"))
    return _jev_client


def _resolves_to_public_address(host: str) -> bool:
    """False unless host resolves only to routable public addresses, so a CT
    entry or a redirect cannot send the request to loopback, link-local
    (169.254.169.254), private, or reserved space."""
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
    Returns (body, content_type), or None if the fetch is unsafe or produced no
    usable response."""
    for _ in range(MAX_REDIRECTS + 1):
        host = urlsplit(url).hostname
        if not host or not _resolves_to_public_address(host):
            return None

        try:
            with session.get(url, timeout=timeout, stream=True, allow_redirects=False) as response:
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        return None
                    url = urljoin(url, location)
                    continue

                if response.status_code >= 400:
                    return None

                chunks, size = [], 0
                for chunk in response.iter_content(8192):
                    size += len(chunk)
                    if size > MAX_FETCH_BYTES:
                        return None
                    chunks.append(chunk)

                return b"".join(chunks), response.headers.get("Content-Type", "")
        except requests.exceptions.RequestException:
            # A truncated or reset body raises here rather than at get(). These
            # pages are attacker-supplied, so a partial read is the expected
            # case, not a rare one, and must not abort the run.
            return None

    return None


def http_probe(domain: str, timeout: int = 10) -> bool:
    """True if ``domain`` answers an HTTP(S) request with any status code. A
    host that responds (even 4xx/5xx) is live. DNS or connection failure means
    it is not. Uses the browser User-Agent, so sites that block script-like
    agents still answer."""
    for url in (f"https://{domain}", f"http://{domain}"):
        host = urlsplit(url).hostname
        if not host or not _resolves_to_public_address(host):
            continue
        try:
            with session.get(url, timeout=timeout, stream=True, allow_redirects=False) as _response:
                return True
        except requests.exceptions.RequestException:
            continue
    return False


def jina_fetch(url: str, timeout: int = 30) -> str | None:
    """Fetch the rendered text of ``url`` through r.jina.ai. Returns the text,
    or None on any request or HTTP error."""
    try:
        response = session.get(JINA_URL + url, timeout=timeout, headers={"User-Agent": JINA_USER_AGENT})
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None
    text = response.text.strip()
    return text or None


def fetch_text(target: str, timeout: int = 10) -> str | None:
    urls = [target] if "://" in target else [f"https://{target}", f"http://{target}"]

    # r.jina.ai first: it renders the page and reaches hosts that block a direct
    # browser-like request.
    jina = jina_fetch(urls[0])
    if jina:
        return jina

    # Fallback: direct HTTP(S) with the same public-address and byte caps.
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


def domain_impersonation(target_label: str, candidate_label: str) -> bool:
    """Jev yes/no: does the candidate domain label look like it's impersonating
    or typosquatting the target brand label?"""
    client = _get_jev_client()
    response = client.system_one(
        state={
            "target_brand": target_label,
            "candidate_domain": candidate_label,
        },
        questions={
            "impersonates": Choice(
                instructions=(
                    "Is the candidate domain name likely a look-alike or typosquat of the "
                    "target brand? Consider character substitutions, added/removed words, "
                    "partial brand fragments, and semantic similarity."
                ),
                criteria={
                    "yes": "the candidate plausibly impersonates or typosquats the target brand",
                    "no": "the candidate is unrelated to the target brand",
                },
            ),
        },
    )
    return response.answers["impersonates"].choice == "yes"


def content_impersonation(reference_text: str, candidate_text: str) -> bool:
    """Jev yes/no: is the candidate page a clone or phishing imitation of the
    reference page?"""
    client = _get_jev_client()
    response = client.system_one(
        state={
            "reference_page": reference_text[:4000],
            "candidate_page": candidate_text[:4000],
        },
        questions={
            "clone": Choice(
                instructions=(
                    "Is the candidate page likely a clone or phishing imitation of the "
                    "reference page? Consider similar branding, product offerings, layout "
                    "patterns, and copied text. Legitimate competitors offering similar "
                    "services are not clones."
                ),
                criteria={
                    "yes": "the candidate page impersonates or clones the reference page",
                    "no": "the candidate page is unrelated or a legitimate different business",
                },
            ),
        },
    )
    return response.answers["clone"].choice == "yes"
