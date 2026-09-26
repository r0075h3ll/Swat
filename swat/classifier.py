import re

import requests
from sentence_transformers import util

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def fetch_text(target: str, timeout: int = 10) -> str | None:
    urls = [target] if "://" in target else [f"https://{target}", f"http://{target}"]

    for url in urls:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            continue

        html = _SCRIPT_STYLE_RE.sub(" ", response.text)
        text = _TAG_RE.sub(" ", html)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text or None

    return None


def content_similarity(model, reference_embedding, domain: str) -> float | None:
    page_text = fetch_text(domain)
    if not page_text:
        return None

    page_embedding = model.encode(page_text)
    return round(util.cos_sim(reference_embedding, page_embedding).item(), 4)
