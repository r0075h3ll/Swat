"""Generate look-alike/typosquat candidate domains with an LLM via OpenRouter.

Discovery once brute-forced crt.sh with split-variant wildcard queries. That
approach produced false positives and structural discovery gaps, so we replaced
it with LLM generation: the model proposes candidate domains for a brand label,
and the pipeline verifies each candidate against CT logs downstream.

The model is any chat/completion model reachable through OpenRouter. The prompt
is provider-agnostic, so the same call works for OpenAI, Anthropic, and the
other models OpenRouter fronts. Tests mock ``_complete``.
"""

import os
import re

import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("SWAT_MODEL", "openai/gpt-4o-mini")
COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"

# Homepage content can be far longer than the prompt needs; the business's
# nature is usually clear from the first stretch of text.
_CONTEXT_LIMIT = 2000

# A candidate must reduce to a plausible bare domain (label + TLD). The
# generator is an LLM: it can emit noise, markdown, or prose. This is the
# boundary that keeps garbage out of the downstream CT/logic steps.
_DOMAIN_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")

# Strip common list formatting the model may emit (numbered items, bullets).
_LEADING_LIST_RE = re.compile(r"^(\d+[.)]\s+|[-*]\s+)")


def build_generation_prompt(brand_label: str, count: int = 20, business_context: str = "") -> str:
    """Return a provider-agnostic prompt that asks any LLM for look-alike and
    typosquat candidate domains for ``brand_label``. ``business_context`` is
    optional text describing what the brand does (typically its homepage), so
    the model can craft domains that read like the real business. The prompt is
    plain text so it can be sent as a single user message to any endpoint."""
    context_block = ""
    if business_context:
        context_block = (
            "\n"
            "Here is what the brand does, taken from its own website, so you can "
            "make the candidates read like the real business:\n"
            f"'''\n{business_context[:_CONTEXT_LIMIT]}\n'''\n"
        )
    return (
        f"Generate {count} distinct domain names that a typosquatter or phishing "
        f"operator might register to impersonate the brand '{brand_label}'.\n"
        f"{context_block}"
        "\n"
        "Vary the technique so the list covers many shapes:\n"
        "- character substitution (paypal -> paypa1, paypol)\n"
        "- character insertion, omission, or transposition (paypal -> paypall, paypl, paypla)\n"
        "- homoglyphs and confusables (paypal -> paypaI)\n"
        "- TLD swaps and added words (paypal.com -> paypal.co, paypal-login.com)\n"
        "- keeping only part of the brand or fusing it with other words (paypal -> pay, pal, mypaypal)\n"
        "- brand label embedded in a subdomain or longer hostname\n"
        "\n"
        "Every candidate must plausibly read as the brand; do not invent unrelated names.\n"
        "\n"
        "Output only the domain names, one per line, lowercase, no numbering, no "
        "bullets, no markdown, no explanation."
    )


def generate_candidates(brand_label: str, count: int = 20, business_context: str = "") -> list[str]:
    """Ask the model for up to ``count`` candidate domains that look like
    ``brand_label``, returning a clean, deduplicated, order-preserving list."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; provide an OpenRouter API key to use LLM generation"
        )
    raw = _complete(build_generation_prompt(brand_label, count, business_context))
    return _parse_candidates(raw)


def _complete(prompt: str) -> str:
    """Send ``prompt`` to the configured OpenRouter model and return the reply
    text. Raises RuntimeError on transport or API errors."""
    try:
        response = requests.post(
            COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"OpenRouter request failed: {e}") from e

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter returned {response.status_code}: {response.text[-400:]}")

    try:
        payload = response.json()
        return payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as e:
        raise RuntimeError(f"OpenRouter returned an unexpected body: {e}") from e


def _parse_candidates(raw: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        token = line.strip().lower()
        if not token:
            continue
        token = _LEADING_LIST_RE.sub("", token).lstrip("*.").rstrip(".")
        if not _DOMAIN_RE.match(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        candidates.append(token)
    return candidates
