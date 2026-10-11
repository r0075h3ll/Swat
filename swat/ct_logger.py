"""Certificate Transparency verification via crt.name's apex index.

The pipeline verifies each LLM-generated candidate against CT logs. A candidate
with no certificate record is not a live counterfeit worth scoring. crt.name is
free, needs no token, and beats crt.sh, which we dropped for ~42% monthly
uptime.
"""

import requests

from .domain import registrable_domain

CRT_NAME_URL = "https://crt.name/v1/search"

session = requests.session()


def cert_exists(domain: str) -> bool:
    """True if crt.name's apex index has a certificate record for ``domain``'s
    registrable apex (or any of its subdomains). An empty result from a source
    that answered means "no certificate we can see"; any request or decode
    error is treated the same way, since we cannot confirm a cert exists."""
    try:
        with session.get(
            CRT_NAME_URL,
            params={"apex": registrable_domain(domain), "format": "json"},
            timeout=30,
        ) as response:
            response.raise_for_status()
            data = response.json()
    except (requests.exceptions.RequestException, ValueError):
        return False
    return isinstance(data, list) and bool(data)
