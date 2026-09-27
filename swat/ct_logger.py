import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC
from email.utils import parsedate_to_datetime

import requests
from requests.adapters import HTTPAdapter

CRT_SH_URL = "https://crt.sh/"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
RATE_LIMIT_BACKOFF_SECONDS = 5

# crt.sh can answer a 429 with Retry-After: 3600. Honouring that verbatim parks
# a pool worker for an hour, in a tool whose request timeout is 30 seconds.
MAX_RETRY_AFTER_SECONDS = 60

# split_variants yields one query per character, so a long brand fans out
# without limit. The cap bounds the worst-case wait; labels at or under
# MAX_QUERY_VARIANTS characters are unaffected.
MAX_QUERY_VARIANTS = 16

# crt.sh has no pagination, so "%shortlabel%" can match a large share of all
# CT. Trust Content-Length to keep that from landing in memory whole.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# crt.sh is unreliable for a single unbroken token (e.g. "examplebrand") but
# far more reliable for space-separated multi-word queries (e.g. "example
# brand"). There's no space in a domain label, so we brute-force every
# single-split position instead of doing real word segmentation. Firing every
# variant at once (no cap) got some of them rate-limited (429) by crt.sh, so
# the fan-out below caps how many are in flight at a time.
VARIANT_MAX_RETRIES = 1
VARIANT_RETRY_BACKOFF_SECONDS = 1
FANOUT_CONCURRENCY = 5

session = requests.session()
session.mount("https://", HTTPAdapter(pool_maxsize=FANOUT_CONCURRENCY))


def _retry_after_seconds(header: str) -> float:
    """Retry-After is either delta-seconds or an HTTP-date (RFC 7231). Clamp
    whatever comes back so neither form can park a worker indefinitely."""
    try:
        seconds = float(header)
    except ValueError:
        try:
            when = parsedate_to_datetime(header)
        except (TypeError, ValueError):
            return RATE_LIMIT_BACKOFF_SECONDS
        if when is None:
            return RATE_LIMIT_BACKOFF_SECONDS
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = when.timestamp() - time.time()

    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


def _reject_oversized(response: requests.Response) -> None:
    """crt.sh serves every match for a query in one body, so a short label can
    come back enormous. A missing or bogus Content-Length is left alone rather
    than guessed at, since crt.sh sets it accurately and a lie is not the
    failure mode worth engineering for here."""
    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_RESPONSE_BYTES:
        raise requests.exceptions.RequestException(
            f"crt.sh response is {content_length} bytes, over the {MAX_RESPONSE_BYTES} byte cap"
        )


def get_ct_logs(
    query: str,
    max_retries: int = MAX_RETRIES,
    backoff_seconds: float = RETRY_BACKOFF_SECONDS,
) -> list[dict]:
    last_error = None
    sleep_seconds = 0.0

    for attempt in range(max_retries + 1):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        sleep_seconds = 0.0

        try:
            response = session.get(CRT_SH_URL, params={"q": query, "output": "json"}, timeout=30)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = (
                    _retry_after_seconds(retry_after) if retry_after else RATE_LIMIT_BACKOFF_SECONDS
                )
                last_error = requests.exceptions.HTTPError(
                    f"429 Too Many Requests (retry-after={retry_after or 'none'})"
                )
                continue

            response.raise_for_status()
            _reject_oversized(response)
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            last_error = e
            sleep_seconds = backoff_seconds * 2**attempt

    attempts = max_retries + 1
    raise RuntimeError(f"crt.sh request failed after {attempts} attempts: {last_error}") from last_error


def split_variants(word: str) -> list[str]:
    """Every single-space-insertion variant of word, e.g. 'abcd' -> ['a bcd', 'ab cd', 'abc d']."""
    return [f"{word[:i]} {word[i:]}" for i in range(1, len(word))]


def get_ct_logs_for_label(root_label: str) -> list[dict]:
    """Query crt.sh with the bare label and every single-split variant of it,
    up to FANOUT_CONCURRENCY at a time, merging all non-empty responses. Each
    variant gets a light retry budget since redundancy comes from trying many
    query shapes, not from retrying any single one.

    Raises RuntimeError when every query fails. Returning an empty list there
    would be indistinguishable from a target that genuinely has no lookalikes,
    which is the one thing this tool must not get wrong.
    """
    queries = [f"%{root_label}%", *split_variants(root_label)[: MAX_QUERY_VARIANTS - 1]]
    merged: dict[object, dict] = {}
    succeeded = 0

    with ThreadPoolExecutor(max_workers=min(FANOUT_CONCURRENCY, len(queries))) as executor:
        futures = {
            executor.submit(
                get_ct_logs,
                query,
                max_retries=VARIANT_MAX_RETRIES,
                backoff_seconds=VARIANT_RETRY_BACKOFF_SECONDS,
            ): query
            for query in queries
        }

        for future in as_completed(futures):
            try:
                entries = future.result()
            except RuntimeError:
                continue

            succeeded += 1
            for entry in entries:
                # A present-but-null id is not a usable key; .get's default only
                # fires on an absent one, which would collapse every such entry
                # onto a single None slot.
                key = entry.get("id") or json.dumps(entry, sort_keys=True)
                merged[key] = entry

    if not succeeded:
        raise RuntimeError(
            f"all {len(queries)} crt.sh queries for '{root_label}' failed, so no result "
            "can be reported either way"
        )

    return list(merged.values())


def log_domains(crt_log_data: list[dict]) -> list[str]:
    domains = set()

    for entry in crt_log_data:
        common_name = entry.get("common_name") or ""
        name_value = entry.get("name_value") or ""
        for name in common_name.split("\n") + name_value.split("\n"):
            name = name.strip().lower().lstrip("*.")
            if name:
                domains.add(name)

    return sorted(domains)
