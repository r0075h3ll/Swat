import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter

CRT_SH_URL = "https://crt.sh/"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
RATE_LIMIT_BACKOFF_SECONDS = 5

# crt.sh is unreliable for a single unbroken token (e.g. "examplebrand") but
# far more reliable for space-separated multi-word queries (e.g. "example
# brand"). There's no space in a domain label, so we brute-force every
# single-split position instead of doing real word segmentation. Firing every
# variant at once (no cap) got some of them rate-limited (429) by crt.sh, so
# the fan-out below caps how many are in flight at a time.
VARIANT_MAX_RETRIES = 1
VARIANT_RETRY_BACKOFF_SECONDS = 1
FANOUT_CONCURRENCY = 5
MAX_CONCURRENT_REQUESTS = 50

session = requests.session()
session.mount("https://", HTTPAdapter(pool_maxsize=MAX_CONCURRENT_REQUESTS))


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
                sleep_seconds = float(retry_after) if retry_after else RATE_LIMIT_BACKOFF_SECONDS
                last_error = requests.exceptions.HTTPError(
                    f"429 Too Many Requests (retry-after={retry_after or 'none'})"
                )
                continue

            response.raise_for_status()
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
    query shapes, not from retrying any single one."""
    queries = [f"%{root_label}%", *split_variants(root_label)]
    merged: dict[object, dict] = {}

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

            for entry in entries:
                key = entry.get("id", json.dumps(entry, sort_keys=True))
                merged[key] = entry

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
