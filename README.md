# SWAT

Monitor CT logs for brand abuse using semantic search.

SWAT pulls newly issued certificates from crt.sh for a target domain, then
scores each result against the target using sentence embeddings and
Levenshtein distance to surface typosquats and look-alike domains.

## How it works

crt.sh indexes Certificate Transparency logs: every domain a CA has issued a
cert for, since ~2013. Its identity search is unreliable for a single
unbroken word (`examplebrand`) but consistent for a real word-boundary
split (`example brand`). Since a domain label has no spaces, SWAT queries
the bare label plus every single-split variant of it, up to 5 at a time, and
merges every response that succeeds. Then it ranks the results:

- **Cosine similarity** (sentence embeddings) catches semantically close
  names that don't share characters.
- **Levenshtein distance** catches character-level typosquats
  (`paypa1.com` for `paypal.com`) that embeddings tend to miss.

A candidate is flagged if it clears either threshold.

Optionally, pass a reference URL (`-r`) for the real brand's homepage. SWAT
fetches it once, then each flagged domain's own homepage, and scores content
similarity via the same model. Catches active phishing clones; an
unreachable domain just gets no `content_similarity` field.

## crt.sh limitations

Verified directly against live crt.sh (it has no documentation beyond its
own search form); full detail and evidence in `DECISIONS.md`.

- **Flaky.** The identical request can return real data, `404`, `502`, or an
  empty `200` seconds apart. SWAT retries with backoff and warns when every
  query variant comes back empty.
- **Rate-limited under concurrency.** Firing many requests at once triggers
  `429`s, server-side, not client-specific. SWAT caps concurrency
  (`FANOUT_CONCURRENCY`) and gives `429` its own backoff.
- **`common_name`/`name_value` can be explicit `null`**, not just absent.
  `.get(key, "")` doesn't guard against that; SWAT does.
- **No date filtering, no pagination.** A query returns everything since CT
  logging began, in one response, or fails outright, no date parameter and
  no page/offset/cursor exists.
- **Unverifiable ceiling on huge successful queries.** SWAT reads the full
  response body; whether crt.sh itself silently caps very large result sets
  isn't something we can confirm from outside their system.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```
git clone https://github.com/r0075h3ll/SWAT.git
cd SWAT
uv sync
```

`uv sync` creates a `.venv` and installs all dependencies, including
`sentence-transformers`, which downloads its embedding model on first run.

## Usage

```
uv run python3 -m swat -d example.com -o output.json
```

| Flag | Description | Default |
|------|-------------|---------|
| `-d` | Target domain (required) | |
| `-s` | Minimum cosine similarity to flag a candidate | `0.5` |
| `-l` | Maximum Levenshtein distance to flag a candidate | `3` |
| `-r` | Reference URL for content-similarity classification (optional) | none |
| `-o` | Output file, or `stdout` | `stdout` |

Sample output: [examples/output.json](examples/output.json)

## Development

```
uv run ruff check .
uv run ruff format .
uv run pytest
```

Run all three before committing. CI runs the same on every push and pull
request to `main`.

## Roadmap

- Option to choose between different embedding models
- Replace the `-r` content-similarity heuristic with a classifier actually
  trained on brand data (the original intent; see `DECISIONS.md`)

## License

[MIT](LICENSE)
