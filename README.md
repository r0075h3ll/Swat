# SWAT

Monitor CT logs for brand abuse using semantic search.

SWAT pulls newly issued certificates from crt.sh for a target domain, then
scores each result against the target using sentence embeddings and
Levenshtein distance to surface typosquats and look-alike domains.

## How it works

crt.sh indexes Certificate Transparency logs: every domain a CA has issued a
cert for, since ~2013. Its identity search is unreliable for a single
unbroken word (`examplebrand`) but consistent for a real word-boundary
split (`example brand`). Since a domain label has no spaces, SWAT takes the
registrable label of the target (`paypal` for `www.paypal.com`), then queries
crt.sh with it plus every single-split variant, capped at 16 queries and 5 in
flight at a time, and merges every response that succeeds. Then it ranks the
results:

- **Cosine similarity** (sentence embeddings) catches semantically close
  names that don't share characters.
- **Levenshtein distance**, normalised by the longer label length, catches
  character-level typosquats (`paypa1.com` for `paypal.com`) that embeddings
  tend to miss. The ratio is used so `-l` means the same thing on a 6-char
  brand as on a 20-char one; the raw distance is still reported.

Both signals score the registrable label (`paypal`) rather than the full
domain (`paypal.com`), because the TLD is a shared constant that pushes
cosine similarity up uniformly and eats into the distance budget without
telling us anything about the brand.

A candidate is flagged when it clears **both** thresholds. Cosine similarity
from a general-purpose sentence embedding on short, out-of-distribution
strings sits well above zero on unrelated pairs, so on its own it does
little filtering; requiring the edit signal too keeps precision up. Pass
`--any` to fall back to OR (flag on either threshold) if the noise is
acceptable — for example when scanning for very short typosquats where the
edit ratio is expected to be small enough that similarity is doing most of
the work.

Independently of the thresholds, a candidate whose hostname contains the
target label as a substring is flagged and marked `brand_in_hostname: true`.
`paypal-secure.evil.com` scores as `evil` against `paypal` under label-only
comparison and fails both thresholds, but the brand token is right there in
the SAN — brand-in-subdomain is one of the most common shapes in CT SAN
lists, and a CT-log monitor exists to catch it.

Optionally, pass a reference URL (`-r`) for the real brand's homepage. SWAT
fetches it once, then each flagged domain's own homepage, and scores content
similarity via the same model. Catches active phishing clones; an
unreachable domain just gets no `content_similarity` field. The comparison
covers the head of each page only, because the embedding model takes at most
256 word pieces, so a clone whose distinguishing text sits below the fold
will score low.

Flagged domains come out of Certificate Transparency logs, which anyone can
write to, so those fetches are treated as untrusted: a candidate that resolves
to a private, loopback, or link-local address is skipped rather than fetched
from your network, redirects are checked hop by hop, and the response body is
capped. Only `text/html` and `application/xhtml+xml` are decoded, and always as
UTF-8 unless the response names a charset.

## crt.sh limitations

Verified directly against live crt.sh, which has no documentation beyond its
own search form.

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
- **Unverifiable ceiling on huge successful queries.** SWAT reads the body of any
  response whose `Content-Length` it accepts, and caps the rest; whether crt.sh
  itself silently caps very large result sets isn't something we can confirm
  from outside their system.

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
| `-d` | Target domain (required). IDN targets are accepted and normalised to their A-label form. IP literals are rejected. | |
| `-s` | Minimum cosine similarity to flag a candidate | `0.5` |
| `-l`, `--max-edit-ratio` | Maximum normalised Levenshtein distance (`edit_distance / longer_label_length`, in `[0, 1]`) | `0.35` |
| `--any` | Flag candidates that clear either threshold (default requires both) | off |
| `-r` | Reference URL for content-similarity classification (optional) | none |
| `-o` | Output file, or `stdout` | `stdout` |

Output is JSON: `{"input_domain": ..., "results": [{"domain", "similarity",
"levenshtein_distance", "levenshtein_ratio", "brand_in_hostname", (optional)
"content_similarity"}]}`.

## Roadmap

- Option to choose between different embedding models
- Replace the `-r` content-similarity heuristic with a classifier actually
  trained on brand data

