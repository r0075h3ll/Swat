# SWAT

Find domains that impersonate a brand.

SWAT asks an LLM for look-alike and typosquat domains, checks which ones are
live, then has [Jev](https://typesafe.ai/) (TypeSafe SDK) decide with a yes/no
whether each domain impersonates the brand and whether its homepage clones the
real site.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```
git clone https://github.com/r0075h3ll/SWAT.git
cd SWAT
uv sync

export OPENROUTER_API_KEY=sk-or-v1-...
export JEV_API_KEY=...
uv run python3 -m swat -d example.com -o output.json
```

## Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | required |
| `JEV_API_KEY` | Jev API key | required |
| `SWAT_MODEL` | OpenRouter model slug | `openai/gpt-4o-mini` |

Any model OpenRouter serves works. The prompt is provider-agnostic and a cheap
chat model is enough.

## Usage

```
uv run python3 -m swat -d <domain> [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-d` | Target domain. Required. Repeat for brand aliases. | |
| `--count` | Candidates to generate per label | `20` |
| `-l`, `--max-edit-ratio` | Max normalised Levenshtein distance (0-1) before a candidate is dropped | `0.6` |
| `-r`, `--reference-url` | Reference URL for content comparison | first target domain |
| `-o` | Output file | stdout |

To cover several brand aliases or regional domains, pass `-d` once per domain:

```
uv run python3 -m swat -d acmecorp.com -d acmebank.co.id -o output.json
```

Domains that reduce to the same label are generated once.

## How it works

```
generate -> pre-filter -> liveness check -> Jev judgment
```

1. **Generate.** An LLM, reached through OpenRouter, proposes candidates for the
   brand's registrable label (`paypal` from `www.paypal.com`).

2. **Pre-filter.** Candidates whose label is too far from the brand label
   (normalised Levenshtein distance) are dropped before any network or Jev call.

3. **Check liveness.** A candidate is live if any one of these succeeds:

   | Check | Passes when |
   |-------|-------------|
   | `cert_exists(domain)` | crt.name's apex index has a CT record |
   | `classifier.http_probe(domain)` | a direct HTTP(S) request gets any response (browser User-Agent) |
   | `classifier.jina_fetch(url)` | r.jina.ai returns rendered index-page content |

   CT logs can miss domains absent from the queried source, so the two network
   probes act as extra gates.

4. **Judge.** Jev answers two yes/no questions through `Choice`:

   - Is the domain a look-alike or typosquat of the brand? (`domain_gate`)
   - Does the candidate's index page impersonate the brand's? (`content_gate`)

   Two "yes" answers mark the domain as counterfeit. If the page can't be
   fetched, the domain goes to manual review instead of being dropped.

## Output

```json
{
  "input_domains": ["paypal.com"],
  "generated_candidates": [
    {"domain": "paypa1.com", "matched_label": "paypal"},
    {"domain": "paypal-secure.com", "matched_label": "paypal"}
  ],
  "decisions": [
    {"domain": "paypa1.com", "matched_label": "paypal", "domain_gate": "yes", "content_gate": "yes"},
    {"domain": "paypal-secure.com", "matched_label": "paypal", "domain_gate": "yes", "content_gate": null}
  ],
  "results": [
    {
      "domain": "paypa1.com",
      "matched_label": "paypal",
      "levenshtein_distance": 1,
      "levenshtein_ratio": 0.1667
    }
  ],
  "manual_review": [
    {"domain": "paypal-secure.com", "matched_label": "paypal", "reason": "content unreachable"}
  ]
}
```

| Key | Contents |
|-----|----------|
| `generated_candidates` | Every domain the model proposed |
| `decisions` | Live candidates with Jev's verdict per gate. A gate is `"yes"`, `"no"`, or `null` if it never ran. |
| `results` | Domains Jev flagged as impersonating (both gates `"yes"`) |
| `manual_review` | Domains that passed the name and liveness checks but whose page couldn't be fetched |

## Notes

- Page content is fetched through r.jina.ai first, then falls back to a direct
  HTTP(S) request with a browser User-Agent. r.jina.ai is free and needs no
  token.
- crt.name's free tier caps at 100 requests/day, so large candidate sets may hit
  the limit. We dropped crt.sh because its monthly uptime was about 42%.
