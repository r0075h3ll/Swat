import argparse
import json
import logging
import sys
import traceback
from datetime import datetime

from . import classifier, generator, logger
from .ct_logger import cert_exists
from .domain import normalize_domain, registrable_domain, search_label
from .levenshtein import distance as levenshtein_distance

ERROR_LOG_PATH = "swat-error.log"

_LEVEL_COLORS = {
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Colors WARNING/ERROR/CRITICAL lines when stderr is a terminal. INFO stays
    plain (it's the routine progress line). Piped output (CI logs, `2> file`)
    gets no ANSI codes, since color in a log file is noise."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno)
        if color and sys.stderr.isatty():
            return f"{color}{message}{_RESET}"
        return message


# Handler setup lives in the entry point, not the package, so importing swat
# never reconfigures logging for whoever imported it.
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ColorFormatter())
    logger.addHandler(handler)


def _domain_arg(raw: str) -> str:
    try:
        return normalize_domain(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _ratio_arg(raw: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{raw!r} is not in the [0, 1] range")
    return value


def _positive_int_arg(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{raw!r} must be at least 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        help="target domain. Repeatable: pass -d once per brand alias or regional "
        "domain (e.g. a company that trades under two different names, or operates "
        "the same brand across several ccTLDs) to generate and score candidates "
        "against all of them in a single run.",
        dest="domains",
        action="append",
        required=True,
        type=_domain_arg,
    )
    parser.add_argument(
        "--count",
        help="number of candidate domains to generate per brand alias/label (default 20)",
        dest="count",
        type=_positive_int_arg,
        default=20,
    )
    parser.add_argument(
        "-l",
        "--max-edit-ratio",
        help="max normalised Levenshtein distance (edit_distance / longer_label_length, "
        "0-1) a candidate's label may sit from the brand label before it is dropped. "
        "This is a cheap pre-filter that discards candidates so far from the brand that "
        "Jev would not call them a lookalike, to avoid paying for the Jev call.",
        dest="max_edit_ratio",
        type=_ratio_arg,
        default=0.6,
    )
    parser.add_argument(
        "-r",
        "--reference-url",
        help="reference URL whose index-page content candidate pages are compared "
        "against. Defaults to the first target domain's homepage for each brand alias.",
        dest="reference_url",
        default=None,
    )
    parser.add_argument(
        "-o", help="write output to a file (default: stdout)", default="stdout", dest="output"
    )
    return parser


def is_live(domain: str) -> bool:
    """True if any independent signal shows ``domain`` is live: a CT certificate
    record, an HTTP response, or rendered index content via r.jina.ai. CT logs
    alone can miss domains whose records aren't in the queried source, so the
    network probes are additional gates."""
    if cert_exists(domain):
        return True
    if classifier.http_probe(domain):
        return True
    return classifier.jina_fetch(f"https://{domain}") is not None


def find_lookalikes(args: argparse.Namespace) -> dict:
    domains = list(dict.fromkeys(args.domains))
    target_registrables = {registrable_domain(d) for d in domains}

    targets_by_label: dict[str, list[str]] = {}
    for domain in domains:
        targets_by_label.setdefault(search_label(domain), []).append(domain)

    results: dict[str, dict] = {}
    manual_review: list[dict] = []
    generated_candidates: list[dict] = []
    decisions: list[dict] = []
    failed_labels: list[str] = []

    for label, label_domains in targets_by_label.items():
        reference_url = args.reference_url or f"https://{label_domains[0]}"
        reference_text = classifier.fetch_text(reference_url)
        if reference_text is None:
            logger.warning(f"could not fetch reference content from '{reference_url}'; content gate skipped")

        try:
            candidates = generator.generate_candidates(
                label, count=args.count, business_context=reference_text or ""
            )
        except RuntimeError as e:
            failed_labels.append(label)
            logger.warning(f"could not generate candidates for '{label}': {e}")
            continue

        logger.info(f"'{label}': {len(candidates)} candidate(s) generated")

        for candidate in candidates:
            generated_candidates.append({"domain": candidate, "matched_label": label})

        if not candidates:
            continue

        for candidate in candidates:
            if registrable_domain(candidate) in target_registrables:
                continue  # first-party

            candidate_label = search_label(candidate)
            edit_distance = levenshtein_distance(label, candidate_label)
            edit_ratio = edit_distance / max(len(label), len(candidate_label), 1)

            if edit_ratio > args.max_edit_ratio:
                continue

            if not is_live(candidate):
                continue  # no CT record, no HTTP response, no index content: not a live counterfeit

            domain_gate = classifier.domain_impersonation(label, candidate_label)
            if not domain_gate:
                decisions.append(
                    {"domain": candidate, "matched_label": label, "domain_gate": "no", "content_gate": None}
                )
                continue  # Jev: not a lookalike/typosquat

            candidate_text = classifier.fetch_text(candidate)
            if candidate_text is None:
                manual_review.append(
                    {"domain": candidate, "matched_label": label, "reason": "content unreachable"}
                )
                decisions.append(
                    {"domain": candidate, "matched_label": label, "domain_gate": "yes", "content_gate": None}
                )
                continue

            if reference_text is None:
                manual_review.append(
                    {"domain": candidate, "matched_label": label, "reason": "reference content unavailable"}
                )
                decisions.append(
                    {"domain": candidate, "matched_label": label, "domain_gate": "yes", "content_gate": None}
                )
                continue

            content_gate = classifier.content_impersonation(reference_text, candidate_text)
            content_verdict = "yes" if content_gate else "no"
            decisions.append(
                {
                    "domain": candidate,
                    "matched_label": label,
                    "domain_gate": "yes",
                    "content_gate": content_verdict,
                }
            )
            if not content_gate:
                continue  # Jev: content is not impersonating

            result = {
                "domain": candidate,
                "matched_label": label,
                "levenshtein_distance": edit_distance,
                "levenshtein_ratio": round(edit_ratio, 4),
            }
            existing = results.get(candidate)
            if existing is None or edit_ratio < existing["levenshtein_ratio"]:
                results[candidate] = result

    if failed_labels and len(failed_labels) == len(targets_by_label):
        raise RuntimeError(
            f"candidate generation failed for every target label ({', '.join(failed_labels)}); "
            "is OPENROUTER_API_KEY set?"
        )

    if failed_labels:
        logger.warning(
            f"brand alias(es) {failed_labels} could not be checked at all this run "
            "(candidate generation failed for them); results below only cover "
            f"{sorted(set(targets_by_label) - set(failed_labels))}."
        )

    ordered = sorted(results.values(), key=lambda r: r["levenshtein_ratio"])
    return {
        "input_domains": domains,
        "generated_candidates": generated_candidates,
        "decisions": decisions,
        "results": ordered,
        "manual_review": manual_review,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sim_details = find_lookalikes(args)
    json_output = json.dumps(sim_details, indent=4)

    if args.output == "stdout":
        print(json_output)
    else:
        with open(args.output, "w") as file:
            file.write(json_output)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        logger.error(str(e))
        raise SystemExit(1) from e
    except Exception as e:
        with open(ERROR_LOG_PATH, "a") as log_file:
            log_file.write(f"{datetime.now().isoformat()} {e}\n{traceback.format_exc()}\n")
        logger.error(f"{e} (full traceback written to {ERROR_LOG_PATH})")
        raise SystemExit(1) from e
