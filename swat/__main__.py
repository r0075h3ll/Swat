import argparse
import json
import logging
import sys

from sentence_transformers import SentenceTransformer, util

from . import classifier, logger
from .ct_logger import get_ct_logs_for_label, log_domains
from .domain import normalize_domain, registrable_domain, search_label
from .levenshtein import distance as levenshtein_distance

# Handler setup lives in the entry point, not the package, so importing swat
# never reconfigures logging for whoever imported it.
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stderr))


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="domain", dest="domain", required=True, type=_domain_arg)
    parser.add_argument("-s", help="similarity threshold (0-1)", dest="sim_thres", type=float, default=0.5)
    parser.add_argument(
        "-l",
        "--max-edit-ratio",
        help="max normalised Levenshtein distance, expressed as edit_distance / "
        "longer_label_length (0-1). Default catches a distance of 2 on a 6-character "
        "label like 'paypal' but not on a 20-character one.",
        dest="max_edit_ratio",
        type=_ratio_arg,
        default=0.35,
    )
    parser.add_argument(
        "--any",
        help="flag candidates that clear either threshold. Default is to require both, "
        "since cosine similarity from a general-purpose sentence embedding on short "
        "out-of-distribution strings sits well above zero and does little filtering "
        "on its own.",
        dest="match_any",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        help="reference URL for content-similarity classification (fetches this and each "
        "flagged domain's homepage, compares via embeddings)",
        dest="reference_url",
        default=None,
    )
    parser.add_argument(
        "-o", help="write output to a file (default: stdout)", default="stdout", dest="output"
    )
    return parser


def find_lookalikes(args: argparse.Namespace) -> dict:
    domain = args.domain
    target_label = search_label(domain)
    target_registrable = registrable_domain(domain)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info(f"Querying crt.sh for certificates matching '{target_label}' and its split variants")
    raw_logs = get_ct_logs_for_label(target_label)

    if not raw_logs:
        logger.warning(
            "crt.sh returned zero certificates across the bare query and every "
            "split variant of it. crt.sh's public instance is known to be "
            "unreliable, so this could still be a silent failure rather than a "
            "real empty result, but it's a much stronger signal after this many "
            "attempts. Consider re-running before concluding there are no "
            "lookalikes."
        )

    # A crt.sh query for a brand returns mostly that brand's own certificates.
    # Comparing registrable domains drops every first party name, not just the
    # exact target: www.example.com, api.example.com and checkout.example.com
    # all score high against the target precisely because they contain it.
    candidates = [c for c in log_domains(raw_logs) if registrable_domain(c) != target_registrable]

    if not candidates:
        return {"input_domain": domain, "results": []}

    # Score labels, not full domains. The TLD is a shared constant that dilutes
    # both signals: every .com candidate scores non-trivially against every .com
    # target on cosine similarity, and every one shares four characters with the
    # target on Levenshtein. And a distance-1 typosquat on "paypal" is a very
    # different signal from a distance-1 near-miss on a 20-char label, so the
    # distance is normalised by the longer of the two labels.
    candidate_labels = [search_label(c) for c in candidates]

    target_emb = model.encode(target_label)
    candidate_embs = model.encode(candidate_labels)
    similarities = util.cos_sim(target_emb, candidate_embs)[0]

    results = []
    for candidate, candidate_label, score in zip(candidates, candidate_labels, similarities, strict=True):
        similarity = score.item()
        edit_distance = levenshtein_distance(target_label, candidate_label)
        edit_ratio = edit_distance / max(len(target_label), len(candidate_label), 1)

        # Cosine similarity from a general-purpose sentence embedding on short,
        # out-of-distribution strings sits well above zero on unrelated pairs,
        # so on its own it does little filtering. The default requires both
        # signals; --any restores the old OR behaviour for callers who want it.
        sim_hit = similarity >= args.sim_thres
        edit_hit = edit_ratio <= args.max_edit_ratio
        threshold_hit = (sim_hit or edit_hit) if args.match_any else (sim_hit and edit_hit)

        # Brand-in-hostname is the shape a CT monitor exists to catch, and
        # scoring labels alone throws away the subdomain context that used to
        # surface it: `paypal-secure.evil.com` scores as 'evil' against
        # 'paypal' and fails both thresholds. A substring hit is enough to
        # flag on its own, and reported alongside the numeric signals so a
        # reviewer can see what triggered the row.
        brand_in_hostname = target_label in candidate
        flagged = threshold_hit or brand_in_hostname

        if flagged:
            results.append(
                {
                    "domain": candidate,
                    "similarity": round(similarity, 4),
                    "levenshtein_distance": edit_distance,
                    "levenshtein_ratio": round(edit_ratio, 4),
                    "brand_in_hostname": brand_in_hostname,
                }
            )

    if args.reference_url:
        reference_text = classifier.fetch_text(args.reference_url)
        if reference_text is None:
            logger.warning(
                f"Could not fetch reference content from '{args.reference_url}'; "
                "skipping content-similarity classification"
            )
        else:
            reference_embedding = model.encode(reference_text)
            for result in results:
                result["content_similarity"] = classifier.content_similarity(
                    model, reference_embedding, result["domain"]
                )

    # Ranked after the enrichment above, not before it: content similarity is
    # the strongest signal available that a domain is actively impersonating the
    # brand, so a live clone should outrank a name that merely looks similar.
    # Candidates with no content score sort last, then the name-based signals
    # break ties.
    results.sort(
        key=lambda r: (
            r.get("content_similarity") is None,
            -(r.get("content_similarity") or 0.0),
            -r["similarity"],
            r["levenshtein_distance"],
        )
    )

    return {"input_domain": domain, "results": results}


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
    except Exception as e:
        logger.exception(e)
        raise SystemExit(1) from e
