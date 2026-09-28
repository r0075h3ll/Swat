import argparse
import json
import logging
import sys

from sentence_transformers import SentenceTransformer, util

from . import classifier, logger
from .ct_logger import get_ct_logs_for_label, log_domains
from .domain import normalize_domain
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="domain", dest="domain", required=True, type=_domain_arg)
    parser.add_argument("-s", help="similarity threshold (0-1)", dest="sim_thres", type=float, default=0.5)
    parser.add_argument("-l", help="max levenshtein distance", dest="max_distance", type=int, default=3)
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
    root_label = domain.split(".")[0]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info(f"Querying crt.sh for certificates matching '{root_label}' and its split variants")
    raw_logs = get_ct_logs_for_label(root_label)

    if not raw_logs:
        logger.warning(
            "crt.sh returned zero certificates across the bare query and every "
            "split variant of it. crt.sh's public instance is known to be "
            "unreliable, so this could still be a silent failure rather than a "
            "real empty result, but it's a much stronger signal after this many "
            "attempts. Consider re-running before concluding there are no "
            "lookalikes."
        )

    candidates = log_domains(raw_logs)
    candidates = [c for c in candidates if c != domain]

    if not candidates:
        return {"input_domain": domain, "results": []}

    target_emb = model.encode(domain)
    candidate_embs = model.encode(candidates)
    similarities = util.cos_sim(target_emb, candidate_embs)[0]

    results = []
    for candidate, score in zip(candidates, similarities, strict=True):
        similarity = score.item()
        edit_distance = levenshtein_distance(domain, candidate)

        if similarity >= args.sim_thres or edit_distance <= args.max_distance:
            results.append(
                {
                    "domain": candidate,
                    "similarity": round(similarity, 4),
                    "levenshtein_distance": edit_distance,
                }
            )

    results.sort(key=lambda r: (-r["similarity"], r["levenshtein_distance"]))

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
