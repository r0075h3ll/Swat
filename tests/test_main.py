import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from swat import __main__ as swat_main

REPO_ROOT = Path(__file__).parent.parent


class _FakeUtil:
    @staticmethod
    def cos_sim(a, b):
        import torch

        a = torch.tensor(a)
        b = torch.tensor(b)
        a = a.unsqueeze(0) if a.ndim == 1 else a
        b = b.unsqueeze(0) if b.ndim == 1 else b
        a_n = a / a.norm(dim=-1, keepdim=True)
        b_n = b / b.norm(dim=-1, keepdim=True)
        return a_n @ b_n.mT


def _fake_model(sim_by_domain):
    class FakeModel:
        def encode(self, x):
            if isinstance(x, str):
                return np.array([1.0, 0.0])
            return np.array([[sim_by_domain[d], (1 - sim_by_domain[d] ** 2) ** 0.5] for d in x])

    return FakeModel()


def test_find_lookalikes_filters_sorts_and_normalizes_input(tmp_path):
    """paypa1.com is a distance-1 typo on the 'paypal' label (ratio 1/6 ≈ 0.17)
    and unrelated.com is neither similar nor close. --any preserves the original
    OR-mode semantics this test was written against."""
    fake_ct_response = [
        {"common_name": "paypal.com", "name_value": "paypal.com"},
        {"common_name": "paypa1.com", "name_value": "paypa1.com"},
        {"common_name": "unrelated.com", "name_value": "unrelated.com"},
    ]
    sim_by_domain = {"paypa1": 0.2, "unrelated": 0.1}

    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "https://PayPal.com/login", "-s", "0.3", "-l", "0.2", "--any"])

    with (
        mock.patch("swat.__main__.get_ct_logs_for_label", return_value=fake_ct_response),
        mock.patch("swat.__main__.SentenceTransformer", return_value=_fake_model(sim_by_domain)),
        mock.patch("swat.__main__.util", _FakeUtil()),
    ):
        result = swat_main.find_lookalikes(args)

    assert result["input_domain"] == "paypal.com"
    assert result["results"] == [
        {
            "domain": "paypa1.com",
            "similarity": 0.2,
            "levenshtein_distance": 1,
            "levenshtein_ratio": 0.1667,
            "brand_in_hostname": False,
        },
    ]


def test_find_lookalikes_scores_against_the_registrable_label():
    """A subdomain target is queried by its label, so it has to be scored by its
    label too. Scoring the full "www.paypal.com" against "paypa1.com" gives a
    Levenshtein distance of 5; the label pair "paypal" vs "paypa1" gives 1."""
    fake_ct_response = [{"common_name": "paypa1.com", "name_value": "paypa1.com"}]
    sim_by_domain = {"paypa1": 0.6}

    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "www.paypal.com", "-s", "0.5"])

    with (
        mock.patch("swat.__main__.get_ct_logs_for_label", return_value=fake_ct_response),
        mock.patch("swat.__main__.SentenceTransformer", return_value=_fake_model(sim_by_domain)),
        mock.patch("swat.__main__.util", _FakeUtil()),
    ):
        result = swat_main.find_lookalikes(args)

    assert result["input_domain"] == "www.paypal.com"
    assert result["results"] == [
        {
            "domain": "paypa1.com",
            "similarity": 0.6,
            "levenshtein_distance": 1,
            "levenshtein_ratio": 0.1667,
            "brand_in_hostname": False,
        },
    ]


def test_find_lookalikes_warns_but_does_not_crash_on_empty_crt_sh_response():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "example.com"])

    with (
        mock.patch("swat.__main__.get_ct_logs_for_label", return_value=[]),
        mock.patch("swat.__main__.SentenceTransformer", return_value=_fake_model({})),
        mock.patch("swat.__main__.util", _FakeUtil()),
    ):
        result = swat_main.find_lookalikes(args)

    assert result == {"input_domain": "example.com", "results": []}


def test_find_lookalikes_content_similarity_encodes_reference_once():
    fake_ct_response = [{"common_name": "evil.com", "name_value": "evil.com"}]
    sim_by_domain = {"evil": 0.9}
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "example.com", "-s", "0.5", "--any", "-r", "https://example.com"])

    encode_calls = []
    model = _fake_model(sim_by_domain)
    original_encode = model.encode

    def tracking_encode(x):
        encode_calls.append(x)
        return original_encode(x)

    model.encode = tracking_encode

    def fake_fetch_text(url_or_domain):
        return "reference page text" if url_or_domain == "https://example.com" else "candidate page text"

    with (
        mock.patch("swat.__main__.get_ct_logs_for_label", return_value=fake_ct_response),
        mock.patch("swat.__main__.SentenceTransformer", return_value=model),
        mock.patch("swat.__main__.util", _FakeUtil()),
        mock.patch("swat.classifier.fetch_text", side_effect=fake_fetch_text),
        mock.patch("swat.classifier.util", _FakeUtil()),
    ):
        result = swat_main.find_lookalikes(args)

    assert "content_similarity" in result["results"][0]
    reference_encodes = [c for c in encode_calls if c == "reference page text"]
    candidate_encodes = [c for c in encode_calls if c == "candidate page text"]
    assert len(reference_encodes) == 1, "reference text should be encoded exactly once, not per candidate"
    assert len(candidate_encodes) == 1


def test_invalid_domain_exits_with_usage_error():
    result = subprocess.run(
        [sys.executable, "-m", "swat", "-d", "not a domain"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    assert "does not look like a fully qualified domain" in result.stderr


def test_missing_required_domain_flag_exits_with_usage_error():
    result = subprocess.run(
        [sys.executable, "-m", "swat"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    assert "-d" in result.stderr


def test_main_writes_json_to_output_file(tmp_path):
    output_file = tmp_path / "out.json"
    parser_args = ["-d", "example.com", "-o", str(output_file)]

    with mock.patch(
        "swat.__main__.find_lookalikes",
        return_value={"input_domain": "example.com", "results": []},
    ):
        swat_main.main(parser_args)

    assert json.loads(output_file.read_text()) == {"input_domain": "example.com", "results": []}


def test_main_prints_json_to_stdout_by_default(capsys):
    with mock.patch(
        "swat.__main__.find_lookalikes",
        return_value={"input_domain": "example.com", "results": []},
    ):
        swat_main.main(["-d", "example.com"])

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"input_domain": "example.com", "results": []}


@pytest.mark.slow
def test_stdout_output_contains_no_log_noise():
    """Regression test for the stdout/stderr logging-contamination bug: log lines must
    never land on stdout, since that's the one thing -o stdout is supposed to guarantee."""
    script = (
        "import sys; sys.path.insert(0, '.')\n"
        "from unittest import mock\n"
        "import swat.__main__ as m\n"
        "with mock.patch('swat.__main__.get_ct_logs_for_label', return_value=[]):\n"
        "    m.main(['-d', 'example.com'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert json.loads(result.stdout) == {"input_domain": "example.com", "results": []}


@pytest.mark.slow
def test_importing_swat_leaves_the_root_logger_alone():
    """Regression test: importing swat must not reconfigure logging for the whole process."""
    script = (
        "import logging\n"
        "handlers, level = list(logging.getLogger().handlers), logging.getLogger().level\n"
        "import swat\n"
        "assert list(logging.getLogger().handlers) == handlers, 'root handlers changed'\n"
        "assert logging.getLogger().level == level, 'root level changed'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.slow
def test_total_crtsh_failure_exits_non_zero_without_printing_results():
    """A lookup that never reached crt.sh must not be reported as a clean empty
    result: exit non zero, and print nothing that a caller could read as 'no lookalikes'."""
    script = (
        "import sys; sys.path.insert(0, '.')\n"
        "from unittest import mock\n"
        "import runpy\n"
        "from swat import ct_logger\n"
        "with mock.patch.object(ct_logger, 'get_ct_logs', side_effect=RuntimeError('down')):\n"
        "    sys.argv = ['swat', '-d', 'example.com']\n"
        "    runpy.run_module('swat', run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 1, result.stderr
    assert result.stdout.strip() == "", f"stdout carried a result: {result.stdout!r}"


def test_find_lookalikes_drops_first_party_subdomains():
    """A crt.sh query for a brand returns mostly the brand's own certificates.
    Only the bare target used to be dropped, so www/api/checkout all scored high."""
    fake_ct_response = [
        {"common_name": "example.com", "name_value": "example.com"},
        {"common_name": "www.example.com", "name_value": "www.example.com"},
        {"common_name": "api.example.com", "name_value": "api.example.com"},
        {"common_name": "checkout.example.com", "name_value": "checkout.example.com"},
        {"common_name": "examp1e.com", "name_value": "examp1e.com"},
    ]
    sim_by_domain = {"examp1e": 0.9}
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "example.com", "-s", "0.5"])

    with (
        mock.patch("swat.__main__.get_ct_logs_for_label", return_value=fake_ct_response),
        mock.patch("swat.__main__.SentenceTransformer", return_value=_fake_model(sim_by_domain)),
        mock.patch("swat.__main__.util", _FakeUtil()),
    ):
        result = swat_main.find_lookalikes(args)

    assert [r["domain"] for r in result["results"]] == ["examp1e.com"]


def test_find_lookalikes_ranks_by_content_similarity():
    """content_similarity is the strongest impersonation signal available, so it
    has to affect ordering. It used to be attached after the sort and ignored."""
    fake_ct_response = [
        {"common_name": "lookalike.com", "name_value": "lookalike.com"},
        {"common_name": "clone.com", "name_value": "clone.com"},
        {"common_name": "unreachable.com", "name_value": "unreachable.com"},
    ]
    # lookalike.com outscores clone.com on name alone; the page content reverses it.
    sim_by_domain = {"lookalike": 0.9, "clone": 0.6, "unreachable": 0.7}
    pages = {
        "https://example.com": "reference page",
        "lookalike.com": "barely related text",
        "clone.com": "the exact same wording as the real brand homepage",
        "unreachable.com": None,
    }

    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "example.com", "-s", "0.5", "--any", "-r", "https://example.com"])
    model = _fake_model(sim_by_domain)

    content_scores = {"lookalike.com": 0.11, "clone.com": 0.93, "unreachable.com": None}

    def fake_content_similarity(_model, _reference, domain):
        return content_scores[domain]

    with (
        mock.patch("swat.__main__.get_ct_logs_for_label", return_value=fake_ct_response),
        mock.patch("swat.__main__.SentenceTransformer", return_value=model),
        mock.patch("swat.__main__.util", _FakeUtil()),
        mock.patch("swat.classifier.fetch_text", side_effect=lambda url: pages.get(url)),
        mock.patch("swat.classifier.util", _FakeUtil()),
        mock.patch(
            "swat.classifier.content_similarity",
            side_effect=lambda _m, _r, domain: fake_content_similarity(None, None, domain),
        ),
    ):
        result = swat_main.find_lookalikes(args)

    assert [r["domain"] for r in result["results"]] == [
        "clone.com",
        "lookalike.com",
        "unreachable.com",
    ]


def test_default_requires_both_signals_and_any_falls_back_to_or():
    """OR-of-thresholds lets the weaker signal set precision on its own, so a
    candidate that clears only cosine similarity (with a label nothing like the
    brand) is dropped in the default AND mode and restored under --any."""
    fake_ct_response = [
        {"common_name": "unrelated-brand-far-away.com", "name_value": "unrelated-brand-far-away.com"},
    ]
    # 0.6 clears the 0.5 similarity default; the label shares no characters
    # with 'paypal' so the edit ratio is well past the 0.35 default.
    sim_by_domain = {"unrelated-brand-far-away": 0.6}

    parser = swat_main.build_parser()
    default_args = parser.parse_args(["-d", "paypal.com"])
    any_args = parser.parse_args(["-d", "paypal.com", "--any"])

    def _run(args):
        with (
            mock.patch("swat.__main__.get_ct_logs_for_label", return_value=fake_ct_response),
            mock.patch("swat.__main__.SentenceTransformer", return_value=_fake_model(sim_by_domain)),
            mock.patch("swat.__main__.util", _FakeUtil()),
        ):
            return swat_main.find_lookalikes(args)

    assert _run(default_args)["results"] == []
    assert [r["domain"] for r in _run(any_args)["results"]] == ["unrelated-brand-far-away.com"]


def test_edit_ratio_normalises_by_the_longer_label_length():
    """A distance of 1 is a sixth of a 6-character label and a fifteenth of a
    15-character one. An absolute cap does not distinguish those cases; a ratio
    cap does. The two candidates are chosen to share the character shape and
    not contain the target label as a substring, so brand-in-hostname does not
    also flag the longer one."""
    fake_ct_response = [
        # 'brandx' vs 'brandy' (length 6): distance 1, ratio 1/6 ≈ 0.17 → flagged
        {"common_name": "brandy.com", "name_value": "brandy.com"},
        # 'brandx' vs 'brandy-holdings' (length 15): distance 10, ratio 10/15 ≈ 0.67 → dropped
        {"common_name": "brandy-holdings.com", "name_value": "brandy-holdings.com"},
    ]
    sim_by_domain = {"brandy": 0.9, "brandy-holdings": 0.9}

    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "brandx.com"])

    with (
        mock.patch("swat.__main__.get_ct_logs_for_label", return_value=fake_ct_response),
        mock.patch("swat.__main__.SentenceTransformer", return_value=_fake_model(sim_by_domain)),
        mock.patch("swat.__main__.util", _FakeUtil()),
    ):
        result = swat_main.find_lookalikes(args)

    flagged = {r["domain"]: r["levenshtein_ratio"] for r in result["results"]}
    assert "brandy.com" in flagged
    assert flagged["brandy.com"] < 0.35
    assert "brandy-holdings.com" not in flagged


def test_max_edit_ratio_rejects_out_of_range_values():
    parser = swat_main.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["-d", "example.com", "-l", "3"])
    with pytest.raises(SystemExit):
        parser.parse_args(["-d", "example.com", "-l", "-0.1"])


def test_brand_in_hostname_flags_subdomain_shapes_thresholds_would_miss():
    """Label-only scoring throws away the subdomain context that used to catch
    brand-in-subdomain candidates. The substring signal restores that class:
    'paypal-secure.evil.com' scores as label 'evil' and fails both thresholds,
    but the brand token is right there in the hostname."""
    fake_ct_response = [
        {"common_name": "paypal-secure.evil.com", "name_value": "paypal-secure.evil.com"},
        {"common_name": "login.paypal.com.secure-login.tk", "name_value": "login.paypal.com.secure-login.tk"},
        {"common_name": "unrelated.com", "name_value": "unrelated.com"},
    ]
    # Deliberately low similarity on the labels: neither would flag on
    # thresholds alone, only the substring signal keeps them in.
    sim_by_domain = {"evil": 0.1, "secure-login": 0.1, "unrelated": 0.1}

    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "paypal.com"])

    with (
        mock.patch("swat.__main__.get_ct_logs_for_label", return_value=fake_ct_response),
        mock.patch("swat.__main__.SentenceTransformer", return_value=_fake_model(sim_by_domain)),
        mock.patch("swat.__main__.util", _FakeUtil()),
    ):
        result = swat_main.find_lookalikes(args)

    by_domain = {r["domain"]: r for r in result["results"]}
    assert by_domain["paypal-secure.evil.com"]["brand_in_hostname"] is True
    assert by_domain["login.paypal.com.secure-login.tk"]["brand_in_hostname"] is True
    assert "unrelated.com" not in by_domain
