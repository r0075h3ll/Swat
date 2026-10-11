import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from swat import __main__ as swat_main

REPO_ROOT = Path(__file__).parent.parent


def _find(
    args,
    candidates_by_label,
    *,
    live_ok=lambda d: True,
    domain_ok=lambda t, c: True,
    content_ok=lambda r, c: True,
    fetch=lambda url: "page text",
):
    def fake_generate(label, count=20, business_context=""):
        return candidates_by_label.get(label, [])

    with (
        mock.patch("swat.generator.generate_candidates", side_effect=fake_generate),
        mock.patch("swat.__main__.is_live", side_effect=live_ok),
        mock.patch("swat.classifier.domain_impersonation", side_effect=domain_ok),
        mock.patch("swat.classifier.content_impersonation", side_effect=content_ok),
        mock.patch("swat.classifier.fetch_text", side_effect=fetch),
    ):
        return swat_main.find_lookalikes(args)


def test_find_lookalikes_flags_candidate_when_both_gates_yes():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "paypal.com"])
    result = _find(args, {"paypal": ["paypa1.com"]})
    assert [r["domain"] for r in result["results"]] == ["paypa1.com"]
    assert result["results"][0]["matched_label"] == "paypal"
    assert result["results"][0]["levenshtein_distance"] == 1
    assert result["manual_review"] == []


def test_find_lookalikes_drops_first_party():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "paypal.com"])
    result = _find(args, {"paypal": ["paypal.com", "www.paypal.com", "paypa1.com"]})
    assert [r["domain"] for r in result["results"]] == ["paypa1.com"]


def test_find_lookalikes_skips_candidate_that_is_not_live():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "paypal.com"])
    result = _find(args, {"paypal": ["paypa1.com"]}, live_ok=lambda d: False)
    assert result["results"] == []


def test_find_lookalikes_reports_generated_candidates_and_jev_decisions():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "paypal.com"])
    result = _find(
        args,
        {"paypal": ["paypa1.com", "paypa2.com", "paypall.com", "paypql.com"]},
        live_ok=lambda d: d != "paypa2.com",
        domain_ok=lambda t, c: c != "paypall",
        content_ok=lambda r, c: c != "paypql.com",
        fetch=lambda url: url,
    )
    assert result["generated_candidates"] == [
        {"domain": "paypa1.com", "matched_label": "paypal"},
        {"domain": "paypa2.com", "matched_label": "paypal"},
        {"domain": "paypall.com", "matched_label": "paypal"},
        {"domain": "paypql.com", "matched_label": "paypal"},
    ]
    assert result["decisions"] == [
        {
            "domain": "paypa1.com",
            "matched_label": "paypal",
            "domain_gate": "yes",
            "content_gate": "yes",
        },
        {
            "domain": "paypall.com",
            "matched_label": "paypal",
            "domain_gate": "no",
            "content_gate": None,
        },
        {
            "domain": "paypql.com",
            "matched_label": "paypal",
            "domain_gate": "yes",
            "content_gate": "no",
        },
    ]


def test_is_live_short_circuits():
    with (
        mock.patch("swat.__main__.cert_exists") as cert_exists,
        mock.patch("swat.__main__.classifier.http_probe") as http_probe,
        mock.patch("swat.__main__.classifier.jina_fetch") as jina_fetch,
    ):
        # CT record present: skip both network probes.
        cert_exists.return_value = True
        assert swat_main.is_live("example.com") is True
        http_probe.assert_not_called()
        jina_fetch.assert_not_called()

        # No CT record, but the HTTP probe answers: skip the Jina fetch.
        cert_exists.return_value = False
        http_probe.return_value = True
        assert swat_main.is_live("example.com") is True
        http_probe.assert_called_once_with("example.com")
        jina_fetch.assert_not_called()

        # No CT record and no HTTP response, but Jina renders content.
        http_probe.return_value = False
        jina_fetch.return_value = "rendered index"
        assert swat_main.is_live("example.com") is True
        jina_fetch.assert_called_once_with("https://example.com")

        # All three signals absent: not live.
        jina_fetch.return_value = None
        assert swat_main.is_live("example.com") is False


def test_find_lookalikes_skips_levenshtein_far_candidate():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "paypal.com"])
    result = _find(args, {"paypal": ["totally-different-brand.com"]})
    assert result["results"] == []


def test_find_lookalikes_skips_domain_gate_no():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "paypal.com"])
    result = _find(args, {"paypal": ["paypa1.com"]}, domain_ok=lambda t, c: False)
    assert result["results"] == []


def test_find_lookalikes_skips_content_gate_no():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "paypal.com"])
    result = _find(args, {"paypal": ["paypa1.com"]}, content_ok=lambda r, c: False)
    assert result["results"] == []


def test_find_lookalikes_manual_review_when_content_unreachable():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "paypal.com"])
    result = _find(args, {"paypal": ["paypa1.com"]}, fetch=lambda url: None)
    assert result["results"] == []
    assert result["manual_review"] == [
        {"domain": "paypa1.com", "matched_label": "paypal", "reason": "content unreachable"}
    ]


def test_find_lookalikes_manual_review_when_reference_unavailable():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "paypal.com"])

    def fetch(url):
        return None if "://" in url else "candidate page text"

    result = _find(args, {"paypal": ["paypa1.com"]}, fetch=fetch)
    assert result["results"] == []
    assert result["manual_review"] == [
        {"domain": "paypa1.com", "matched_label": "paypal", "reason": "reference content unavailable"}
    ]


def test_find_lookalikes_multiple_aliases():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "acmecorp.com", "-d", "acmebank.co.id"])
    result = _find(
        args,
        {"acmecorp": ["acmecorp-clone.com"], "acmebank": ["acmebank-clone.com"]},
    )
    by_domain = {r["domain"]: r for r in result["results"]}
    assert by_domain["acmecorp-clone.com"]["matched_label"] == "acmecorp"
    assert by_domain["acmebank-clone.com"]["matched_label"] == "acmebank"


def test_find_lookalikes_dedupes_identical_domain_argument():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "example.com", "-d", "example.com"])
    result = _find(args, {"example": []})
    assert result["input_domains"] == ["example.com"]


def test_find_lookalikes_generates_per_unique_label_only_once():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "acmecorp.com", "-d", "acmecorp.co.th"])
    calls = []
    with mock.patch(
        "swat.generator.generate_candidates",
        side_effect=lambda label, count=20, business_context="": calls.append(label) or [],
    ):
        swat_main.find_lookalikes(args)
    assert calls == ["acmecorp"]


def test_find_lookalikes_raises_when_generation_fails_everywhere():
    parser = swat_main.build_parser()
    args = parser.parse_args(["-d", "example.com"])
    with (
        mock.patch("swat.generator.generate_candidates", side_effect=RuntimeError("no model")),
        pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"),
    ):
        swat_main.find_lookalikes(args)


# --- CLI / error handling (subprocess) ---


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
    with mock.patch(
        "swat.__main__.find_lookalikes",
        return_value={"input_domains": ["example.com"], "results": [], "manual_review": []},
    ):
        swat_main.main(["-d", "example.com", "-o", str(output_file)])

    assert json.loads(output_file.read_text()) == {
        "input_domains": ["example.com"],
        "results": [],
        "manual_review": [],
    }


def test_main_prints_json_to_stdout_by_default(capsys):
    with mock.patch(
        "swat.__main__.find_lookalikes",
        return_value={"input_domains": ["example.com"], "results": [], "manual_review": []},
    ):
        swat_main.main(["-d", "example.com"])

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "input_domains": ["example.com"],
        "results": [],
        "manual_review": [],
    }


@pytest.mark.slow
def test_stdout_output_contains_no_log_noise():
    script = (
        "import sys; sys.path.insert(0, '.')\n"
        "from unittest import mock\n"
        "import swat.__main__ as m\n"
        "with mock.patch('swat.generator.generate_candidates', return_value=[]):\n"
        "    m.main(['-d', 'example.com'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert json.loads(result.stdout) == {
        "input_domains": ["example.com"],
        "generated_candidates": [],
        "decisions": [],
        "results": [],
        "manual_review": [],
    }


@pytest.mark.slow
def test_importing_swat_leaves_the_root_logger_alone():
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
def test_total_generation_failure_exits_non_zero_without_printing_results():
    script = (
        "import sys; sys.path.insert(0, '.')\n"
        "from unittest import mock\n"
        "import runpy\n"
        "from swat import generator\n"
        "with mock.patch.object(generator, 'generate_candidates', side_effect=RuntimeError('no model')):\n"
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


@pytest.mark.slow
def test_unexpected_exception_writes_traceback_to_a_file_not_stderr(tmp_path):
    script = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from unittest import mock\n"
        "import runpy\n"
        "from swat import generator\n"
        "with mock.patch.object(generator, 'generate_candidates', side_effect=ValueError('boom')):\n"
        "    sys.argv = ['swat', '-d', 'example.com']\n"
        "    runpy.run_module('swat', run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "Traceback" not in result.stderr
    assert "boom" in result.stderr
    log_path = tmp_path / "swat-error.log"
    assert log_path.exists()
    assert "Traceback" in log_path.read_text()
    assert "boom" in log_path.read_text()


def test_color_formatter_colors_warnings_on_a_tty():
    import logging

    record = logging.LogRecord("swat", logging.WARNING, __file__, 1, "careful now", None, None)
    formatter = swat_main._ColorFormatter()
    with mock.patch.object(sys.stderr, "isatty", return_value=True):
        formatted = formatter.format(record)
    assert formatted.startswith(swat_main._LEVEL_COLORS[logging.WARNING])
    assert formatted.endswith(swat_main._RESET)
    assert "careful now" in formatted


def test_color_formatter_leaves_plain_output_when_not_a_tty():
    import logging

    record = logging.LogRecord("swat", logging.ERROR, __file__, 1, "broken", None, None)
    formatter = swat_main._ColorFormatter()
    with mock.patch.object(sys.stderr, "isatty", return_value=False):
        formatted = formatter.format(record)
    assert formatted == "broken"


def test_color_formatter_leaves_info_uncolored_even_on_a_tty():
    import logging

    record = logging.LogRecord("swat", logging.INFO, __file__, 1, "routine", None, None)
    formatter = swat_main._ColorFormatter()
    with mock.patch.object(sys.stderr, "isatty", return_value=True):
        formatted = formatter.format(record)
    assert formatted == "routine"
