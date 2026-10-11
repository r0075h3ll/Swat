import pytest

from swat import generator


def test_parse_candidates_extracts_domains_dedupes_and_preserves_order():
    raw = "paypail.com\npaypal.com\npay-pal.com\npaypail.com\n"
    assert generator._parse_candidates(raw) == ["paypail.com", "paypal.com", "pay-pal.com"]


def test_parse_candidates_strips_list_formatting_and_case():
    raw = "1. PayPail.com\n- pay-pal.com\n* pay-bpal.com\n"
    assert generator._parse_candidates(raw) == ["paypail.com", "pay-pal.com", "pay-bpal.com"]


def test_parse_candidates_drops_non_domain_tokens():
    raw = "paypail.com\njust some prose\npaypal\nfoo..com\n"
    assert generator._parse_candidates(raw) == ["paypail.com"]


def test_generate_candidates_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(generator, "OPENROUTER_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        generator.generate_candidates("paypal")


def test_generate_candidates_builds_prompt_and_parses(monkeypatch):
    monkeypatch.setattr(generator, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(generator, "_complete", lambda prompt: "paypail.com\npay-bpal.com\n")
    assert generator.generate_candidates("paypal", count=5) == ["paypail.com", "pay-bpal.com"]


def test_build_generation_prompt_contains_brand_and_count():
    prompt = generator.build_generation_prompt("paypal", count=7)
    assert "paypal" in prompt
    assert "7" in prompt
    assert "one per line" in prompt


def test_build_generation_prompt_includes_business_context_when_provided():
    prompt = generator.build_generation_prompt("paypal", count=3, business_context="peer-to-peer payments")
    assert "peer-to-peer payments" in prompt


def test_build_generation_prompt_omits_context_block_when_empty():
    prompt = generator.build_generation_prompt("paypal", count=3, business_context="")
    assert "taken from its own website" not in prompt


def test_complete_raises_on_non_200(monkeypatch):
    class _Response:
        status_code = 401
        text = "unauthorized"

    monkeypatch.setattr(generator, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(generator.requests, "post", lambda *a, **k: _Response())
    with pytest.raises(RuntimeError, match="401"):
        generator._complete("prompt")
