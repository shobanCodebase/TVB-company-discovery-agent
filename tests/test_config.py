import os

from core.config import Settings


def test_llm_model_defaults_to_smaller_model_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = Settings()
    assert settings.llm_model == "openai/gpt-oss-20b"


def test_llm_model_is_configurable_via_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    settings = Settings()
    assert settings.llm_model == "openai/gpt-oss-120b"


def test_research_max_source_chars_configurable(monkeypatch):
    monkeypatch.setenv("RESEARCH_MAX_SOURCE_CHARS", "3000")
    settings = Settings()
    assert settings.research_max_source_chars == 3000


def test_llm_extraction_max_tokens_has_expected_default(monkeypatch):
    monkeypatch.delenv("LLM_EXTRACTION_MAX_TOKENS", raising=False)
    settings = Settings()
    assert settings.llm_extraction_max_tokens == 900

def test_max_candidates_for_research_defaults_to_20(monkeypatch):
    monkeypatch.delenv("MAX_CANDIDATES_FOR_RESEARCH", raising=False)
    settings = Settings()
    assert settings.max_candidates_for_research == 20


def test_groq_min_request_interval_defaults_to_20(monkeypatch):
    monkeypatch.delenv("GROQ_MIN_REQUEST_INTERVAL_SECONDS", raising=False)
    settings = Settings()
    assert settings.groq_min_request_interval_seconds == 20.0


def test_groq_min_request_interval_configurable(monkeypatch):
    monkeypatch.setenv("GROQ_MIN_REQUEST_INTERVAL_SECONDS", "5")
    settings = Settings()
    assert settings.groq_min_request_interval_seconds == 5.0