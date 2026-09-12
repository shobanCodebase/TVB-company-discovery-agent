from unittest.mock import MagicMock

import pytest

from core.config import Settings
from services.email_service import (
    EmailService,
    generate_email_candidates,
)


def make_settings(**overrides) -> Settings:
    defaults = dict(request_timeout_seconds=5.0, hunter_api_key="", use_email_verification_api=True)
    defaults.update(overrides)
    return Settings(**defaults)


# --- pattern generation ---

def test_generate_email_candidates_produces_expected_patterns():
    candidates = generate_email_candidates("Jane Doe", "acme.io")
    emails = {c.email for c in candidates}
    assert "jane.doe@acme.io" in emails
    assert "jane@acme.io" in emails
    assert "jdoe@acme.io" in emails
    assert "jane.d@acme.io" in emails


def test_generate_email_candidates_handles_single_word_name():
    candidates = generate_email_candidates("Cher", "acme.io")
    assert candidates == []


def test_generate_email_candidates_handles_middle_names():
    candidates = generate_email_candidates("Jane Q. Doe", "acme.io")
    emails = {c.email for c in candidates}
    assert "jane.doe@acme.io" in emails  # uses first + last token, ignores middle initial


# --- MX check ---

def test_domain_has_mx_record_true(monkeypatch):
    service = EmailService(settings=make_settings())
    monkeypatch.setattr(
        "services.email_service.dns.resolver.resolve",
        lambda domain, rtype, lifetime=None: [object()],
    )
    assert service.domain_has_mx_record("acme.io") is True


def test_domain_has_mx_record_false_on_nxdomain(monkeypatch):
    import dns.resolver
    service = EmailService(settings=make_settings())

    def raise_nxdomain(domain, rtype, lifetime=None):
        raise dns.resolver.NXDOMAIN()

    monkeypatch.setattr("services.email_service.dns.resolver.resolve", raise_nxdomain)
    assert service.domain_has_mx_record("nonexistent-domain-xyz.io") is False


def test_domain_has_mx_record_false_on_unexpected_error(monkeypatch):
    service = EmailService(settings=make_settings())

    def raise_weird(domain, rtype, lifetime=None):
        raise RuntimeError("dns server down")

    monkeypatch.setattr("services.email_service.dns.resolver.resolve", raise_weird)
    assert service.domain_has_mx_record("acme.io") is False


# --- verify_email pipeline (no Hunter key configured) ---

def test_verify_email_rejects_bad_syntax(monkeypatch):
    service = EmailService(settings=make_settings())
    result = service.verify_email("not-an-email", "acme.io")
    assert result.verified is False
    assert result.method == "syntax_check"


def test_verify_email_rejects_generic_address(monkeypatch):
    service = EmailService(settings=make_settings())
    result = service.verify_email("info@acme.io", "acme.io")
    assert result.verified is False
    assert result.method == "generic_check"


def test_verify_email_rejects_domain_mismatch(monkeypatch):
    service = EmailService(settings=make_settings())
    result = service.verify_email("jane@othercompany.com", "acme.io")
    assert result.verified is False
    assert result.method == "domain_match"


def test_verify_email_rejects_no_mx_record(monkeypatch):
    service = EmailService(settings=make_settings())
    monkeypatch.setattr(service, "domain_has_mx_record", lambda domain: False)
    result = service.verify_email("jane@acme.io", "acme.io")
    assert result.verified is False
    assert result.method == "mx_check"


def test_verify_email_passes_deterministic_only_without_hunter_key():
    service = EmailService(settings=make_settings(hunter_api_key=""))
    service.domain_has_mx_record = lambda domain: True
    result = service.verify_email("jane@acme.io", "acme.io")
    assert result.verified is True
    assert result.method == "deterministic_only"


# --- verify_email with Hunter configured ---

def test_verify_email_uses_hunter_score_pass(monkeypatch):
    service = EmailService(settings=make_settings(hunter_api_key="fake-key"))
    service.domain_has_mx_record = lambda domain: True
    monkeypatch.setattr(service, "hunter_verify_email", lambda email: 85)

    result = service.verify_email("jane@acme.io", "acme.io")
    assert result.verified is True
    assert result.method == "hunter_verify_email"
    assert result.confidence == 85


def test_verify_email_uses_hunter_score_fail(monkeypatch):
    service = EmailService(settings=make_settings(hunter_api_key="fake-key"))
    service.domain_has_mx_record = lambda domain: True
    monkeypatch.setattr(service, "hunter_verify_email", lambda email: 30)

    result = service.verify_email("jane@acme.io", "acme.io")
    assert result.verified is False
    assert result.confidence == 30


def test_verify_email_falls_back_when_hunter_unavailable(monkeypatch):
    service = EmailService(settings=make_settings(hunter_api_key="fake-key"))
    service.domain_has_mx_record = lambda domain: True
    monkeypatch.setattr(service, "hunter_verify_email", lambda email: None)

    result = service.verify_email("jane@acme.io", "acme.io")
    assert result.verified is True
    assert result.method == "deterministic_only"


# --- Hunter email-finder (mocked HTTP) ---

class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


def test_hunter_find_email_returns_none_without_key():
    service = EmailService(settings=make_settings(hunter_api_key=""))
    assert service.hunter_find_email("Jane", "Doe", "acme.io") is None


def test_hunter_find_email_parses_valid_response(monkeypatch):
    service = EmailService(settings=make_settings(hunter_api_key="fake-key"))
    monkeypatch.setattr(
        service._client, "get",
        lambda *a, **k: FakeResponse(200, {"data": {"email": "jane.doe@acme.io"}}),
    )
    candidate = service.hunter_find_email("Jane", "Doe", "acme.io")
    assert candidate is not None
    assert candidate.email == "jane.doe@acme.io"
    assert candidate.pattern == "hunter_finder"


def test_hunter_find_email_handles_no_match(monkeypatch):
    service = EmailService(settings=make_settings(hunter_api_key="fake-key"))
    monkeypatch.setattr(service._client, "get", lambda *a, **k: FakeResponse(200, {"data": {}}))
    assert service.hunter_find_email("Jane", "Doe", "acme.io") is None


def test_hunter_find_email_handles_non_200(monkeypatch):
    service = EmailService(settings=make_settings(hunter_api_key="fake-key"))
    monkeypatch.setattr(service._client, "get", lambda *a, **k: FakeResponse(403, {}, "forbidden"))
    assert service.hunter_find_email("Jane", "Doe", "acme.io") is None